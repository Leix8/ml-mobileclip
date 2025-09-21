'''


# load frames and other metadata from given video and downsample to the target fps to play video, if it's smaller than orginal fps
def load_frames_from_video(video_path: str, play_fps) -> video_frames: list[frame], durtation: float, width: int, height: int

# load the embeddings to retrieve upon
def load_embedding_database(scene_json: str) -> embedding_database: json_object

# load model
def load_model(model_path, **kwargs): -> model, tokenizer, image_processor

# encode image
def encode_image(image) -> image_embedding

# calculate the sampling frame idx according to video_frames and play fps, so that the sampling is at encoding fps
def get_encoding_idx(video_frames: list[frame], play_fps: int, encoding_fps: int) -> sampled_frame_idx: [int]

# core function, to predict the highlight frame index from the given video frames and methods
def predict_highlight_idx(video_frames: [frame], sampled_frame_idx: [int], embedding_database: json_object, method, nms, topk) -> highlight_idx: [{index: tag}, {index: tag}, ...], cos_sim_score_stats
    # 1. calculate image embeddings for all sampled frames with sampled frame idx
    # 2. with the given method, use its embeddings for each tag to calculate the cos similartiy with each frame's image embeddings
    # 3. with the stats in step 2, do proper filtering, including nms on each cos similarity of each tags's embedding
    # 4. combine all cos similarity score across all tags properly, for example, for each frame idx keeping only the score of the tag with the highest score
    # 5. do another nms if needed
    # 6. select the topk as the predicted highlight idx

# core function, to predict the highlight video clip duration based on highlight frame index
def predict highlight_clip(video_frames: [frame], highlight_idx: [int], **kwargs) -> video_highlight_clips: {[{start_idx: int, end_idx: int}, {}, ...]}
    # from the predicted highlight index, convert it into video clips of [start_idx, end_index]
    # by default make the start idx / end index to be 1 second before / after the predicted highlight index

# visualize the save the result as video
def visualize_video_clipping(output_path: str, cos_sim_score_stats, highlight_idx, video_highlight_clips)
    # core visualize function
    # make a proper layout to visualize the highlight selection and clipping versus original video
    # e.g. on the upper section of each frame, draw the frame of the video, but make a way to differentiate the predicted clips or not
    # e.g. on the lower section of each frame, draw a plot of the cos_score stats and make a mark on the selected highlight frames and mark which tag it comes from, and make a sliding vertical line to align with the video progress

def main():
    ap = argparse.ArgumentParser("clip video based on MobileCLIP embeddings")
    ap.add_argument("--scene_json", type=str, required=True, help="directory containing images (will recurse)")
    ap.add_argument("--video", type=str, required=True, help="path to input video file")
    ap.add_argument("--model_path", type=str, default="./checkpoints/mobileclip2_b.pt")
    ap.add_argument("--device", type=str, default="cuda:0")
    ap.add_argument("--outdir", type=str, default="./highlight_clipping")
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--fps", type=int, default=1)
    
'''


#!/usr/bin/env python3
"""
Video highlight retrieval & clipping using MobileCLIP(-2) style embeddings.

Scene JSON schema (flexible; any subset works):
{
  "tags": [
    {
      "tag": "dog jumping",
      "image_embeddings": [[...], [...], ...],     # optional, list of D
      "text_embeddings":  [[...]],                 # optional, list of D (or single)
      "weight": 1.0                                 # optional global weight
    },
    ...
  ],
  "encoding_fps": 2                                 # optional: source fps for embeddings
}
"""

import argparse
import os
import sys
import json
import math
import random
from typing import List, Dict, Any, Tuple, Optional

import numpy as np
from pathlib import Path

import cv2
from PIL import Image
from tqdm import tqdm

import torch
import torch.nn.functional as F
import open_clip
from mobileclip.modules.common.mobileone import reparameterize_model
from model_name_map import MODEL_NAME_MAP, infer_model_name_from_ckpt

# ---------------- Utils ---------------- #

def setup_determinism(seed: int = 1234):
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":16:8")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    try:
        torch.use_deterministic_algorithms(True)
    except Exception:
        pass


def _norm(x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    return x / (x.norm(dim=-1, keepdim=True) + eps)


def seconds_to_frames(seconds: float, fps: float) -> int:
    return int(round(seconds * fps))


def merge_intervals(intervals: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
    if not intervals:
        return []
    intervals = sorted(intervals)
    merged = [intervals[0]]
    for s, e in intervals[1:]:
        last_s, last_e = merged[-1]
        if s <= last_e + 1:
            merged[-1] = (last_s, max(last_e, e))
        else:
            merged.append((s, e))
    return merged


# ---------------- Core API (your skeleton) ---------------- #

# load frames and other metadata from given video and downsample to the target fps to play video, if it's smaller than original fps
def load_frames_from_video(video_path: str, play_fps: Optional[float], max_side: int = 500) -> Tuple[List[np.ndarray], float, int, int, float]:
    """
    Returns:
        video_frames: list of BGR frames (np.ndarray HxWx3)
        duration_s: float
        width: int
        height: int
        effective_play_fps: float (actual fps of returned frames)
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {video_path}")

    orig_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    duration_s = total_frames / max(orig_fps, 1e-6)

    # compute scaling factor so that max(h,w) <= max_side
    max_dim = max(width, height)
    if max_dim > max_side:
        scale = max_side / max_dim
        target_w = int(round(width * scale))
        target_h = int(round(height * scale))
    else:
        target_w, target_h = width, height
    
    # If play_fps is None or 0, keep original; else, downsample if play_fps < orig_fps
    if not play_fps or play_fps <= 0:
        target_fps = orig_fps
    else:
        target_fps = min(play_fps, orig_fps)

    # Timestamp-based sampling to avoid drift
    step = 1.0 / target_fps
    timestamps = np.arange(0.0, duration_s + 1e-6, step)

    frames = []
    for t in tqdm(timestamps, desc = "loading video frames"):
        ok, frame = cap.read()
        if not ok:
            break
        # --- Downsample to fit within max_side ---
        if (frame.shape[1], frame.shape[0]) != (target_w, target_h):
            frame = cv2.resize(frame, (target_w, target_h), interpolation=cv2.INTER_AREA)
        frames.append(frame)

    cap.release()
    effective_play_fps = target_fps if len(frames) > 1 else orig_fps
    return frames, duration_s, width, height, effective_play_fps


# load the embeddings to retrieve upon
def load_embedding_database(scene_json: str, method: str) -> Dict[str, Any]:
    """
    Parse embedding_database.json in 'pet_scenes' format.
    Args:
        scene_json: path to json file
        method: which embedding key to use
                e.g. 'tag_embedding', 'multi_ref_TAF_embedding', 'multi_ref_push_embedding'
    Returns:
        {
          'tags': {
            tag: {
              'embedding': torch.Tensor [1,D],
              'weight': float,
              'params': dict   # only if provided
            }, ...
          }
        }
    """
    with open(scene_json, "r", encoding="utf-8") as f:
        data = json.load(f)

    db = {"tags": {}}
    scenes = data.get("pet_scenes", [])

    for entry in tqdm(scenes, desc = "loading embedding database"):
        tag = entry.get("tag", None)
        if not tag:
            continue

        emb_pack = entry.get("embedding", {})
        if method not in emb_pack:
            continue

        raw_emb = emb_pack[method]

        # Case 1: direct format (tag_embedding)
        if "value" in raw_emb:
            arr = np.array(raw_emb["value"], dtype=np.float32)  # cast to float32
            tensor = torch.tensor(arr, dtype=torch.float32)
            tensor = F.normalize(tensor, dim=-1).unsqueeze(0)  # [1,D]
            db["tags"][tag] = {
                "embedding": tensor,
                "weight": 1.0,
                "params": {}
            }

        # Case 2: nested format with params+data (multi_ref_TAF_embedding / push_embedding)
        elif "data" in raw_emb and "value" in raw_emb["data"]:
            arr = np.array(raw_emb["data"]["value"], dtype=np.float32)
            tensor = torch.tensor(arr, dtype=torch.float32)
            tensor = F.normalize(tensor, dim=-1).unsqueeze(0)
            db["tags"][tag] = {
                "embedding": tensor,
                "weight": 1.0,
                "params": raw_emb.get("params", {})
            }

    return db


# load model
def load_model(model_path: str, device: str = "cuda:0", precision: str = "fp16", **kwargs):
    """
    Loads MobileCLIP/MobileCLIP2 via the 'mobileclip' package.
    Returns: model, tokenizer (None), image_processor (callable PIL->Tensor)
    """
    model_name = infer_model_name_from_ckpt(model_path)
    model_kwargs = {}
    if not (model_name.endswith("S3") or model_name.endswith("S4") or model_name.endswith("L-14")):
        model_kwargs = {"image_mean": (0, 0, 0), "image_std": (1, 1, 1)}
    model, _, image_processor = open_clip.create_model_and_transforms(model_name, pretrained=model_path, **model_kwargs)
    device = torch.device(device if torch.cuda.is_available() else "cpu")
    model = model.to(device).eval()
    model = reparameterize_model(model)
    tokenizer = open_clip.get_tokenizer(model_name)

    if precision == "fp16" and torch.cuda.is_available():
        model = model.half()
    elif precision == "bf16" and torch.cuda.is_available():
        model = model.to(torch.bfloat16)

    return model, tokenizer, image_processor, model_name


# encode image
@torch.no_grad()
def encode_image(model, image_processor, image_bgr: np.ndarray, device: str = "cuda:0", precision: str = "fp16") -> torch.Tensor:
    """
    Returns L2-normalized embedding [D] as torch.Tensor (float32 on CPU).
    """
    img_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    pil = Image.fromarray(img_rgb)
    pixel = image_processor(pil).unsqueeze(0)  # [1,3,H,W]
    pixel = pixel.to(device)
    if precision == "fp16":
        pixel = pixel.half()
    elif precision == "bf16":
        pixel = pixel.to(torch.bfloat16)
    feat = model.encode_image(pixel)  # [1,D]
    feat = F.normalize(feat.float(), dim=-1).squeeze(0).cpu()
    return feat


# calculate the sampling frame idx according to video_frames and play fps, so that the sampling is at encoding fps
def get_encoding_idx(video_frames: List[np.ndarray], play_fps: float, encoding_fps: float) -> List[int]:
    """
    Map frames sampled at play_fps into indices that approximate encoding_fps sampling.
    """
    if encoding_fps <= 0 or play_fps <= 0:
        return list(range(len(video_frames)))
    duration_s = len(video_frames) / play_fps
    enc_times = np.arange(0.0, duration_s + 1e-9, 1.0 / encoding_fps)
    idx = [min(int(round(t * play_fps)), len(video_frames) - 1) for t in enc_times]
    # Dedup & in-bounds
    seen = set()
    out = []
    for i in idx:
        if 0 <= i < len(video_frames) and i not in seen:
            out.append(i)
            seen.add(i)
    return out


# ----- NMS / peak picking on 1D scores ----- #

def nms_1d(scores: np.ndarray, window: int, threshold: float) -> List[int]:
    """
    Simple 1D local-maxima NMS:
    - Keep indices that are local maxima within +/- window
    - And exceed threshold
    """
    if len(scores) == 0:
        return []
    keep = []
    N = len(scores)
    for i in range(N):
        s = scores[i]
        if s < threshold:
            continue
        left = max(0, i - window)
        right = min(N, i + window + 1)
        if s >= scores[left:right].max():
            keep.append(i)
    return keep


# core function, to predict the highlight frame index from the given video frames and methods
@torch.no_grad()
def predict_highlight_idx(
    model,
    image_processor,
    device: str,
    precision: str,
    video_frames: List[np.ndarray],
    sampled_frame_idx: List[int],
    embedding_database: Dict[str, Any],
    nms_window: int,
    nms_threshold: float,
    topk: int,
    **kwargs
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """
    Predict highlight frames from video given reference embeddings.

    Args:
        model, image_processor: MobileCLIP model and processor
        device, precision: inference device & dtype
        video_frames: list of frames (BGR np.ndarray)
        sampled_frame_idx: indices of frames to encode
        embedding_database: {
            "tags": {
                tag: {
                    "embedding": torch.Tensor [1,D],
                    "weight": float,
                    "params": dict   # optional parameters from JSON
                }
            }
        }
        nms_window: half-window for local maxima suppression
        nms_threshold: score threshold for NMS
        topk: number of peaks to keep

    Returns:
        highlight_idx: list of {"index": idx_global, "tag": tag, "score": float, "params": dict}
        stats: dictionary with per-tag score curves
    """
    # 1) Encode sampled frames
    feats = []
    for i in tqdm(sampled_frame_idx, desc = "embedding sampled frames"):
        emb = encode_image(model, image_processor, video_frames[i],
                           device=device, precision=precision)
        feats.append(emb)
    feats = torch.stack(feats, dim=0)  # [Ns,D]
    feats = F.normalize(feats, dim=-1)

    # 2) Compute cosine similarity with each tag embedding
    per_tag_scores: Dict[str, Dict[str, Any]] = {}
    for tag, pack in embedding_database["tags"].items():
        refE = pack["embedding"]   # [1,D]
        w    = float(pack.get("weight", 1.0))
        params = pack.get("params", {})

        if refE is None:
            continue

        # Cosine similarity [Ns]
        scores_raw = torch.matmul(feats, refE.T).squeeze(-1)  # [Ns]
        scores = (scores_raw * w).cpu().numpy()

        per_tag_scores[tag] = {
            "scores": scores,
            "sampled_idx": sampled_frame_idx,
            "weight": w,
            "params": params,  # keep params here for analysis/logging
        }

    # 3) Fuse across tags: take best scoring tag per frame
    Ns = len(sampled_frame_idx)
    fused_scores = np.full(Ns, -1e9, dtype=np.float32)
    fused_tags   = np.array([""] * Ns, dtype=object)

    for tag, pack in per_tag_scores.items():
        s = pack["scores"]
        better = s > fused_scores
        fused_scores[better] = s[better]
        fused_tags[better] = tag

    # 4) Apply NMS on fused scores
    keep_idx_local = nms_1d(fused_scores, window=nms_window, threshold=nms_threshold)

    # 5) Select top-k peaks
    keep_idx_local = sorted(
        keep_idx_local,
        key=lambda i: float(fused_scores[i]),
        reverse=True
    )[:topk]

    highlight_idx = []
    for i_local in keep_idx_local:
        idx_global = sampled_frame_idx[i_local]
        tag = str(fused_tags[i_local])
        params = per_tag_scores[tag]["params"]
        highlight_idx.append({
            "index": int(idx_global),
            "tag": tag,
            "score": float(fused_scores[i_local]),
            "params": params,  # pass along tag params for context
        })

    # 6) Collect stats
    stats = {
        "per_tag": per_tag_scores,
        "best_per_frame": {
            "score": fused_scores.tolist(),
            "tag": fused_tags.tolist(),
        },
        "sampled_idx": sampled_frame_idx,
    }

    return highlight_idx, stats

# core function, to predict the highlight video clip duration based on highlight frame index
def predict_highlight_clip(
    video_frames: List[np.ndarray],
    highlight_idx: List[Dict[str, Any]],
    pad_sec: float,
    play_fps: float,
    min_clip_sec: float = 1.0,
    merge_if_overlap: bool = True
) -> List[Dict[str, int]]:
    """
    Returns list of dicts: [{"start_idx": int, "end_idx": int}, ...]
    """
    pad = seconds_to_frames(pad_sec, play_fps)
    min_len = seconds_to_frames(min_clip_sec, play_fps)
    N = len(video_frames)
    raw_intervals = []
    for item in highlight_idx:
        i = int(item["index"])
        s = max(0, i - pad)
        e = min(N - 1, i + pad)
        if e - s + 1 < min_len:
            center = i
            s = max(0, center - (min_len // 2))
            e = min(N - 1, s + min_len - 1)
        raw_intervals.append((s, e))

    if merge_if_overlap:
        merged = merge_intervals(raw_intervals)
    else:
        merged = raw_intervals

    return [{"start_idx": s, "end_idx": e} for (s, e) in merged]


# visualize the save the result as video
def visualize_video_clipping(
    output_path: str,
    cos_sim_score_stats: Dict[str, Any],
    highlight_idx: List[Dict[str, Any]],
    video_highlight_clips: List[Dict[str, int]],
    video_frames: List[np.ndarray],
    play_fps: float,
    method: str,
    model_name: str,
    params: Dict[str, Any],
    max_tags_to_draw: int = 6,
    show_score_ranges: Tuple[float, float] = (-0.1, 1.0),
    font_scale: float = 0.6,
    thickness: int = 1,
    canvas_max_side: int = 800,
    frame_max_side: int = 400,
    thumb_h: int = 60,
):
    sampled_idx = cos_sim_score_stats["sampled_idx"]
    Ns = len(sampled_idx)
    per_tag = cos_sim_score_stats["per_tag"]
    best_per_frame = cos_sim_score_stats["best_per_frame"]

    # --- Resize video frame dimension if too big ---
    frame_h, frame_w = video_frames[0].shape[:2]
    frame_max_dim = max(frame_w, frame_h)
    if frame_max_dim > frame_max_side:
        frame_scale = frame_max_side / frame_max_dim
        frame_w = int(round(frame_w * frame_scale))
        frame_h = int(round(frame_h * frame_scale))
    else:
        frame_scale = 1.0

    pad = 10
    meta_w = 320
    plot_h = 200

    canvas_w = frame_w + meta_w + 3 * pad
    canvas_h = frame_h + thumb_h + plot_h + 5 * pad

    max_dim = max(canvas_w, canvas_h)
    if max_dim > canvas_max_side:
        scale = canvas_max_side / max_dim
        out_w, out_h = int(round(canvas_w * scale)), int(round(canvas_h * scale))
    else:
        scale = 1.0
        out_w, out_h = canvas_w, canvas_h

    # --- Precompute thumbnails (fixed width) ---
    num_thumbs = min(20, len(video_frames))
    step = max(1, len(video_frames) // num_thumbs)
    thumb_w = (canvas_w - 2 * pad) // num_thumbs
    thumbs = []
    for i in range(0, len(video_frames), step):
        th = cv2.resize(video_frames[i], (thumb_w, thumb_h), interpolation=cv2.INTER_AREA)
        thumbs.append((i, th))

    # --- Highlight mask ---
    highlight_mask = np.zeros(len(video_frames), dtype=np.uint8)
    for clip in video_highlight_clips:
        s, e = clip["start_idx"], clip["end_idx"]
        highlight_mask[s:e+1] = 1

    # --- VideoWriter ---
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, play_fps, (out_w, out_h))

    global_to_local = {g: i for i, g in enumerate(sampled_idx)}

    rng = np.random.RandomState(42)
    colors = {tag: tuple(int(c) for c in rng.randint(50, 220, size=3))
              for tag in list(per_tag.keys())[:max_tags_to_draw]}

    for g_idx, frame in enumerate(video_frames):
        # resize frame if needed
        if frame_scale != 1.0:
            frame = cv2.resize(frame, (frame_w, frame_h), interpolation=cv2.INTER_AREA)

        # --- canvas ---
        canvas = np.ones((canvas_h, canvas_w, 3), dtype=np.uint8) * 255

        # upper section: frame
        y0, x0 = pad, pad
        canvas[y0:y0+frame_h, x0:x0+frame_w] = frame

        # upper section: metadata
        meta_x = frame_w + 2*pad
        y_line = y0 + 20
        info_lines = [
            f"Frame {g_idx}/{len(video_frames)}",
            f"Method: {method}",
            f"Model: {model_name}",
            f"Params: {params}",
        ]
        if g_idx in global_to_local:
            li = global_to_local[g_idx]
            info_lines.append(f"Best tag: {best_per_frame['tag'][li]}")
            info_lines.append(f"Score: {best_per_frame['score'][li]:.3f}")
        for txt in info_lines:
            cv2.putText(canvas, txt, (meta_x, y_line),
                        cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0,0,0), 1, cv2.LINE_AA)
            y_line += 22

        # middle section: timeline bar
        bar_y, bar_x = frame_h + 2*pad, pad
        for idx, (fi, th) in enumerate(thumbs):
            x1 = bar_x + idx * thumb_w
            canvas[bar_y:bar_y+thumb_h, x1:x1+thumb_w] = th

            if highlight_mask[fi]:
                overlay = canvas[bar_y:bar_y+thumb_h, x1:x1+thumb_w].copy()
                overlay[:] = (50, 200, 50)
                cv2.addWeighted(overlay, 0.3, canvas[bar_y:bar_y+thumb_h, x1:x1+thumb_w],
                                0.7, 0, canvas[bar_y:bar_y+thumb_h, x1:x1+thumb_w])

        # cursor on timeline
        cur_idx = int(g_idx * num_thumbs / len(video_frames))
        if cur_idx < len(thumbs):
            x_c = bar_x + cur_idx * thumb_w + thumb_w // 2
            cv2.line(canvas, (x_c, bar_y), (x_c, bar_y+thumb_h), (0,0,255), 2)

        # lower section: score curves
        plot_y = bar_y + thumb_h + pad
        plot_x0, plot_x1 = pad, canvas_w - pad
        plot_y0, plot_y1 = plot_y, plot_y + plot_h

        cv2.rectangle(canvas, (plot_x0, plot_y0), (plot_x1, plot_y1), (200,200,200), 1)

        for tag, pack in list(per_tag.items())[:max_tags_to_draw]:
            s = pack["scores"]
            pts = []
            for i, sv in enumerate(s):
                x = int(np.interp(i, [0, Ns-1], [plot_x0, plot_x1]))
                y = int(np.interp(sv, show_score_ranges, [plot_y1, plot_y0]))
                pts.append((x,y))
            for i in range(1,len(pts)):
                cv2.line(canvas, pts[i-1], pts[i], colors[tag], 2)

        # moving cursor
        if g_idx in global_to_local:
            li = global_to_local[g_idx]
            x_c = int(np.interp(li, [0,Ns-1],[plot_x0,plot_x1]))
            cv2.line(canvas, (x_c, plot_y0), (x_c, plot_y1), (0,0,255), 2)

        # highlight markers
        for clip in video_highlight_clips:
            s, e = clip["start_idx"], clip["end_idx"]
            x_s = int(np.interp(s, [0,len(video_frames)-1],[plot_x0,plot_x1]))
            x_e = int(np.interp(e, [0,len(video_frames)-1],[plot_x0,plot_x1]))
            cv2.rectangle(canvas, (x_s, plot_y0), (x_e, plot_y1), (0,255,0), 1)

        # legend
        legend_y = plot_y1 + 20
        lx = plot_x0
        for tag, color in colors.items():
            cv2.rectangle(canvas, (lx, legend_y-10), (lx+15, legend_y+5), color, -1)
            cv2.putText(canvas, tag[:20], (lx+20, legend_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1, cv2.LINE_AA)
            lx += 150

        if scale != 1.0:
            canvas = cv2.resize(canvas, (out_w, out_h), interpolation=cv2.INTER_AREA)

        out.write(canvas)

    out.release()


# ---------------- Main ---------------- #

def main():
    ap = argparse.ArgumentParser("clip video based on MobileCLIP embeddings")
    ap.add_argument("--scene_json", type=str, required=True, help="path to scene JSON with tag embeddings")
    ap.add_argument("--video", type=str, required=True, help="path to input video file")
    ap.add_argument("--model_path", type=str, default="./checkpoints/mobileclip2_b.pt")
    ap.add_argument("--device", type=str, default="cuda:0")
    ap.add_argument("--precision", type=str, default="fp16", choices=["fp32", "fp16", "bf16"])
    ap.add_argument("--outdir", type=str, default="./highlight_clipping")
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--play_fps", type=float, default=5.0, help="playback/downsample fps (0=original)")
    ap.add_argument("--encoding_fps", type=float, default=1.0, help="fps used to generate embeddings (0=auto from scene_json or use play_fps)")
    ap.add_argument("--method", type=str, default="tag_embedding",
                choices=["tag_embedding", "multi_ref_TAF_embedding", "multi_ref_push_embedding"],
                help="which embedding type from JSON to use")
    ap.add_argument("--per_tag_reduce", type=str, default="max", choices=["max", "mean"])
    ap.add_argument("--tag_mix", type=str, default="best", choices=["best", "sum", "both"])
    ap.add_argument("--nms_window", type=int, default=5, help="half-width (in sampled frames) for 1D NMS")
    ap.add_argument("--nms_threshold", type=float, default=0.25)
    ap.add_argument("--topk", type=int, default=5, help="number of peaks to keep")
    ap.add_argument("--clip_pad_sec", type=float, default=1.0, help="padding seconds before/after highlight frame")
    ap.add_argument("--min_clip_sec", type=float, default=1.0)
    ap.add_argument("--max_tags_to_draw", type=int, default=6)
    ap.add_argument("--no_video", action="store_true", help="do not render visualization video")
    args = ap.parse_args()

    setup_determinism(args.seed)
    os.makedirs(args.outdir, exist_ok=True)

    # Load video
    frames, duration_s, W, H, play_fps = load_frames_from_video(args.video, args.play_fps)
    if len(frames) == 0:
        raise RuntimeError("No frames decoded.")
    print(f"[INFO] Video loaded: {len(frames)} frames @ {play_fps:.3f} fps, size {W}x{H}, duration {duration_s:.2f}s")

    # Load embeddings DB
    db = load_embedding_database(args.scene_json, method=args.method)
    enc_fps = args.encoding_fps if args.encoding_fps > 0 else (db.get("encoding_fps") or play_fps)
    if not enc_fps or enc_fps <= 0:
        enc_fps = play_fps
    print(f"[INFO] Encoding fps: {enc_fps}")

    # Load model
    model, tokenizer, image_processor, model_name = load_model(args.model_path, device=args.device, precision=args.precision)
    print(f"[INFO] {model_name} model has been loaded")

    # Map frames for encoding
    sampled_idx = get_encoding_idx(frames, play_fps=play_fps, encoding_fps=enc_fps)
    print(f"[INFO] Sampled {len(sampled_idx)} frames for scoring (out of {len(frames)}).")

    # Predict highlight indices
    highlights, stats = predict_highlight_idx(
        model=model,
        image_processor=image_processor,
        device=args.device,
        precision=args.precision,
        video_frames=frames,
        sampled_frame_idx=sampled_idx,
        embedding_database=db,
        nms_window=args.nms_window,
        nms_threshold=args.nms_threshold,
        topk=args.topk,
    )

    # Predict clips from highlight frames
    clips = predict_highlight_clip(
        video_frames=frames,
        highlight_idx=highlights,
        pad_sec=args.clip_pad_sec,
        play_fps=play_fps,
        min_clip_sec=args.min_clip_sec,
        merge_if_overlap=True
    )

    # Save results
    base = Path(args.outdir) / (Path(f"{model_name}")) / (Path(args.video).stem + f"_highlights_{args.method}")
    json_out = str(base) + ".json"
    os.makedirs(os.path.dirname(json_out), exist_ok=True)
    with open(json_out, "w", encoding="utf-8") as f:
        json.dump({
            "highlights": highlights,
            "clips": clips,
            "play_fps": play_fps,
            "encoding_fps": enc_fps
        }, f, indent=2, ensure_ascii=False)
    print(f"[INFO] Wrote: {json_out}")

    if not args.no_video:
        mp4_out = str(base) + ".mp4"
        visualize_video_clipping(
            output_path=mp4_out,
            cos_sim_score_stats=stats,
            highlight_idx=highlights,
            video_highlight_clips=clips,
            video_frames=frames,
            play_fps=play_fps,
            max_tags_to_draw=args.max_tags_to_draw,
            method = args.method,
            model_name = model_name,
            params = {"nms_window": args.nms_window, "nms_threshold": args.nms_threshold}
        )
        print(f"[INFO] Wrote: {mp4_out}")

    # Also dump clipped segments list (frame indices & timestamp ranges) for convenience
    txt_out = str(base) + "_clips.txt"
    os.makedirs(os.path.dirname(txt_out), exist_ok=True)
    with open(txt_out, "w", encoding="utf-8") as f:
        for c in clips:
            s, e = c["start_idx"], c["end_idx"]
            ts = s / max(play_fps, 1e-6)
            te = e / max(play_fps, 1e-6)
            f.write(f"{s}-{e}  ({ts:.3f}s - {te:.3f}s)\n")
    print(f"[INFO] Wrote: {txt_out}")


if __name__ == "__main__":
    main()