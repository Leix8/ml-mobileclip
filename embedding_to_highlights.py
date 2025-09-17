#!/usr/bin/env python3
import argparse
import os
import sys
import json
import random
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional

import torch
import torch.nn.functional as F
from torch.cuda.amp import autocast
from PIL import Image, ImageOps, ImageDraw, ImageFont

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from tqdm import tqdm

try:
    import cv2
    _HAS_CV2 = True
except Exception:
    _HAS_CV2 = False
    try:
        import imageio
        _HAS_IMAGEIO = True
    except Exception:
        _HAS_IMAGEIO = False


# ================= Determinism =================
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
    torch.use_deterministic_algorithms(True)


# ================= MobileCLIP loader =================
def _import_mobileclip():
    try:
        import mobileclip
        return sys.modules["mobileclip"]
    except Exception as e:
        raise ImportError("Could not import 'mobileclip'. Please install/ensure it's on PYTHONPATH.") from e


# ================= IO helpers =================
def list_image_paths(dir_path: Path) -> List[Path]:
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
    return [p for p in sorted(dir_path.iterdir()) if p.is_file() and p.suffix.lower() in exts]


def timecode_from_seconds(sec: float) -> str:
    if sec is None:
        return ""
    msec = int(round(sec * 1000))
    hh = msec // 3600000
    mm = (msec % 3600000) // 60000
    ss = (msec % 60000) // 1000
    ms = msec % 1000
    return f"{hh:02d}:{mm:02d}:{ss:02d}.{ms:03d}"


def get_name_from_path(p: str) -> str:
    path = Path(p)
    if path.is_dir():
        return path.name
    else:
        return path.stem


# ================= Video / Frames loading (with mapping) =================
def sample_video_frames_with_map(video_path: str, target_fps: int) -> Tuple[List[Image.Image], List[str], List[int], float, int, int]:
    """
    Returns:
      pil_frames: sampled frames (downsampled)
      frame_names: synthetic names for sampled frames
      original_indices: original frame indices in source video (0-based)
      native_fps: fps reported by container
      stride: sampling stride in frames
      total_original_frames: number of frames in source video
    """
    if not _HAS_CV2:
        raise RuntimeError("OpenCV required for --video mode.")
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")
    native_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    stride = max(1, int(round(native_fps / max(1, target_fps))))

    pil_frames: List[Image.Image] = []
    frame_names: List[str] = []
    original_indices: List[int] = []

    idx = 0
    with tqdm(total=total_frames if total_frames > 0 else None,
              desc="Reading video frames", unit="f") as pbar:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if idx % stride == 0:
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                pil_frames.append(Image.fromarray(rgb))
                frame_names.append(f"frame_{idx:08d}.jpg")  # embed original idx in name
                original_indices.append(idx)
            idx += 1
            if total_frames > 0:
                pbar.update(1)
    cap.release()
    return pil_frames, frame_names, original_indices, float(native_fps), int(stride), int(total_frames)


def load_frames_dir_with_map(frames_dir: str, target_fps: int, assumed_base_fps: float = 30.0) -> Tuple[List[Image.Image], List[str], List[int], Optional[float], int, int]:
    """
    For a directory of images, we assume an effective base fps to define stride.
    Returns similar tuple to video loader, with native_fps=None and total_original_frames=len(all_paths).
    original_indices are indices in the sorted file list.
    """
    all_paths = list_image_paths(Path(frames_dir))
    if len(all_paths) == 0:
        raise RuntimeError(f"No images found in: {frames_dir}")

    stride = max(1, int(round(assumed_base_fps / max(1, target_fps))))
    selected_ids = [i for i in range(len(all_paths)) if i % stride == 0]

    pil_frames: List[Image.Image] = []
    frame_names: List[str] = []
    original_indices: List[int] = []

    for i in tqdm(selected_ids, desc="Reading frame images", unit="img"):
        p = all_paths[i]
        im = Image.open(p).convert("RGB")
        im = ImageOps.exif_transpose(im)
        pil_frames.append(im)
        frame_names.append(p.name)  # keep original file name
        original_indices.append(i)

    return pil_frames, frame_names, original_indices, None, int(stride), len(all_paths)


# ================= Encoders =================
@torch.no_grad()
def encode_images(model, image_processor, pil_images: List[Image.Image],
                  device: str, batch_size: int, use_amp: bool = True) -> torch.Tensor:
    if len(pil_images) == 0:
        return torch.empty(0, device=device)
    embs = []
    for i in tqdm(range(0, len(pil_images), batch_size), desc="Encoding frames", unit="batch"):
        imgs = [image_processor(im) for im in pil_images[i: i + batch_size]]
        pixel_batch = torch.stack(imgs, dim=0).to(device)
        with autocast(enabled=use_amp):
            feats = model.encode_image(pixel_batch)
        feats = F.normalize(feats, dim=-1)
        embs.append(feats)
    return torch.cat(embs, dim=0)


def cosine_sim_batch(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return (a * b).sum(dim=-1)


# ================= Plotting helpers =================
def draw_score_panel(
    scores_dict: Dict[str, np.ndarray],
    current_idx: int,
    topk_indices: Dict[str, np.ndarray],
    width_px: int = 1400,
    height_per_row: int = 120,
    dpi: int = 100,
    tag_labels: Dict[str, List[str]] = None
) -> Image.Image:
    methods = list(scores_dict.keys())
    R = len(methods)
    fig = plt.figure(figsize=(width_px / dpi, (R * height_per_row) / dpi), dpi=dpi)
    gs = fig.add_gridspec(R, 1, hspace=0.6)

    for r, m in enumerate(methods):
        ax = fig.add_subplot(gs[r])
        s = scores_dict[m]
        x = np.arange(len(s))
        ax.plot(x, s, marker=".", linewidth=1)
        ax.axvline(current_idx, color="red", linestyle="--", linewidth=2)

        ymin, ymax = float(s.min()), float(s.max())
        span = max(0.1, ymax - ymin)

        # top-k markers
        if m in topk_indices and topk_indices[m].size > 0:
            idxs = np.asarray(topk_indices[m], dtype=int)
            vals = s[idxs]
            ax.scatter(idxs, vals, s=24, marker="v", color="black", zorder=3)  # smaller marker
            for xi, yi in zip(idxs, vals):
                ax.text(xi, yi + 0.02 * span, "✂", fontsize=10,
                        ha="center", va="center", color="red", zorder=4)
                if tag_labels and m in tag_labels:
                    ax.text(xi, yi + 0.05 * span, tag_labels[m][xi],
                            fontsize=8, ha="center", va="bottom", alpha=0.8, color="blue")

        ax.set_ylim(ymin - 0.1 * span, ymax + 0.1 * span)
        ax.set_ylabel("cos")
        ax.set_title(m)
        ax.grid(True, alpha=0.3)
        if r == R - 1:
            ax.set_xlabel("frame idx")
        else:
            ax.tick_params(axis="x", which="both", labelbottom=False)

    fig.canvas.draw()
    buf = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
    plot_img = buf.reshape(fig.canvas.get_width_height()[::-1] + (4,))
    plt.close(fig)
    return Image.fromarray(plot_img[:, :, :3])


def load_font(size: int = 18) -> ImageFont.ImageFont:
    try:
        return ImageFont.truetype("DejaVuSans.ttf", size)
    except Exception:
        return ImageFont.load_default()


def make_info_panel(
    frame_img: Image.Image,
    meta_text: str,
    right_panel_width: int = 420,
    pad: int = 12,
    bg=(15, 15, 15),
    fg=(240, 240, 240),
) -> Image.Image:
    w_frame, h_frame = frame_img.size
    panel_w = w_frame + right_panel_width
    panel_h = h_frame
    canvas = Image.new("RGB", (panel_w, panel_h), bg)
    canvas.paste(frame_img, (0, 0))

    draw = ImageDraw.Draw(canvas)
    font = load_font(18)

    def wrap_text(text, font, max_w):
        words = text.split()
        lines = []
        cur = ""
        for w in words:
            test = (cur + " " + w).strip()
            if draw.textlength(test, font=font) <= max_w:
                cur = test
            else:
                if cur:
                    lines.append(cur)
                cur = w
        if cur:
            lines.append(cur)
        return "\n".join(lines)

    wrapped = "\n".join([wrap_text(line, font, right_panel_width - 2 * pad)
                         for line in meta_text.splitlines()])
    draw.multiline_text((w_frame + pad, pad), wrapped, font=font, fill=fg, spacing=4)

    return canvas


# ================= Embeddings JSON helpers =================
def extract_tag_entries(emb_json: Dict[str, Any]) -> List[Dict[str, Any]]:
    out = []
    for key, items in emb_json.items():
        if isinstance(items, list):
            for it in items:
                if isinstance(it, dict) and "embedding" in it:
                    out.append(it)
    return out


def get_vector_from_entry(entry: Dict[str, Any], path: List[str]) -> np.ndarray:
    d = entry
    for p in path:
        d = d[p]
    return np.array(d["value"], dtype=np.float32)


# ================= NMS (1D temporal) =================
def nms_1d(scores: np.ndarray, window: int, topk: int) -> np.ndarray:
    """
    scores: 1D scores per sampled frame
    window: suppression half-width (in sampled frames). Frames within +/- window are suppressed.
    topk: keep at most topk after suppression
    Returns kept indices (descending by score).
    """
    if len(scores) == 0:
        return np.array([], dtype=int)
    order = np.argsort(scores)[::-1]
    keep = []
    suppressed = np.zeros_like(scores, dtype=bool)

    for idx in order:
        if suppressed[idx]:
            continue
        keep.append(idx)
        if len(keep) >= topk:
            break
        # suppress neighbors
        lo = max(0, idx - window)
        hi = min(len(scores) - 1, idx + window)
        suppressed[lo:hi+1] = True
        suppressed[idx] = False  # keep the peak itself

    keep = np.array(keep, dtype=int)
    # ensure strictly sorted by score desc (already is)
    return keep


# ================= Main =================
def main():
    parser = argparse.ArgumentParser(
        description="Video highlight retrieval visualization with temporal NMS and result JSON export."
    )
    parser.add_argument("--embeddings", type=str, required=True)
    parser.add_argument("--frames", type=str, help="Directory of frames")
    parser.add_argument("--video", type=str, help="Video file path")
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--fps", type=int, default=5, help="Sampling fps for video or effective fps for frames directory")
    parser.add_argument("--topk", type=int, default=10)
    parser.add_argument("--bg_samples", type=int, default=128)
    parser.add_argument("--model_path", type=str, default="./checkpoints/mobileclip_s0.pt")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--no_amp", action="store_true")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--all_in_one", action="store_true",
                        help="If set, combine scores across tags (per method, keep highest score per frame and label tag).")
    parser.add_argument("--nms_window_frames", type=int, default=5,
                        help="NMS suppression half-width in sampled frames (used if --nms_window_sec is not provided or not applicable).")
    parser.add_argument("--nms_window_sec", type=float, default=None,
                        help="NMS suppression half-width in seconds (preferred for videos; converted to sampled frames).")
    args = parser.parse_args()

    setup_determinism(args.seed)

    # Load embeddings JSON
    with open(args.embeddings, "r") as f:
        emb_json = json.load(f)
    entries = extract_tag_entries(emb_json)
    if len(entries) == 0:
        raise ValueError("No tag entries found in embeddings JSON.")

    # Load MobileCLIP
    mobileclip = _import_mobileclip()
    model_name = os.path.splitext(os.path.basename(args.model_path))[0]
    model, _, image_processor = mobileclip.create_model_and_transforms(model_name, pretrained=args.model_path)
    model = model.to(args.device).eval()
    use_amp = not args.no_amp

    # Load frames (with mapping to original indices)
    src_type = None
    output_prefix = None
    pil_frames: List[Image.Image] = []
    frame_names: List[str] = []
    original_indices: List[int] = []
    native_fps: Optional[float] = None
    stride: int = 1
    total_original_frames: int = 0

    if args.video:
        src_type = "video"
        output_prefix = get_name_from_path(args.video)
        (pil_frames, frame_names, original_indices,
         native_fps, stride, total_original_frames) = sample_video_frames_with_map(args.video, args.fps)
    elif args.frames:
        src_type = "frames"
        output_prefix = get_name_from_path(args.frames)
        (pil_frames, frame_names, original_indices,
         native_fps, stride, total_original_frames) = load_frames_dir_with_map(args.frames, args.fps, assumed_base_fps=30.0)
    else:
        raise ValueError("Either --frames or --video must be given")

    # Encode frames
    frame_embs = encode_images(model, image_processor, pil_frames, args.device, args.batch_size, use_amp=use_amp)
    N = frame_embs.size(0)

    # Background centroid (for debias)
    bg_count = min(args.bg_samples, N)
    bg_idxs = np.sort(np.random.choice(np.arange(N), size=bg_count, replace=False))
    bg_centroid = F.normalize(frame_embs[torch.from_numpy(bg_idxs)].mean(dim=0), dim=-1)

    # Collect all scores per tag
    def _get_vec(entry: Dict[str, Any], key_path: List[str]) -> torch.Tensor:
        vec = get_vector_from_entry(entry, key_path)  # np
        return F.normalize(torch.tensor(vec, device=args.device), dim=-1)

    all_tag_scores: Dict[str, List[Tuple[np.ndarray, str]]] = {m: [] for m in ["Text", "TAF", "Cos-Push", "Debiased TAF", "Debiased Cos-Push"]}
    for entry in entries:
        tag_text = entry.get("tag", entry.get("id", "tag"))
        text_t = _get_vec(entry, ["embedding", "tag_embedding"])
        taf_t  = _get_vec(entry, ["embedding", "multi_ref_TAF_embedding", "data"])
        push_t = _get_vec(entry, ["embedding", "multi_ref_push_embedding", "data"])

        debiased_taf  = F.normalize(taf_t  - bg_centroid, dim=-1)
        debiased_push = F.normalize(push_t - bg_centroid, dim=-1)

        sim_text = cosine_sim_batch(frame_embs, text_t).cpu().numpy()
        sim_taf  = cosine_sim_batch(frame_embs, taf_t).cpu().numpy()
        sim_push = cosine_sim_batch(frame_embs, push_t).cpu().numpy()
        sim_taf_deb  = cosine_sim_batch(frame_embs, debiased_taf).cpu().numpy()
        sim_push_deb = cosine_sim_batch(frame_embs, debiased_push).cpu().numpy()

        all_tag_scores["Text"].append((sim_text, tag_text))
        all_tag_scores["TAF"].append((sim_taf, tag_text))
        all_tag_scores["Cos-Push"].append((sim_push, tag_text))
        all_tag_scores["Debiased TAF"].append((sim_taf_deb, tag_text))
        all_tag_scores["Debiased Cos-Push"].append((sim_push_deb, tag_text))

    # Combine across tags (per method, per frame keep highest tag score + label)
    combined_scores: Dict[str, np.ndarray] = {}
    combined_labels: Dict[str, List[str]] = {}
    for m, sims_tags in all_tag_scores.items():
        stacked = np.stack([s for s, _ in sims_tags], axis=1)  # [N, T]
        tag_names = [t for _, t in sims_tags]
        best_idx = stacked.argmax(axis=1)      # [N]
        best_scores = stacked.max(axis=1)      # [N]
        best_labels = [tag_names[j] for j in best_idx]
        combined_scores[m] = best_scores
        combined_labels[m] = best_labels

    # -------- NMS selection (compute effective window in sampled frames) --------
    # If time-based window provided and we know sampled fps, convert; else use frames window
    sampled_fps = None
    if src_type == "video" and native_fps and stride:
        sampled_fps = native_fps / stride
    elif src_type == "frames":
        sampled_fps = float(args.fps)

    if args.nms_window_sec is not None and sampled_fps is not None:
        nms_window_frames_eff = max(1, int(round(args.nms_window_sec * sampled_fps)))
    else:
        nms_window_frames_eff = max(1, int(args.nms_window_frames))

    # For visualization markers
    topk_map: Dict[str, np.ndarray] = {}
    # For JSON export (detailed highlight info)
    method_highlights: Dict[str, List[Dict[str, Any]]] = {}

    for m, scores in combined_scores.items():
        # NMS over the per-frame best scores
        kept = nms_1d(scores, window=nms_window_frames_eff, topk=args.topk)
        topk_map[m] = kept  # markers in plots

        # Build highlight entries using original frame idx & names
        highlights = []
        for rank, si in enumerate(kept, start=1):
            orig_idx = int(original_indices[si]) if si < len(original_indices) else int(si)
            fname = frame_names[si] if si < len(frame_names) else f"frame_{si:08d}.jpg"
            label = combined_labels[m][si]
            score = float(scores[si])

            ts = None
            if src_type == "video" and native_fps and orig_idx is not None:
                ts = orig_idx / native_fps

            highlights.append({
                "rank": rank,
                "sampled_frame_index": int(si),
                "original_frame_index": orig_idx,
                "frame_name": fname,
                "score": score,
                "tag": label,
                "timestamp_sec": (float(ts) if ts is not None else None),
                "timecode": (timecode_from_seconds(ts) if ts is not None else None),
            })
        method_highlights[m] = highlights

    # -------- CSV of full frame scores (unchanged behavior) --------
    import pandas as pd
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    csv_path = out_dir / f"{output_prefix}_scores.csv"
    df = pd.DataFrame({"frame_name": frame_names, "original_frame_index": original_indices})
    for m, arr in combined_scores.items():
        df[m] = arr
        df[m + "_BestTag"] = combined_labels[m]
    df.to_csv(csv_path, index=False)
    print(f"Saved CSV to {csv_path}")

    # -------- JSON export with metadata & NMS highlights --------
    json_out = {
        "input_metadata": {
            "source_type": src_type,                          # "video" | "frames"
            "video_path": args.video if src_type == "video" else None,
            "frames_dir": args.frames if src_type == "frames" else None,
            "native_fps": native_fps,
            "total_original_frames": total_original_frames,
            "frame_names_in_sampled_order": frame_names,      # sampled sequence order
            "original_indices_in_sampled_order": original_indices
        },
        "processing_metadata": {
            "model_name": model_name,
            "device": args.device,
            "batch_size": args.batch_size,
            "bg_samples": args.bg_samples,
            "target_fps": args.fps,
            "sampling_stride_frames": stride,
            "sampled_fps": sampled_fps,
            "num_sampled_frames": N,
            "all_in_one": True,  # scores combined across tags per method
            "nms_window_frames_used": nms_window_frames_eff,
            "nms_window_sec_requested": args.nms_window_sec,
            "seed": args.seed
        },
        "results": method_highlights
    }
    results_json_path = out_dir / f"{output_prefix}_highlights.json"
    with open(results_json_path, "w") as f:
        json.dump(json_out, f, indent=2)
    print(f"Saved highlight results JSON to {results_json_path}")

    # -------- Visualization video with markers at NMS-selected highlights --------
    # Simple resize helper
    sample_im = pil_frames[0]
    target_h = 360
    scale = target_h / sample_im.height
    target_w = int(sample_im.width * scale)
    def resize_img(im): return im.resize((target_w, target_h), Image.BILINEAR)

    # Compose video
    sample_upper = make_info_panel(resize_img(sample_im), "Combined across tags (NMS selected markers)")
    width_upper, height_upper = sample_upper.size
    sample_plot = draw_score_panel(combined_scores, 0, topk_map, width_px=width_upper, height_per_row=120, tag_labels=combined_labels)
    width_plot, height_plot = sample_plot.size
    canvas_w, canvas_h = width_upper, height_upper + height_plot

    out_video_path = out_dir / f"{output_prefix}_highlights.mp4"
    if _HAS_CV2:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        vw = cv2.VideoWriter(str(out_video_path), fourcc, args.fps, (canvas_w, canvas_h))
    elif _HAS_IMAGEIO:
        vw = imageio.get_writer(str(out_video_path), fps=args.fps, codec="libx264", quality=8)
    else:
        raise RuntimeError("No video writer available")

    for idx, (im, fname) in enumerate(tqdm(zip(pil_frames, frame_names), total=N, desc="Composing video", unit="f")):
        header_txt = f"Frame {idx+1}/{N} | sampled_idx={idx} | orig_idx={original_indices[idx]} | {fname}"
        upper_panel = make_info_panel(resize_img(im), header_txt)
        plot_panel = draw_score_panel(combined_scores, idx, topk_map, width_px=canvas_w, height_per_row=120, tag_labels=combined_labels)
        canvas = Image.new("RGB", (canvas_w, canvas_h), (0, 0, 0))
        canvas.paste(upper_panel, (0, 0))
        canvas.paste(plot_panel, (0, height_upper))
        frame_np = np.array(canvas)
        if _HAS_CV2:
            vw.write(frame_np[:, :, ::-1])
        else:
            vw.append_data(frame_np)
    if _HAS_CV2:
        vw.release()
    else:
        vw.close()
    print(f"Saved highlight video to {out_video_path}")


if __name__ == "__main__":
    main()