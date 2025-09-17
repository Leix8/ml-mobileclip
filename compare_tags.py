#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Video frame retrieval visualization with MobileCLIP.
- Deterministic runs
- Safe MobileCLIP import
- Multiple prompt JSON support
- Visualization video with synced vertical lines
"""

import argparse
import os
import sys
import json
import time
import random
from pathlib import Path
from typing import List, Dict, Any, Tuple

import numpy as np
from PIL import Image

import torch
import torch.nn.functional as F
from tqdm import tqdm

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


# ---- Visualization / IO ----
import matplotlib
matplotlib.use("Agg")  # headless rendering
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

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


# -------------------- Utilities --------------------
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp", ".heic", ".HEIC"}


def natural_key(p: Path):
    import re
    s = p.name
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r'(\d+)', s)]


def load_prompt_jsons(paths: List[str]) -> List[Dict[str, Any]]:
    results = []
    for p in paths:
        with open(p, "r") as f:
            data = json.load(f)
        groups = []
        flat_tags, flat_refdirs = [], []
        for gname, items in data.items():
            if not isinstance(items, list):
                continue
            clean_items = []
            for it in items:
                tag = str(it.get("tag", "")).strip()
                ref = str(it.get("ref_dir", "")).strip()
                clean_items.append({"tag": tag, "ref_dir": ref})
                flat_tags.append(tag)
                flat_refdirs.append(ref)
            groups.append({"name": gname, "items": clean_items})
        results.append({
            "source_name": os.path.basename(p),
            "groups": groups,
            "flat_tags": flat_tags,
            "flat_refdirs": flat_refdirs
        })
    return results


def iter_frame_paths(frames_dir: str) -> List[Path]:
    root = Path(frames_dir)
    if not root.exists():
        raise FileNotFoundError(f"Frames directory not found: {frames_dir}")
    paths = [p for p in root.rglob("*") if p.suffix in IMG_EXTS]
    paths.sort(key=natural_key)
    return paths


def sample_indices(total_count: int, source_fps: float, target_fps: float) -> List[int]:
    if total_count <= 0:
        return []
    if target_fps <= 0:
        target_fps = 1.0
    if source_fps <= 0:
        source_fps = 30.0

    step = max(1, int(round(source_fps / target_fps)))
    idxs = list(range(0, total_count, step))
    if idxs[-1] != total_count - 1:
        idxs.append(total_count - 1)
    return idxs


def read_video_frames(video_path: str) -> Tuple[List[Image.Image], float]:
    if not os.path.isfile(video_path):
        raise FileNotFoundError(f"Video not found: {video_path}")

    frames = []
    source_fps = 30.0

    if _HAS_CV2:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise RuntimeError(f"Failed to open video: {video_path}")
        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps and fps > 0:
            source_fps = float(fps)
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(Image.fromarray(frame_rgb))
        cap.release()
    elif _HAS_IMAGEIO:
        reader = imageio.get_reader(video_path)
        try:
            meta = reader.get_meta_data()
            source_fps = float(meta.get("fps", 30.0))
        except Exception:
            source_fps = 30.0
        for frame in reader:
            frames.append(Image.fromarray(frame))
        reader.close()
    else:
        raise RuntimeError("Neither OpenCV nor imageio is available to read video.")

    return frames, source_fps


def write_video(path: str, frames_bgr: List[np.ndarray], fps: float):
    h, w = frames_bgr[0].shape[:2]
    if _HAS_CV2:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(path, fourcc, fps, (w, h))
        if not out.isOpened():
            raise RuntimeError("Failed to open VideoWriter.")
        for f in frames_bgr:
            out.write(f)
        out.release()
    elif _HAS_IMAGEIO:
        frames_rgb = [f[:, :, ::-1] for f in frames_bgr]
        imageio.mimsave(path, frames_rgb, fps=fps, quality=8)
    else:
        raise RuntimeError("Neither OpenCV nor imageio available for writing video.")


def pil_to_tensor_for_mobileclip(img: Image.Image, image_processor) -> torch.Tensor:
    return image_processor(img).unsqueeze(0)


# -------------------- Core Pipeline --------------------
def main():
    parser = argparse.ArgumentParser(description="Video frame retrieval visualization with MobileCLIP")
    parser.add_argument("--prompt_json", type=str, nargs="+", required=False)
    parser.add_argument("--embeddings", type=str, required=True)
    grp = parser.add_mutually_exclusive_group(required=True)
    grp.add_argument("--video", type=str, default=None)
    grp.add_argument("--frames", type=str, default=None)
    parser.add_argument("--fps", type=float, default=1.0)
    parser.add_argument("--output_dir", type=str, default="./output_viz")
    parser.add_argument("--model_path", type=str, default="./checkpoints/mobileclip_s0.pt")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--fig_width", type=int, default=1280)
    parser.add_argument("--fig_height", type=int, default=720)
    parser.add_argument("--dpi", type=int, default=100)
    args = parser.parse_args()

    setup_determinism(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    # ---- Load prompt JSONs ----
    
    prompt_specs = load_prompt_jsons(args.prompt_json)
    tag_counts = [len(ps["flat_tags"]) for ps in prompt_specs]
    num_rows = min(tag_counts)
    row_tag_labels = [prompt_specs[0]["flat_tags"][i] for i in range(num_rows)]

    # ---- Load frames ----
    if args.video is not None:
        frames_all, src_fps = read_video_frames(args.video)
        src_desc = f"video: {Path(args.video).name}"
    else:
        frame_paths = iter_frame_paths(args.frames)
        frames_all = [Image.open(p).convert("RGB") for p in frame_paths]
        src_fps = 30.0
        src_desc = f"frames_dir: {Path(args.frames).name} (assume 30 fps)"

    sel_idxs = sample_indices(len(frames_all), src_fps, args.fps)
    sampled_frames = [frames_all[i] for i in sel_idxs]
    T = len(sampled_frames)
    print(f"{T} frames have been loaded")

    # ---- Load MobileCLIP ----
    mobileclip = _import_mobileclip()
    model_name = os.path.splitext(os.path.basename(args.model_path))[0]
    model, _, image_processor = mobileclip.create_model_and_transforms(model_name)

    if os.path.isfile(args.model_path):
        try:
            sd = torch.load(args.model_path, map_location="cpu")
            if "state_dict" in sd:
                sd = sd["state_dict"]
            model.load_state_dict(sd, strict=False)
        except Exception as e:
            print(f"[WARN] Could not load weights: {e}")

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    model = model.to(device).eval()
    tokenizer = mobileclip.get_tokenizer(model_name)

    with open(args.embeddings, "r") as f:
        emb_json = json.load(f)
    entries = extract_tag_entries(emb_json)
    if len(entries) == 0:
        raise ValueError("No tag entries found in embeddings JSON.")
    
    def _get_vec(entry: Dict[str, Any], key_path: List[str]) -> torch.Tensor:
        vec = get_vector_from_entry(entry, key_path)  # np
        return F.normalize(torch.tensor(vec, device=args.device), dim=-1)
    
    def cosine_sim_batch(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        return (a * b).sum(dim=-1)

    text_t = None
    for entry in entries:
        tag_text = entry.get("tag", entry.get("id", "tag"))
        text_t = _get_vec(entry, ["embedding", "tag_embedding"])

        sim_text = cosine_sim_batch(frame_embs, text_t).cpu().numpy()

    # ---- Encode text ----
    with torch.no_grad():
        text_embeds = []
        for ps in tqdm(prompt_specs, desc = "encoding tags"):
            tags = [ps["flat_tags"][i] for i in range(num_rows)]
            toks = tokenizer(tags).to(device)
            te = model.encode_text(toks)
            te = F.normalize(te, dim=-1)
            text_embeds.append(te)

    # ---- Encode frames ----
    with torch.no_grad():
        img_embeds = []
        for img in tqdm(sampled_frames, desc = "encoding frames"):
            x = pil_to_tensor_for_mobileclip(img, image_processor).to(device)
            fe = model.encode_image(x)
            fe = F.normalize(fe, dim=-1)
            img_embeds.append(fe.squeeze(0).cpu())
        img_embeds = torch.stack(img_embeds, dim=0)

    sims = torch.zeros((num_rows, len(prompt_specs), T), dtype=torch.float32)
    for s, te in enumerate(text_t):
        s_mat = torch.matmul(img_embeds, te.cpu().T)
        sims[:, s, :] = s_mat.T

    npy_out_name = ("compare_tag_sim_" + (Path(args.video).stem if args.video else Path(args.frames).name) + ".npy")
    np.save(os.path.join(args.output_dir, npy_out_name), sims.numpy())
    # np.save(os.path.join(args.output_dir, "similarities.npy"), sims.numpy())

    # ---- Visualization ----
    fig_w_in = args.fig_width / args.dpi
    fig_h_in = args.fig_height / args.dpi
    fig = plt.figure(figsize=(fig_w_in, fig_h_in), dpi=args.dpi)
    gs = GridSpec(nrows=num_rows + 5, ncols=1, figure=fig)
    top_rows = max(3, int((num_rows + 5) * 0.55))
    ax_img = fig.add_subplot(gs[:top_rows, 0])
    plot_axes = [fig.add_subplot(gs[top_rows + r: top_rows + r + 1, 0]) for r in range(num_rows)]
    lines = []
    x_vals = np.arange(T)

    for r in range(num_rows):
        ax = plot_axes[r]
        per_row_lines = []
        for s in range(len(prompt_specs)):
            (ln,) = ax.plot(x_vals, sims[r, s, :].numpy(), linewidth=1.2)
            per_row_lines.append(ln)
        lines.append(per_row_lines)
        if len(prompt_specs) > 1:
            labels = [ps["groups"][0]["items"][r]["tag"] for ps in prompt_specs]
            ax.legend(labels, fontsize=8, loc="upper right", framealpha=0.6)
        ax.set_xlim(0, T - 1)
        ydata = sims[r, :, :].numpy().reshape(-1)
        ax.set_ylim(float(np.min(ydata)), float(np.max(ydata)))
        ax.set_ylabel("cos sim", fontsize=9)
        # ax.set_title(f"Tag {r+1}: {row_tag_labels[r]}", fontsize=10, pad=2)
        ax._vline = ax.axvline(0, linestyle="--", linewidth=1.0)
        ax.tick_params(axis='both', which='major', labelsize=8)

    ax_img.axis("off")
    suptitle = fig.suptitle("", fontsize=11)

    def render_frame(t: int, pil_img: Image.Image) -> np.ndarray:
        ax_img.clear()
        ax_img.axis("off")
        ax_img.imshow(pil_img)
        timestamp_sec = t / max(args.fps, 1e-6)
        meta = [
            f"Source: {src_desc}",
            f"Model: {model_name}",
            f"Device: {str(device)}",
            f"Target FPS: {args.fps:g} | Src FPS: {src_fps:.2f}",
            f"Frame idx: {t}/{T-1} | Time: {timestamp_sec:.2f}s"
        ]
        ax_img.set_title(" | ".join(meta), fontsize=10, pad=6)
        for r in range(num_rows):
            plot_axes[r]._vline.set_xdata([t, t])
        suptitle.set_text("Video Retrieval Similarity (cosine)")
        fig.canvas.draw()

        w, h = fig.canvas.get_width_height()
        argb = np.frombuffer(fig.canvas.tostring_argb(), dtype=np.uint8).reshape(h, w, 4)
        # ARGB → RGB
        rgb = argb[:, :, 1:4]
        return rgb[:, :, ::-1]   # to BGR

    out_name = ("compare_tag_viz_" + (Path(args.video).stem if args.video else Path(args.frames).name) + ".mp4")
    out_path = os.path.join(args.output_dir, out_name)
    bgr_frames = [render_frame(t, sampled_frames[t]) for t in range(T)]
    write_video(out_path, bgr_frames, fps=max(args.fps, 1.0))
    print(f"[OUT] Wrote visualization video: {out_path}")


if __name__ == "__main__":
    main()