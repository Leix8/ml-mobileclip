#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Video frame retrieval visualization with MobileCLIP (AMP enabled, OpenCV only).
- Deterministic runs
- Safe MobileCLIP import
- Multiple prompt JSON support
- Cosine similarity saved to .npy
- Visualization video with synced vertical lines
"""

import argparse
import os
import sys
import json
import random
from pathlib import Path
from typing import List, Dict, Any, Tuple

import numpy as np
from PIL import Image

import torch
import torch.nn.functional as F
from torch.cuda.amp import autocast
from tqdm import tqdm

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec


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


def read_video_cv2(video_path: str) -> Tuple[List[Image.Image], float]:
    if not os.path.isfile(video_path):
        raise FileNotFoundError(f"Video not found: {video_path}")
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frames = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(Image.fromarray(frame_rgb))
    cap.release()
    return frames, float(fps)


def pil_to_tensor_for_mobileclip(img: Image.Image, image_processor) -> torch.Tensor:
    return image_processor(img).unsqueeze(0)


# -------------------- Core Pipeline --------------------
def main():
    parser = argparse.ArgumentParser(description="Video frame retrieval visualization with MobileCLIP + AMP (OpenCV only)")
    parser.add_argument("--prompt_json", type=str, nargs="+", required=True)
    grp = parser.add_mutually_exclusive_group(required=True)
    grp.add_argument("--video", type=str, default=None)
    grp.add_argument("--frames", type=str, default=None)
    parser.add_argument("--fps", type=float, default=1.0)
    parser.add_argument("--output_dir", type=str, default="./output_viz")
    parser.add_argument("--model_path", type=str, default="./checkpoints/mobileclip_s0.pt")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--no_amp", action="store_true", help="Disable AMP, use strict float32")
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
        frames_all, src_fps = read_video_cv2(args.video)
        src_desc = f"video: {Path(args.video).name}"
    else:
        frame_paths = iter_frame_paths(args.frames)
        frames_all = [Image.open(p).convert("RGB") for p in frame_paths]
        src_fps = 30.0
        src_desc = f"frames_dir: {Path(args.frames).name} (assume 30 fps)"

    step = max(1, int(round(src_fps / args.fps)))
    sel_idxs = list(range(0, len(frames_all), step))
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

    use_amp = not args.no_amp

    # ---- Encode text ----
    with torch.no_grad():
        text_embeds = []
        for ps in tqdm(prompt_specs, desc="encoding tags"):
            tags = [ps["flat_tags"][i] for i in range(num_rows)]
            toks = tokenizer(tags).to(device)  # must stay long
            with autocast(enabled=use_amp):
                te = model.encode_text(toks)
            te = F.normalize(te.float(), dim=-1)
            text_embeds.append(te)

    # ---- Encode frames ----
    with torch.no_grad():
        img_embeds = []
        for img in tqdm(sampled_frames, desc="encoding frames"):
            x = pil_to_tensor_for_mobileclip(img, image_processor).to(device).float()
            with autocast(enabled=use_amp):
                fe = model.encode_image(x)
            fe = F.normalize(fe.float(), dim=-1)
            img_embeds.append(fe.squeeze(0).cpu())
        img_embeds = torch.stack(img_embeds, dim=0)

    sims = torch.zeros((num_rows, len(prompt_specs), T), dtype=torch.float32)
    for s, te in enumerate(text_embeds):
        s_mat = torch.matmul(img_embeds, te.cpu().T)
        sims[:, s, :] = s_mat.T

    # ---- Save sims to .npy ----
    npy_out = Path(args.output_dir) / f"compare_tag_sim_{src_desc.replace(' ', '_')}.npy"
    np.save(npy_out, sims.numpy())
    print(f"[OUT] Saved similarities to {npy_out}")
    # ---- Visualization video ----
    fig_w_in = args.fig_width / args.dpi
    fig_h_in = args.fig_height / args.dpi
    fig = plt.figure(figsize=(fig_w_in, fig_h_in), dpi=args.dpi)
    gs = GridSpec(nrows=num_rows + 5, ncols=1, figure=fig)
    top_rows = max(3, int((num_rows + 5) * 0.55))
    ax_img = fig.add_subplot(gs[:top_rows, 0])
    plot_axes = [fig.add_subplot(gs[top_rows + r: top_rows + r + 1, 0]) for r in range(num_rows)]
    x_vals = np.arange(T)

    ax_img.axis("off")
    suptitle = fig.suptitle("", fontsize=11)

    def render_frame(t: int, pil_img: Image.Image) -> np.ndarray:
        ax_img.clear()
        ax_img.axis("off")
        ax_img.imshow(pil_img)

        # update text above
        timestamp_sec = t / max(args.fps, 1e-6)
        meta = [
            f"Source: {src_desc}",
            f"Model: {model_name}",
            f"Device: {str(device)}",
            f"Target FPS: {args.fps:g} | Src FPS: {src_fps:.2f}",
            f"Frame idx: {t}/{T-1} | Time: {timestamp_sec:.2f}s"
        ]
        ax_img.set_title(" | ".join(meta), fontsize=10, pad=6)

        # draw similarity curves fresh for this frame
        for r in range(num_rows):
            ax = plot_axes[r]
            ax.clear()
            for s in range(len(prompt_specs)):
                y = sims[r, s, :].numpy().clip(-1, 1)  # safe range
                ax.plot(x_vals, y, linewidth=1.2)
            if len(prompt_specs) > 1:
                labels = [ps["groups"][0]["items"][r]["tag"] for ps in prompt_specs]
                ax.legend(labels, fontsize=8, loc="upper right", framealpha=0.6)
            ax.set_xlim(0, T - 1)
            ymin, ymax = np.nanmin(y), np.nanmax(y)
            if not np.isfinite(ymin) or not np.isfinite(ymax) or ymin == ymax:
                ymin, ymax = -1.0, 1.0
            ax.set_ylim(ymin, ymax)
            ax.set_ylabel("cos sim", fontsize=9)
            ax.axvline(t, linestyle="--", linewidth=1.0, color="red")

        suptitle.set_text("Video Retrieval Similarity (cosine)")
        fig.canvas.draw()

        w, h = fig.canvas.get_width_height()
        argb = np.frombuffer(fig.canvas.tostring_argb(), dtype=np.uint8).reshape(h, w, 4)
        rgb = argb[:, :, 1:4]
        return rgb[:, :, ::-1]  # RGB→BGR

    out_name = ("compare_tag_viz_" + (Path(args.video).stem if args.video else Path(args.frames).name) + ".mp4")
    out_path = os.path.join(args.output_dir, out_name)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    vw = cv2.VideoWriter(out_path, fourcc, args.fps, (args.fig_width, args.fig_height))

    for t in tqdm(range(T), desc="Composing video"):
        frame_bgr = render_frame(t, sampled_frames[t])
        vw.write(frame_bgr)
    vw.release()
    print(f"[OUT] Wrote visualization video: {out_path}")


if __name__ == "__main__":
    main()