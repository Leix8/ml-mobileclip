#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Category-aware video highlight proposal & visualization (MobileCLIP local).

Features
--------
- MobileCLIP (local) embeddings:
    - Tags (text) ↔ Frames
    - Ref Images ↔ Frames
- FastVLM caption-assisted tag selection (optional):
    - Produce a single video caption, rank tags by caption similarity,
      keep top-k, then compute per-frame curves.
- Peak detection to propose highlight timestamps.
- Visualization video: top = frames; bottom = similarity curves for each enabled method.
- CSV export of plotted series; JSON export of highlight proposals.

Dependencies
------------
pip install numpy pillow torch matplotlib opencv-python
# For caption fallback (if not providing fastvlm_impl):
pip install transformers accelerate

You must have your local `mobileclip` module accessible, with:
    model, _, image_processor = mobileclip.create_model_and_transforms(model_name, pretrained=model_path)
    tokenizer = mobileclip.get_tokenizer(model_name)
and the model exposes .encode_image() and .encode_text().

Usage (examples)
----------------
# Tags via local MobileCLIP, tags in JSON (category->tags[])
python video_highlights.py \
  --model_path ./checkpoints/mobileclip_s0.pt \
  --device cuda:0 \
  --frame_dir /data/my_video_frames \
  --tags ./tags_v1_1_enhanced_specified.json \
  --use_tags \
  --output_dir ./highlight_detection

# Caption-assisted + tags (still MobileCLIP for text embeddings)
python video_highlights.py \
  --model_path ./checkpoints/mobileclip_s0.pt \
  --device cuda:0 \
  --frames_dir /data/my_video_frames \
  --tags ./tags_v1_1_enhanced_specified.json \
  --use_caption --topk_from_caption 5 \
  --fastvlm_impl ./fastvlm_impl.py \
  --output_dir ./highlight_detection

# Reference images only
python video_highlights.py \
  --model_path ./checkpoints/mobileclip_s0.pt \
  --device cuda:0 \
  --frames_dir /data/my_video_frames \
  --use_ref_images --ref_images_dir ./ref_imgs \
  --output_dir ./highlight_detection
"""

import argparse
import csv
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np
from PIL import Image

import torch
import torch.nn.functional as F

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ==========================
# MobileCLIP (local) loader
# ==========================

@dataclass
class MobileCLIPConfig:
    model_name: str
    model_path: str
    device: str = "cuda:0"


def load_mobileclip(cfg: MobileCLIPConfig):
    """
    Uses local `mobileclip` package.

    Expects:
      model, _, image_processor = mobileclip.create_model_and_transforms(model_name, pretrained=model_path)
      tokenizer = mobileclip.get_tokenizer(model_name)

    Returns:
      encode_image(pil_list) -> (B,D) L2-normalized tensor
      encode_text(texts)     -> (B,D) L2-normalized tensor
      image_processor (callable for PIL -> tensor)
    """
    try:
        import mobileclip
    except Exception as e:
        print("ERROR: Could not import your local `mobileclip` module. Make sure it's on PYTHONPATH.", file=sys.stderr)
        raise e

    model, _, image_processor = mobileclip.create_model_and_transforms(
        cfg.model_name, pretrained=cfg.model_path
    )
    model = model.to(cfg.device).eval()
    tokenizer = mobileclip.get_tokenizer(cfg.model_name)

    @torch.no_grad()
    def encode_image(pil_list: List[Image.Image]) -> torch.Tensor:
        imgs = torch.stack([image_processor(im) for im in pil_list]).to(cfg.device)
        feats = model.encode_image(imgs)
        feats = F.normalize(feats, dim=-1)
        return feats

    @torch.no_grad()
    def encode_text(texts: List[str]) -> torch.Tensor:
        toks = tokenizer(texts).to(cfg.device)
        feats = model.encode_text(toks)
        feats = F.normalize(feats, dim=-1)
        return feats

    return encode_image, encode_text, image_processor


# ==========================
# FastVLM caption wrapper
# ==========================

class FastVLMCaptioner:
    """
    Plug your FastVLM via --fastvlm_impl (path to python file exposing class FastVLMImpl(device)
    with method caption_video(frames) -> str).
    Otherwise falls back to BLIP base for a rough caption.
    """
    def __init__(self, device="cuda:0", impl_path=None):
        self.device = device
        self.impl = None
        if impl_path:
            import importlib.util
            spec = importlib.util.spec_from_file_location("fastvlm_impl", impl_path)
            m = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(m)  # type: ignore
            self.impl = m.FastVLMImpl(device=device)
        else:
            try:
                from transformers import BlipProcessor, BlipForConditionalGeneration
                self.processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
                self.model = BlipForConditionalGeneration.from_pretrained(
                    "Salesforce/blip-image-captioning-base"
                ).to(device)
                self.model.eval()
            except Exception as e:
                print("Caption fallback unavailable. Provide --fastvlm_impl.", file=sys.stderr)
                raise e

    @torch.no_grad()
    def caption_video(self, frames: List[Image.Image]) -> str:
        if self.impl is not None:
            return self.impl.caption_video(frames)

        # Fallback: caption a small subset & join
        if len(frames) > 16:
            idx = np.linspace(0, len(frames) - 1, 8).astype(int).tolist()
            sample = [frames[i] for i in idx]
        else:
            sample = frames

        caps = []
        for im in sample:
            inputs = self.processor(images=im, return_tensors="pt").to(self.model.device)
            out = self.model.generate(**inputs, max_new_tokens=20)
            cap = self.processor.decode(out[0], skip_special_tokens=True)
            caps.append(cap)

        return " ".join(dict.fromkeys(caps)).strip()


# ==========================
# IO helpers
# ==========================

def load_frames(frames_dir: Path, max_frames: int = 0) -> Tuple[List[Path], List[Image.Image]]:
    paths = sorted([p for p in frames_dir.iterdir() if p.suffix.lower() in (".jpg", ".jpeg", ".png", ".bmp", ".webp")])
    if max_frames and max_frames > 0:
        paths = paths[:max_frames]
    images = [Image.open(p).convert("RGB") for p in paths]
    print(f"{len(images)} frame images have been loaded.")
    return paths, images


def load_reference_images(ref_dir: Path) -> Tuple[List[Path], List[Image.Image]]:
    if not ref_dir:
        return [], []
    paths = sorted([p for p in ref_dir.iterdir() if p.suffix.lower() in (".jpg", ".jpeg", ".png", ".bmp", ".webp")])
    images = [Image.open(p).convert("RGB") for p in paths]
    return paths, images


# ---- Tags readers: .json (category->tags[]) or .txt (one per line) ----

def load_tags_txt(path: str) -> Tuple[List[str], List[str], List[str]]:
    tags_plain, tag_labels, tag_category = [], [], []
    with open(path, "r") as f:
        for line in f:
            t = line.strip().strip(",")
            if not t:
                continue
            tags_plain.append(t)
            tag_labels.append(t)
            tag_category.append("")
    if not tags_plain:
        raise ValueError("No tags found in .txt.")
    return tags_plain, tag_labels, tag_category


def load_tags_json(
    path: str, allow_categories: List[str] = None, max_per_category: int = None
) -> Tuple[List[str], List[str], List[str]]:
    with open(path, "r") as f:
        obj = json.load(f)
    if not isinstance(obj, dict):
        raise ValueError("Expected JSON dict {category: [tags...]}.")
    allow = set([c.strip() for c in allow_categories]) if allow_categories else None
    tags_plain, tag_labels, tag_category = [], [], []
    for cat, tags in obj.items():
        if allow and cat not in allow:
            continue
        picked = list(tags)
        if max_per_category and max_per_category > 0:
            picked = picked[:max_per_category]
        for t in picked:
            t = str(t).strip()
            if not t:
                continue
            tags_plain.append(t)
            tag_labels.append(f"{cat}: {t}")
            tag_category.append(cat)
    if not tags_plain:
        raise ValueError("No tags after filters in JSON.")
    return tags_plain, tag_labels, tag_category


def load_tags_any(
    path: str, allow_categories: List[str] = None, max_per_category: int = None
) -> Tuple[List[str], List[str], List[str]]:
    if path.lower().endswith(".json"):
        return load_tags_json(path, allow_categories, max_per_category)
    return load_tags_txt(path)


# ==========================
# Math & detection
# ==========================

def cosine_sim_mat(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    return A @ B.t()  # assume already L2-normalized


def top3_indices_overall(curves: np.ndarray) -> List[int]:
    if curves.shape[1] <= 3:
        return list(range(curves.shape[1]))
    scores = curves.max(axis=0)
    return np.argsort(-scores)[:3].tolist()


def detect_peaks_from_curve(curve: np.ndarray, fps: float, min_separation_s=2.0, min_percentile=95) -> List[int]:
    if len(curve) < 3:
        return []
    thresh = np.percentile(curve, min_percentile)
    w = max(1, int(min_separation_s * fps // 2))
    peaks = []
    for i in range(len(curve)):
        if curve[i] < thresh:
            continue
        lo = max(0, i - w); hi = min(len(curve), i + w + 1)
        if curve[i] == curve[lo:hi].max():
            if peaks and (i - peaks[-1]) < int(min_separation_s * fps):
                if curve[i] > curve[peaks[-1]]:
                    peaks[-1] = i
            else:
                peaks.append(i)
    return peaks


def write_csv_curves(path: Path, header: List[str], rows: List[List[float]]):
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)


# ==========================
# Core pipelines (3 methods)
# ==========================

def method_tags_vs_frames(
    encode_img, encode_txt, frames: List[Image.Image],
    tags_plain: List[str], tag_labels: List[str],
    device="cuda:0", batch=32
) -> Dict:
    with torch.no_grad():
        feats_frames = []
        for i in range(0, len(frames), batch):
            feats_frames.append(encode_img(frames[i:i+batch]).float().to("cpu"))
        F_frames = torch.cat(feats_frames, dim=0)  # (T, D)

        T_text = encode_txt(tags_plain).float().to("cpu")  # (K, D)

        S = cosine_sim_mat(F_frames, T_text).numpy()       # (T, K)

    return {"ref_names": tag_labels, "curves": S}


def method_caption_assisted(
    encode_img, encode_txt, captioner: FastVLMCaptioner,
    frames: List[Image.Image], tags_plain: List[str], tag_labels: List[str],
    device="cuda:0", batch=32, topk_from_caption=5
) -> Dict:
    caption = captioner.caption_video(frames).strip()
    if not caption:
        raise RuntimeError("Empty caption produced by the captioner.")

    with torch.no_grad():
        cap_emb = encode_txt([caption]).float().to("cpu")           # (1, D)
        tag_emb = encode_txt(tags_plain).float().to("cpu")          # (K, D)
        tag_scores = (cap_emb @ tag_emb.t()).squeeze(0).numpy()     # (K,)

    order = np.argsort(-tag_scores)[:max(1, min(topk_from_caption, len(tags_plain)))]
    sel_plain  = [tags_plain[i] for i in order.tolist()]
    sel_labels = [tag_labels[i] for i in order.tolist()]

    out = method_tags_vs_frames(encode_img, encode_txt, frames, sel_plain, sel_labels,
                                device=device, batch=batch)
    out["caption"] = caption
    out["caption_tag_ranking"] = [(tag_labels[i], float(tag_scores[i])) for i in order.tolist()]
    return out


def method_refimages_vs_frames(
    encode_img, frames: List[Image.Image], ref_imgs: List[Image.Image],
    ref_names: List[str], device="cuda:0", batch=32
) -> Dict:
    with torch.no_grad():
        feats_frames = []
        for i in range(0, len(frames), batch):
            feats_frames.append(encode_img(frames[i:i+batch]).float().to("cpu"))
        F_frames = torch.cat(feats_frames, dim=0)  # (T, D)

        feats_refs = []
        for i in range(0, len(ref_imgs), batch):
            feats_refs.append(encode_img(ref_imgs[i:i+batch]).float().to("cpu"))
        R = torch.cat(feats_refs, dim=0)  # (M, D)

        S = cosine_sim_mat(F_frames, R).numpy()  # (T, M)

    return {"ref_names": ref_names, "curves": S}


# ==========================
# Visualization
# ==========================

def render_plot_panel(method_curves: Dict[str, Dict], idx: int, fps: float,
                      width: int = 1280, height: int = 360) -> np.ndarray:
    T = next(iter(method_curves.values()))["curves"].shape[0] if method_curves else idx+1
    t_axis = np.arange(T) / max(1e-6, fps)

    n_methods = len(method_curves)
    if n_methods == 0:
        return np.zeros((height, width, 3), dtype=np.uint8)

    fig_h = 2.6 * n_methods
    fig = plt.figure(figsize=(width/160, fig_h), dpi=160)
    gs = fig.add_gridspec(n_methods, 1, hspace=0.5)
    method_titles = {
        "tags": "MobileCLIP: Tags↔Frames (category-aware)",
        "caption": "FastVLM-Caption-Assisted (Tags↔Frames)",
        "refs": "MobileCLIP: RefImages↔Frames"
    }

    for r, (mkey, mdata) in enumerate(method_curves.items()):
        ax = fig.add_subplot(gs[r, 0])
        curves = mdata["curves"]; names = mdata["ref_names"]
        top_idx = top3_indices_overall(curves)
        for j in top_idx:
            y = curves[:, j]
            ax.plot(t_axis, y, linewidth=1.2, alpha=0.85, label=names[j])
        ax.axvline(x=idx / max(1e-6, fps), linestyle="--", linewidth=1.0)
        ax.set_xlim(0, t_axis[-1] if len(t_axis) else 1.0)
        ax.set_ylim(-0.05, 1.05)
        ax.set_xlabel("time (s)")
        ax.set_ylabel("cos sim")
        ax.set_title(method_titles.get(mkey, mkey))
        ax.legend(fontsize=8, loc="upper right", ncol=1)

    # --- draw and read pixels from the canvas (backend-safe) ---
    fig.canvas.draw()

    # Preferred: RGBA buffer → np.ndarray (H, W, 4)
    try:
        plot_rgba = np.asarray(fig.canvas.buffer_rgba())
    except Exception:
        # Fallback for very old/new combos
        try:
            # Some builds expose renderer.buffer_rgba()
            plot_rgba = np.asarray(fig.canvas.renderer.buffer_rgba())
        except Exception:
            # Last resort: tostring_argb → reshape → ARGB→RGBA
            w, h = fig.canvas.get_width_height()
            argb = np.frombuffer(fig.canvas.tostring_argb(), dtype=np.uint8)
            argb = argb.reshape((h, w, 4))
            # ARGB → RGBA (roll channels)
            plot_rgba = np.roll(argb, -1, axis=2)

    plt.close(fig)  # free the figure ASAP

    # Convert RGBA → BGR for cv2 stacking
    plot_bgr = cv2.cvtColor(plot_rgba, cv2.COLOR_RGBA2BGR)

    # Resize to target panel size
    plot_bgr = cv2.resize(plot_bgr, (width, height), interpolation=cv2.INTER_AREA)
    return plot_bgr


def stack_frame_with_plot(frame_bgr: np.ndarray, plot_bgr: np.ndarray) -> np.ndarray:
    w = max(frame_bgr.shape[1], plot_bgr.shape[1])

    def pad_to_width(img):
        if img.shape[1] == w:
            return img
        pad = np.zeros((img.shape[0], w - img.shape[1], 3), dtype=img.dtype)
        return np.concatenate([img, pad], axis=1)

    return np.vstack([pad_to_width(frame_bgr), pad_to_width(plot_bgr)])


# ==========================
# Main / Orchestration
# ==========================

def main():
    # --- argparse (compatible with your pattern) ---
    ap = argparse.ArgumentParser(description="Detect & visualize highlight moments (category-aware, local MobileCLIP).")

    # keep your style/flags
    ap.add_argument("--model_path", type=str, default="./checkpoints/mobileclip_s0.pt",
                    help="path to MobileCLIP checkpoint (.pt)")
    ap.add_argument("--device", type=str, default="cuda:0", help="device, e.g., cuda:0 / cpu")
    ap.add_argument("--output_dir", type=str, default="./highlight_detection", help="dir to save results")
    ap.add_argument("--video_dir", type=str, default=None, help="(optional) original video dir (unused)")
    ap.add_argument("--frame_dir", type=str, default=None, help="frames dir (alias of --frames_dir)")
    ap.add_argument("--frames_dir", type=Path, default=None, help="frames dir (preferred)")
    ap.add_argument("--tags", type=str, default="../Koala-36M/scenes_and_tags_v2.json",
                    help="path to tags file: supports .json (category->tags[]) or .txt (one tag per line)")

    # original script flags that still matter
    ap.add_argument("--fps", type=float, default=25.0)
    ap.add_argument("--resize_width", type=int, default=1280)
    ap.add_argument("--max_frames", type=int, default=0)

    # enable/disable methods
    ap.add_argument("--use_tags", action="store_true")
    ap.add_argument("--use_caption", action="store_true")
    ap.add_argument("--use_ref_images", action="store_true")
    ap.add_argument("--ref_images_dir", type=Path)

    # caption specifics
    ap.add_argument("--fastvlm_impl", type=Path, default=None)
    ap.add_argument("--topk_from_caption", type=int, default=5)

    # tag filtering (when .json)
    ap.add_argument("--tag_categories", type=str, help="comma-separated allow-list (for .json)")
    ap.add_argument("--max_tags_per_category", type=int, default=0)

    # batching
    ap.add_argument("--batch", type=int, default=32)

    # outputs (under output_dir)
    ap.add_argument("--out_video", type=str, default="highlights_vis.mp4")
    ap.add_argument("--out_json",  type=str, default="highlights.json")
    ap.add_argument("--out_csv",   type=str, default="curves.csv")

    # peaks
    ap.add_argument("--peak_min_separation_s", type=float, default=2.0)
    ap.add_argument("--peak_percentile", type=float, default=95.0)

    args = ap.parse_args()
    args.model_name = os.path.splitext(os.path.basename(args.model_path))[0]

    # resolve frames_dir (support both flags)
    if args.frames_dir is None and args.frame_dir is not None:
        args.frames_dir = Path(args.frame_dir)
    if args.frames_dir is None:
        raise SystemExit("Please provide --frames_dir or --frame_dir")

    # create output_dir and resolve output paths
    os.makedirs(args.output_dir, exist_ok=True)
    args.out_video = Path(args.output_dir) / args.out_video
    args.out_json  = Path(args.output_dir) / args.out_json
    args.out_csv   = Path(args.output_dir) / args.out_csv

    # check that at least one method is enabled
    if not (args.use_tags or args.use_caption or args.use_ref_images):
        print("Enable at least one method: --use_tags / --use_caption / --use_ref_images", file=sys.stderr)
        sys.exit(1)

    # --- Load frames ---
    frame_paths, frame_imgs = load_frames(args.frames_dir, max_frames=args.max_frames)
    if not frame_imgs:
        print("No frames found.", file=sys.stderr)
        sys.exit(1)

    # --- Prepare MobileCLIP ---
    mc_cfg = MobileCLIPConfig(model_name=args.model_name, model_path=args.model_path, device=args.device)
    encode_img, encode_txt, _ = load_mobileclip(mc_cfg)
    print(f"mobileCLIP model has been loaded")
    
    # --- Prepare methods ---
    methods: Dict[str, Dict] = {}

    # Load tags (if any tag-based mode is on)
    tags_plain: List[str] = []
    tag_labels: List[str] = []
    tag_category: List[str] = []
    if args.use_tags or args.use_caption:
        if not args.tags:
            print("Tag-based modes require --tags", file=sys.stderr); sys.exit(1)
        allow_cats = [s.strip() for s in args.tag_categories.split(",")] if args.tag_categories else None
        max_per_cat = args.max_tags_per_category if args.max_tags_per_category and args.max_tags_per_category > 0 else None
        tags_plain, tag_labels, tag_category = load_tags_any(args.tags, allow_categories=allow_cats, max_per_category=max_per_cat)

    # Method 1: MobileCLIP Tags↔Frames
    if args.use_tags:
        out1 = method_tags_vs_frames(encode_img, encode_txt, frame_imgs, tags_plain, tag_labels,
                                     device=args.device, batch=args.batch)
        methods["tags"] = out1

    # Method 2: FastVLM Caption-assisted (then Tags↔Frames)
    if args.use_caption:
        captioner = FastVLMCaptioner(device=args.device, impl_path=str(args.fastvlm_impl) if args.fastvlm_impl else None)
        out2 = method_caption_assisted(encode_img, encode_txt, captioner, frame_imgs,
                                       tags_plain, tag_labels,
                                       device=args.device, batch=args.batch,
                                       topk_from_caption=args.topk_from_caption)
        methods["caption"] = out2

    # Method 3: MobileCLIP RefImages↔Frames
    if args.use_ref_images:
        if not args.ref_images_dir:
            print("--use_ref_images requires --ref_images_dir", file=sys.stderr); sys.exit(1)
        ref_paths, ref_imgs = load_reference_images(args.ref_images_dir)
        if len(ref_imgs) == 0:
            print("No reference images found in --ref_images_dir.", file=sys.stderr); sys.exit(1)
        out3 = method_refimages_vs_frames(encode_img, frame_imgs, ref_imgs,
                                          [p.stem for p in ref_paths],
                                          device=args.device, batch=args.batch)
        methods["refs"] = out3

    # --- Highlight proposals per method ---
    proposals = {}
    T = len(frame_imgs)
    fps = args.fps

    csv_header = ["frame_idx", "time_s"]
    csv_rows: List[List[float]] = []
    top3_per_method: Dict[str, List[int]] = {}

    for key, obj in methods.items():
        curves = obj["curves"]  # (T, M)
        idx3 = top3_indices_overall(curves)
        top3_per_method[key] = idx3

        # aggregate (max over top-3) and detect peaks
        top3_curves = curves[:, idx3]
        agg = top3_curves.max(axis=1)
        peak_idx = detect_peaks_from_curve(agg, fps, args.peak_min_separation_s, args.peak_percentile)

        # also include which reference wins at each peak
        peaks = []
        for i in peak_idx:
            mcol = int(np.argmax(curves[i, idx3]))
            ref_global_idx = idx3[mcol]
            ref_name = obj["ref_names"][ref_global_idx]
            if ": " in ref_name:
                cat, tag = ref_name.split(": ", 1)
            else:
                cat, tag = "", ref_name
            peaks.append({
                "frame_idx": int(i),
                "time_sec": float(i / max(1e-6, fps)),
                "score": float(curves[i, ref_global_idx]),
                "ref_label": ref_name,
                "category": cat,
                "tag": tag
            })
        proposals[key] = peaks

    # CSV columns: top-3 series per method (as plotted)
    per_method_series = []
    for key, obj in methods.items():
        names = obj["ref_names"]; curves = obj["curves"]; idx3 = top3_per_method[key]
        for j in idx3:
            label = f"{key}:{names[j]}"
            csv_header.append(label)
            per_method_series.append(curves[:, j])

    per_method_series = np.stack(per_method_series, axis=1) if per_method_series else np.zeros((T, 0))
    for i in range(T):
        row = [i, i / max(1e-6, fps)]
        if per_method_series.shape[1] > 0:
            row.extend(per_method_series[i].tolist())
        csv_rows.append(row)
    write_csv_curves(args.out_csv, csv_header, csv_rows)

    # JSON: proposals (+ caption info if any)
    j = {"fps": fps, "num_frames": T, "proposals": proposals}
    if "caption" in methods:
        j["video_caption"] = methods["caption"].get("caption", "")
        j["caption_tag_ranking"] = methods["caption"].get("caption_tag_ranking", [])
    with open(args.out_json, "w") as f:
        json.dump(j, f, indent=2)

    # --- Render visualization video ---
    vis_w = args.resize_width
    probe = np.array(frame_imgs[0])
    H0, W0 = probe.shape[0], probe.shape[1]
    top_h = int(vis_w * H0 / W0)
    bottom_h = 360

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out_h = top_h + bottom_h
    vw = cv2.VideoWriter(str(args.out_video), fourcc, fps, (vis_w, out_h))

    # pre-resize frames to speed up
    resized_frames_bgr = []
    for im in frame_imgs:
        fr = np.array(im)[:, :, ::-1]  # RGB->BGR
        fr = cv2.resize(fr, (vis_w, top_h), interpolation=cv2.INTER_AREA)
        resized_frames_bgr.append(fr)

    for i in range(T):
        plot_panel = render_plot_panel(methods, i, fps=fps, width=vis_w, height=bottom_h)
        stacked = stack_frame_with_plot(resized_frames_bgr[i], plot_panel)
        vw.write(stacked)

    vw.release()

    # Console summary
    print(f"[OK] Saved curves CSV:    {args.out_csv}")
    print(f"[OK] Saved proposals JSON:{args.out_json}")
    print(f"[OK] Saved visualization: {args.out_video}")

    print("\n=== Highlight Proposals ===")
    for key, pe in proposals.items():
        print(f"\n[{key}] {len(pe)} peaks")
        for p in pe:
            print(f"  - t={p['time_sec']:.2f}s  frame={p['frame_idx']}  score={p['score']:.3f}  -> {p['ref_label']}")


if __name__ == "__main__":
    main()