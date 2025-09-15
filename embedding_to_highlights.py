#!/usr/bin/env python3
import argparse
import os
import sys
import json
import random
import numpy as np
from pathlib import Path
from typing import List, Dict, Any

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


def sample_video_frames(video_path: str, target_fps: int) -> List[Image.Image]:
    frames = []
    if not _HAS_CV2:
        raise RuntimeError("OpenCV required for --video mode.")
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")
    native_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    stride = max(1, int(round(native_fps / max(1, target_fps))))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    idx = 0
    with tqdm(total=total_frames if total_frames > 0 else None,
              desc="Reading video frames", unit="f") as pbar:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if idx % stride == 0:
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(Image.fromarray(rgb))
            idx += 1
            if total_frames > 0:
                pbar.update(1)
    cap.release()
    return frames


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
            ax.scatter(idxs, vals, s=40, marker="v", color="black", zorder=3)
            for xi, yi in zip(idxs, vals):
                ax.text(xi, yi + 0.02 * span, "✂", fontsize=10,
                        ha="center", va="center", color="red", zorder=4)
                # tag annotation
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


# ================= JSON helpers =================
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

# ================= Utils =================
def get_name_from_path(p: str) -> str:
    path = Path(p)
    if path.is_dir():
        # directory → return last folder name
        return path.name
    else:
        # file → return filename without extension
        return path.stem


# ================= Main =================
def main():
    parser = argparse.ArgumentParser(
        description="Video highlight retrieval visualization, combine across tags if --all_in_one."
    )
    parser.add_argument("--embeddings", type=str, required=True)
    parser.add_argument("--frames", type=str, help="Directory of frames")
    parser.add_argument("--video", type=str, help="Video file path")
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--fps", type=int, default=5)
    parser.add_argument("--topk", type=int, default=10)
    parser.add_argument("--bg_samples", type=int, default=128)
    parser.add_argument("--model_path", type=str, default="./checkpoints/mobileclip_s0.pt")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--no_amp", action="store_true")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--all_in_one", action="store_true",
                        help="If set, combine scores across tags (per method, keep highest score per frame and label tag).")
    args = parser.parse_args()

    setup_determinism(args.seed)

    with open(args.embeddings, "r") as f:
        emb_json = json.load(f)
    entries = extract_tag_entries(emb_json)
    if len(entries) == 0:
        raise ValueError("No tag entries found.")

    # Load MobileCLIP
    mobileclip = _import_mobileclip()
    model_name = os.path.splitext(os.path.basename(args.model_path))[0]
    model, _, image_processor = mobileclip.create_model_and_transforms(model_name, pretrained=args.model_path)
    model = model.to(args.device).eval()
    use_amp = not args.no_amp

    # Load frames
    pil_frames, frame_names = [], []
    output_prefix = None
    if args.video:
        output_prefix = get_name_from_path(args.video)
        pil_frames = sample_video_frames(args.video, args.fps)
        frame_names = [f"frame_{i:05d}.jpg" for i in range(len(pil_frames))]
    elif args.frames:
        output_prefix = get_name_from_path(args.frames)
        all_paths = list_image_paths(Path(args.frames))
        base_fps = 30.0
        stride = max(1, int(round(base_fps / max(1, args.fps))))
        sel = [i for i in range(len(all_paths)) if i % stride == 0]
        for i in tqdm(sel, desc="Reading frame images", unit="img"):
            p = all_paths[i]
            im = Image.open(p).convert("RGB")
            im = ImageOps.exif_transpose(im)
            pil_frames.append(im)
            frame_names.append(p.name)
    else:
        raise ValueError("Either --frames or --video must be given")

    frame_embs = encode_images(model, image_processor, pil_frames, args.device, args.batch_size, use_amp=use_amp)
    N = frame_embs.size(0)

    # Background centroid
    bg_count = min(args.bg_samples, N)
    bg_idxs = np.sort(np.random.choice(np.arange(N), size=bg_count, replace=False))
    bg_centroid = F.normalize(frame_embs[torch.from_numpy(bg_idxs)].mean(dim=0), dim=-1)

    # Collect all scores
    all_tag_scores = {m: [] for m in ["Text", "TAF", "Cos-Push", "Debiased TAF", "Debiased Cos-Push"]}
    all_tag_labels = []

    for entry in entries:
        tag_text = entry.get("tag", entry.get("id", "tag"))
        text_vec = get_vector_from_entry(entry, ["embedding", "tag_embedding"])
        taf_vec = np.array(entry["embedding"]["multi_ref_TAF_embedding"]["data"]["value"], dtype=np.float32)
        push_vec = np.array(entry["embedding"]["multi_ref_push_embedding"]["data"]["value"], dtype=np.float32)

        text_t = F.normalize(torch.tensor(text_vec, device=args.device), dim=-1)
        taf_t = F.normalize(torch.tensor(taf_vec, device=args.device), dim=-1)
        push_t = F.normalize(torch.tensor(push_vec, device=args.device), dim=-1)
        debiased_taf = F.normalize(taf_t - bg_centroid, dim=-1)
        debiased_push = F.normalize(push_t - bg_centroid, dim=-1)

        sim_text = cosine_sim_batch(frame_embs, text_t).cpu().numpy()
        sim_taf = cosine_sim_batch(frame_embs, taf_t).cpu().numpy()
        sim_push = cosine_sim_batch(frame_embs, push_t).cpu().numpy()
        sim_taf_deb = cosine_sim_batch(frame_embs, debiased_taf).cpu().numpy()
        sim_push_deb = cosine_sim_batch(frame_embs, debiased_push).cpu().numpy()

        all_tag_scores["Text"].append((sim_text, tag_text))
        all_tag_scores["TAF"].append((sim_taf, tag_text))
        all_tag_scores["Cos-Push"].append((sim_push, tag_text))
        all_tag_scores["Debiased TAF"].append((sim_taf_deb, tag_text))
        all_tag_scores["Debiased Cos-Push"].append((sim_push_deb, tag_text))

    # Combine across tags
    combined_scores = {}
    combined_labels = {}
    topk_map = {}
    for m, sims_tags in all_tag_scores.items():
        stacked = np.stack([s for s, _ in sims_tags], axis=1)
        tag_names = [t for _, t in sims_tags]
        best_idx = stacked.argmax(axis=1)
        best_scores = stacked.max(axis=1)
        best_labels = [tag_names[j] for j in best_idx]
        combined_scores[m] = best_scores
        combined_labels[m] = best_labels

        # compute top-k
        k = min(args.topk, N)
        idxs = np.argpartition(best_scores, -k)[-k:]
        topk_map[m] = idxs[np.argsort(best_scores[idxs])][::-1]

    # Save CSV
    import pandas as pd
    csv_path = Path(args.output_dir) / f"{output_prefix}_scores.csv"
    df = pd.DataFrame({"frame": frame_names})
    for m, arr in combined_scores.items():
        df[m] = arr
        df[m + "_BestTag"] = combined_labels[m]
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(csv_path, index=False)
    print(f"Saved CSV to {csv_path}")

    # Visualization
    sample_im = pil_frames[0]
    target_h = 360
    scale = target_h / sample_im.height
    target_w = int(sample_im.width * scale)
    def resize_img(im): return im.resize((target_w, target_h), Image.BILINEAR)

    sample_upper = make_info_panel(resize_img(sample_im), "Combined across tags")
    width_upper, height_upper = sample_upper.size
    sample_plot = draw_score_panel(combined_scores, 0, topk_map, width_px=width_upper, height_per_row=120, tag_labels=combined_labels)
    width_plot, height_plot = sample_plot.size
    canvas_w, canvas_h = width_upper, height_upper + height_plot

    out_video_path = Path(args.output_dir) / f"{output_prefix}_highlights.mp4"
    if _HAS_CV2:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        vw = cv2.VideoWriter(str(out_video_path), fourcc, args.fps, (canvas_w, canvas_h))
    elif _HAS_IMAGEIO:
        vw = imageio.get_writer(str(out_video_path), fps=args.fps, codec="libx264", quality=8)
    else:
        raise RuntimeError("No video writer available")

    for idx, (im, fname) in enumerate(tqdm(zip(pil_frames, frame_names), total=N, desc="Composing video", unit="f")):
        upper_panel = make_info_panel(resize_img(im), f"Frame {idx+1}/{N} {fname}")
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