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
def sample_video_frames_with_map(video_path: str, target_fps: int):
    if not _HAS_CV2:
        raise RuntimeError("OpenCV required for --video mode.")
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")
    native_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    stride = max(1, int(round(native_fps / max(1, target_fps))))

    pil_frames, frame_names, original_indices = [], [], []
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
                frame_names.append(f"frame_{idx:08d}.jpg")
                original_indices.append(idx)
            idx += 1
            if total_frames > 0:
                pbar.update(1)
    cap.release()
    return pil_frames, frame_names, original_indices, float(native_fps), int(stride), int(total_frames)


def load_frames_dir_with_map(frames_dir: str, target_fps: int, assumed_base_fps: float = 30.0):
    all_paths = list_image_paths(Path(frames_dir))
    if len(all_paths) == 0:
        raise RuntimeError(f"No images found in: {frames_dir}")

    stride = max(1, int(round(assumed_base_fps / max(1, target_fps))))
    selected_ids = [i for i in range(len(all_paths)) if i % stride == 0]

    pil_frames, frame_names, original_indices = [], [], []
    for i in tqdm(selected_ids, desc="Reading frame images", unit="img"):
        p = all_paths[i]
        im = Image.open(p).convert("RGB")
        im = ImageOps.exif_transpose(im)
        pil_frames.append(im)
        frame_names.append(p.name)
        original_indices.append(i)

    return pil_frames, frame_names, original_indices, None, int(stride), len(all_paths)


# ================= Encoders =================
@torch.no_grad()
def encode_images(model, image_processor, pil_images, device: str, batch_size: int, use_amp: bool = True):
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
    all_tag_scores: Dict[str, List[Tuple[np.ndarray, str]]],
    current_idx: int,
    width_px: int = 1400,
    height_per_row: int = 150,
    dpi: int = 100
) -> Image.Image:
    methods = list(all_tag_scores.keys())
    methods = ['Text']
    R = len(methods)
    fig = plt.figure(figsize=(width_px / dpi, (R * height_per_row) / dpi), dpi=dpi)
    gs = fig.add_gridspec(R, 1, hspace=0.6)

    for r, m in enumerate(methods):
        ax = fig.add_subplot(gs[r])
        sims_tags = all_tag_scores[m]
        for scores, tag in sims_tags:
            x = np.arange(len(scores))
            ax.plot(x, scores, linewidth=1, label=tag)
        ax.axvline(current_idx, color="red", linestyle="--", linewidth=2)
        ax.set_ylabel("cos")
        ax.set_title(m)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=7, loc="upper right")
        if r == R - 1:
            ax.set_xlabel("frame idx")
        else:
            ax.tick_params(axis="x", labelbottom=False)

    fig.canvas.draw()
    buf = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
    plot_img = buf.reshape(fig.canvas.get_width_height()[::-1] + (4,))
    plt.close(fig)
    return Image.fromarray(plot_img[:, :, :3])


def load_font(size: int = 18):
    try:
        return ImageFont.truetype("DejaVuSans.ttf", size)
    except Exception:
        return ImageFont.load_default()


def make_info_panel(frame_img: Image.Image, meta_text: str, right_panel_width: int = 420, pad: int = 12,
                    bg=(15, 15, 15), fg=(240, 240, 240)) -> Image.Image:
    w_frame, h_frame = frame_img.size
    panel_w = w_frame + right_panel_width
    panel_h = h_frame
    canvas = Image.new("RGB", (panel_w, panel_h), bg)
    canvas.paste(frame_img, (0, 0))

    draw = ImageDraw.Draw(canvas)
    font = load_font(18)
    draw.multiline_text((w_frame + pad, pad), meta_text, font=font, fill=fg, spacing=4)
    return canvas


# ================= Embeddings JSON helpers =================
def extract_tag_entries(emb_json: Dict[str, Any]):
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


# ================= Main =================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--embeddings", type=str, required=True)
    parser.add_argument("--frames", type=str)
    parser.add_argument("--video", type=str)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--fps", type=int, default=5)
    parser.add_argument("--bg_samples", type=int, default=128)
    parser.add_argument("--model_path", type=str, default="./checkpoints/mobileclip_s0.pt")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--no_amp", action="store_true")
    parser.add_argument("--seed", type=int, default=1234)
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

    # Load frames
    if args.video:
        (pil_frames, frame_names, original_indices,
         native_fps, stride, total_original_frames) = sample_video_frames_with_map(args.video, args.fps)
        output_prefix = get_name_from_path(args.video)
    elif args.frames:
        (pil_frames, frame_names, original_indices,
         native_fps, stride, total_original_frames) = load_frames_dir_with_map(args.frames, args.fps)
        output_prefix = get_name_from_path(args.frames)
    else:
        raise ValueError("Either --frames or --video must be given")

    # Encode frames
    frame_embs = encode_images(model, image_processor, pil_frames, args.device, args.batch_size, use_amp=use_amp)

    # Background centroid
    N = frame_embs.size(0)
    bg_count = min(args.bg_samples, N)
    bg_idxs = np.sort(np.random.choice(np.arange(N), size=bg_count, replace=False))
    bg_centroid = F.normalize(frame_embs[torch.from_numpy(bg_idxs)].mean(dim=0), dim=-1)

    def _get_vec(entry, key_path):
        vec = get_vector_from_entry(entry, key_path)
        return F.normalize(torch.tensor(vec, device=args.device), dim=-1)

    # Collect all scores per tag
    all_tag_scores = {m: [] for m in ["Text", "TAF", "Cos-Push", "Debiased TAF", "Debiased Cos-Push"]}
    for entry in entries:
        tag_text = entry.get("tag", entry.get("id", "tag"))
        text_t = _get_vec(entry, ["embedding", "tag_embedding"])
        taf_t = _get_vec(entry, ["embedding", "multi_ref_TAF_embedding", "data"])
        push_t = _get_vec(entry, ["embedding", "multi_ref_push_embedding", "data"])

        debiased_taf = F.normalize(taf_t - bg_centroid, dim=-1)
        debiased_push = F.normalize(push_t - bg_centroid, dim=-1)

        all_tag_scores["Text"].append((cosine_sim_batch(frame_embs, text_t).cpu().numpy(), tag_text))
        all_tag_scores["TAF"].append((cosine_sim_batch(frame_embs, taf_t).cpu().numpy(), tag_text))
        all_tag_scores["Cos-Push"].append((cosine_sim_batch(frame_embs, push_t).cpu().numpy(), tag_text))
        all_tag_scores["Debiased TAF"].append((cosine_sim_batch(frame_embs, debiased_taf).cpu().numpy(), tag_text))
        all_tag_scores["Debiased Cos-Push"].append((cosine_sim_batch(frame_embs, debiased_push).cpu().numpy(), tag_text))

    # -------- Visualization video --------
    sample_im = pil_frames[0]
    target_h = 360
    scale = target_h / sample_im.height
    target_w = int(sample_im.width * scale)
    def resize_img(im): return im.resize((target_w, target_h), Image.BILINEAR)

    sample_upper = make_info_panel(resize_img(sample_im), "Per-tag scores (all curves plotted)")
    width_upper, height_upper = sample_upper.size
    sample_plot = draw_score_panel(all_tag_scores, 0, width_px=width_upper, height_per_row=150)
    width_plot, height_plot = sample_plot.size
    canvas_w, canvas_h = width_upper, height_upper + height_plot

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_video_path = out_dir / f"{output_prefix}_alltags.mp4"

    if _HAS_CV2:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        vw = cv2.VideoWriter(str(out_video_path), fourcc, args.fps, (canvas_w, canvas_h))
    elif _HAS_IMAGEIO:
        vw = imageio.get_writer(str(out_video_path), fps=args.fps, codec="libx264", quality=8)
    else:
        raise RuntimeError("No video writer available")

    for idx, (im, fname) in enumerate(tqdm(zip(pil_frames, frame_names), total=N, desc="Composing video", unit="f")):
        header_txt = f"Frame {idx+1}/{N} | orig_idx={original_indices[idx]} | {fname}"
        upper_panel = make_info_panel(resize_img(im), header_txt)
        plot_panel = draw_score_panel(all_tag_scores, idx, width_px=canvas_w, height_per_row=150)
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
    print(f"Saved video to {out_video_path}")


if __name__ == "__main__":
    main()