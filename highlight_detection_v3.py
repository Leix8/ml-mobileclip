#!/usr/bin/env python3
import argparse
import os
import sys
from pathlib import Path
from typing import List, Tuple
from datetime import datetime

# ---- Optional determinism env var before importing torch ----
# (Still set again inside --deterministic block for safety.)
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":16:8")

import random
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFont, ImageOps
import pandas as pd
import matplotlib.pyplot as plt


# ========== Determinism helper ==========
def setup_determinism(seed: int = 1234):
    """
    Make runs deterministic as much as possible (PyTorch + cuDNN/cuBLAS).
    Call this early in main() if args.deterministic is True.
    """
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


# ========== MobileCLIP + tokenizer loader ==========
def _import_mobileclip_and_tokenize():
    """
    Returns (mobileclip_module, tokenizer_fn)
    tokenizer_fn comes from mobileclip.tokenize or open_clip.tokenize
    """
    mobileclip_mod = None
    tokenizer_fn = None
    try:
        import mobileclip  # noqa: F401
        mobileclip_mod = sys.modules["mobileclip"]
        tokenizer_fn = getattr(mobileclip_mod, "tokenize", None)
    except Exception:
        mobileclip_mod = None

    if tokenizer_fn is None:
        try:
            import open_clip  # noqa: F401
            tokenizer_fn = sys.modules["open_clip"].tokenize
        except Exception:
            tokenizer_fn = None

    if mobileclip_mod is None:
        raise ImportError("Could not import 'mobileclip'. Please install/ensure it's on PYTHONPATH.")
    if tokenizer_fn is None:
        raise ImportError("Could not find a tokenizer. Ensure either 'mobileclip.tokenize' or 'open_clip.tokenize' is available.")
    return mobileclip_mod, tokenizer_fn


# ========== IO helpers ==========
def list_image_paths(frames_dir: Path) -> List[Path]:
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    files = [p for p in sorted(frames_dir.iterdir()) if p.suffix.lower() in exts]
    if not files:
        raise FileNotFoundError(f"No images found in {frames_dir}")
    return files


# ========== Encoders ==========
@torch.no_grad()
def encode_images(model, processor, image_paths: List[Path], device: str, batch_size: int) -> torch.Tensor:
    """Return L2-normalized image embeddings of shape [N, D]."""
    embs = []
    for i in range(0, len(image_paths), batch_size):
        batch_paths = image_paths[i : i + batch_size]
        imgs = []
        for p in batch_paths:
            # EXIF-aware, deterministic load
            img = Image.open(p).convert("RGB")
            img = ImageOps.exif_transpose(img)
            imgs.append(processor(img))
        pixel_batch = torch.stack(imgs, dim=0).to(device)
        feats = model.encode_image(pixel_batch)
        feats = F.normalize(feats, dim=-1)
        embs.append(feats)
    return torch.cat(embs, dim=0)


@torch.no_grad()
def encode_text(model, tokenizer_fn, text: str, device: str, context_length: int) -> torch.Tensor:
    """Return L2-normalized text embedding of shape [D]."""
    tokens = tokenizer_fn([text], context_length=context_length).to(device)
    tfeat = model.encode_text(tokens)
    tfeat = F.normalize(tfeat, dim=-1)
    return tfeat.squeeze(0)


def cosine_sim(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Cosine similarity for batched vectors: a [N,D], b [D] -> [N]."""
    return (a * b).sum(dim=-1)


# ========== Visualization builders ==========
def _truncate_middle(s: str, max_chars: int) -> str:
    if len(s) <= max_chars:
        return s
    if max_chars <= 3:
        return s[:max_chars]
    keep = (max_chars - 3) // 2
    return f"{s[:keep]}...{s[-keep:]}"


def build_strip_with_labels(
    image_paths: List[Path],
    frame_height: int = 200,
    frame_margin: int = 8,
    label_height: int = 28,
    font: ImageFont.ImageFont | None = None,
    text_color=(240, 240, 240),
    bg_color=(0, 0, 0),
) -> Tuple[Image.Image, np.ndarray, np.ndarray]:
    """
    Build a single-row strip of frames (kept order) with margins + filename labels.
    Returns:
      strip (PIL.Image)
      centers_idx_space (np.ndarray): frame indices 0..N-1
      centers_px (np.ndarray): pixel x-centers in the final strip image
    """
    N = len(image_paths)
    if N == 0:
        raise ValueError("No images to montage")

    font = font or ImageFont.load_default()

    # Resize to unified height
    resized, widths = [], []
    for p in image_paths:
        im = Image.open(p).convert("RGB")
        im = ImageOps.exif_transpose(im)
        w, h = im.size
        scale = frame_height / max(1, h)
        new_w = max(1, int(round(w * scale)))
        resized_im = im.resize((new_w, frame_height), Image.BILINEAR)
        resized.append(resized_im)
        widths.append(new_w)

    total_w = sum(widths) + frame_margin * (N - 1)
    total_h = frame_height + label_height

    strip = Image.new("RGB", (total_w, total_h), bg_color)
    draw = ImageDraw.Draw(strip)

    x = 0
    centers_idx_space, centers_px = [], []
    for i, (im, p, w) in enumerate(zip(resized, image_paths, widths)):
        strip.paste(im, (x, 0))

        # Label (truncate to fit)
        name = p.name
        max_px = w - 4
        text = name
        if hasattr(font, "getlength"):
            while font.getlength(text) > max_px and len(text) > 4:
                text = _truncate_middle(text, len(text) - 1)
            tw = font.getlength(text)
        else:
            avg_char_px = 6.0
            max_chars = max(4, int(max_px / avg_char_px))
            text = _truncate_middle(text, max_chars)
            tw = len(text) * 6.0

        tx = x + (w - tw) / 2
        ty = frame_height + (label_height - (font.size if hasattr(font, "size") else 10)) / 2 - 1
        draw.text((tx, ty), text, font=font, fill=text_color)

        centers_idx_space.append(i)
        centers_px.append(x + w / 2.0)

        x += w + frame_margin

    return strip, np.array(centers_idx_space, dtype=float), np.array(centers_px, dtype=float)


# ========== Main ==========
def main():
    parser = argparse.ArgumentParser(
        description="MobileCLIP retrieval with TAF, pixel-aligned plots, determinism option, and timestamped outputs."
    )
    parser.add_argument("--model_path", type=str, default="./checkpoints/mobileclip_s0.pt")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--frames_dir", type=str, required=True)
    parser.add_argument("--ref_image", type=str, required=True)
    parser.add_argument("--text", type=str, required=True)
    # Blends
    parser.add_argument("--alpha", type=float, default=0.5, help="Classic blend weight for text (0=image, 1=text).")
    parser.add_argument("--beta", type=float, default=0.4, help="TAF weight for text-aligned visual detail (parallel).")
    parser.add_argument("--gamma", type=float, default=0.1, help="TAF weight for orthogonal visual detail.")
    # Batching / text
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--context_length", type=int, default=77)
    # Outputs
    parser.add_argument("--output_csv", type=str, default=None)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--topk", type=int, default=10)
    # Visualization
    parser.add_argument("--frame_height", type=int, default=200)
    parser.add_argument("--frame_margin", type=int, default=8)
    parser.add_argument("--label_height", type=int, default=28)
    parser.add_argument("--label_font_size", type=int, default=12)
    parser.add_argument("--dpi", type=int, default=150)
    # Determinism
    parser.add_argument("--deterministic", action="store_true", help="Enable deterministic execution.")
    parser.add_argument("--seed", type=int, default=1234, help="Random seed when --deterministic is set.")
    args = parser.parse_args()

    if args.deterministic:
        setup_determinism(args.seed)

    # ---- Timestamped output naming ----
    time_tag = datetime.now().strftime("%Y%m%d_%H%M%S")
    ref_stem = Path(args.ref_image).stem
    prompt_words = args.text.strip().lower().split()
    clue = "_".join(prompt_words[:5]) if prompt_words else "prompt"
    clue = "".join([c if c.isalnum() or c in "_-" else "_" for c in clue])
    prefix = f"{ref_stem}__{clue}__{time_tag}"

    out_root = Path(args.output_dir)
    out_dir = out_root / prefix
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---- Load model & tokenizer ----
    mobileclip, tokenizer_fn = _import_mobileclip_and_tokenize()
    device = args.device

    frames_dir = Path(args.frames_dir)
    image_paths = list_image_paths(frames_dir)
    ref_image_path = Path(args.ref_image)
    if not ref_image_path.exists():
        raise FileNotFoundError(f"Reference image not found: {ref_image_path}")

    model_name = os.path.splitext(os.path.basename(args.model_path))[0]
    model, _, image_processor = mobileclip.create_model_and_transforms(model_name)
    model.to(device).eval()

    # Load checkpoint (flexible keys)
    ckpt = torch.load(args.model_path, map_location="cpu")
    state_dict = ckpt.get("state_dict", ckpt)
    model.load_state_dict(state_dict, strict=False)

    # ---- Encode ----
    print(f"Encoding {len(image_paths)} frames ...")
    frame_embs = encode_images(model, image_processor, image_paths, device, args.batch_size)  # [N, D]

    # Reference image (EXIF-aware for encoding & plotting)
    ref_im_for_encode = Image.open(ref_image_path).convert("RGB")
    ref_im_for_encode = ImageOps.exif_transpose(ref_im_for_encode)
    ref_img_emb = encode_images(model, image_processor, [ref_image_path], device, 1).squeeze(0)  # [D]

    text_emb = encode_text(model, tokenizer_fn, args.text, device, args.context_length)  # [D]

    # ---- Base similarities ----
    sim_ref = cosine_sim(frame_embs, ref_img_emb).cpu().numpy()
    sim_text = cosine_sim(frame_embs, text_emb).cpu().numpy()

    # Classic weighted-sum blend
    alpha = float(args.alpha)
    blend_vec = F.normalize((1.0 - alpha) * ref_img_emb + alpha * text_emb, dim=-1)
    sim_blend = cosine_sim(frame_embs, blend_vec).cpu().numpy()

    # ---- Text-Aligned Fusion (TAF) ----
    beta = float(args.beta)
    gamma = float(args.gamma)

    dot_it = torch.clamp((ref_img_emb * text_emb).sum(), -1.0, 1.0)
    i_parallel = dot_it * text_emb            # along text direction
    i_perp = ref_img_emb - i_parallel         # orthogonal residual

    def safe_norm(x: torch.Tensor) -> torch.Tensor:
        n = x.norm(p=2)
        return x / n if n > 0 else x

    i_parallel_n = safe_norm(i_parallel)
    i_perp_n = safe_norm(i_perp)

    taf_vec = F.normalize((1.0 - beta) * text_emb + beta * i_parallel_n + gamma * i_perp_n, dim=-1)
    sim_taf = cosine_sim(frame_embs, taf_vec).cpu().numpy()

    # ---- CSV (timestamped) ----
    csv_path = Path(args.output_csv) if args.output_csv else out_dir / f"{prefix}_scores.csv"
    df = pd.DataFrame({
        "frame": [str(p) for p in image_paths],
        "sim_text": sim_text,
        "sim_ref_image": sim_ref,
        f"sim_blend_alpha_{alpha:.2f}": sim_blend,
        f"sim_taf_beta_{beta:.2f}_gamma_{gamma:.2f}": sim_taf,
    })
    df.to_csv(csv_path, index=False)
    print(f"Saved CSV to: {csv_path}")

    # ---- Visualization (pixel-accurate alignment) ----
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", args.label_font_size)
    except Exception:
        font = ImageFont.load_default()

    strip_img, _, centers_px = build_strip_with_labels(
        image_paths=image_paths,
        frame_height=args.frame_height,
        frame_margin=args.frame_margin,
        label_height=args.label_height,
        font=font,
        text_color=(240, 240, 240),
        bg_color=(0, 0, 0),
    )
    strip_np = np.asarray(strip_img)
    strip_h, strip_w = strip_np.shape[0], strip_np.shape[1]
    N = len(image_paths)

    base_dpi = float(args.dpi)
    fig_w_in = max(10.0, strip_w / base_dpi)  # whole figure scales with number of frames (no hard cap)
    fig_h_in = 15.0                            # tall enough for 6 rows

    fig = plt.figure(figsize=(fig_w_in, fig_h_in), dpi=base_dpi)
    gs = fig.add_gridspec(6, 1, height_ratios=[2.0, 2.3, 1.4, 1.4, 1.4, 1.4], hspace=0.5)

    # Row 1: reference image + text (EXIF-aware for display)
    gs_row1 = gs[0].subgridspec(1, 2, width_ratios=[1.2, 1.0], wspace=0.2)
    ax_ref = fig.add_subplot(gs_row1[0, 0])
    ref_im_for_plot = Image.open(ref_image_path).convert("RGB")
    ref_im_for_plot = ImageOps.exif_transpose(ref_im_for_plot)
    ax_ref.imshow(ref_im_for_plot)
    ax_ref.set_title("Reference Image", fontsize=12)
    ax_ref.axis("off")

    ax_text = fig.add_subplot(gs_row1[0, 1])
    ax_text.axis("off")
    info = (
        f"Text Prompt:\n{args.text}\n\n"
        f"Model: {model_name}\n"
        f"Frames: {N}\n"
        f"alpha (classic): {alpha:.2f}\n"
        f"β (TAF parallel): {beta:.2f}\n"
        f"γ (TAF orthogonal): {gamma:.2f}\n"
        f"Deterministic: {'Yes' if args.deterministic else 'No'}"
    )
    ax_text.text(0.01, 0.98, info, va="top", ha="left", fontsize=11, wrap=True)

    # Row 2: frame strip (pixel extent so axes match curves)
    ax_strip = fig.add_subplot(gs[1])
    ax_strip.imshow(strip_np, extent=[0, strip_w, 0, strip_h])
    ax_strip.set_xlim(0, strip_w)
    ax_strip.set_ylim(0, strip_h)
    ax_strip.set_title("Frame Sequence (names under each frame)", fontsize=12)
    ax_strip.axis("off")

    # Rows 3–6: similarity curves at pixel centers, sharing x-range with strip
    def plot_curve(ax, title, y):
        ax.plot(centers_px, y)
        ax.set_xlim(0, strip_w)
        ax.set_ylabel("cosine")
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
        ax.tick_params(axis='x', which='both', labelbottom=False)

    ax3 = fig.add_subplot(gs[2])
    plot_curve(ax3, "Text→Frame Similarity", sim_text)

    ax4 = fig.add_subplot(gs[3], sharex=ax3)
    plot_curve(ax4, "Ref Image→Frame Similarity", sim_ref)

    ax5 = fig.add_subplot(gs[4], sharex=ax3)
    plot_curve(ax5, f"Blended→Frame Similarity (α={alpha:.2f})", sim_blend)

    ax6 = fig.add_subplot(gs[5], sharex=ax3)
    ax6.plot(centers_px, sim_taf)
    ax6.set_xlim(0, strip_w)
    ax6.set_xlabel("Frame centers (pixel aligned)")
    ax6.set_ylabel("cosine")
    ax6.set_title(f"TAF→Frame Similarity (β={beta:.2f}, γ={gamma:.2f})")
    ax6.grid(True, alpha=0.3)
    ax6.tick_params(axis='x', which='both', labelbottom=False)

    vis_path = out_dir / f"{prefix}_overview.png"
    plt.savefig(vis_path, bbox_inches="tight", dpi=args.dpi)
    plt.close(fig)
    print(f"Saved visualization to: {vis_path}")

    # ---- Top-K prints (after plotting, for clean logs) ----
    def print_topk(series: pd.Series, title: str, k: int):
        order = series.to_numpy().argsort()[::-1][:k]
        print(f"\nTop {k} frames — {title}")
        for rank, idx in enumerate(order, start=1):
            print(f"{rank:>2}. {Path(image_paths[idx]).name:50s}  score={series.iloc[idx]:.4f}")

    print_topk(df["sim_text"], "Text Retrieval", args.topk)
    print_topk(df["sim_ref_image"], "Reference Image Retrieval", args.topk)
    blend_col = [c for c in df.columns if c.startswith("sim_blend_alpha_")][0]
    print_topk(df[blend_col], f"Blended Retrieval (alpha={alpha:.2f})", args.topk)
    taf_col = [c for c in df.columns if c.startswith("sim_taf_beta_")][0]
    print_topk(df[taf_col], f"TAF Retrieval (β={beta:.2f}, γ={gamma:.2f})", args.topk)


if __name__ == "__main__":
    main()