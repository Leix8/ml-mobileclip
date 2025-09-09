#!/usr/bin/env python3
import argparse
import os
import sys
from pathlib import Path
from typing import List

import torch
import torch.nn.functional as F
from PIL import Image
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# --- Robust imports for MobileCLIP + tokenization ---
def _import_mobileclip_and_tokenize():
    mobileclip = None
    tokenizer_fn = None
    try:
        import mobileclip  # noqa: F401
        mobileclip = sys.modules["mobileclip"]
        tokenizer_fn = getattr(mobileclip, "tokenize", None)
    except Exception:
        mobileclip = None

    if tokenizer_fn is None:
        try:
            import open_clip  # noqa: F401
            tokenizer_fn = sys.modules["open_clip"].tokenize
        except Exception:
            tokenizer_fn = None

    if mobileclip is None:
        raise ImportError(
            "Could not import 'mobileclip'. Please install/ensure it's on PYTHONPATH."
        )

    if tokenizer_fn is None:
        raise ImportError(
            "Could not find a tokenizer. Ensure either 'mobileclip.tokenize' or 'open_clip.tokenize' is available."
        )

    return mobileclip, tokenizer_fn


def list_image_paths(frames_dir: Path) -> List[Path]:
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    files = [p for p in sorted(frames_dir.iterdir()) if p.suffix.lower() in exts]
    if not files:
        raise FileNotFoundError(f"No images found in {frames_dir}")
    return files


@torch.no_grad()
def encode_images(model, processor, image_paths: List[Path], device: str, batch_size: int) -> torch.Tensor:
    """Return L2-normalized image embeddings of shape [N, D]."""
    embs = []
    for i in range(0, len(image_paths), batch_size):
        batch_paths = image_paths[i : i + batch_size]
        imgs = []
        for p in batch_paths:
            img = Image.open(p).convert("RGB")
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


def build_horizontal_montage(image_paths: List[Path], target_height: int = 120, max_width: int = 8000) -> Image.Image:
    """
    Create a single-row montage of all frames (kept in order).
    Each frame is resized to target_height while preserving aspect ratio.
    If total width exceeds max_width, we downscale the entire strip at the end.
    """
    # Load & resize each image to target_height
    resized = []
    widths = []
    total_w = 0
    for p in image_paths:
        im = Image.open(p).convert("RGB")
        w, h = im.size
        if h == 0:
            continue
        scale = target_height / h
        new_w = max(1, int(round(w * scale)))
        im_resized = im.resize((new_w, target_height), Image.BILINEAR)
        resized.append(im_resized)
        widths.append(new_w)
        total_w += new_w

    if total_w == 0:
        raise ValueError("Could not build montage: zero total width")

    strip = Image.new("RGB", (total_w, target_height), (0, 0, 0))
    x = 0
    for im in resized:
        strip.paste(im, (x, 0))
        x += im.size[0]

    # Downscale if too wide for typical renderers
    if strip.size[0] > max_width:
        scale = max_width / strip.size[0]
        new_w = int(round(strip.size[0] * scale))
        new_h = int(round(strip.size[1] * scale))
        strip = strip.resize((new_w, new_h), Image.BILINEAR)

    return strip


def main():
    parser = argparse.ArgumentParser(
        description="MobileCLIP retrieval over frames with visualization (ref image, text, blended)."
    )
    parser.add_argument("--model_path", type=str, default="./checkpoints/mobileclip_s0.pt",
                        help="Path to MobileCLIP checkpoint (.pt/.pth).")
    parser.add_argument("--device", type=str, default="cuda:0", help="Device, e.g., cuda:0 or cpu.")
    parser.add_argument("--frames_dir", type=str, required=True,
                        help="Directory containing extracted video frames (images).")
    parser.add_argument("--ref_image", type=str, required=True, help="Reference image file path.")
    parser.add_argument("--text", type=str, required=True, help="Text prompt.")
    parser.add_argument("--alpha", type=float, default=0.5,
                        help="Blend weight for text (0=image-only, 1=text-only).")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size for image encoding.")
    parser.add_argument("--context_length", type=int, default=77, help="Text tokenizer context length.")
    parser.add_argument("--output_csv", type=str, default=None,
                        help="CSV path; if omitted, saved under --output_dir.")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Directory to save outputs (CSV + visualization).")
    parser.add_argument("--topk", type=int, default=10, help="Print top-K frames for each method.")
    parser.add_argument("--montage_height", type=int, default=120,
                        help="Pixel height for the frame-strip montage row.")
    parser.add_argument("--fig_width_per_100_frames", type=float, default=10.0,
                        help="Figure width grows with frame count; width ~= (N/100)*this.")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # --- Build output naming prefix ---
    ref_stem = Path(args.ref_image).stem
    # use first 5 words from text prompt as clue
    prompt_words = args.text.strip().lower().split()
    clue = "_".join(prompt_words[:5]) if prompt_words else "prompt"
    clue = "".join([c if c.isalnum() or c in "_-" else "_" for c in clue])
    prefix = f"{ref_stem}__{clue}"

    # make dedicated subfolder under --output_dir
    out_subdir = out_dir / prefix
    out_subdir.mkdir(parents=True, exist_ok=True)
    
    # Imports
    mobileclip, tokenizer_fn = _import_mobileclip_and_tokenize()

    device = args.device
    frames_dir = Path(args.frames_dir)
    ref_image_path = Path(args.ref_image)
    if not ref_image_path.exists():
        raise FileNotFoundError(f"Reference image not found: {ref_image_path}")

    image_paths = list_image_paths(frames_dir)

    # Create model & preprocess
    model_name = os.path.splitext(os.path.basename(args.model_path))[0]
    model, _, image_processor = mobileclip.create_model_and_transforms(model_name)
    model.to(device)
    model.eval()

    # Load checkpoint
    ckpt = torch.load(args.model_path, map_location="cpu")
    state_dict = ckpt.get("state_dict", ckpt)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        print(f"[Warn] Missing keys in state_dict: {len(missing)}", file=sys.stderr)
    if unexpected:
        print(f"[Warn] Unexpected keys in state_dict: {len(unexpected)}", file=sys.stderr)

    # Encode frames
    print(f"Encoding {len(image_paths)} frames ...")
    frame_embs = encode_images(model, image_processor, image_paths, device, args.batch_size)

    # Encode ref image + text
    ref_img_emb = encode_images(model, image_processor, [ref_image_path], device, 1).squeeze(0)
    text_emb = encode_text(model, tokenizer_fn, args.text, device, args.context_length)

    # Similarities
    sim_ref = cosine_sim(frame_embs, ref_img_emb).cpu().numpy()     # [N]
    sim_text = cosine_sim(frame_embs, text_emb).cpu().numpy()       # [N]
    alpha = float(args.alpha)
    blend_vec = F.normalize((1.0 - alpha) * ref_img_emb + alpha * text_emb, dim=-1)
    sim_blend = cosine_sim(frame_embs, blend_vec).cpu().numpy()     # [N]

    # Save CSV
    df = pd.DataFrame({
        "frame": [str(p) for p in image_paths],
        "sim_text": sim_text,
        "sim_ref_image": sim_ref,
        f"sim_blend_alpha_{alpha:.2f}": sim_blend,
    })
    csv_path = Path(args.output_csv) if args.output_csv else out_subdir / "retrieval_scores.csv"
    df.to_csv(csv_path, index=False)
    print(f"Saved per-frame similarity scores to: {csv_path}")

    # Print top-K
    def print_topk(series: pd.Series, title: str, k: int):
        order = series.to_numpy().argsort()[::-1][:k]
        print(f"\nTop {k} frames — {title}")
        for rank, idx in enumerate(order, start=1):
            print(f"{rank:>2}. {Path(image_paths[idx]).name:50s}  score={series.iloc[idx]:.4f}")

    print_topk(df["sim_text"], "Text Retrieval", args.topk)
    print_topk(df["sim_ref_image"], "Reference Image Retrieval", args.topk)
    blend_col = [c for c in df.columns if c.startswith("sim_blend_alpha_")][0]
    print_topk(df[blend_col], f"Blended Retrieval (alpha={alpha:.2f})", args.topk)

    # ---------- Visualization ----------
    # Build the montage strip for row 2 (all frames in sequence)
    montage = build_horizontal_montage(image_paths, target_height=args.montage_height)
    montage_np = np.asarray(montage)

    # Compute figure size adaptively:
    N = len(image_paths)
    fig_w = max(10.0, (N / 100.0) * args.fig_width_per_100_frames)  # scales with frame count
    # Use a tall figure to host 5 rows (images + 3 curves)
    fig_h = 12.0
    fig = plt.figure(figsize=(fig_w, fig_h))
    gs = fig.add_gridspec(5, 1, height_ratios=[2.0, 2.0, 1.5, 1.5, 1.5], hspace=0.45)

    # Row 1: reference image (left) + text prompt (right) using two columns within row 1
    gs_row1 = gs[0].subgridspec(1, 2, width_ratios=[1.2, 1.0], wspace=0.2)
    ax_ref = fig.add_subplot(gs_row1[0, 0])
    ref_im = Image.open(ref_image_path).convert("RGB")
    ax_ref.imshow(ref_im)
    ax_ref.set_title("Reference Image", fontsize=12)
    ax_ref.axis("off")

    ax_text = fig.add_subplot(gs_row1[0, 1])
    ax_text.axis("off")
    txt = (f"Text Prompt:\n{args.text}\n\n"
           f"Model: {model_name}\n"
           f"Frames: {N}\n"
           f"alpha (blend weight for text): {alpha:.2f}")
    ax_text.text(0.01, 0.98, txt, va="top", ha="left", fontsize=11, wrap=True)

    # Row 2: montage strip of frames
    ax_strip = fig.add_subplot(gs[1])
    ax_strip.imshow(montage_np)
    ax_strip.set_title("Frame Sequence (in order)", fontsize=12)
    ax_strip.axis("off")

    # Rows 3-5: aligned similarity curves
    x = np.arange(N)

    ax3 = fig.add_subplot(gs[2])
    ax3.plot(x, sim_text)
    ax3.set_xlim([0, N - 1 if N > 1 else 1])
    ax3.set_ylabel("cosine")
    ax3.set_title("Text→Frame Similarity")

    ax4 = fig.add_subplot(gs[3], sharex=ax3)
    ax4.plot(x, sim_ref)
    ax4.set_xlim([0, N - 1 if N > 1 else 1])
    ax4.set_ylabel("cosine")
    ax4.set_title("Ref Image→Frame Similarity")

    ax5 = fig.add_subplot(gs[4], sharex=ax3)
    ax5.plot(x, sim_blend)
    ax5.set_xlim([0, N - 1 if N > 1 else 1])
    ax5.set_xlabel("Frame Index (sorted by filename)")
    ax5.set_ylabel("cosine")
    ax5.set_title(f"Blended→Frame Similarity (alpha={alpha:.2f})")

    # Light grid on all three curve plots
    for ax in (ax3, ax4, ax5):
        ax.grid(True, alpha=0.3)

    vis_path = out_subdir / "retrieval_overview.png"
    plt.savefig(vis_path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"Saved visualization to: {vis_path}")


if __name__ == "__main__":
    main()