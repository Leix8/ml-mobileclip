#!/usr/bin/env python3
import argparse
import os
import sys
import json
import random
import re
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple, Union

import torch
import torch.nn.functional as F
from torch.cuda.amp import autocast
from PIL import Image, ImageOps
import open_clip
from mobileclip.modules.common.mobileone import reparameterize_model
from model_name_map import MODEL_NAME_MAP, infer_model_name_from_ckpt


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
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}

def list_image_paths(frames_dir: Path) -> List[Path]:
    return [p for p in sorted(frames_dir.iterdir()) if p.is_file() and p.suffix.lower() in IMG_EXTS]

def normalize_tag_id(tag: str, llm_generated: bool = False) -> str:
    """ Fallback tag_id normalization. """
    if llm_generated:
        tag = tag.lower()
        tag = re.sub(r"[^\w\s]", "", tag)
        tag = re.sub(r"^(a|the)\s+", "", tag)
        tag = re.sub(r"\s+", " ", tag).strip()
    return tag.replace(" ", "_")

def resolve_path(p: Union[str, Path]) -> Path:
    return Path(p).expanduser().resolve()

def gather_images_from_item(
    parent_dir: Path,
    item: Union[str, Dict[str, Any]],
    llm_generated: bool
) -> Tuple[str, str, List[Path], Optional[Path]]:
    """
    Returns (tag_text, tag_id, ref_paths, ref_dir_used)
    Supports:
      - str: old tag text
      - dict with "tag" + optional "ref_dir" (new format)
    """
    if isinstance(item, str):
        tag_text = item
        tag_id = normalize_tag_id(tag_text, llm_generated)
        tag_dir = parent_dir / tag_id
        ref_paths = list_image_paths(tag_dir) if tag_dir.exists() else []
        return tag_text, tag_id, ref_paths, (tag_dir if tag_dir.exists() else None)

    if not isinstance(item, dict):
        raise TypeError(f"Expected str or dict for scene entry, got {type(item)}")

    tag_text = str(item.get("tag", "")).strip()
    if not tag_text:
        raise ValueError("Tag object missing 'tag' field.")

    tag_id = str(item.get("id") or normalize_tag_id(tag_text, llm_generated))

    ref_dir_used = None
    ref_paths: List[Path] = []
    if "ref_dir" in item and item["ref_dir"]:
        rd = resolve_path(item["ref_dir"])
        if rd.exists() and rd.is_dir():
            ref_paths = list_image_paths(rd)
            ref_dir_used = rd
        else:
            print(f"[WARN] ref_dir not found or not a directory: {rd}", file=sys.stderr)
    else:
        # fallback to old behavior
        tag_dir = parent_dir / tag_id
        if tag_dir.exists() and tag_dir.is_dir():
            ref_paths = list_image_paths(tag_dir)
            ref_dir_used = tag_dir

    return tag_text, tag_id, ref_paths, ref_dir_used

# ================= Encoders =================
@torch.no_grad()
def encode_images(model, image_processor, image_paths: List[Path],
                  device: str, batch_size: int, use_amp: bool = True) -> torch.Tensor:
    embs = []
    for i in range(0, len(image_paths), batch_size):
        batch_paths = image_paths[i: i + batch_size]
        imgs = []
        for p in batch_paths:
            img = Image.open(p).convert("RGB")
            img = ImageOps.exif_transpose(img)
            imgs.append(image_processor(img))
        pixel_batch = torch.stack(imgs, dim=0).to(device)
        with autocast(enabled=use_amp):
            feats = model.encode_image(pixel_batch)
        feats = F.normalize(feats, dim=-1)
        embs.append(feats)
    return torch.cat(embs, dim=0) if embs else torch.empty(0, device=device)

@torch.no_grad()
def encode_text(model, tokenizer, text: str, device: str,
                context_length: int, use_amp: bool = True) -> torch.Tensor:
    tokens = tokenizer([text], context_length=context_length).to(device)
    with autocast(enabled=use_amp):
        tfeat = model.encode_text(tokens)
    return F.normalize(tfeat, dim=-1).squeeze(0)

# ================= Fusion helpers =================
def safe_norm(x: torch.Tensor) -> torch.Tensor:
    n = x.norm(p=2)
    return x / n if n > 0 else x

def taf_fuse_multi_refs(text_emb: torch.Tensor, ref_embs: torch.Tensor,
                        beta: float, gamma: float) -> torch.Tensor:
    if ref_embs.size(0) == 0:
        return text_emb
    fused = []
    for k in range(ref_embs.size(0)):
        ref = ref_embs[k]
        dot_it = torch.clamp((ref * text_emb).sum(), -1.0, 1.0)
        i_parallel = dot_it * text_emb
        i_perp = ref - i_parallel
        fused.append((1.0 - beta) * text_emb +
                     beta * safe_norm(i_parallel) +
                     gamma * safe_norm(i_perp))
    return F.normalize(torch.stack(fused, dim=0).mean(dim=0), dim=-1)

def cos_push_multi_refs(text_emb: torch.Tensor, ref_embs: torch.Tensor,
                        push_lambda: float) -> torch.Tensor:
    if ref_embs.size(0) == 0:
        return text_emb
    pushed = [F.normalize(ref_embs[k] + push_lambda * text_emb, dim=-1)
              for k in range(ref_embs.size(0))]
    return F.normalize(torch.stack(pushed, dim=0).mean(dim=0), dim=-1)

# ================= Save helper =================
def tensor_to_jsonable(t: torch.Tensor) -> Dict[str, Any]:
    arr = t.detach().cpu().numpy()
    return {"shape": list(arr.shape), "dtype": str(arr.dtype), "value": arr.tolist()}

# ================= Main =================
def main():
    parser = argparse.ArgumentParser(
        description="Encode scene tags with per-tag reference directories using MobileCLIP."
    )
    parser.add_argument("--scene_json", type=str, required=True,
                        help="Path to scene/tag json file.")
    parser.add_argument("--ref_dir", type=str, required=True,
                        help="Parent directory for fallback lookup.")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Where to save the output json.")
    parser.add_argument("--output_suffix", type=str, required=False)
    parser.add_argument("--llm_generated", action="store_true",
                        help="Enable more aggressive tag normalization for fallback mode.")
    parser.add_argument("--model_path", type=str, default="./checkpoints/mobileclip2_b.pt")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--context_length", type=int, default=77)
    parser.add_argument("--beta", type=float, default=0.4)
    parser.add_argument("--gamma", type=float, default=0.1)
    parser.add_argument("--push_lambda", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--no_amp", action="store_true")

    args = parser.parse_args()

    setup_determinism(args.seed)

    # Load MobileCLIP
    # mobileclip = _import_mobileclip()
    # model_name = os.path.splitext(os.path.basename(args.model_path))[0]
    model_name = infer_model_name_from_ckpt(args.model_path)
    model, _, image_processor = open_clip.create_model_and_transforms(
        model_name, pretrained=args.model_path)
    model = model.to(args.device).eval()
    tokenizer = open_clip.get_tokenizer(model_name)
    model.eval()
    model = reparameterize_model(model)

    use_amp = not args.no_amp

    # Load JSON
    with open(args.scene_json, "r") as f:
        scenes_data = json.load(f)

    parent_dir = Path(args.ref_dir)
    results: Dict[str, List[Dict[str, Any]]] = {}

    for key, tag_list in scenes_data.items():
        results[key] = []
        if not isinstance(tag_list, list):
            print(f"[WARN] Expected list under key '{key}'", file=sys.stderr)
            continue

        for item in tag_list:
            tag_text, tag_id, ref_paths, ref_dir_used = gather_images_from_item(
                parent_dir, item, args.llm_generated
            )

            ref_embs = encode_images(model, image_processor, ref_paths,
                                     args.device, args.batch_size, use_amp=use_amp) if ref_paths else torch.empty(0, device=args.device)
            text_emb = encode_text(model, tokenizer, tag_text, args.device,
                                   args.context_length, use_amp=use_amp)

            taf_vec = taf_fuse_multi_refs(text_emb, ref_embs, args.beta, args.gamma)
            push_vec = cos_push_multi_refs(text_emb, ref_embs, args.push_lambda)

            entry = {
                "id": tag_id,
                "tag": tag_text,
                "reference_dir": str(ref_dir_used) if ref_dir_used else None,
                "num_ref": int(ref_embs.size(0)),
                "model": model_name,
                "embedding": {
                    "tag_embedding": tensor_to_jsonable(text_emb),
                    "multi_ref_TAF_embedding": {
                        "params": {"beta": args.beta, "gamma": args.gamma},
                        "data": tensor_to_jsonable(taf_vec)
                    },
                    "multi_ref_push_embedding": {
                        "params": {"push_lambda": args.push_lambda},
                        "data": tensor_to_jsonable(push_vec)
                    }
                }
            }
            results[key].append(entry)

    # Save
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    base = Path(args.scene_json).stem
    flag = "_llmgen" if args.llm_generated else ""
    out_path = out_dir / f"{base}{flag}_{model_name}_embeddings.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved embeddings to {out_path}")

if __name__ == "__main__":
    main()