#!/usr/bin/env python3
import os
import json
import shutil
from pathlib import Path
from PIL import Image
from concurrent.futures import ThreadPoolExecutor, as_completed

import torch
import numpy as np
from icrawler.builtin import GoogleImageCrawler

import open_clip
from mobileclip.modules.common.mobileone import reparameterize_model
from model_name_map import MODEL_NAME_MAP, infer_model_name_from_ckpt

# ========== Monkey-patch icrawler to avoid NoneType bug ==========
from icrawler import parser
_old_parse = parser.Parser.parse
def _safe_parse(self, response, **kwargs):
    result = _old_parse(self, response, **kwargs)
    return result or []  # return [] instead of None
parser.Parser.parse = _safe_parse


# ================== Image Crawler ==================
def download_images_from_prompt(prompt: str, tmp_dir: str, max_images: int = 100):
    """Download raw images into tmp_dir"""
    os.makedirs(tmp_dir, exist_ok=True)
    crawler = GoogleImageCrawler(storage={"root_dir": tmp_dir})
    crawler.crawl(
        keyword=prompt,
        max_num=max_images,
        min_size=(200, 200),
        file_idx_offset=0
    )


# ================== CLIP-based Filtering ==================
def encode_image(model, processor, path: Path, device="cuda"):
    """Encode single image with MobileCLIP"""
    try:
        img = Image.open(path).convert("RGB")
    except Exception:
        return None
    img_tensor = processor(img).unsqueeze(0).to(device)
    with torch.no_grad():
        feat = model.encode_image(img_tensor).cpu().numpy()[0]
    feat = feat / np.linalg.norm(feat)
    return feat


def filter_images(model, processor, tokenizer, tag: str, img_paths, keep_num=30,
                  device="cuda", num_threads=8, threshold=0.1):
    """Filter images based on CLIP similarity with text prompt"""
    # text embedding
    tok = tokenizer([tag]).to(device)
    with torch.no_grad():
        text_feat = model.encode_text(tok).cpu().numpy()[0]
    text_feat = text_feat / np.linalg.norm(text_feat)

    results = []

    # multithread encoding
    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        futures = {executor.submit(encode_image, model, processor, p, device): p for p in img_paths}
        for fut in as_completed(futures):
            feat = fut.result()
            if feat is None:
                continue
            sim = float(np.dot(text_feat, feat))
            results.append((sim, futures[fut]))

    # apply loose threshold
    keep = [p for sim, p in results if sim >= threshold]

    # fallback: if too few survived, keep top-K
    if len(keep) < keep_num:
        results.sort(key=lambda x: x[0], reverse=True)
        keep = [p for _, p in results[:keep_num]]

    return keep


# ================== Main ==================
def main(json_file: str, model_path: str, output_dir: str,
         device="cuda", num_per_tag=30, threshold=0.2):

    os.makedirs(output_dir, exist_ok=True)

    # ---- Load model ----
    model_name = infer_model_name_from_ckpt(args.model_path)
    model_kwargs = {}
    if not (model_name.endswith("S3") or model_name.endswith("S4") or model_name.endswith("L-14")):
        model_kwargs = {"image_mean": (0, 0, 0), "image_std": (1, 1, 1)}
    model, _, image_processor = open_clip.create_model_and_transforms(model_name, pretrained=args.model_path, **model_kwargs)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    model = model.to(device).eval()
    model = reparameterize_model(model)
    tokenizer = open_clip.get_tokenizer(model_name)

    with open(json_file, "r") as f:
        data = json.load(f)

    # ====== update ref_dir inline ======
    for entry in data["pet_scenes"]:
        tag = entry["tag"]
        ref_dir = Path(output_dir) / tag.replace(" ", "_")
        os.makedirs(ref_dir, exist_ok=True)
        entry["ref_dir"] = str(ref_dir)  # update JSON

        tmp_dir = Path(output_dir) / f"tmp_{tag.replace(' ', '_')}"
        if tmp_dir.exists():
            shutil.rmtree(tmp_dir)

        print(f"[INFO] Searching for '{tag}' ...")
        download_images_from_prompt(tag, str(tmp_dir), max_images=num_per_tag * 5)  # overfetch

        img_paths = list(tmp_dir.rglob("*.jpg")) + list(tmp_dir.rglob("*.png"))
        if not img_paths:
            print(f"[WARN] No images found for {tag}")
            continue

        print(f"[INFO] Filtering {len(img_paths)} images with CLIP...")
        keep_paths = filter_images(
            model, image_processor, tokenizer, tag,
            img_paths, keep_num=num_per_tag,
            device=device, threshold=threshold
        )

        # copy filtered images
        for i, p in enumerate(keep_paths):
            out_path = ref_dir / f"{tag.replace(' ', '_')}_{i:03d}{p.suffix}"
            shutil.copy(p, out_path)

        print(f"[DONE] {tag}: kept {len(keep_paths)} images -> {ref_dir}")

        shutil.rmtree(tmp_dir)  # clean temp dir

    # save updated JSON with new ref_dir
    updated_json = Path(output_dir) / "scene_updated.json"
    with open(updated_json, "w") as f:
        json.dump(data, f, indent=2)

    print(f"[DONE] Updated JSON saved to {updated_json}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene_json", type=str, default = "./scene_tags/pet_scenes_crawled.json", help="Path to original scene_json")
    parser.add_argument("--model_path", type=str, default = "./checkpoints/mobileclip2_s4.pt", help="Path to MobileCLIP2 checkpoint")
    parser.add_argument("--output_dir", type=str, default="./ref_images", help="Directory to save images + new JSON")
    parser.add_argument("--device", type=str, default="cuda", help="Device (cuda or cpu)")
    parser.add_argument("--num_per_tag", type=int, default=30, help="Number of filtered samples per tag")
    parser.add_argument("--threshold", type=float, default=0.2, help="Similarity threshold for filtering")
    args = parser.parse_args()

    main(args.scene_json, args.model_path, args.output_dir,
         device=args.device, num_per_tag=args.num_per_tag, threshold=args.threshold)