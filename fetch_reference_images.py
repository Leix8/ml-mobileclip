#!/usr/bin/env python3
import os, sys, json, time, re, math, hashlib, argparse
from pathlib import Path
from typing import List, Dict, Optional
import requests
from mobileclip.modules.common.mobileone import reparameterize_model

# Optional MobileCLIP scoring
def _try_open_clip():
    try:
        import open_clip, torch, torch.nn.functional as F
        return open_clip, torch, F
    except Exception:
        return None, None, None

WM_API = "https://commons.wikimedia.org/w/api.php"

CATEGORIES = [
    # High-signal jumping/action categories
    "Jumping dogs",
    "Airborne jumping dogs",
    "Airborne galloping dogs",
    "Dogs jumping into water",
    "Dock jumping",
    "Dog agility",
    "Dog agility jumps",
    # Bonus mixed sources that still often include jumps
    "Disc dog",
    "Flyball",
]

def commons_category_files(category: str, max_files: int = 200) -> List[str]:
    """Return file titles (e.g., 'File:Dog_Leaping.jpg') from a Commons category."""
    titles = []
    cmcontinue = None
    while True:
        params = {
            "action": "query",
            "list": "categorymembers",
            "cmtitle": f"Category:{category.replace(' ', '_')}",  # FIX HERE
            "cmtype": "file",
            "cmlimit": "500",
            "format": "json",
        }
        if cmcontinue:
            params["cmcontinue"] = cmcontinue
        headers = {
            "User-Agent": "mobileclip-dataset/1.0 (https://github.com/leix8/ml-mobileclip)"
        }        
        r = requests.get(WM_API, params=params, timeout=30, headers = headers)
        r.raise_for_status()
        data = r.json()
        for m in data.get("query", {}).get("categorymembers", []):
            titles.append(m["title"])
            if len(titles) >= max_files:
                return titles
        cmcontinue = data.get("continue", {}).get("cmcontinue")
        if not cmcontinue:
            return titles

def commons_file_urls(file_titles: List[str], min_width=512, max_count=999) -> List[Dict]:
    """Fetch best image URLs for given 'File:...' titles."""
    out = []
    for i in range(0, len(file_titles), 50):
        chunk = file_titles[i:i+50]
        params = {
            "action": "query",
            "prop": "imageinfo",
            "titles": "|".join(chunk),
            "iiprop": "url|mime|size",
            "iiurlwidth": str(max(min_width, 512)),
            "format": "json",
        }
        r = requests.get(WM_API, params=params, timeout=30)
        r.raise_for_status()
        pages = r.json().get("query", {}).get("pages", {})
        for p in pages.values():
            infos = p.get("imageinfo", [])
            if not infos:
                continue
            info = infos[0]
            mime = info.get("mime", "")
            if not mime.startswith("image/"):
                continue
            url = info.get("thumburl") or info.get("url")
            width = info.get("thumbwidth") or info.get("width", 0)
            height = info.get("thumbheight") or info.get("height", 0)
            if url and width and width >= min_width:
                out.append({"title": p.get("title", ""), "url": url, "w": width, "h": height})
            if len(out) >= max_count:
                return out
        time.sleep(0.2)  # polite
    return out

def safe_name(s: str) -> str:
    s = s.replace("File:", "").strip()
    s = re.sub(r"[^A-Za-z0-9._-]+", "_", s)
    return s

def download(url: str, outpath: Path) -> bool:
    try:
        with requests.get(url, stream=True, timeout=60) as r:
            r.raise_for_status()
            with open(outpath, "wb") as f:
                for chunk in r.iter_content(1 << 15):
                    if chunk:
                        f.write(chunk)
        return True
    except Exception:
        return False

def perceptual_key(path: Path) -> Optional[str]:
    """Quick byte-hash (not true pHash) to dedupe downloads; sufficient for URL duplicates."""
    try:
        h = hashlib.sha1()
        with open(path, "rb") as f:
            for b in iter(lambda: f.read(1 << 20), b""):
                h.update(b)
        return h.hexdigest()
    except Exception:
        return None

def score_with_open_clip(img_paths: List[Path], text: str, device: str = "cuda:0"):
    open_clip, torch, F = _try_open_clip()
    if open_clip is None:
        print("[INFO] mobileclip not available; skipping similarity filtering.")
        return {p: 0.0 for p in img_paths}

    # minimal MobileCLIP loader
    model_name = "mobileclip2_b"  # adjust to your local checkpoint naming
    try:
        model, _, preprocess = open_clip.create_model_and_transforms(model_name)
    except Exception:
        # fallback to import style some users use
        model = open_clip.create_model(model_name)
        preprocess = open_clip.get_image_transform(model_name)

    model.eval().to(device)
    model = reparameterize_model(model)

    with torch.no_grad():
        text_tok = open_clip.tokenize([text]).to(device)
        text_feat = model.encode_text(text_tok)
        text_feat = text_feat / text_feat.norm(dim=-1, keepdim=True)

    from PIL import Image
    scores = {}
    bs = 16
    batch = []
    paths = []
    for p in img_paths:
        try:
            im = Image.open(p).convert("RGB")
            batch.append(preprocess(im))
            paths.append(p)
            if len(batch) == bs:
                scores.update(_score_batch(model, torch, F, batch, paths, text_feat, device))
                batch, paths = [], []
        except Exception:
            continue
    if batch:
        scores.update(_score_batch(model, torch, F, batch, paths, text_feat, device))
    return scores

def _score_batch(model, torch, F, batch, paths, text_feat, device):
    with torch.no_grad():
        imgs = torch.stack(batch).to(device)
        img_feat = model.encode_image(imgs)
        img_feat = img_feat / img_feat.norm(dim=-1, keepdim=True)
        sim = (img_feat @ text_feat.T).squeeze(1).float().cpu().tolist()
    return {p: s for p, s in zip(paths, sim)}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", type=Path, default=Path(".reference_images/pet_scenes/dog_jumping"))
    ap.add_argument("--target_count", type=int, default=120, help="Stop after this many successful downloads (pre-filter).")
    ap.add_argument("--min_width", type=int, default=768)
    ap.add_argument("--keep_top", type=int, default=60, help="After MobileCLIP scoring, keep top-N (set 0 to keep all).")
    ap.add_argument("--device", type=str, default="cuda:0")
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    # 1) Harvest Commons
    titles = []
    for cat in CATEGORIES:
        ts = commons_category_files(cat, max_files=400)
        titles.extend(ts)
    titles = list(dict.fromkeys(titles))  # dedupe order-preserving

    files = commons_file_urls(titles, min_width=args.min_width, max_count=2000)
    print(f"[INFO] candidate URLs from Commons: {len(files)}")

    # 2) Download until target_count (skipping tiny images)
    got = 0
    seen_hash = set()
    saved_paths = []
    for f in files:
        name = safe_name(f["title"])
        out = args.out_dir / name
        if out.exists():
            hk = perceptual_key(out)
            if hk:
                seen_hash.add(hk)
            saved_paths.append(out)
            got += 1
            if got >= args.target_count:
                break
            continue
        ok = download(f["url"], out)
        if not ok:
            continue
        hk = perceptual_key(out)
        if hk and hk in seen_hash:
            try:
                out.unlink()
            except Exception:
                pass
            continue
        if hk:
            seen_hash.add(hk)
        saved_paths.append(out)
        got += 1
        if got >= args.target_count:
            break

    print(f"[INFO] downloaded images: {len(saved_paths)}")

    # 3) Optional: MobileCLIP similarity filtering
    scores = score_with_mobileclip(saved_paths, "a dog jumping", device=args.device)
    if scores and args.keep_top > 0:
        ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)[:args.keep_top]
        keep_set = set(p for p, _ in ranked)
        removed = 0
        for p in saved_paths:
            if p not in keep_set:
                try:
                    p.unlink()
                    removed += 1
                except Exception:
                    pass
        print(f"[INFO] kept top {len(keep_set)} by MobileCLIP score; removed {removed}.")
    print("[DONE] Your reference set is ready at:", args.out_dir)

if __name__ == "__main__":
    main()