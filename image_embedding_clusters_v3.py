#!/usr/bin/env python3
import os
import json
import torch
import numpy as np
from pathlib import Path
from typing import List, Dict
from PIL import Image
from tqdm import tqdm

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import open_clip
from mobileclip.modules.common.mobileone import reparameterize_model
from model_name_map import MODEL_NAME_MAP, infer_model_name_from_ckpt

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

from embedding_processor import *

try:
    import umap
    HAS_UMAP = True
except ImportError:
    HAS_UMAP = False
    print("[WARN] umap-learn not installed; skipping UMAP.")

import open_clip  # must include your MobileCLIP2 setup


# ================== Helpers ==================
def load_json(json_path: str) -> Dict:
    with open(json_path, "r") as f:
        return json.load(f)

def load_images_from_dir(ref_dir: str, exts={".jpg", ".jpeg", ".png"}):
    paths = []
    for p in Path(ref_dir).rglob("*"):
        if p.suffix.lower() in exts:
            paths.append(p)
    return paths

def get_embeddings(model, processor, tokenizer, tag: str, image_paths: List[Path], device="cuda"):
    # text
    tok = tokenizer([tag]).to(device)
    with torch.no_grad():
        text_feat = model.encode_text(tok).cpu().numpy()[0]

    # images
    image_feats = []
    for img_path in image_paths:
        try:
            img = Image.open(img_path).convert("RGB")
        except Exception:
            continue
        img_tensor = processor(img).unsqueeze(0).to(device)
        with torch.no_grad():
            feat = model.encode_image(img_tensor).cpu().numpy()[0]
        image_feats.append(feat)

    return text_feat, np.array(image_feats)

def normalize(x: np.ndarray) -> np.ndarray:
    return x / np.linalg.norm(x, axis=-1, keepdims=True)


# ================== Visualization ==================
def run_projection(X: np.ndarray, method: str, seed=1234) -> np.ndarray:
    method = method.lower()
    if method == "pca":
        Z = PCA(n_components=2, random_state=seed).fit_transform(X)
        return Z
    elif method == "umap" and HAS_UMAP:
        reducer = umap.UMAP(n_components=2, random_state=seed, metric="cosine", min_dist=0.1)
        return reducer.fit_transform(X)
    elif method == "tsne":
        Z = TSNE(n_components=2, random_state=seed, metric="cosine", perplexity=30).fit_transform(X)
        return Z
    else:
        raise ValueError(f"Unknown or unavailable method: {method}")

from matplotlib.offsetbox import OffsetImage, AnnotationBbox

def run_projection_fast(X: np.ndarray, method: str, seed=1234) -> np.ndarray:
    """Fast projection: PCA to 50D first, then apply chosen method."""
    n_comp = min(50, X.shape[0]-1, X.shape[1])  # cannot exceed n_samples-1
    X_reduced = PCA(n_components=n_comp, random_state=seed).fit_transform(X)    
    if method == "pca":
        return PCA(n_components=2, random_state=seed).fit_transform(X_reduced)
    elif method == "umap" and HAS_UMAP:
        reducer = umap.UMAP(n_components=2, random_state=seed, metric="cosine", min_dist=0.1)
        return reducer.fit_transform(X_reduced)
    elif method == "tsne":
        max_perp = max(2, min(30, X_reduced.shape[0] // 3))
        return TSNE(n_components=2, random_state=seed, metric="cosine", perplexity=max_perp).fit_transform(X_reduced)
    else:
        raise ValueError(f"Unknown or unavailable method: {method}")

def plot_embeddings_with_thumbnails(Z, labels, modalities, image_paths, texts, method, save_path):
    """Plot with thumbnails for images and text labels for text points."""
    fig, ax = plt.subplots(figsize=(12, 10))

    # faint scatter backdrop
    ax.scatter(Z[:,0], Z[:,1], alpha=0.1, s=5, color="gray")

    zoom = 0.5 if len(Z) < 100 else 0.25

    for i, (x, y) in enumerate(Z):
        if modalities[i] == "image" and image_paths[i] is not None:
            try:
                img = Image.open(image_paths[i]).convert("RGB")
                img.thumbnail((64, 64))
                imagebox = OffsetImage(img, zoom=zoom)
                ab = AnnotationBbox(imagebox, (x, y), frameon=False)
                ax.add_artist(ab)
            except Exception:
                ax.text(x, y, "[img]", fontsize=6, color="blue")
        else:  # text
            ax.text(x, y, texts[i], fontsize=8, color="red", ha="center")

    ax.set_title(f"Projection with {method} (Thumbnails/Text)")
    ax.set_xlim(Z[:,0].min()-1, Z[:,0].max()+1)
    ax.set_ylim(Z[:,1].min()-1, Z[:,1].max()+1)
    plt.savefig(save_path, dpi=150)
    plt.close()

def plot_embeddings(Z, labels, modalities, method, save_path):
    plt.figure(figsize=(8, 6))
    markers = {"image": "o", "text": "X"}
    for lab in set(labels):
        for mod in ["image", "text"]:
            idx = [i for i, (l, m) in enumerate(zip(labels, modalities)) if l == lab and m == mod]
            if not idx:
                continue
            plt.scatter(
                Z[idx, 0], Z[idx, 1],
                label=f"{lab}-{mod}",
                marker=markers[mod], alpha=0.7
            )
    plt.legend()
    plt.title(f"Projection with {method}")
    plt.savefig(save_path, dpi=150)
    plt.close()

def plot_similarity_heatmap(embeddings, labels, modalities, save_path):
    sims = embeddings @ embeddings.T
    sns.heatmap(sims, xticklabels=labels, yticklabels=labels, cmap="coolwarm", center=0)
    plt.title("Cosine Similarity Heatmap")
    plt.savefig(save_path, dpi=150)
    plt.close()


# ================== Main ==================
def main(json_path: str, model_path: str, output_dir: str, device="cuda"):
    os.makedirs(output_dir, exist_ok=True)

    # ---- Load JSON ----
    data = load_json(json_path)["pet_scenes"]

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

    all_embeddings = []
    labels, modalities, image_paths, texts = [], [], [], []

    for entry in tqdm(data, desc="Processing scenes"):
        tag = entry["tag"]
        ref_dir = entry.get("ref_dir")
        img_paths = load_images_from_dir(ref_dir) if ref_dir else []

        text_feat, img_feats = get_embeddings(model, image_processor, tokenizer, tag, img_paths, device=device)

        text_feat = normalize(text_feat.reshape(1, -1))[0]
        img_feats = normalize(img_feats) if len(img_feats) > 0 else np.zeros((0, text_feat.shape[0]))

        # text
        all_embeddings.append(text_feat)
        labels.append(tag)
        modalities.append("text")
        texts.append(tag)
        image_paths.append(None)

        # images
        for p, f in zip(img_paths, img_feats):
            all_embeddings.append(f)
            labels.append(tag)
            modalities.append("image")
            texts.append("")   # no text for image
            image_paths.append(p)
    
    all_embeddings = np.array(all_embeddings)
    print(f"complete encoding part, now calculate projection...")
    
    # ---- Projections ----
    for method in ["pca", "tsne", "umap"]:
            if method == "umap" and not HAS_UMAP:
                continue
            Z = run_projection_fast(all_embeddings, method)

            # scatter plot
            plot_embeddings(Z, labels, modalities, method, os.path.join(output_dir, f"proj_{method}.png"))

            # thumbnail/text plot
            plot_embeddings_with_thumbnails(
                Z, labels, modalities, image_paths, texts,
                method, os.path.join(output_dir, f"proj_{method}_thumbs.png")
            )

    # ---- Similarity Heatmap ----
    plot_similarity_heatmap(all_embeddings, [f"{l}-{m}" for l, m in zip(labels, modalities)], modalities,
                            os.path.join(output_dir, "similarity_heatmap.png"))

    # ---- Save embeddings & metadata ----
    np.save(os.path.join(output_dir, "embeddings.npy"), all_embeddings)
    with open(os.path.join(output_dir, "metadata.json"), "w") as f:
        json.dump({"labels": labels, "modalities": modalities}, f, indent=2)

    print(f"Done. Results saved to {output_dir}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene_json", type=str, default="./scene_tags/pet_scenes.json")
    parser.add_argument("--model_path", type=str, default="./checkpoints/mobileclip2_s4.pt")
    parser.add_argument("--output_dir", type=str, default="./experiements/multimodal_embedding_cluster", help="Directory to save results")
    parser.add_argument("--device", type=str, default="cuda", help="cuda or cpu")
    args = parser.parse_args()

    main(args.scene_json, args.model_path, args.output_dir, device=args.device)