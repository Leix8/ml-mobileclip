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
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA

from embedding_processor import *

try:
    import umap
    HAS_UMAP = True
except ImportError:
    HAS_UMAP = False
    print("[WARN] umap-learn not installed; skipping UMAP.")

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

def get_embeddings(model, processor, tokenizer, tag: str, ref_paths: List[Path], cand_paths: List[Path], device="cuda"):
    # text
    tok = tokenizer([tag]).to(device)
    with torch.no_grad():
        text_feat = model.encode_text(tok).cpu().numpy()[0]

    # ref_images
    ref_feats = []
    if ref_paths:
        for ref_path in ref_paths:
            try:
                img = Image.open(ref_path).convert("RGB")
            except Exception:
                continue
            img_tensor = processor(img).unsqueeze(0).to(device)
            with torch.no_grad():
                feat = model.encode_image(img_tensor).cpu().numpy()[0]
            ref_feats.append(feat)
        
    cand_feats = []
    if cand_paths:
        for cand_path in cand_paths:
            try:
                img = Image.open(cand_path).convert("RGB")
            except Exception:
                continue
            img_tensor = processor(img).unsqueeze(0).to(device)
            with torch.no_grad():
                feat = model.encode_image(img_tensor).cpu().numpy()[0]
            cand_feats.append(feat)

    return text_feat, np.array(ref_feats), np.array(cand_feats) 

def normalize(x: np.ndarray) -> np.ndarray:
    return x / np.linalg.norm(x, axis=-1, keepdims=True)

# ================== Whitening & ISD ==================
def compute_whitening_matrix(X: np.ndarray, eps=1e-6):
    mu = X.mean(axis=0, keepdims=True)
    Xc = X - mu
    cov = np.cov(Xc, rowvar=False)
    U, S, _ = np.linalg.svd(cov)
    W = U @ np.diag(1.0 / np.sqrt(S + eps)) @ U.T
    return W, mu

def apply_whitening(X: np.ndarray, W: np.ndarray, mu: np.ndarray):
    Xc = X - mu
    Xw = Xc @ W
    return Xw / np.linalg.norm(Xw, axis=1, keepdims=True)

def apply_isd(X: np.ndarray, topk_remove: int = 1):
    mu = X.mean(axis=0, keepdims=True)
    Xc = X - mu  # [N, D]
    # PCA via SVD on features
    U, S, Vt = np.linalg.svd(Xc, full_matrices=False)  # Vt: [min(N,D), D]
    V = Vt.T  # [D, min(N,D)]

    if topk_remove > 0:
        Xc_proj = Xc - (Xc @ V[:, :topk_remove]) @ V[:, :topk_remove].T
    else:
        Xc_proj = Xc
    return normalize(Xc_proj)

# ================== Visualization ==================
def run_projection(X: np.ndarray, method: str, seed=1234) -> np.ndarray:
    method = method.lower()
    if method == "pca":
        return PCA(n_components=2, random_state=seed).fit_transform(X)
    elif method == "umap" and HAS_UMAP:
        reducer = umap.UMAP(n_components=2, random_state=seed, metric="cosine", min_dist=0.1)
        return reducer.fit_transform(X)
    elif method == "tsne":
        return TSNE(n_components=2, random_state=seed, metric="cosine", perplexity=30).fit_transform(X)
    else:
        raise ValueError(f"Unknown or unavailable method: {method}")

from matplotlib.offsetbox import OffsetImage, AnnotationBbox

def run_projection_fast(X: np.ndarray, method: str, seed=1234) -> np.ndarray:
    n_comp = min(50, X.shape[0]-1, X.shape[1])
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

def run_projection_supervised(X: np.ndarray, y: List[str], method: str) -> np.ndarray:
    if method.lower() == "lda":
        lda = LDA(n_components=2)
        return lda.fit_transform(X, y)
    else:
        raise ValueError(f"Unknown supervised projection method: {method}")

def plot_embeddings_with_thumbnails(Z, labels, modalities, image_paths, texts, method, save_path):
    fig, ax = plt.subplots(figsize=(12, 10))
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
        else:
            ax.text(x, y, texts[i], fontsize=8, color="red", ha="center")
    ax.set_title(f"Projection with {method} (Thumbnails/Text)")
    ax.set_xlim(Z[:,0].min()-1, Z[:,0].max()+1)
    ax.set_ylim(Z[:,1].min()-1, Z[:,1].max()+1)
    plt.savefig(save_path, dpi=150)
    plt.close()

def plot_embeddings(Z, labels, modalities, method, save_path):
    plt.figure(figsize=(8, 6))
    markers = {"reference": "o", "text": "X", "candidate": "*",}
    for lab in set(labels):
        for mod in ["reference", "candidate", "text"]:
            idx = [i for i, (l, m) in enumerate(zip(labels, modalities)) if l == lab and m == mod]
            if not idx:
                continue
            plt.scatter(Z[idx, 0], Z[idx, 1], label=f"{lab}-{mod}", marker=markers[mod], alpha=0.7)
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

# ================== Quantitative Metrics ==================
def compute_cluster_stats(X: np.ndarray, labels: List[str]):
    sims = X @ X.T
    labels = np.array(labels)
    intra, inter = [], []
    for i in range(len(labels)):
        for j in range(i+1, len(labels)):
            if labels[i] == labels[j]:
                intra.append(sims[i,j])
            else:
                inter.append(sims[i,j])
    intra_mean = np.mean(intra) if intra else 0
    inter_mean = np.mean(inter) if inter else 0
    ratio = intra_mean / (inter_mean + 1e-6)
    return {"intra": float(intra_mean), "inter": float(inter_mean), "ratio": float(ratio)}

# ================== Main ==================
def main(args):
    json_path = args.scene_json
    model_path = args.model_path
    json_file_name = os.path.splitext(os.path.basename(args.scene_json))[0]
    output_dir = os.path.join(args.output_dir, json_file_name)
    device= args.device
    os.makedirs(output_dir, exist_ok=True)
    data = load_json(json_path)["pet_scenes"]

    model_name = infer_model_name_from_ckpt(model_path)
    model_kwargs = {}
    if not (model_name.endswith("S3") or model_name.endswith("S4") or model_name.endswith("L-14")):
        model_kwargs = {"image_mean": (0,0,0), "image_std": (1,1,1)}
    model, _, image_processor = open_clip.create_model_and_transforms(model_name, pretrained=model_path, **model_kwargs)
    device = torch.device(device if torch.cuda.is_available() else "cpu")
    model = model.to(device).eval()
    model = reparameterize_model(model)
    tokenizer = open_clip.get_tokenizer(model_name)

    all_embeddings, labels, modalities, image_paths, texts = [], [], [], [], []
    for entry in tqdm(data, desc="Processing scenes"):
        tag = entry["tag"]
        ref_dir = entry.get("ref_dir", "None")
        ref_paths = load_images_from_dir(ref_dir) if ref_dir else []
        cand_dir = entry.get("candidate_dir", None)
        cand_paths = load_images_from_dir(cand_dir) if cand_dir else []
        text_feat, ref_feats, cand_feats = get_embeddings(model, image_processor, tokenizer, tag, ref_paths, cand_paths, device=device)
        text_feat = normalize(text_feat.reshape(1, -1))[0]
        ref_feats = normalize(ref_feats) if len(ref_feats) > 0 else np.zeros((0, text_feat.shape[0]))
        all_embeddings.append(text_feat)
        labels.append(tag); modalities.append("text"); texts.append(tag); image_paths.append(None)
        for p, f in zip(ref_paths, ref_feats):
            all_embeddings.append(f)
            labels.append(tag); modalities.append("reference"); texts.append(""); image_paths.append(p)
        for p, f in zip(cand_paths, cand_feats):
            all_embeddings.append(f)
            labels.append(tag); modalities.append("candidate"); texts.append(""); image_paths.append(p)
    
    all_embeddings = np.array(all_embeddings)
    print("Embeddings built, now compute transforms...")

    # Whitening
    if args.whitening:
        try:
            W, mu = compute_whitening_matrix(all_embeddings)
            all_embeddings_whiten = apply_whitening(all_embeddings, W, mu)
        except Exception as e:
            print(f"[WARN] Whitening failed: {e}")
            all_embeddings_whiten = None

    # ISD
    if args.isd:
        try:
            all_embeddings_isd = apply_isd(all_embeddings, topk_remove=1)
        except Exception as e:
            print(f"[WARN] ISD failed: {e}")
            all_embeddings_isd = None

    # Run projections & plots
    def run_all(X, suffix):
        method = args.projection
        if method == "umap" and not HAS_UMAP:
            print(f"[ERROR] running umap projection while HAS_UMAP has not been set")
            return
        try:
            Z = run_projection_fast(X, method)
            plot_embeddings(Z, labels, modalities, f"{method}{suffix}", os.path.join(output_dir, f"proj_{method}{suffix}.png"))
            plot_embeddings_with_thumbnails(Z, labels, modalities, image_paths, texts, f"{method}{suffix}", os.path.join(output_dir, f"proj_{method}{suffix}_thumbs.png"))
        except Exception as e:
            print(f"[WARN] {method} on {suffix} failed: {e}")
        # Similarity heatmap
        plot_similarity_heatmap(X, [f"{l}-{m}" for l,m in zip(labels, modalities)], modalities, os.path.join(output_dir, f"similarity_heatmap{suffix}.png"))
        # Stats
        stats = compute_cluster_stats(X, labels)
        with open(os.path.join(output_dir, f"cluster_stats{suffix}.json"), "w") as f:
            json.dump(stats, f, indent=2)
        print(f"[STATS-{suffix}] {stats}")

    run_all(all_embeddings, "")
    if args.whitening and all_embeddings_whiten is not None:
        run_all(all_embeddings_whiten, "_whiten")
    if args.isd and all_embeddings_isd is not None:
        run_all(all_embeddings_isd, "_isd")

    # Supervised LDA (on original)
    try:
        Z_lda = run_projection_supervised(all_embeddings, labels, "lda")
        plot_embeddings(Z_lda, labels, modalities, "lda", os.path.join(output_dir, "proj_lda.png"))
        plot_embeddings_with_thumbnails(Z_lda, labels, modalities, image_paths, texts, "lda", os.path.join(output_dir, "proj_lda_thumbs.png"))
    except Exception as e:
        print(f"[WARN] LDA projection failed: {e}")

    # Save embeddings & metadata
    np.save(os.path.join(output_dir, "embeddings.npy"), all_embeddings)
    with open(os.path.join(output_dir, "metadata.json"), "w") as f:
        json.dump({"labels": labels, "modalities": modalities}, f, indent=2)

    print(f"Done. Results saved to {output_dir}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene_json", type=str, default="./scene_tags/pet_scenes.json")
    parser.add_argument("--model_path", type=str, default="./checkpoints/mobileclip2_s4.pt")
    parser.add_argument("--output_dir", type=str, default="./experiments/multimodal_embedding_cluster")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--whitening", action = "store_true")
    parser.add_argument("--isd", action = "store_true")
    parser.add_argument("--projection", type = str, default = "pca", choices=["pca", "tsne", "umap"])

    args = parser.parse_args()

    main(args)