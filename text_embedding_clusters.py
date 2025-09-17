#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Text embedding clustering visualization with MobileCLIP.
- Input: one or more JSON files containing { "group_name": [ {"tag": "..."} ] }
- Output: scatter plots (PCA/UMAP/t-SNE), cosine heatmap, KMeans report, dendrogram
"""

import argparse, os, sys, json, random
from pathlib import Path
from typing import List, Dict

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import torch
import torch.nn.functional as F

# ---------------- MobileCLIP safe import ----------------
def _import_mobileclip():
    try:
        import mobileclip
        return sys.modules["mobileclip"]
    except Exception as e:
        raise ImportError("Could not import 'mobileclip'. Please check PYTHONPATH.") from e

# ---------------- Determinism ----------------
def set_seed(seed=1234):
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":16:8")
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(True)

# ---------------- JSON Loader ----------------
def load_tags_from_jsons(paths: List[str]) -> List[str]:
    tags = []
    for p in paths:
        with open(p, "r") as f:
            data = json.load(f)
        for group_name, items in data.items():
            if isinstance(items, list):
                for it in items:
                    tag = str(it.get("tag", "")).strip()
                    if tag and tag.lower() != "na":
                        tags.append(tag)
    return tags

# ---------------- Projections ----------------
def run_projection(X: np.ndarray, method: str, seed=1234) -> np.ndarray:
    method = method.lower()
    if method == "pca":
        from sklearn.decomposition import PCA
        return PCA(n_components=2, random_state=seed).fit_transform(X)
    elif method == "umap":
        try:
            import umap
            reducer = umap.UMAP(n_components=2, random_state=seed, metric="cosine", min_dist=0.1)
            return reducer.fit_transform(X)
        except Exception:
            print("[WARN] umap-learn not installed, falling back to t-SNE")
            method = "tsne"
    if method == "tsne":
        from sklearn.manifold import TSNE
        return TSNE(n_components=2, random_state=seed, metric="cosine", init="pca",
                    perplexity=min(30, max(5, len(X)//2))).fit_transform(X)
    raise ValueError("Unknown method, use pca|umap|tsne.")

# ---------------- Plots ----------------
def save_scatter(X2, labels, out, title=""):
    plt.figure(figsize=(10,7), dpi=120)
    plt.scatter(X2[:,0], X2[:,1], s=40, alpha=0.8)
    for (x,y,l) in zip(X2[:,0], X2[:,1], labels):
        plt.text(x, y, l, fontsize=8, ha="center", va="center")
    plt.title(title)
    plt.grid(True, alpha=0.25)
    plt.tight_layout()
    plt.savefig(out); plt.close()

def save_heatmap(C, labels, out):
    import seaborn as sns
    plt.figure(figsize=(8,7), dpi=120)
    sns.heatmap(C, vmin=-1, vmax=1, cmap="viridis",
                xticklabels=labels, yticklabels=labels)
    plt.title("Cosine Similarity")
    plt.tight_layout()
    plt.savefig(out); plt.close()

def save_dendrogram(C, labels, out):
    from scipy.cluster.hierarchy import linkage, dendrogram
    from scipy.spatial.distance import squareform
    D = 1 - np.clip(C, -1, 1)
    D = D + np.eye(D.shape[0])*1e-6
    Y = squareform(D, checks=False)
    Z = linkage(Y, method="average")
    plt.figure(figsize=(12, 5 + 0.08*len(labels)), dpi=120)
    dendrogram(Z, labels=labels, leaf_rotation=90)
    plt.title("Hierarchical (average-link) on cosine distance")
    plt.tight_layout()
    plt.savefig(out); plt.close()

# ---------------- Clustering ----------------
def auto_kmeans(X: np.ndarray, kmin=2, kmax=10, seed=1234):
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score
    best_k, best_score, best_labels = 1, -1, None
    for k in range(kmin, min(kmax, len(X))+1):
        km = KMeans(n_clusters=k, n_init=10, random_state=seed)
        labels = km.fit_predict(X)
        if len(set(labels)) < 2: continue
        sc = silhouette_score(X, labels, metric="cosine")
        if sc > best_score:
            best_k, best_score, best_labels = k, sc, labels
    if best_labels is None:
        return np.zeros(len(X), dtype=int), 1, 0.0
    return best_labels, best_k, float(best_score)

# ---------------- Main ----------------
def main():
    ap = argparse.ArgumentParser("MobileCLIP text embedding visualization")
    ap.add_argument("--jsons", type=str, nargs="+", required=True,
                    help="JSON file(s) with {group:[{tag:..., ref_dir:...}]}")
    ap.add_argument("--model_path", type=str, default="./checkpoints/mobileclip_s2.pt")
    ap.add_argument("--device", type=str, default="cuda:0")
    ap.add_argument("--outdir", type=str, default="./experiements/text_embedding_cluster")
    ap.add_argument("--proj", type=str, default="umap", choices=["pca","umap","tsne"])
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--auto_kmax", type=int, default=10)
    args = ap.parse_args()

    set_seed(args.seed)
    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)

    # ---- Load tags ----
    tags = load_tags_from_jsons(args.jsons)
    print(f"Loaded {len(tags)} text prompts.")

    # ---- Load MobileCLIP ----
    mobileclip = _import_mobileclip()
    model_name = os.path.splitext(os.path.basename(args.model_path))[0]
    model, _, _ = mobileclip.create_model_and_transforms(model_name, pretrained=args.model_path)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    model = model.to(device).eval()
    tokenizer = mobileclip.get_tokenizer(model_name)

    # ---- Encode text ----
    with torch.no_grad():
        toks = tokenizer(tags).to(device)
        te = model.encode_text(toks)
        te = F.normalize(te.float(), dim=-1)
    E = te.cpu().numpy()

    # ---- Cosine similarity ----
    C = E @ E.T
    save_heatmap(C, tags, outdir/"cosine_heatmap.png")

    # ---- Projection ----
    X2 = run_projection(E, args.proj, seed=args.seed)
    save_scatter(X2, tags, outdir/f"scatter_{args.proj}.png",
                 title=f"Text embeddings ({args.proj.upper()})")

    # ---- KMeans ----
    labels, k_opt, sil = auto_kmeans(E, kmin=2, kmax=args.auto_kmax, seed=args.seed)
    save_scatter(X2, [f"{t} (c{labels[i]})" for i,t in enumerate(tags)],
                 outdir/f"kmeans_{args.proj}.png",
                 title=f"KMeans (k={k_opt}, silhouette={sil:.3f})")

    with open(outdir/"kmeans_report.json","w") as f:
        json.dump({"k": int(k_opt), "silhouette_cosine": sil,
                   "clusters": {int(c): [tags[i] for i in np.where(labels==c)[0]]
                                for c in np.unique(labels)}}, f, indent=2)

    # ---- Dendrogram ----
    save_dendrogram(C, tags, outdir/"dendrogram.png")

    print(f"[OK] Wrote outputs to {outdir}")

if __name__ == "__main__":
    main()