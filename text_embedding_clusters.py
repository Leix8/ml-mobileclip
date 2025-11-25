#!/usr/bin/env python3
from __future__ import annotations
import os
import json
import torch
from torch.cuda.amp import autocast

import numpy as np
from pathlib import Path
from PIL import Image
from tqdm import tqdm
import random
import itertools

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.offsetbox import OffsetImage, AnnotationBbox

import open_clip
from mobileclip.modules.common.mobileone import reparameterize_model
from model_name_map import MODEL_NAME_MAP, infer_model_name_from_ckpt

import pandas as pd

import seaborn as sns

from sklearn.decomposition import PCA
from sklearn.cluster import DBSCAN
from sklearn.metrics import pairwise_distances
from sklearn.preprocessing import normalize
from sklearn.feature_extraction.text import TfidfVectorizer

from embedding_processor import *
from typing import List, Dict, Any, Optional, Callable, Sequence, Tuple

import plotly.express as px
from plotly.offline import plot as plot_offline

# ================== Quantitative Metrics ==================
def compute_cluster_stats(X: np.ndarray, labels: List[str]):
    # add stats for clusters counts, average cluster size, intra/inter-cluster similarity, noise ratio, etc.
    labels_set = set(labels)
    cluster_cnt = len(labels_set) - (1 if "-1" in labels_set else 0)
    average_cluster_size = len(labels) / len(labels_set) if labels_set else 0
    noise_cnt = labels.count("-1")
    noise_ratio = labels.count("-1") / len(labels) if len(labels) > 0 else 0

    X = normalize(X)
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
    
    return {"intra": float(intra_mean), "inter": float(inter_mean), "ratio": float(ratio), "cluster_count": cluster_cnt, "average_cluster_size": float(average_cluster_size), "nosie_cnt": noise_cnt, "noise_ratio": float(noise_ratio)}


# ================== Prompt Generation & Clustering ==================
subjects = [
    "a dog", "a cat", "a puppy", "a kitten", "two dogs", "two cats",
    # "a corgi", "a husky", "a golden retriever", "a small dog", "a large dog",
    # "a small cat", "a pet", "a domestic animal"
]

actions = [
    # Motion
    "is running", "is jumping", "is walking", "is chasing", "is rolling",
    # # Interaction
    "is playing", "is fetching", "is tugging a toy", "is biting", "is scratching",
    "is licking its paw", "is sniffing something", "is shaking its body",
    # Emotion / Expression
    "is barking", "is meowing", "is wagging its tail", "is stretching",
    "is yawning", "is curious", "is watching something",
    # Static / Rest
    "is sleeping", "is lying down", "is sitting", "is resting", "is staying still"
]

scenes = [
    "on the grass", "in the park", "on the beach", "in the living room",
    "in the bedroom", "in the kitchen", "in the backyard", "in the garden",
    "on the sofa", "on the bed", "by the window", "on the floor",
    "in the yard", "at home", "outdoors", "under the table"
]

objects = [
    "with a ball", "with a frisbee", "with a toy", "with a stick", "with a bone",
    "with a rope", "with a plush toy", "with a pillow", "with food", "with a bowl",
    "chasing another pet", "playing with a person", "looking at the camera",
    "being brushed", "taking a bath", "wearing a collar", "next to its owner"
]
templates = [
    "{s} {a}",
    "{s} {a} {sc}",
    "{s} {a} {o}",
    "{s} {a} {sc} {o}"
]

def generate_prompts(subjects, actions, scenes, objects, templates,
                     max_samples=200000, seed=42):
    random.seed(seed)
    prompts = set()

    combos = list(itertools.product(subjects, actions, scenes, objects))
    random.shuffle(combos)

    for (s, a, sc, o) in tqdm(combos, desc="Generating prompts", total=len(combos)):
        for t in templates:
            p = t.format(s=s, a=a, sc=sc, o=o).strip()
            # small cleaning
            p = p.replace("  ", " ").replace(" ,", ",")
            prompts.add(p)
            if len(prompts) >= max_samples:
                break
        if len(prompts) >= max_samples:
            break

    return sorted(list(prompts))

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


def _l2_normalize(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    n = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.clip(n, eps, None)

@torch.no_grad()
def encode_text_batch(model, tokenizer, texts: list[str], device: str,
                     context_length: int, use_amp: bool = True) -> torch.Tensor:
    """
    Batch encoder: texts (B) -> [B, D] tensor on CPU (float32), L2-normalized.
    """
    # Tokenize once on CPU; move as a batch
    tokens = tokenizer(
        texts, context_length=context_length
    ).to(device, non_blocking=True)

    with autocast(enabled=use_amp):
        tfeat = model.encode_text(tokens)      # [B, D] on device
        tfeat = F.normalize(tfeat, dim=-1)

    return tfeat.float().cpu()                 # [B, D] on CPU

def _embed_prompts_with_mobileclip2(
    prompts: List[str],
    model,
    tokenizer,
    device: str,
    context_length: int,
    use_amp: bool = True,
    batch_size: int = 1024,
) -> np.ndarray:
    """
    Encode a list of prompts using your encode_text() (single-string API).
    Falls back to provided encode_text_fn if passed explicitly.
    """
    embs = []
    model.eval()
    with torch.no_grad():
        for i in tqdm(range(0, len(prompts), batch_size), desc="Encoding Prompts"):
            # Pass the batch of prompts as a list
            e = encode_text_batch(
                model=model,
                tokenizer=tokenizer,
                texts=prompts[i:i + batch_size],  # Pass the batch of prompts
                device=device,
                context_length=context_length,
                use_amp=use_amp
            )
            if isinstance(e, torch.Tensor):
                e = e.detach().float().cpu().numpy()
            embs.append(e)  # Collect embeddings directly
    embs = np.concatenate(embs, axis=0)  # Properly stack embeddings along the first dimension
    embs = _l2_normalize(embs)
    print(f"[INFO] Encoded {len(prompts)} prompts into embeddings of shape {embs.shape}.")
    return embs

def embed_promts(
    prompts: List[str],
    model,
    tokenizer,
    device: str = "cuda",
    context_length: int = 77,
    use_amp: bool = True,
    max_output: Optional[int] = None,
    return_info: bool = False,
    topk_per_row: int = 0,       # 0 => full O(N^2). For large N, set e.g. 64 for memory savings.
    progress: bool = True,
) -> List[str] | Dict[str, Any]:
    
    N = len(prompts)
    if N == 0:
        return []

    E = _embed_prompts_with_mobileclip2(
        prompts, model, tokenizer, device, context_length, use_amp=use_amp
    ).astype(np.float32)
    # Safety: normalize
    E /= (np.linalg.norm(E, axis=1, keepdims=True) + 1e-12)
    
    return E


def cluster_and_plot_dbscan(
    prompts: Sequence[str],
    embeddings: Sequence[Sequence[float]],
    eps: float = 0.25,
    min_samples: int = 5,
    metric: str = "cosine",
    annotate: bool = True,
    legend_outside: bool = True,
) -> Tuple[np.ndarray, plt.Figure]:
    if len(prompts) == 0:
        raise ValueError("`prompts` is empty.")
    if len(prompts) != len(embeddings):
        raise ValueError("`prompts` and `embeddings` must have the same length.")

    X = np.asarray(embeddings, dtype=float)
    if X.ndim != 2:
        raise ValueError("`embeddings` must be 2D array-like of shape (N, D).")

    # Normalize for stability (useful for cosine or euclidean on embeddings)
    Xn = normalize(X, norm="l2", axis=1)

    # DBSCAN
    db = DBSCAN(eps=eps, min_samples=min_samples, metric=metric)
    labels = db.fit_predict(Xn)

    # 2D projection (PCA just for viz)
    if Xn.shape[1] == 1:
        Xn = np.hstack([Xn, np.zeros((Xn.shape[0], 1))])
    X2 = PCA(n_components=2, random_state=0).fit_transform(Xn)

    # Plot
    # Use constrained layout to help with legend placement
    fig, ax = plt.subplots(figsize=(8, 6), layout="constrained")

    unique_labels = np.unique(labels)
    # Plot non-noise clusters
    for lab in [l for l in unique_labels if l != -1]:
        idx = labels == lab
        sc = ax.scatter(X2[idx, 0], X2[idx, 1], s=18, alpha=0.85, label=f"Cluster {lab}")
        if annotate:
            cx, cy = X2[idx].mean(axis=0)
            ax.annotate(f"C{lab} ({idx.sum()})", (cx, cy), xytext=(5, 5), textcoords="offset points")

    # Plot noise
    if -1 in unique_labels:
        idx_noise = labels == -1
        if idx_noise.any():
            ax.scatter(X2[idx_noise, 0], X2[idx_noise, 1], s=16, marker="x", alpha=0.85, label="Noise (-1)")

    ax.set_title("embedding clusters (DBScan | PCA 2D)")
    ax.set_xlabel("PC 1")
    ax.set_ylabel("PC 2")
    ax.margins(0.05)          # a bit of breathing room
    ax.set_aspect("auto")     # ensure not forced to square

    # Legend handling: place outside at the bottom if many entries
    handles, labels_txt = ax.get_legend_handles_labels()
    if handles:
        if legend_outside:
            ncols = min(len(handles), 6)
            # Put legend below plot; constrained layout will allocate space
            leg = ax.legend(handles, labels_txt, loc="upper center",
                            bbox_to_anchor=(0.5, -0.12), ncol=ncols, frameon=False)
            # If your Matplotlib is older and ignores constrained layout, uncomment next line:
            # plt.tight_layout(rect=[0, 0.06, 1, 1])
        else:
            ax.legend(loc="best")

    return labels, fig

def plot_clusters_with_text_samples(
    prompts,
    embeddings,
    labels,
    sample_per_cluster: int = 3,
    sample_of_cluster: int = 15,  # Fixed number of clusters to sample
    random_seed: int = 42,
    figsize=(14, 10),
    metric: str = "cosine"
):
    """
    Visualize DBSCAN clusters in 2D PCA with sampled text annotations.

    Parameters
    ----------
    prompts : list[str]
        List of text prompts.
    embeddings : array-like, shape (N, D)
        Embedding vectors.
    labels : array-like, shape (N,)
        Cluster labels from DBSCAN (-1 = noise).
    sample_per_cluster : int, default=3
        Number of text prompts to annotate per cluster.
    sample_of_cluster : int, default=5
        Fixed number of clusters to sample.
    random_seed : int, default=42
        Random seed for reproducible sampling.
    figsize : tuple, default=(14, 10)
        Size of the output figure.
    metric : str, default="cosine"
        (Not used directly here but can be useful if extended to t-SNE/UMAP.)
    """
    np.random.seed(random_seed)
    random.seed(random_seed)

    X = np.asarray(embeddings, dtype=float)
    Xn = normalize(X, norm="l2", axis=1)
    X2 = PCA(n_components=2, random_state=random_seed).fit_transform(Xn)

    unique_labels = sorted(set(labels))
    n_clusters = len([l for l in unique_labels if l != -1])

    fig, ax = plt.subplots(figsize=figsize)
    ax.set_title(f"PCA 2D projection of embeddings ({n_clusters} clusters)")
    ax.set_xlabel("PC 1")
    ax.set_ylabel("PC 2")

    # Plot all points lightly
    ax.scatter(X2[:, 0], X2[:, 1], c="lightgray", s=10, alpha=0.4, label="All points")

    # Sample a fixed number of clusters
    sampled_labels = np.random.choice(unique_labels[1:], size=min(sample_of_cluster, n_clusters - 1), replace=False)

    for cid in sampled_labels: 
        idx = np.where(labels == cid)[0]
        if cid == -1:
            # Plot noise points separately
            ax.scatter(X2[idx, 0], X2[idx, 1], s=25, marker="x", c="k", alpha=0.6, label="Noise (-1)")
            continue

        # Cluster points
        ax.scatter(X2[idx, 0], X2[idx, 1], s=22, alpha=0.7, label=f"Cluster {cid}")

        # Randomly sample a few for annotation
        sample_idx = np.random.choice(idx, size=min(sample_per_cluster, len(idx)), replace=False)
        for si in sample_idx:
            text = prompts[si]
            # shorten long prompts
            if len(text) > 60:
                text = text[:57] + "…"
            ax.annotate(text,
                        (X2[si, 0], X2[si, 1]),
                        xytext=(5, 3),
                        textcoords="offset points",
                        fontsize=8,
                        alpha=0.8)

    ax.legend(loc="upper right", fontsize=9, ncol=2)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

    return fig

def make_interactive_cluster_plot(
    prompts,
    embeddings,
    labels,
    *,
    method: str = "pca",
    random_state: int = 42,
    filename: str = "dbscan_clusters_interactive.html",
):
    """
    Create an interactive 2D scatter (hover to see text prompts) as a standalone HTML.

    Parameters
    ----------
    prompts : Sequence[str]
        Text prompts, same order as embeddings.
    embeddings : array-like of shape (N, D)
        Embedding vectors.
    labels : array-like of shape (N,)
        Cluster labels from DBSCAN (-1 = noise).
    method : {"pca","umap"}, default="pca"
        2D projection method for visualization only.
    random_state : int, default=42
        Random seed for reproducibility (affects PCA init a bit, UMAP more).
    filename : str, default="dbscan_clusters_interactive.html"
        Output HTML path (standalone, can open locally in a browser).

    Returns
    -------
    filename : str
        The path to the saved HTML file.
    """

    prompts = list(prompts)
    X = np.asarray(embeddings, dtype=float)
    labels = np.asarray(labels)

    if X.ndim != 2:
        raise ValueError("`embeddings` must be 2D (N, D).")
    if len(prompts) != X.shape[0] or len(labels) != X.shape[0]:
        raise ValueError("Length mismatch among prompts, embeddings, and labels.")

    # Normalize for stable geometry before projection
    Xn = normalize(X, norm="l2", axis=1)

    # --- 2D projection
    method_used = "PCA"
    if method.lower() == "umap":
        try:
            import umap  # type: ignore
            reducer = umap.UMAP(n_components=2, random_state=random_state)
            X2 = reducer.fit_transform(Xn)
            method_used = "UMAP"
        except Exception:
            # Fallback to PCA if UMAP isn't available
            X2 = PCA(n_components=2, random_state=random_state).fit_transform(Xn)
            method_used = "PCA (UMAP unavailable)"
    else:
        X2 = PCA(n_components=2, random_state=random_state).fit_transform(Xn)
        method_used = "PCA"

    # Build frame for plotting
    df = pd.DataFrame({
        "PC1": X2[:, 0],
        "PC2": X2[:, 1],
        "cluster": labels.astype(int),
        "prompt": prompts,
    })

    # Interactive scatter (hover shows prompt + cluster)
    fig = px.scatter(
        df,
        x="PC1",
        y="PC2",
        color="cluster",
        hover_data={"prompt": True, "cluster": True, "PC1": False, "PC2": False},
        title=f"{method_used} 2D projection of embeddings (hover to see prompts)",
    )
    # Slightly smaller points & a bit of transparency to reduce occlusion
    fig.update_traces(marker=dict(size=6, opacity=0.8))

    # Save as a self-contained HTML (loads Plotly from CDN for smaller file)
    plot_offline(fig, filename=filename, auto_open=False, include_plotlyjs="cdn")
    return filename

# ================== Main ==================
def main(args):

    print(f"Generating pet prompts from {len(subjects)} subjects, {len(actions)} actions, {len(scenes)} scenes, {len(objects)} objects.")
    prompts = generate_prompts(subjects, actions, scenes, objects, templates, max_samples=args.max_samples)
    print(f"[INFO] Generated {len(prompts)} initial prompts.")

    # cluster embeddings and prune similar ones
    setup_determinism(args.seed)
    model_name = infer_model_name_from_ckpt(args.model_path)
    model, _, image_processor = open_clip.create_model_and_transforms(
        model_name, pretrained=args.model_path)
    model = model.to(args.device).eval()
    tokenizer = open_clip.get_tokenizer(model_name)
    model.eval()
    model = reparameterize_model(model)
    use_amp = not args.no_amp

    embeddings = embed_promts(
    prompts,
    model=model,
    tokenizer=tokenizer,
    device=args.device,
    context_length=77,          # match your runtime
    use_amp=use_amp,
    max_output=1000            # or set e.g. 1000
    )

    labels, fig = cluster_and_plot_dbscan(
        prompts,
        embeddings,
        eps=0.05,
        min_samples=5,
        metric="cosine"
    )
    
    output_dir = os.path.join(args.output_dir)
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    cluster_plot_path = os.path.join(output_dir, "dbscan_pet_prompts.png")
    fig.savefig(cluster_plot_path, dpi=150)
    print(f"[INFO] Saved DBSCAN clustering plot to {cluster_plot_path}.")

    # save clustered prompts and cluster stats into a json file 
    clustered_prompts = {}
    for lab in set(labels):
        clustered_prompts[str(lab)] = [prompts[i] for i in range(len(prompts)) if labels[i] == lab]
        stats = compute_cluster_stats(embeddings, [str(l) for l in labels])
        stats["eps"] = 0.05
        stats["min_samples"] = 5
        stats["metric"] = "cosine"

        stats_path = os.path.join(args.output_dir, "dbscan_cluster_stats.json")
        with open(stats_path, "w") as f:
            json.dump(stats, f, indent=4)
        prompts_path = os.path.join(args.output_dir, "dbscan_clustered_pet_prompts.json")
        with open(prompts_path, "w") as f:
            json.dump(clustered_prompts, f, indent=4)   
    print(f"[INFO] Saved clustered prompts to {prompts_path}.")

    # summarize and visualize semantics
    plot_clusters_with_text_samples(prompts, embeddings, labels, sample_per_cluster= min(int(len(labels) / 3), 15), figsize=(16, 12))
    sementic_plot_path = os.path.join(output_dir, "dbscan_pet_prompts_with_texts.png")
    plt.savefig(sementic_plot_path, dpi=150)
    print(f"[INFO] Saved semantic cluster plot to {sementic_plot_path}.")

    # interactive plot
    interactive_plot_path = os.path.join(output_dir, "dbscan_pet_prompts_interactive.html")
    make_interactive_cluster_plot(prompts=prompts, embeddings=embeddings, labels=labels,
                                  method=args.projection,
                                  filename=interactive_plot_path)
    print(f"[INFO] Saved interactive cluster plot to {interactive_plot_path}.")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="./checkpoints/mobileclip2_s4.pt")
    parser.add_argument("--output_dir", type=str, default="./experiments/text_embedding_cluster")
    parser.add_argument("--projection", type = str, default = "pca", choices=["pca", "tsne", "umap"])
    parser.add_argument('--max_samples', type=int, default=200000, help='Maximum number of prompts to generate.')
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--no_amp", action="store_true")
    parser.add_argument("--device", type=str, default="cuda:0")
    args = parser.parse_args()

    main(args)