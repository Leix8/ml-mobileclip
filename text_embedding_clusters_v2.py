#!/usr/bin/env python3
from __future__ import annotations
import os
import json
import torch
from torch.cuda.amp import autocast
import time
from datetime import datetime

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
import numpy as np
from sklearn.preprocessing import normalize
from prompt_components import prompt_components

def estimate_intra_inter_ratio(
    X,
    labels,
    max_intra_pairs=200_000,
    max_inter_pairs=200_000,
    seed=42,
):
    """
    Efficiently estimate mean intra- and inter-class similarity
    using random pairs, without forming full N x N matrix.

    X: (N, D) embeddings
    labels: (N,) int/str labels
    """

    rng = np.random.default_rng(seed)

    # 1) Normalize to unit vectors (cosine sim = dot)
    X = normalize(X.astype(np.float32), axis=1)
    labels = np.asarray(labels)
    N = X.shape[0]

    # ---------- Intra-class ----------
    intra_sims = []

    for lab in np.unique(labels):
        idx = np.where(labels == lab)[0]
        m = len(idx)
        if m < 2:
            continue

        # total possible pairs = m*(m-1)/2
        num_pairs = min(max_intra_pairs, m * (m - 1) // 2)

        # sample with replacement (good enough statistically, fast)
        i = rng.choice(idx, size=num_pairs, replace=True)
        j = rng.choice(idx, size=num_pairs, replace=True)

        # drop i == j
        mask = i != j
        i, j = i[mask], j[mask]
        if i.size == 0:
            continue

        sims = np.sum(X[i] * X[j], axis=1)
        intra_sims.append(sims)

    intra_mean = np.concatenate(intra_sims).mean() if intra_sims else 0.0

    # ---------- Inter-class ----------
    # oversample then filter to ensure enough cross-label pairs
    inter_i = rng.integers(0, N, size=max_inter_pairs * 2)
    inter_j = rng.integers(0, N, size=max_inter_pairs * 2)

    mask = labels[inter_i] != labels[inter_j]
    inter_i, inter_j = inter_i[mask], inter_j[mask]

    if inter_i.size == 0:
        inter_mean = 0.0
    else:
        inter_i = inter_i[:max_inter_pairs]
        inter_j = inter_j[:max_inter_pairs]
        sims_inter = np.sum(X[inter_i] * X[inter_j], axis=1)
        inter_mean = sims_inter.mean()

    ratio = intra_mean / (inter_mean + 1e-6)
    return intra_mean, inter_mean, ratio

def compute_cluster_stats(X: np.ndarray, labels: List[str]):
    # add stats for clusters counts, average cluster size, intra/inter-cluster similarity, noise ratio, etc.
    labels_set = set(labels)
    num_cluster = len(labels_set) - (1 if "-1" in labels_set else 0)
    average_cluster_size = len(labels) / len(labels_set) if labels_set else 0
    num_noise = labels.count("-1")
    noise_ratio = labels.count("-1") / len(labels) if len(labels) > 0 else 0

    X = normalize(X)

    # sims = X @ X.T
    # labels = np.array(labels)
    # intra, inter = [], []
    # for i in range(len(labels)):
    #     for j in range(i+1, len(labels)):
    #         if labels[i] == labels[j]:
    #             intra.append(sims[i,j])
    #         else:
    #             inter.append(sims[i,j])
    # intra_mean = np.mean(intra) if intra else 0
    # inter_mean = np.mean(inter) if inter else 0
    # ratio = intra_mean / (inter_mean + 1e-6)
    
    intra_mean, inter_mean, ratio = estimate_intra_inter_ratio(X, labels)

    return {"intra": float(intra_mean), "inter": float(inter_mean), "ratio": float(ratio), "cluster_count": num_cluster, "average_cluster_size": float(average_cluster_size), "nosie_cnt": num_noise, "noise_ratio": float(noise_ratio)}

def generate_prompts(subjects, actions, scenarios, objects, templates,
                     max_samples=200000, seed=42):
    print("[INFO] Generating prompts form {} subjects, {} actions, {} scenarios, {} objects.".format(
        len(subjects), len(actions), len(scenarios), len(objects)
    ))
    random.seed(seed)
    prompts = set()

    combos = list(itertools.product(subjects, actions, scenarios, objects))
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

# add profiling to this function
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

    start = time.perf_counter()
    # DBSCAN
    db = DBSCAN(eps=eps, min_samples=min_samples, metric=metric, n_jobs=-1)
    labels = db.fit_predict(Xn)
    end = time.perf_counter()
    print(f"[PROFILING] DBSCAN clustering completed in {end - start:.2f} seconds. Found {len(set(labels)) - (1 if -1 in labels else 0)} clusters.")

    # 2D projection (PCA just for viz)
    start = time.perf_counter()
    if Xn.shape[1] == 1:
        Xn = np.hstack([Xn, np.zeros((Xn.shape[0], 1))])
    X2 = PCA(n_components=2, random_state=0).fit_transform(Xn)
    enmd = time.perf_counter()
    print(f"[PROFILING] PCA 2D projection completed in {enmd - start:.2f} seconds.")

    # Plot
    # Use constrained layout to help with legend placement
    fig, ax = plt.subplots(figsize=(8, 10), layout="constrained")

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
    # load prompt components
    scene = prompt_components[args.scene]
    subjects = scene["subjects"]
    actions = scene["actions"]
    scenarios = scene["scenarios"]
    objects = scene["objects"]
    templates = prompt_components["templates"]

    print(f"Generating pet prompts from {len(subjects)} subjects, {len(actions)} actions, {len(scenarios)} scenarios, {len(objects)} objects.")
    prompts = generate_prompts(subjects, actions, scenarios, objects, templates, max_samples=args.max_samples)
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

    start = time.perf_counter()
    embeddings = embed_promts(
    prompts,
    model=model,
    tokenizer=tokenizer,
    device=args.device,
    context_length=77,          # match your runtime
    use_amp=use_amp,
    max_output=1000            # or set e.g. 1000
    )
    end = time.perf_counter()
    print(f"[PROFILING] Text embedding completed in {end - start:.2f} seconds.")
    print(f"[INFO] Obtained embeddings of shape {embeddings.shape}.")

    start = time.perf_counter()
    labels, fig = cluster_and_plot_dbscan(
        prompts,
        embeddings,
        eps=args.eps,
        min_samples=args.min_points,
        metric=args.metric,
    )
    end = time.perf_counter()
    print(f"[PROFILING] DBSCAN clustering and plotting completed in {end - start:.2f} seconds.")
    print(f"[INFO] Completed DBSCAN clustering with eps={args.eps}, min_samples={args.min_points}.")

    # add eps and min_points to folder name if needed
    output_dir_name = f"{args.scene}_dbscan_eps{args.eps}_minpts{args.min_points}"
    output_dir = os.path.join(args.output_dir, output_dir_name)
    print(f"[INFO] Saving results to {output_dir}")
    if not os.path.exists(output_dir):
        try:
            os.makedirs(output_dir, exist_ok=True)
            print(f"[INFO] Created output directory at {output_dir}")
        except Exception as e:
            print(f"[ERROR] Failed to create output directory at {output_dir}: {e}")
            return

    cluster_plot_path = os.path.join(output_dir, f"{args.scene}_dbscan_full_cluster_layout.png")
    fig.savefig(cluster_plot_path, dpi=150)
    print(f"[INFO] Saved DBSCAN clustering plot to {cluster_plot_path}.")

    # save clustered prompts and cluster stats into a json file 
    start = time.perf_counter()
    clustered_prompts = {}
    from collections import defaultdict

    clustered_prompts = defaultdict(list)
    for i, lab in enumerate(labels):
        clustered_prompts[str(lab)].append(prompts[i])
    
    print(f"[INFO] Prepared clustered prompts for {len(clustered_prompts)} clusters.")
    stats = compute_cluster_stats(embeddings, [str(l) for l in labels])
    
    stats["eps"] = args.eps
    stats["min_samples"] = args.min_points
    stats["metric"] = args.metric
    
    stats["num_subjects"] = len(subjects)
    stats["num_actions"] = len(actions)
    stats["num_scenarios"] = len(scenarios)
    stats["num_objects"] = len(objects)
    stats["num_prompts"] = len(prompts)
    
    print(f"[INFO] prepared to dump cluster stats: {stats}")
    stats_path = os.path.join(output_dir, f"{args.scene}_dbscan_cluster_stats.json")
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=4)

    print(f"[INFO] prepare to dump clustered prompts.")
    prompts_path = os.path.join(output_dir, f"{args.scene}_dbscan_full_cluster_prompts.json")
    with open(prompts_path, "w") as f:
        json.dump(clustered_prompts, f, indent=4)   
    end = time.perf_counter()
    print(f"[PROFILING] Saving clustered prompts and stats completed in {end - start:.2f} seconds.")
    print(f"[INFO] Saved clustered prompts to {prompts_path}.")

    # summarize and visualize semantics
    plot_clusters_with_text_samples(prompts, embeddings, labels, sample_per_cluster= min(int(len(labels) / 3), 15), figsize=(16, 12))
    sementic_plot_path = os.path.join(output_dir, f"{args.scene}_dbscan_cluster_with_text_sample.png")
    plt.savefig(sementic_plot_path, dpi=150)
    print(f"[INFO] Saved semantic cluster plot to {sementic_plot_path}.")

    # interactive plot
    interactive_plot_path = os.path.join(output_dir, f"{args.scene}_dbscan_prompts_cluster_interactive.html")
    make_interactive_cluster_plot(prompts=prompts, embeddings=embeddings, labels=labels,
                                  method=args.projection,
                                  filename=interactive_plot_path)
    print(f"[INFO] Saved interactive cluster plot to {interactive_plot_path}.")

    db_full = {
        "metadata": {
            "scene": args.scene,
            "generator": f"{args.scene}_full",
            "created_at": datetime.now().isoformat(),
            "prompt_count": len(prompts)
        },
        "prompts": [{"tag": p} for p in prompts],
    }

    out_file = os.path.join(output_dir, f"{args.scene}_dbscan_full.json")
    with open(out_file, "w") as f:
        json.dump(db_full, f, indent=2, ensure_ascii=False)

    pruned_prompts = []
    # make a small set of pruned prompts by taking one from each cluster
    unique_labels = set(labels)
    for lab in unique_labels:
        idx = np.where(labels == lab)[0]
        if len(idx) > 0:
            pruned_prompts.append(prompts[idx[0]])
    db_pruned = {
        "metadata": {
            "scene": args.scene,
            "generator": f"{args.scene}_dbscan_cluster_pruned",
            "created_at": datetime.now().isoformat(),
            "eps": args.eps,
            "min_samples": args.min_points,
            "metric": args.metric,
            "prompt_count": len(pruned_prompts)
        },
        "prompts": [{"tag": p} for p in pruned_prompts],
    }

    out_file = os.path.join(output_dir, f"{args.scene}_dbscan_cluster_pruned.json")
    with open(out_file, "w") as f:
        json.dump(db_pruned, f, indent=2, ensure_ascii=False)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="./checkpoints/mobileclip2_s4.pt")
    parser.add_argument("--output_dir", type=str, default="./experiments/text_embedding_cluster")
    parser.add_argument("--projection", type = str, default = "pca", choices=["pca", "tsne", "umap"])
    parser.add_argument('--max_samples', type=int, default=400000, help='Maximum number of prompts to generate.')
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--no_amp", action="store_true")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--eps", type=float, default=0.05)
    parser.add_argument("--min_points", type=int, default=2)
    parser.add_argument("--metric", type=str, default="cosine")
    parser.add_argument("--scene", type=str, default="pet_scene", choices=["test_scene", "pet_scene", "family_scene", "daily_life_scene", "sport_and_workout_scene", "fishing_scene", "basketball_scene"])
    args = parser.parse_args()

    main(args)