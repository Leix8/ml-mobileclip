#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Visualize & analyze clustering of MobileCLIP image embeddings.
Outputs:
  - scatter_{method}.png  (2D projection with thumbnails)
  - cosine_heatmap.png
  - kmeans_report.json
  - dendrogram.png
Optional:
  - text_projection_{method}.png (color by projection onto a text prompt direction)
"""

import argparse, os, sys, json, math, random
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.offsetbox import OffsetImage, AnnotationBbox

import torch
import torch.nn.functional as F

import open_clip
from mobileclip.modules.common.mobileone import reparameterize_model
from model_name_map import MODEL_NAME_MAP, infer_model_name_from_ckpt

from embedding_processor import *

# ---------- MobileCLIP safe import ----------
def _import_mobileclip():
    try:
        import mobileclip
        return sys.modules["mobileclip"]
    except Exception as e:
        raise ImportError("Could not import 'mobileclip'. Put it on PYTHONPATH.") from e

# ---------- Utils ----------
IMG_EXTS = {".jpg",".jpeg",".png",".bmp",".webp",".tif",".tiff",".heic",".HEIC"}

def list_images(paths: List[str]) -> List[Path]:
    out = []
    for p in paths:
        pp = Path(p)
        if pp.is_dir():
            out.extend([q for q in pp.rglob("*") if q.suffix.lower() in IMG_EXTS])
        elif pp.suffix.lower() in IMG_EXTS:
            out.append(pp)
    out = sorted(list({q.resolve() for q in out}))
    if not out:
        raise RuntimeError("No images found.")
    return out

def set_seed(seed=1234):
    import numpy as _np, random as _rnd, torch as _th
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":16:8")
    _rnd.seed(seed); _np.random.seed(seed)
    _th.manual_seed(seed); _th.cuda.manual_seed_all(seed)
    _th.backends.cudnn.benchmark = False
    _th.backends.cudnn.deterministic = True
    _th.backends.cuda.matmul.allow_tf32 = False
    _th.backends.cudnn.allow_tf32 = False
    _th.use_deterministic_algorithms(True)

def pil_thumb(path: Path, max_w=96) -> np.ndarray:
    im = Image.open(path).convert("RGB")
    w, h = im.size
    if w > max_w:
        im = im.resize((max_w, int(h*max_w/w)), Image.BILINEAR)
    return np.asarray(im)

def add_thumbnails(ax, X2: np.ndarray, img_paths: List[Path], zoom=1.0):
    for (x, y), p in zip(X2, img_paths):
        im = pil_thumb(p, max_w=96)
        oi = OffsetImage(im, zoom=zoom/2.0)  # adjust density
        ab = AnnotationBbox(oi, (x, y), frameon=False, pad=0.0)
        ax.add_artist(ab)

def cosine_matrix(E: np.ndarray) -> np.ndarray:
    # E: [N, D], assumed normalized
    return E @ E.T

def remove_top_pcs(E: np.ndarray, k=1) -> np.ndarray:
    # project out top-k PCs (simple whitening-ish)
    X = E - E.mean(0, keepdims=True)
    try:
        U,S,Vt = np.linalg.svd(X, full_matrices=False)
        P = Vt[:k].T
        Ew = X - X @ P @ P.T
        Ew = Ew / (np.linalg.norm(Ew, axis=1, keepdims=True) + 1e-6)
        return Ew
    except Exception:
        return E

def auto_kmeans(X: np.ndarray, kmin=2, kmax=10, seed=1234) -> Tuple[np.ndarray, int, float]:
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score
    best_k, best_score, best_labels = 2, -1.0, None
    for k in range(kmin, min(kmax, len(X)-1)+1):
        km = KMeans(n_clusters=k, n_init=10, random_state=seed)
        labels = km.fit_predict(X)
        if len(set(labels)) < 2: continue
        sc = silhouette_score(X, labels, metric="cosine")
        if sc > best_score:
            best_k, best_score, best_labels = k, sc, labels
    if best_labels is None:
        best_k, best_score, best_labels = 1, 0.0, np.zeros(len(X), dtype=int)
    return best_labels, best_k, float(best_score)

def run_projection(X: np.ndarray, method: str, seed=1234) -> np.ndarray:
    method = method.lower()
    if method == "pca":
        from sklearn.decomposition import PCA
        Z = PCA(n_components=2, random_state=seed).fit_transform(X)
        return Z
    elif method == "umap":
        try:
            import umap
        except Exception:
            print("[WARN] umap-learn not installed; falling back to t-SNE.")
            method = "tsne"
        else:
            reducer = umap.UMAP(n_components=2, random_state=seed, metric="cosine", min_dist=0.1)
            return reducer.fit_transform(X)
    if method == "tsne":
        from sklearn.manifold import TSNE
        Z = TSNE(n_components=2, random_state=seed, metric="cosine", init="pca", perplexity=min(30, max(5, len(X)//4)))
        return Z.fit_transform(X)
    raise ValueError("Unknown method. Use pca|umap|tsne.")

def save_scatter(X2, paths, labels=None, title="", out="scatter.png", thumbnails=True):
    plt.figure(figsize=(10,7), dpi=120)
    ax = plt.gca()
    if labels is None:
        ax.scatter(X2[:,0], X2[:,1], s=10, alpha=0.6)
    else:
        import matplotlib.cm as cm
        u = np.unique(labels)
        colors = cm.get_cmap('tab10')(np.linspace(0,1,len(u)))
        for c,lab in zip(colors,u):
            m = labels==lab
            ax.scatter(X2[m,0], X2[m,1], s=12, alpha=0.7, label=f"c{lab}")
        ax.legend(loc="best", fontsize=8)
    if thumbnails and len(paths) <= 300:  # avoid overplotting
        add_thumbnails(ax, X2, paths, zoom=1.0)
    ax.set_title(title)
    ax.grid(True, alpha=0.25)
    plt.tight_layout()
    plt.savefig(out)
    plt.close()

def save_scatter_modalities(
    image_embeddings, image_paths,
    text_embeddings=None, text_labels=None,
    method="pca", seed=1234,
    title="", out="scatter.png", thumbnails=True
):
    # ===== 1. Stack all embeddings =====
    all_embeddings = []
    modalities = []

    if image_embeddings is not None and len(image_embeddings) > 0:
        all_embeddings.append(image_embeddings)
        modalities += ["image"] * len(image_embeddings)

    if text_embeddings is not None and len(text_embeddings) > 0:
        all_embeddings.append(text_embeddings)
        modalities += ["text"] * len(text_embeddings)

    all_embeddings = np.vstack(all_embeddings)
    modalities = np.array(modalities)

    # ===== 2. Project jointly =====
    X2 = run_projection(all_embeddings, method=method, seed=seed)

    # ===== 3. Split results =====
    X2_images = X2[modalities == "image"]
    X2_text   = X2[modalities == "text"]

    # ===== 4. Plot =====
    plt.figure(figsize=(10,7), dpi=120)
    ax = plt.gca()

    # Images
    if len(X2_images) > 0:
        ax.scatter(X2_images[:,0], X2_images[:,1], s=10, alpha=0.3, c="blue")
        if thumbnails and len(image_paths) <= 300:
            add_thumbnails(ax, X2_images, image_paths, zoom=1.0)

    # Text
    if len(X2_text) > 0 and text_labels is not None:
        ax.scatter(X2_text[:,0], X2_text[:,1], s=40, alpha=0.8, c="red", marker="x")
        for (x,y), txt in zip(X2_text, text_labels):
            ax.text(x, y, txt, fontsize=9, ha="center", va="center",
                    bbox=dict(facecolor="white", alpha=0.6, edgecolor="none", pad=1))

    ax.set_title(title)
    ax.grid(True, alpha=0.25)
    plt.tight_layout()
    plt.savefig(out)
    plt.close()

def save_heatmap(C, paths, out="cosine_heatmap.png"):
    import seaborn as sns  # if not installed, comment and use plt.imshow
    names = [p.stem[:18] for p in paths]
    plt.figure(figsize=(8,7), dpi=120)
    sns.heatmap(C, vmin=-1, vmax=1, cmap="viridis", xticklabels=names, yticklabels=names)
    plt.title("Cosine Similarity")
    plt.tight_layout()
    plt.savefig(out)
    plt.close()

def save_dendrogram(C, paths, out="dendrogram.png"):
    # hierarchical clustering on cosine distance
    from scipy.cluster.hierarchy import linkage, dendrogram
    from scipy.spatial.distance import squareform
    D = 1 - np.clip(C, -1, 1)
    # small jitter to avoid singular ties
    D = D + np.eye(D.shape[0]) * 1e-6
    Y = squareform(D, checks=False)
    Z = linkage(Y, method="average")
    plt.figure(figsize=(12, 4 + 0.08*len(paths)), dpi=120)
    dendrogram(Z, labels=[p.stem[:30] for p in paths], leaf_rotation=90)
    plt.title("Hierarchical (average-link) on cosine distance")
    plt.tight_layout()
    plt.savefig(out)
    plt.close()

def load_text_dirs(mobileclip, model, tokenizer, prompts: List[str], device: torch.device) -> np.ndarray:
    toks = tokenizer(prompts).to(device)
    with torch.no_grad():
        te = model.encode_text(toks)
        te = F.normalize(te.float(), dim=-1)
    return te.cpu().numpy()  # [P, D]

def safe_norm(x: torch.Tensor) -> torch.Tensor:
    n = x.norm(p=2)
    return x / n if n > 0 else x

def list_all_images(root_dir: str) -> List[Path]:
    root = Path(root_dir)
    if not root.exists():
        raise FileNotFoundError(f"Directory not found: {root_dir}")
    imgs = [p for p in root.rglob("*") if p.suffix.lower() in IMG_EXTS]
    if not imgs:
        raise RuntimeError(f"No images found in {root_dir}")
    return sorted(imgs, key=lambda p: p.name.lower())

def taf_fuse_multi_refs(text_emb: torch.Tensor, ref_embs: torch.Tensor,
                        beta: float = 0.4, gamma: float = 0.1) -> torch.Tensor:
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
    return F.normalize(torch.stack(fused, dim=0), dim=-1)

# ---------- Main ----------
def main():
    ap = argparse.ArgumentParser("MobileCLIP embedding visualization")
    ap.add_argument("--scene_json", type=str, required=True, help="directory containing images (will recurse)")
    ap.add_argument("--model_path", type=str, default="./checkpoints/mobileclip2_b.pt")
    ap.add_argument("--device", type=str, default="cuda:0")
    ap.add_argument("--outdir", type=str, default="./experiements/visual_embedding_cluster")
    ap.add_argument("--proj", type=str, default="umap", choices=["pca","umap","tsne"])
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--remove_top_pcs", type=int, default=0, help="remove top-K PCs from image embeddings")
    ap.add_argument("--debias_bg", action="store_true", help="subtract mean of random subset (background centroid)")
    ap.add_argument("--text_prompts", type=str, nargs="*", default=[], help="optional text prompts to project onto")
    ap.add_argument("--taf", action="store_true", help="optional to run cluster with taf")
    ap.add_argument("--auto_kmax", type=int, default=10, help="max clusters to scan for KMeans")
    args = ap.parse_args()

    set_seed(args.seed)

    with open(args.scene_json, "r") as f:
        scenes_data = json.load(f)
    for key, tag_list in scenes_data.items():
        groups_raw = {}
        if not isinstance(tag_list, list):
            print(f"[WARN] Expected list under key '{key}'", file=sys.stderr)
            continue
        for item in tag_list:
            groups_raw[item["tag"]] = list_all_images(item["ref_dir"])

    # ---- MobileCLIP ----
    model_name = infer_model_name_from_ckpt(args.model_path)
    model_kwargs = {}
    if not (model_name.endswith("S3") or model_name.endswith("S4") or model_name.endswith("L-14")):
        model_kwargs = {"image_mean": (0, 0, 0), "image_std": (1, 1, 1)}
    model, _, image_processor = open_clip.create_model_and_transforms(model_name, pretrained=args.model_path, **model_kwargs)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    model = model.to(device).eval()
    model = reparameterize_model(model)
    tokenizer = open_clip.get_tokenizer(model_name)

    outdir = Path(f"{args.outdir}/{model_name}"); outdir.mkdir(parents=True, exist_ok=True)

    groups_embedding = {}
    # ---- Encode data ----
    for tag, paths in groups_raw.items():
        ims = [Image.open(p).convert("RGB") for p in paths]
        with torch.no_grad(),  torch.cuda.amp.autocast():
            #images 
            embs = []
            for i in range(0, len(ims), 32):
                batch = ims[i:i+32]
                x = torch.stack([image_processor(im) for im in batch], dim=0).to(device)
                fe = model.encode_image(x)
                fe = F.normalize(fe, dim=-1)
                embs.append(fe.cpu())
            image_embeddings = torch.cat(embs, dim=0).numpy()  # [N, D], L2-normalized

            #text
            toks = tokenizer(tag).to(device)
            te = model.encode_text(toks)
            te = F.normalize(te.float(), dim=-1)
            text_embedding = te.cpu().numpy()  # [P, D]
            
            groups_embedding[tag] = {"image_embeddings": image_embeddings, "text_embedding": text_embedding}

    # ---- Optional group debias / whitening ----
    if args.debias_bg:
        all_embeddings_bg = []
        paths_all = []

        for tag, embeddings in groups_embedding.items():
            image_embeddings = embeddings["image_embeddings"]

            # sample subset for background mean
            idx = np.random.choice(len(image_embeddings),
                                size=min(128, len(image_embeddings)),
                                replace=False)
            bg = image_embeddings[idx].mean(0, keepdims=True)

            # subtract background + normalize
            image_embeddings_bg = image_embeddings - bg
            image_embeddings_bg = image_embeddings_bg / (
                np.linalg.norm(image_embeddings_bg, axis=1, keepdims=True) + 1e-6
            )

            # collect debiased embeddings
            all_embeddings_bg.append(image_embeddings_bg)

            # collect corresponding image paths
            paths_all.extend(groups_raw[tag])

        # 🔹 Concatenate all groups into one array [Total, D]
        all_embeddings_bg = np.concatenate(all_embeddings_bg, axis=0)

        # 🔹 Run projection ONCE on all debiased embeddings
        X2_bg_all = run_projection(all_embeddings_bg, args.proj, seed=args.seed)
        print(f"X2_bg_all: {X2_bg_all.shape}")

        # Save scatter plot
        save_scatter(
            X2_bg_all, paths_all, labels=None,
            title=f"image embeddings (group bg debiased) ({args.proj.upper()})",
            out=str(outdir / f"scatter_{args.proj}_group_bg_debiased.png"),
            thumbnails=True
        )
        print(f"debiased image embedding cluster scatter has been saved to {outdir}")

    # ---- Optional text aligned fusion ----
    if args.taf:
        all_fused = []
        paths_all = []

        for tag, embeddings in groups_embedding.items():
            image_embeddings = embeddings["image_embeddings"]   # [N,D]
            text_embedding   = embeddings["text_embedding"]     # [D]

            fused_group = []
            for image_embedding in image_embeddings:
                image_embedding_taf = taf(
                    torch.tensor(image_embedding, dtype=torch.float32),
                    torch.tensor(text_embedding, dtype=torch.float32)
                )
                # ensure 1D shape [D]
                fused_group.append(image_embedding_taf.squeeze().cpu().numpy())

            fused_group = np.stack(fused_group, axis=0)  # [N,D]
            all_fused.append(fused_group)

            # collect corresponding paths
            paths_all.extend(groups_raw[tag])

        # 🔹 Combine all groups into one [Total, D]
        all_fused = np.concatenate(all_fused, axis=0)

        # 🔹 Run projection ONCE across all fused embeddings
        X2_taf_proj = run_projection(all_fused, args.proj, seed=args.seed)
        print(f"X2_taf_proj: {X2_taf_proj.shape}")

        # Save scatter
        save_scatter(
            X2_taf_proj, paths_all, labels=None,
            title=f"image embeddings (group taf) ({args.proj.upper()})",
            out=str(outdir / f"scatter_{args.proj}_group_taf.png"),
            thumbnails=True
        )
        print(f"taf image embedding cluster scatter has been saved to {outdir}")

    # fall back to visualize global info, regardless of group
    E = []
    text_embeddings = []
    text_labels = []
    for tag, embeddings in groups_embedding.items():
        E.append(embeddings["image_embeddings"])
        text_embeddings.append(embeddings["text_embedding"])
        text_labels.append(tag)
    E = np.concatenate(E, axis = 0)
    text_embeddings = np.concatenate(text_embeddings, axis = 0)
    paths = paths_all

    if args.remove_top_pcs > 0:
        E = remove_top_pcs(E, k=args.remove_top_pcs)

    # ---- Cosine heatmap ----
    C = cosine_matrix(E)
    try:
        import seaborn  # noqa
        save_heatmap(C, paths, out=str(outdir/"cosine_heatmap.png"))
    except Exception:
        # fallback without seaborn
        plt.figure(figsize=(8,7), dpi=120)
        plt.imshow(np.clip(C, -1, 1), vmin=-1, vmax=1, cmap="viridis")
        plt.colorbar()
        plt.title("Cosine Similarity")
        plt.tight_layout()
        plt.savefig(outdir/"cosine_heatmap.png")
        plt.close()

    # ---- 2D projection ----
    save_scatter_modalities(
        E, paths,
        text_embeddings=text_embeddings, text_labels=text_labels,
        method="pca", seed=1234,
        title=f"multi-modality embeddings ({args.proj.upper()})", 
        out=str(outdir/f"scatter_{args.proj}_multimodal.png"),
        thumbnails=True
    )

    # ----- other stats ---------
    X2 = run_projection(E, args.proj, seed=args.seed)
    # save_scatter(X2, paths, labels=None,
    #              title=f"MobileCLIP image embeddings ({args.proj.upper()})",
    #              out=str(outdir/f"scatter_{args.proj}.png"),
    #              thumbnails=True)
    

    # ---- KMeans auto-k + report ----
    try:
        from sklearn.preprocessing import StandardScaler
        X_for_k = StandardScaler(with_mean=True, with_std=True).fit_transform(E)  # cosine ~ angular; scaling okay
    except Exception:
        X_for_k = E
    labels, k_opt, sil = auto_kmeans(X_for_k, kmin=2, kmax=args.auto_kmax, seed=args.seed)
    save_scatter(X2, paths, labels=labels,
                 title=f"KMeans (k={k_opt}, silhouette={sil:.3f}) on {args.proj.upper()}",
                 out=str(outdir/f"kmeans_{args.proj}.png"),
                 thumbnails=False)
    with open(outdir/"kmeans_report.json","w") as f:
        json.dump({"k": int(k_opt), "silhouette_cosine": float(sil),
                   "cluster_counts": {int(c): int((labels==c).sum()) for c in np.unique(labels)}}, f, indent=2)

    # ---- Hierarchical dendrogram ----
    try:
        save_dendrogram(C, paths, out=str(outdir/"dendrogram.png"))
    except Exception as e:
        print("[WARN] dendrogram failed:", e)

    # ---- Optional: project onto text prompts (semantic axes) ----
    if args.text_prompts:
        T = load_text_dirs(open_clip, model, tokenizer, args.text_prompts, device)  # [P, D], L2-normalized
        # pick first prompt or compute direction between two prompts
        if len(T) == 1:
            w = T[0]
        else:
            # direction between first two prompts
            w = T[0] - T[1]
            w = w / (np.linalg.norm(w) + 1e-6)
        scores = (E @ w)  # cosine projection
        # color by scores
        plt.figure(figsize=(10,7), dpi=120)
        sc = plt.scatter(X2[:,0], X2[:,1], c=scores, cmap="coolwarm", s=18)
        plt.colorbar(sc, label=f"projection on '{args.text_prompts[0]}'" if len(T)==1 else f"direction '{args.text_prompts[0]}' – '{args.text_prompts[1]}'")
        plt.title(f"Text projection on {args.proj.upper()}")
        plt.grid(True, alpha=0.25); plt.tight_layout()
        plt.savefig(outdir/f"text_projection_{args.proj}.png"); plt.close()

    print(f"[OK] Wrote outputs to: {outdir}")

if __name__ == "__main__":
    main()