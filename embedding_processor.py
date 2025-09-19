"""
encode_text(text: str) -> text_embedding: tensor[D]

encode_image(image: tensor[3, H, W]) -> image_embedding: tensor[D]

debias(target_embdding: tensor[D], bias_embeddings: tensor[D]) -> debiased_embedding: tensor[D]

taf(target_embedding: tensor[D], guiding_embedding: tensor[D], beta: float, gamma: float) -> taf_embedding: tensor[D]

cluster(embeddings: tensor[N,D], proj_args) -> plot

### generate ref embeddings and visualize clustering ###
----------------------------
ref_embeddings = [encode_image(ref) for ref in ref_images] -> tensor[R,D]
----------------------------
ref_embeddings_centroid = ref_embeddings.pool() -> tensor[D]
ref_embeddings_debiased = ref_embeddings - ref_centroid -> tensor[R,D]
----------------------------
ref_embeddings_taf = [taf(emb, text_embedding) for emb in ref_embeddings] -> tensor[R,D]
cluster(ref_embeddings_taf)
----------------------------
ref_embeddings_debiased_taf = [taf(emb, text_embedding) for emb in ref_embeddings_debiased] -> tensor[R, D]
cluster(ref_embeddings_debiased_taf) # expect to see best clustering here
----------------------------

### process frames and calculate similarity ###
----------------------------
frames_embeddings = [encode_image(f) for f in temporal_frames] -> tensor[F,D]
frame_embedding = encode_image(current_frame) -> tensor[D]
----------------------------
frames_embeddings_centroid = frames_embeddings.pool() -> tensor[D]
frame_embedding_debiased = frame_embedding - frames_embeddings_centroid -> tensor[D]
----------------------------
cos_sim = frame_embeddings_debiased @ ref_embeddings_debiased_taf -> float

"""

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import umap


# ============ 1. Encode functions ============

def encode_text(text: str, model=None, tokenizer=None, device="cuda") -> torch.Tensor:
    """Encode a text prompt into an embedding [D]."""
    assert model is not None and tokenizer is not None, "Need model + tokenizer"
    with torch.no_grad(), torch.cuda.amp.autocast():
        tokens = tokenizer([text]).to(device)
        emb = model.encode_text(tokens)
        emb = F.normalize(emb, dim=-1)  # [1,D]
    return emb.squeeze(0)              # [D]


def encode_image(image: torch.Tensor, model=None, image_processor=None, device="cuda") -> torch.Tensor:
    """Encode a single image tensor [3,H,W] into embedding [D]."""
    assert model is not None and image_processor is not None, "Need model + image_processor"
    with torch.no_grad(), torch.cuda.amp.autocast():
        x = image_processor(image).unsqueeze(0).to(device)  # [1,3,H,W]
        emb = model.encode_image(x)
        emb = F.normalize(emb, dim=-1)
    return emb.squeeze(0)  # [D]

# ============ 2. Debias ============

def debias(target_embedding: torch.Tensor, bias_embeddings: torch.Tensor) -> torch.Tensor:
    """
    Subtract mean background embedding from the target.
    target_embedding: [D]
    bias_embeddings: [N,D]
    """
    centroid = F.normalize(bias_embeddings.mean(dim=0, keepdim=True), dim=-1)
    debiased = target_embedding - centroid.squeeze(0)
    return F.normalize(debiased, dim=-1)

# ============ 3. TAF Fusion ============

def safe_norm(x: torch.Tensor) -> torch.Tensor:
    return x / (x.norm(dim=-1, keepdim=True) + 1e-6)

def taf(target_embedding: torch.Tensor,
        guiding_embedding: torch.Tensor,
        beta: float = 0.5,
        gamma: float = 0.1) -> torch.Tensor:
    """
    Fuse target embedding with a guiding (e.g. text) embedding.
    """
    dot_it = torch.clamp((guiding_embedding * target_embedding).sum(), -1.0, 1.0)
    i_parallel = dot_it * target_embedding
    i_perp = guiding_embedding - i_parallel
    fused = ((1.0 - beta) * target_embedding +
             beta * safe_norm(i_parallel) +
             gamma * safe_norm(i_perp))
    # fused = (beta * safe_norm(i_parallel) +
    #          gamma * safe_norm(i_perp))
    return F.normalize(fused, dim=-1).squeeze(0)

# ============ 4. Clustering & Visualization ============

def add_thumbnails(ax, X2, paths, zoom=0.6):
    """Overlay thumbnails on a scatter plot."""
    from matplotlib.offsetbox import OffsetImage, AnnotationBbox
    for (x, y, path) in zip(X2[:,0], X2[:,1], paths):
        try:
            im = plt.imread(path)
            imagebox = OffsetImage(im, zoom=zoom)
            ab = AnnotationBbox(imagebox, (x,y), frameon=False)
            ax.add_artist(ab)
        except Exception as e:
            print(f"[WARN] Skipping {path}: {e}")

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
        add_thumbnails(ax, X2, paths, zoom=0.6)

    ax.set_title(title)
    ax.grid(True, alpha=0.25)
    plt.tight_layout()
    plt.savefig(out)
    plt.close()


def cluster(embeddings, paths=None, labels=None, proj_args=None,
            out_dir="cluster_out", out_name="cluster.png", thumbnails=False):
    """
    Reduce embeddings to 2D, save clustering scatter plot.
    
    Args:
        embeddings: torch.Tensor [N,D]
        paths: list of image paths (optional, for thumbnails)
        labels: cluster labels (optional, ints or None)
        proj_args: {"method": "umap"/"pca"/"tsne", "seed": 1234}
        out_dir: directory where output image is saved
        out_name: filename for saved scatter
        thumbnails: overlay thumbnails if True
    """
    os.makedirs(out_dir, exist_ok=True)
    proj_args = proj_args or {}
    method = proj_args.get("method", "umap")
    seed = proj_args.get("seed", 1234)

    X = embeddings.cpu().numpy()

    if method == "pca":
        reducer = PCA(n_components=2, random_state=seed)
        X2 = reducer.fit_transform(X)
    elif method == "tsne":
        reducer = TSNE(n_components=2, random_state=seed, metric="cosine")
        X2 = reducer.fit_transform(X)
    else:  # default UMAP
        reducer = umap.UMAP(n_components=2, random_state=seed, metric="cosine")
        X2 = reducer.fit_transform(X)

    out_path = os.path.join(out_dir, out_name)
    save_scatter(X2, paths or ["" for _ in range(len(X2))],
                 labels=labels, title=f"Cluster ({method.upper()})",
                 out=out_path, thumbnails=thumbnails)

    print(f"[OUT] Saved scatter to {out_path}")
    return X2