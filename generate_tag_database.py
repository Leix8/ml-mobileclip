#!/usr/bin/env python3
"""
Generate a Pet-Related Text Prompt Database
-------------------------------------------
Creates compositional text prompts for zero-shot highlight retrieval tasks,
suitable for CLIP/MobileCLIP2-based embedding and retrieval systems.

Output: pet_prompts_v1.json
"""

import itertools
import random
import json
from datetime import datetime
import argparse  # Add this import
from mobileclip.modules.common.mobileone import reparameterize_model
from model_name_map import MODEL_NAME_MAP, infer_model_name_from_ckpt
import open_clip
import torch
import os
import numpy as np
import heapq

from torch.cuda.amp import autocast
import torch.nn.functional as F
from typing import List, Dict, Any, Optional, Callable

from tqdm import tqdm

# -----------------------------------------------------
# 1. Define Core Vocabularies
# -----------------------------------------------------

# -----------------------------------------------------
# pet scenes
# -----------------------------------------------------

subjects = [
    "a dog", "a cat", "a puppy", "a kitten", "two dogs", "two cats",
    "a corgi", "a husky", "a golden retriever", "a small dog", "a large dog",
    "a small cat", "a pet", "a domestic animal"
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


# -----------------------------------------------------
# daily life scenes (scaled down)
# -----------------------------------------------------
# subjects = [
#     "a person", "a man", "a woman", "a family", "a child", "a group of people",
#     "a couple", "a parent and child", "a group of friends", "a family with kids",
#     "a person walking a dog", "a photographer", "a street performer",
#     "a crowd", "several people", "a young boy", "a girl", "a group of kids",
#     "a senior couple", "a jogging man", "a cyclist"
# ]
# actions = [
#     # Movement
#     "is walking", "is running", "is jogging", "is strolling slowly",
#     "is entering the mall", "is walking along the street", "is crossing the road",
#     # Social
#     "is talking", "is chatting", "is smiling", "is laughing", "is taking selfies",
#     "is waving", "is hugging", "is meeting friends", "is sitting and talking",
#     # Observation / interaction
#     "is watching the performance", "is taking photos", "is filming with a phone",
#     "is interacting with art installation", "is admiring the lights",
#     # Family / kids
#     "is holding a child’s hand", "is playing with kids", "is feeding pigeons",
#     "is shopping", "is eating ice cream", "is buying snacks",
#     # Sport / activity
#     "is playing basketball", "is playing ping pong", "is riding a bicycle",
#     "is stretching", "is exercising", "is playing badminton",
#     "is enjoying outdoor activities"
# ]
# scenes = [
#     # Indoor
#     "inside a shopping mall", "in front of a store", "near an escalator",
#     "in a POP MART area", "inside an anime-themed shop", "by a large art installation",
#     "in a food court", "in a bookstore", "inside a bright modern mall",
#     # Outdoor / public
#     "in the park", "in a community park", "in a plaza", "on the street",
#     "on a pedestrian walkway", "by the fountain", "near the playground",
#     "by the lake", "at the basketball court", "at the ping pong area",
#     "in the square", "under the trees", "by the grass field",
#     "in front of a modern building", "under a glass dome", "in a public garden",
#     # Time / ambience
#     "during the day", "at night", "under city lights", "in a lively evening scene"
# ]
# objects = [
#     # Installations & art
#     "with a cartoon sculpture", "with an art installation", "with a flower display",
#     "with a Totoro sculpture", "with a large anime figure", "with a neon light sign",
#     "with a LED screen", "with interactive exhibits", "with a light tunnel",
#     # Leisure & shopping
#     "with a shopping bag", "with an ice cream", "with a drink", "with a stroller",
#     "with friends", "with family", "with other shoppers",
#     # Sports
#     "with a basketball", "with a ping pong paddle", "with a bicycle", "with a camera tripod",
#     # Environment
#     "surrounded by greenery", "surrounded by lights", "under colorful lights",
#     "with buildings around", "near a flower sculpture", "next to a fountain",
#     "under a canopy", "on the grass", "near the playground",
#     "with reflections on the glass", "in front of tall buildings"
# ]
# -----------------------------------------------------
# 2. Define Compositional Templates
# -----------------------------------------------------

templates = [
    # "{s} {a}",
    # "{s} {a} {sc}",
    # "{s} {a} {o}",
    "{s} {a} {sc} {o}"
]

# -----------------------------------------------------
# 3. Generate Prompts (sampled for diversity)
# -----------------------------------------------------


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
# -----------------------------------------------------
# 3.1 cluster prompt embeddings and prune (optional)
# -----------------------------------------------------


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

@torch.no_grad()
def encode_text(model, tokenizer, text: str, device: str,
                context_length: int, use_amp: bool = True) -> torch.Tensor:
    tokens = tokenizer([text], context_length=context_length).to(device)
    with autocast(enabled=use_amp):
        tfeat = model.encode_text(tokens)
    return F.normalize(tfeat, dim=-1).squeeze(0)

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


# Your encode_text signature:
# def encode_text(model, tokenizer, text: str, device: str,
#                 context_length: int, use_amp: bool = True) -> torch.Tensor

def _l2_normalize(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    n = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.clip(n, eps, None)

class _UnionFind:
    def __init__(self, n: int):
        self.p = list(range(n))
        self.sz = [1] * n
    def find(self, x: int) -> int:
        while self.p[x] != x:
            self.p[x] = self.p[self.p[x]]
            x = self.p[x]
        return x
    def union(self, a: int, b: int):
        ra, rb = self.find(a), self.find(b)
        if ra == rb: return
        if self.sz[ra] < self.sz[rb]:
            ra, rb = rb, ra
        self.p[rb] = ra
        self.sz[ra] += self.sz[rb]

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

def cluster_and_prune_prompts(
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
    """
    Greedy agglomerative clustering by highest cosine similarity pairs until
    the number of clusters <= max_output. No fixed threshold is used.

    - Start with each prompt as its own cluster.
    - Maintain a centroid for each cluster (normalized sum of member embeddings).
    - Use a max-heap of pairwise similarities (cosine on normalized vectors).
    - Iteratively merge the highest-similarity pair whose clusters are still active.
    - Stop when the number of clusters <= max_output.
    - Representative for each cluster = member closest to the cluster centroid.

    Notes:
      * For N <= ~2000, full pairwise is fine (O(N^2) memory). For larger N,
        set `topk_per_row` (e.g., 32–64) to push only per-row top-K candidate pairs.
      * Embeddings are assumed L2-normalized. If not, they are normalized here.
    """
    N = len(prompts)
    if N == 0:
        return [] if not return_info else {"optimized": [], "clusters": [], "reps_idx": []}

    if (max_output is None) or (max_output >= N):
        return prompts if not return_info else {
            "optimized": prompts, "clusters": [[i] for i in range(N)], "reps_idx": list(range(N))
        }

    # 1) Encode prompts to embeddings (N, D), ideally already normalized
    E = _embed_prompts_with_mobileclip2(
        prompts, model, tokenizer, device, context_length, use_amp=use_amp
    ).astype(np.float32)
    # Safety: normalize
    E /= (np.linalg.norm(E, axis=1, keepdims=True) + 1e-12)

    # 2) Initialize clusters: each item is its own cluster
    #    We'll keep: active flags, members list, and centroid sums (unnormalized),
    #    where centroid = normalize(sum_vec).
    members = {i: [i] for i in range(N)}
    sums = {i: E[i].copy() for i in range(N)}  # unnormalized sum of embeddings
    active = {i: True for i in range(N)}
    parent = list(range(N))  # disjoint-set-like "current id" (not strictly union-find, but helpful)

    def find_root(x: int) -> int:
        # 'parent' here only collapses trivial chains from merges
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    # 3) Build a max-heap of pairwise similarities
    heap: list[tuple[float, int, int]] = []  # stores (-sim, i, j)
    def push_pair(i: int, j: int):
        if i == j: 
            return
        # cosine between centroids (use normalized sums)
        ci = sums[i] / (np.linalg.norm(sums[i]) + 1e-12)
        cj = sums[j] / (np.linalg.norm(sums[j]) + 1e-12)
        sim = float(ci @ cj)
        heapq.heappush(heap, (-sim, i, j))

    if topk_per_row and topk_per_row > 0 and N > 2048:
        # Memory-friendly: push only top-K neighbors per row initially
        # (We’ll keep updating similarities against the merged cluster later.)
        for i in tqdm(range(N), disable=not progress, desc="Init topK heap"):
            sims = E @ E[i]  # (N,)
            print(f"probing original sims: mean={sims.mean():.4f}, max={sims.max():.4f}, min={sims.min():.4f}, std={sims.std():.4f}, sample_value ={sims[N//2]:.4f}")
            sims[i] = -1.0
            if topk_per_row < N - 1:
                idxs = np.argpartition(-sims, topk_per_row)[:topk_per_row]
            else:
                idxs = np.arange(N); idxs = idxs[idxs != i]
            for j in idxs:
                if j > i:
                    heapq.heappush(heap, (-float(sims[j]), i, j))
    else:
        # Full O(N^2) initialization
        # Compute in blocks to limit peak memory if needed
        B = 2048
        for i0 in tqdm(range(0, N, B), disable=not progress, desc="Init full heap"):
            i1 = min(N, i0 + B)
            block_vecs = E[i0:i1]        # (B, D)
            S = block_vecs @ E.T         # (B, N)
            for ii in range(i1 - i0):
                i = i0 + ii
                # Only upper triangle to avoid duplicates
                S[ii, :i+1] = -1.0
                js = np.where(S[ii] > 0)[0]  # speed tweak: only push positive sims
                for j in js:
                    heapq.heappush(heap, (-float(S[ii, j]), i, j))

    # 4) Greedy merges until cluster count <= max_output
    cluster_count = N

    def merge(a: int, b: int) -> int:
        """Merge cluster b into a. Returns the surviving root id."""
        # Choose larger cluster as the survivor to keep ids more stable
        if len(members[b]) > len(members[a]):
            a, b = b, a
        # Merge b -> a
        members[a].extend(members[b])
        sums[a] += sums[b]
        active[b] = False
        parent[b] = a
        # Recompute similarities for new 'a' against all active clusters
        for c in list(members.keys()):
            if c == a or not active.get(c, False):
                continue
            push_pair(a, c)
        return a

    # Keep merging best pairs while we have too many clusters
    with tqdm(total=cluster_count, disable=not progress, desc="Merging clusters") as pbar:
        while cluster_count > max_output and heap:
            neg_sim, i, j = heapq.heappop(heap)
            i = find_root(i)
            j = find_root(j)
            if i == j or not active.get(i, False) or not active.get(j, False):
                continue  # stale pair, skip
            # Merge these two clusters
            merge(i, j)
            cluster_count -= 1
            pbar.update(1)  # Update progress bar
    
    # 5) Collect active clusters
    final_clusters: List[List[int]] = [sorted(v) for k, v in members.items() if active.get(k, False)]

    # 6) Representatives: pick item closest to centroid
    reps_idx: List[int] = []
    for k in final_clusters:
        sub = E[k]  # (m, D)
        centroid = sums[find_root(k[0])]
        centroid = centroid / (np.linalg.norm(centroid) + 1e-12)
        sims = sub @ centroid
        reps_idx.append(k[int(np.argmax(sims))])

    reps_idx = sorted(set(reps_idx))
    optimized = [prompts[i] for i in reps_idx]

    # reconstruct the sim matrix of final picked prompts for analysis (optional)
    # calculate the mean, min, max, std statics for the final similarity matrix and probe
    final_embs = E[reps_idx]  # (M, D)
    S_final = final_embs @ final_embs.T  # (M, M)
    print(f"[INFO] Final similarity matrix stats: mean={S_final.mean():.4f}, max={S_final.max():.4f}, min={S_final.min():.4f}, std={S_final.std():.4f}, sample_value ={S_final[len(S_final)//2, len(S_final)//2]:.4f}")

    if return_info:
        return {"optimized": optimized, "clusters": final_clusters, "reps_idx": reps_idx}
    return optimized

# -----------------------------------------------------
# 4. Generate and Save JSON Database
# -----------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description="Generate a Pet-Related Text Prompt Database.")
    parser.add_argument('--max_samples', type=int, default=200000, help='Maximum number of prompts to generate.')
    parser.add_argument('--output_file', type=str, default='scene_tags/pet_scenes_scaled_default.json', help='Output JSON file name.')
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--no_amp", action="store_true")
    parser.add_argument("--model_path", type=str, default="./checkpoints/mobileclip2_s4.pt")
    parser.add_argument("--device", type=str, default="cuda:0")
    args = parser.parse_args()

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
    
    optimized = cluster_and_prune_prompts(
    prompts,
    model=model,
    tokenizer=tokenizer,
    device=args.device,
    context_length=77,          # match your runtime
    use_amp=use_amp,
    max_output=1000            # or set e.g. 1000
    )
    print(f"pruned from {len(prompts)} initial prompts → {len(optimized)} after cluster pruning")

    db = {
        "metadata": {
            "domain": "pet scenes",
            "generator": "pet_scenes_prompt_generator_v1",
            "created_at": datetime.now().isoformat(),
            "prompt_count": len(prompts)
        },
        "prompts": [{"tag": p} for p in optimized]
    }

    out_file = args.output_file  # Use the output file from args
    with open(out_file, "w") as f:
        json.dump(db, f, indent=2, ensure_ascii=False)

    print(f"[INFO] Generated {len(optimized)} prompts.")
    print(f"[INFO] Saved to {out_file}")

# -----------------------------------------------------
# 5. Run
# -----------------------------------------------------

if __name__ == "__main__":
    main()