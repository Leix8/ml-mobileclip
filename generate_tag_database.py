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
    # "a small cat", "a pet", "a domestic animal"
]

actions = [
    # Motion
    "is running", "is jumping", "is walking", "is chasing", "is rolling",
    # # Interaction
    "is playing", "is fetching", "is tugging a toy", "is biting", "is scratching",
    # "is licking its paw", "is sniffing something", "is shaking its body",
    # # Emotion / Expression
    # "is barking", "is meowing", "is wagging its tail", "is stretching",
    # "is yawning", "is curious", "is watching something",
    # # Static / Rest
    # "is sleeping", "is lying down", "is sitting", "is resting", "is staying still"
]

scenes = [
    "on the grass", "in the park", "on the beach", "in the living room",
    "in the bedroom", "in the kitchen", "in the backyard", "in the garden",
    # "on the sofa", "on the bed", "by the window", "on the floor",
    # "in the yard", "at home", "outdoors", "under the table"
]

objects = [
    "with a ball", "with a frisbee", "with a toy", "with a stick", "with a bone",
    "with a rope", "with a plush toy", "with a pillow", "with food", "with a bowl",
    # "chasing another pet", "playing with a person", "looking at the camera",
    # "being brushed", "taking a bath", "wearing a collar", "next to its owner"
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
    similarity_threshold: float = 0.88,
    max_output: Optional[int] = None,
    return_info: bool = False,
) -> List[str] | Dict[str, Any]:
    """
    Cluster prompts by cosine similarity (same text geometry as your downstream MobileCLIP2),
    pick one representative per cluster (closest to centroid), optionally cap the output.

    similarity_threshold: merge prompts if cosine >= threshold (0.85–0.92 typical).
    max_output: if set, keep representatives from the largest clusters first.
    """
    N = len(prompts)
    if N == 0:
        return [] if not return_info else {"optimized": [], "clusters": [], "reps_idx": []}

    # 1) Embed with your MobileCLIP2 text encoder
    E = _embed_prompts_with_mobileclip2(
        prompts, model, tokenizer, device, context_length,
        use_amp=use_amp
    )  # (N, D), L2-normalized
    print(f"[INFO] Prompt embeddings shape: {E.shape}")

    # 2) Cosine similarity graph + union-find clustering
    #    (cosine on normalized vectors is just dot product)
    uf = _UnionFind(N)
    # Compute in blocks to keep memory modest if N is large
    block = 1024
    for i0 in tqdm(range(0, N, block), desc="Processing blocks to optimize prompts"):
        i1 = min(N, i0 + block)
        Ei = E[i0:i1]                # (Bi, D)
        # Full sim against all j>i0 to avoid double union operations
        S = Ei @ E.T                 # (Bi, N) cosine similarities
        print(f"probing similarity values in block {i0}:{i1}, min={S.min():.4f}, max={S.max():.4f}, mean={S.mean():.4f}, std={S.std():.4f}, sample vals={S.ravel()[::len(S.ravel())//10]}")
        for ii in range(i1 - i0):
            i = i0 + ii
            # only consider j > i to avoid duplicate unions
            sims = S[ii, i+1:]
            js = np.where(sims >= similarity_threshold)[0]
            for offset in js:
                j = i + 1 + int(offset)
                uf.union(i, j)
    print(f"check union-find parents: {[uf.find(i) for i in range(N)]}")

    # 3) Collect clusters
    root_to_members: Dict[int, List[int]] = {}
    for i in range(N):
        r = uf.find(i)
        root_to_members.setdefault(r, []).append(i)
    clusters = list(root_to_members.values())

    # 4) Pick representatives: closest to centroid (max cosine to centroid)
    reps_idx = []
    print(f"check size of clusters: {[len(m) for m in clusters]}")
    for idxs in clusters:
        sub = E[idxs]                       # (k, D)
        centroid = sub.mean(axis=0, keepdims=True)
        centroid = centroid / (np.linalg.norm(centroid) + 1e-12)
        sims = (sub @ centroid.T).ravel()   # cosine to centroid
        best_local = idxs[int(np.argmax(sims))]
        reps_idx.append(best_local)

    # 5) Optional cap: keep reps from largest clusters first
    if max_output is not None and len(reps_idx) > max_output:
        sizes = [len(m) for m in clusters]
        order = np.argsort(sizes)[::-1]
        reps_idx = [reps_idx[i] for i in order[:max_output]]

    reps_idx = sorted(set(reps_idx))
    optimized = [prompts[i] for i in reps_idx]

    if return_info:
        return {"optimized": optimized, "clusters": clusters, "reps_idx": reps_idx}
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
    similarity_threshold=0.95,  # tune 0.85–0.92
    max_output=10000            # or set e.g. 1000
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