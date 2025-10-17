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

# -----------------------------------------------------
# 1. Define Core Vocabularies
# -----------------------------------------------------

# -----------------------------------------------------
# pet scenes
# -----------------------------------------------------
'''
subjects = [
    "a dog", "a cat", "a puppy", "a kitten", "two dogs", "two cats",
    "a corgi", "a husky", "a golden retriever", "a small dog", "a large dog",
    "a small cat", "a pet", "a domestic animal"
]

actions = [
    # Motion
    "is running", "is jumping", "is walking", "is chasing", "is rolling",
    # Interaction
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
'''

# -----------------------------------------------------
# daily life scenes (scaled down)
# -----------------------------------------------------
subjects = [
    "a person", "a man", "a woman", "a family", "a child", "a group of people",
    "a couple", "a parent and child", "a group of friends", "a family with kids",
    "a person walking a dog", "a photographer", "a street performer",
    "a crowd", "several people", "a young boy", "a girl", "a group of kids",
    "a senior couple", "a jogging man", "a cyclist"
]
actions = [
    # Movement
    "is walking", "is running", "is jogging", "is strolling slowly",
    "is entering the mall", "is walking along the street", "is crossing the road",
    # Social
    "is talking", "is chatting", "is smiling", "is laughing", "is taking selfies",
    "is waving", "is hugging", "is meeting friends", "is sitting and talking",
    # Observation / interaction
    "is watching the performance", "is taking photos", "is filming with a phone",
    "is interacting with art installation", "is admiring the lights",
    # Family / kids
    "is holding a child’s hand", "is playing with kids", "is feeding pigeons",
    "is shopping", "is eating ice cream", "is buying snacks",
    # Sport / activity
    "is playing basketball", "is playing ping pong", "is riding a bicycle",
    "is stretching", "is exercising", "is playing badminton",
    "is enjoying outdoor activities"
]
scenes = [
    # Indoor
    "inside a shopping mall", "in front of a store", "near an escalator",
    "in a POP MART area", "inside an anime-themed shop", "by a large art installation",
    "in a food court", "in a bookstore", "inside a bright modern mall",
    # Outdoor / public
    "in the park", "in a community park", "in a plaza", "on the street",
    "on a pedestrian walkway", "by the fountain", "near the playground",
    "by the lake", "at the basketball court", "at the ping pong area",
    "in the square", "under the trees", "by the grass field",
    "in front of a modern building", "under a glass dome", "in a public garden",
    # Time / ambience
    "during the day", "at night", "under city lights", "in a lively evening scene"
]
objects = [
    # Installations & art
    "with a cartoon sculpture", "with an art installation", "with a flower display",
    "with a Totoro sculpture", "with a large anime figure", "with a neon light sign",
    "with a LED screen", "with interactive exhibits", "with a light tunnel",
    # Leisure & shopping
    "with a shopping bag", "with an ice cream", "with a drink", "with a stroller",
    "with friends", "with family", "with other shoppers",
    # Sports
    "with a basketball", "with a ping pong paddle", "with a bicycle", "with a camera tripod",
    # Environment
    "surrounded by greenery", "surrounded by lights", "under colorful lights",
    "with buildings around", "near a flower sculpture", "next to a fountain",
    "under a canopy", "on the grass", "near the playground",
    "with reflections on the glass", "in front of tall buildings"
]
# -----------------------------------------------------
# 2. Define Compositional Templates
# -----------------------------------------------------

templates = [
    "{s} {a}",
    "{s} {a} {sc}",
    "{s} {a} {o}",
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

    for (s, a, sc, o) in combos:
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
# 4. Generate and Save JSON Database
# -----------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description="Generate a Pet-Related Text Prompt Database.")
    parser.add_argument('--max_samples', type=int, default=200000, help='Maximum number of prompts to generate.')
    parser.add_argument('--output_file', type=str, default='scene_tags/pet_scenes_scaled_v1.json', help='Output JSON file name.')
    args = parser.parse_args()

    print(f"Generating pet prompts from {len(subjects)} subjects, {len(actions)} actions, {len(scenes)} scenes, {len(objects)} objects.")
    prompts = generate_prompts(subjects, actions, scenes, objects, templates, max_samples=args.max_samples)

    db = {
        "metadata": {
            "domain": "daily life scenes",
            "generator": "daily_life_prompt_generator_v1",
            "created_at": datetime.now().isoformat(),
            "prompt_count": len(prompts)
        },
        "prompts": [{"tag": p} for p in prompts]
    }

    out_file = args.output_file  # Use the output file from args
    with open(out_file, "w") as f:
        json.dump(db, f, indent=2, ensure_ascii=False)

    print(f"[INFO] Generated {len(prompts)} prompts.")
    print(f"[INFO] Saved to {out_file}")

# -----------------------------------------------------
# 5. Run
# -----------------------------------------------------

if __name__ == "__main__":
    main()