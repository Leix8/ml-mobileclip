#!/usr/bin/env python3
import os
import requests
import json
import time
from pathlib import Path
from urllib.parse import quote

#!/usr/bin/env python3
import os
import json
from icrawler.builtin import GoogleImageCrawler

def download_images_from_prompt(prompt: str, save_dir: str, max_images: int = 30):
    os.makedirs(save_dir, exist_ok=True)
    print(f"[INFO] Downloading '{prompt}' -> {save_dir}")

    crawler = GoogleImageCrawler(storage={"root_dir": save_dir})
    crawler.crawl(
        keyword=prompt,
        max_num=max_images,
        min_size=(200, 200),   # filter out tiny icons
        file_idx_offset=0
    )
    print(f"[DONE] {prompt} : images saved in {save_dir}")


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene_json", type=str, default="./scene_tags/pet_scenes.json")
    parser.add_argument("--base_dir", type=str, default="./ref_images")
    args = parser.parse_args()
    
    with open(args.scene_json, "r") as f:
        data = json.load(f)

    for entry in data["pet_scenes"]:
        tag = entry["tag"]
        ref_dir = entry["ref_dir"]
        save_dir = os.path.join(args.base_dir, tag)
        download_images_from_prompt(tag, save_dir, max_images=50)

if __name__ == "__main__":
    # Example usage
    main()