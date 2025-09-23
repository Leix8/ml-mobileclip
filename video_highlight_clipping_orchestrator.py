#!/usr/bin/env python3
import argparse
import subprocess
import sys
from pathlib import Path

def main():
    # Step 1: Only define *new* args that orchestrator needs
    ap = argparse.ArgumentParser("Batch orchestrator for video highlight clipping")
    ap.add_argument("--root_dir", type=str, required=True,
                    help="Root directory to search for mp4 videos")
    ap.add_argument("--script", type=str, default="video_highlight_clipping.py",
                    help="Path to the actual clipping script")
    # Parse known args for orchestrator, leave others untouched
    args, unknown = ap.parse_known_args()

    root_dir = Path(args.root_dir)
    videos = list(root_dir.rglob("*.mp4"))
    if not videos:
        print(f"[WARN] No .mp4 files found under {root_dir}")
        return
    print(f"[INFO] Found {len(videos)} videos under {root_dir}")

    # Step 2: Pass through all other args directly to the working script
    for vid in videos:
        print(f"[INFO] Processing {vid}")
        cmd = [
            sys.executable, args.script,
            "--video", str(vid),
        ] + unknown  # reuse whatever args user provided
        print("[CMD]", " ".join(cmd))

        ret = subprocess.call(cmd)
        if ret != 0:
            print(f"[ERROR] Failed processing {vid}, exit code {ret}")

if __name__ == "__main__":
    main()