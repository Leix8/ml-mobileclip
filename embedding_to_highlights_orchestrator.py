#!/usr/bin/env python3
import argparse
import subprocess
import sys
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

def is_video_file(p: Path):
    exts = {".mp4", ".avi", ".mov", ".mkv", ".webm"}
    return p.is_file() and p.suffix.lower() in exts

def is_frame_dir(p: Path):
    if not p.is_dir():
        return False
    # check for at least 5 image files inside
    img_exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
    imgs = [f for f in p.iterdir() if f.suffix.lower() in img_exts]
    return len(imgs) >= 5

def run_one(item: Path, args):
    """
    Run the highlight retrieval script for one video file or frames directory.
    """
    cmd = [
        sys.executable, args.script,             # python interpreter + retrieval script
        "--embeddings", args.embeddings,
        "--output_dir", str(args.output_dir),
        "--fps", str(args.fps),
        "--topk", str(args.topk),
        "--bg_samples", str(args.bg_samples),
        "--model_path", args.model_path,
        "--device", args.device
    ]

    if args.no_amp:
        cmd.append("--no_amp")
    if args.all_in_one:
        cmd.append("--all_in_one")
    if args.nms_window_sec is not None:
        cmd += ["--nms_window_sec", str(args.nms_window_sec)]
    else:
        cmd += ["--nms_window_frames", str(args.nms_window_frames)]

    if is_video_file(item):
        cmd += ["--video", str(item)]
    elif is_frame_dir(item):
        cmd += ["--frames", str(item)]
    else:
        return f"Skipped {item} (not video or frame dir)"

    print(f"[INFO] Running: {' '.join(cmd)}")
    try:
        subprocess.run(cmd, check=True)
        return f"Completed {item}"
    except subprocess.CalledProcessError as e:
        return f"Failed {item}: {e}"

def main():
    parser = argparse.ArgumentParser(description="Orchestrator for batch highlight retrieval")
    parser.add_argument("--root_dir", type=str, required=True,
                        help="Root directory containing videos or frame subdirectories")
    parser.add_argument("--script", type=str, default = "./embedding_to_highlights.py",
                        help="Path to highlight retrieval script (with NMS)")
    parser.add_argument("--embeddings", type=str, required=True,
                        help="Path to embeddings JSON file")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Directory to store outputs")
    parser.add_argument("--fps", type=int, default=5)
    parser.add_argument("--topk", type=int, default=10)
    parser.add_argument("--bg_samples", type=int, default=128)
    parser.add_argument("--model_path", type=str, default="./checkpoints/mobileclip_s0.pt")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--no_amp", action="store_true")
    parser.add_argument("--all_in_one", action="store_true")
    parser.add_argument("--nms_window_frames", type=int, default=15)
    parser.add_argument("--nms_window_sec", type=float, default=None,
                        help="Preferred for video inputs; overrides --nms_window_frames")
    parser.add_argument("--workers", type=int, default=1,
                        help="Number of parallel processes")
    args = parser.parse_args()

    root = Path(args.root_dir)
    items = [p for p in root.iterdir() if is_video_file(p) or is_frame_dir(p)]
    if not items:
        print(f"[WARN] No valid videos or frame dirs found in {root}")
        return

    args.output_dir = Path(args.output_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    print(f"[INFO] Found {len(items)} items to process under {root}")
    if args.workers > 1:
        with ProcessPoolExecutor(max_workers=args.workers) as ex:
            futures = {ex.submit(run_one, item, args): item for item in items}
            for fut in as_completed(futures):
                print(fut.result())
    else:
        for item in items:
            print(run_one(item, args))

if __name__ == "__main__":
    main()