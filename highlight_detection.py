import torch
import mobileclip
import argparse
import os
import re
from tqdm import tqdm
import torch.nn.functional as F

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from pathlib import Path

def natural_key(path: Path):
    """Natural sort: splits 'frame_10.png' into ['frame_', 10, '.png'] so 2 < 10."""
    s = path.name
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r'(\d+)', s)]

def iter_frame_dirs(
    root_dir,
    img_exts={".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp", ".heic", ".HEIC"},
    min_images=2,
    include_root=True,
):
    root = Path(root_dir)
    dirs = [root] if include_root else []
    dirs += [p for p in root.rglob("*") if p.is_dir()]

    seen = set()  # avoid duplicates if symlinks/etc.
    for d in dirs:
        if d in seen:
            continue
        seen.add(d)

        # Only images directly in this directory (do not recurse here)
        imgs = [p for p in d.iterdir()
                if p.is_file()
                and not p.name.startswith(".")
                and p.suffix.lower() in img_exts]
        # print(f"imgs={imgs}")
        if len(imgs) >= min_images:
            imgs.sort(key=natural_key)  # natural numeric order
            # print(d, imgs)
            yield d, imgs

def encode_text(model, tokenizer, tags):
    text_embeddings_dict = {}
    for idx, tag in enumerate(tags):
        input_id = tokenizer(tag).to(args.device)
        with torch.no_grad(), torch.cuda.amp.autocast():
            text_embedding = model.encode_text(input_id)
            text_embedding /= text_embedding.norm(dim=-1, keepdim=True)
        text_embeddings_dict[tag] = text_embedding
    
    return text_embeddings_dict

def encode_vision(model, image_processor, args):
    video_embeddings = {}
    for d, image_dirs in iter_frame_dirs(root_dir = args.frame_dir): # iterating videos from a root dir
        frame_embeddings = {}
        for idx, image_dir in tqdm(enumerate(image_dirs), total = len(image_dirs), desc = f"encoding frames in: {d}"): # iterating frames from one video
            try:
                image = Image.open(image_dir).convert('RGB')
            except Exception as e:
                print(f"❌ Skipping {image_dir}: {e}")
                continue    

            image_tensor = image_processor(image).unsqueeze(0).to(args.device)

            with torch.no_grad(), torch.cuda.amp.autocast():
                image_embedding = model.encode_image(image_tensor)
                image_embedding /= image_embedding.norm(dim=-1, keepdim=True)
            frame_embeddings[image_dir] = image_embedding
        video_embeddings[d] = frame_embeddings
    return video_embeddings


def visualize_tagging_on_video(frame_dirs, tags, frames_tagging_cos, threshold, args, frames_dir, max_tags_to_show=5):
    """
    Visualizes tag similarity scores by appending a bottom panel to each frame and saves as video.
    
    Args:
        frame_dirs (List[Path]): List of N image paths.
        tags (List[str]): List of M tag strings.
        frames_tagging_cos (List[List[float]]): N x M cosine similarity scores.
        threshold (float): Minimum score to display tag.
        args: Argument object with args.output_dir.
        frames_dir: Current directory name for output naming.
        max_tags_to_show (int): Maximum number of tags to display per frame, sorted by score descending.
    """
    print(len(frame_dirs), len(frames_tagging_cos))
    assert len(frame_dirs) == len(frames_tagging_cos), "Frame count mismatch"

    os.makedirs(args.output_dir, exist_ok=True)
    output_path = os.path.join(args.output_dir, f"{Path(frames_dir).stem}.mp4")

    # Get size from one sample frame
    sample_frame = cv2.imread(str(frame_dirs[0]))
    if sample_frame is None:
        raise ValueError(f"Cannot read sample frame: {frame_dirs[0]}")
    h, w, _ = sample_frame.shape

    # Reserve extra space at bottom for tag overlay
    panel_height = 60 + 30 * max_tags_to_show
    output_size = (w, h + panel_height)

    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), 10, output_size)

    for idx, frame_path in enumerate(frame_dirs):
        frame = cv2.imread(str(frame_path))
        if frame is None:
            print(f"❌ Failed to read {frame_path}")
            continue

        tag_scores = frames_tagging_cos[idx]
        filtered = [(tags[i], tag_scores[i]) for i in range(len(tag_scores)) if tag_scores[i] >= threshold]
        top_tags = sorted(filtered, key=lambda x: x[1], reverse=True)[:max_tags_to_show]

        # Create a bottom panel
        panel = np.ones((panel_height, w, 3), dtype=np.uint8) * 255  # white background

        for i, (tag, score) in enumerate(top_tags):
            y = 30 + i * 30
            text = f"{tag}: {score:.2f}"
            cv2.putText(panel, text, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

        # Stack frame and panel vertically
        combined_frame = np.vstack((frame, panel))

        out.write(combined_frame)

    out.release()
    print(f"✅ Saved visualized video: {output_path}")

def load_tags(tags_arg):
    """Load tags from a comma-separated string, a .txt file, or a JSON file.
    Robust to quotes, extra spaces, and structured JSON dicts.
    """
    def clean(tag: str) -> str:
        if not isinstance(tag, str):
            tag = str(tag)
        # strip whitespace and trailing commas/semicolons
        tag = tag.strip().strip(",;")
        # strip surrounding brackets if passed accidentally
        tag = tag.strip("[]{}")
        # remove surrounding quotes repeatedly if present
        while len(tag) > 1 and tag[0] in ("'", '"') and tag[-1] in ("'", '"'):
            tag = tag[1:-1].strip()
        return tag

    tags = []

    if isinstance(tags_arg, (list, tuple)):
        # Already a list of tags
        tags = tags_arg

    elif isinstance(tags_arg, str):
        if os.path.isfile(tags_arg):
            # File path input
            if tags_arg.endswith(".json"):
                with open(tags_arg, "r", encoding="utf-8") as f:
                    data = json.load(f)
                # flatten JSON if it’s a dict of categories
                if isinstance(data, dict):
                    for v in data.values():
                        if isinstance(v, list):
                            tags.extend(v)
                elif isinstance(data, list):
                    tags = data
                else:
                    raise ValueError("Unsupported JSON structure for tags")
            elif tags_arg.endswith(".txt"):
                with open(tags_arg, "r", encoding="utf-8") as f:
                    for line in f:
                        tags.extend(line.split(","))
            else:
                raise ValueError("Unsupported file format: must be .txt or .json")
        else:
            # Treat as raw comma-separated string
            tags = tags_arg.split(",")
    else:
        raise TypeError("tags_arg must be a string, list, or tuple")

    # Final cleaning + deduplication (preserve order)
    cleaned = []
    seen = set()
    for tag in tags:
        ct = clean(tag)
        if ct and ct not in seen:
            cleaned.append(ct)
            seen.add(ct)

    return cleaned

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_path", \
        type = str, \
        required = False, \
        default = "./checkpoints/mobileclip_s0.pt",\
        help = "path to checkpoints"
    )
    parser.add_argument("--output_dir", type = str, default = "./highlight_detection", help = "directory to save the results")
    parser.add_argument("--device", type = str, default = "cuda:0", help = "device to be used")
    parser.add_argument("--video_dir", type=str, default=None, help="location of image file")
    parser.add_argument("--frame_dir", type=str, default=None, help="location of image file")
    parser.add_argument("--tags", type=str, default="./tags_v1_1_enhanced_specified.txt")
    args = parser.parse_args()

    args.model_name = os.path.splitext(os.path.basename(args.model_path))[0]

    # process tags
    tags = load_tags(args.tags)

    model, _, image_processor = mobileclip.create_model_and_transforms(args.model_name, pretrained=args.model_path)
    model = model.to(args.device).eval()
    tokenizer = mobileclip.get_tokenizer(args.model_name)

    # process text 
    text_embeddings = encode_text(model, tokenizer, tags) # dict: {tag : text_embedding}
    text_embeddings_batch = model.encode_text(tokenizer(tags).to(args.device)).half() # text embeddings tensor

    # for k,v in enumerate(text_embeddings):
    #     print(f"[{k}] {v}: {(text_embeddings[v].type(), text_embeddings[v].shape, text_embeddings[v][:, 10:20])}")
    
    # process vision 
    vision_embeddings = encode_vision(model, image_processor, args)

    for frames_dir_idx, (frames_dir, frame_embeddings) in enumerate(vision_embeddings.items()):
        print(f"Processing {frames_dir}")
        
        frame_dirs = []
        frames_tagging_cos = []
        frames_tagging_softmax = []
        for image_dir_idx, (image_dir, image_embedding) in tqdm(enumerate(frame_embeddings.items()), total = len(frame_embeddings), desc = f"Calculating similarity for: {frames_dir}"):
            text_embeddings_batch /= text_embeddings_batch.norm(dim = -1, keepdim = True)
            text_probs = (100.0 * image_embedding @ text_embeddings_batch.T).softmax(dim=-1)
            
            frame_tagging_cos = []
            detection = {}
            for idx, tag in enumerate(tags):
                detection[tag] = text_probs[0, idx].item()

            for text_idx, (text, text_embedding) in enumerate(text_embeddings.items()):
                cos_sim = F.cosine_similarity(image_embedding, text_embedding.expand_as(image_embedding), dim=1)  # shape: [256]
                # print(f"|---->{text}: {cos_sim.max()}")
                frame_tagging_cos.append(cos_sim.max())
            frames_tagging_cos.append(frame_tagging_cos)
            frame_dirs.append(image_dir)
        visualize_tagging_on_video(frame_dirs, tags, frames_tagging_cos, 0.15, args, frames_dir) #frame_dirs is of (#frames), frames_tagging_cos is of (#frames, #tags)





'''
image = preprocess(Image.open("docs/fig_accuracy_latency.png").convert('RGB')).unsqueeze(0)
text = tokenizer(["a diagram", "a dog", "a cat"])

with torch.no_grad(), torch.cuda.amp.autocast():
    image_features = model.encode_image(image)
    text_features = model.encode_text(text)
    image_features /= image_features.norm(dim=-1, keepdim=True)
    text_features /= text_features.norm(dim=-1, keepdim=True)

    text_probs = (100.0 * image_features @ text_features.T).softmax(dim=-1)

print("Label probs:", text_probs)

'''