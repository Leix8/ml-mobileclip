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


def visualize_tagging_on_video(frame_dirs, tags, frames_tagging_cos, threshold, args, frames_dir):
    """
    Visualizes tag similarity scores on video frames and saves as a video.
    Args:
        frame_dirs (List[Path]): List of N image paths.
        tags (List[str]): List of M tag strings.
        frames_tagging_cos (List[List[float]]): N x M cosine similarity scores.
        threshold (float): Minimum score to display tag.
        args: Argument object with args.output_dir.
        frames_dir: Current directory name for output naming.
    """
    print(len(frame_dirs), len(frames_tagging_cos))
    assert len(frame_dirs) == len(frames_tagging_cos), "Frame count mismatch"

    os.makedirs(args.output_dir, exist_ok=True)
    output_path = os.path.join(args.output_dir, f"{Path(frames_dir).stem}.mp4")

    # Read one frame to get size
    sample_frame = cv2.imread(str(frame_dirs[0]))
    height, width, _ = sample_frame.shape
    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), 10, (width, height))

    for idx, frame_path in enumerate(frame_dirs):
        frame = cv2.imread(str(frame_path))
        if frame is None:
            print(f"❌ Failed to read {frame_path}")
            continue

        tag_scores = frames_tagging_cos[idx]
        tag_texts = []

        for i, score in enumerate(tag_scores):
            if score >= threshold:
                tag_texts.append(f"{tags[i]}: {score:.2f}")

        # Draw tags
        y0 = 30
        dy = 30
        for i, tag_text in enumerate(tag_texts):
            y = y0 + i * dy
            cv2.putText(frame, tag_text, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        out.write(frame)

    out.release()
    print(f"✅ Saved visualized video: {output_path}")

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
    args = parser.parse_args()

    args.model_name = os.path.splitext(os.path.basename(args.model_path))[0]

    # process tags
    tags = ["volcano",
      "sea view",
      "beach",
      "forest",
      "grassland",
      "lake",
      "river",
      "waterfall",
      "canyon",
      "mountain",
      "sunrise",
      "sunset",
      "starry sky",
      "Milky Way",
      "aurora",
      "desert",
      "farmland",
      "meadow",
      "plateau",
      "snow-capped mountain",
      "glacier",
      "cave",
      "field",
      "riverside",
      "sailing",
      "aviation",
      "island",
      "terraced fields",
      "moon",
      "rock formation",
      "diving",
      "rafting",
      "swimming",
      "skiing",
      "ice cave",
      "bonfire",
      "hot air balloon",
      "cable car",
      "train",
      "amusement park",
      "bridge",
      "cityscape", 

      "sea view",
      "beach",
      "coral reef",
      "sailing",
      "surfing",
      "waves",
      "diving",
      "snorkeling",

      "volcano",
      "forest",
      "grassland",
      "lake",
      "mountain",
      "sunrise",
      "sunset",
      "glacier",
      "aurora",
      "waterfall",
      "hot spring",
      "rafting",
      "desert",
      "starry sky",
      "moon",
      "snow-capped mountain",
      "cable car",
      "hot air balloon",
      "bonfire",
      "skiing",
      "camping",
      "riverside",
      "meadow", 

      "kissing",
      "flying kiss",
      "eating",
      "drinking",
      "blowing air",
      "singing",
      "smiling",
      "drinking coffee",
      "pointing",
      "nodding",
      "shaking head",
      "patting head",
      "covering face",
      "cute blink",
      "adorable head tilt",

      "hugging",
      "clapping",
      "waving",
      "throwing objects",
      "catching objects",
      "height comparison",
      "cheering",
      "raising cups",
      "handshake",
      "high-five",
      "patting",
      "holding",
      "hugging child",
      "lifting child",
      "carrying",
      "spreading arms",
      
      "jumping",
      "running",
      "crawling",
      "kneeling on one knee",

      "running toward someone",
      "hugging",
      "playing together",
      "carrying child",
      "eating",
      "smiling",
      "hugging child",
      "pushing stroller",
      "blowing bubbles",
      "child crying",
      "child walking",
      
      "rolling",
      "eating",
      "running around",
      "cuddling",
      "sleeping",
      "playful rolling",

      "city walk",
      "neon lights",
      "bar",
      "aquarium",
      "square",
      "amusement park",
      "church",
      "alley",
      "pet",
      "park",
      "sports",
      "swimming",
      "water park",
      "zoo",
      "theater",
      "exhibition",
      "gym",
      "flower market",
      "cat cafe",
      "shopping",
      "train station",
      "bus",
      "airport",
      "scenic viewpoint",
      "family trip",

      "dishes",
      "desserts",
      "drinks",

      "main course",
        "baked goods",
        "sweets",
        "snacks",
        "beverages", 

    "food gathering",
    "group photo"
      ]

    model, _, image_processor = mobileclip.create_model_and_transforms(args.model_name, pretrained=args.model_path)
    model = model.to(args.device).eval()
    tokenizer = mobileclip.get_tokenizer(args.model_name)

    # process text 
    text_embeddings = encode_text(model, tokenizer, tags) # dict: {tag : text_embedding}
    text_embeddings_batch = model.encode_text(tokenizer(tags).to(args.device)).half() # text embeddings tensor

    for k,v in enumerate(text_embeddings):
        print(f"[{k}] {v}: {(text_embeddings[v].type(), text_embeddings[v].shape, text_embeddings[v][:, 10:20])}")
    
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

            print(f"| {image_dir}: {detection}")

            for text_idx, (text, text_embedding) in enumerate(text_embeddings.items()):
                cos_sim = F.cosine_similarity(image_embedding, text_embedding.expand_as(image_embedding), dim=1)  # shape: [256]
                print(f"|---->{text}: {cos_sim.max()}")
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