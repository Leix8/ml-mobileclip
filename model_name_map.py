# model_name_map.py
MODEL_NAME_MAP = {
    # MobileCLIP1
    "mobileclip-s0": "mobileclip_s0",
    "mobileclip-s1": "MobileCLIP-S1",
    "mobileclip-s2": "MobileCLIP-S2",
    "mobileclip-b": "MobileCLIP-B",
    "mobileclip-blt": "MobileCLIP-B-LT",
    "mobileclip-s3": "MobileCLIP-S3",
    "mobileclip-l14": "MobileCLIP-L-14",
    "mobileclip-s4": "MobileCLIP-S4",

    # MobileCLIP2
    "mobileclip2-s0": "MobileCLIP2-S0",
    "mobileclip2-s2": "MobileCLIP2-S2",
    "mobileclip2-b": "MobileCLIP2-B",
    "mobileclip2-s3": "MobileCLIP2-S3",
    "mobileclip2-l14": "MobileCLIP2-L-14",
    "mobileclip2-s4": "MobileCLIP2-S4",
}

import os

def infer_model_name_from_ckpt(ckpt_path: str) -> str:
    """
    Infer MobileCLIP model_name given a checkpoint filename.
    E.g. './checkpoints/mobileclip2-b.pt' -> 'MobileCLIP2-B'
    """
    base = os.path.splitext(os.path.basename(ckpt_path))[0]
    key = base.lower().replace("_", "-")  # normalize
    if key in MODEL_NAME_MAP:
        return MODEL_NAME_MAP[key]
    raise ValueError(f"Unknown checkpoint-to-model mapping for '{base} -> {key}'")