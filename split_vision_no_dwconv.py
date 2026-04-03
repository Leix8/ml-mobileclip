"""
Load the mobileclip_s1 vision encoder checkpoint, find all depthwise Conv2d
layers, replace them with normal Conv2d (groups=1) with random weights,
and save the modified vision encoder checkpoint.
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
import torch.nn as nn
from mobileclip import create_model_and_transforms


def replace_dwconv(module, prefix=""):
    """Recursively replace all depthwise/grouped Conv2d with normal Conv2d (groups=1)."""
    replaced = []
    for name, child in module.named_children():
        full_name = f"{prefix}.{name}" if prefix else name
        if isinstance(child, nn.Conv2d) and child.groups > 1:
            old_groups = child.groups
            new_conv = nn.Conv2d(
                in_channels=child.in_channels,
                out_channels=child.out_channels,
                kernel_size=child.kernel_size,
                stride=child.stride,
                padding=child.padding,
                dilation=child.dilation,
                groups=1,
                bias=child.bias is not None,
                padding_mode=child.padding_mode,
            )
            setattr(module, name, new_conv)
            replaced.append(
                f"{full_name}: groups {old_groups} -> 1, "
                f"in={child.in_channels} out={child.out_channels} "
                f"k={child.kernel_size} s={child.stride}"
            )
        else:
            replaced.extend(replace_dwconv(child, full_name))
    return replaced


def main():
    ckpt_path = "checkpoints/mobileclip_s1.pt"

    print("Loading mobileclip_s1 model (reparameterized)...")
    model, _, _ = create_model_and_transforms(
        "mobileclip_s1", pretrained=ckpt_path, reparameterize=True
    )

    vision_encoder = model.image_encoder

    # Replace depthwise convolutions in vision encoder only
    replaced = replace_dwconv(vision_encoder)

    print(f"\nReplaced {len(replaced)} depthwise/grouped Conv2d layers in vision encoder:")
    for r in replaced:
        print(f"  {r}")

    # Extract vision encoder state dict (with image_encoder. prefix to match original format)
    vision_state = {"image_encoder." + k: v for k, v in vision_encoder.state_dict().items()}
    logit_scale = model.logit_scale

    # Save
    out_dir = "checkpoints/s1/vision_tower"
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "mobileclip_s1_vision_no_dwconv.pt")

    torch.save({
        "state_dict": vision_state,
        "logit_scale": logit_scale,
    }, out_path)

    print(f"\nSaved to {out_path}")
    print(f"File size: {os.path.getsize(out_path) / 1024 / 1024:.1f} MB")

    # Summary
    total_params = sum(p.numel() for p in vision_encoder.parameters())
    print(f"Total vision encoder params: {total_params:,}")


if __name__ == "__main__":
    main()
