"""
Analyze mobileclip_s1 checkpoint, find all depthwise Conv2d layers,
replace them with normal Conv2d (groups=1) with random weights,
and save the modified model.
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
import torch.nn as nn
from mobileclip import create_model_and_transforms

def main():
    ckpt_path = "checkpoints/mobileclip_s1.pt"

    # Load model (reparameterized for inference)
    print("Loading mobileclip_s1 model...")
    model, _, preprocess = create_model_and_transforms(
        "mobileclip_s1", pretrained=ckpt_path, reparameterize=True
    )

    # Find and replace all depthwise Conv2d layers
    replaced = []

    def replace_dwconv(module, prefix=""):
        for name, child in module.named_children():
            full_name = f"{prefix}.{name}" if prefix else name
            if isinstance(child, nn.Conv2d) and child.groups > 1:
                # This is a depthwise (or grouped) conv
                old_groups = child.groups
                new_conv = nn.Conv2d(
                    in_channels=child.in_channels,
                    out_channels=child.out_channels,
                    kernel_size=child.kernel_size,
                    stride=child.stride,
                    padding=child.padding,
                    dilation=child.dilation,
                    groups=1,  # normal conv
                    bias=child.bias is not None,
                    padding_mode=child.padding_mode,
                )
                # Random init (default kaiming) is fine - user only needs speed test
                setattr(module, name, new_conv)
                replaced.append(
                    f"{full_name}: groups {old_groups} -> 1, "
                    f"in={child.in_channels} out={child.out_channels} "
                    f"k={child.kernel_size} s={child.stride}"
                )
            else:
                replace_dwconv(child, full_name)

    replace_dwconv(model)

    print(f"\nReplaced {len(replaced)} depthwise/grouped Conv2d layers:")
    for r in replaced:
        print(f"  {r}")

    # Save as state_dict (same format as original)
    out_path = "checkpoints/mobileclip_s1_no_dwconv.pt"
    torch.save(model.state_dict(), out_path)
    print(f"\nSaved modified model to {out_path}")
    print(f"File size: {os.path.getsize(out_path) / 1024 / 1024:.1f} MB")

if __name__ == "__main__":
    main()
