"""
Load mobileclip_s1, replace depthwise Conv2d in vision encoder with bottleneck:
  1x1 Conv (C -> r) -> KxK Conv (r -> r, groups=1) -> 1x1 Conv (r -> C)
Rank r is chosen so bottleneck params ≈ original depthwise params.
Save the vision encoder checkpoint to checkpoints/s1/vision_tower/.
"""
import sys
import os
import math
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
import torch.nn as nn
from mobileclip import create_model_and_transforms


class BottleneckConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding,
                 dilation, bias, padding_mode, rank):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, rank, kernel_size=1, bias=False),
            nn.Conv2d(rank, rank, kernel_size=kernel_size, stride=stride,
                      padding=padding, dilation=dilation, groups=1,
                      bias=False, padding_mode=padding_mode),
            nn.Conv2d(rank, out_channels, kernel_size=1, bias=bias),
        )

    def forward(self, x):
        return self.conv(x)


def compute_rank(in_channels, out_channels, kernel_size):
    if isinstance(kernel_size, tuple):
        k_h, k_w = kernel_size
    else:
        k_h = k_w = kernel_size
    kk = k_h * k_w
    dw_params = max(in_channels, out_channels) * kk
    a = kk
    b = in_channels + out_channels
    c = -dw_params
    r = (-b + math.sqrt(b * b - 4 * a * c)) / (2 * a)
    return max(1, round(r))


def replace_dwconv(module, prefix=""):
    replaced = []
    total_dw, total_bn = 0, 0
    for name, child in module.named_children():
        full_name = f"{prefix}.{name}" if prefix else name
        if isinstance(child, nn.Conv2d) and child.groups > 1:
            old_params = sum(p.numel() for p in child.parameters())
            r = compute_rank(child.in_channels, child.out_channels, child.kernel_size)
            new_module = BottleneckConv(
                in_channels=child.in_channels,
                out_channels=child.out_channels,
                kernel_size=child.kernel_size,
                stride=child.stride,
                padding=child.padding,
                dilation=child.dilation,
                bias=child.bias is not None,
                padding_mode=child.padding_mode,
                rank=r,
            )
            new_params = sum(p.numel() for p in new_module.parameters())
            setattr(module, name, new_module)
            total_dw += old_params
            total_bn += new_params
            replaced.append(
                f"{full_name}: groups {child.groups}->1, "
                f"C_in={child.in_channels} C_out={child.out_channels} "
                f"k={child.kernel_size} rank={r}, params {old_params}->{new_params}"
            )
        else:
            sub_replaced, sub_dw, sub_bn = replace_dwconv(child, full_name)
            replaced.extend(sub_replaced)
            total_dw += sub_dw
            total_bn += sub_bn
    return replaced, total_dw, total_bn


def main():
    print("Loading mobileclip_s1 model (reparameterized)...")
    model, _, _ = create_model_and_transforms(
        "mobileclip_s1", pretrained="checkpoints/mobileclip_s1.pt", reparameterize=True
    )

    vision = model.image_encoder
    orig_params = sum(p.numel() for p in vision.parameters())

    replaced, total_dw, total_bn = replace_dwconv(vision)

    new_params = sum(p.numel() for p in vision.parameters())

    print(f"\nReplaced {len(replaced)} depthwise Conv2d -> bottleneck:")
    for r in replaced:
        print(f"  {r}")

    print(f"\n--- Vision Encoder Parameter Summary ---")
    print(f"Original params:          {orig_params:>12,}")
    print(f"New params:               {new_params:>12,}")
    print(f"Depthwise params removed: {total_dw:>12,}")
    print(f"Bottleneck params added:  {total_bn:>12,}")
    print(f"Param change:             {new_params - orig_params:>+12,} "
          f"({(new_params / orig_params - 1) * 100:+.1f}%)")

    vision_state = {"image_encoder." + k: v for k, v in vision.state_dict().items()}

    out_dir = "checkpoints/s1/vision_tower"
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "mobileclip_s1_vision_bottleneck.pt")

    torch.save({
        "state_dict": vision_state,
        "logit_scale": model.logit_scale,
    }, out_path)

    print(f"\nSaved to {out_path}")
    print(f"File size: {os.path.getsize(out_path) / 1024 / 1024:.1f} MB")


if __name__ == "__main__":
    main()
