"""
Replace depthwise Conv2d layers in mobileclip_s1 with bottleneck structure:
  1x1 Conv (C -> r) -> KxK Conv (r -> r, groups=1) -> 1x1 Conv (r -> C)

This keeps parameter count close to the original depthwise conv while
using only normal (groups=1) convolutions. Useful for speed benchmarking.

Original depthwise params: C * K * K
Bottleneck params:         C_in*r + r*r*K*K + r*C_out  (+ bias terms)

We choose r so that bottleneck params ≈ original depthwise params:
  r = max(1, round( C*K*K / (C_in + C_out + K*K) ))
"""
import sys
import os
import math
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
import torch.nn as nn
from mobileclip import create_model_and_transforms


class BottleneckConv(nn.Module):
    """Drop-in replacement for a depthwise/grouped Conv2d using bottleneck factorization."""

    def __init__(self, in_channels, out_channels, kernel_size, stride, padding,
                 dilation, bias, padding_mode, rank):
        super().__init__()
        self.conv = nn.Sequential(
            # 1x1 pointwise: reduce channels
            nn.Conv2d(in_channels, rank, kernel_size=1, bias=False),
            # KxK spatial conv on reduced channels (normal groups=1)
            nn.Conv2d(rank, rank, kernel_size=kernel_size, stride=stride,
                      padding=padding, dilation=dilation, groups=1,
                      bias=False, padding_mode=padding_mode),
            # 1x1 pointwise: expand channels back
            nn.Conv2d(rank, out_channels, kernel_size=1, bias=bias),
        )

    def forward(self, x):
        return self.conv(x)


def compute_rank(in_channels, out_channels, kernel_size):
    """Compute bottleneck rank so that param count ≈ original depthwise param count."""
    if isinstance(kernel_size, tuple):
        k_h, k_w = kernel_size
    else:
        k_h = k_w = kernel_size
    kk = k_h * k_w

    # Original depthwise params (ignoring bias): max(C_in, C_out) * K * K
    dw_params = max(in_channels, out_channels) * kk

    # Bottleneck params: C_in*r + r*r*K*K + r*C_out
    # Solve: r^2*KK + r*(C_in+C_out) - dw_params = 0  (quadratic in r)
    a = kk
    b = in_channels + out_channels
    c = -dw_params
    r = (-b + math.sqrt(b * b - 4 * a * c)) / (2 * a)
    r = max(1, round(r))
    return r


def main():
    ckpt_path = "checkpoints/mobileclip_s1.pt"

    print("Loading mobileclip_s1 model...")
    model, _, preprocess = create_model_and_transforms(
        "mobileclip_s1", pretrained=ckpt_path, reparameterize=True
    )

    # Count original params
    orig_params = sum(p.numel() for p in model.parameters())

    replaced = []
    total_dw_params = 0
    total_bn_params = 0

    def replace_dwconv(module, prefix=""):
        nonlocal total_dw_params, total_bn_params
        for name, child in module.named_children():
            full_name = f"{prefix}.{name}" if prefix else name
            if isinstance(child, nn.Conv2d) and child.groups > 1:
                old_groups = child.groups
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
                total_dw_params += old_params
                total_bn_params += new_params

                setattr(module, name, new_module)
                replaced.append(
                    f"{full_name}: groups {old_groups}->1, "
                    f"C_in={child.in_channels} C_out={child.out_channels} "
                    f"k={child.kernel_size} s={child.stride} "
                    f"rank={r}, params {old_params}->{new_params}"
                )
            else:
                replace_dwconv(child, full_name)

    replace_dwconv(model)

    new_total_params = sum(p.numel() for p in model.parameters())

    print(f"\nReplaced {len(replaced)} depthwise/grouped Conv2d layers:")
    for r in replaced:
        print(f"  {r}")

    print(f"\n--- Parameter Summary ---")
    print(f"Original total params:     {orig_params:>12,}")
    print(f"New total params:          {new_total_params:>12,}")
    print(f"Depthwise params removed:  {total_dw_params:>12,}")
    print(f"Bottleneck params added:   {total_bn_params:>12,}")
    print(f"Param change:              {new_total_params - orig_params:>+12,} "
          f"({(new_total_params / orig_params - 1) * 100:+.1f}%)")

    out_path = "checkpoints/mobileclip_s1_bottleneck.pt"
    torch.save(model.state_dict(), out_path)
    print(f"\nSaved to {out_path}")
    print(f"File size: {os.path.getsize(out_path) / 1024 / 1024:.1f} MB")


if __name__ == "__main__":
    main()
