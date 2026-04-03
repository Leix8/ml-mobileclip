"""
Convert 3 mobileclip_s1 vision encoder checkpoints to TFLite (fp32 + fp16).

Checkpoints:
  1. mobileclip_s1_original_vision_encoder.pt  (original depthwise)
  2. mobileclip_s1_vision_bottleneck.pt        (bottleneck replacement)
  3. mobileclip_s1_vision_no_dwconv.pt         (direct groups=1 replacement)
"""
import sys
import os
import math
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
import torch.nn as nn
import numpy as np
import litert_torch
from mobileclip import create_model_and_transforms


# --------------- BottleneckConv (needed to rebuild bottleneck model) ---------------
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


# --------------- Model builders ---------------

def build_original_vision():
    """Build original reparameterized vision encoder and load weights."""
    model, _, _ = create_model_and_transforms(
        "mobileclip_s1", pretrained="checkpoints/mobileclip_s1.pt", reparameterize=True
    )
    return model.image_encoder


def _replace_dwconv_direct(module):
    """Replace depthwise conv with groups=1 conv (random weights)."""
    for name, child in module.named_children():
        if isinstance(child, nn.Conv2d) and child.groups > 1:
            new_conv = nn.Conv2d(
                in_channels=child.in_channels, out_channels=child.out_channels,
                kernel_size=child.kernel_size, stride=child.stride,
                padding=child.padding, dilation=child.dilation,
                groups=1, bias=child.bias is not None,
                padding_mode=child.padding_mode,
            )
            setattr(module, name, new_conv)
        else:
            _replace_dwconv_direct(child)


def _replace_dwconv_bottleneck(module):
    """Replace depthwise conv with bottleneck structure."""
    for name, child in module.named_children():
        if isinstance(child, nn.Conv2d) and child.groups > 1:
            r = compute_rank(child.in_channels, child.out_channels, child.kernel_size)
            new_module = BottleneckConv(
                in_channels=child.in_channels, out_channels=child.out_channels,
                kernel_size=child.kernel_size, stride=child.stride,
                padding=child.padding, dilation=child.dilation,
                bias=child.bias is not None, padding_mode=child.padding_mode,
                rank=r,
            )
            setattr(module, name, new_module)
        else:
            _replace_dwconv_bottleneck(child)


def build_no_dwconv_vision(ckpt_path):
    """Build vision encoder with depthwise conv replaced by groups=1."""
    model, _, _ = create_model_and_transforms(
        "mobileclip_s1", pretrained="checkpoints/mobileclip_s1.pt", reparameterize=True
    )
    vis = model.image_encoder
    _replace_dwconv_direct(vis)
    # Load saved weights
    ckpt = torch.load(ckpt_path, map_location="cpu")
    state = ckpt["state_dict"]
    # Strip image_encoder. prefix
    state = {k.replace("image_encoder.", ""): v for k, v in state.items()}
    vis.load_state_dict(state)
    return vis


def build_bottleneck_vision(ckpt_path):
    """Build vision encoder with depthwise conv replaced by bottleneck."""
    model, _, _ = create_model_and_transforms(
        "mobileclip_s1", pretrained="checkpoints/mobileclip_s1.pt", reparameterize=True
    )
    vis = model.image_encoder
    _replace_dwconv_bottleneck(vis)
    # Load saved weights
    ckpt = torch.load(ckpt_path, map_location="cpu")
    state = ckpt["state_dict"]
    state = {k.replace("image_encoder.", ""): v for k, v in state.items()}
    vis.load_state_dict(state)
    return vis


# --------------- TFLite conversion ---------------

def quantize_tflite_to_fp16(input_path, output_path):
    """Cast float32 weights to float16 in a TFLite flatbuffer."""
    try:
        from ai_edge_litert import schema_py_generated as schema
    except ImportError:
        from tflite_runtime import schema_py_generated as schema
    import flatbuffers

    with open(input_path, "rb") as f:
        buf = f.read()

    model = schema.ModelT.InitFromPackedBuf(bytearray(buf), 0)

    for subgraph in model.subgraphs:
        for tensor in subgraph.tensors:
            if tensor.type == 0:  # FLOAT32
                buffer = model.buffers[tensor.buffer]
                if buffer.data is not None and len(buffer.data) > 0:
                    fp32_data = np.frombuffer(bytes(buffer.data), dtype=np.float32)
                    fp16_data = fp32_data.astype(np.float16)
                    buffer.data = list(fp16_data.tobytes())
                    tensor.type = 1  # FLOAT16

    builder = flatbuffers.Builder(len(buf) * 2)
    packed = model.Pack(builder)
    builder.Finish(packed)
    out_buf = builder.Output()

    with open(output_path, "wb") as f:
        f.write(bytes(out_buf))


def convert_model(vis_encoder, name, out_dir):
    """Convert a vision encoder to fp32 and fp16 TFLite."""
    vis_encoder.eval()
    sample_inputs = (torch.randn(1, 3, 256, 256),)

    with torch.no_grad():
        out = vis_encoder(*sample_inputs)
    print(f"  PyTorch output: {out.shape}")

    # FP32
    print(f"  Converting {name} -> fp32 tflite...")
    edge_model = litert_torch.convert(vis_encoder, sample_inputs)
    fp32_path = os.path.join(out_dir, f"{name}_fp32.tflite")
    edge_model.export(fp32_path)
    print(f"    Saved: {fp32_path} ({os.path.getsize(fp32_path) / 1024 / 1024:.1f} MB)")

    # FP16
    print(f"  Converting {name} -> fp16 tflite...")
    fp16_path = os.path.join(out_dir, f"{name}_fp16.tflite")
    quantize_tflite_to_fp16(fp32_path, fp16_path)
    print(f"    Saved: {fp16_path} ({os.path.getsize(fp16_path) / 1024 / 1024:.1f} MB)")


def main():
    out_dir = "checkpoints/s1/vision_tower"
    ckpt_dir = out_dir

    # 1. Original
    print("\n[1/3] Original vision encoder (depthwise)")
    vis = build_original_vision()
    convert_model(vis, "mobileclip_s1_vision_original", out_dir)
    del vis

    # 2. No-dwconv (groups=1)
    print("\n[2/3] No-dwconv vision encoder (groups=1)")
    vis = build_no_dwconv_vision(os.path.join(ckpt_dir, "mobileclip_s1_vision_no_dwconv.pt"))
    convert_model(vis, "mobileclip_s1_vision_no_dwconv", out_dir)
    del vis

    # 3. Bottleneck
    print("\n[3/3] Bottleneck vision encoder")
    vis = build_bottleneck_vision(os.path.join(ckpt_dir, "mobileclip_s1_vision_bottleneck.pt"))
    convert_model(vis, "mobileclip_s1_vision_bottleneck", out_dir)
    del vis

    print("\nAll done!")


if __name__ == "__main__":
    main()
