"""
Convert mobileclip_s1 text encoder to TFLite format (fp32 and fp16)
using litert_torch.
"""
import sys
import os
import tempfile
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
import torch.nn as nn
import numpy as np
import litert_torch
from mobileclip import create_model_and_transforms


class TextEncoderWrapper(nn.Module):
    """Wrapper to make the text encoder export-friendly.

    Replaces argmax-based EOT selection (unsupported in TFLite) with
    manual reimplementation using only supported ops.
    """
    def __init__(self, text_encoder):
        super().__init__()
        self.enc = text_encoder

    def forward(self, text_tokens: torch.Tensor) -> torch.Tensor:
        # Embedding + positional encoding
        token_emb = self.enc.forward_embedding(text_tokens)

        # Causal attention mask
        bsz, seq_len = text_tokens.shape
        attn_mask = torch.zeros(seq_len, seq_len, dtype=token_emb.dtype, device=token_emb.device)
        attn_mask = attn_mask.masked_fill(
            torch.triu(torch.ones(seq_len, seq_len, dtype=torch.bool, device=token_emb.device), diagonal=1),
            float("-inf"),
        )
        attn_mask = attn_mask.unsqueeze(0).expand(bsz, -1, -1)

        # Transformer layers
        for layer in self.enc.transformer:
            token_emb = layer(token_emb, key_padding_mask=None, attn_mask=attn_mask)

        token_emb = self.enc.final_layer_norm(token_emb)

        # EOT token selection: replace argmax with cast+sum approach
        nonzero_mask = (text_tokens != 0).to(torch.int32)
        lengths = nonzero_mask.sum(dim=-1)  # [bsz]
        eot_indices = lengths - 1  # [bsz]

        # Gather EOT embeddings: [bsz, hidden_dim]
        eot_emb = token_emb[torch.arange(bsz, device=token_emb.device), eot_indices]

        # Project
        eot_emb = eot_emb @ self.enc.projection_layer
        return eot_emb


def main():
    ckpt_path = "checkpoints/mobileclip_s1.pt"
    out_dir = "checkpoints/s1/text_tower"
    os.makedirs(out_dir, exist_ok=True)

    print("Loading mobileclip_s1 model...")
    model, _, _ = create_model_and_transforms(
        "mobileclip_s1", pretrained=ckpt_path, reparameterize=True
    )

    text_enc = TextEncoderWrapper(model.text_encoder)
    text_enc.eval()

    # Sample input: batch=1, context_length=77, token ids
    sample_inputs = (torch.randint(0, 49408, (1, 77)),)

    # Verify forward works
    with torch.no_grad():
        out = text_enc(*sample_inputs)
    print(f"PyTorch output shape: {out.shape}, dtype: {out.dtype}")

    # === FP32 ===
    print("\nConverting to TFLite (fp32)...")
    edge_model = litert_torch.convert(text_enc, sample_inputs)
    fp32_path = os.path.join(out_dir, "mobileclip_s1_text_encoder_fp32.tflite")
    edge_model.export(fp32_path)
    print(f"  Saved: {fp32_path} ({os.path.getsize(fp32_path) / 1024 / 1024:.1f} MB)")

    # === FP16 via TFLite post-training float16 quantization ===
    print("\nConverting to TFLite (fp16)...")
    import tensorflow as tf

    # Read the fp32 tflite and apply float16 quantization
    # We need to go through SavedModel. Export edge model to saved model first.
    with tempfile.TemporaryDirectory() as tmpdir:
        saved_model_dir = os.path.join(tmpdir, "saved_model")
        # Export the torch model via tf saved model
        # Actually, the simplest approach: take the fp32 tflite flatbuffer
        # and requantize weights to fp16 using TFLite converter
        converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir) if False else None

    # Alternative: directly quantize the fp32 tflite weights to fp16
    # by reading the flatbuffer and casting float32 buffers to float16
    fp16_path = os.path.join(out_dir, "mobileclip_s1_text_encoder_fp16.tflite")
    quantize_tflite_to_fp16(fp32_path, fp16_path)
    print(f"  Saved: {fp16_path} ({os.path.getsize(fp16_path) / 1024 / 1024:.1f} MB)")

    print("\nDone!")


def quantize_tflite_to_fp16(input_path, output_path):
    """Quantize float32 weights in a TFLite flatbuffer to float16.

    Reads the flatbuffer using the TFLite schema, finds all float32 tensor
    buffers, and casts them to float16. This halves the model size.
    """
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
            # TensorType.FLOAT32 = 0, FLOAT16 = 1
            if tensor.type == 0:  # FLOAT32
                buffer = model.buffers[tensor.buffer]
                if buffer.data is not None and len(buffer.data) > 0:
                    fp32_data = np.frombuffer(bytes(buffer.data), dtype=np.float32)
                    fp16_data = fp32_data.astype(np.float16)
                    buffer.data = list(fp16_data.tobytes())
                    tensor.type = 1  # FLOAT16

    # Serialize back
    builder = flatbuffers.Builder(len(buf) * 2)
    packed = model.Pack(builder)
    builder.Finish(packed)
    out_buf = builder.Output()

    with open(output_path, "wb") as f:
        f.write(bytes(out_buf))


if __name__ == "__main__":
    main()
