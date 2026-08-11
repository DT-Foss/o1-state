#!/usr/bin/env python3
"""
Extract Layer 10+18 weights from Qwen3-1.7B for FOSS-KI pipeline.
==================================================================
Extracts Q/K/V projections, norms, MLP down_proj from deep layers
where actual specialization happens (not shallow layers 0-2).

Output: data/qwen3_deep_layers.npz
"""

import os
import numpy as np

base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
data_dir = os.path.join(base, 'data')

print("Loading Qwen3-1.7B via MLX...")
import mlx.core as mx
from mlx_lm import load
model, tokenizer = load("Qwen/Qwen3-1.7B")


def to_np(w):
    """Convert MLX weight to numpy float32 (handles bf16)."""
    return np.array(mx.array(w, dtype=mx.float32), copy=False)


# Target layers: 10 and 18 (where specialization happens)
LAYERS = [10, 18]
arrays = {}

for layer_idx in LAYERS:
    print(f"\nExtracting layer {layer_idx}...")
    layer = model.model.layers[layer_idx]
    attn = layer.self_attn
    mlp = layer.mlp

    # Attention projections
    arrays[f'q_proj_{layer_idx}'] = to_np(attn.q_proj.weight)
    arrays[f'k_proj_{layer_idx}'] = to_np(attn.k_proj.weight)
    arrays[f'v_proj_{layer_idx}'] = to_np(attn.v_proj.weight)
    arrays[f'o_proj_{layer_idx}'] = to_np(attn.o_proj.weight)

    # QK norms (Qwen3 feature)
    arrays[f'q_norm_{layer_idx}'] = to_np(attn.q_norm.weight)
    arrays[f'k_norm_{layer_idx}'] = to_np(attn.k_norm.weight)

    # Input layernorm
    arrays[f'input_norm_{layer_idx}'] = to_np(layer.input_layernorm.weight)
    arrays[f'post_attention_layernorm_{layer_idx}'] = to_np(layer.post_attention_layernorm.weight)

    # MLP projections (down_proj = fact memory per Geva et al. 2021)
    arrays[f'mlp_gate_proj_{layer_idx}'] = to_np(mlp.gate_proj.weight)
    arrays[f'mlp_up_proj_{layer_idx}'] = to_np(mlp.up_proj.weight)
    arrays[f'mlp_down_proj_{layer_idx}'] = to_np(mlp.down_proj.weight)

    for name in [f'q_proj_{layer_idx}', f'k_proj_{layer_idx}', f'v_proj_{layer_idx}',
                 f'mlp_down_proj_{layer_idx}']:
        print(f"  {name}: {arrays[name].shape}")

# Also extract final norm for LM head
arrays['final_norm'] = to_np(model.model.norm.weight)
print(f"\nfinal_norm: {arrays['final_norm'].shape}")

# Save
out_path = os.path.join(data_dir, 'qwen3_deep_layers.npz')
np.savez_compressed(out_path, **arrays)
size_mb = os.path.getsize(out_path) / 1024 / 1024
print(f"\nSaved to {out_path} ({size_mb:.0f}MB)")
print(f"Arrays: {len(arrays)}")
for k, v in sorted(arrays.items()):
    print(f"  {k}: {v.shape}")
