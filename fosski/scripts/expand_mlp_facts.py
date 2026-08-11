#!/usr/bin/env python3
"""
Expand MLP facts from 198 → all KB subjects.
Uses Qwen3-1.7B full forward pass to extract MLP activations at Layer 18.
"""

import os
import json
import numpy as np
import mlx.core as mx
from mlx_lm import load

base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
data_dir = os.path.join(base, 'data')

print("Loading Qwen3-1.7B...")
model, tokenizer = load('Qwen/Qwen3-1.7B')

# Load existing MLP facts
mlp_path = os.path.join(data_dir, 'qwen3_mlp_facts.npz')
existing = dict(np.load(mlp_path)) if os.path.exists(mlp_path) else {}
print(f"Existing MLP facts: {len(existing)}")

# Load KB subjects
with open(os.path.join(data_dir, 'knowledge_full.json')) as f:
    kb = json.load(f)

# All subjects + entity objects
subjects = sorted(set(t[0] for t in kb['triplets']))
objects_raw = set()
for t in kb['triplets']:
    obj = t[2] if len(t) > 2 else ''
    if obj and len(obj.split()) <= 4 and obj[0].isupper():
        objects_raw.add(obj)
all_entities = sorted(set(subjects) | objects_raw)
print(f"Total entities: {len(all_entities)}")

# Filter already extracted
new_entities = [e for e in all_entities if e.lower() not in existing]
print(f"New to extract: {len(new_entities)}")


def extract_mlp_activation(text, layer_idx=18):
    """Extract MLP output at layer_idx after full forward pass through all prior layers."""
    tokens = tokenizer.encode(text)
    if not tokens:
        return None

    x = mx.array([tokens])
    h = model.model.embed_tokens(x)

    for i, layer in enumerate(model.model.layers):
        if i == layer_idx:
            normed = layer.input_layernorm(h)
            attn_out = layer.self_attn(normed, mask=None, cache=None)
            if isinstance(attn_out, tuple):
                attn_out = attn_out[0]
            h_post_attn = h + attn_out
            normed2 = layer.post_attention_layernorm(h_post_attn)
            mlp_out = layer.mlp(normed2)
            return np.array(mx.array(mlp_out[0, -1, :], dtype=mx.float32), copy=False)
        else:
            h = layer(h, mask=None, cache=None)
            if isinstance(h, tuple):
                h = h[0]

    return None


# Extract
n_done = 0
n_fail = 0
new_facts = {}

for entity in new_entities:
    try:
        prompt = f"{entity} is"
        vec = extract_mlp_activation(prompt, layer_idx=18)
        if vec is not None:
            new_facts[entity.lower()] = vec
            n_done += 1
    except Exception as e:
        n_fail += 1
        if n_fail <= 5:
            print(f"  Skip '{entity}': {e}")

    if n_done % 100 == 0 and n_done > 0:
        print(f"  {n_done}/{len(new_entities)} done")

print(f"\nExtracted: {n_done}/{len(new_entities)} ({n_fail} failed)")

# Merge and save
merged = dict(existing)
merged.update(new_facts)
np.savez_compressed(mlp_path, **merged)
print(f"Saved: {len(merged)} total MLP facts to {mlp_path}")
size_mb = os.path.getsize(mlp_path) / 1024 / 1024
print(f"File size: {size_mb:.1f}MB")
