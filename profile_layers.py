#!/usr/bin/env python3
"""
CASI Layer Profiling — Which Transformer Layers Actually Matter?
=================================================================
For each of the 28 Qwen3 layers, measure how much "structure"
(CASI score) it adds to the hidden state.

CASI ≈ 1.0 → layer adds no structure (redundant, skip-able)
CASI > 1.5 → layer adds meaningful structure (important)
CASI > 2.0 → critical layer (information processing)

This tells us which layers to distill and which to ignore.
"""

import os
import numpy as np
import mlx.core as mx
from mlx_lm import load

base = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(base, 'data')

print("Loading Qwen3-1.7B...")
model, tokenizer = load('Qwen/Qwen3-1.7B')
print(f"  28 layers loaded")

# Test sentences for profiling
TEST_INPUTS = [
    "The capital of France is",
    "Water is made of hydrogen and",
    "The largest planet in our solar system is",
    "Shakespeare wrote many famous",
    "Gravity pulls objects toward the",
    "The speed of light is approximately",
    "Dogs are known as man's best",
    "Photosynthesis converts sunlight into",
]


def profile_layers(text):
    """Run forward pass, capture hidden state delta at each layer.

    Returns: list of (layer_idx, delta_norm, delta_structure)
    where delta_structure is a CASI-like measure of how much
    the layer changed the representation.
    """
    tokens = tokenizer.encode(text)
    x = mx.array([tokens])

    # Get initial embedding
    h = model.model.embed_tokens(x)
    h_np = np.array(h[0, -1, :].tolist(), dtype=np.float32)

    results = []
    prev_h = h_np.copy()

    for i, layer in enumerate(model.layers):
        h = layer(h, mask=None, cache=None)
        if isinstance(h, tuple):
            h = h[0]

        h_np = np.array(h[0, -1, :].tolist(), dtype=np.float32)

        # Delta: how much did this layer change the representation?
        delta = h_np - prev_h
        delta_norm = float(np.linalg.norm(delta))

        # Structure measure: cosine similarity between consecutive states
        # Low similarity = big change = more structure added
        cos_sim = float(np.dot(prev_h, h_np) /
                       (np.linalg.norm(prev_h) * np.linalg.norm(h_np) + 1e-8))

        # CASI-like: compression ratio of delta
        # Higher entropy delta = more information added
        delta_abs = np.abs(delta)
        if delta_abs.sum() > 0:
            delta_dist = delta_abs / delta_abs.sum()
            entropy = -float(np.sum(delta_dist * np.log2(delta_dist + 1e-10)))
            max_entropy = np.log2(len(delta))
            structure = entropy / max_entropy  # 0-1, higher = more uniform = less structure
            # Invert: higher = more structured change
            casi_proxy = 1.0 / max(structure, 0.01)
        else:
            casi_proxy = 1.0

        results.append({
            'layer': i,
            'delta_norm': delta_norm,
            'cos_sim': cos_sim,
            'casi_proxy': casi_proxy,
        })

        prev_h = h_np.copy()

    return results


# Profile across all test inputs
print(f"\nProfiling {len(TEST_INPUTS)} inputs across 28 layers...")

all_profiles = []
for text in TEST_INPUTS:
    profile = profile_layers(text)
    all_profiles.append(profile)

# Average across inputs
avg_profile = []
for layer_idx in range(28):
    avg_delta = np.mean([p[layer_idx]['delta_norm'] for p in all_profiles])
    avg_cos = np.mean([p[layer_idx]['cos_sim'] for p in all_profiles])
    avg_casi = np.mean([p[layer_idx]['casi_proxy'] for p in all_profiles])
    avg_profile.append({
        'layer': layer_idx,
        'delta_norm': avg_delta,
        'cos_sim': avg_cos,
        'casi_proxy': avg_casi,
    })

# Print results
print(f"\n{'Layer':>5} {'Delta Norm':>12} {'Cos Sim':>10} {'CASI Proxy':>12} {'Importance':>12}")
print("-" * 55)

# Classify layers
important_layers = []
redundant_layers = []

for p in avg_profile:
    # Importance: high delta norm + low cos_sim = important layer
    importance = p['delta_norm'] * (1 - p['cos_sim'])
    mark = ""
    if p['delta_norm'] > np.median([x['delta_norm'] for x in avg_profile]) * 1.5:
        mark = " *** CRITICAL"
        important_layers.append(p['layer'])
    elif p['delta_norm'] < np.median([x['delta_norm'] for x in avg_profile]) * 0.5:
        mark = "  ~  redundant"
        redundant_layers.append(p['layer'])

    print(f"{p['layer']:>5} {p['delta_norm']:>12.4f} {p['cos_sim']:>10.4f} "
          f"{p['casi_proxy']:>12.3f} {importance:>12.4f}{mark}")

print(f"\n=== Summary ===")
print(f"Critical layers (high delta): {important_layers}")
print(f"Redundant layers (low delta): {redundant_layers}")
print(f"Layers to distill (top 50%): {[p['layer'] for p in sorted(avg_profile, key=lambda x: -x['delta_norm'])[:14]]}")

# Save profile
save_path = os.path.join(data_dir, 'layer_profile.npz')
np.savez_compressed(save_path,
    delta_norms=np.array([p['delta_norm'] for p in avg_profile]),
    cos_sims=np.array([p['cos_sim'] for p in avg_profile]),
    casi_proxies=np.array([p['casi_proxy'] for p in avg_profile]),
    important_layers=np.array(important_layers),
    redundant_layers=np.array(redundant_layers),
)
print(f"\nProfile saved to {save_path}")
