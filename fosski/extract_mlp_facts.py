#!/usr/bin/env python3
"""
MLP Facts Extraction — Steal Knowledge from Qwen3 Layers 10-18
================================================================
For every KB subject, run Qwen3 forward pass and capture hidden
states at layers 10 and 18. This massively expands our residual
Hopfield memory from 137 words to 1300+ subjects.

Layer 10: syntactic/relational patterns (Delta=54, transition zone)
Layer 18: semantic/factual composition (Delta=242, critical zone)

Merges with existing qwen3_residuals.npz (preserving old entries).
Also extracts MLP-specific activations for fact probing.
"""

import os
import sys
import json
import numpy as np
import mlx.core as mx
from mlx_lm import load

base = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(base, 'data')

print("Loading Qwen3-1.7B...")
model, tokenizer = load('Qwen/Qwen3-1.7B')
print(f"  {len(model.layers)} layers loaded")

# Load existing residuals to merge
existing_path = os.path.join(data_dir, 'qwen3_residuals.npz')
if os.path.exists(existing_path):
    existing = dict(np.load(existing_path))
    print(f"  Loaded {len(existing)} existing residual vectors")
else:
    existing = {}
    print("  No existing residuals, starting fresh")

# Load KB subjects
with open(os.path.join(data_dir, 'knowledge_full.json')) as f:
    kb = json.load(f)

triplets = kb['triplets']
subjects = sorted(set(t[0] for t in triplets))
print(f"\nKB has {len(subjects)} unique subjects")

# Also extract for multi-word objects that are entities
objects_raw = set()
for t in triplets:
    obj = t[2] if len(t) > 2 else ''
    # Only entity-like objects (capitalized, not too long)
    if obj and len(obj.split()) <= 4 and obj[0].isupper():
        objects_raw.add(obj)
objects = sorted(objects_raw - set(subjects))
print(f"Entity objects to add: {len(objects)}")

all_words = subjects + objects

# Filter out already extracted
already = set()
for key in existing:
    word = key.rsplit('__', 1)[0]
    already.add(word)
new_words = [w for w in all_words if w.lower() not in already]
print(f"Already extracted: {len(already)}")
print(f"New to extract: {len(new_words)}")


def extract_hidden_states(text, layers_to_capture=[10, 18]):
    """Run forward pass and capture hidden states at specific layers."""
    tokens = tokenizer.encode(text)
    if not tokens:
        return {}

    x = mx.array([tokens])
    h = model.model.embed_tokens(x)

    results = {}
    for i, layer in enumerate(model.layers):
        h = layer(h, mask=None, cache=None)
        if isinstance(h, tuple):
            h = h[0]
        if i in layers_to_capture:
            state = np.array(h[0, -1, :].tolist(), dtype=np.float32)
            results[i] = state

    return results


def extract_mlp_activation(text, layer_idx=18):
    """Extract MLP-specific activation (not full hidden state).

    MLP output = the "fact recall" component per Mechanistic Interpretability.
    Separate from attention output which handles routing.
    """
    tokens = tokenizer.encode(text)
    if not tokens:
        return None

    x = mx.array([tokens])
    h = model.model.embed_tokens(x)

    for i, layer in enumerate(model.layers):
        if i == layer_idx:
            # Run attention first
            normed = layer.input_layernorm(h)
            attn_out = layer.self_attn(normed, mask=None, cache=None)
            if isinstance(attn_out, tuple):
                attn_out = attn_out[0]
            h_post_attn = h + attn_out

            # Now run MLP separately
            normed2 = layer.post_attention_layernorm(h_post_attn)
            mlp_out = layer.mlp(normed2)
            # mlp_out is the pure fact component
            return np.array(mlp_out[0, -1, :].tolist(), dtype=np.float32)
        else:
            h = layer(h, mask=None, cache=None)
            if isinstance(h, tuple):
                h = h[0]

    return None


# Phase 1: Extract hidden states for all new subjects
print(f"\n=== Phase 1: Hidden State Extraction ({len(new_words)} words) ===")
new_residuals = {}
n_done = 0
n_fail = 0

for word in new_words:
    try:
        states = extract_hidden_states(word)
        if states:
            for layer_idx, vec in states.items():
                key = f"{word.lower()}__layer_{layer_idx}"
                new_residuals[key] = vec
            n_done += 1
    except Exception as e:
        n_fail += 1
        if n_fail <= 5:
            print(f"  Skip '{word}': {e}")

    if n_done % 100 == 0 and n_done > 0:
        print(f"  {n_done}/{len(new_words)} done")

print(f"  Extracted: {n_done}/{len(new_words)} ({n_fail} failed)")

# Phase 2: Extract MLP activations for top subjects (fact probing)
# Use subjects that appear in many facts
subject_counts = {}
for t in triplets:
    subject_counts[t[0]] = subject_counts.get(t[0], 0) + 1
top_subjects = sorted(subject_counts.items(), key=lambda x: -x[1])[:200]
print(f"\n=== Phase 2: MLP Fact Extraction (top {len(top_subjects)} subjects) ===")

mlp_facts = {}
n_mlp = 0
for subj, count in top_subjects:
    try:
        # For subjects with many facts, use contextual prompts
        # "France" → "France is" (activates factual recall)
        prompt = f"{subj} is"
        mlp_vec = extract_mlp_activation(prompt, layer_idx=18)
        if mlp_vec is not None:
            mlp_facts[subj.lower()] = mlp_vec
            n_mlp += 1
    except Exception as e:
        if n_mlp < 5:
            print(f"  MLP skip '{subj}': {e}")

    if n_mlp % 50 == 0 and n_mlp > 0:
        print(f"  {n_mlp}/{len(top_subjects)} MLP done")

print(f"  MLP extracted: {n_mlp}/{len(top_subjects)}")

# Merge and save
print(f"\n=== Saving ===")
merged = dict(existing)
merged.update(new_residuals)
print(f"  Residuals: {len(existing)} old + {len(new_residuals)} new = {len(merged)} total")

save_path = os.path.join(data_dir, 'qwen3_residuals.npz')
np.savez_compressed(save_path, **merged)
print(f"  Saved to {save_path}")

# Save MLP facts separately
if mlp_facts:
    mlp_path = os.path.join(data_dir, 'qwen3_mlp_facts.npz')
    np.savez_compressed(mlp_path, **mlp_facts)
    print(f"  MLP facts: {len(mlp_facts)} vectors saved to {mlp_path}")

# Stats
n_words = len(set(k.rsplit('__', 1)[0] for k in merged))
print(f"\n=== Summary ===")
print(f"Total words/phrases with residuals: {n_words}")
print(f"Total vectors: {len(merged)} (2 layers × {n_words} words)")
print(f"MLP fact vectors: {len(mlp_facts)}")
print(f"Each vector: 2048d float32")
