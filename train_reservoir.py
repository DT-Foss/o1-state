#!/usr/bin/env python3
"""
Train Reservoir Readout on ALL KB Facts
========================================
One-shot Ridge regression: state → target embedding.
No gradients. No GPU. Just a matmul.

Input: 137K facts from Brain → (subject+relation) → reservoir state
Target: object embedding in 2048d raw space
Output: W_out (2048, 4096) saved to data/reservoir_readout.npz
"""

import os, sys, time, json
import numpy as np

base = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, base)
data_dir = os.path.join(base, 'data')

print("=== Reservoir Readout Training ===\n")

# 1. Load components
t0 = time.perf_counter()

from core.reservoir_lm import build_reservoir_lm
from core.brain import BrainSnapshot

reservoir, emb = build_reservoir_lm(data_dir)
if reservoir is None:
    print("ERROR: Could not build reservoir")
    sys.exit(1)

# Load raw embeddings for target vectors
raw_emb = np.load(os.path.join(data_dir, 'qwen3_1.7b_embeddings.npy'))
with open(os.path.join(data_dir, 'qwen3_1.7b_vocab.json')) as f:
    token2id = json.load(f)

# Load Brain facts
brain = BrainSnapshot.load(os.path.join(data_dir, 'foss-ki.brain'))
parts = brain.restore_parts()
kb = parts['knowledge']

t_load = time.perf_counter() - t0
print(f"Loaded in {t_load:.1f}s")
print(f"  KB facts: {len(kb.facts)}")
print(f"  Vocab: {len(token2id)} tokens")
print(f"  Reservoir: {reservoir.reservoir_size}d, input {reservoir.input_dim}d")
print()

# 2. Collect training pairs: (subject+relation words) → object embedding
t0 = time.perf_counter()
n_collected = 0
n_skipped = 0

for fact in kb.facts:
    if len(fact) < 3:
        n_skipped += 1
        continue

    subj, rel, obj = fact[0], fact[1], fact[2]

    # Find target: object embedding in raw 2048d space
    obj_str = str(obj).lower().strip()
    target = None
    for variant in [obj_str, f'Ġ{obj_str}', obj_str.capitalize(),
                    f'Ġ{obj_str.capitalize()}']:
        tid = token2id.get(variant)
        if tid is not None:
            target = raw_emb[tid].astype(np.float32)
            break

    if target is None:
        # Try first word of multi-word object
        first_word = obj_str.split()[0] if ' ' in obj_str else None
        if first_word:
            for variant in [first_word, f'Ġ{first_word}', first_word.capitalize(),
                            f'Ġ{first_word.capitalize()}']:
                tid = token2id.get(variant)
                if tid is not None:
                    target = raw_emb[tid].astype(np.float32)
                    break

    if target is None:
        n_skipped += 1
        continue

    # Encode input: subject + relation words through reservoir
    words = [w for w in f"{subj} {rel}".lower().split() if len(w) > 1]
    if not words:
        n_skipped += 1
        continue

    state, _ = reservoir.encode_context(words, emb, mix_steps=3)
    reservoir.collect_training_sample(state, target)
    n_collected += 1

    if n_collected % 5000 == 0:
        print(f"  Collected {n_collected} samples...")

t_collect = time.perf_counter() - t0
print(f"\nCollected {n_collected} training pairs in {t_collect:.1f}s")
print(f"  Skipped: {n_skipped} (no target embedding found)")
print()

# 3. Train readout (Ridge regression)
print("Training readout (Ridge regression)...")
t0 = time.perf_counter()
n_trained = reservoir.train_readout(ridge_alpha=10.0)
t_train = time.perf_counter() - t0
print(f"  Trained on {n_trained} samples in {t_train:.1f}s")
print(f"  W_out shape: {reservoir.W_out.shape}")
print()

# 4. Save
readout_path = os.path.join(data_dir, 'reservoir_readout.npz')
reservoir.save(readout_path)
size_mb = os.path.getsize(readout_path) / 1024 / 1024
print(f"Saved to {readout_path} ({size_mb:.1f} MB)")
print()

# 5. Quick validation
print("=== Quick Validation ===")
norms = np.linalg.norm(raw_emb, axis=1, keepdims=True)
raw_normed = (raw_emb / np.clip(norms, 1e-8, None)).astype(np.float32)
id2token = {v: k for k, v in token2id.items()}

test_queries = [
    ["capital", "france"],
    ["author", "hamlet"],
    ["color", "sky"],
    ["water", "formula"],
    ["discovered", "gravity"],
    ["born", "shakespeare"],
    ["language", "japan"],
    ["currency", "uk"],
]

for words in test_queries:
    state, _ = reservoir.encode_context(words, emb, mix_steps=5)
    pred = reservoir.predict(state)
    if pred is None:
        print(f"  {' '.join(words):25s} -> NO PREDICTION")
        continue

    pred_n = pred / (np.linalg.norm(pred) + 1e-8)
    sims = raw_normed @ pred_n.astype(np.float32)
    top5 = np.argsort(sims)[-5:][::-1]

    results = []
    for idx in top5:
        token = id2token.get(idx, '?').replace('Ġ', '').strip()
        if token and len(token) > 1 and token[0].isalpha():
            results.append(f"{token}({sims[idx]:.3f})")
    if not results:
        results = ["<no clean tokens>"]

    print(f"  {' '.join(words):25s} -> {', '.join(results[:5])}")

print(f"\nTotal time: {time.perf_counter() - t0:.1f}s")
