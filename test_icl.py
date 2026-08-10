#!/usr/bin/env python3
"""Test ICL mechanisms: Translation Vector + Sherman-Morrison."""

import sys
import os
import numpy as np
import json

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

base = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(base, 'data')

# Load embeddings
from core.reservoir_lm import EmbeddingStore, build_reservoir_lm

emb_path = os.path.join(data_dir, 'qwen3_1.7b_embeddings.npy')
vocab_path = os.path.join(data_dir, 'qwen3_1.7b_vocab.json')

print("Loading embeddings...")
store = EmbeddingStore(emb_path, vocab_path)
store._load()
print(f"  {store.vocab_size} tokens, {store.dim}d projected")

# Load raw 2048d embeddings
raw_emb = np.load(emb_path).astype(np.float32)
print(f"  Raw: {raw_emb.shape}")

# Build id2token
with open(vocab_path) as f:
    t2i = json.load(f)
id2token = {v: k for k, v in t2i.items()}

# ============================================================
# Test 1: Translation Vector ICL in RAW 2048d space
# ============================================================
print("\n" + "="*60)
print("TEST 1: Translation Vector ICL (2048d raw space)")
print("="*60)

from core.icl import TranslationVectorICL

tv_icl = TranslationVectorICL(store, raw_emb=raw_emb, id2token=id2token)

# Test: capital city analogy
print("\n--- Capital City Analogy ---")
demos = [
    ("france", "paris"),
    ("germany", "berlin"),
    ("japan", "tokyo"),
    ("spain", "madrid"),
]
for query in ["italy", "norway", "australia", "brazil"]:
    results = tv_icl.icl(demos, query, top_k=5)
    clean = [(t.replace('Ġ', '').strip(), f"{s:.3f}") for t, s in results[:5]]
    print(f"  {query} → {clean}")

# Test: EN→DE translation
print("\n--- English→German Translation ---")
demos_de = [
    ("cat", "Katze"),
    ("dog", "Hund"),
    ("house", "Haus"),
    ("tree", "Baum"),
]
for query in ["fish", "bird", "water", "sun"]:
    results = tv_icl.icl(demos_de, query, top_k=5)
    clean = [(t.replace('Ġ', '').strip(), f"{s:.3f}") for t, s in results[:5]]
    print(f"  {query} → {clean}")

# Test: comparative (big→small)
print("\n--- Comparative: big→small ---")
demos_comp = [
    ("big", "small"),
    ("tall", "short"),
    ("fast", "slow"),
    ("hot", "cold"),
]
for query in ["loud", "heavy", "bright", "old"]:
    results = tv_icl.icl(demos_comp, query, top_k=5)
    clean = [(t.replace('Ġ', '').strip(), f"{s:.3f}") for t, s in results[:5]]
    print(f"  {query} → {clean}")

# Test: past tense
print("\n--- Past Tense ---")
demos_past = [
    ("go", "went"),
    ("see", "saw"),
    ("take", "took"),
    ("give", "gave"),
]
for query in ["come", "run", "make", "eat"]:
    results = tv_icl.icl(demos_past, query, top_k=5)
    clean = [(t.replace('Ġ', '').strip(), f"{s:.3f}") for t, s in results[:5]]
    print(f"  {query} → {clean}")

# ============================================================
# Test 2: Sherman-Morrison ICL
# ============================================================
print("\n" + "="*60)
print("TEST 2: Sherman-Morrison ICL (rank-1 updates)")
print("="*60)

from core.icl import ShermanMorrisonICL

# Build reservoir
reservoir, emb = build_reservoir_lm()
if reservoir is None:
    print("ERROR: reservoir not built")
    sys.exit(1)

# Load or train readout
readout_path = os.path.join(data_dir, 'reservoir_readout.npz')
if os.path.exists(readout_path):
    reservoir.load_readout(readout_path)
    print(f"  Readout loaded: W_out shape = {reservoir.W_out.shape}")
else:
    print("  No readout found — training from KB...")
    kb_path = os.path.join(data_dir, 'knowledge_full.json')
    if os.path.exists(kb_path):
        with open(kb_path) as f:
            kb_data = json.load(f)
        triplets = kb_data.get('triplets', [])
        facts = [(t[0], t[1], t[2]) for t in triplets
                 if isinstance(t[0], str) and not t[0].startswith('/m/')]
        emb._load()
        n = 0
        for subj, rel, obj in facts[:500]:
            obj_str = str(obj).lower()
            target = None
            for variant in [obj_str, 'Ġ' + obj_str, obj_str.capitalize(),
                            'Ġ' + obj_str.capitalize()]:
                tid = emb._token2id.get(variant)
                if tid is not None:
                    target = raw_emb[tid]
                    break
            if target is None:
                continue
            words = [w for w in f"{subj} {rel}".lower().split()
                     if len(w) > 1]
            if not words:
                continue
            state, _ = reservoir.encode_context(words, emb, mix_steps=5)
            reservoir.collect_training_sample(state, target)
            n += 1
        if n > 0:
            reservoir.train_readout(ridge_alpha=10.0)
            reservoir.save(readout_path)
            print(f"  Trained on {n} facts, saved readout")
    else:
        print("  ERROR: no KB data")
        sys.exit(1)
    print(f"  W_out shape = {reservoir.W_out.shape}")

sm_icl = ShermanMorrisonICL(reservoir, learning_rate=5.0)

# Test: teach capital cities via multiple examples
print("\n--- Sherman-Morrison: Capital Cities (5 examples) ---")

# Before ICL
test_queries = ["capital australia", "capital italy", "capital brazil"]
print("BEFORE ICL:")
for query in test_queries:
    words = query.lower().split()
    state, _ = reservoir.encode_context(words, emb, mix_steps=5)
    pred = reservoir.predict(state)
    if pred is not None:
        # Search in raw space
        vec_norm = pred / (np.linalg.norm(pred) + 1e-8)
        norms = np.linalg.norm(raw_emb, axis=1, keepdims=True)
        raw_normed = raw_emb / np.clip(norms, 1e-8, None)
        sims = raw_normed @ vec_norm
        top_idx = np.argpartition(sims, -5)[-5:]
        top_idx = top_idx[np.argsort(sims[top_idx])[::-1]]
        results = [(id2token.get(int(i), '?').replace('Ġ', '').strip(),
                     float(sims[i])) for i in top_idx]
        print(f"  '{query}' → {results[:3]}")

# Apply ICL with 5 capital city examples
demos = [
    ("france", "paris"),
    ("germany", "berlin"),
    ("japan", "tokyo"),
    ("spain", "madrid"),
    ("china", "beijing"),
]
sm_icl.icl_from_word_pairs(demos, emb, raw_emb=raw_emb)
print(f"\nApplied {sm_icl._n_updates} ICL updates (lr={sm_icl.learning_rate})")

# After ICL
print("\nAFTER ICL:")
for query in test_queries:
    words = query.lower().split()
    state, _ = reservoir.encode_context(words, emb, mix_steps=5)
    pred = reservoir.predict(state)
    if pred is not None:
        vec_norm = pred / (np.linalg.norm(pred) + 1e-8)
        norms = np.linalg.norm(raw_emb, axis=1, keepdims=True)
        raw_normed = raw_emb / np.clip(norms, 1e-8, None)
        sims = raw_normed @ vec_norm
        top_idx = np.argpartition(sims, -5)[-5:]
        top_idx = top_idx[np.argsort(sims[top_idx])[::-1]]
        results = [(id2token.get(int(i), '?').replace('Ġ', '').strip(),
                     float(sims[i])) for i in top_idx]
        print(f"  '{query}' → {results[:3]}")

# Restore original W_out
sm_icl.restore()

# Test with more examples and higher learning rate
print("\n--- Sherman-Morrison: Animals→Habitats (8 examples, lr=10) ---")
sm_icl2 = ShermanMorrisonICL(reservoir, learning_rate=10.0)
demos2 = [
    ("cat", "house"),
    ("fish", "ocean"),
    ("bird", "nest"),
    ("bear", "forest"),
    ("whale", "sea"),
    ("lion", "savanna"),
    ("penguin", "ice"),
    ("monkey", "jungle"),
]
sm_icl2.icl_from_word_pairs(demos2, emb, raw_emb=raw_emb)
print(f"Applied {sm_icl2._n_updates} ICL updates (lr={sm_icl2.learning_rate})")

for query in ["dog", "eagle", "shark", "wolf"]:
    words = [query]
    state, _ = reservoir.encode_context(words, emb, mix_steps=5)
    pred = reservoir.predict(state)
    if pred is not None:
        vec_norm = pred / (np.linalg.norm(pred) + 1e-8)
        norms = np.linalg.norm(raw_emb, axis=1, keepdims=True)
        raw_normed = raw_emb / np.clip(norms, 1e-8, None)
        sims = raw_normed @ vec_norm
        top_idx = np.argpartition(sims, -10)[-10:]
        top_idx = top_idx[np.argsort(sims[top_idx])[::-1]]
        results = []
        for i in top_idx:
            tok = id2token.get(int(i), '?').replace('Ġ', '').strip()
            if tok and len(tok) > 1 and tok[0].isalpha() and all(ord(c) < 128 for c in tok):
                results.append((tok, f"{sims[i]:.3f}"))
            if len(results) >= 5:
                break
        print(f"  '{query}' habitat → {results}")

sm_icl2.restore()

print("\nDone.")
