"""
Reservoir ESN → lm_head Text Generation
========================================
Train reservoir with 2048d raw embedding targets.
Query → reservoir state → predict 2048d → lm_head → vocabulary logits → words.
"""
import numpy as np
import json
import os
import sys
import time

print("=" * 60)
print("FOSS-KI: Reservoir → lm_head Text Generation")
print("=" * 60)

# --- Load embeddings ---
t0 = time.time()
print("\n[1] Loading Qwen3-1.7B embeddings...")
raw_emb = np.load('data/qwen3_1.7b_embeddings.npy').astype(np.float32)
with open('data/qwen3_1.7b_vocab.json') as f:
    token2id = json.load(f)
id2token = {v: k for k, v in token2id.items()}
print(f"    Embeddings: {raw_emb.shape} ({raw_emb.nbytes/1e6:.0f} MB)")

# --- Load lm_head ---
print("\n[2] Loading lm_head (151936 × 2048)...")
lm_head = np.load('data/qwen3_lm_head.npy').astype(np.float32)
print(f"    lm_head: {lm_head.shape} ({lm_head.nbytes/1e6:.0f} MB)")
print(f"    Load time: {time.time()-t0:.1f}s")

# --- Build reservoir with 2048d output ---
print("\n[3] Building Reservoir (2048 nodes, 2048d output)...")
from core.reservoir_lm import ReservoirLM, EmbeddingStore

# We need 512d projected embeddings for reservoir INPUT
# But 2048d raw embeddings as TRAINING TARGETS (for lm_head compatibility)
emb_store = EmbeddingStore('data/qwen3_1.7b_embeddings.npy', 'data/qwen3_1.7b_vocab.json')
emb_store._load()
print(f"    Reservoir input dim: {emb_store.dim}d (projected)")
print(f"    Readout target dim: 2048d (raw, for lm_head)")

reservoir = ReservoirLM(input_dim=emb_store.dim, reservoir_size=2048)

# --- Training data: subject→object triplets ---
print("\n[4] Training on subject→object triplets (2048d targets)...")
triplets = [
    # Geography
    ("japan", "tokyo"), ("france", "paris"), ("germany", "berlin"),
    ("italy", "rome"), ("spain", "madrid"), ("brazil", "brasilia"),
    ("china", "beijing"), ("russia", "moscow"), ("india", "delhi"),
    ("egypt", "cairo"), ("australia", "canberra"), ("canada", "ottawa"),
    ("mexico", "mexico"), ("argentina", "buenos"), ("peru", "lima"),
    ("norway", "oslo"), ("sweden", "stockholm"), ("finland", "helsinki"),
    ("portugal", "lisbon"), ("greece", "athens"), ("turkey", "ankara"),
    # Science
    ("water", "hydrogen"), ("sun", "star"), ("earth", "planet"),
    ("moon", "satellite"), ("mars", "red"), ("jupiter", "largest"),
    ("oxygen", "gas"), ("iron", "metal"), ("gold", "element"),
    ("diamond", "carbon"), ("photon", "light"), ("electron", "particle"),
    ("gravity", "newton"), ("relativity", "einstein"), ("quantum", "planck"),
    ("evolution", "darwin"), ("dna", "helix"), ("cell", "biology"),
    # Literature/Culture
    ("shakespeare", "hamlet"), ("orwell", "1984"), ("tolkien", "hobbit"),
    ("picasso", "guernica"), ("beethoven", "symphony"), ("mozart", "requiem"),
    ("bach", "fugue"), ("da vinci", "mona"), ("michelangelo", "sistine"),
    # Technology
    ("python", "programming"), ("linux", "kernel"), ("internet", "network"),
    ("computer", "processor"), ("algorithm", "computation"), ("database", "storage"),
    # Animals
    ("cat", "feline"), ("dog", "canine"), ("eagle", "raptor"),
    ("whale", "marine"), ("spider", "eight"), ("snake", "reptile"),
    # Food
    ("pizza", "italian"), ("sushi", "japanese"), ("chocolate", "cocoa"),
    # Context sentences — teach simple patterns
    ("capital of japan", "tokyo"), ("capital of france", "paris"),
    ("capital of germany", "berlin"), ("capital of italy", "rome"),
    ("wrote hamlet", "shakespeare"), ("wrote 1984", "orwell"),
    ("discovered gravity", "newton"), ("theory of relativity", "einstein"),
    ("largest planet", "jupiter"), ("closest star", "sun"),
    ("chemical formula of water", "h2o"), ("speed of light", "photon"),
]

# Augment with more context patterns
augmented = []
for subj, obj in triplets:
    augmented.append((subj, obj))
    # Reverse: object→subject associations
    augmented.append((obj, subj))

n_train = 0
t_train = time.time()
for subj, obj in augmented:
    # Encode subject through reservoir
    words = subj.lower().split()
    reservoir.reset_state()
    zero = np.zeros(emb_store.dim, dtype=np.float32)
    for w in words:
        v = emb_store.encode(w)
        if v is None:
            v = zero
        reservoir.step(v)
    # Mixing steps
    for _ in range(3):
        reservoir.step(zero)
    state = (reservoir.state_pos + reservoir.state_neg) / 2.0

    # Target: 2048d raw embedding of the object's FIRST token
    obj_words = obj.lower().split()
    target = None
    for w in obj_words:
        for variant in [w, 'Ġ' + w, w.capitalize(), 'Ġ' + w.capitalize()]:
            tid = token2id.get(variant)
            if tid is not None:
                target = raw_emb[tid]
                break
        if target is not None:
            break

    if target is not None:
        reservoir.collect_training_sample(state, target)
        n_train += 1

print(f"    Collected {n_train} training samples")
n = reservoir.train_readout(ridge_alpha=10.0)
print(f"    Trained readout on {n} samples ({time.time()-t_train:.1f}s)")

# --- Test: Query → Reservoir → lm_head → Text ---
print("\n[5] GENERATION TEST: Query → Reservoir → lm_head → Words")
print("=" * 60)

# Pre-normalize lm_head rows for cosine similarity
lm_norms = np.linalg.norm(lm_head, axis=1, keepdims=True)
lm_normed = lm_head / np.clip(lm_norms, 1e-8, None)

test_queries = [
    "capital of japan",
    "capital of france",
    "capital of germany",
    "wrote hamlet",
    "wrote 1984",
    "largest planet",
    "discovered gravity",
    "theory of relativity",
    "chemical formula of water",
    "python",
    "shakespeare",
    "sun",
]

correct = 0
total = len(test_queries)

for query in test_queries:
    # Encode through reservoir
    words = query.lower().split()
    reservoir.reset_state()
    zero = np.zeros(emb_store.dim, dtype=np.float32)
    for w in words:
        v = emb_store.encode(w)
        if v is None:
            v = zero
        reservoir.step(v)
    for _ in range(3):
        reservoir.step(zero)
    state = (reservoir.state_pos + reservoir.state_neg) / 2.0

    # Predict 2048d embedding
    pred = reservoir.predict(state)
    if pred is None:
        print(f"  {query:35s} → [no readout]")
        continue

    # Method 1: lm_head logits (matrix multiply)
    logits = lm_head @ pred  # (151936,)
    top_idx = np.argpartition(logits, -10)[-10:]
    top_idx = top_idx[np.argsort(logits[top_idx])[::-1]]
    top_tokens_logit = [(id2token.get(int(i), '?'), float(logits[i])) for i in top_idx]

    # Method 2: Cosine similarity in embedding space
    pred_norm = pred / (np.linalg.norm(pred) + 1e-8)
    sims = lm_normed @ pred_norm
    top_idx2 = np.argpartition(sims, -10)[-10:]
    top_idx2 = top_idx2[np.argsort(sims[top_idx2])[::-1]]
    top_tokens_cos = [(id2token.get(int(i), '?'), float(sims[i])) for i in top_idx2]

    # Display
    top5_logit = ', '.join(f"{t}({s:.1f})" for t, s in top_tokens_logit[:5])
    top5_cos = ', '.join(f"{t}({s:.3f})" for t, s in top_tokens_cos[:5])
    print(f"\n  Q: {query}")
    print(f"  lm_head logits: {top5_logit}")
    print(f"  cosine sim:     {top5_cos}")

print(f"\n{'='*60}")
print(f"Total time: {time.time()-t0:.1f}s")
print(f"{'='*60}")
