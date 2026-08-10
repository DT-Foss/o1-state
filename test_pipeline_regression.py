#!/usr/bin/env python3
"""
Pipeline Regression Tests — Gold Standard
==========================================
20 Q&A pairs that verify FossPipeline.query() returns correct answers
with all components loaded (Reservoir, Hopfield Bank, KB, MLP, Residual).

Run: python test_pipeline_regression.py
"""

import os, sys, time, json
import numpy as np

base = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, base)
data_dir = os.path.join(base, 'data')

# ── Load everything ──────────────────────────────────────────
print("Loading pipeline...")
t0 = time.perf_counter()

from core.brain import BrainSnapshot
brain = BrainSnapshot.load(os.path.join(data_dir, 'foss-ki.brain'))
parts = brain.restore_parts()
kb = parts['knowledge']

raw_emb = np.load(os.path.join(data_dir, 'qwen3_1.7b_embeddings.npy'))
with open(os.path.join(data_dir, 'qwen3_1.7b_vocab.json')) as f:
    token2id = json.load(f)
id2token = {v: k for k, v in token2id.items()}
lm_head = np.load(os.path.join(data_dir, 'qwen3_lm_head.npy')).astype(np.float32)
V = np.load(os.path.join(data_dir, 'qwen3_svd_V512.npy')).astype(np.float32)
mean_vec = np.load(os.path.join(data_dir, 'qwen3_svd_mean.npy')).astype(np.float32)

class EmbShim:
    def __init__(self, raw, t2i, V, mean):
        self.raw = raw; self.t2i = t2i; self.V = V; self.mean = mean
        self.dim = 512; self._token2id = t2i
    def encode(self, word):
        # Multi-word: split, encode each, average
        if ' ' in word or '_' in word:
            parts = word.replace('_', ' ').split()
            vecs = [self.encode(p) for p in parts]
            vecs = [v for v in vecs if v is not None]
            if vecs:
                return np.mean(vecs, axis=0).astype(np.float32)
            return None
        w = word.lower()
        for check in [w, f'Ġ{w}', w.capitalize(), f'Ġ{w.capitalize()}', w.upper()]:
            if check in self.t2i:
                vec = self.raw[self.t2i[check]].astype(np.float32)
                return (self.V @ (vec - self.mean)).astype(np.float32)
        return None
    def get(self, word): return self.encode(word)

emb_shim = EmbShim(raw_emb, token2id, V, mean_vec)

from core.reservoir_lm import build_reservoir_lm
reservoir, _ = build_reservoir_lm(data_dir)
if reservoir is not None:
    readout_path = os.path.join(data_dir, 'reservoir_readout.npz')
    if os.path.exists(readout_path):
        reservoir.load_readout(readout_path)

from core.residual_hopfield import ResidualHopfield
res_hop = ResidualHopfield(data_dir)
from core.commonsense import CommonSenseEngine
cs = CommonSenseEngine()
from core.multi_hop import MultiHopReasoner
mh = MultiHopReasoner(commonsense=cs, knowledge=kb)

from core.foss_pipeline import FossPipeline
pipeline = FossPipeline()
pipeline.emb_store = emb_shim
pipeline.configure(
    knowledge=kb, raw_emb=raw_emb, lm_head=lm_head,
    id2token=id2token, residual_hopfield=res_hop,
    commonsense=cs, multi_hop=mh,
)
if reservoir is not None and reservoir.W_out is not None:
    pipeline.reservoir = reservoir

t_load = time.perf_counter() - t0
print(f"Loaded in {t_load:.1f}s\n")

# ── Gold Standard Tests ──────────────────────────────────────
TESTS = [
    # Basic knowledge (8)
    ("capital france", "Paris", "capital"),
    ("author hamlet", "Shakespeare", "author"),
    ("color of sky", "blue", "property"),
    ("largest planet solar system", "Jupiter", "superlative"),
    ("who discovered gravity", "Newton", "discoverer"),
    ("speed of light", "299", "numeric"),
    ("water made of", "hydrogen", "composition"),
    ("Shakespeare born", "1564", "date"),
    # Extended knowledge (6)
    ("capital germany", "Berlin", "capital"),
    ("language japan", "Japanese", "language"),
    ("currency uk", "Pound", "currency"),
    ("population china", "1.4", "numeric"),
    ("inventor telephone", "Bell", "inventor"),
    ("chemical symbol gold", "Au", "symbol"),
    # Edge cases (6)
    ("capital of brazil", "Brasilia", "capital_variant"),
    ("who wrote romeo and juliet", "Shakespeare", "author_variant"),
    ("what is the biggest ocean", "Ocean", "superlative_variant"),
    ("born einstein", "1879", "date_variant"),
    ("france borders", "switzerland", "relation"),  # Switzerland, Germany, etc. — any neighbor correct
    ("type python", "interpret", "type"),  # "interpreted" or "programming language" both valid
]

# ── Run tests ────────────────────────────────────────────────
print("=" * 70)
print("PIPELINE REGRESSION TESTS")
print("=" * 70)
print()

passed = 0
failed = 0
errors = []
latencies = []

for query, expected_contains, category in TESTS:
    t0 = time.perf_counter()
    try:
        result = pipeline.query(query)
        elapsed = (time.perf_counter() - t0) * 1000
        latencies.append(elapsed)

        answer = str(result.get('answer', '')) if result else ''
        answer_lower = answer.lower()
        expected_lower = expected_contains.lower()

        if expected_lower in answer_lower:
            status = "PASS"
            passed += 1
        else:
            status = "FAIL"
            failed += 1
            errors.append((query, expected_contains, answer, category))

        conf = result.get('confidence', 0) if result else 0
        sources = [s[0] for s in result.get('sources', [])] if result else []
        print(f"  [{status}] \"{query}\" → \"{answer}\" "
              f"(expect: {expected_contains}, {elapsed:.0f}ms, "
              f"src: {','.join(sources)})")

    except Exception as e:
        elapsed = (time.perf_counter() - t0) * 1000
        failed += 1
        errors.append((query, expected_contains, f"ERROR: {e}", category))
        print(f"  [FAIL] \"{query}\" → ERROR: {e}")

print()
print("=" * 70)
print(f"RESULT: {passed}/{passed+failed} passed ({100*passed/(passed+failed):.0f}%)")
if latencies:
    print(f"Latency: avg={np.mean(latencies):.0f}ms, "
          f"median={np.median(latencies):.0f}ms, "
          f"max={np.max(latencies):.0f}ms, "
          f"budget=500ms")
print("=" * 70)

if errors:
    print("\nFAILURES:")
    for q, exp, got, cat in errors:
        print(f"  [{cat}] \"{q}\" expected \"{exp}\" got \"{got}\"")

sys.exit(0 if failed == 0 else 1)
