#!/usr/bin/env python3
"""Profile pipeline queries with fully configured components."""

import os, sys, time, json
import numpy as np

base = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, base)
data_dir = os.path.join(base, 'data')

print("Loading FOSS-KI with full configuration...")
t0 = time.perf_counter()

# ── 1. Load Brain (4855 facts) ────────────────────────────
from core.brain import BrainSnapshot
brain_path = os.path.join(data_dir, 'foss-ki.brain')
brain = BrainSnapshot.load(brain_path)
parts = brain.restore_parts()
kb = parts['knowledge']
print(f"  Brain loaded: {len(kb.facts)} facts")

# ── 2. Load embeddings ────────────────────────────────────
raw_emb = np.load(os.path.join(data_dir, 'qwen3_1.7b_embeddings.npy'))
with open(os.path.join(data_dir, 'qwen3_1.7b_vocab.json')) as f:
    token2id = json.load(f)
id2token = {v: k for k, v in token2id.items()}
lm_head = np.load(os.path.join(data_dir, 'qwen3_lm_head.npy')).astype(np.float32)
V = np.load(os.path.join(data_dir, 'qwen3_svd_V512.npy')).astype(np.float32)
mean_vec = np.load(os.path.join(data_dir, 'qwen3_svd_mean.npy')).astype(np.float32)

# ── 3. Embedding shim ─────────────────────────────────────
class EmbShim:
    def __init__(self, raw, t2i, V, mean):
        self.raw = raw
        self.t2i = t2i
        self.V = V
        self.mean = mean
        self.dim = 512
        self._token2id = t2i
    def encode(self, word):
        w = word.lower()
        for check in [w, f'Ġ{w}', w.capitalize(), f'Ġ{w.capitalize()}', w.upper()]:
            if check in self.t2i:
                vec_2048 = self.raw[self.t2i[check]].astype(np.float32)
                return (self.V @ (vec_2048 - self.mean)).astype(np.float32)
        return None
    def get(self, word):
        return self.encode(word)
    def get_raw(self, word):
        w = word.lower()
        for check in [w, f'Ġ{w}', w.capitalize(), f'Ġ{w.capitalize()}']:
            if check in self.t2i:
                return self.raw[self.t2i[check]].astype(np.float32)
        return None

emb_shim = EmbShim(raw_emb, token2id, V, mean_vec)

# ── 4. Reservoir LM (pre-trained ESN with matching weights) ──
from core.reservoir_lm import build_reservoir_lm
reservoir, reservoir_emb = build_reservoir_lm(data_dir)
if reservoir is not None:
    readout_path = os.path.join(data_dir, 'reservoir_readout.npz')
    if os.path.exists(readout_path):
        reservoir.load_readout(readout_path)
    print(f"  Reservoir: READY (W_out: {reservoir.W_out.shape if reservoir.W_out is not None else 'N/A'})")

# ── 4b. Residual Hopfield ─────────────────────────────────
from core.residual_hopfield import ResidualHopfield
res_hop = ResidualHopfield(data_dir)

# ── 5. CommonSense Engine ─────────────────────────────────
from core.commonsense import CommonSenseEngine
cs = CommonSenseEngine()

# ── 6. MultiHop Reasoner ──────────────────────────────────
from core.multi_hop import MultiHopReasoner
mh = MultiHopReasoner(commonsense=cs, knowledge=kb)

# ── 7. Configure Pipeline ─────────────────────────────────
from core.foss_pipeline import FossPipeline
pipeline = FossPipeline()
# Set emb_store BEFORE configure so HopfieldTemplateBank can be built
pipeline.emb_store = emb_shim
pipeline.configure(
    knowledge=kb,
    raw_emb=raw_emb,
    lm_head=lm_head,
    id2token=id2token,
    residual_hopfield=res_hop,
    commonsense=cs,
    multi_hop=mh,
)
if reservoir is not None and reservoir.W_out is not None:
    pipeline.reservoir = reservoir

t_load = time.perf_counter() - t0
print(f"  Total load: {t_load:.1f}s")
print(f"  NN index: {pipeline._nn_index.backend if pipeline._nn_index else 'none'}")
print()

# ── 8. Profile queries ────────────────────────────────────
queries = [
    "capital france",
    "author hamlet",
    "color of sky",
    "largest planet solar system",
    "who discovered gravity",
    "speed of light",
    "water made of",
    "Shakespeare born",
]

print("=" * 70)
print("QUERY PROFILING — Full Pipeline")
print("=" * 70)
print()

latencies = []
for q in queries:
    times = []
    for _ in range(3):
        t0 = time.perf_counter()
        result = pipeline.query(q)
        elapsed = (time.perf_counter() - t0) * 1000
        times.append(elapsed)

    avg = np.mean(times[1:])  # skip warmup
    latencies.append(avg)
    answer = str(result.get('answer', '?'))[:50] if result else '?'
    sources = [(s[0], s[2]) for s in result.get('sources', [])] if result else []

    print(f"\"{q}\"")
    print(f"  → {answer}")
    print(f"  Time: {avg:.1f} ms (runs: {', '.join(f'{t:.0f}' for t in times)})")
    if sources:
        print(f"  Sources: {sources}")
    trace = result.get('trace', []) if result else []
    for t in trace:
        print(f"  {t}")
    print()

print(f"Average: {np.mean(latencies):.1f} ms | Median: {np.median(latencies):.1f} ms | Max: {np.max(latencies):.1f} ms")
