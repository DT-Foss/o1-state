#!/usr/bin/env python3
"""
Multiple-Choice Benchmark — HellaSwag + PIQA via FossPipeline
=============================================================
Tests the full pipeline's ability to score and rank options.
Uses all components: embeddings, commonsense, MLP facts, reservoir, attention.

Usage:
    python benchmark_mc.py [--n 100] [--bench hellaswag|piqa|both]
"""

import os, sys, time, json, argparse
import numpy as np

base = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, base)
data_dir = os.path.join(base, 'data')

# ── Args ──────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument('--n', type=int, default=100, help='Number of examples')
parser.add_argument('--bench', default='both', choices=['hellaswag', 'piqa', 'both'])
args = parser.parse_args()

# ── Load pipeline ─────────────────────────────────────────────
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
    def __init__(self, raw, t2i, Vmat, mean):
        self.raw = raw; self.t2i = t2i; self.V = Vmat; self.mean = mean
        self.dim = 512; self._token2id = t2i
    def encode(self, word):
        if ' ' in word or '_' in word:
            parts = word.replace('_', ' ').split()
            vecs = [self.encode(p) for p in parts]
            vecs = [v for v in vecs if v is not None]
            if vecs: return np.mean(vecs, axis=0).astype(np.float32)
            return None
        w = word.lower()
        for check in [w, f'Ġ{w}', w.capitalize(), f'Ġ{w.capitalize()}', w.upper()]:
            if check in self.t2i:
                vec = self.raw[self.t2i[check]].astype(np.float32)
                return (self.V @ (vec - self.mean)).astype(np.float32)
        return None


emb_shim = EmbShim(raw_emb, token2id, V, mean_vec)

from core.reservoir_lm import build_reservoir_lm
reservoir, _ = build_reservoir_lm(data_dir)
if reservoir is not None:
    readout_path = os.path.join(data_dir, 'reservoir_readout.npz')
    if os.path.exists(readout_path):
        reservoir.load_readout(readout_path)

from core.commonsense import CommonSenseEngine
cs = CommonSenseEngine()
from core.multi_hop import MultiHopReasoner
mh = MultiHopReasoner(commonsense=cs, knowledge=kb)

from core.foss_pipeline import FossPipeline
pipeline = FossPipeline()
pipeline.emb_store = emb_shim
pipeline.configure(
    knowledge=kb, raw_emb=raw_emb, lm_head=lm_head,
    id2token=id2token, commonsense=cs, multi_hop=mh,
)
if reservoir is not None and reservoir.W_out is not None:
    pipeline.reservoir = reservoir

from core.mc_scorer import MultipleChoiceScorer
scorer = MultipleChoiceScorer(pipeline)

t_load = time.perf_counter() - t0
print(f"Loaded in {t_load:.1f}s\n")


# ── HellaSwag ─────────────────────────────────────────────────
def run_hellaswag(n_examples):
    path = os.path.join(data_dir, 'external_benchmarks', 'hellaswag_val.jsonl')
    if not os.path.exists(path):
        print("HellaSwag data not found"); return

    with open(path) as f:
        examples = [json.loads(l) for l in f]

    # Stratified sample for reproducibility
    rng = np.random.default_rng(42)
    indices = rng.choice(len(examples), size=min(n_examples, len(examples)), replace=False)

    correct = 0
    total = 0
    per_scorer = {}
    latencies = []

    print(f"{'='*60}")
    print(f"HELLASWAG ({len(indices)} examples, random=25%)")
    print(f"{'='*60}")

    for idx in indices:
        ex = examples[idx]
        ctx = ex['ctx']
        endings = ex['endings']
        label = ex['label']

        t0 = time.perf_counter()
        pred = scorer.predict(ctx, endings)
        elapsed = (time.perf_counter() - t0) * 1000
        latencies.append(elapsed)

        if pred == label:
            correct += 1
        total += 1

        # Track individual scorer accuracy
        for name, method in [('emb', scorer._embedding_score),
                              ('qwen', scorer._qwen_ppl),
                              ('res', scorer._reservoir_score)]:
            s = method(ctx, endings)
            if s is not None and int(np.argmax(s)) == label:
                per_scorer[name] = per_scorer.get(name, 0) + 1

        if total % 20 == 0:
            print(f"  {total}/{len(indices)}: {correct}/{total} ({100*correct/total:.1f}%)")

    acc = 100 * correct / total
    print(f"\nHELLASWAG RESULT: {correct}/{total} ({acc:.1f}%)")
    print(f"  Random baseline: 25.0%")
    print(f"  Latency: avg={np.mean(latencies):.0f}ms, median={np.median(latencies):.0f}ms")
    print(f"  Per-scorer accuracy:")
    for name in per_scorer:
        if total > 0:
            print(f"    {name:6s}: {per_scorer[name]}/{total} ({100*per_scorer[name]/total:.1f}%)")
    return acc


# ── PIQA ──────────────────────────────────────────────────────
def run_piqa(n_examples):
    path = os.path.join(data_dir, 'external_benchmarks', 'piqa_valid.jsonl')
    label_path = os.path.join(data_dir, 'external_benchmarks', 'piqa_valid_labels.lst')
    if not os.path.exists(path) or not os.path.exists(label_path):
        print("PIQA data not found"); return

    with open(path) as f:
        examples = [json.loads(l) for l in f]
    with open(label_path) as f:
        labels = [int(l.strip()) for l in f]

    rng = np.random.default_rng(42)
    indices = rng.choice(len(examples), size=min(n_examples, len(examples)), replace=False)

    correct = 0
    total = 0
    latencies = []

    print(f"\n{'='*60}")
    print(f"PIQA ({len(indices)} examples, random=50%)")
    print(f"{'='*60}")

    for idx in indices:
        ex = examples[idx]
        label = labels[idx]
        goal = ex['goal']
        options = [ex['sol1'], ex['sol2']]

        t0 = time.perf_counter()
        pred = scorer.predict(goal, options)
        elapsed = (time.perf_counter() - t0) * 1000
        latencies.append(elapsed)

        if pred == label:
            correct += 1
        total += 1

        if total % 20 == 0:
            print(f"  {total}/{len(indices)}: {correct}/{total} ({100*correct/total:.1f}%)")

    acc = 100 * correct / total
    print(f"\nPIQA RESULT: {correct}/{total} ({acc:.1f}%)")
    print(f"  Random baseline: 50.0%")
    print(f"  Latency: avg={np.mean(latencies):.0f}ms, median={np.median(latencies):.0f}ms")
    return acc


# ── Run ───────────────────────────────────────────────────────
results = {}
if args.bench in ('hellaswag', 'both'):
    results['hellaswag'] = run_hellaswag(args.n)
if args.bench in ('piqa', 'both'):
    results['piqa'] = run_piqa(args.n)

print(f"\n{'='*60}")
print(f"SUMMARY")
print(f"{'='*60}")
for bench, acc in results.items():
    if acc is not None:
        print(f"  {bench:12s}: {acc:.1f}%")
