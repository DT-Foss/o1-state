#!/usr/bin/env python3
"""
HellaSwag Benchmark
====================
Tests commonsense sentence completion on 10,042 examples.
GPT-4 achieves ~95.3%. Random baseline: 25%.

Usage:
    python3 benchmarks/benchmark_hellaswag.py [--flm PATH]
"""

import sys
import os
import time
import argparse
import pickle

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.hellaswag_solver import HellaSwagSolver, evaluate_hellaswag


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--flm', help='Path to FLM pickle', default=None)
    args = parser.parse_args()

    base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_path = os.path.join(base, 'data', 'external_benchmarks', 'hellaswag_val.jsonl')

    if not os.path.exists(data_path):
        print(f"HellaSwag data not found at {data_path}")
        return

    glove_path = os.path.join(base, 'data', 'glove.6B.100d.txt')
    conceptnet_path = os.path.join(base, 'data', 'conceptnet_index.pkl')
    pmi_path = os.path.join(base, 'data', 'pmi_index.pkl')

    # Load FLM if provided
    flm = None
    if args.flm:
        print(f"Loading FLM from {args.flm}...")
        t0 = time.time()
        with open(args.flm, 'rb') as f:
            flm = pickle.load(f)
        print(f"Loaded in {time.time()-t0:.1f}s")

    # Build Stacked Hopfield from GloVe (if available)
    stacked = None
    word2idx = None
    vecs = None
    import numpy as np
    words_npy = glove_path.replace('.txt', '_words.npy')
    vecs_npy = glove_path.replace('.txt', '.npy')
    if os.path.exists(words_npy) and os.path.exists(vecs_npy):
        try:
            from core.stacked_hopfield import StackedHopfield
            words = np.load(words_npy, allow_pickle=True)
            vecs = np.load(vecs_npy)
            word2idx = {str(w): i for i, w in enumerate(words)}
            stacked = StackedHopfield(dim=vecs.shape[1])
            # Store top 5000 words in Layer 1
            for i in range(min(5000, len(words))):
                stacked.store_word(vecs[i], str(words[i]))
            print(f"  Stacked Hopfield: {stacked.stats}")
        except Exception as e:
            print(f"  Stacked Hopfield init failed: {e}")

    # Build Hopfield Sequence Memory from commonsense sequences (if GloVe available)
    seq_mem = None
    tok_ret = None
    if word2idx is not None and vecs is not None:
        try:
            from core.hopfield_lm import (
                build_sequence_memory, build_token_retriever,
                COMMONSENSE_RELATIONS
            )
            # Build sequence memory from commonsense relation text
            # Convert relations to natural sequences for training
            relation_text = '. '.join(
                f"{s} {r.replace('_', ' ')} {o}"
                for s, r, o in COMMONSENSE_RELATIONS
            )
            seq_mem = build_sequence_memory(word2idx, vecs, text=relation_text, window=3)
            print(f"  Sequence Memory: {seq_mem.n_sequences} sequences")

            # Build Per-Token Retriever
            tok_ret = build_token_retriever(word2idx, vecs)
            print(f"  Token Retriever: ready")
        except Exception as e:
            print(f"  Sequence Memory/Retriever init failed: {e}")

    solver = HellaSwagSolver(
        glove_path=glove_path,
        lm_model=flm,
        stacked_hopfield=stacked,
        sequence_memory=seq_mem,
        token_retriever=tok_ret,
        conceptnet_path=conceptnet_path,
        pmi_path=pmi_path,
    )

    print("=" * 65)
    print("  HELLASWAG — Commonsense Sentence Completion")
    print("=" * 65)

    t0 = time.time()
    correct, total, errors = evaluate_hellaswag(solver, data_path)
    elapsed = time.time() - t0

    pct = correct / total * 100 if total > 0 else 0

    print(f"\n  Result: {correct}/{total} ({pct:.1f}%)")
    print(f"  Time: {elapsed:.1f}s")

    if errors:
        print(f"\n  Sample errors ({min(len(errors), 10)}):")
        for e in errors[:10]:
            print(f"    Ctx: {e['ctx']}")
            print(f"      Pred: {e['pred']}")
            print(f"      True: {e['correct']}")
            print()

    print(f"\n  {'Metric':<35} {'FOSS-KI':>10} {'GPT-4':>10}")
    print(f"  {'─' * 57}")
    print(f"  {'HellaSwag accuracy':<35} {pct:>9.1f}% {'~95%':>10}")
    print(f"  {'Random baseline (4 choices)':<35} {'25%':>10} {'':>10}")
    print(f"  {'Parameters':<35} {'0':>10} {'~200B':>10}")
    print(f"{'=' * 65}")


if __name__ == '__main__':
    main()
