#!/usr/bin/env python3
"""
FLM Training on Merged Corpus (Tech + Wiki).

Usage:
    cd ~/Desktop/foss-ki
    nohup python3 train_flm_merged.py > data/flm_merge_training.log 2>&1 &

Monitor:
    tail -f data/flm_merge_training.log
"""

import pickle
import time
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from core.foss_lm import FossLanguageModel

CORPUS_PATH = 'data/flm_corpus_merged.txt'
FINAL_PATH = 'data/flm_merged.pkl'
CHUNK_SIZE = 100_000  # 100KB per batch


def main():
    total_size = os.path.getsize(CORPUS_PATH)
    total_batches = (total_size + CHUNK_SIZE - 1) // CHUNK_SIZE

    flm = FossLanguageModel()
    print(f'Fresh FLM: orders {flm.CHAR_ORDERS}')
    print(f'Corpus: {total_size/1e6:.1f}MB, {total_batches} batches')
    sys.stdout.flush()

    t0 = time.time()
    trained = 0
    batch = 0

    with open(CORPUS_PATH, 'r', encoding='utf-8') as f:
        while True:
            chunk = f.read(CHUNK_SIZE)
            if not chunk:
                break
            flm.train(chunk)
            trained += len(chunk)
            batch += 1
            elapsed = time.time() - t0
            rate = trained / elapsed / 1000 if elapsed > 0 else 0
            remaining = total_size - trained
            eta = remaining / (rate * 1000) / 60 if rate > 0 else 0

            if batch % 10 == 0:
                print(f'  Batch {batch}/{total_batches}: {trained/1e6:.1f}MB '
                      f'({trained*100/total_size:.0f}%) @ {rate:.1f}KB/s, '
                      f'ETA {eta:.0f}min')
                sys.stdout.flush()

    elapsed_total = time.time() - t0
    print(f'\nTRAINING COMPLETE: {trained/1e6:.1f}MB in {elapsed_total:.0f}s ({elapsed_total/60:.1f}min)')
    print(f'Rate: {trained/elapsed_total/1000:.1f}KB/s')

    # Node counts before prune
    for i, chain in enumerate(flm.chains):
        print(f'  Chain {i} (order {chain.max_order}): {chain.count_nodes():,} nodes')

    # Aggressive per-chain pruning for merged corpus
    prune_levels = {0: 3, 1: 10, 2: 25}  # More aggressive for larger corpus
    total_pruned = 0
    for i, chain in enumerate(flm.chains):
        mc = prune_levels.get(i, 5)
        before = chain.count_nodes()
        removed = chain.prune(min_count=mc)
        total_pruned += removed
        print(f'  Pruned chain {i}: mc={mc}, {before:,} → {chain.count_nodes():,}')

    # Prune word chain
    wc = flm.word_chain
    dead = [k for k, v in wc.transitions.items() if wc.state_totals[k] <= 3]
    for k in dead:
        del wc.transitions[k]
        del wc.state_totals[k]
    print(f'  Word chain pruned: {len(dead):,} removed')
    sys.stdout.flush()

    # Save (atomic)
    print('Saving...')
    sys.stdout.flush()
    tmp = FINAL_PATH + '.tmp'
    with open(tmp, 'wb') as f:
        pickle.dump(flm, f)
    os.rename(tmp, FINAL_PATH)
    fsize = os.path.getsize(FINAL_PATH)
    print(f'SAVED: {FINAL_PATH} ({fsize/1e6:.0f}MB)')
    print('DONE.')


if __name__ == '__main__':
    main()
