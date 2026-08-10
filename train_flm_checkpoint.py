#!/usr/bin/env python3
"""
FLM Training — Clean, simple, no Pickle checkpoints.

Order 8 max. Prune before save. No checkpoint pickle (retraining is
faster than pickle dump in swap). Only saves byte offset for resume.

Usage:
    cd ~/Desktop/foss-ki
    nohup python3 train_flm_checkpoint.py > data/flm_training.log 2>&1 &

Monitor:
    tail -f data/flm_training.log
"""

import pickle
import time
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from core.foss_lm import FossLanguageModel

CORPUS_PATH = 'data/flm_corpus.txt'
FINAL_PATH = 'data/flm_trained.pkl'
OFFSET_PATH = 'data/flm_train_offset.txt'
CHUNK_SIZE = 100_000  # 100KB per batch


def main():
    total_size = os.path.getsize(CORPUS_PATH)
    total_batches = (total_size + CHUNK_SIZE - 1) // CHUNK_SIZE

    # Always train from scratch (faster than loading a multi-GB pickle)
    flm = FossLanguageModel()
    print(f'Fresh FLM: orders {flm.CHAR_ORDERS}, {len(flm.chains)} char chains + word chain')
    print(f'Corpus: {total_size/1e6:.1f}MB, {total_batches} batches of {CHUNK_SIZE//1000}KB')
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
    print(f'Char vocab: {len(flm.chains[0].vocab)}')

    # Prune hapax legomena before save
    total_pruned = 0
    total_nodes = 0
    for chain in flm.chains:
        total_nodes += chain.count_nodes()
        total_pruned += chain.prune(min_count=2)
    print(f'Pruned: {total_pruned} nodes removed ({total_nodes}→{total_nodes - total_pruned})')
    sys.stdout.flush()

    # Save (atomic)
    print('Saving final model...')
    sys.stdout.flush()
    tmp = FINAL_PATH + '.tmp'
    with open(tmp, 'wb') as pf:
        pickle.dump(flm, pf)
    os.rename(tmp, FINAL_PATH)  # atomic
    fsize = os.path.getsize(FINAL_PATH)
    print(f'SAVED: {FINAL_PATH} ({fsize/1e6:.0f}MB)')

    # Clean up old checkpoints
    import glob
    for cp in glob.glob(os.path.join('data', 'flm_checkpoint_b*.pkl')):
        os.remove(cp)
        print(f'Cleaned: {cp}')

    print('ALL DONE.')


if __name__ == '__main__':
    main()
