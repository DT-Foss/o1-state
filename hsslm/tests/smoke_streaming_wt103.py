"""
Closing smoke test: HSSLM streaming with a surprise-gate, on real WT-103
text, using the Organism's own vocabulary (length_extrap_v2.build_vocab,
5000 words + unk + mask = 5002 IDs) instead of HSSLM's placeholder
BPE-like tokenizer -- per lead's sovereignty decision: same vocabulary as
the A3-life Organism, no external BPE dependency, direct comparability.

Runs 200 gated chunks through HSSLMStreamer (hsslm/neural/streaming.py) on
real WT-103 text (offline, from the local HF cache). Reports:
  - loss trajectory (should trend down)
  - gate rate (fraction of chunks that triggered a real backward step)
  - the F2 streaming-exactness number (measured separately in
    smoke_neural.py; not re-measured here -- this is the live-data run)

Run:
    OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
    TOKENIZERS_PARALLELISM=false HF_DATASETS_OFFLINE=1 HF_HUB_OFFLINE=1 \
    python3 smoke_streaming_wt103.py
"""
import os
import re
import sys

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
os.environ.setdefault("HF_HUB_OFFLINE", "1")

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, "/Users/bhkmie/Documents/Forschung/O1_juli/src")

import torch
torch.set_num_threads(1)
torch.manual_seed(43)

from neural.model import HSSLM
from neural.config import HSSLMConfig
from neural.streaming import HSSLMStreamer

# Reuse the Organism's exact vocab-building recipe (not importing
# portable_organism.py itself, per the lead's boundary -- length_extrap_v2 is
# the shared dependency both already use).
from length_extrap_v2 import build_vocab, tokenize, VOCAB_MAX


def load_wikitext103_train_text(max_chars: int = 20_000_000) -> str:
    """Load WT-103 train split text, offline, from local HF cache.
    Capped at max_chars for a smoke run (full WT-103 train is huge)."""
    from datasets import load_dataset
    ds = load_dataset("Salesforce/wikitext", "wikitext-103-raw-v1")
    text = "\n\n".join(ds["train"]["text"][:20000])  # first N rows, capped below anyway
    return text[:max_chars]


def main():
    print("=" * 70)
    print("HSSLM STREAMING SMOKE -- real WT-103, surprise-gated, Organism vocab")
    print("=" * 70)

    print("\n[1] Loading WT-103 (offline, HF cache)...")
    text = load_wikitext103_train_text()
    print(f"    chars loaded: {len(text):,}")

    print("\n[2] Building Organism vocabulary (length_extrap_v2.build_vocab)...")
    vocab, stoi, unk, mask = build_vocab(text)
    total_ids = mask + 1  # 0..unk-1 words, unk, mask -> mask+1 total IDs
    print(f"    vocab words: {len(vocab)} (VOCAB_MAX={VOCAB_MAX})")
    print(f"    unk id: {unk}  mask id: {mask}  total ids: {total_ids}")
    assert total_ids == VOCAB_MAX + 2, f"expected {VOCAB_MAX}+2={VOCAB_MAX+2}, got {total_ids}"

    print("\n[3] Tokenizing...")
    ids = tokenize(text, stoi, unk)
    print(f"    tokens: {len(ids):,}")

    print("\n[4] Building HSSLM with vocab_size={} (weight-tied LM head)...".format(total_ids))
    config = HSSLMConfig()
    config.vocab_size = total_ids
    config.hierarchical = False  # flat mode: streaming.py doesn't pass boundaries
    config.dropout = 0.0  # deterministic-ish streaming; no dropout noise in the gate signal
    config.pad_token_id = mask  # no real pad token in this vocab; reuse mask id as ignore_index
    model = HSSLM(config)
    model.train()
    counts = model.get_parameter_count()
    print(f"    total params: {counts['total']:,}")

    opt = torch.optim.AdamW(model.parameters(), lr=6e-4, betas=(0.9, 0.95), weight_decay=0.1)
    streamer = HSSLMStreamer(
        model, opt, vocab_size=total_ids, pad_token_id=-1,  # -1: no padding in this stream, don't ignore anything
        grad_clip=1.0,
    )

    print("\n[5] Running 200 gated chunks (chunk_size=32)...")
    chunk_size = 32
    n_chunks = 200
    needed_tokens = n_chunks * chunk_size + 1
    if len(ids) < needed_tokens:
        raise RuntimeError(f"not enough tokens: have {len(ids)}, need {needed_tokens}")

    losses = []
    surprises = []
    gates = []
    for c in range(n_chunks):
        start = c * chunk_size
        chunk_ids = ids[start:start + chunk_size + 1]
        x = torch.tensor([chunk_ids[:-1]], dtype=torch.long)  # (1, chunk_size)
        y = torch.tensor([chunk_ids[1:]], dtype=torch.long)   # (1, chunk_size)

        surprise, gated, nll = streamer.step_gated(x, y)
        surprises.append(surprise)
        gates.append(gated)
        if gated:
            losses.append(surprise)

        if c % 20 == 0 or c == n_chunks - 1:
            stats = streamer.stats()
            print(
                f"    chunk {c:3d}: surprise={surprise:.4f} gated={gated!s:5s} "
                f"gate_rate={stats['gate_rate']:.3f} position={streamer.position}"
            )

    print("\n[6] SUMMARY")
    stats = streamer.stats()
    first10 = sum(surprises[:10]) / 10
    last10 = sum(surprises[-10:]) / 10
    gate_rate = stats["gate_rate"]
    print(f"    chunks processed:     {stats['n_chunks']}")
    print(f"    gated (trained) steps: {stats['n_bwd']}")
    print(f"    gate rate:             {gate_rate:.3f}")
    print(f"    grad tokens:            {stats['grad_tokens']:,}")
    print(f"    surprise first-10 avg: {first10:.4f}")
    print(f"    surprise last-10 avg:  {last10:.4f}")
    print(f"    all surprises finite:  {all(s == s and abs(s) != float('inf') for s in surprises)}")

    # Gate-rate sanity: with GATE_Q=0.75 and ignition warmup, expect the
    # post-ignition gate rate to trend toward ~(1-0.75)=25% as the rolling
    # window fills (spike-triggered: only chunks ABOVE the 75th percentile
    # of recent surprise get trained on).
    ok_finite = all(s == s and abs(s) != float("inf") for s in surprises)
    ok_progress = last10 <= first10 * 1.5  # loose bound: not exploding
    print(f"\nVERDICT: finite={'PASS' if ok_finite else 'FAIL'}  "
          f"no-explosion={'PASS' if ok_progress else 'FAIL'}  "
          f"gate_rate={gate_rate:.3f} (expect roughly near 0.25 post-warmup, band is descriptive not a hard gate)")


if __name__ == "__main__":
    main()
