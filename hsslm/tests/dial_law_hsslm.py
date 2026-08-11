"""
P81 Dial-Law harness: HSSLM arm + Organism control arm.

The Organism's law (measured elsewhere in this project) is: the surprise-gate
rate converges to approximately (1 - q), invariant across width/seed/age, once
past ignition warmup and once the rolling window is full. This script measures
the SAME question for two architectures under IDENTICAL streaming/gating
mechanics and IDENTICAL WT-103 text + vocabulary:

  --arch hsslm      HSSLMStreamer (hsslm/neural/streaming.py), HSSLM's own
                     port of the gate mechanics, on the S6 core.
  --arch organism   src/portable_organism.py's Organism.step_gated used
                     DIRECTLY (not reimplemented) -- the original gate
                     mechanics, on StreamingNoPELM (d_model configurable via
                     po.D_MODEL, default 128 per the lead's spec).

Both arms use the same q grid, same WT-103 text, same
length_extrap_v2.build_vocab vocabulary, same GATE_WINDOW/MIN_WINDOW/
IGNITION_CHUNKS, same chunk_size, same chunk count -- so any curve difference
is attributable to the architecture (S6+hierarchy vs NoPE-transformer-scan),
not to the harness.

IMPORTANT re: portable_organism.py is NOT modified. GATE_Q, GATE_WINDOW,
MIN_WINDOW, IGNITION_CHUNKS, D_MODEL, SEED are module-level constants that
Organism.step_gated and Organism.__init__ read directly from the
portable_organism module's namespace at call time (not as parameters) --
so this harness sets them as module attributes (`po.GATE_Q = q`, etc.)
before constructing/using an Organism, exactly the way editing a config
file would, just done at import-time instead. No source line in
portable_organism.py is touched.

Usage:
    OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
    TOKENIZERS_PARALLELISM=false HF_DATASETS_OFFLINE=1 HF_HUB_OFFLINE=1 \
    python3 dial_law_hsslm.py --arch hsslm --q 0.5 0.65 0.75 0.85 0.9 \
        --chunks 400 --seeds 44 45 --out dial_law_hsslm_scored.json

    python3 dial_law_hsslm.py --arch organism --q 0.5 0.65 0.75 0.85 0.9 \
        --chunks 400 --seeds 44 45 --out dial_law_organism_scored.json
"""
import argparse
import json
import os
import sys
import time

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
os.environ.setdefault("HF_HUB_OFFLINE", "1")

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, "/Users/bhkmie/Documents/Forschung/O1_juli/src")

import torch

from length_extrap_v2 import build_vocab, tokenize, VOCAB_MAX

_TEXT_CACHE = {}
_VOCAB_CACHE = {}


def load_wikitext103_train_text(max_chars: int = 20_000_000) -> str:
    if "text" in _TEXT_CACHE:
        return _TEXT_CACHE["text"]
    from datasets import load_dataset
    ds = load_dataset("Salesforce/wikitext", "wikitext-103-raw-v1")
    text = "\n\n".join(ds["train"]["text"][:20000])[:max_chars]
    _TEXT_CACHE["text"] = text
    return text


def get_vocab_and_ids(text: str):
    if "ids" in _VOCAB_CACHE:
        return _VOCAB_CACHE["vocab"], _VOCAB_CACHE["stoi"], _VOCAB_CACHE["unk"], \
            _VOCAB_CACHE["mask"], _VOCAB_CACHE["ids"]
    vocab, stoi, unk, mask = build_vocab(text)
    ids = tokenize(text, stoi, unk)
    _VOCAB_CACHE.update(vocab=vocab, stoi=stoi, unk=unk, mask=mask, ids=ids)
    return vocab, stoi, unk, mask, ids


# ===========================================================================
# HSSLM arm
# ===========================================================================

def run_one_hsslm(q, chunks, seed, chunk_size, ids, total_ids,
                   gate_window, min_window, ignition_chunks) -> dict:
    from neural.model import HSSLM
    from neural.config import HSSLMConfig
    from neural.streaming import HSSLMStreamer

    torch.manual_seed(seed)
    torch.set_num_threads(1)

    config = HSSLMConfig()
    config.vocab_size = total_ids
    config.hierarchical = False
    config.dropout = 0.0
    config.pad_token_id = -1
    model = HSSLM(config)
    model.train()

    opt = torch.optim.AdamW(model.parameters(), lr=6e-4, betas=(0.9, 0.95), weight_decay=0.1)
    streamer = HSSLMStreamer(
        model, opt, vocab_size=total_ids, pad_token_id=-1,
        gate_q=q, gate_window=gate_window, min_window=min_window,
        ignition_chunks=ignition_chunks, grad_clip=1.0,
    )

    needed_tokens = chunks * chunk_size + 1
    if len(ids) < needed_tokens:
        raise RuntimeError(f"not enough tokens: have {len(ids)}, need {needed_tokens}")

    warmup_end = max(ignition_chunks, min_window)

    surprises, gates = [], []
    t0 = time.time()
    for c in range(chunks):
        start = c * chunk_size
        chunk_ids = ids[start:start + chunk_size + 1]
        x = torch.tensor([chunk_ids[:-1]], dtype=torch.long)
        y = torch.tensor([chunk_ids[1:]], dtype=torch.long)
        surprise, gated, _ = streamer.step_gated(x, y)
        surprises.append(surprise)
        gates.append(gated)
    elapsed = time.time() - t0

    return _summarize(q, seed, chunks, chunk_size, warmup_end, surprises, gates,
                       elapsed, arch="hsslm", extra={"n_params": sum(p.numel() for p in model.parameters())})


# ===========================================================================
# Organism control arm -- uses src/portable_organism.py's Organism.step_gated
# DIRECTLY. portable_organism.py itself is NOT modified; module-level policy
# constants (GATE_Q, GATE_WINDOW, MIN_WINDOW, IGNITION_CHUNKS, D_MODEL) are
# set as module attributes before constructing/using the Organism, since
# step_gated and __init__ read them from the module namespace at call time.
# ===========================================================================

def run_one_organism(q, chunks, seed, chunk_size, ids, total_ids, mask_id,
                      gate_window, min_window, ignition_chunks,
                      d_model, batch) -> dict:
    import portable_organism as po  # local import: this module forces CPU +
    # single-threaded torch as an import-time side effect (see its own
    # torch.backends.mps.is_available = lambda: False / set_num_threads(1)) --
    # importing it only when this arm actually runs keeps that side effect
    # scoped to when we asked for it.

    # Set the module-level policy constants this run needs. Organism.__init__
    # reads D_MODEL/N_LAYERS/N_HEADS/CHUNK; step_gated reads GATE_Q/
    # IGNITION_CHUNKS/(MIN_WINDOW via window-fill check); Organism.__init__
    # builds self.window = deque(maxlen=GATE_WINDOW).
    po.D_MODEL = d_model
    po.CHUNK = chunk_size
    po.GATE_Q = q
    po.GATE_WINDOW = gate_window
    po.MIN_WINDOW = min_window
    po.IGNITION_CHUNKS = ignition_chunks

    organism = po.Organism("dial_law_control", total_ids, mask_id, seed=seed)
    organism.model.train()

    needed_tokens = batch * chunks * chunk_size + batch  # rough upper bound, batch rows drawn contiguously below
    warmup_end = max(ignition_chunks, min_window)

    # Build batch rows the same way the Organism's own training loop draws
    # chunks: B independent contiguous windows advancing together, chunk_size
    # tokens each, +1 for the next-token target. To keep this comparable to
    # the HSSLM arm (which streams B=1 sequentially through the WHOLE token
    # stream), we use batch=1 unless the Organism's own path needs more --
    # StreamingNoPELM has no batch-size constraint of its own (it's just the
    # leading tensor dim), so batch=1 is valid and documented here as the
    # choice: makes rate/surprise directly comparable to the HSSLM arm's
    # single-stream measurement, not an artifact of averaging across B
    # parallel streams.
    B = batch
    assert B == 1, (
        "batch>1 not implemented in this harness -- Organism.step_gated "
        "accepts any leading batch dim, but comparing surprise across "
        "different B would confound architecture with batch averaging. "
        "Documented per lead's instruction: batch kept at 1."
    )

    total_needed = chunks * chunk_size + 1
    if len(ids) < total_needed:
        raise RuntimeError(f"not enough tokens: have {len(ids)}, need {total_needed}")

    surprises, gates = [], []
    t0 = time.time()
    for c in range(chunks):
        start = c * chunk_size
        chunk_ids = ids[start:start + chunk_size + 1]
        x = torch.tensor([chunk_ids[:-1]], dtype=torch.long)
        y = torch.tensor([chunk_ids[1:]], dtype=torch.long)
        surprise, gated, _ = organism.step_gated(x, y)
        surprises.append(surprise)
        gates.append(gated)
    elapsed = time.time() - t0

    return _summarize(q, seed, chunks, chunk_size, warmup_end, surprises, gates,
                       elapsed, arch="organism",
                       extra={"n_params": sum(p.numel() for p in organism.model.parameters()),
                              "d_model": d_model, "batch": B})


# ===========================================================================
# Shared summary
# ===========================================================================

def _summarize(q, seed, chunks, chunk_size, warmup_end, surprises, gates, elapsed, arch, extra=None):
    stationary_gates = gates[warmup_end:]
    warmup_gates = gates[:warmup_end]

    n_gated_total = sum(gates)
    n_gated_stationary = sum(stationary_gates)
    n_stationary = len(stationary_gates)

    rate_total = n_gated_total / len(gates) if gates else float("nan")
    rate_stationary = (
        n_gated_stationary / n_stationary if n_stationary > 0 else float("nan")
    )
    rate_warmup = (
        sum(warmup_gates) / len(warmup_gates) if warmup_gates else float("nan")
    )

    surprise_first10 = sum(surprises[:10]) / min(10, len(surprises))
    surprise_last10 = sum(surprises[-10:]) / min(10, len(surprises[-10:]))

    result = {
        "arch": arch,
        "q": q,
        "seed": seed,
        "chunks": chunks,
        "chunk_size": chunk_size,
        "warmup_end_chunk": warmup_end,
        "rate_total": rate_total,
        "rate_stationary": rate_stationary,
        "rate_warmup": rate_warmup,
        "n_gated": n_gated_total,
        "n_gated_stationary": n_gated_stationary,
        "n_stationary_chunks": n_stationary,
        "one_minus_q": 1.0 - q,
        "stationary_minus_expected": rate_stationary - (1.0 - q) if n_stationary > 0 else float("nan"),
        "surprise_first10": surprise_first10,
        "surprise_last10": surprise_last10,
        "elapsed_sec": elapsed,
    }
    if extra:
        result.update(extra)
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arch", choices=["hsslm", "organism"], default="hsslm")
    ap.add_argument("--q", type=float, nargs="+", default=[0.5, 0.65, 0.75, 0.85, 0.9])
    ap.add_argument("--chunks", type=int, default=400)
    ap.add_argument("--chunk-size", type=int, default=32)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--seeds", type=int, nargs="+", default=None)
    ap.add_argument("--gate-window", type=int, default=200)
    ap.add_argument("--min-window", type=int, default=30)
    ap.add_argument("--ignition-chunks", type=int, default=15)
    ap.add_argument("--d-model", type=int, default=128, help="organism arm only")
    ap.add_argument("--batch", type=int, default=1, help="organism arm only; must be 1, see run_one_organism")
    ap.add_argument("--out", type=str, default=None)
    args = ap.parse_args()

    seeds = args.seeds if args.seeds is not None else [args.seed]

    print("=" * 74)
    print(f"P81 DIAL-LAW HARNESS -- arch={args.arch}")
    print("=" * 74)
    print(f"q values:   {args.q}")
    print(f"chunks:     {args.chunks}  (chunk_size={args.chunk_size})")
    print(f"seeds:      {seeds}")
    print(f"gate params: window={args.gate_window} min_window={args.min_window} ignition={args.ignition_chunks}")
    print(f"warmup_end: max(ignition={args.ignition_chunks}, min_window={args.min_window}) = "
          f"{max(args.ignition_chunks, args.min_window)}")
    if args.arch == "organism":
        print(f"organism d_model={args.d_model}  batch={args.batch}")

    print("\n[1] Loading WT-103 + building Organism vocabulary (shared across all runs)...")
    text = load_wikitext103_train_text()
    vocab, stoi, unk, mask, ids = get_vocab_and_ids(text)
    total_ids = mask + 1
    print(f"    vocab words: {len(vocab)}  total ids: {total_ids}  tokens: {len(ids):,}")

    results = []
    for seed in seeds:
        for q in args.q:
            print(f"\n[RUN] arch={args.arch}  q={q}  seed={seed}  chunks={args.chunks} ...")
            if args.arch == "hsslm":
                r = run_one_hsslm(
                    q, args.chunks, seed, args.chunk_size, ids, total_ids,
                    args.gate_window, args.min_window, args.ignition_chunks,
                )
            else:
                r = run_one_organism(
                    q, args.chunks, seed, args.chunk_size, ids, total_ids, mask,
                    args.gate_window, args.min_window, args.ignition_chunks,
                    args.d_model, args.batch,
                )
            results.append(r)
            print(
                f"    rate_total={r['rate_total']:.3f}  "
                f"rate_stationary={r['rate_stationary']:.3f}  "
                f"1-q={r['one_minus_q']:.3f}  "
                f"delta={r['stationary_minus_expected']:+.3f}  "
                f"surprise {r['surprise_first10']:.3f}->{r['surprise_last10']:.3f}  "
                f"({r['elapsed_sec']:.1f}s)"
            )

    print("\n" + "=" * 74)
    print("SUMMARY TABLE")
    print("=" * 74)
    header = f"{'arch':>9} {'seed':>5} {'q':>6} {'1-q':>6} {'rate_stat':>10} {'delta':>8} {'rate_total':>10} {'n_gated':>8} {'surp_first10':>13} {'surp_last10':>12}"
    print(header)
    print("-" * len(header))
    for r in results:
        print(
            f"{r['arch']:>9} {r['seed']:>5} {r['q']:>6.2f} {r['one_minus_q']:>6.2f} "
            f"{r['rate_stationary']:>10.3f} {r['stationary_minus_expected']:>+8.3f} "
            f"{r['rate_total']:>10.3f} {r['n_gated']:>8d} "
            f"{r['surprise_first10']:>13.3f} {r['surprise_last10']:>12.3f}"
        )

    out_path = args.out or os.path.join(
        os.path.dirname(__file__), f"dial_law_{args.arch}_results.json"
    )
    with open(out_path, "w") as f:
        json.dump({
            "config": {
                "arch": args.arch,
                "q_values": args.q,
                "chunks": args.chunks,
                "chunk_size": args.chunk_size,
                "seeds": seeds,
                "gate_window": args.gate_window,
                "min_window": args.min_window,
                "ignition_chunks": args.ignition_chunks,
                "vocab_total_ids": total_ids,
                **({"d_model": args.d_model, "batch": args.batch} if args.arch == "organism" else {}),
            },
            "results": results,
        }, f, indent=2)
    print(f"\nOutput JSON written: {out_path}")


if __name__ == "__main__":
    main()
