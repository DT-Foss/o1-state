#!/usr/bin/env python3 -u
"""
CONSULT-BACK — the graph learns from being used. MVP-5, the last MVP
baustein: closes analysis/LIVE_CAUSAL_SPEC.md SS4's loop back into the
organism (stage 4, "Consult") and folds successful consults into
src/livecausal/evidence.py's UseLedger (SS3).

Mirrors src/closed_loop.py's MEASURED consultation pattern exactly, aimed
at a LIVE-CAUSAL graph instead of a static .causal Attic file:

    stream -> surprise spike (a GAP) -> query(key) on the live graph
    -> inject the best path's outcome text as tokens -> read the REAL
    continuation -> does surprise on the next tokens DROP, vs the same
    gap WITHOUT injection (identical fork, same pre-gap state)?

Honest control (closed_loop.py's own load-bearing comparison, unchanged):
IDENTICAL gap and IDENTICAL continuation, read TWICE from the SAME
pre-gap state -- once WITH the path injected, once WITHOUT. We own the
state, so the two runs are exactly comparable, no confound.

SECOND control (this file's addition, the falsifier the build brief
names): a RANDOM-path injection arm -- inject some OTHER edge's outcome
text (not the graph's best-evidence answer for this gap) from the same
pre-gap state, read the same continuation. If the real path's mean delta
does not clear the random path's, consultation is not doing anything the
graph's mere presence (any token injection) wouldn't already do.

Every consult whose delta > 0 (real arm) logs UseLedger.append_use for
every base edge cited in the injected path's derivation, seq += 1 per
edge use (SS3: use_count is a distinct number from evidence_count, never
merged, never wall-clock-timestamped).

Does NOT mutate the store's segments or the evidence ledger -- only
use.ledger grows (per the build brief: "NICHT den Store mutieren außer
use.ledger").

Usage:
  python3 src/livecausal/consult_run.py --smoke
  python3 src/livecausal/consult_run.py --store results/p72_store_run1 --text-file corpus.txt
  python3 src/livecausal/consult_run.py --store results/p72_store_run1 --source wt103
"""
import argparse
import json
import os
import random
import re
import sys
import time

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SRC = os.path.join(REPO_ROOT, "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from livecausal.infer import LiveGraph  # noqa: E402
from livecausal.evidence import EvidenceLedger, UseLedger  # noqa: E402

_WORD_RE = re.compile(r"[a-zA-Z]{2,}")


# ─────────────────────────────────────────────────────────────────────────
#  Token-level read/advance, straight from closed_loop.py (unchanged
#  mechanics -- same no-grad forward, same state-fork discipline).
# ─────────────────────────────────────────────────────────────────────────
def _clone_state(states):
    return None if states is None else [s.clone() for s in states]


def _read(model, token_ids, states, dev, torch_mod):
    """Run token_ids through the model from `states`, return (per-token
    surprise list, new states). surprise[i] = NLL of predicting
    token_ids[i+1] given everything up to i. No-grad (this is inference-
    only scoring, mirrors closed_loop.py's @torch.no_grad() decorator --
    inlined as a context manager since torch is passed in, not imported
    at module scope)."""
    import torch.nn.functional as F

    with torch_mod.no_grad():
        if len(token_ids) < 2:
            x = torch_mod.tensor(token_ids, dtype=torch_mod.long).unsqueeze(0).to(dev)
            _, st = model(x, states)
            return [], st
        x = torch_mod.tensor(token_ids[:-1], dtype=torch_mod.long).unsqueeze(0).to(dev)
        y = torch_mod.tensor(token_ids[1:], dtype=torch_mod.long).unsqueeze(0).to(dev)
        logits, st = model(x, states)
        nll = F.cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1), reduction="none")
        return [float(v) for v in nll], st


def _advance(model, token_ids, states, dev, torch_mod):
    """Push tokens through to update the state, no scoring (used to
    inject a path). No-grad, same reasoning as _read."""
    if not token_ids:
        return states
    with torch_mod.no_grad():
        x = torch_mod.tensor(token_ids, dtype=torch_mod.long).unsqueeze(0).to(dev)
        _, st = model(x, states)
        return st


# ─────────────────────────────────────────────────────────────────────────
#  Word-level stream sources (token-by-token control, like closed_loop.py
#  -- NOT the chunk-batched ChunkFeeder path the builder loop uses; a
#  consultation needs to fork the state at an arbitrary single-word
#  position, which a fixed chunk boundary would not allow).
# ─────────────────────────────────────────────────────────────────────────
def load_text_file_words(path):
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        text = f.read()
    return _WORD_RE.findall(text.lower())


def load_wt103_words(stoi, unk, n_words, seed=42):
    """Streams WT-103 (via curator_yield_run's own HFStreamWithText, so
    this uses the SAME verbatim-text mechanism the builder's online path
    already relies on) and returns the first n_words words, lowercased."""
    import curator_yield_run as cyr  # noqa: E402

    stream = cyr.HFStreamWithText("wt103", stoi, unk)
    words = []
    while len(words) < n_words:
        toks, segs = stream.next_block()
        for seg in segs:
            words.extend(_WORD_RE.findall(seg.lower()))
    return words[:n_words]


# ─────────────────────────────────────────────────────────────────────────
#  Path selection: the graph's best-evidence outgoing edge for a gap key,
#  and its outcome TEXT (loaded from the citing record, not just the key
#  -- the injected tokens must be real prose the model can score, exactly
#  like closed_loop.py injects the attic's path TEXT, not its key).
# ─────────────────────────────────────────────────────────────────────────
def _record_for_citation(graph, sha, idx):
    for seg_sha, rec_idx, record in graph.store.iter_records(sha):
        if rec_idx == idx:
            return record
    return None


def best_edge_for_key(graph, evidence_ledger, valid_segments, key):
    """The graph's best-evidence outgoing edge from `key`: among base +
    inferred edges from query(key), pick the one with the highest
    evidence_count on its (from_key, to_key) pair (ties broken by lower
    depth, i.e. more direct evidence first, then by to_key for
    determinism). Returns (edge_dict, outcome_text) or (None, None) if
    key has no outgoing edges or no record text can be recovered."""
    edges = graph.query(key)
    if not edges:
        return None, None

    def score(edge):
        ek = (edge["from_key"], edge["to_key"])
        ec = evidence_ledger.evidence_count(ek, valid_segments)
        depth = edge.get("depth", 1)
        return (-ec, depth, edge["to_key"])

    edges_sorted = sorted(edges, key=score)
    for edge in edges_sorted:
        sha, idx = edge["derivation"][-1]  # last hop's citing record carries this edge's own outcome text
        record = _record_for_citation(graph, sha, idx)
        if record is not None and record.get("outcome"):
            return edge, record["outcome"]
    return None, None


def random_edge_for_key(graph, valid_segments, exclude_key, rng):
    """A random OTHER base edge in the graph (not `exclude_key`'s own
    edges), for the random-injection control arm. Returns (edge_key_tuple,
    outcome_text) or (None, None) if the graph has no other edges at all."""
    candidates = []
    for from_key, targets in graph._base_edges.items():
        if from_key == exclude_key:
            continue
        for to_key, citations in targets.items():
            candidates.append((from_key, to_key, citations[0]))
    if not candidates:
        return None, None
    from_key, to_key, (sha, idx) = rng.choice(candidates)
    record = _record_for_citation(graph, sha, idx)
    if record is None or not record.get("outcome"):
        return None, None
    return (from_key, to_key), record["outcome"]


def calibrate_surprise_threshold(words, stoi, unk, model, dev, torch_mod, quantile=0.9, n_calib_words=500):
    """Measures the model's own per-token surprise distribution over the
    first n_calib_words of `words` (from a fresh state, exactly how
    run_consult itself reads), and returns the given quantile as the gap
    threshold. Fixes closed_loop.py's hardcoded 9.0, which assumed a
    specific model/vocab pairing this loop's smaller configurations do
    not share (see --surprise-thresh's help text)."""
    calib_words = words[:n_calib_words]
    ids = [stoi.get(w, unk) for w in calib_words]
    if len(ids) < 2:
        return 1.0  # degenerate corpus; any positive threshold is arbitrary here
    surprises, _ = _read(model, ids, None, dev, torch_mod)
    if not surprises:
        return 1.0
    surprises_sorted = sorted(surprises)
    idx = min(len(surprises_sorted) - 1, int(quantile * (len(surprises_sorted) - 1)))
    return surprises_sorted[idx]


# ─────────────────────────────────────────────────────────────────────────
#  Main consult loop
# ─────────────────────────────────────────────────────────────────────────
def run_consult(
    graph,
    evidence_ledger,
    use_ledger,
    words,
    stoi,
    unk,
    model,
    dev,
    torch_mod,
    surprise_thresh=9.0,
    lookahead=12,
    max_gaps=40,
    seed=60,
):
    """The loop itself, factored out of main() so tests can drive it
    directly. Returns the result dict (also the JSON payload's shape)."""
    rng = random.Random(seed)
    ids = [stoi.get(w, unk) for w in words]
    valid_segments = graph.store.segments()

    results = []
    n_use_entries = 0
    use_seq = use_ledger.max_seq()

    states = None
    i = 0
    L = lookahead
    while i < len(ids) - L - 1 and len(results) < max_gaps:
        gap_surprise, _ = _read(model, ids[i:i + 2], _clone_state(states), dev, torch_mod)
        w = words[i + 1] if i + 1 < len(words) else ""
        is_gap = bool(gap_surprise) and gap_surprise[0] > surprise_thresh
        edge, outcome_text = (None, None)
        if is_gap:
            edge, outcome_text = best_edge_for_key(graph, evidence_ledger, valid_segments, w)
            is_gap = is_gap and edge is not None

        if not is_gap:
            states = _advance(model, ids[i:i + 1], states, dev, torch_mod)
            i += 1
            continue

        pre_state = _clone_state(states)
        cont = ids[i + 1:i + 1 + L]

        s_without, _ = _read(model, [ids[i]] + cont, _clone_state(pre_state), dev, torch_mod)

        real_path_ids = [stoi.get(t, unk) for t in _WORD_RE.findall(outcome_text.lower())]
        st_after_real = _advance(model, real_path_ids, _clone_state(pre_state), dev, torch_mod)
        s_with_real, _ = _read(model, [ids[i]] + cont, st_after_real, dev, torch_mod)

        rand_key, rand_text = random_edge_for_key(graph, valid_segments, w, rng)
        s_with_random = None
        if rand_text is not None:
            rand_path_ids = [stoi.get(t, unk) for t in _WORD_RE.findall(rand_text.lower())]
            st_after_rand = _advance(model, rand_path_ids, _clone_state(pre_state), dev, torch_mod)
            s_with_random, _ = _read(model, [ids[i]] + cont, st_after_rand, dev, torch_mod)

        avg_without = sum(s_without) / max(1, len(s_without))
        avg_with_real = sum(s_with_real) / max(1, len(s_with_real))
        # Round BEFORE the > 0 decision, not after: the decision to log a
        # use entry must agree with the drop_real value the JSON report
        # actually shows, or a reader sees "drop_real: 0.0" next to a
        # populated used_edges list and cannot reconcile the two (caught
        # by test_use_ledger_only_grows_on_positive_delta -- a raw
        # ~0.00004 drop rounded display-side to 0.0 while still counting
        # as "helped" internally).
        drop_real = round(avg_without - avg_with_real, 4)

        avg_with_random = None
        drop_random = None
        if s_with_random is not None:
            avg_with_random = sum(s_with_random) / max(1, len(s_with_random))
            drop_random = round(avg_without - avg_with_random, 4)

        edge_key = (edge["from_key"], edge["to_key"])
        used_edges = []
        if drop_real > 0:
            # SS2.3-mirrored: log a use for every base edge in the
            # injected path's derivation (an inferred edge's own hops),
            # not just the top-level edge_key -- the graph learns which
            # BASE facts actually helped, same granularity infer.py's
            # own contested-propagation already reasons at.
            hop_keys = graph.edge_keys_for_derivation(edge["derivation"]) if edge["derivation"] else [edge_key]
            for hop_key in hop_keys:
                use_seq += 1
                sha, idx = graph.base_edge_citations(*hop_key)[0] if graph.base_edge_citations(*hop_key) else edge["derivation"][0]
                use_ledger.append_use(hop_key, use_seq, sha, idx)
                used_edges.append(list(hop_key))
                n_use_entries += 1

        results.append({
            "gap": w,
            "gap_surprise": round(gap_surprise[0], 4),
            "edge": {"from_key": edge_key[0], "to_key": edge_key[1], "depth": edge.get("depth", 1)},
            "cont_surprise_without": round(avg_without, 4),
            "cont_surprise_with_real_path": round(avg_with_real, 4),
            "drop_real": drop_real,  # already rounded above, before the >0 decision
            "cont_surprise_with_random_path": round(avg_with_random, 4) if avg_with_random is not None else None,
            "drop_random": drop_random,  # already rounded above
            "used_edges": used_edges,
        })

        states = _advance(model, ids[i:i + 1], states, dev, torch_mod)
        i += 1

    drops_real = [r["drop_real"] for r in results]
    drops_random = [r["drop_random"] for r in results if r["drop_random"] is not None]
    mean_drop_real = sum(drops_real) / max(1, len(drops_real))
    mean_drop_random = (sum(drops_random) / len(drops_random)) if drops_random else None
    n_helped_real = sum(1 for d in drops_real if d > 0)

    # Top-5 most-used edges by use_count, per the build brief's output spec.
    edge_use_counts = {}
    for r in results:
        for ek in r["used_edges"]:
            key = tuple(ek)
            edge_use_counts[key] = use_ledger.use_count(key, valid_segments)
    top5 = sorted(edge_use_counts.items(), key=lambda kv: (-kv[1], kv[0]))[:5]

    return {
        "n_spikes": sum(1 for _ in results) + 0,  # gaps that cleared the surprise threshold AND had a queryable edge
        "n_consults": len(results),
        "mean_delta_real": round(mean_drop_real, 4),
        "mean_delta_random": round(mean_drop_random, 4) if mean_drop_random is not None else None,
        "n_helped_real": n_helped_real,
        "n_use_entries": n_use_entries,
        "top5_used_edges": [{"edge_key": list(k), "use_count": v} for k, v in top5],
        "results": results,
    }


# ─────────────────────────────────────────────────────────────────────────
#  Smoke corpus: causal chains a trained-from-scratch tiny model + the
#  smoke graph can plausibly show a surprise drop on -- reuses
#  builder_run.generate_smoke_corpus's chain shape so the injected paths
#  are the SAME items the store was built from.
# ─────────────────────────────────────────────────────────────────────────
def generate_vocab_aware_smoke_corpus(path, stoi, seed=42, n_sentences=200, chain_len=4):
    """Like builder_run.generate_smoke_corpus, but the chain "entity" words
    are drawn from the model's OWN in-vocabulary word list, not synthetic
    itemXX strings. builder_run's generator deliberately uses OOV words
    (to test the extractor against fresh entities); consult_run needs the
    OPPOSITE property -- an injected path's outcome text must be a token
    the model actually distinguishes, or a real-vs-random comparison is
    comparing two `unk` injections (caught running the first smoke pass:
    ALL 'itemXX' words hashed to the identical unk id, making the real and
    random arms indistinguishable by construction, not by finding-- see
    the build report). Draws distinct, rare-but-in-vocabulary words
    (indices 500..3000 of the 5000-word list: common enough to have a
    stable embedding, rare enough not to already carry strong bigram
    priors that would dominate the injection signal) as the chain nodes.
    Returns the list of chains (as key lists), same shape as the OOV
    generator, so callers don't need to branch on which corpus flavor ran.
    """
    rng = random.Random(seed)
    # Deterministic pool: words ranked 500..3000 by token id (stoi's ids
    # are frequency-ranked by build_vocab) -- common enough for a stable
    # embedding, rare enough to not already carry dominant bigram priors.
    by_id = sorted(stoi.keys(), key=lambda w: stoi[w])
    vocab_pool = by_id[500:3000] if len(by_id) >= 3000 else by_id

    chains = []
    sentences = []
    used = set()

    def _fresh_word():
        # Sample without replacement from the pool (falls back to
        # with-replacement once exhausted -- a smoke corpus can outgrow a
        # 2500-word pool, and a repeated entity word is still a valid,
        # in-vocabulary token, just not a fresh one).
        available = [w for w in vocab_pool if w not in used]
        w = rng.choice(available) if available else rng.choice(vocab_pool)
        used.add(w)
        return w

    n_chains = max(1, n_sentences // (chain_len + 2))
    for _ in range(n_chains):
        keys = [_fresh_word() for _ in range(chain_len + 1)]
        chains.append(keys)
        for i in range(chain_len):
            sentences.append("{} causes {}.".format(keys[i], keys[i + 1]))
        sentences.append("the weather was {} today.".format(_fresh_word()))
        sentences.append("a report about {} was published.".format(_fresh_word()))

    while len(sentences) < n_sentences:
        sentences.append("nothing {} happened here.".format(_fresh_word()))

    rng.shuffle(sentences)
    text = " ".join(sentences[:n_sentences])
    with open(path, "w", encoding="utf-8") as f:
        for _ in range(40):
            f.write(text)
            f.write(" ")
    return chains


def generate_smoke_store(store_dir, corpus_path, seed=42):
    """Builds a small smoke LiveGraph (fake-extractor causal chains, using
    the vocab-aware corpus above -- see its docstring for why) so
    consult_run has something to consult against without needing
    curator_yield_run or network."""
    from livecausal.builder_run import (
        TextFileStream,
        fake_extractor,
        run_builder,
        stream_windows,
    )
    import torch
    import portable_organism as po

    torch.set_num_threads(1)
    po.D_MODEL, po.BATCH, po.CHUNK = 64, 4, 32
    po.GATE_Q, po.GATE_WINDOW, po.MIN_WINDOW, po.IGNITION_CHUNKS = 0.75, 50, 10, 5

    vocab, stoi, unk, mask, val_ids = po.get_vocab()
    chains = generate_vocab_aware_smoke_corpus(corpus_path, stoi, seed=seed)
    V = len(vocab)
    torch.manual_seed(seed)
    organism = po.Organism("consult-smoke-builder", V, mask, seed=seed)
    stream = TextFileStream(corpus_path, stoi, unk)
    feeder = po.ChunkFeeder(stream, po.BATCH, po.CHUNK)
    window_iter = stream_windows(organism, stream, feeder, window_tokens=32)

    status_path = os.path.join(store_dir, "..", "consult_smoke_builder_status.json")
    metrics_path = os.path.join(store_dir, "..", "consult_smoke_builder_metrics.jsonl")
    graph, metrics = run_builder(
        store_dir, status_path, metrics_path,
        window_iter, fake_extractor, windows_per_segment=5,
        max_windows=60, print_every=1000, stream=stream,
    )
    return graph, chains


def main():
    ap = argparse.ArgumentParser(description="LIVE-CAUSAL consult-back + use-reinforcement (MVP-5)")
    ap.add_argument("--store", default=None, help="LiveStore directory to consult (required unless --smoke)")
    ap.add_argument("--text-file", default=None, help="local text file as the reading stream (offline path)")
    ap.add_argument("--source", default=None, choices=("c4", "wt103"), help="HF corpus (needs network)")
    ap.add_argument("--n-words", type=int, default=4000, help="--source path only: words to pull from the stream")
    ap.add_argument("--surprise-thresh", type=float, default=None,
                    help="fixed gap threshold; if omitted, calibrated AFTER warmup as a quantile "
                         "over the consult-portion's own surprise distribution (closed_loop.py's "
                         "fixed 9.0 assumed a specific model/vocab combination this loop's smaller "
                         "smoke config does not share -- a trained-but-tiny d_model=64 model's "
                         "surprise ceiling sits closer to ~2, not ~9)")
    ap.add_argument("--surprise-quantile", type=float, default=0.9,
                    help="quantile used for auto-calibration when --surprise-thresh is omitted")
    ap.add_argument("--lookahead", type=int, default=12)
    ap.add_argument("--max-gaps", type=int, default=40)
    ap.add_argument("--seed", type=int, default=60)
    ap.add_argument("--d-model", type=int, default=64)
    ap.add_argument("--warmup-chunks", type=int, default=200,
                    help="chunks of gradient training on the stream's PREFIX before the consult "
                         "phase starts (a fresh, never-trained model has no representations for "
                         "consultation to act on -- an in-flight surprise drop needs a model that "
                         "already predicts SOMETHING sensibly; 0 disables warmup, reproducing the "
                         "untrained-model condition this flag was added to fix)")
    ap.add_argument("--warmup-batch", type=int, default=4)
    ap.add_argument("--warmup-chunk-size", type=int, default=32)
    ap.add_argument("--out", default=os.path.join(REPO_ROOT, "results", "livecausal_consult.json"))
    ap.add_argument("--smoke", action="store_true", help="build a tiny smoke store + demo corpus, run offline")
    args = ap.parse_args()

    smoke_dir = None
    if args.smoke:
        import tempfile

        smoke_dir = tempfile.mkdtemp(prefix="livecausal_consult_smoke_")
        args.store = os.path.join(smoke_dir, "store")
        args.text_file = os.path.join(smoke_dir, "corpus.txt")
        args.max_gaps = args.max_gaps or 20
        args.surprise_thresh = args.surprise_thresh
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        os.environ.setdefault("HF_DATASETS_OFFLINE", "1")

    if not args.store:
        raise SystemExit("--store is required (or pass --smoke)")

    if args.text_file:
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        os.environ.setdefault("HF_DATASETS_OFFLINE", "1")

    import torch  # noqa: E402
    torch.set_num_threads(1)
    import portable_organism as po  # noqa: E402

    po.D_MODEL = args.d_model
    vocab, stoi, unk, mask, val_ids = po.get_vocab()
    V = len(vocab)

    if args.smoke:
        print("[consult] building smoke store + corpus...", flush=True)
        generate_smoke_store(args.store, args.text_file, seed=args.seed)

    graph = LiveGraph(args.store)
    evidence_ledger = EvidenceLedger(args.store)
    use_ledger = UseLedger(args.store)

    if args.text_file:
        words = load_text_file_words(args.text_file)
    elif args.source:
        words = load_wt103_words(stoi, unk, args.n_words, seed=args.seed)
    else:
        raise SystemExit("one of --text-file, --source, or --smoke is required")

    torch.manual_seed(args.seed)
    po.D_MODEL = args.d_model  # re-assert: get_vocab() above may have been called under a
    # different po.D_MODEL from a prior --smoke build step; the READER organism below must
    # use the d_model this run actually asked for.
    organism = po.Organism("consult-reader", V, mask, seed=args.seed)
    model = organism.model
    dev = next(model.parameters()).device

    warmup_words, consult_words = words, words
    if args.warmup_chunks > 0:
        n_warmup_tokens = args.warmup_chunks * args.warmup_batch * args.warmup_chunk_size
        split = min(len(words) // 2, n_warmup_tokens // 1)  # word-level proxy for token count (>=1 token/word here)
        warmup_words, consult_words = words[:split], words[split:]
        if not consult_words:
            consult_words = words  # fallback: corpus too short to split, reuse everything

        print("[consult] warmup: {} chunks on {} prefix words...".format(args.warmup_chunks, len(warmup_words)), flush=True)
        po.BATCH, po.CHUNK = args.warmup_batch, args.warmup_chunk_size
        warmup_ids = [stoi.get(w, unk) for w in warmup_words]

        class _ListStream:
            """Minimal next_block() source over a fixed id list, wrapping on
            exhaustion -- enough for ChunkFeeder, nothing else needed here."""

            def __init__(self, ids, block=4096):
                self.ids, self.block, self._cursor = ids, block, 0
                self.docs = 0

            def next_block(self):
                out = []
                while len(out) < self.block:
                    if self._cursor >= len(self.ids):
                        self._cursor = 0
                        self.docs += 1
                    take = min(self.block - len(out), len(self.ids) - self._cursor)
                    out.extend(self.ids[self._cursor:self._cursor + take])
                    self._cursor += take
                return out

        warmup_stream = _ListStream(warmup_ids)
        warmup_feeder = po.ChunkFeeder(warmup_stream, po.BATCH, po.CHUNK)
        for _ in range(args.warmup_chunks):
            x, y = warmup_feeder.next_xy()
            organism.step_gated(x, y, ignition=True)  # ignition=True: always trains during warmup, gate untouched by consult concerns
        model.eval()
        print("[consult] warmup done, {} grad steps ({} tokens)".format(
            organism.n_bwd, organism.grad_tokens), flush=True)
    else:
        model.eval()

    surprise_thresh = args.surprise_thresh
    if surprise_thresh is None:
        surprise_thresh = calibrate_surprise_threshold(
            consult_words, stoi, unk, model, dev, torch, quantile=args.surprise_quantile
        )
        print("[consult] auto-calibrated surprise_thresh={:.4f} (quantile={})".format(
            surprise_thresh, args.surprise_quantile), flush=True)

    print("[consult] store={} words={} gaps<={}".format(args.store, len(consult_words), args.max_gaps), flush=True)
    t0 = time.time()
    result = run_consult(
        graph, evidence_ledger, use_ledger, consult_words, stoi, unk, model, dev, torch,
        surprise_thresh=surprise_thresh,
        lookahead=args.lookahead,
        max_gaps=args.max_gaps,
        seed=args.seed,
    )
    elapsed = time.time() - t0

    payload = {
        "store": args.store,
        "text_file": args.text_file,
        "source": args.source,
        "surprise_thresh": round(surprise_thresh, 4),
        "surprise_thresh_was_calibrated": args.surprise_thresh is None,
        "lookahead": args.lookahead,
        "max_gaps": args.max_gaps,
        "seed": args.seed,
        "warmup_chunks": args.warmup_chunks,
        "n_warmup_words": len(warmup_words) if args.warmup_chunks > 0 else 0,
        "n_consult_words": len(consult_words),
        "n_warmup_grad_steps": organism.n_bwd if args.warmup_chunks > 0 else 0,
        "elapsed_seconds": round(elapsed, 2),
        **result,
    }
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    print("=" * 74)
    print("CONSULT-BACK: does querying the live graph lower post-gap surprise?")
    for r in result["results"][:12]:
        arrow = "v" if r["drop_real"] > 0 else "^"
        rr = r["drop_random"]
        print("  gap {!r} (s={:.1f}): without {:.2f} -> with(real) {:.2f} {} {:+.3f}"
              "  | with(random) {}".format(
                  r["gap"], r["gap_surprise"], r["cont_surprise_without"],
                  r["cont_surprise_with_real_path"], arrow, r["drop_real"],
                  "{:+.3f}".format(rr) if rr is not None else "n/a",
              ))
    print()
    print("n_spikes={} n_consults={} n_helped_real={}/{}".format(
        result["n_spikes"], result["n_consults"], result["n_helped_real"], result["n_consults"]))
    print("mean_delta_real={:+.4f}  mean_delta_random={}".format(
        result["mean_delta_real"],
        "{:+.4f}".format(result["mean_delta_random"]) if result["mean_delta_random"] is not None else "n/a",
    ))
    print("n_use_entries={}".format(result["n_use_entries"]))
    print("top5 used edges:")
    for e in result["top5_used_edges"]:
        print("  {} -> {}  use_count={}".format(e["edge_key"][0], e["edge_key"][1], e["use_count"]))
    print("wrote {}".format(args.out))

    if smoke_dir:
        import shutil
        shutil.rmtree(smoke_dir, ignore_errors=True)


if __name__ == "__main__":
    main()
