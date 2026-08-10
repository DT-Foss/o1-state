#!/usr/bin/env python3 -u
"""
BUILDER LOOP v0 — the organism streams, the gate curates, the extractor
validates, the live graph grows. analysis/LIVE_CAUSAL_SPEC.md SS4 ("the
organism as builder"), stages 1-3 (curate / extract / fold) wired end to
end: no LLM anywhere in the loop.

    window source -> window text -> extract_validated -> LiveStore.append_segment
              -> LiveGraph.on_append

Wired against src/curator_yield_run.py, the REAL bridge (mvp3-p70's
built-and-scored MVP-4 CONTRACT, P70 measurement):
    extract_validated(text: str) -> list[dict]
        Validated triplets, each carrying trigger/mechanism/outcome and
        trigger_key/outcome_key (normalize_entity(trigger/outcome), set
        directly by curator_yield_run -- a fallback key-derivation stays
        in make_record() for any OTHER extractor wired in that doesn't
        set them, e.g. a stub in a test).
    iter_windows(...) -> generator of (tape_pos, window_text, surprise,
        gated), one per chunk, EVERY window (P70 policy: extraction is
        ungated -- see stream_windows()'s docstring for the same rule on
        the --text-file path). Used directly for the ONLINE corpus path
        (--source c4/wt103); resolve_extractor's lazy-import hook still
        lets tests stub extract_validated without curator_yield_run
        needing to exist for THOSE tests, but the online window source
        is no longer a placeholder this file re-implements -- see
        build_window_iterator.

Two corpus sources:
  --text-file PATH   a local text file streamed as an iterator of blocks
                      (F6: "the corpus is an iterator" -- this is the
                      offline-capable, no-HF-network path used for the
                      smoke test and for beast while DNS is down). Drives
                      TextFileStream + stream_windows() locally, UNCHANGED
                      by this bridge-integration pass.
  --source c4|wt103   delegates entirely to curator_yield_run.iter_windows
                      (verbatim source text via its own HFStreamWithText;
                      needs network / a warm HF cache).

Outputs (results/livecausal_builder_<tag>*):
  status.json    heartbeat (tmpfile + os.replace, pos_run.py pattern)
  metrics.jsonl  per-append-batch metrics record
  the graph itself lives in --store-dir (a src/livecausal LiveStore dir)

Usage:
  python3 src/livecausal/builder_run.py --smoke
  python3 src/livecausal/builder_run.py --text-file corpus.txt --store-dir results/my_graph
"""
import argparse
import json
import os
import random
import re
import sys
import tempfile
import time

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SRC = os.path.join(REPO_ROOT, "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from livecausal.infer import LiveGraph  # noqa: E402
from livecausal.store import LiveStore  # noqa: E402

_WORD_RE = re.compile(r"[a-zA-Z]+")


# ─────────────────────────────────────────────────────────────────────────
#  Corpus sources: both expose next_block() -> list[int] (tokenized) AND
#  raw_words(lo, hi) -> str (the SAME span, as original words joined by
#  space) so gated windows can be handed to the extractor as TEXT, not
#  token ids -- extract_validated's contract takes text, not ids.
# ─────────────────────────────────────────────────────────────────────────
class TextFileStream:
    """Reads a local text file once into memory, tokenizes it with the
    SAME word regex portable_organism/length_extrap_v2 use, and re-streams
    it as an iterator of token blocks (F6: the corpus is an iterator --
    this makes a static file behave like the C4Stream/HFStream sources
    everywhere this loop touches them). No network, no HF `datasets` call:
    this is the offline path (built for the beast-DNS-is-down constraint).

    Wraps on exhaustion (like C4Stream's doc-skip-to-0 restart), so a short
    file can still feed an arbitrarily long run -- `.docs` counts wraps.
    """

    def __init__(self, path, stoi, unk, block=8192):
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            text = f.read()
        self.words = []
        self.word_terminators = []  # the raw text immediately after each
        # word up to (and including) the next word's start -- carries
        # sentence punctuation (".", "!", "\n\n", ...) so raw text
        # reconstruction below does not silently fuse separate sentences
        # into one run-on line (the word regex itself strips all
        # punctuation, which would otherwise erase every sentence
        # boundary and let an extractor's pattern match across sentences
        # that were never causally connected -- caught by the smoke test).
        matches = list(_WORD_RE.finditer(text.lower()))
        for i, m in enumerate(matches):
            self.words.append(m.group(0))
            end = m.end()
            next_start = matches[i + 1].start() if i + 1 < len(matches) else len(text)
            self.word_terminators.append(text[end:next_start])
        if not self.words:
            raise ValueError("text file {} has no tokenizable words".format(path))
        self.stoi, self.unk, self.block = stoi, unk, block
        self.docs = 0          # wrap count, mirrors C4Stream.docs's role as a position stamp
        self.reconnects = 0    # kept for status.json symmetry with C4Stream; always 0 here
        self.pending = []
        self._cursor = 0       # word index into self.words (pre-wrap)

    def next_block(self):
        out = []
        while len(out) < self.block:
            if self._cursor >= len(self.words):
                self._cursor = 0
                self.docs += 1
            take = min(self.block - len(out), len(self.words) - self._cursor)
            chunk_words = self.words[self._cursor:self._cursor + take]
            out.extend(self.stoi.get(w, self.unk) for w in chunk_words)
            self._cursor += take
        return out

    def raw_text_for_tape_range(self, tape, lo, hi):
        """Best-effort raw-word recovery for tape[lo:hi]: since the tape is
        just token ids (post-tokenization, post-wrap), this re-derives the
        source words by matching the wrapped word sequence position, which
        is exact as long as `lo` is within one file-length of `hi`'s wrap
        count (true for any window smaller than the file). For a window
        that straddles a wrap boundary, words after the wrap are still
        correct (tokenization is wrap-invariant: self.words is fixed).
        Reconstructs the ORIGINAL inter-word text (including punctuation)
        via word_terminators, so sentence boundaries survive into the
        extractor's input instead of being silently fused."""
        n = len(self.words)
        if n == 0:
            return ""
        start = lo % n
        length = hi - lo
        idxs = [(start + i) % n for i in range(length)]
        parts = []
        for j, i in enumerate(idxs):
            parts.append(self.words[i])
            if j < len(idxs) - 1:
                parts.append(self.word_terminators[i])
        return "".join(parts)


# HFCorpusStream (the earlier online-source placeholder, token-id text
# fallback and all) is RETIRED: superseded by curator_yield_run.iter_windows
# (see build_window_iterator below), which carries verbatim source text via
# its own HFStreamWithText -- the placeholder-text gap this class used to
# flag no longer exists for the online path.


def build_window_iterator(args, organism_seed=42):
    """Returns (window_iter, stream_or_none) for the selected corpus source.

    --text-file: builds a TextFileStream + a fresh po.Organism/ChunkFeeder
        locally and drives stream_windows() over them (unchanged from
        before this bridge-integration pass, per the build brief: "the
        --text-file path stays unchanged"). stream_or_none is the
        TextFileStream (exposes .docs for the n_streamed_tape_pos metric).

    --source (c4/wt103): delegates ENTIRELY to
        curator_yield_run.iter_windows -- the MVP-4 CONTRACT function,
        which owns its own Organism/HFStreamWithText/ChunkFeederWithText
        internally and already yields the exact (tape_pos, window_text,
        surprise, gated) shape this loop needs, with VERBATIM source text
        (not a token-id placeholder -- iter_windows's HFStreamWithText
        keeps original-case text in lockstep with the token tape). This
        loop does not construct its own Organism/stream for this path
        anymore; stream_or_none is None (see run_builder's docstring for
        how the n_streamed_tape_pos metric degrades gracefully).
    """
    if args.text_file:
        import torch  # noqa: E402
        import portable_organism as po  # noqa: E402

        vocab, stoi, unk, mask, val_ids = po.get_vocab()
        V = len(vocab)
        torch.manual_seed(organism_seed)
        organism = po.Organism("builder", V, mask, seed=organism_seed)
        stream = TextFileStream(args.text_file, stoi, unk)
        feeder = po.ChunkFeeder(stream, args.batch, args.chunk_size)
        win_iter = stream_windows(organism, stream, feeder, args.window_tokens, tape_cap=args.tape_cap)
        return win_iter, stream

    import curator_yield_run  # noqa: E402

    win_iter = curator_yield_run.iter_windows(
        substrate=args.source,
        chunks=args.chunks if args.chunks else 10 ** 9,  # iter_windows needs a finite bound; effectively unbounded, max_windows/max_seconds in run_builder still cut it off
        window_tokens=args.window_tokens,
        d_model=args.d_model,
        batch=args.batch,
        chunk_size=args.chunk_size,
        seed=organism_seed,
        q=args.q,
        window=args.window,
        min_window=args.min_window,
        ignition_chunks=args.ignition_chunks,
    )
    return win_iter, None


# ─────────────────────────────────────────────────────────────────────────
#  Window generator: the FENCE the SPEC calls out -- SS4 stage 1 (curate).
#  Wraps po.Organism.step_gated (surprise_filter_run.py's live gate) to get
#  a per-chunk surprise value and gate decision.
#
#  POLICY UPDATE (P70 rescoring, MVP-3 -- "Builder-Politik: Extraktion
#  ungated, Gate kuratiert Storage"): P70 measured causal-structure yield
#  as EVENLY DISTRIBUTED across the stream (surprise-gated extraction
#  windows and random windows both land ~18 validated triplets/kilotoken,
#  the falsifier's bars of 1.3x/1.5x both missed) -- surprise-gating the
#  EXTRACTION step buys nothing. This generator therefore now yields EVERY
#  chunk's window, gated or not, carrying the gate's surprise/gated signal
#  alongside the text so a LATER stage (storage/dedup/memory -- P55/P58/P67
#  territory, not this generator) can curate on it instead. The organism
#  still streams gated (its own training gate is untouched -- step_gated
#  still decides whether to backprop on this chunk); only the WINDOW-YIELD
#  side stopped filtering.
# ─────────────────────────────────────────────────────────────────────────
def stream_windows(organism, stream, feeder, window_tokens, tape_cap=None):
    """Generator over (tape_pos: int, window_text: str, surprise: float,
    gated: bool) for EVERY chunk (ungated extraction, per the P70 policy
    update above). tape_pos is the tape offset of the window START (the
    chunk's first token) -- this IS the builder's doc_coord. window_text is
    window_tokens tokens' worth of ORIGINAL words (not the tokenized ids)
    recovered from the stream via raw_text_for_tape_range (TextFileStream
    supports this; this function is only called with a TextFileStream --
    the online --source path uses curator_yield_run.iter_windows instead,
    which has its own verbatim-text mechanism, see build_window_iterator).
    surprise is the chunk's mean NLL (the
    same value step_gated computes for its own rolling-quantile decision);
    gated is whether the organism's OWN training gate accepted this chunk
    for backprop -- carried through as a signal for the storage layer, not
    used here to filter which windows get yielded.

    tape_cap: if given, the generator's internal token tape is trimmed to
    the last `tape_cap` tokens after each chunk (keeps memory flat on long
    runs) -- callers needing wider windows than tape_cap should not use it.
    """
    tape = []
    tape_base = 0  # absolute position of tape[0] (after trimming)
    while True:
        x, y = feeder.next_xy()
        s, gated, nll = organism.step_gated(x, y)
        chunk_start = tape_base + len(tape)
        tape.extend(int(v) for v in x[0])  # lane 0 carries the tape, surprise_filter_run.py's convention
        lo, hi = chunk_start, chunk_start + window_tokens
        text = stream.raw_text_for_tape_range(tape, lo, hi)
        if text is None:
            # Fallback for sources without raw-text recovery: encode
            # the token ids as a placeholder string so the extractor
            # contract (text: str) is still satisfied. Flagged: this
            # degrades extraction quality for HF corpora until the
            # extractor is also handed the vocab to detokenize.
            text = " ".join(str(t) for t in x[0][:window_tokens].tolist())
        yield chunk_start, text, s, gated
        if tape_cap and len(tape) > tape_cap:
            drop = len(tape) - tape_cap
            tape = tape[drop:]
            tape_base += drop


# ─────────────────────────────────────────────────────────────────────────
#  Extractor contract resolution (curator_yield_run.py, built in parallel).
#  We NEVER import curator_yield_run at module load time -- only inside
#  this resolver, and only on first use -- so this file works standalone
#  (tests, --smoke) whether or not curator_yield_run.py exists yet.
# ─────────────────────────────────────────────────────────────────────────
def resolve_extractor(override=None):
    """Returns a callable extract_validated(text) -> list[dict]. `override`
    (a callable) takes priority -- this is how tests inject a stub. Absent
    an override, lazily imports curator_yield_run.extract_validated (the
    CONTRACT this file was built against, per the build brief)."""
    if override is not None:
        return override

    def _lazy(text):
        # Vertrag mit curator_yield_run: extract_validated(text) -> list[dict]
        import curator_yield_run  # noqa: E402
        return curator_yield_run.extract_validated(text)

    return _lazy


_TRIGGER_KEY_RE = re.compile(r"[^a-z0-9]+")


def _default_key(s):
    """Fallback key derivation (lower, non-alnum -> underscore, stripped)
    for stub/test records that don't already carry trigger_key/outcome_key
    -- extract_validated is contracted to provide real keys; this is only
    used by the built-in FakeExtractor's minimal dict."""
    return _TRIGGER_KEY_RE.sub("_", s.strip().lower()).strip("_")


_SENTENCE_SPLIT_RE = re.compile(r"[.!?\n]+")
_CAUSES_RE = re.compile(r"\b([a-z]+(?:\s+[a-z]+){0,2})\s+causes\s+([a-z]+(?:\s+[a-z]+){0,2})\b")


def fake_extractor(text):
    """Built-in stub extractor for --smoke and tests: regex 'X causes Y'
    sentences -> one triplet each. Deterministic, no ML, mirrors the shape
    extract_validated is contracted to return.

    Splits on sentence-ending punctuation FIRST, then matches within each
    sentence -- matching the raw {0,2}-word-either-side pattern directly
    against the whole window text would run PAST a sentence boundary
    whenever the punctuation got lost upstream (it does not here, since
    TextFileStream.raw_text_for_tape_range preserves it, but a real
    extractor -- and this stub -- should not depend on that alone; a
    sentence-first split is the correct defense either way)."""
    triplets = []
    for sentence in _SENTENCE_SPLIT_RE.split(text):
        for m in _CAUSES_RE.finditer(sentence):
            trigger, outcome = m.group(1).strip(), m.group(2).strip()
            triplets.append({
                "trigger": trigger,
                "mechanism": "causes",
                "outcome": outcome,
                "trigger_key": _default_key(trigger),
                "outcome_key": _default_key(outcome),
            })
    return triplets


# ─────────────────────────────────────────────────────────────────────────
#  Record building: extractor triplet dict -> Store schema record
#  (MVP-1's schema: trigger/mechanism/outcome/trigger_key/outcome_key/
#  doc_coord/evidence_count/use_count/meta).
# ─────────────────────────────────────────────────────────────────────────
def make_record(triplet, tape_pos, surprise, gated, extractor_version="builder_v0"):
    """Extractor triplet -> Store schema record.

    surprise/gated (P70 policy update, MVP-3): the organism's per-window
    gate signal, carried into meta so a LATER storage/dedup/memory stage
    can curate on it -- extraction itself no longer gates (see
    stream_windows' docstring). Every record gets this signal regardless
    of whether its window was gated, so a downstream policy can filter
    retroactively without re-streaming.

    curator_yield_run.extract_validated (the real bridge, P70-scored) now
    sets trigger_key/outcome_key directly via normalize_entity -- taken
    as-is when present. The _default_key fallback stays for any extractor
    that does not provide keys (e.g. a stub in a test), so this loop never
    crashes on a missing key regardless of which extractor is wired in."""
    trigger_key = triplet.get("trigger_key") or _default_key(triplet["trigger"])
    outcome_key = triplet.get("outcome_key") or _default_key(triplet["outcome"])
    return {
        "trigger": triplet["trigger"],
        "mechanism": triplet["mechanism"],
        "outcome": triplet["outcome"],
        "trigger_key": trigger_key,
        "outcome_key": outcome_key,
        "doc_coord": tape_pos,
        "evidence_count": 1,
        "use_count": 0,
        "meta": {
            "extractor_version": extractor_version,
            "surprise": float(surprise),
            "gated": bool(gated),
        },
    }


# ─────────────────────────────────────────────────────────────────────────
#  status / metrics I/O (pos_run.py's pattern: atomic status, appended
#  jsonl metrics).
# ─────────────────────────────────────────────────────────────────────────
def write_atomic(path, obj):
    d = os.path.dirname(path) or "."
    fd, tmp = tempfile.mkstemp(dir=d, prefix=".tmp_")
    with os.fdopen(fd, "w") as f:
        json.dump(obj, f, indent=2)
    os.replace(tmp, path)


def append_jsonl(path, obj):
    with open(path, "a") as f:
        f.write(json.dumps(obj) + "\n")


# ─────────────────────────────────────────────────────────────────────────
#  Main loop
# ─────────────────────────────────────────────────────────────────────────
def run_builder(
    store_dir,
    status_path,
    metrics_path,
    window_iter,
    extractor,
    windows_per_segment,
    max_windows=None,
    max_seconds=None,
    print_every=10,
    stream=None,
):
    """The loop itself, factored out of main() so tests can drive it
    directly against a stub window_iter/extractor without going through
    argv. window_iter: any iterable of (tape_pos, window_text, surprise,
    gated) -- either stream_windows(...) (the --text-file / TextFileStream
    path) or curator_yield_run.iter_windows(...) (the online --source
    c4/wt103 path, per the MVP-4 CONTRACT it documents) satisfy this the
    same way; run_builder no longer knows or cares which. `stream` is
    OPTIONAL and used only for the n_streamed_tape_pos metric (a stream
    object exposing `.docs`) -- curator_yield_run.iter_windows owns its
    stream internally and does not expose one, so this is None on that
    path and the metric falls back to n_windows_total (still monotonic,
    just coarser)."""
    graph = LiveGraph(store_dir)
    t0 = time.time()

    # P70 policy update (MVP-3): extraction runs on EVERY window, gated or
    # not. n_windows_total/n_triplets_total count the ungated firehose;
    # n_windows_gated/n_triplets_from_gated break out the subset the
    # organism's own gate accepted, purely as a metric -- nothing here
    # filters storage on it yet (the storage/dedup/memory policy P70 named
    # is a later stage, not built by this loop; meta.surprise/meta.gated on
    # every record is what that later stage will curate on).
    n_windows_total = 0
    n_windows_gated = 0
    n_triplets_total = 0
    n_triplets_from_gated = 0
    n_segments = 0
    n_base_edges_seen = 0
    append_seconds_total = 0.0

    pending_records = []
    status = {
        "pid": os.getpid(),
        "phase": "running",
        "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "store_dir": store_dir,
    }

    def current_metrics():
        return {
            "wall_s": round(time.time() - t0, 2),
            # doc/wrap count when a stream object is available (TextFileStream
            # path); falls back to the window count itself when the window
            # source owns its stream internally (curator_yield_run.iter_windows).
            "n_streamed_tape_pos": stream.docs if stream is not None else n_windows_total,
            "n_windows_total": n_windows_total,
            "n_windows_gated": n_windows_gated,
            "n_triplets_total": n_triplets_total,
            "n_triplets_from_gated": n_triplets_from_gated,
            "n_segments": n_segments,
            "n_base_edges": sum(len(v) for v in graph._base_edges.values()),
            "n_inferred_edges": len(graph.inferred_edges()),
        }

    def write_status():
        st = dict(status)
        st.update(current_metrics())
        write_atomic(status_path, st)

    write_status()

    def flush_segment():
        nonlocal n_segments, append_seconds_total, pending_records
        if not pending_records:
            return
        records = pending_records
        pending_records = []
        t_append0 = time.perf_counter()
        sha = graph.store.append_segment(records)
        new_inferred = graph.on_append(sha)
        t_append1 = time.perf_counter()
        append_s = t_append1 - t_append0
        append_seconds_total += append_s
        n_segments += 1
        append_jsonl(metrics_path, {
            "type": "append",
            "wall_s": round(time.time() - t0, 2),
            "segment_sha": sha,
            "n_records": len(records),
            "n_new_inferred": len(new_inferred),
            "append_seconds": round(append_s, 6),
            **current_metrics(),
        })
        return sha

    for tape_pos, window_text, surprise, gated in window_iter:
        n_windows_total += 1
        if gated:
            n_windows_gated += 1
        # Extraction is UNGATED (P70 policy update): runs on every window
        # regardless of `gated`. The gate signal rides along on each
        # resulting record's meta instead of filtering which windows reach
        # the extractor at all.
        triplets = extractor(window_text)
        for t in triplets:
            pending_records.append(make_record(t, tape_pos, surprise, gated))
            n_triplets_total += 1
            if gated:
                n_triplets_from_gated += 1

        if len(pending_records) >= windows_per_segment or (
            pending_records and n_windows_total % windows_per_segment == 0
        ):
            flush_segment()

        if n_windows_total % print_every == 0:
            m = current_metrics()
            print(
                "[builder] windows={n_windows_total} (gated={n_windows_gated}) "
                "triplets={n_triplets_total} (from_gated={n_triplets_from_gated}) "
                "segments={n_segments} base_edges={n_base_edges} inferred_edges={n_inferred_edges} "
                "wall={wall_s:.1f}s".format(**m),
                flush=True,
            )
            write_status()

        if max_windows and n_windows_total >= max_windows:
            break
        if max_seconds and (time.time() - t0) >= max_seconds:
            break

    flush_segment()  # final partial batch
    status["phase"] = "done"
    write_status()
    return graph, current_metrics()


# ─────────────────────────────────────────────────────────────────────────
#  Smoke corpus generator: 200 sentences with known causal chains
#  ("A causes B", "B causes C", ...), deterministic (seeded), so the
#  smoke test can assert the resulting graph contains the expected chains.
# ─────────────────────────────────────────────────────────────────────────
def generate_smoke_corpus(path, seed=42, n_sentences=200, chain_len=4):
    """Writes a synthetic text file of n_sentences sentences to `path`.
    Each sentence is 'wordA causes wordB.'; sentences are grouped into
    chains of chain_len links (wordA causes wordB, wordB causes wordC, ...)
    so the extracted graph has REAL transitive structure to verify, plus
    filler sentences with no 'causes' (noise the gate/extractor must not
    hallucinate triplets from). Returns the list of chains (as key lists)
    for the caller to assert against."""
    rng = random.Random(seed)
    chains = []
    sentences = []
    word_counter = [0]

    def _fresh_word():
        # Pure a-z letters only (base-26, Excel-column style: a, b, ...,
        # z, aa, ab, ...): the tokenizer this loop shares with
        # portable_organism (_WORD_RE = [a-zA-Z]+) strips digits entirely,
        # so an "item123"-style word would silently collapse to the
        # non-unique token "item" on every occurrence -- caught by the
        # first smoke run, where every generated word tokenized down to
        # the same few strings and no expected chain survived intact.
        word_counter[0] += 1
        n = word_counter[0]
        letters = []
        while True:
            n, r = divmod(n - 1, 26)
            letters.append(chr(ord("a") + r))
            if n == 0:
                break
        return "item" + "".join(reversed(letters))

    n_chains = max(1, n_sentences // (chain_len + 2))
    for _ in range(n_chains):
        keys = [_fresh_word() for _ in range(chain_len + 1)]
        chains.append(keys)
        for i in range(chain_len):
            sentences.append("{} causes {}.".format(keys[i], keys[i + 1]))
        # filler / noise sentences (no causal connective)
        sentences.append("the weather was {} today.".format(_fresh_word()))
        sentences.append("a report about {} was published.".format(_fresh_word()))

    while len(sentences) < n_sentences:
        sentences.append("nothing {} happened here.".format(_fresh_word()))

    rng.shuffle(sentences)
    text = " ".join(sentences[:n_sentences])
    # Repeat the text several times so the tape has enough chunks to clear
    # IGNITION_CHUNKS and MIN_WINDOW before the gate's rolling quantile has
    # anything to compare against -- a single short pass would gate
    # everything as "ignition" and never exercise the real gate.
    with open(path, "w", encoding="utf-8") as f:
        for _ in range(40):
            f.write(text)
            f.write(" ")
    return chains


def main():
    ap = argparse.ArgumentParser(description="LIVE-CAUSAL builder loop v0")
    ap.add_argument("--text-file", default=None, help="local text file as the corpus (offline path)")
    ap.add_argument("--source", default="wt103", choices=("c4", "wt103"), help="HF corpus (needs network)")
    ap.add_argument("--store-dir", default=os.path.join(REPO_ROOT, "results", "livecausal_builder_store"))
    ap.add_argument("--out-prefix", default=os.path.join(REPO_ROOT, "results", "livecausal_builder"))
    ap.add_argument("--window-tokens", type=int, default=32)
    ap.add_argument("--windows-per-segment", type=int, default=5)
    ap.add_argument("--d-model", type=int, default=64)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--chunk-size", type=int, default=32)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--q", type=float, default=0.75)
    ap.add_argument("--window", type=int, default=50)
    ap.add_argument("--min-window", type=int, default=10)
    ap.add_argument("--ignition-chunks", type=int, default=5)
    ap.add_argument("--max-windows", type=int, default=None)
    ap.add_argument("--max-seconds", type=float, default=None)
    ap.add_argument("--chunks", type=int, default=None,
                    help="online (--source) path only: bound passed to curator_yield_run.iter_windows "
                         "(default: a large finite bound; --max-windows/--max-seconds still cut the loop off early)")
    ap.add_argument("--tape-cap", type=int, default=200_000,
                    help="--text-file path only: rolling tape trim window (memory cap on long runs)")
    ap.add_argument("--smoke", action="store_true", help="generate a synthetic causal corpus + fake extractor, run offline")
    ap.add_argument("--tag", default="v0")
    args = ap.parse_args()

    extractor_override = None
    smoke_corpus_path = None
    if args.smoke:
        smoke_corpus_path = os.path.join(tempfile.gettempdir(), "livecausal_smoke_corpus_{}.txt".format(os.getpid()))
        generate_smoke_corpus(smoke_corpus_path, seed=args.seed)
        args.text_file = smoke_corpus_path
        args.max_windows = args.max_windows or 40
        extractor_override = fake_extractor
        args.store_dir = args.store_dir if "--store-dir" in sys.argv else os.path.join(
            tempfile.gettempdir(), "livecausal_smoke_store_{}".format(os.getpid())
        )

    if args.text_file:
        # --text-file is the offline path (F6: the corpus is an iterator,
        # a local file counts): get_vocab() still needs WikiText2 for its
        # vocabulary, which datasets can serve from its local cache, but
        # only if we force it not to probe the Hub first (that probe is
        # what fails, not slowly, when DNS is down -- an unhandled
        # exception, not a graceful fallback). This makes the offline path
        # actually offline instead of "offline unless the Hub probe hangs".
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        os.environ.setdefault("HF_DATASETS_OFFLINE", "1")

    import torch  # noqa: E402
    torch.set_num_threads(1)
    import portable_organism as po  # noqa: E402

    # Set the gate cadence knobs before EITHER path builds an Organism:
    # the --text-file path builds one directly in build_window_iterator;
    # the --source path delegates to curator_yield_run.iter_windows, which
    # re-sets these same module globals itself from the args it's given
    # (same values, passed explicitly below) -- setting them here too is
    # redundant but harmless for that path, and required for the
    # --text-file path since stream_windows calls organism.step_gated
    # against whatever these currently are.
    po.D_MODEL, po.BATCH, po.CHUNK = args.d_model, args.batch, args.chunk_size
    po.GATE_Q, po.GATE_WINDOW, po.MIN_WINDOW, po.IGNITION_CHUNKS = (
        args.q, args.window, args.min_window, args.ignition_chunks,
    )

    extractor = resolve_extractor(extractor_override)
    window_iter, stream = build_window_iterator(args, organism_seed=args.seed)

    status_path = args.out_prefix + "_status.json"
    metrics_path = args.out_prefix + "_metrics.jsonl"
    os.makedirs(os.path.dirname(status_path) or ".", exist_ok=True)

    graph, final_metrics = run_builder(
        args.store_dir,
        status_path,
        metrics_path,
        window_iter,
        extractor,
        args.windows_per_segment,
        max_windows=args.max_windows,
        max_seconds=args.max_seconds,
        stream=stream,
    )

    print("=" * 74)
    print("[builder] done: {}".format(final_metrics))
    print("[builder] graph store: {}".format(args.store_dir))

    if smoke_corpus_path and os.path.exists(smoke_corpus_path):
        os.remove(smoke_corpus_path)


if __name__ == "__main__":
    main()
