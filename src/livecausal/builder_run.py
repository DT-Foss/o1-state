#!/usr/bin/env python3 -u
"""
BUILDER LOOP v0 — the organism streams, the gate curates, the extractor
validates, the live graph grows. analysis/LIVE_CAUSAL_SPEC.md SS4 ("the
organism as builder"), stages 1-3 (curate / extract / fold) wired end to
end: no LLM anywhere in the loop.

    stream -> po.Organism.step_gated (P58's live gate, surprise_filter_run.py
              pattern) -> gated chunk -> window text -> extract_validated
              (curator_yield_run's contract, imported lazily -- SEE BELOW)
              -> LiveStore.append_segment -> LiveGraph.on_append

CONTRACT WITH src/curator_yield_run.py (lead-defined, built in parallel by
mvp3-p70; this file imports it only through the two functions below, never
its internals):
    extract_validated(text: str) -> list[dict]
        Validated triplets, each dict carrying at least {trigger, mechanism,
        outcome, trigger_key, outcome_key}.
    a window source iterable over (tape_pos: int, window_text: str) for
        gated windows -- this file OWNS that iterable (gated_windows()
        below implements it against po.Organism.step_gated), the contract
        only fixes its shape so curator_yield_run's extractor can consume
        it identically to how this loop does.

Both are resolved via a lazy import hook (`_default_extractor`,
`resolve_extractor`) so tests can stub them (see --extractor-stub / the
FakeExtractor path) without needing curator_yield_run to exist yet.

Two corpus sources:
  --text-file PATH   a local text file streamed as an iterator of blocks
                      (F6: "the corpus is an iterator" -- this is the
                      offline-capable, no-HF-network path used for the
                      smoke test and for beast while DNS is down).
  --source c4|wt103   the HF streaming corpus (HFStream, source_swap_run.py
                      pattern) -- needs network / a warm HF cache.

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


class HFCorpusStream:
    """Thin wrapper around portable_organism's C4Stream/source_swap_run's
    HFStream that also retains the raw words it has streamed so far (for
    raw_text_for_tape_range), bounded to a rolling window so memory stays
    flat on a long run. Needs network (HF `datasets`) unless the dataset
    is already warm in the local HF cache (verified true for wikitext2 on
    beast; c4/wt103 streaming still needs a live connection per shard)."""

    def __init__(self, source, stoi, unk, block=8192, keep_words=200_000):
        import portable_organism as po  # noqa: E402
        from source_swap_run import HFStream  # noqa: E402

        self._stream = HFStream(source, stoi, unk, block=block)
        self._po = po
        self.keep_words = keep_words
        self._word_buffer = []   # rolling (tape_pos_of_first, [words...])
        self._word_buffer_start = 0

    @property
    def docs(self):
        return self._stream.docs

    @property
    def reconnects(self):
        return self._stream.reconnects

    @property
    def pending(self):
        return self._stream.pending

    def next_block(self):
        # HFStream tokenizes internally without keeping the source words;
        # we cannot recover them post hoc for an HF corpus without
        # re-tokenizing the same underlying text, which the base class
        # does not expose. Documented gap (flagged in the build report):
        # raw_text_for_tape_range on this source returns None, so
        # gated_windows() falls back to a token-id placeholder text for
        # HF corpora. The --text-file path is the one with exact raw text.
        return self._stream.next_block()

    def raw_text_for_tape_range(self, tape, lo, hi):
        return None


def build_corpus_stream(args, stoi, unk):
    if args.text_file:
        return TextFileStream(args.text_file, stoi, unk)
    return HFCorpusStream(args.source, stoi, unk)


# ─────────────────────────────────────────────────────────────────────────
#  Gated window generator: the FENCE the SPEC calls out -- SS4 stage 1
#  (curate). Wraps po.Organism.step_gated (surprise_filter_run.py's live
#  gate, not a post-hoc top-M sort: a window is curated exactly when the
#  gate accepts its chunk in real time).
# ─────────────────────────────────────────────────────────────────────────
def gated_windows(organism, stream, feeder, window_tokens, tape_cap=None):
    """Generator over (tape_pos: int, window_text: str) for every chunk the
    live gate accepts. tape_pos is the tape offset of the window START
    (the gated chunk's first token) -- this IS the builder's doc_coord.
    window_text is window_tokens tokens' worth of ORIGINAL words (not the
    tokenized ids) recovered from the stream, when the stream supports it
    (TextFileStream always does; HFCorpusStream currently does not -- see
    its raw_text_for_tape_range note).

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
        if gated:
            lo, hi = chunk_start, chunk_start + window_tokens
            text = stream.raw_text_for_tape_range(tape, lo, hi)
            if text is None:
                # Fallback for sources without raw-text recovery: encode
                # the token ids as a placeholder string so the extractor
                # contract (text: str) is still satisfied. Flagged: this
                # degrades extraction quality for HF corpora until the
                # extractor is also handed the vocab to detokenize.
                text = " ".join(str(t) for t in x[0][:window_tokens].tolist())
            yield chunk_start, text
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
def make_record(triplet, tape_pos, extractor_version="builder_v0"):
    """Extractor triplet -> Store schema record.

    CONTRACT DEVIATION (flagged in the build report): the build brief's
    contract requires extract_validated to return trigger_key/outcome_key
    directly. The REAL curator_yield_run.extract_validated (already
    present in this repo, P70-scored) does not -- it returns
    {trigger, mechanism, outcome, confidence, evidence_sentence,
    quantification, domain, source, quality_score}, no *_key fields. This
    derives keys the same way the built-in fake_extractor's _default_key
    does (lower, non-alnum -> underscore) so the loop stays runnable
    against the real extractor without crashing on a missing key; if
    curator_yield_run later adds real keys (e.g. entity-normalized, not
    just string-normalized), swap this fallback for triplet['trigger_key']
    directly and drop the .get() defaulting below."""
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
        "meta": {"extractor_version": extractor_version},
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
    stream,
    organism,
    feeder,
    extractor,
    window_tokens,
    windows_per_segment,
    max_windows=None,
    max_seconds=None,
    tape_cap=200_000,
    print_every=10,
):
    """The loop itself, factored out of main() so tests can drive it
    directly against a stub stream/extractor without going through argv."""
    graph = LiveGraph(store_dir)
    t0 = time.time()

    n_windows_curated = 0
    n_triplets_validated = 0
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
            "n_streamed_tape_pos": stream.docs,  # coarse: doc/wrap count, exact tape pos is per-window
            "n_windows_curated": n_windows_curated,
            "n_triplets_validated": n_triplets_validated,
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

    win_iter = gated_windows(organism, stream, feeder, window_tokens, tape_cap=tape_cap)
    for tape_pos, window_text in win_iter:
        n_windows_curated += 1
        triplets = extractor(window_text)
        for t in triplets:
            pending_records.append(make_record(t, tape_pos))
            n_triplets_validated += 1

        if len(pending_records) >= windows_per_segment or (
            pending_records and n_windows_curated % windows_per_segment == 0
        ):
            flush_segment()

        if n_windows_curated % print_every == 0:
            m = current_metrics()
            print(
                "[builder] windows={n_windows_curated} triplets={n_triplets_validated} "
                "segments={n_segments} base_edges={n_base_edges} inferred_edges={n_inferred_edges} "
                "wall={wall_s:.1f}s".format(**m),
                flush=True,
            )
            write_status()

        if max_windows and n_windows_curated >= max_windows:
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

    po.D_MODEL, po.BATCH, po.CHUNK = args.d_model, args.batch, args.chunk_size
    po.GATE_Q, po.GATE_WINDOW, po.MIN_WINDOW, po.IGNITION_CHUNKS = (
        args.q, args.window, args.min_window, args.ignition_chunks,
    )
    vocab, stoi, unk, mask, val_ids = po.get_vocab()
    V = len(vocab)

    torch.manual_seed(args.seed)
    organism = po.Organism("builder", V, mask, seed=args.seed)
    stream = build_corpus_stream(args, stoi, unk)
    feeder = po.ChunkFeeder(stream, args.batch, args.chunk_size)
    extractor = resolve_extractor(extractor_override)

    status_path = args.out_prefix + "_status.json"
    metrics_path = args.out_prefix + "_metrics.jsonl"
    os.makedirs(os.path.dirname(status_path) or ".", exist_ok=True)

    graph, final_metrics = run_builder(
        args.store_dir,
        status_path,
        metrics_path,
        stream,
        organism,
        feeder,
        extractor,
        args.window_tokens,
        args.windows_per_segment,
        max_windows=args.max_windows,
        max_seconds=args.max_seconds,
    )

    print("=" * 74)
    print("[builder] done: {}".format(final_metrics))
    print("[builder] graph store: {}".format(args.store_dir))

    if smoke_corpus_path and os.path.exists(smoke_corpus_path):
        os.remove(smoke_corpus_path)


if __name__ == "__main__":
    main()
