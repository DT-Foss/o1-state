"""Plain-assert tests for the consult-back loop (src/livecausal/consult_run.py).
Run: python3 src/livecausal/test_consult_run.py

Offline-only.
"""
import os
import random
import shutil
import sys
import tempfile

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("HF_DATASETS_OFFLINE", "1")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SRC = os.path.join(REPO_ROOT, "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from livecausal.infer import LiveGraph  # noqa: E402
from livecausal.store import LiveStore  # noqa: E402
from livecausal.evidence import EvidenceLedger, UseLedger  # noqa: E402
from livecausal.consult_run import (  # noqa: E402
    best_edge_for_key,
    calibrate_surprise_threshold,
    generate_smoke_store,
    generate_vocab_aware_smoke_corpus,
    random_edge_for_key,
    run_consult,
)


def with_tmpdir(fn):
    d = tempfile.mkdtemp(prefix="consult-test-")
    try:
        fn(d)
    finally:
        shutil.rmtree(d, ignore_errors=True)


def _build_smoke_graph(d, seed=42):
    store_dir = os.path.join(d, "store")
    corpus_path = os.path.join(d, "corpus.txt")
    graph, chains = generate_smoke_store(store_dir, corpus_path, seed=seed)
    return graph, chains, store_dir, corpus_path


# ---------------------------------------------------------------------
# Test 1: best_edge_for_key finds a real edge with real evidence, and
# its outcome text is recoverable (non-empty) prose from the store.
# ---------------------------------------------------------------------

def test_best_edge_for_key_finds_real_edge(d):
    graph, chains, store_dir, corpus_path = _build_smoke_graph(d)
    led = EvidenceLedger(store_dir)
    valid = graph.store.segments()

    start_key = chains[0][0]
    edge, text = best_edge_for_key(graph, led, valid, start_key)
    assert edge is not None, "expected chains[0][0] to have an outgoing edge"
    assert edge["from_key"] == start_key
    assert text, "outcome text must be non-empty prose"
    assert isinstance(text, str)


def test_best_edge_for_key_none_for_unknown(d):
    graph, chains, store_dir, corpus_path = _build_smoke_graph(d)
    led = EvidenceLedger(store_dir)
    valid = graph.store.segments()
    edge, text = best_edge_for_key(graph, led, valid, "this_key_was_never_in_the_corpus")
    assert edge is None and text is None


# ---------------------------------------------------------------------
# Test 2: random_edge_for_key never returns the excluded key's own edges,
# and (with the vocab-aware corpus) its text tokenizes to something other
# than pure unk -- the exact bug the build report flags and fixes.
# ---------------------------------------------------------------------

def test_random_edge_excludes_source_and_is_in_vocab(d):
    graph, chains, store_dir, corpus_path = _build_smoke_graph(d)
    valid = graph.store.segments()
    rng = random.Random(1)

    import portable_organism as po
    vocab, stoi, unk, mask, val_ids = po.get_vocab()

    start_key = chains[0][0]
    for _ in range(10):
        rand_key, rand_text = random_edge_for_key(graph, valid, start_key, rng)
        if rand_key is None:
            continue
        assert rand_key[0] != start_key, "random edge must not originate from the excluded key"
        assert rand_text
        toks = [stoi.get(w, unk) for w in rand_text.lower().split()]
        assert any(t != unk for t in toks), (
            "random-arm injection text is entirely unk -- the exact bug "
            "the vocab-aware smoke corpus was built to eliminate"
        )


# ---------------------------------------------------------------------
# Test 3: calibrate_surprise_threshold returns a finite, positive number
# that actually sits inside the observed surprise distribution (not some
# fixed constant unrelated to the model/corpus in front of it).
# ---------------------------------------------------------------------

def test_calibrate_surprise_threshold_tracks_distribution(d):
    graph, chains, store_dir, corpus_path = _build_smoke_graph(d)

    import torch
    import portable_organism as po
    from livecausal.consult_run import _read

    torch.set_num_threads(1)
    vocab, stoi, unk, mask, val_ids = po.get_vocab()
    torch.manual_seed(1)
    organism = po.Organism("calib-test", len(vocab), mask, seed=1)
    model = organism.model
    model.eval()
    dev = next(model.parameters()).device

    with open(corpus_path, "r", encoding="utf-8") as f:
        import re
        words = re.findall(r"[a-zA-Z]{2,}", f.read().lower())

    thresh = calibrate_surprise_threshold(words, stoi, unk, model, dev, torch, quantile=0.9, n_calib_words=300)
    assert thresh > 0, "threshold must be positive"
    assert thresh < 50, "threshold implausibly large for a tiny model -- calibration likely broken"

    # Cross-check: the SAME words scored again must show ~90% at or below
    # threshold (quantile calibration did what it claims).
    ids = [stoi.get(w, unk) for w in words[:300]]
    surprises, _ = _read(model, ids, None, dev, torch)
    n_at_or_below = sum(1 for s in surprises if s <= thresh + 1e-6)
    frac = n_at_or_below / max(1, len(surprises))
    assert frac >= 0.85, "quantile calibration should place ~90% of the SAME sample at or below threshold, got {:.2f}".format(frac)


# ---------------------------------------------------------------------
# Test 4: run_consult only logs a use-ledger entry when drop_real > 0 --
# the core "the graph learns from being used, not merely queried" rule.
# ---------------------------------------------------------------------

def test_use_ledger_only_grows_on_positive_delta(d):
    graph, chains, store_dir, corpus_path = _build_smoke_graph(d)
    evidence_ledger = EvidenceLedger(store_dir)
    use_ledger = UseLedger(store_dir)

    import torch
    import portable_organism as po
    import re

    torch.set_num_threads(1)
    vocab, stoi, unk, mask, val_ids = po.get_vocab()
    torch.manual_seed(7)
    organism = po.Organism("use-test", len(vocab), mask, seed=7)
    model = organism.model
    model.eval()
    dev = next(model.parameters()).device

    with open(corpus_path, "r", encoding="utf-8") as f:
        words = re.findall(r"[a-zA-Z]{2,}", f.read().lower())

    thresh = calibrate_surprise_threshold(words, stoi, unk, model, dev, torch, quantile=0.9, n_calib_words=300)

    result = run_consult(
        graph, evidence_ledger, use_ledger, words, stoi, unk, model, dev, torch,
        surprise_thresh=thresh, lookahead=8, max_gaps=15, seed=1,
    )

    n_helped = sum(1 for r in result["results"] if r["drop_real"] > 0)
    assert n_helped == result["n_helped_real"]

    n_used_edges_in_helped_rows = sum(len(r["used_edges"]) for r in result["results"] if r["drop_real"] > 0)
    n_used_edges_in_unhelped_rows = sum(len(r["used_edges"]) for r in result["results"] if r["drop_real"] <= 0)
    assert n_used_edges_in_unhelped_rows == 0, (
        "a row with drop_real <= 0 logged a use entry -- the graph would "
        "be reinforced by consultations that did NOT help"
    )
    assert result["n_use_entries"] == n_used_edges_in_helped_rows

    if result["n_use_entries"] > 0:
        # Fold reproducibility: a fresh UseLedger mount over the same
        # store agrees with the one that wrote the entries.
        valid = graph.store.segments()
        used_key = tuple(result["results"][
            next(i for i, r in enumerate(result["results"]) if r["used_edges"])
        ]["used_edges"][0])
        led2 = UseLedger(store_dir)
        assert led2.use_count(used_key, valid) == use_ledger.use_count(used_key, valid)


# ---------------------------------------------------------------------
# Test 5: determinism -- two independent run_consult calls with the SAME
# seed, model init, and corpus produce byte-identical result rows (the
# consult loop is pure inference + a fixed rng, no hidden nondeterminism).
# ---------------------------------------------------------------------

def test_run_consult_is_deterministic(d):
    graph, chains, store_dir, corpus_path = _build_smoke_graph(d)
    evidence_ledger = EvidenceLedger(store_dir)

    import torch
    import portable_organism as po
    import re

    torch.set_num_threads(1)
    vocab, stoi, unk, mask, val_ids = po.get_vocab()

    with open(corpus_path, "r", encoding="utf-8") as f:
        words = re.findall(r"[a-zA-Z]{2,}", f.read().lower())

    def _run_once(use_ledger_path):
        torch.manual_seed(3)
        organism = po.Organism("det-test", len(vocab), mask, seed=3)
        model = organism.model
        model.eval()
        dev = next(model.parameters()).device
        thresh = calibrate_surprise_threshold(words, stoi, unk, model, dev, torch, quantile=0.9, n_calib_words=300)
        use_ledger = UseLedger(use_ledger_path)
        return run_consult(
            graph, evidence_ledger, use_ledger, words, stoi, unk, model, dev, torch,
            surprise_thresh=thresh, lookahead=8, max_gaps=10, seed=1,
        )

    dir_a = os.path.join(d, "runA")
    dir_b = os.path.join(d, "runB")
    os.makedirs(dir_a)
    os.makedirs(dir_b)

    result_a = _run_once(dir_a)
    result_b = _run_once(dir_b)

    assert len(result_a["results"]) == len(result_b["results"])
    for ra, rb in zip(result_a["results"], result_b["results"]):
        assert ra["gap"] == rb["gap"]
        assert ra["gap_surprise"] == rb["gap_surprise"]
        assert ra["cont_surprise_without"] == rb["cont_surprise_without"]
        assert ra["cont_surprise_with_real_path"] == rb["cont_surprise_with_real_path"]
        assert ra["drop_real"] == rb["drop_real"]


# ---------------------------------------------------------------------
# P75 read-side wiring: best_edge_for_key(..., canon=True) finds edges
# across raw-key variants that differ only by determiner/adjective (the
# same join shape src/livecausal/test_canon.py's connectivity test
# exercises directly on LiveGraph), while the exact-string path (canon
# default False) cannot see across that gap at all.
# ---------------------------------------------------------------------

def _make_record(trigger, outcome, doc_coord):
    """Direct store record, real noun-phrase trigger/outcome text (not
    the trig::X placeholder shape) so canonical_key has something to
    parse -- same construction test_canon.py uses, kept local here since
    this test builds its own tiny store rather than going through
    generate_smoke_store's ML/fake_extractor pipeline (no organism/
    training needed to exercise query/best_edge_for_key selection logic
    in isolation)."""
    return {
        "trigger": trigger,
        "mechanism": "causes",
        "outcome": outcome,
        "trigger_key": trigger.strip().lower(),
        "outcome_key": outcome.strip().lower(),
        "doc_coord": doc_coord,
        "evidence_count": 1,
        "use_count": 0,
        "meta": {},
    }


def test_best_edge_for_key_canon_finds_edge_exact_string_misses(d):
    # r1's raw trigger is "the resulting recession" (canon_key "recession");
    # the consult loop queries with the bare word "recession" -- a
    # DIFFERENT raw surface string that never appears as trigger_key in
    # the store at all, but shares r1's canon_key. Exact-string lookup
    # must find nothing; canon=True lookup must find r1.
    store_dir = os.path.join(d, "store")
    store = LiveStore(store_dir)
    store.append_segment([_make_record("the resulting recession", "a sharp rise in unemployment", 0)])

    led = EvidenceLedger(store_dir)
    graph_raw = LiveGraph(store_dir, canon=False)
    valid = graph_raw.store.segments()

    # Exact-string path: the query key "recession" is not any record's
    # trigger_key ("the resulting recession" is), so no edge is found.
    edge_raw, text_raw = best_edge_for_key(graph_raw, led, valid, "recession", canon=False)
    assert edge_raw is None and text_raw is None, (
        "exact-string lookup should NOT find an edge for a raw key that "
        "never appears verbatim as a trigger_key: {}".format(edge_raw)
    )

    # Canon path: query key "recession" canonicalizes to "recession",
    # which matches r1's trigger's canon_key ("the resulting recession"
    # -> "recession") -- the edge must be found, with real outcome prose
    # recoverable exactly like the exact-string path recovers it.
    graph_canon = LiveGraph(store_dir, canon=True)
    edge_canon, text_canon = best_edge_for_key(graph_canon, led, valid, "recession", canon=True)
    assert edge_canon is not None, "canon=True lookup should find the joined edge"
    assert edge_canon["from_key"] == "recession"
    assert text_canon, "outcome text must be real, non-empty prose"
    assert "unemployment" in text_canon.lower()

    # canon_env_pin travels on the graph that produced this result --
    # available for a caller (main()) to stamp into the run's payload.
    assert graph_canon.canon_env_pin is not None
    assert graph_canon.canon_env_pin["canon_version"]


# ---------------------------------------------------------------------
# P75 regression: best_edge_for_key/run_consult with canon left at its
# default (False) must behave identically to the pre-P75 call shape --
# checked directly against the smoke-graph fixture the existing tests
# already use, not just a fresh assertion on the new corpus above.
# ---------------------------------------------------------------------

def test_consult_canon_false_is_regression_identical(d):
    graph, chains, store_dir, corpus_path = _build_smoke_graph(d)
    led = EvidenceLedger(store_dir)
    valid = graph.store.segments()
    start_key = chains[0][0]

    # Same call, with and without the new keyword argument explicit --
    # must agree exactly (edge dict equal field-for-field, same text).
    edge_implicit, text_implicit = best_edge_for_key(graph, led, valid, start_key)
    edge_explicit, text_explicit = best_edge_for_key(graph, led, valid, start_key, canon=False)
    assert edge_implicit == edge_explicit
    assert text_implicit == text_explicit

    # A canon=True graph is not required to run the canon=False path --
    # this graph was mounted with the default (canon=False) and no
    # canon_inferred.jsonl file exists, exactly like before P75.
    assert graph.canon_enabled is False
    assert not os.path.exists(os.path.join(store_dir, "canon_inferred.jsonl"))

    # run_consult with canon left at its default must match a run that
    # passes canon=False explicitly, row for row (same seed, same corpus,
    # same model init) -- the keyword existing at all must not perturb
    # the default code path.
    import torch
    import portable_organism as po
    import re

    torch.set_num_threads(1)
    vocab, stoi, unk, mask, val_ids = po.get_vocab()

    with open(corpus_path, "r", encoding="utf-8") as f:
        words = re.findall(r"[a-zA-Z]{2,}", f.read().lower())

    def _run_once(use_ledger_path, explicit_canon_false):
        torch.manual_seed(5)
        organism = po.Organism("regress-test", len(vocab), mask, seed=5)
        model = organism.model
        model.eval()
        dev = next(model.parameters()).device
        thresh = calibrate_surprise_threshold(words, stoi, unk, model, dev, torch, quantile=0.9, n_calib_words=300)
        use_ledger = UseLedger(use_ledger_path)
        kwargs = {"canon": False} if explicit_canon_false else {}
        return run_consult(
            graph, led, use_ledger, words, stoi, unk, model, dev, torch,
            surprise_thresh=thresh, lookahead=8, max_gaps=10, seed=1,
            **kwargs,
        )

    dir_default = os.path.join(d, "runDefault")
    dir_explicit = os.path.join(d, "runExplicit")
    os.makedirs(dir_default)
    os.makedirs(dir_explicit)

    result_default = _run_once(dir_default, explicit_canon_false=False)
    result_explicit = _run_once(dir_explicit, explicit_canon_false=True)

    assert result_default["results"] == result_explicit["results"]
    assert result_default["mean_delta_real"] == result_explicit["mean_delta_real"]
    assert result_default["coverage"] == result_explicit["coverage"]


def run_all():
    tests = [
        test_best_edge_for_key_finds_real_edge,
        test_best_edge_for_key_none_for_unknown,
        test_random_edge_excludes_source_and_is_in_vocab,
        test_calibrate_surprise_threshold_tracks_distribution,
        test_use_ledger_only_grows_on_positive_delta,
        test_run_consult_is_deterministic,
        test_best_edge_for_key_canon_finds_edge_exact_string_misses,
        test_consult_canon_false_is_regression_identical,
    ]
    for t in tests:
        with_tmpdir(t)
        print("OK  {}".format(t.__name__))
    print("ALL {} TESTS PASSED".format(len(tests)))


if __name__ == "__main__":
    run_all()
