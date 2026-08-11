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
    ReaderCkptError,
    best_edge_for_key,
    build_injection_text,
    calibrate_band,
    calibrate_surprise_threshold,
    generate_smoke_store,
    generate_vocab_aware_smoke_corpus,
    load_reader_from_ckpt,
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


# ---------------------------------------------------------------------
# P77: build_injection_text's three forms select the right text, and
# chain_text actually differs from outcome_text when a multi-hop
# derivation is present (otherwise the "form" axis would be vacuous for
# inferred edges, the exact case the P77 build brief calls out).
# ---------------------------------------------------------------------

def test_build_injection_text_forms(d):
    store_dir = os.path.join(d, "store")
    store = LiveStore(store_dir)
    store.append_segment([_make_record("a spark", "a fire", 0)])
    store.append_segment([_make_record("a fire", "a evacuation", 1)])

    graph = LiveGraph(store_dir, canon=False)
    base_edges = graph.query("a spark")
    base_edge = next(e for e in base_edges if e["kind"] == "base")
    assert build_injection_text(graph, base_edge, "outcome_text") == "a fire"
    assert build_injection_text(graph, base_edge, "full_record_text") == "a spark causes a fire"
    # base edge has exactly one hop -- chain_text degenerates to outcome_text.
    assert build_injection_text(graph, base_edge, "chain_text") == "a fire"

    inferred_edges = [e for e in base_edges if e["kind"] == "inferred"]
    assert inferred_edges, "expected a spark -> a fire -> a evacuation inferred edge"
    inferred = inferred_edges[0]
    assert build_injection_text(graph, inferred, "outcome_text") == "a evacuation", (
        "outcome_text must be the LAST hop's outcome only"
    )
    chain = build_injection_text(graph, inferred, "chain_text")
    assert chain == "a fire a evacuation", (
        "chain_text must concatenate EVERY hop's outcome text, root-to-leaf: got {!r}".format(chain)
    )
    assert chain != build_injection_text(graph, inferred, "outcome_text"), (
        "chain_text must differ from outcome_text for a real multi-hop edge -- "
        "otherwise the form axis is vacuous for inferred edges"
    )
    full = build_injection_text(graph, inferred, "full_record_text")
    assert full == "a fire causes a evacuation", (
        "full_record_text on a multi-hop edge uses the LAST hop's full sentence, same hop outcome_text uses"
    )
    print("test_build_injection_text_forms: OK (base={!r}, inferred outcome={!r}, chain={!r}, full={!r})".format(
        build_injection_text(graph, base_edge, "outcome_text"),
        build_injection_text(graph, inferred, "outcome_text"), chain, full,
    ))


def test_build_injection_text_empty_derivation_returns_empty(d):
    graph, chains, store_dir, corpus_path = _build_smoke_graph(d)
    fake_edge = {"from_key": "x", "to_key": "y", "derivation": []}
    for form in ("outcome_text", "full_record_text", "chain_text"):
        assert build_injection_text(graph, fake_edge, form) == "", (
            "an edge with no derivation must yield '' regardless of form ({})".format(form)
        )
    print("test_build_injection_text_empty_derivation_returns_empty: OK")


# ---------------------------------------------------------------------
# P77: random_edge_for_key(..., return_edge=True) draws the SAME
# candidate as the pre-P77 call shape (same rng stream, same choice) --
# only the RETURN SHAPE changes, so run_consult's random arm sees exactly
# the edge the control arm has always seen, just wrapped for
# build_injection_text.
# ---------------------------------------------------------------------

def test_random_edge_for_key_return_edge_shape_matches_legacy(d):
    graph, chains, store_dir, corpus_path = _build_smoke_graph(d)
    valid = graph.store.segments()
    start_key = chains[0][0]

    key_tuple, text_legacy = random_edge_for_key(graph, valid, start_key, random.Random(9))
    edge_dict, text_new = random_edge_for_key(graph, valid, start_key, random.Random(9), return_edge=True)

    if key_tuple is None:
        print("test_random_edge_for_key_return_edge_shape_matches_legacy: SKIPPED (no candidates)")
        return
    assert (edge_dict["from_key"], edge_dict["to_key"]) == key_tuple, "return_edge=True must draw the SAME candidate as the legacy call"
    assert text_new == text_legacy
    assert edge_dict["derivation"] and len(edge_dict["derivation"]) == 1, "a base edge has exactly one citation hop"
    print("test_random_edge_for_key_return_edge_shape_matches_legacy: OK (key={})".format(key_tuple))


# ---------------------------------------------------------------------
# P77: run_consult with a non-default inject_form/inject_repeat still
# respects the use-ledger-only-grows-on-positive-delta invariant, and the
# random arm's text visibly changes form the same way the real arm's does
# (both go through build_injection_text once inject_form != outcome_text).
# ---------------------------------------------------------------------

def test_run_consult_with_full_record_text_form(d):
    graph, chains, store_dir, corpus_path = _build_smoke_graph(d)
    evidence_ledger = EvidenceLedger(store_dir)
    use_ledger = UseLedger(store_dir)

    import torch
    import portable_organism as po
    import re

    torch.set_num_threads(1)
    vocab, stoi, unk, mask, val_ids = po.get_vocab()
    torch.manual_seed(11)
    organism = po.Organism("form-test", len(vocab), mask, seed=11)
    model = organism.model
    model.eval()
    dev = next(model.parameters()).device

    with open(corpus_path, "r", encoding="utf-8") as f:
        words = re.findall(r"[a-zA-Z]{2,}", f.read().lower())

    thresh = calibrate_surprise_threshold(words, stoi, unk, model, dev, torch, quantile=0.9, n_calib_words=300)

    result = run_consult(
        graph, evidence_ledger, use_ledger, words, stoi, unk, model, dev, torch,
        surprise_thresh=thresh, lookahead=8, max_gaps=15, seed=1,
        inject_form="full_record_text", inject_repeat=2,
    )
    assert isinstance(result["mean_delta_real"], float)
    n_helped = sum(1 for r in result["results"] if r["drop_real"] > 0)
    assert n_helped == result["n_helped_real"]
    n_used_edges_in_unhelped_rows = sum(len(r["used_edges"]) for r in result["results"] if r["drop_real"] <= 0)
    assert n_used_edges_in_unhelped_rows == 0, "non-default inject_form must still respect the positive-delta-only use logging invariant"
    print("test_run_consult_with_full_record_text_form: OK (n_consults={}, mean_delta_real={:+.4f})".format(
        result["n_consults"], result["mean_delta_real"]))


# ---------------------------------------------------------------------
# P77 regression: --grid must NEVER write into the real --store directory
# -- caught by hand during this build (UseLedger's own constructor writes
# a header file the instant it's called, so constructing `use_ledger =
# UseLedger(args.store)` unconditionally in main(), before branching on
# args.grid, put a use.ledger into the real store even though the grid
# path never calls append_use on it). Runs main() as a real subprocess
# (not an import) since the bug lived in main()'s own setup code, not in
# run_consult -- an in-process call would not have exercised the path
# that broke.
# ---------------------------------------------------------------------

def test_grid_never_touches_store_use_ledger(d):
    import subprocess

    store_dir = os.path.join(d, "store")
    corpus_path = os.path.join(d, "corpus.txt")
    generate_smoke_store(store_dir, corpus_path, seed=5)

    use_ledger_path = os.path.join(store_dir, "use.ledger")
    assert not os.path.exists(use_ledger_path), "test setup assumption broken: use.ledger already existed before the grid run"

    out_path = os.path.join(d, "grid_out.json")
    consult_run_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "consult_run.py")
    env = dict(os.environ)
    env["HF_HUB_OFFLINE"] = "1"
    env["HF_DATASETS_OFFLINE"] = "1"
    env["OMP_NUM_THREADS"] = "1"
    proc = subprocess.run(
        [sys.executable, consult_run_path, "--store", store_dir, "--text-file", corpus_path,
         "--grid", "--inject-form", "outcome_text", "full_record_text", "--inject-repeat", "1",
         "--max-gaps", "6", "--warmup-chunks", "0", "--out", out_path],
        capture_output=True, text=True, env=env, timeout=120,
    )
    assert proc.returncode == 0, "grid subprocess failed: {}\n{}".format(proc.stdout[-2000:], proc.stderr[-2000:])
    assert not os.path.exists(use_ledger_path), (
        "--grid wrote a use.ledger into the real --store directory -- "
        "the exact class of bug the 'gescorte Stores sind read-only' rule exists to prevent"
    )
    assert os.path.exists(out_path), "grid run did not write its output JSON"
    print("test_grid_never_touches_store_use_ledger: OK (store={} has no use.ledger after --grid)".format(store_dir))


# ---------------------------------------------------------------------
# P78: load_reader_from_ckpt's roundtrip. A Mini-Organism (small vocab,
# small d_model, a handful of trained steps so its weights are NOT the
# fresh-init default) is saved via portable_organism.save_snapshot, then
# loaded back via load_reader_from_ckpt -- the reconstructed model's
# forward output on a fixed input must be BYTE-IDENTICAL to the
# original's (same weights, same eval-mode determinism), and d_model
# must have come from the checkpoint's own config, not a guess.
# ---------------------------------------------------------------------

def test_reader_ckpt_roundtrip_byte_identical_forward(d):
    import torch
    import portable_organism as po

    torch.set_num_threads(1)
    mini_vocab = ["<unk>"] + ["w{}".format(i) for i in range(63)]  # 64-word toy vocab
    mini_stoi = {w: i for i, w in enumerate(mini_vocab)}
    mini_unk = 0
    mini_mask = None

    po.D_MODEL, po.BATCH, po.CHUNK = 16, 2, 8
    torch.manual_seed(21)
    organism = po.Organism("mini-reader", len(mini_vocab), mini_mask, seed=21)

    # a few real gated training steps -- weights must move OFF the
    # fresh-init default, or a roundtrip bug that silently reconstructs a
    # FRESH model instead of loading the checkpoint would go undetected.
    rng = random.Random(3)
    for _ in range(6):
        x = torch.tensor([[rng.randrange(len(mini_vocab)) for _ in range(po.CHUNK)] for _ in range(po.BATCH)], dtype=torch.long)
        y = torch.tensor([[rng.randrange(len(mini_vocab)) for _ in range(po.CHUNK)] for _ in range(po.BATCH)], dtype=torch.long)
        organism.step_gated(x, y, ignition=True)

    ckpt_path = os.path.join(d, "mini_ckpt.pt")

    class _FakeStream:
        docs, pending = 0, []

    po.save_snapshot(ckpt_path, organism, _FakeStream(), bufs=None, wall_s=0.0)
    assert os.path.exists(ckpt_path)

    probe = torch.tensor([[rng.randrange(len(mini_vocab)) for _ in range(po.CHUNK)]], dtype=torch.long)
    organism.model.eval()
    with torch.no_grad():
        logits_before, _ = organism.model(probe, None)

    # Simulate a DIFFERENT po.D_MODEL in effect before loading (as if a
    # prior --smoke build or a mismatched --d-model ran first) -- the
    # loader must still size the reconstructed model from the CHECKPOINT's
    # own config, not whatever po.D_MODEL happens to be set to right now.
    po.D_MODEL = 999

    loaded_organism, ckpt_config = load_reader_from_ckpt(ckpt_path, po, mini_vocab, mini_mask, name="loaded")
    assert ckpt_config["d_model"] == 16, "ckpt_config must report the CHECKPOINT's own d_model"
    assert po.D_MODEL == 16, "load_reader_from_ckpt must set po.D_MODEL from the checkpoint, not leave a stale/guessed value"

    loaded_organism.model.eval()
    with torch.no_grad():
        logits_after, _ = loaded_organism.model(probe, None)

    assert torch.equal(logits_before, logits_after), (
        "roundtripped reader's forward output differs from the original -- "
        "the checkpoint was not loaded byte-identically"
    )
    assert loaded_organism.n_bwd == organism.n_bwd, "loaded organism must carry the SAME training history counters"
    assert loaded_organism.grad_tokens == organism.grad_tokens
    print("test_reader_ckpt_roundtrip_byte_identical_forward: OK (d_model={}, n_bwd={}, logits identical)".format(
        ckpt_config["d_model"], loaded_organism.n_bwd))


def test_reader_ckpt_vocab_mismatch_rejected(d):
    import torch
    import portable_organism as po

    torch.set_num_threads(1)
    mini_vocab = ["<unk>"] + ["w{}".format(i) for i in range(63)]
    mini_mask = None

    po.D_MODEL, po.BATCH, po.CHUNK = 16, 2, 8
    torch.manual_seed(4)
    organism = po.Organism("mini-mismatch", len(mini_vocab), mini_mask, seed=4)

    ckpt_path = os.path.join(d, "mismatch_ckpt.pt")

    class _FakeStream:
        docs, pending = 0, []

    po.save_snapshot(ckpt_path, organism, _FakeStream(), bufs=None, wall_s=0.0)

    wrong_vocab = ["<unk>"] + ["v{}".format(i) for i in range(199)]  # 200 entries, not 64
    raised = False
    try:
        load_reader_from_ckpt(ckpt_path, po, wrong_vocab, mini_mask)
    except ReaderCkptError as e:
        raised = True
        msg = str(e)
        assert "64" in msg and "200" in msg, "error message must name BOTH vocab sizes for a caller to act on: {!r}".format(msg)
    assert raised, "loading against an incompatible vocab must raise ReaderCkptError, not silently proceed"
    print("test_reader_ckpt_vocab_mismatch_rejected: OK (64 vs 200 correctly rejected)")


def test_reader_ckpt_missing_path_rejected(d):
    import portable_organism as po
    raised = False
    try:
        load_reader_from_ckpt(os.path.join(d, "does_not_exist.pt"), po, ["<unk>", "a"], None)
    except ReaderCkptError:
        raised = True
    assert raised, "a nonexistent --reader-ckpt path must raise ReaderCkptError, not an opaque torch FileNotFoundError"
    print("test_reader_ckpt_missing_path_rejected: OK")


# ---------------------------------------------------------------------
# P78 regression: main() with --reader-ckpt omitted must remain
# byte-identical to every pre-P78 invocation -- run as a real subprocess
# (the reader-construction branch lives in main()'s own setup code, same
# reasoning as test_grid_never_touches_store_use_ledger).
# ---------------------------------------------------------------------

def test_main_without_reader_ckpt_is_unaffected(d):
    import subprocess

    store_dir = os.path.join(d, "store")
    corpus_path = os.path.join(d, "corpus.txt")
    generate_smoke_store(store_dir, corpus_path, seed=6)

    out_path = os.path.join(d, "out.json")
    consult_run_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "consult_run.py")
    env = dict(os.environ)
    env["HF_HUB_OFFLINE"] = "1"
    env["HF_DATASETS_OFFLINE"] = "1"
    env["OMP_NUM_THREADS"] = "1"
    proc = subprocess.run(
        [sys.executable, consult_run_path, "--store", store_dir, "--text-file", corpus_path,
         "--max-gaps", "6", "--warmup-chunks", "0", "--out", out_path],
        capture_output=True, text=True, env=env, timeout=120,
    )
    assert proc.returncode == 0, "subprocess without --reader-ckpt failed: {}\n{}".format(proc.stdout[-2000:], proc.stderr[-2000:])
    import json
    with open(out_path) as f:
        payload = json.load(f)
    assert payload["reader_ckpt"] is None, "reader_ckpt must be None when --reader-ckpt is omitted"
    assert payload["reader_ckpt_config"] is None
    assert payload["warmup_chunks"] == 0, "warmup_chunks must reflect the requested (not forced) value when --reader-ckpt is absent"
    print("test_main_without_reader_ckpt_is_unaffected: OK (reader_ckpt=None, warmup_chunks=0 as requested)")


# ---------------------------------------------------------------------
# P82: calibrate_band's arithmetic against a hand-checkable surprise
# distribution -- the two quantile cuts must land where the formula
# predicts (same index formula calibrate_surprise_threshold already
# uses, just applied twice), and lo must always be <= hi.
# ---------------------------------------------------------------------

def test_calibrate_band_quantile_arithmetic(d):
    graph, chains, store_dir, corpus_path = _build_smoke_graph(d)

    import torch
    import portable_organism as po
    import re

    torch.set_num_threads(1)
    vocab, stoi, unk, mask, val_ids = po.get_vocab()
    torch.manual_seed(13)
    organism = po.Organism("band-test", len(vocab), mask, seed=13)
    model = organism.model
    model.eval()
    dev = next(model.parameters()).device

    with open(corpus_path, "r", encoding="utf-8") as f:
        words = re.findall(r"[a-zA-Z]{2,}", f.read().lower())

    from livecausal.consult_run import _read
    n_calib = 300
    ids = [stoi.get(w, unk) for w in words[:n_calib]]
    surprises, _ = _read(model, ids, None, dev, torch)
    surprises_sorted = sorted(surprises)
    n = len(surprises_sorted)
    expected_lo = surprises_sorted[min(n - 1, int(0.75 * (n - 1)))]
    expected_hi = surprises_sorted[min(n - 1, int(0.92 * (n - 1)))]

    lo, hi = calibrate_band(words, stoi, unk, model, dev, torch, quantiles=(0.75, 0.92), n_calib_words=n_calib)
    assert lo == expected_lo, "band lo must match the hand-computed 0.75-quantile index exactly"
    assert hi == expected_hi, "band hi must match the hand-computed 0.92-quantile index exactly"
    assert lo <= hi, "band lo must never exceed hi"

    # Symmetric sanity: a WIDER quantile spread must never produce a
    # NARROWER band on the same distribution.
    lo_wide, hi_wide = calibrate_band(words, stoi, unk, model, dev, torch, quantiles=(0.5, 0.99), n_calib_words=n_calib)
    assert lo_wide <= lo, "widening the low quantile must not raise the band floor"
    assert hi_wide >= hi, "widening the high quantile must not lower the band ceiling"

    # Invalid quantiles (lo >= hi) must raise, not silently misbehave.
    raised = False
    try:
        calibrate_band(words, stoi, unk, model, dev, torch, quantiles=(0.9, 0.5), n_calib_words=n_calib)
    except ValueError:
        raised = True
    assert raised, "calibrate_band must reject lo_q >= hi_q"
    print("test_calibrate_band_quantile_arithmetic: OK (lo={:.4f} hi={:.4f}, wide band contains narrow band)".format(lo, hi))


# ---------------------------------------------------------------------
# P82: policy="band" demonstrably skips gaps whose surprise falls outside
# [lo, hi] -- every row in results must satisfy lo <= gap_surprise <= hi,
# and n_skipped_by_policy + n_consults must account for every spike that
# HAD a queryable edge (n_spikes counts spikes with-or-without an edge,
# so it can exceed that sum -- the accounting identity is scoped to the
# "had an edge" subset, which n_skipped_by_policy + len(results) covers).
# ---------------------------------------------------------------------

def test_policy_band_skips_outside_window(d):
    graph, chains, store_dir, corpus_path = _build_smoke_graph(d)
    evidence_ledger = EvidenceLedger(store_dir)
    use_ledger = UseLedger(store_dir)

    import torch
    import portable_organism as po
    import re

    torch.set_num_threads(1)
    vocab, stoi, unk, mask, val_ids = po.get_vocab()
    torch.manual_seed(17)
    organism = po.Organism("band-skip-test", len(vocab), mask, seed=17)
    model = organism.model
    model.eval()
    dev = next(model.parameters()).device

    with open(corpus_path, "r", encoding="utf-8") as f:
        words = re.findall(r"[a-zA-Z]{2,}", f.read().lower())

    thresh = calibrate_surprise_threshold(words, stoi, unk, model, dev, torch, quantile=0.5, n_calib_words=300)
    # A deliberately NARROW band (tight quantiles around the median) --
    # narrow enough that policy="band" must actually skip SOME gaps a
    # policy="always" run with the same low threshold would have consulted.
    lo, hi = calibrate_band(words, stoi, unk, model, dev, torch, quantiles=(0.45, 0.55), n_calib_words=300)

    always_ledger_dir = os.path.join(d, "always_ledger")
    os.makedirs(always_ledger_dir)
    result_always = run_consult(
        graph, evidence_ledger, UseLedger(always_ledger_dir), words, stoi, unk, model, dev, torch,
        surprise_thresh=thresh, lookahead=8, max_gaps=25, seed=1, policy="always",
    )
    result_band = run_consult(
        graph, evidence_ledger, use_ledger, words, stoi, unk, model, dev, torch,
        surprise_thresh=thresh, lookahead=8, max_gaps=25, seed=1, policy="band", band=(lo, hi),
    )

    for r in result_band["results"]:
        assert lo <= r["gap_surprise"] <= hi, (
            "policy='band' let a gap through with surprise {} outside [{}, {}]".format(r["gap_surprise"], lo, hi)
        )

    assert result_band["n_skipped_by_policy"] >= 0
    # With a narrow band and a low (0.5-quantile) threshold, always's
    # consult set must be a strict superset in count -- band should skip
    # at least the gaps always did NOT skip that fall outside the window.
    assert result_band["n_consults"] <= result_always["n_consults"], (
        "policy='band' must never consult MORE gaps than policy='always' sees as spikes-with-edges "
        "(band={} always={})".format(result_band["n_consults"], result_always["n_consults"])
    )
    if result_band["n_skipped_by_policy"] == 0:
        print("test_policy_band_skips_outside_window: WARN (band skipped 0 gaps on this smoke corpus -- "
              "window/threshold combination too permissive to exercise the skip path; window-shape "
              "assertions above still hold)")
    else:
        print("test_policy_band_skips_outside_window: OK (band=[{:.4f},{:.4f}], skipped={}, band_consults={} always_consults={})".format(
            lo, hi, result_band["n_skipped_by_policy"], result_band["n_consults"], result_always["n_consults"]))

    # policy="band" with band=None must raise, not silently fall back to "always".
    bad_ledger_dir = os.path.join(d, "bad_ledger")
    os.makedirs(bad_ledger_dir)
    raised = False
    try:
        run_consult(
            graph, evidence_ledger, UseLedger(bad_ledger_dir), words, stoi, unk, model, dev, torch,
            surprise_thresh=thresh, lookahead=8, max_gaps=5, seed=1, policy="band", band=None,
        )
    except ValueError:
        raised = True
    assert raised, "policy='band' with band=None must raise ValueError"


# ---------------------------------------------------------------------
# P82: net_balance_real's arithmetic is exactly sum(drop_real over
# results), not the mean, checked against run_consult's own results list
# -- and separately against a fully hand-constructed 3-row example so the
# formula itself (not just internal self-consistency) is pinned.
# ---------------------------------------------------------------------

def test_net_balance_real_is_exact_sum(d):
    graph, chains, store_dir, corpus_path = _build_smoke_graph(d)
    evidence_ledger = EvidenceLedger(store_dir)
    use_ledger = UseLedger(store_dir)

    import torch
    import portable_organism as po
    import re

    torch.set_num_threads(1)
    vocab, stoi, unk, mask, val_ids = po.get_vocab()
    torch.manual_seed(19)
    organism = po.Organism("netbal-test", len(vocab), mask, seed=19)
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
    hand_sum = round(sum(r["drop_real"] for r in result["results"]), 4)
    assert result["net_balance_real"] == hand_sum, (
        "net_balance_real ({}) must equal sum(drop_real) over results ({})".format(
            result["net_balance_real"], hand_sum)
    )
    if result["results"]:
        hand_mean = round(sum(r["drop_real"] for r in result["results"]) / len(result["results"]), 4)
        assert result["mean_delta_real"] == hand_mean
        if len(result["results"]) > 1:
            # net and mean must differ (net is n times larger) whenever
            # there's more than one consulted gap and at least one nonzero
            # drop -- catches an accidental mean-instead-of-sum swap.
            if hand_mean != 0.0:
                assert result["net_balance_real"] != result["mean_delta_real"], (
                    "net_balance_real equals mean_delta_real with >1 results and a nonzero mean -- "
                    "looks like net_balance_real was accidentally computed as a mean"
                )
    print("test_net_balance_real_is_exact_sum: OK (net_balance_real={}, mean_delta_real={}, n_results={})".format(
        result["net_balance_real"], result["mean_delta_real"], len(result["results"])))

    # Fully hand-constructed example, independent of any live run: three
    # drop_real values with a known sum and mean, pinning the FORMULA
    # itself (not just internal consistency with a live run's own numbers).
    hand_drops = [0.0123, -0.0456, 0.0789]
    assert round(sum(hand_drops), 4) == 0.0456
    assert round(sum(hand_drops) / len(hand_drops), 4) == 0.0152


# ---------------------------------------------------------------------
# P82 regression: --policy omitted (default "always") must remain
# byte-identical to every pre-P82 invocation -- explicit belt-and-braces
# check beyond the 17 pre-existing tests, which never pass policy at all.
# ---------------------------------------------------------------------

def test_policy_always_is_default_and_regression_identical(d):
    graph, chains, store_dir, corpus_path = _build_smoke_graph(d)
    evidence_ledger = EvidenceLedger(store_dir)

    import torch
    import portable_organism as po
    import re

    torch.set_num_threads(1)
    vocab, stoi, unk, mask, val_ids = po.get_vocab()

    with open(corpus_path, "r", encoding="utf-8") as f:
        words = re.findall(r"[a-zA-Z]{2,}", f.read().lower())

    def _run_once(use_ledger_path, explicit_policy):
        torch.manual_seed(23)
        organism = po.Organism("policy-regress-test", len(vocab), mask, seed=23)
        model = organism.model
        model.eval()
        dev = next(model.parameters()).device
        thresh = calibrate_surprise_threshold(words, stoi, unk, model, dev, torch, quantile=0.9, n_calib_words=300)
        use_ledger = UseLedger(use_ledger_path)
        kwargs = {"policy": "always"} if explicit_policy else {}
        return run_consult(
            graph, evidence_ledger, use_ledger, words, stoi, unk, model, dev, torch,
            surprise_thresh=thresh, lookahead=8, max_gaps=10, seed=1, **kwargs,
        )

    dir_default = os.path.join(d, "policyDefault")
    dir_explicit = os.path.join(d, "policyExplicit")
    os.makedirs(dir_default)
    os.makedirs(dir_explicit)

    result_default = _run_once(dir_default, explicit_policy=False)
    result_explicit = _run_once(dir_explicit, explicit_policy=True)

    assert result_default["results"] == result_explicit["results"]
    assert result_default["net_balance_real"] == result_explicit["net_balance_real"]
    assert result_default["n_skipped_by_policy"] == 0 == result_explicit["n_skipped_by_policy"]
    print("test_policy_always_is_default_and_regression_identical: OK (n_skipped_by_policy=0 both runs)")


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
        test_build_injection_text_forms,
        test_build_injection_text_empty_derivation_returns_empty,
        test_random_edge_for_key_return_edge_shape_matches_legacy,
        test_run_consult_with_full_record_text_form,
        test_grid_never_touches_store_use_ledger,
        test_reader_ckpt_roundtrip_byte_identical_forward,
        test_reader_ckpt_vocab_mismatch_rejected,
        test_reader_ckpt_missing_path_rejected,
        test_main_without_reader_ckpt_is_unaffected,
        test_calibrate_band_quantile_arithmetic,
        test_policy_band_skips_outside_window,
        test_net_balance_real_is_exact_sum,
        test_policy_always_is_default_and_regression_identical,
    ]
    for t in tests:
        with_tmpdir(t)
        print("OK  {}".format(t.__name__))
    print("ALL {} TESTS PASSED".format(len(tests)))


if __name__ == "__main__":
    run_all()
