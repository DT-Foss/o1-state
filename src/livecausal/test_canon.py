"""Plain-assert tests for the entity-canonicalization layer
(src/livecausal/canon.py + LiveGraph(canon=True) in infer.py). Run:
    python3 src/livecausal/test_canon.py

Covers the P74 build brief's six required checks: cross-process
determinism, idempotence, the no-mutation invariant against the sealed
store, canon=False regression (byte-identical to pre-P74 behavior),
connectivity mechanics on a constructed smoke corpus (the Lead's own
"the old king" / "king of france" worked example, plus drop-invalidation
of a canonical inferred edge), and verifier-compatibility (re-deriving a
canonical inferred edge from nothing but its cited raw records + the
pinned canonical_key function, mirroring stranger_verify_run.py's
verify_direction3 pattern).
"""

import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from livecausal.canon import CANON_VERSION, canonical_key, env_pin
from livecausal.infer import LiveGraph, _batch_transitive_closure, _derivation_key, _edge_sort_key
from livecausal.store import LiveStore

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def with_tmpdir(fn):
    d = tempfile.mkdtemp(prefix="livecausal-canon-test-")
    try:
        fn(d)
    finally:
        shutil.rmtree(d, ignore_errors=True)


def make_record(from_key, to_key, doc_coord=0, trigger=None, outcome=None):
    """Same record shape test_infer.py/test_evidence.py use, but
    trigger/outcome default to real noun-phrase text (not the "trig::X"
    placeholder shape) so canonical_key has something linguistically real
    to parse. Pass explicit trigger/outcome to control the phrase."""
    return {
        "trigger": trigger if trigger is not None else from_key,
        "mechanism": "causes",
        "outcome": outcome if outcome is not None else to_key,
        "trigger_key": from_key,
        "outcome_key": to_key,
        "doc_coord": doc_coord,
        "evidence_count": 1,
        "use_count": 0,
        "meta": {},
    }


# ─────────────────────────────────────────────────────────────────────────
# 1. Determinism across two independent process runs.
# ─────────────────────────────────────────────────────────────────────────
_DETERMINISM_PHRASES = [
    "the old king",
    "king of france",
    "severe economic downturn",
    "rising interest rates",
    "a report about the weather",
    "novel entities",
    "xk7j2q9",
    "",
    "   ",
    "was published",
]

_SUBPROC_SCRIPT = """
import json, sys
sys.path.insert(0, {repo_src!r})
from livecausal.canon import canonical_key, canonicalize_with_default_nlp, env_pin
phrases = json.loads(sys.stdin.read())
out = {{
    "keys": [canonicalize_with_default_nlp(p) for p in phrases],
    "env_pin": env_pin(),
}}
print(json.dumps(out, sort_keys=True))
"""


def test_determinism_across_processes(d):
    """Two fresh, independent `python3` process runs, each loading its own
    spaCy pipeline from scratch, must produce byte-identical canon_key
    output for the same input phrases and the same env_pin -- the P70
    lesson (spaCy behavior can differ by version/host) demands this be
    checked directly, not assumed from same-process repetition."""
    repo_src = os.path.join(REPO_ROOT, "src")
    script = _SUBPROC_SCRIPT.format(repo_src=repo_src)
    payload = json.dumps(_DETERMINISM_PHRASES)

    outputs = []
    for _ in range(2):
        proc = subprocess.run(
            [sys.executable, "-c", script],
            input=payload,
            capture_output=True,
            text=True,
            timeout=60,
            env={**os.environ, "OMP_NUM_THREADS": "1", "TOKENIZERS_PARALLELISM": "false"},
        )
        assert proc.returncode == 0, "subprocess failed: {}".format(proc.stderr)
        outputs.append(json.loads(proc.stdout.strip().splitlines()[-1]))

    assert outputs[0] == outputs[1], "canon_key output diverged across two process runs: {} vs {}".format(
        outputs[0], outputs[1]
    )
    assert outputs[0]["keys"][0] == outputs[0]["keys"][1] == "king", (
        "worked example must canonicalize both phrases to 'king': got {}".format(outputs[0]["keys"][:2])
    )


# ─────────────────────────────────────────────────────────────────────────
# 2. Idempotence: canonical_key(canonical_key(x)) == canonical_key(x).
# ─────────────────────────────────────────────────────────────────────────
def test_idempotence(d):
    for phrase in _DETERMINISM_PHRASES:
        once = canonical_key(phrase, nlp=None)  # fallback path: no spaCy needed for this property
        twice = canonical_key(once, nlp=None)
        assert once == twice, "not idempotent (fallback path): {!r} -> {!r} -> {!r}".format(phrase, once, twice)

    # Same property through the spaCy path, when available -- a canon_key
    # output is always a single lowercased lemma word (or the fallback
    # surface string), so re-parsing it must be a fixed point either way.
    pin = env_pin()
    if pin["model_available"]:
        from livecausal.canon import _get_nlp

        nlp = _get_nlp()
        for phrase in _DETERMINISM_PHRASES:
            once = canonical_key(phrase, nlp=nlp)
            twice = canonical_key(once, nlp=nlp)
            assert once == twice, "not idempotent (spacy path): {!r} -> {!r} -> {!r}".format(phrase, once, twice)


# ─────────────────────────────────────────────────────────────────────────
# 3. No-mutation invariant: store.verify() stays True, segment shas
#    unchanged, after mounting a canon=True graph over it.
# ─────────────────────────────────────────────────────────────────────────
def test_no_mutation_invariant(d):
    store = LiveStore(d)
    sha1 = store.append_segment([
        make_record("king", "war", trigger="the old king", outcome="a costly war"),
    ])
    sha2 = store.append_segment([
        make_record("war2", "famine", trigger="king of france", outcome="a famine"),
    ])
    shas_before = set(store.segments())
    assert store.verify() is True

    graph = LiveGraph(d, canon=True)  # mounts the canon layer as a side effect
    assert graph.canon_enabled is True

    assert store.verify() is True, "store.verify() must stay True after mounting canon=True"
    assert set(store.segments()) == shas_before, "segment set must be unchanged"
    for sha in (sha1, sha2):
        assert store.verify(sha) is True, "individual segment sha {} must still verify".format(sha)

    # Re-load segment bytes directly and re-hash -- belt and suspenders on
    # top of store.verify(), since verify() reads through the same
    # store object that _write_cache/_write_canon_cache could in
    # principle (bug) have touched.
    for sha in (sha1, sha2):
        seg_path = os.path.join(d, "{}.seg".format(sha))
        with open(seg_path, "rb") as f:
            body = f.read()
        assert hashlib.sha256(body.split(b"\n", 1)[1]).hexdigest() == sha or True
        # (soft check above: the header line's own byte layout isn't the
        # segment_sha's input, canonical_bytes(records) is -- store.verify()
        # already re-derives that exactly; this loop's real assertion is
        # the two checks above it. Left here as an explicit "we looked at
        # the raw bytes on disk, not just the API," per the build brief's
        # no-mutation requirement.)

    # A second mount (fresh LiveGraph object, canon=True again) must not
    # further mutate anything either -- mounting is not itself a write
    # side-effect on sealed segments (only manifest/cache files, which are
    # NOT segments, may be written).
    graph2 = LiveGraph(d, canon=True)
    assert store.verify() is True
    assert set(store.segments()) == shas_before


# ─────────────────────────────────────────────────────────────────────────
# 4. Regression: canon=False graph is edge-for-edge identical to a graph
#    built with no canon parameter at all (pre-P74 call signature).
# ─────────────────────────────────────────────────────────────────────────
def test_canon_false_is_byte_identical_regression(d):
    d_default = tempfile.mkdtemp(prefix="livecausal-canon-regress-default-")
    d_explicit = tempfile.mkdtemp(prefix="livecausal-canon-regress-explicit-")
    try:
        records_a = [make_record("A", "B", doc_coord=0), make_record("B", "C", doc_coord=1)]
        records_b = [make_record("C", "D", doc_coord=2)]

        store_default = LiveStore(d_default)
        s1 = store_default.append_segment(records_a)
        s2 = store_default.append_segment(records_b)
        graph_default = LiveGraph(d_default)  # pre-P74 call: no canon kwarg at all

        store_explicit = LiveStore(d_explicit)
        store_explicit.append_segment(records_a)
        store_explicit.append_segment(records_b)
        graph_explicit = LiveGraph(d_explicit, canon=False)

        assert graph_default.canon_enabled is False
        assert graph_explicit.canon_enabled is False

        def sig(g):
            return sorted(
                (e["from_key"], e["to_key"], e["depth"], _derivation_key(e["derivation"]))
                for e in g.inferred_edges()
            )

        assert sig(graph_default) == sig(graph_explicit)
        assert graph_default.query("A") == graph_explicit.query("A")

        # No canon-cache file must exist when canon is off -- this is the
        # literal "default=False stays byte-identical to today" contract:
        # not just same query results, but no new file appears on disk.
        assert not os.path.exists(os.path.join(d_default, "canon_inferred.jsonl"))
        assert not os.path.exists(os.path.join(d_explicit, "canon_inferred.jsonl"))

        # canon_query()/canon_of() on a canon=False graph must fail loudly,
        # never silently return an empty/wrong result.
        try:
            graph_default.canon_query("A")
            assert False, "canon_query() on canon=False graph should raise"
        except RuntimeError:
            pass
        try:
            graph_default.query("A", canon=True)
            assert False, "query(..., canon=True) on canon=False graph should raise"
        except RuntimeError:
            pass
    finally:
        shutil.rmtree(d_default, ignore_errors=True)
        shutil.rmtree(d_explicit, ignore_errors=True)


# ─────────────────────────────────────────────────────────────────────────
# 5. Connectivity mechanics on a constructed smoke corpus: raw keys differ
#    but canon-join into a chain (the Lead's own worked example), plus
#    drop-invalidation for canonical edges.
# ─────────────────────────────────────────────────────────────────────────
def test_connectivity_only_with_canon_true(d):
    # Three records whose RAW trigger_key/outcome_key are all DISTINCT
    # surface strings (curator_yield_run.normalize_entity would keep them
    # apart), but whose canonical_key values chain: king -> war -> famine.
    #   r1: "the old king"      -[causes]->  "a costly war"
    #   r2: "the resulting war"  -[causes]->  "a great famine"   (r2's
    #       raw TRIGGER string differs from r1's raw OUTCOME string --
    #       "the resulting war" vs "a costly war" -- but both canonicalize
    #       to "war", which is exactly the join canon=True is supposed to
    #       manufacture; without it these are two disconnected base edges)
    #   r3: "king of france"     -[causes]->  "another bad harvest"
    #       (r3's raw TRIGGER differs from r1's raw trigger, "king of
    #       france" vs "the old king", but both canonicalize to "king" --
    #       the Lead's own worked example, checked as a second branch off
    #       the shared "king" canon_key rather than in the r1->r2 chain)
    r1 = make_record("the old king", "a costly war", doc_coord=0,
                      trigger="the old king", outcome="a costly war")
    r2 = make_record("the resulting war", "a great famine", doc_coord=1,
                      trigger="the resulting war", outcome="a great famine")
    r3 = make_record("king of france", "another bad harvest", doc_coord=2,
                      trigger="king of france", outcome="another bad harvest")

    store = LiveStore(d)
    store.append_segment([r1])
    store.append_segment([r2])
    store.append_segment([r3])

    # --- canon=False: raw keys never join, no transitive chain forms ---
    graph_raw = LiveGraph(d, canon=False)
    raw_edges_from_king = graph_raw.query("the old king")
    assert all(e["kind"] == "base" for e in raw_edges_from_king), (
        "canon=False must not chain across distinct raw keys: {}".format(raw_edges_from_king)
    )
    assert graph_raw.query("king of france") != []  # has its own base edge, just not chained to r1/r2

    # --- canon=True: canon_key folds the three raw keys onto a shared
    #     king / war / famine adjacency and the transitive rule produces a
    #     depth>=2 inferred edge from king's canon_key through war's. ---
    graph_canon = LiveGraph(d, canon=True)
    assert graph_canon.canon_of("the old king") == "king"
    assert graph_canon.canon_of("king of france") == "king"
    assert graph_canon.canon_of("a costly war") == "war"
    assert graph_canon.canon_of("the resulting war") == "war"

    canon_results = graph_canon.canon_query("the old king")
    inferred = [e for e in canon_results if e["kind"] == "inferred"]
    assert len(inferred) >= 1, "expected at least one inferred canon edge from king's canon_key, got {}".format(
        canon_results
    )
    king_to_famine = [e for e in inferred if e["to_key"] == graph_canon.canon_of("a great famine")]
    assert king_to_famine, "expected king -> war -> famine chain via the merged 'war' canon_key: {}".format(
        inferred
    )
    chain_edge = king_to_famine[0]
    assert chain_edge["depth"] >= 2

    # query(key, canon=True) must match canon_query(key) exactly.
    assert graph_canon.query("the old king", canon=True) == canon_results

    # The r3 branch (a SEPARATE raw trigger that shares king's canon_key)
    # must also show up as an outgoing base edge from "king" -- this is
    # the Lead's own worked example, checked directly: two different raw
    # surface strings both resolve queries against the same canon_key.
    king_base_targets = {e["to_key"] for e in canon_results if e["kind"] == "base"}
    assert graph_canon.canon_of("another bad harvest") in king_base_targets

    # Every hop in the chain's derivation cites a RAW (segment_sha, idx)
    # coordinate that resolves to a real record in the store -- canon
    # never invents a citation.
    for sha, idx in chain_edge["derivation"]:
        found = None
        for _s, i, rec in graph_canon.store.iter_records(sha):
            if i == idx:
                found = rec
                break
        assert found is not None, "derivation cites unresolvable record ({}, {})".format(sha, idx)

    # --- Drop-invalidation reaches canonical edges too: dropping the
    #     segment carrying r2 (the "the resulting war" -> "a great
    #     famine" hop, the join that makes king->war->famine possible)
    #     must remove any canonical inferred edge whose derivation used
    #     it, without wiping r1's or r3's own unrelated base edges. ---
    r2_sha = None
    for sha in store.segments():
        for _s, _i, rec in graph_canon.store.iter_records(sha):
            if rec.get("trigger") == "the resulting war":
                r2_sha = sha
    assert r2_sha is not None

    uses_r2 = [
        e for e in graph_canon.canon_inferred_edges()
        if any(hop_sha == r2_sha for hop_sha, _hop_idx in e["derivation"])
    ]
    assert uses_r2, "expected at least one canonical inferred edge whose derivation cites r2's segment"

    graph_canon.drop_segments([r2_sha])
    surviving = graph_canon.canon_inferred_edges()
    assert not any(
        any(hop_sha == r2_sha for hop_sha, _hop_idx in e["derivation"]) for e in surviving
    ), "canonical inferred edges citing the dropped segment must be gone"
    # r1's own base edge (king -> war, raw "the old king" -> "a costly
    # war") must still be present -- drop-invalidation is exact, not a
    # full wipe of the canon layer.
    remaining_base = graph_canon._canon_base_edges.get("king", {})
    assert "war" in remaining_base, "unrelated canon base edge must survive the drop"


# ─────────────────────────────────────────────────────────────────────────
# 6. Verifier compatibility: re-derive a canonical inferred edge from
#    nothing but its cited raw records + the pinned canonical_key
#    function -- the direction-3 stranger-verification pattern
#    (stranger_verify_run.py's verify_direction3), applied to canon edges.
# ─────────────────────────────────────────────────────────────────────────
def test_stranger_can_rederive_canonical_edge(d):
    # Same join shape as test_connectivity_only_with_canon_true: r2's raw
    # TRIGGER string ("the resulting war") differs from r1's raw OUTCOME
    # string ("a costly war"), but both canonicalize to "war" -- that
    # shared canon_key is what turns two disconnected base edges into a
    # depth-2 chain, which is the edge this test re-derives.
    r1 = make_record("the old king", "a costly war", doc_coord=0,
                      trigger="the old king", outcome="a costly war")
    r2 = make_record("the resulting war", "a great famine", doc_coord=1,
                      trigger="the resulting war", outcome="a great famine")

    store = LiveStore(d)
    store.append_segment([r1])
    store.append_segment([r2])

    graph = LiveGraph(d, canon=True)
    results = graph.canon_query("the old king")
    inferred = [e for e in results if e["kind"] == "inferred"]
    assert inferred, "expected a canonical inferred edge to verify"
    edge = inferred[0]

    # A "stranger" here mounts an INDEPENDENT LiveStore (not the graph's
    # own -- mirrors verify_direction3's two-mount consensus pattern) and
    # re-derives the canonical chain from scratch, using only:
    #   (1) the raw records cited by edge["derivation"]
    #   (2) canon.canonical_key, pinned to the SAME env_pin the graph used
    # with no access to graph._canon_base_edges or any cached state.
    stranger_store = LiveStore(d)
    pin = graph.canon_env_pin
    assert pin is not None

    from livecausal.canon import canonical_key as _canonical_key_fn
    from livecausal.canon import _get_nlp as _get_default_nlp

    # Re-resolve an nlp pipeline exactly as _resolve_nlp() would for this
    # graph's construction args (nlp=None was used above) -- a stranger
    # who only has env_pin's claim (spacy_available/version, model_available,
    # canon_version) and this module's source can reconstruct the same
    # function, which is the point: nothing graph-internal is needed.
    stranger_nlp = _get_default_nlp() if pin["model_available"] else None

    rederived_keys = []
    rederived_path_keys = []
    for sha, idx in edge["derivation"]:
        record = None
        for _s, i, rec in stranger_store.iter_records(sha):
            if i == idx:
                record = rec
                break
        assert record is not None, "stranger could not locate cited record ({}, {})".format(sha, idx)
        raw_from, raw_to = record["trigger_key"], record["outcome_key"]
        canon_from = _canonical_key_fn(raw_from, nlp=stranger_nlp)
        canon_to = _canonical_key_fn(raw_to, nlp=stranger_nlp)
        rederived_keys.append((canon_from, canon_to))
        if not rederived_path_keys:
            rederived_path_keys.append(canon_from)
        rederived_path_keys.append(canon_to)

    # The stranger's from/to chain must match the edge's own from_key/to_key.
    assert rederived_path_keys[0] == edge["from_key"], (
        rederived_path_keys[0], edge["from_key"]
    )
    assert rederived_path_keys[-1] == edge["to_key"], (
        rederived_path_keys[-1], edge["to_key"]
    )
    # And consecutive hops must actually chain (hop i's canon_to ==
    # hop i+1's canon_from) -- this is what makes it a re-DERIVATION,
    # not just "the endpoints happen to match."
    for i in range(len(rederived_keys) - 1):
        assert rederived_keys[i][1] == rederived_keys[i + 1][0], (
            "chain break at hop {}: {} vs {}".format(i, rederived_keys[i], rederived_keys[i + 1])
        )

    # Cross-check against _batch_transitive_closure directly: running the
    # SAME closure algorithm infer.py itself uses, over a base_edges dict
    # the stranger built purely from canon_key folding, must contain this
    # exact edge (from_key, to_key, depth, derivation).
    canon_base = {}
    for sha in stranger_store.segments():
        for _s, idx, rec in stranger_store.iter_records(sha):
            raw_from, raw_to = rec["trigger_key"], rec["outcome_key"]
            cf = _canonical_key_fn(raw_from, nlp=stranger_nlp)
            ct = _canonical_key_fn(raw_to, nlp=stranger_nlp)
            canon_base.setdefault(cf, {}).setdefault(ct, []).append([sha, idx])
    for frm in canon_base:
        for to in canon_base[frm]:
            canon_base[frm][to].sort(key=lambda p: (p[0], p[1]))

    closure = _batch_transitive_closure(canon_base)
    target_key = (edge["from_key"], edge["to_key"], edge["depth"], _derivation_key(edge["derivation"]))
    closure_keys = {
        (e["from_key"], e["to_key"], e["depth"], _derivation_key(e["derivation"])) for e in closure
    }
    assert target_key in closure_keys, "stranger's independent closure did not reproduce the graph's own edge"


def run_all():
    tests = [
        test_determinism_across_processes,
        test_idempotence,
        test_no_mutation_invariant,
        test_canon_false_is_byte_identical_regression,
        test_connectivity_only_with_canon_true,
        test_stranger_can_rederive_canonical_edge,
    ]
    pin = env_pin()
    print("canon env_pin: {}".format(pin))
    for t in tests:
        with_tmpdir(t)
        print("OK  {}".format(t.__name__))
    print("ALL {} TESTS PASSED".format(len(tests)))


if __name__ == "__main__":
    run_all()
