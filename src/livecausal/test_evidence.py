"""Plain-assert tests for the evidence calculus (src/livecausal/evidence.py).
Run: python3 src/livecausal/test_evidence.py

Covers analysis/EVIDENCE_CALCULUS_DRAFT.md's binding recommendations:
fold reproducibility (a second, independent ledger mount recomputes the
same numbers), drop-invalidation (dead ledger lines filtered, not
rewritten), conflict coexistence + read-time dominance flip, contested
propagation through infer.py's derivations, and use-ledger sequencing.
"""

import os
import shutil
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from livecausal.infer import LiveGraph
from livecausal.evidence import (
    EvidenceLedger,
    UseLedger,
    DEFAULT_DOC_WINDOW_W,
    contested,
    contested_for_derivation,
    dominance,
    is_dominant,
)


def with_tmpdir(fn):
    d = tempfile.mkdtemp(prefix="evidence-test-")
    try:
        fn(d)
    finally:
        shutil.rmtree(d, ignore_errors=True)


def make_record(from_key, to_key, doc_coord, mechanism="causes"):
    return {
        "trigger": "t::{}".format(from_key),
        "mechanism": mechanism,
        "outcome": "o::{}".format(to_key),
        "trigger_key": from_key,
        "outcome_key": to_key,
        "doc_coord": doc_coord,
        "evidence_count": 1,
        "use_count": 0,
        "meta": {},
    }


def append_and_log(graph, ledger, records):
    sha = graph.store.append_segment(records)
    graph.on_append(sha)
    ledger.append_observations_for_segment(graph, sha)
    return sha


# ---------------------------------------------------------------------
# Test 1: fold reproducibility -- a second, independent EvidenceLedger
# mount over the same store directory recomputes the identical count.
# ---------------------------------------------------------------------

def test_fold_reproducible_on_fresh_mount(d):
    graph = LiveGraph(d)
    led = EvidenceLedger(d)

    # Same edge A->B, three structured doc_coords: two distinct documents
    # (1 and 2) plus a second occurrence within document 2 (should not
    # count as a third independent source).
    append_and_log(graph, led, [make_record("A", "B", doc_coord=[1, 0])])
    append_and_log(graph, led, [make_record("A", "B", doc_coord=[2, 0])])
    append_and_log(graph, led, [make_record("A", "B", doc_coord=[2, 7])])

    valid = graph.store.segments()
    count_first_mount = led.evidence_count(("A", "B"), valid)
    assert count_first_mount == 2, "expected 2 independent documents, got {}".format(count_first_mount)

    # Fresh, independent ledger object over the SAME store_dir: must fold
    # to the identical number purely by reading the file from scratch.
    led2 = EvidenceLedger(d)
    count_second_mount = led2.evidence_count(("A", "B"), valid)
    assert count_second_mount == count_first_mount, (
        "fold is not reproducible: {} vs {}".format(count_first_mount, count_second_mount)
    )

    # Window W must be self-described in the ledger header (Lead's review
    # note): a fresh mount picks up the SAME w without being told it again.
    assert led2.window_w == led.window_w == DEFAULT_DOC_WINDOW_W


# ---------------------------------------------------------------------
# Test 2: drop-filtering -- dropping the segment that supplied one
# document's evidence reduces the fold, without rewriting the ledger.
# ---------------------------------------------------------------------

def test_drop_filters_dead_evidence_lines(d):
    graph = LiveGraph(d)
    led = EvidenceLedger(d)

    sha_doc1 = append_and_log(graph, led, [make_record("A", "B", doc_coord=[1, 0])])
    sha_doc2a = append_and_log(graph, led, [make_record("A", "B", doc_coord=[2, 0])])
    sha_doc2b = append_and_log(graph, led, [make_record("A", "B", doc_coord=[2, 9])])

    valid_before = graph.store.segments()
    assert led.evidence_count(("A", "B"), valid_before) == 2

    # Drop BOTH segments that observed document 2 -> document 2's evidence
    # vanishes entirely; document 1's does not.
    graph.drop_segments([sha_doc2a, sha_doc2b])
    valid_after = graph.store.segments()
    assert led.evidence_count(("A", "B"), valid_after) == 1, (
        "dropping all of document 2's segments should leave only document 1's evidence"
    )

    # The ledger FILE itself is untouched (invalidation is a read-time
    # filter, not a rewrite) -- the dead lines are still physically present.
    with open(led.path, "r", encoding="utf-8") as f:
        raw_lines = [l for l in f.readlines() if l.strip()]
    # header + 3 observation lines, none removed
    assert len(raw_lines) == 4, "ledger file should be untouched by drop (append-only, no rewrite)"


# ---------------------------------------------------------------------
# Test 3: conflict coexistence + dominance flip -- both sides of a
# conflicting node-pair remain queryable (never suppressed at write
# time); the read-time dominant side flips as evidence accumulates.
# ---------------------------------------------------------------------

def test_conflict_coexistence_and_dominance_flip(d):
    graph = LiveGraph(d)
    led = EvidenceLedger(d)

    edge_pos = ("A", "B_pos")
    edge_neg = ("A", "B_neg")

    # Start 1-1: neither side should dominate (ratio 1.0 < 2.0 default).
    append_and_log(graph, led, [make_record("A", "B_pos", doc_coord=[1, 0], mechanism="causes")])
    append_and_log(graph, led, [make_record("A", "B_neg", doc_coord=[100, 0], mechanism="prevents")])

    valid = graph.store.segments()
    # Coexistence: both edges are still present as base edges in the graph
    # (never suppressed at write time).
    assert "B_pos" in graph._base_edges.get("A", {})
    assert "B_neg" in graph._base_edges.get("A", {})

    dom1 = dominance(led, edge_pos, edge_neg, valid)
    assert dom1["count_a"] == 1 and dom1["count_b"] == 1
    assert is_dominant(dom1) is None, "1-vs-1 must not be dominant"
    assert contested(led, edge_pos, edge_neg, valid) is True, "both sides have evidence -> contested"

    # Add 3 more independent documents' worth of evidence for edge_pos ->
    # now 4-vs-1, clears both the 2x ratio and the floor=2 default.
    for i in range(2, 5):
        append_and_log(graph, led, [make_record("A", "B_pos", doc_coord=[i, 0], mechanism="causes")])

    valid2 = graph.store.segments()
    dom2 = dominance(led, edge_pos, edge_neg, valid2)
    assert dom2["count_a"] == 4 and dom2["count_b"] == 1
    winner = is_dominant(dom2)
    assert winner == edge_pos, "expected edge_pos to dominate 4-vs-1, got {}".format(winner)

    # Still coexisting in storage -- dominance never deletes the loser.
    assert "B_neg" in graph._base_edges.get("A", {}), "the minority edge must survive in storage"
    assert contested(led, edge_pos, edge_neg, valid2) is True, "still contested (both sides nonzero)"


# ---------------------------------------------------------------------
# Test 3b: contested status propagates through infer.py's derivation
# chain -- SS2.3's inheritance rule, using infer.py's own
# edge_keys_for_derivation hook (never re-deriving chains inside
# evidence.py itself).
# ---------------------------------------------------------------------

def test_contested_propagates_through_derivation(d):
    graph = LiveGraph(d)
    led = EvidenceLedger(d)

    # A->B contested (built via a real conflicting pair with dominance
    # NOT yet decisive); B->C uncontested.
    append_and_log(graph, led, [make_record("A", "B", doc_coord=[1, 0])])
    append_and_log(graph, led, [make_record("A", "B_alt", doc_coord=[2, 0])])  # conflicts with A->B
    append_and_log(graph, led, [make_record("B", "C", doc_coord=[1, 0])])

    valid = graph.store.segments()
    ab_contested = contested(led, ("A", "B"), ("A", "B_alt"), valid)
    assert ab_contested is True

    def base_contested_lookup(ek):
        return ek == ("A", "B")  # the only edge we've established as contested

    inferred = [e for e in graph.inferred_edges() if e["from_key"] == "A" and e["to_key"] == "C"]
    assert inferred, "expected the transitive A->C edge to exist"
    edge = inferred[0]
    hop_keys = graph.edge_keys_for_derivation(edge["derivation"])
    assert hop_keys == [("A", "B"), ("B", "C")]

    result = contested_for_derivation(base_contested_lookup, edge["derivation"], hop_keys)
    assert result is True, "inferred edge must inherit contested status from its A->B hop"

    # And an edge whose derivation never touches a contested hop must not
    # be flagged.
    def nothing_contested(ek):
        return False

    result2 = contested_for_derivation(nothing_contested, edge["derivation"], hop_keys)
    assert result2 is False


# ---------------------------------------------------------------------
# Test 4: use-ledger sequencing -- logical sequence numbers, not
# timestamps; use_count folds correctly and respects drop-invalidation
# exactly like the evidence ledger.
# ---------------------------------------------------------------------

def test_use_ledger_sequence_and_invalidation(d):
    graph = LiveGraph(d)
    sha = graph.store.append_segment([make_record("A", "B", doc_coord=[1, 0])])
    graph.on_append(sha)

    use_led = UseLedger(d)
    # No wall-clock anywhere in the line shape.
    line = use_led.append_use(("A", "B"), seq=1, sha=sha, idx=0)
    assert "ts" not in line and "timestamp" not in line and "time" not in line
    assert line["seq"] == 1

    use_led.append_use(("A", "B"), seq=2, sha=sha, idx=0)
    use_led.append_use(("A", "B"), seq=3, sha=sha, idx=0)

    valid = graph.store.segments()
    assert use_led.use_count(("A", "B"), valid) == 3
    assert use_led.max_seq() == 3

    # A second, independent mount recomputes identically (fold
    # reproducibility, same requirement as the evidence ledger).
    use_led2 = UseLedger(d)
    assert use_led2.use_count(("A", "B"), valid) == 3
    assert use_led2.max_seq() == 3

    # Drop the cited segment -> use_count folds to 0 (dead lines filtered,
    # not rewritten); max_seq is UNAFFECTED (it's the caller's own logical
    # counter, not scoped to live segments).
    graph.drop_segments([sha])
    valid_after = graph.store.segments()
    assert use_led.use_count(("A", "B"), valid_after) == 0
    assert use_led.max_seq() == 3


def run_all():
    tests = [
        test_fold_reproducible_on_fresh_mount,
        test_drop_filters_dead_evidence_lines,
        test_conflict_coexistence_and_dominance_flip,
        test_contested_propagates_through_derivation,
        test_use_ledger_sequence_and_invalidation,
    ]
    for t in tests:
        with_tmpdir(t)
        print("OK  {}".format(t.__name__))
    print("ALL {} TESTS PASSED".format(len(tests)))


if __name__ == "__main__":
    run_all()
