"""Plain-assert tests for LiveGraph delta-inference. Run:
    python3 src/livecausal/test_infer.py
"""

import os
import random
import shutil
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from livecausal.infer import MAX_DEPTH, LiveGraph, _batch_transitive_closure
from livecausal.store import LiveStore


def with_tmpdir(fn):
    d = tempfile.mkdtemp(prefix="livecausal-infer-test-")
    try:
        fn(d)
    finally:
        shutil.rmtree(d, ignore_errors=True)


def make_record(from_key, to_key, doc_coord=0, evidence_count=1):
    return {
        "trigger": "trig::{}".format(from_key),
        "mechanism": "causes",
        "outcome": "outc::{}".format(to_key),
        "trigger_key": from_key,
        "outcome_key": to_key,
        "doc_coord": doc_coord,
        "evidence_count": evidence_count,
        "use_count": 0,
        "meta": {},
    }


def sorted_inferred_signature(graph):
    """A comparable, order-independent signature of the inferred-edge set:
    sorted list of (from_key, to_key, depth, tuple(derivation-as-tuples)).
    """
    sig = []
    for e in graph.inferred_edges():
        deriv = tuple((sha, idx) for sha, idx in e["derivation"])
        sig.append((e["from_key"], e["to_key"], e["depth"], deriv))
    sig.sort()
    return sig


def random_chain_batches(rng, n_batches, chain_len_range=(2, 6)):
    """Generate n_batches of segments, each segment a single-hop record in
    a chain A->B->C->..., producing multiple independent chains across
    batches. Returns a list of (list_of_records) -- one inner list per
    segment to append, in append order.
    """
    segments = []
    key_counter = [0]

    def _fresh():
        key_counter[0] += 1
        return "K{}".format(key_counter[0])

    for _ in range(n_batches):
        chain_len = rng.randint(*chain_len_range)
        keys = [_fresh() for _ in range(chain_len + 1)]
        # Each hop of this chain becomes its own segment (so append order
        # matters and on_append is exercised hop by hop).
        for i in range(chain_len):
            segments.append([make_record(keys[i], keys[i + 1], doc_coord=i)])
    return segments


# ---------------------------------------------------------------------
# Test 1: incremental (append-by-append) ≡ batch (fresh rebuild)
# ---------------------------------------------------------------------

def test_incremental_equals_batch_equivalence(d):
    rng = random.Random(12345)
    segments = random_chain_batches(rng, n_batches=5, chain_len_range=(2, 6))

    incr_dir = os.path.join(d, "incremental")
    batch_dir = os.path.join(d, "batch")
    os.makedirs(incr_dir)
    os.makedirs(batch_dir)

    incr_graph = LiveGraph(incr_dir)
    shas = []
    for records in segments:
        sha = incr_graph.store.append_segment(records)
        shas.append(sha)
        incr_graph.on_append(sha)

    # Batch: append everything to a fresh store first, then mount a fresh
    # LiveGraph over it (forces the full-rebuild path in _mount).
    batch_store = LiveStore(batch_dir)
    for records in segments:
        batch_store.append_segment(records)
    batch_graph = LiveGraph(batch_dir)
    assert batch_graph.was_rebuilt_on_mount() is True

    incr_sig = sorted_inferred_signature(incr_graph)
    batch_sig = sorted_inferred_signature(batch_graph)
    assert incr_sig == batch_sig, "incremental and batch inferred-edge sets diverge"
    assert len(incr_sig) > 0, "test is vacuous: no inferred edges produced at all"

    # Also check base edges match (sanity on top of inferred-edge equality).
    assert incr_graph._base_edges == batch_graph._base_edges


# ---------------------------------------------------------------------
# Test 2: drop a middle segment + on_drop ≡ batch rebuild without it
# ---------------------------------------------------------------------

def test_drop_equals_batch_without_segment(d):
    rng = random.Random(999)
    segments = random_chain_batches(rng, n_batches=5, chain_len_range=(3, 6))

    incr_dir = os.path.join(d, "incremental")
    batch_dir = os.path.join(d, "batch")
    os.makedirs(incr_dir)
    os.makedirs(batch_dir)

    incr_graph = LiveGraph(incr_dir)
    shas = []
    for records in segments:
        sha = incr_graph.store.append_segment(records)
        shas.append(sha)
        incr_graph.on_append(sha)

    # Drop a middle segment via the graph's drop_segments (store + graph).
    mid_idx = len(shas) // 2
    dropped_sha = shas[mid_idx]
    incr_graph.drop_segments([dropped_sha])

    # Batch reference: append everything EXCEPT that one segment's records,
    # to a fresh store, then a fresh LiveGraph (full rebuild).
    batch_store = LiveStore(batch_dir)
    for i, records in enumerate(segments):
        if i == mid_idx:
            continue
        batch_store.append_segment(records)
    batch_graph = LiveGraph(batch_dir)

    incr_sig = sorted_inferred_signature(incr_graph)
    batch_sig = sorted_inferred_signature(batch_graph)
    assert incr_sig == batch_sig, "post-drop incremental graph diverges from batch-without-segment"
    assert incr_graph._base_edges == batch_graph._base_edges

    # No surviving inferred edge may cite the dropped segment.
    for e in incr_graph.inferred_edges():
        for hop_sha, _hop_idx in e["derivation"]:
            assert hop_sha != dropped_sha


# ---------------------------------------------------------------------
# Test 3: depth cap -- a length-8 chain only infers up to depth 5
# ---------------------------------------------------------------------

def test_depth_cap_on_long_chain(d):
    graph = LiveGraph(d)
    chain_len = 8
    keys = ["C{}".format(i) for i in range(chain_len + 1)]
    for i in range(chain_len):
        sha = graph.store.append_segment([make_record(keys[i], keys[i + 1], doc_coord=i)])
        graph.on_append(sha)

    depths = sorted(set(e["depth"] for e in graph.inferred_edges()))
    assert depths, "expected some inferred edges on an 8-hop chain"
    assert max(depths) == MAX_DEPTH, "depth cap not enforced: max depth {} (expected {})".format(
        max(depths), MAX_DEPTH
    )
    assert all(2 <= d_ <= MAX_DEPTH for d_ in depths)

    # The longest possible depth-5 chain should exist: C0->C1->...->C5.
    longest = [e for e in graph.inferred_edges() if e["depth"] == MAX_DEPTH]
    assert any(e["from_key"] == "C0" and e["to_key"] == "C5" for e in longest)
    # And nothing reaches C6/C7/C8 from C0 (would require depth 6/7/8).
    assert not any(e["from_key"] == "C0" and e["to_key"] in ("C6", "C7", "C8") for e in graph.inferred_edges())


# ---------------------------------------------------------------------
# Test 4: derivation correctness -- every inferred edge is re-derivable
# from its cited base records alone.
# ---------------------------------------------------------------------

def test_derivation_is_recomputable(d):
    rng = random.Random(4242)
    segments = random_chain_batches(rng, n_batches=4, chain_len_range=(2, 5))

    graph = LiveGraph(d)
    for records in segments:
        sha = graph.store.append_segment(records)
        graph.on_append(sha)

    # Rebuild a raw (sha, idx) -> record lookup straight from the store,
    # independent of the graph's internal indices.
    record_by_coord = {}
    for sha, idx, record in graph.store.iter_records():
        record_by_coord[(sha, idx)] = record

    checked = 0
    for e in graph.inferred_edges():
        # Re-walk the derivation: hop i must be a base record whose
        # trigger_key/outcome_key chain from e["from_key"] to e["to_key"].
        derivation = e["derivation"]
        assert len(derivation) == e["depth"]
        cursor = e["from_key"]
        seen_keys = {cursor}
        for sha, idx in derivation:
            rec = record_by_coord.get((sha, idx))
            assert rec is not None, "derivation cites a record not present in the store"
            assert rec["trigger_key"] == cursor, "derivation hop does not chain from the current key"
            nxt = rec["outcome_key"]
            assert nxt not in seen_keys, "recomputed chain is not cycle-free"
            seen_keys.add(nxt)
            cursor = nxt
        assert cursor == e["to_key"], "derivation does not land on the claimed to_key"
        checked += 1

    assert checked > 0, "test is vacuous: no inferred edges to check"


# ---------------------------------------------------------------------
# Test 5: cache validity -- matching manifest stamp skips rebuild;
# a changed manifest forces one.
# ---------------------------------------------------------------------

def test_cache_validity_mount_behavior(d):
    graph = LiveGraph(d)
    sha1 = graph.store.append_segment([make_record("A", "B", doc_coord=0)])
    graph.on_append(sha1)
    sha2 = graph.store.append_segment([make_record("B", "C", doc_coord=0)])
    graph.on_append(sha2)

    assert graph.was_rebuilt_on_mount() is True  # fresh dir, no cache yet at construction

    # Re-mount over the same dir: cache file now exists and matches the
    # manifest stamp -> must load from cache, not rebuild.
    graph2 = LiveGraph(d)
    assert graph2.was_loaded_from_cache() is True
    assert graph2.was_rebuilt_on_mount() is False
    assert sorted_inferred_signature(graph2) == sorted_inferred_signature(graph)

    # Now change the manifest out from under the cache (append directly via
    # the store, bypassing on_append, so the cache file is left stale).
    sha3 = graph2.store.append_segment([make_record("C", "D", doc_coord=0)])

    graph3 = LiveGraph(d)
    assert graph3.was_loaded_from_cache() is False
    assert graph3.was_rebuilt_on_mount() is True
    # And the rebuild picked up the new segment's consequences: A->B->C->D
    # is a 3-hop chain, so the inferred A->D edge has depth 3.
    assert any(
        e["from_key"] == "A" and e["to_key"] == "D" and e["depth"] == 3
        for e in graph3.inferred_edges()
    )


def run_all():
    tests = [
        test_incremental_equals_batch_equivalence,
        test_drop_equals_batch_without_segment,
        test_depth_cap_on_long_chain,
        test_derivation_is_recomputable,
        test_cache_validity_mount_behavior,
    ]
    for t in tests:
        with_tmpdir(t)
        print("OK  {}".format(t.__name__))
    print("ALL {} TESTS PASSED".format(len(tests)))


if __name__ == "__main__":
    run_all()
