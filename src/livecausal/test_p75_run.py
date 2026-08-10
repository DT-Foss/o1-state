"""Plain-assert tests for p75_run.py's OWN building blocks: deterministic
sampling, re-derivation on a locally-built smoke store, and the
no-mutation check. NOT a measurement against any real P72/P73/P74
artifact -- every store here is built fresh, locally, offline.

Run: python3 src/livecausal/test_p75_run.py
"""
import os
import shutil
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("HF_DATASETS_OFFLINE", "1")

from livecausal import p75_run
from livecausal.infer import LiveGraph
from livecausal.store import LiveStore


def _build_smoke_store(tmp_root, seed=42, max_windows=60):
    store_dir = os.path.join(tmp_root, "smoke_source")
    p75_run.build_smoke_source_store(store_dir, seed=seed, max_windows=max_windows)
    return store_dir


def test_clone_store_never_touches_source():
    tmp_root = tempfile.mkdtemp(prefix="p75-test-clone-")
    try:
        source_dir = _build_smoke_store(tmp_root)
        before_shas = LiveStore(source_dir).segments()
        before_mtimes = {
            f: os.path.getmtime(os.path.join(source_dir, f))
            for f in os.listdir(source_dir)
        }

        dest_dir = os.path.join(tmp_root, "copy1")
        p75_run.clone_store(source_dir, dest_dir)

        after_shas = LiveStore(source_dir).segments()
        after_mtimes = {
            f: os.path.getmtime(os.path.join(source_dir, f))
            for f in os.listdir(source_dir)
        }
        assert before_shas == after_shas, "source segment list changed after clone_store"
        assert before_mtimes == after_mtimes, "source file mtimes changed after clone_store -- something wrote into it"
        assert os.path.exists(dest_dir)
        assert LiveStore(dest_dir).segments() == before_shas

        # clone_store must refuse to clone onto an existing dest_dir
        raised = False
        try:
            p75_run.clone_store(source_dir, dest_dir)
        except ValueError:
            raised = True
        assert raised, "clone_store did not refuse an existing dest_dir"
        print("test_clone_store_never_touches_source: OK ({} segments, dest guard OK)".format(len(before_shas)))
    finally:
        shutil.rmtree(tmp_root, ignore_errors=True)


def test_sampling_is_deterministic():
    tmp_root = tempfile.mkdtemp(prefix="p75-test-sample-")
    try:
        source_dir = _build_smoke_store(tmp_root)
        copy_dir = os.path.join(tmp_root, "copy")
        p75_run.clone_store(source_dir, copy_dir)

        graph_a = LiveGraph(copy_dir, canon=True)
        moat_a = p75_run.measure_moat(copy_dir, graph_a, n_samples=10, seed=777)

        graph_b = LiveGraph(copy_dir, canon=True)
        moat_b = p75_run.measure_moat(copy_dir, graph_b, n_samples=10, seed=777)

        keys_a = [(r["from_key"], r["to_key"], r["depth"]) for r in moat_a["rows"]]
        keys_b = [(r["from_key"], r["to_key"], r["depth"]) for r in moat_b["rows"]]
        assert keys_a == keys_b, "same seed produced different samples: {} vs {}".format(keys_a, keys_b)

        moat_c = p75_run.measure_moat(copy_dir, graph_a, n_samples=10, seed=778)
        keys_c = [(r["from_key"], r["to_key"], r["depth"]) for r in moat_c["rows"]]
        # a different seed is allowed to coincidentally match on a tiny smoke
        # store, so this is informational only, not asserted -- the load-
        # bearing guarantee is same-seed reproducibility, checked above.
        print("test_sampling_is_deterministic: OK (n={}, same-seed match={}, diff-seed identical={})".format(
            len(keys_a), keys_a == keys_b, keys_a == keys_c))
    finally:
        shutil.rmtree(tmp_root, ignore_errors=True)


def test_rederivation_on_smoke_store():
    tmp_root = tempfile.mkdtemp(prefix="p75-test-rederive-")
    try:
        source_dir = _build_smoke_store(tmp_root, max_windows=80)
        copy_dir = os.path.join(tmp_root, "copy")
        p75_run.clone_store(source_dir, copy_dir)

        graph = LiveGraph(copy_dir, canon=True)
        all_edges = graph.canon_inferred_edges()
        assert isinstance(all_edges, list)

        moat = p75_run.measure_moat(copy_dir, graph, n_samples=30, seed=59075)
        if moat["n_sampled"] == 0:
            print("test_rederivation_on_smoke_store: SKIPPED (smoke store has 0 canon inferred edges -- widen max_windows if this persists)")
            return

        for row in moat["rows"]:
            assert row["ok"], "re-derivation failed for {}: {}".format(
                (row["from_key"], row["to_key"]), row["detail"])
        assert moat["all_rederived_ok"]
        assert moat["n_rederived_ok"] == moat["n_sampled"]
        print("test_rederivation_on_smoke_store: OK ({}/{} rederived, {} total canon inferred edges available)".format(
            moat["n_rederived_ok"], moat["n_sampled"], len(all_edges)))
    finally:
        shutil.rmtree(tmp_root, ignore_errors=True)


def test_rederivation_rejects_tampered_edge():
    """A stranger's re-derivation must FAIL closed: feeding it an edge
    whose from_key/to_key was tampered with (a cache-injection style
    attack, mirroring MVP-6's tamper test) must not silently pass."""
    tmp_root = tempfile.mkdtemp(prefix="p75-test-tamper-")
    try:
        source_dir = _build_smoke_store(tmp_root, max_windows=80)
        copy_dir = os.path.join(tmp_root, "copy")
        p75_run.clone_store(source_dir, copy_dir)

        graph = LiveGraph(copy_dir, canon=True)
        all_edges = graph.canon_inferred_edges()
        if not all_edges:
            print("test_rederivation_rejects_tampered_edge: SKIPPED (smoke store has 0 canon inferred edges)")
            return

        real_edge = all_edges[0]
        tampered = dict(real_edge)
        tampered["to_key"] = tampered["to_key"] + "__tampered__"

        pin = graph.canon_env_pin
        from livecausal import canon as canon_mod
        stranger_nlp = canon_mod._get_nlp() if (pin and pin.get("model_available")) else None
        stranger_store = LiveStore(copy_dir)

        ok, detail = p75_run.rederive_canon_edge(stranger_store, tampered, stranger_nlp)
        assert ok is False, "tampered edge was accepted as re-derivable -- moat has a hole"
        assert detail["reason"] == "to_key mismatch"
        print("test_rederivation_rejects_tampered_edge: OK (rejected as: {})".format(detail["reason"]))
    finally:
        shutil.rmtree(tmp_root, ignore_errors=True)


def test_no_mutation_check():
    tmp_root = tempfile.mkdtemp(prefix="p75-test-nomut-")
    try:
        source_dir = _build_smoke_store(tmp_root)
        copy_dir = os.path.join(tmp_root, "copy")
        expected_shas = p75_run.source_segment_shas(source_dir)
        p75_run.clone_store(source_dir, copy_dir)

        before = p75_run.check_no_mutation(copy_dir, expected_shas)
        assert before["store_verify_true"] is True
        assert before["segment_shas_unchanged"] is True

        # exercise canon mount + moat sampling exactly as run_p75 does,
        # then re-check: segments must still be untouched (only
        # canon_inferred.jsonl / use.ledger are allowed to appear/change).
        write_side, raw_graph, canon_graph = p75_run.measure_write_side(copy_dir)
        p75_run.measure_canon_cost(copy_dir)
        graph2 = LiveGraph(copy_dir, canon=True)
        p75_run.measure_moat(copy_dir, graph2, n_samples=10, seed=1)

        after = p75_run.check_no_mutation(copy_dir, expected_shas)
        assert after["store_verify_true"] is True
        assert after["segment_shas_unchanged"] is True
        assert after["n_segments"] == before["n_segments"]
        print("test_no_mutation_check: OK ({} segments, verify True before and after full clause a/c/d run)".format(
            after["n_segments"]))

        # negative control: prove the check actually detects a mutation,
        # so a passing positive check above is not vacuous.
        fake_expected = list(expected_shas) + ["deadbeef" * 8]
        tampered_check = p75_run.check_no_mutation(copy_dir, fake_expected)
        assert tampered_check["segment_shas_unchanged"] is False
        print("test_no_mutation_check: OK (negative control: mismatched expected-shas list correctly flagged)")
    finally:
        shutil.rmtree(tmp_root, ignore_errors=True)


def test_write_side_lift_and_compression_are_sane():
    tmp_root = tempfile.mkdtemp(prefix="p75-test-writeside-")
    try:
        source_dir = _build_smoke_store(tmp_root, max_windows=80)
        copy_dir = os.path.join(tmp_root, "copy")
        p75_run.clone_store(source_dir, copy_dir)

        write_side, raw_graph, canon_graph = p75_run.measure_write_side(copy_dir)
        assert write_side["n_inferred_edges_raw"] >= 0
        assert write_side["n_inferred_edges_canon"] >= 0
        assert write_side["n_distinct_canon_keys"] <= write_side["n_distinct_raw_keys"], \
            "canonicalization must never INCREASE the distinct key count"
        if write_side["n_distinct_canon_keys"]:
            assert write_side["raw_to_canon_compression_ratio"] >= 1.0
        if write_side["n_inferred_edges_raw"]:
            assert write_side["lift_factor_canon_over_raw"] >= 0.0
        assert write_side["canon_env_pin"] is not None
        assert "spacy_available" in write_side["canon_env_pin"]
        print("test_write_side_lift_and_compression_are_sane: OK (raw_inferred={} canon_inferred={} "
              "raw_keys={} canon_keys={} compression={})".format(
                  write_side["n_inferred_edges_raw"], write_side["n_inferred_edges_canon"],
                  write_side["n_distinct_raw_keys"], write_side["n_distinct_canon_keys"],
                  write_side["raw_to_canon_compression_ratio"]))
    finally:
        shutil.rmtree(tmp_root, ignore_errors=True)


def test_canon_cost_cold_vs_warm():
    tmp_root = tempfile.mkdtemp(prefix="p75-test-cost-")
    try:
        source_dir = _build_smoke_store(tmp_root)
        copy_dir = os.path.join(tmp_root, "copy")
        p75_run.clone_store(source_dir, copy_dir)

        cost = p75_run.measure_canon_cost(copy_dir)
        assert cost["cold_was_rebuilt_on_mount"] is True, "deleting the cache first must force a rebuild"
        assert cost["warm_loaded_from_cache"] is True, "second mount right after must adopt the cache"
        assert cost["cold_mount_seconds"] >= 0.0
        assert cost["warm_mount_seconds"] >= 0.0
        cache_path = os.path.join(copy_dir, "canon_inferred.jsonl")
        assert os.path.exists(cache_path), "warm mount should have left a cache file behind"
        print("test_canon_cost_cold_vs_warm: OK (cold={:.4f}s warm={:.4f}s, rebuilt={}, loaded_from_cache={})".format(
            cost["cold_mount_seconds"], cost["warm_mount_seconds"],
            cost["cold_was_rebuilt_on_mount"], cost["warm_loaded_from_cache"]))
    finally:
        shutil.rmtree(tmp_root, ignore_errors=True)


def test_full_smoke_end_to_end_no_read_side():
    """Exercises run_p75() itself (minus the consult_run.py subprocess,
    which needs network/HF access) against a freshly built local store."""
    tmp_root = tempfile.mkdtemp(prefix="p75-test-e2e-")
    try:
        source_dir = _build_smoke_store(tmp_root, max_windows=80)
        result = p75_run.run_p75(source_dir, n_moat_samples=15, moat_seed=59075, run_read_side=False)

        assert result["read_side"] is None
        assert result["clause_a_write_side"]["n_inferred_raw"] >= 0
        assert result["clause_c_cost"]["cold_was_rebuilt_on_mount"] is True
        assert result["clause_d_moat"]["store_verify_true"] is True
        assert result["clause_d_moat"]["segment_shas_unchanged"] is True
        assert result["env_pin"] is not None
        assert not os.path.exists(result["copy_dir"]), "run_p75 must clean up its own work_root when it owns it"
        print("test_full_smoke_end_to_end_no_read_side: OK (clause a={}, c cold_rebuilt={}, d {}/{} rederived, "
              "no_mutation store_verify={})".format(
                  result["clause_a_write_side"], result["clause_c_cost"]["cold_was_rebuilt_on_mount"],
                  result["clause_d_moat"]["n_rederived_ok"], result["clause_d_moat"]["n_sampled"],
                  result["clause_d_moat"]["store_verify_true"]))
    finally:
        shutil.rmtree(tmp_root, ignore_errors=True)


TESTS = [
    test_clone_store_never_touches_source,
    test_sampling_is_deterministic,
    test_rederivation_on_smoke_store,
    test_rederivation_rejects_tampered_edge,
    test_no_mutation_check,
    test_write_side_lift_and_compression_are_sane,
    test_canon_cost_cold_vs_warm,
    test_full_smoke_end_to_end_no_read_side,
]


if __name__ == "__main__":
    import torch
    torch.set_num_threads(1)

    n_pass = 0
    n_fail = 0
    for t in TESTS:
        try:
            t()
            n_pass += 1
        except Exception as e:
            n_fail += 1
            print("{}: FAIL -- {}: {}".format(t.__name__, type(e).__name__, e))
    print("=" * 60)
    print("{}/{} passed".format(n_pass, len(TESTS)))
    if n_fail:
        sys.exit(1)
