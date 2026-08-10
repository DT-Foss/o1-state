"""Plain-assert tests for the cross-version canon probe
(src/livecausal/canon_probe.py). Run:
    python3 src/livecausal/test_canon_probe.py

Covers the build brief's two required checks: dump is byte-identical
across two independent process runs on the same machine (the local
baseline the x86-runner comparison depends on being meaningful at all --
if same-machine, same-process-twice already diverged, a cross-machine
diff would be uninterpretable noise), and compare() correctly flags a
divergence introduced into an artificially modified copy of a dump.
"""
import copy
import json
import os
import shutil
import subprocess
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from livecausal.canon_probe import PROBES, run_compare, run_dump

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def with_tmpdir(fn):
    d = tempfile.mkdtemp(prefix="canon-probe-test-")
    try:
        fn(d)
    finally:
        shutil.rmtree(d, ignore_errors=True)


# ─────────────────────────────────────────────────────────────────────────
# 1. dump is byte-identical across two independent process runs on the
#    same machine -- each process loads its own spaCy pipeline from
#    scratch and runs the full fixed PROBES list.
# ─────────────────────────────────────────────────────────────────────────
def test_dump_byte_identical_across_processes(d):
    out_a = os.path.join(d, "dump_a.json")
    out_b = os.path.join(d, "dump_b.json")

    script = os.path.join(REPO_ROOT, "src", "livecausal", "canon_probe.py")
    env = {**os.environ, "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1",
           "MKL_NUM_THREADS": "1", "TOKENIZERS_PARALLELISM": "false"}

    for out_path in (out_a, out_b):
        proc = subprocess.run(
            [sys.executable, script, "dump", "--out", out_path],
            capture_output=True, text=True, timeout=120, env=env,
        )
        assert proc.returncode == 0, "dump subprocess failed: {}".format(proc.stderr)

    with open(out_a, "rb") as f:
        bytes_a = f.read()
    with open(out_b, "rb") as f:
        bytes_b = f.read()

    assert bytes_a == bytes_b, (
        "dump output diverged across two independent process runs on the "
        "SAME machine -- if this fails, a cross-machine (local vs x86 "
        "runner) comparison would be meaningless noise on top of it"
    )

    # Sanity on the content itself, not just byte equality (a byte-equal
    # but empty/truncated file would trivially "pass" the check above).
    payload = json.loads(bytes_a)
    assert payload["n_probes"] == len(PROBES)
    assert payload["n_probes"] >= 200, "build brief requires >= ~200 probe phrases"
    assert len(payload["probes"]) == payload["n_probes"]
    assert "env_pin" in payload and payload["env_pin"]["canon_version"]
    # Every probe entry must have both fields; canon is always a string
    # (canon.py's fallback path guarantees this even for symbol/empty
    # inputs -- see canon.py's _surface_fallback, never None/missing).
    for p in payload["probes"]:
        assert isinstance(p["raw"], str)
        assert isinstance(p["canon"], str)


# ─────────────────────────────────────────────────────────────────────────
# 2. compare() detects a divergence introduced into an artificially
#    modified copy of a dump -- both the direct run_compare() API and the
#    CLI's --out report path.
# ─────────────────────────────────────────────────────────────────────────
def test_compare_detects_artificial_divergence(d):
    dump_a = run_dump()
    dump_b = copy.deepcopy(dump_a)

    # Self-comparison first: a dump against itself must show zero
    # divergence (the baseline the "detects a real divergence" check
    # below is contrasted against).
    report_same = run_compare(dump_a, copy.deepcopy(dump_a))
    assert report_same["n_divergent"] == 0
    assert report_same["fraction_identical"] == 1.0
    assert report_same["divergences"] == []

    # Artificially corrupt exactly TWO entries in dump_b's copy -- one
    # phrase gets an obviously-wrong canon value, another gets its canon
    # value blanked -- simulating what a genuine cross-version spaCy
    # divergence would produce in the dump file (a differing string for
    # the same raw phrase).
    assert len(dump_b["probes"]) >= 2
    target_raw_1 = dump_b["probes"][0]["raw"]
    target_raw_2 = dump_b["probes"][1]["raw"]
    original_canon_1 = dump_b["probes"][0]["canon"]
    dump_b["probes"][0]["canon"] = original_canon_1 + "_CORRUPTED_FOR_TEST"
    dump_b["probes"][1]["canon"] = ""

    # Also change env_pin's spacy_version, mirroring what a REAL
    # local-vs-x86-runner dump pair would show (3.8.11 vs 3.8.15) -- the
    # report must carry both env_pins through untouched so a reader can
    # attribute the divergence to a specific version pair.
    dump_b["env_pin"] = dict(dump_b["env_pin"])
    dump_b["env_pin"]["spacy_version"] = "9.9.99-test"

    report = run_compare(dump_a, dump_b)

    assert report["n_total"] == len(dump_a["probes"])
    assert report["n_divergent"] == 2
    assert report["n_identical"] == report["n_total"] - 2
    assert report["fraction_identical"] == round((report["n_total"] - 2) / report["n_total"], 4)

    divergent_raws = {dd["raw"] for dd in report["divergences"]}
    assert divergent_raws == {target_raw_1, target_raw_2}

    for dd in report["divergences"]:
        if dd["raw"] == target_raw_1:
            assert dd["canon_a"] == original_canon_1
            assert dd["canon_b"] == original_canon_1 + "_CORRUPTED_FOR_TEST"
        if dd["raw"] == target_raw_2:
            assert dd["canon_b"] == ""
        # Both env_pins travel with every divergence row, untouched --
        # this is the field a reader needs to attribute a real
        # divergence to a spaCy version pair.
        assert dd["env_pin_a"] == dump_a["env_pin"]
        assert dd["env_pin_b"] == dump_b["env_pin"]

    assert report["env_pin_a"]["spacy_version"] != report["env_pin_b"]["spacy_version"]

    # --- CLI path: same corruption, via files + the actual subprocess
    #     entry point, confirming compare's exit code is ALWAYS 0 (report,
    #     not gate) even when real divergences are present. ---
    path_a = os.path.join(d, "cli_a.json")
    path_b = os.path.join(d, "cli_b.json")
    path_out = os.path.join(d, "cli_report.json")
    with open(path_a, "w", encoding="utf-8") as f:
        json.dump(dump_a, f)
    with open(path_b, "w", encoding="utf-8") as f:
        json.dump(dump_b, f)

    script = os.path.join(REPO_ROOT, "src", "livecausal", "canon_probe.py")
    proc = subprocess.run(
        [sys.executable, script, "compare", "--a", path_a, "--b", path_b, "--out", path_out],
        capture_output=True, text=True, timeout=60,
    )
    assert proc.returncode == 0, "compare must always exit 0 (report, not gate), even with real divergences"
    assert "n_divergent=2" in proc.stdout

    with open(path_out, "r", encoding="utf-8") as f:
        cli_report = json.load(f)
    assert cli_report["n_divergent"] == 2
    assert {dd["raw"] for dd in cli_report["divergences"]} == {target_raw_1, target_raw_2}


def run_all():
    tests = [
        test_dump_byte_identical_across_processes,
        test_compare_detects_artificial_divergence,
    ]
    for t in tests:
        with_tmpdir(t)
        print("OK  {}".format(t.__name__))
    print("ALL {} TESTS PASSED".format(len(tests)))


if __name__ == "__main__":
    run_all()
