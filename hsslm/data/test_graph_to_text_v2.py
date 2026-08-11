"""pytest-Tests fuer graph_to_text_v2.py (Sprecher-Datenbasis v2,
Baustein 3: Junk-Filter + Dedupe + deterministische Sortierung).

Laeuft gegen results/p72_store_local (kleiner, lokal verfuegbarer Store --
NICHT den vollen wt103_full_store_local, dessen Rekonstruktion Minuten
dauert; die Logik ist store-groessenunabhaengig, siehe build_pairs_v2).

Run:
    OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
    HF_DATASETS_OFFLINE=1 HF_HUB_OFFLINE=1 \
    python3 -m pytest hsslm/data/test_graph_to_text_v2.py -q
"""
import os
import sys

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
os.environ.setdefault("HF_HUB_OFFLINE", "1")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pytest

from graph_to_text_v2 import build_pairs_v2, P72_CADENCE
from key_filter import is_junk_key

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
STORE_DIR = os.path.join(REPO_ROOT, "results", "p72_store_local")

pytestmark = pytest.mark.skipif(
    not os.path.isdir(STORE_DIR),
    reason="results/p72_store_local not present (read-only fixture store)",
)


@pytest.fixture(scope="module")
def v2_pairs_and_report():
    # Kleine chunks (p72-Groesse), Store ist p72_store_local selbst --
    # deckt dieselbe Rekonstruktions- und Filterlogik ab wie ein voller Lauf.
    cadence = dict(P72_CADENCE)
    return build_pairs_v2(STORE_DIR, cadence)


def test_report_has_expected_keys(v2_pairs_and_report):
    pairs, report = v2_pairs_and_report
    assert set(report.keys()) == {"n_raw", "n_junk_filtered", "n_dupe_filtered", "n_final"}


def test_report_accounting_is_consistent(v2_pairs_and_report):
    pairs, report = v2_pairs_and_report
    assert report["n_final"] == len(pairs)
    assert report["n_raw"] - report["n_junk_filtered"] - report["n_dupe_filtered"] == report["n_final"]


def test_no_junk_keys_in_output(v2_pairs_and_report):
    pairs, report = v2_pairs_and_report
    for p in pairs:
        assert not is_junk_key(p["trigger"])
        assert not is_junk_key(p["outcome"])


def test_output_is_deduplicated(v2_pairs_and_report):
    pairs, report = v2_pairs_and_report
    keys = [(p["structure"], p["text"]) for p in pairs]
    assert len(keys) == len(set(keys))


def test_output_is_sorted_by_citation(v2_pairs_and_report):
    pairs, report = v2_pairs_and_report
    keys = [(p["citation"]["sha"], p["citation"]["idx"]) for p in pairs]
    assert keys == sorted(keys)


def test_determinism_two_runs_identical(v2_pairs_and_report):
    # Vergleicht die (bereits gecachte) Modul-Fixture gegen einen zweiten
    # unabhaengigen Lauf -- ein Lauf statt zwei zusaetzlichen, spart die
    # teure iter_windows-Rekonstruktion (mehrere Minuten pro Aufruf).
    pairs1, report1 = v2_pairs_and_report
    cadence = dict(P72_CADENCE)
    pairs2, report2 = build_pairs_v2(STORE_DIR, cadence)
    assert pairs1 == pairs2
    assert report1 == report2


def test_schema_matches_v1(v2_pairs_and_report):
    pairs, report = v2_pairs_and_report
    required = {"structure", "text", "citation", "doc_coord",
                "trigger", "mechanism", "outcome"}
    for p in pairs[:20]:
        assert required <= set(p.keys())
        assert p["structure"].startswith("<fact> ")
        assert p["structure"].endswith(" <say>")
