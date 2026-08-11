"""pytest-Tests fuer fertig_teacher.py (Determinismus, Schema-Konformitaet,
Citation-Durchreichung).

Run:
    OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
    python3 -m pytest hsslm/data/test_fertig_teacher.py -q
"""
import os
import sys

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pytest

from fertig_teacher import (
    build_teacher_pairs,
    verbalize_record,
    _humanize,
    _agrees_plural,
)

STORE_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "results", "p72_store_local",
)

pytestmark = pytest.mark.skipif(
    not os.path.isdir(STORE_DIR),
    reason="results/p72_store_local not present (read-only fixture store)",
)


@pytest.fixture(scope="module")
def sample_pairs():
    return list(build_teacher_pairs(STORE_DIR, limit=50))


def test_determinism_two_runs_byte_identical():
    """Gleicher Store + gleicher limit => byte-identische Paare (keine
    Zufallsquelle im Verbalisierungspfad, siehe Modul-Docstring)."""
    run1 = list(build_teacher_pairs(STORE_DIR, limit=50))
    run2 = list(build_teacher_pairs(STORE_DIR, limit=50))
    assert run1 == run2


def test_determinism_verbalize_record_is_pure():
    """verbalize_record ist eine reine Funktion von (record, index) --
    zweimaliger Aufruf mit denselben Argumenten liefert dasselbe Ergebnis."""
    rec = {"trigger": "the engine", "mechanism": "causes", "outcome": "overheating"}
    r1 = verbalize_record(rec, 3)
    r2 = verbalize_record(rec, 3)
    assert r1 == r2


def test_schema_has_required_keys(sample_pairs):
    required = {"structure", "text", "citation", "doc_coord",
                "trigger", "mechanism", "outcome", "family"}
    for p in sample_pairs:
        assert required <= set(p.keys())


def test_schema_structure_format(sample_pairs):
    for p in sample_pairs:
        s = p["structure"]
        assert s.startswith("<fact> ")
        assert s.endswith(" <say>")
        assert " | " in s


def test_schema_text_is_nonempty_sentence(sample_pairs):
    for p in sample_pairs:
        assert p["text"]
        assert p["text"].endswith(".")


def test_citation_passthrough(sample_pairs):
    """citation.sha/idx im Paar muss exakt dem LiveStore-Record entsprechen,
    aus dem das Paar erzeugt wurde (die Quittung reist mit dem Satz)."""
    sys.path.insert(0, os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
        "src", "livecausal"))
    from store import LiveStore
    store = LiveStore(STORE_DIR)
    records_by_pos = {(sha, idx): rec for sha, idx, rec in store.iter_records()}

    for p in sample_pairs:
        key = (p["citation"]["sha"], p["citation"]["idx"])
        assert key in records_by_pos
        rec = records_by_pos[key]
        assert rec["doc_coord"] == p["doc_coord"]
        assert rec["trigger"] == p["trigger"]
        assert rec["mechanism"] == p["mechanism"]
        assert rec["outcome"] == p["outcome"]


def test_family_label_is_known_or_unknown(sample_pairs):
    from fertig import primitives
    valid = {f.value for f in primitives.RelationFamily} | {"unknown"}
    for p in sample_pairs:
        assert p["family"] in valid


def test_humanize_strips_wt103_hyphen_artifact():
    assert _humanize("co @-@ operate") == "co-operate"
    assert _humanize("grammy @-@ award") == "grammy-award"
    assert _humanize("plain text") == "plain text"


def test_humanize_collapses_whitespace():
    assert _humanize("a   b\t c") == "a b c"


def test_agrees_plural_heuristic():
    assert _agrees_plural("its predecessors") is True
    assert _agrees_plural("events") is True
    assert _agrees_plural("story") is False
    assert _agrees_plural("the class") is False
    # Ausnahmen (endet auf -ss/-us/-is, KEIN Plural-Indikator)
    assert _agrees_plural("the boss") is False
    assert _agrees_plural("the campus") is False


def test_is_caused_by_numerus_fix():
    plural_rec = {"trigger": "its predecessors", "mechanism": "is caused by", "outcome": "story"}
    text, _ = verbalize_record(plural_rec, 0)
    assert "are caused by" in text
    assert "is caused by" not in text

    singular_rec = {"trigger": "the story", "mechanism": "is caused by", "outcome": "events"}
    text2, _ = verbalize_record(singular_rec, 0)
    assert "is caused by" in text2
