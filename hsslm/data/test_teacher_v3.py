"""pytest-Tests fuer teacher_v3.py (IR-verifizierte FERTIG-Form-Engine als
Lehrer-Satz-Generator, Sprache-Haupt-Track, Anschluss an fertig_teacher.py
[v2] und graph_to_text_v3.py [Rohtext-Spur]).

Zwei Testgruppen:
  1. Reine Funktions-Tests (build_prompt, verify_variant, choose_best,
     is_prompt_echo_only, content_words) -- kein FormEngine-Laden noetig,
     schnell.
  2. Store+Engine-Tests (Determinismus, Schema, verified-only-Property,
     Citation-Durchreichung) -- laden die echte FormEngine (langsam, ~15s
     pro Record wegen n=3 HSSLM-Generierungen), daher mit SEHR kleinem
     Limit (2 Records) und als eigene, teure Fixture einmal pro Modul
     gebaut.

Run:
    OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
    python3 -m pytest hsslm/data/test_teacher_v3.py -q
"""
import os
import sys

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pytest

from teacher_v3 import (
    build_prompt, verify_variant, choose_best, is_prompt_echo_only,
    content_words, build_teacher_v3_pairs,
)

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
STORE_DIR = os.path.join(REPO_ROOT, "results", "p72_store_local")
MODEL_PATH = os.path.join(REPO_ROOT, "fertig", "data", "hsslm_form.pt")


# ---------------------------------------------------------------------------
# Reine Funktions-Tests
# ---------------------------------------------------------------------------

def test_build_prompt_format():
    p = build_prompt("the engine", "causes", "overheating")
    assert p == "The engine causes overheating."


def test_build_prompt_capitalizes_subject_only():
    p = build_prompt("smoking", "reduces", "lung capacity")
    assert p.startswith("Smoking")
    assert "Lung capacity" not in p


def test_content_words_filters_short_words():
    words = content_words("a it is of the very loud engine roars")
    assert "very" in words
    assert "loud" in words
    assert "engine" in words
    assert "roars" in words
    assert "a" not in words
    assert "it" not in words
    assert "is" not in words
    assert "of" not in words
    assert "the" not in words


def test_verify_variant_accepts_when_both_present():
    assert verify_variant(
        "the engine", "overheating",
        "The engine causes severe overheating in summer.",
    ) is True


def test_verify_variant_rejects_missing_trigger():
    assert verify_variant(
        "the engine", "overheating",
        "Something else entirely causes overheating.",
    ) is False


def test_verify_variant_rejects_missing_outcome():
    assert verify_variant(
        "the engine", "overheating",
        "The engine causes something completely different.",
    ) is False


def test_verify_variant_rejects_when_no_content_words():
    # trigger/outcome bestehen nur aus kurzen Woertern (<4 Zeichen) --
    # nichts Pruefbares, kein Pass (siehe verify_variant Docstring)
    assert verify_variant("it is", "of a", "some random text here") is False


def test_choose_best_prefers_shortest():
    texts = ["A very long sentence with lots of extra words.",
             "Short sentence.",
             "Medium length sentence here."]
    assert choose_best(texts) == "Short sentence."


def test_choose_best_alphabetic_tiebreak():
    texts = ["Zebra text here.", "Apple text here."]
    # gleiche Laenge -> alphabetischer Tie-Break
    assert len(texts[0]) == len(texts[1])
    assert choose_best(texts) == "Apple text here."


def test_is_prompt_echo_only_detects_pure_echo():
    prompt = "The engine causes overheating."
    # Reiner Echo-Fall: nach dem Prompt folgt nichts (oder nur Whitespace) --
    # keine neuen Content-Woerter.
    assert is_prompt_echo_only(prompt, prompt + "\n\n") is True


def test_is_prompt_echo_only_true_even_for_nonsense_repeats():
    prompt = "The engine causes overheating."
    variant = prompt + " " + prompt  # wortwoertliche Wiederholung, keine NEUEN Woerter
    assert is_prompt_echo_only(prompt, variant) is True


def test_is_prompt_echo_only_false_when_new_words_appear():
    prompt = "The engine causes overheating."
    variant = prompt + " This leads to further damage."
    assert is_prompt_echo_only(prompt, variant) is False


def test_is_prompt_echo_only_false_when_variant_diverges_early():
    prompt = "The engine causes overheating."
    variant = "Something completely different was said."
    assert is_prompt_echo_only(prompt, variant) is False


# ---------------------------------------------------------------------------
# Store+Engine-Tests (teuer -- echte FormEngine, sehr kleines Limit)
# ---------------------------------------------------------------------------

pytestmark = pytest.mark.skipif(
    not (os.path.isdir(STORE_DIR) and os.path.exists(MODEL_PATH)),
    reason="results/p72_store_local or fertig/data/hsslm_form.pt not present",
)


@pytest.fixture(scope="module")
def small_run():
    pairs, rejects = [], []
    for pair, reject in build_teacher_v3_pairs(STORE_DIR, limit=2, n_variants=2):
        if pair is not None:
            pairs.append(pair)
        else:
            rejects.append(reject)
    return pairs, rejects


def test_determinism_two_runs_identical():
    run1_pairs, run1_rejects = [], []
    for pair, reject in build_teacher_v3_pairs(STORE_DIR, limit=2, n_variants=2):
        (run1_pairs if pair is not None else run1_rejects).append(pair or reject)
    run2_pairs, run2_rejects = [], []
    for pair, reject in build_teacher_v3_pairs(STORE_DIR, limit=2, n_variants=2):
        (run2_pairs if pair is not None else run2_rejects).append(pair or reject)
    assert run1_pairs == run2_pairs
    assert run1_rejects == run2_rejects


def test_schema_has_required_keys(small_run):
    pairs, rejects = small_run
    required = {"structure", "text", "citation", "doc_coord",
                "trigger", "mechanism", "outcome", "family",
                "n_variants_verified", "n_variants_total", "prompt_echo_only"}
    for p in pairs:
        assert required <= set(p.keys())


def test_schema_structure_format(small_run):
    pairs, rejects = small_run
    for p in pairs:
        s = p["structure"]
        assert s.startswith("<fact> ")
        assert s.endswith(" <say>")
        assert " | " in s


def test_verified_only_property(small_run):
    """Jedes Paar im Output MUSS die IR-Verifikation bestehen -- keine
    unverifizierte Variante darf ins Schema gelangen."""
    pairs, rejects = small_run
    for p in pairs:
        assert verify_variant(p["trigger"], p["outcome"], p["text"])
        assert p["n_variants_verified"] >= 1
        assert p["n_variants_verified"] <= p["n_variants_total"]


def test_citation_passthrough(small_run):
    pairs, rejects = small_run
    sys.path.insert(0, os.path.join(REPO_ROOT, "src", "livecausal"))
    from store import LiveStore
    store = LiveStore(STORE_DIR)
    records_by_pos = {(sha, idx): rec for sha, idx, rec in store.iter_records()}

    for p in pairs:
        key = (p["citation"]["sha"], p["citation"]["idx"])
        assert key in records_by_pos
        rec = records_by_pos[key]
        assert rec["doc_coord"] == p["doc_coord"]
        assert rec["trigger"] == p["trigger"]
        assert rec["mechanism"] == p["mechanism"]
        assert rec["outcome"] == p["outcome"]


def test_rejects_carry_diagnostic_info(small_run):
    pairs, rejects = small_run
    for r in rejects:
        assert "family" in r
        assert "prompt" in r
        assert "rejected" in r
        assert "citation" in r
