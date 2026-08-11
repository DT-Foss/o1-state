"""FERTIG — Tests der Grammar- und Code-Schicht."""

from __future__ import annotations

from fertig import grammar
from fertig import code as code_mod


# ---------------------------------------------------------------------------
# grammar: Kongruenz-Regeln
# ---------------------------------------------------------------------------

def test_anaphor_gender_herself_ok():
    # über structural_score (setzt die Case-Info des Originalsatzes)
    good = "Katherine can't help herself."
    bad = "Katherine can't help himself."
    assert grammar.structural_score(good) > grammar.structural_score(bad)


def test_anaphor_gender_mismatch():
    grammar.apply_rules("Katherine can't help himself.")
    toks = grammar._toks("Katherine can't help himself.")
    assert grammar.rule_anaphor_gender(toks) == -1


def test_anaphor_number_mismatch():
    assert grammar.rule_anaphor_number(
        "the boy hurt themselves".split()) == -1


def test_determiner_plural_mismatch():
    assert grammar.rule_determiner_noun("this cats".split()) == -1
    assert grammar.rule_determiner_noun("these cat".split()) == -1


def test_determiner_ok():
    assert grammar.rule_determiner_noun("this cat".split()) == 1


def test_subject_verb_plural_mismatch():
    assert grammar.rule_subject_verb("the children runs".split()) == -1


def test_subject_verb_ok():
    assert grammar.rule_subject_verb("the children run".split()) == 1


def test_structural_score_discriminates():
    good = "katherine can't help herself"
    bad = "katherine can't help himself"
    assert grammar.structural_score(good) > grammar.structural_score(bad)


# ---------------------------------------------------------------------------
# code: Assemblierung + Sandbox
# ---------------------------------------------------------------------------

def test_load_fragments_and_triplets():
    frags = code_mod.load_fragments()
    assert len(frags) >= 50
    trips = code_mod.load_triplets()
    assert len(trips) >= 100  # FORGE-Wissen (python_stdlib etc.)


def test_assemble_matches():
    frags = code_mod.load_fragments()
    trips = code_mod.load_triplets()
    code_txt, used = code_mod.assemble("list all files in the directory",
                                       frags, trips)
    assert used  # mindestens ein Fragment gematcht
    assert "def main()" in code_txt
    assert "if __name__" in code_txt


def test_assemble_honest_dead_end():
    frags = code_mod.load_fragments()
    code_txt, used = code_mod.assemble("zzz qqq xxx yyy", frags, [])
    assert used == [] and code_txt == ""


def test_sandbox_runs_and_fails():
    rc, out, err = code_mod.run_sandbox("print('hello')")
    assert rc == 0 and "hello" in out
    rc2, _, err2 = code_mod.run_sandbox("raise ValueError('boom')")
    assert rc2 != 0
