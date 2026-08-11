"""FERTIG — Tests des Primitiv-Schemas (fertig.primitives + relations).

Nach dem Lab-Uebernahme-Umbau (primitives.py/relations.py sind jetzt die
_codex_lab/primitive_schema_snapshot-Versionen): die vier Noether-/
Vortexmath-Tests (digital_root, digital_root_class, detect_periodicity,
detect_parity) sind nach tests/test_noether.py umgezogen, siehe dort und
fertig/noether.py's Modul-Docstring fuer die Begruendung. intrinsic_dimension
ist Live-only (kein Lab-Gegenpart) und lebt jetzt in fertig.diagnostics
neben der strengeren Lab-Variante two_nn_intrinsic_dimension."""

from __future__ import annotations

import numpy as np

from fertig import diagnostics, primitives, relations


def test_normalize_synonyms():
    assert primitives.normalize_mechanism("leads_to") == "causes"
    assert primitives.normalize_mechanism("associated_with") == "related_to"
    assert primitives.normalize_mechanism("described_as") == "defined_as"
    assert primitives.normalize_mechanism("indirectly_linked_to") == "related_to"
    assert primitives.normalize_mechanism("improves") == "improves"
    # unbekannt bleibt unbekannt (preserve_unknown=True-Default), aber
    # normalisiert -- kein Mod-9-Raten mehr im Fallback (siehe noether.py)
    assert primitives.normalize_mechanism("frobnicates") == "frobnicates"
    assert primitives.normalize_mechanism(
        "frobnicates", preserve_unknown=False) is None


def test_intrinsic_dimension():
    # TwoNN auf realen (verrauschten) Punktwolken — perfekt gleichmäßige
    # Gitter degenerieren (r2/r1 -> 1), das ist eine bekannte Eigenschaft.
    # Live-only-Funktion (kein Lab-Gegenpart), jetzt in fertig.diagnostics.
    rng = np.random.RandomState(0)
    X = np.linspace(0, 1, 40).reshape(-1, 1) + 0.01 * rng.randn(40, 1)
    id1 = diagnostics.intrinsic_dimension(X)
    assert 0.5 < id1 < 2.5
    # 2D: zufällige Punkte im Quadrat (Gitter degenerieren: exakte
    # Distanz-Gleichstände brechen r2/r1 -> 1)
    g = rng.rand(100, 2)
    id2 = diagnostics.intrinsic_dimension(g)
    assert 1.2 < id2 < 3.5


def test_question_primitives():
    # Phrasen-/Token-basiertes Matching (Lab-Schema) statt loser
    # Teilstring-Treffer -- siehe primitives.question_primitives-Docstring.
    assert "causes" in primitives.question_primitives(
        "What factor has the greatest effect on motion?")
    assert primitives.question_primitives(
        "Which technology was developed in the latest year?"
    ) == frozenset({"created_in"})
    comparative = primitives.question_primitives(
        "How is a pond different from a lake?")
    assert "smaller_than" in comparative
    assert "larger_than" in comparative
    # 'for' in 'formula' darf 'used_for' nicht per Teilstring aktivieren
    assert "used_for" not in primitives.question_primitives(
        "Which formula predicts motion?")


def test_relations_schema_extraction():
    trips = relations.extract_relations(
        "The telephone was invented in 1876 by Bell. "
        "Gold is a metal. The heart is responsible for pumping blood. "
        "A pond is smaller than a lake.")
    rows = {(a, m, b) for a, m, b, _ in trips}
    assert ("telephone", "created_in", "1876") in rows
    assert ("gold", "is_a", "metal") in rows
    assert ("heart", "responsible_for", "pumping blood") in rows
    assert ("pond", "smaller_than", "lake") in rows


def test_relations_canonical_causal():
    trips = relations.extract_relations(
        "Smoking leads to lung cancer. Exercise reduces stress.")
    d = {(a, m): c for a, m, b, c in trips}
    assert ("smoking", "causes") in d
    assert ("exercise", "reduces") in d


def test_unknown_relation_is_not_fabricated():
    # Der Kern der Lead-Entscheidung: kein Mod-9-Raten im Fallback mehr.
    assert primitives.canonicalize_mechanism("frobnicates") is None
    assert primitives.normalize_mechanism("frobnicates") == "frobnicates"
    assert primitives.normalize_mechanism(
        "frobnicates", preserve_unknown=False) is None


def test_schema_coverage_reports_unknowns_and_diversity():
    report = primitives.schema_coverage(
        ["causes", "leads_to", "made_of", "mystery_relation"])
    assert report.total == 4
    assert report.canonical == 3
    assert report.ratio == 0.75
    assert report.unknown == {"mystery_relation": 1}
