"""FERTIG — Tests des Primitiv-Schemas (fertig.primitives + relations)."""

from __future__ import annotations

import numpy as np

from fertig import primitives, relations


def test_normalize_synonyms():
    assert primitives.normalize_mechanism("leads_to") == "causes"
    assert primitives.normalize_mechanism("associated_with") == "related_to"
    assert primitives.normalize_mechanism("described_as") == "defined_as"
    assert primitives.normalize_mechanism("indirectly_linked_to") == "related_to"
    assert primitives.normalize_mechanism("improves") == "improves"
    # unbekannt bleibt
    assert primitives.normalize_mechanism("frobnicates") == "frobnicates"


def test_digital_root_ring_homomorphism():
    # dr(a+b) = dr(dr(a)+dr(b))
    a, b = "sugar", "candy"
    assert primitives.digital_root(a + b) == \
        primitives.digital_root(primitives.digital_root(a) +
                                primitives.digital_root(b))
    # 9 statt 0
    assert primitives.digital_root("ii") == 9  # i=9, 9+9=18 -> 9


def test_digital_root_class_groups():
    # Kommutativität: gleiche Buchstaben -> gleiche Klasse
    assert primitives.digital_root_class("ab") == \
        primitives.digital_root_class("ba")
    # Anhängen von 9er-Schritten ändert die Klasse nicht (9 ≡ 0 mod 9)
    assert primitives.digital_root_class("a") == \
        primitives.digital_root_class("ai")


def test_detect_periodicity():
    seq = [1.0, 2.0, 1.0, 2.0, 1.0, 2.0]
    assert primitives.detect_periodicity(seq) == 2
    assert primitives.detect_periodicity([1.0, 2.0, 3.0, 4.0]) is None


def test_detect_parity():
    assert primitives.detect_parity([1.0, 2.0, 2.0, 1.0]) == "even"
    assert primitives.detect_parity([1.0, 2.0, -2.0, -1.0]) == "odd"
    assert primitives.detect_parity([1.0, 2.0, 3.0]) is None


def test_intrinsic_dimension():
    # TwoNN auf realen (verrauschten) Punktwolken — perfekt gleichmäßige
    # Gitter degenerieren (r2/r1 -> 1), das ist eine bekannte Eigenschaft
    rng = np.random.RandomState(0)
    X = np.linspace(0, 1, 40).reshape(-1, 1) + 0.01 * rng.randn(40, 1)
    id1 = primitives.intrinsic_dimension(X)
    assert 0.5 < id1 < 2.5
    # 2D: zufällige Punkte im Quadrat (Gitter degenerieren: exakte
    # Distanz-Gleichstände brechen r2/r1 -> 1)
    g = rng.rand(100, 2)
    id2 = primitives.intrinsic_dimension(g)
    assert 1.2 < id2 < 3.5

def test_question_primitives():
    assert "causes" in primitives.question_primitives(
        "What factor has the greatest effect on motion?")
    assert "created_in" in primitives.question_primitives(
        "Which technology was developed most recently?")
    assert "comparative" in primitives.question_primitives(
        "How is a pond different from a lake?")


def test_relations_schema_extraction():
    trips = relations.extract_relations(
        "The telephone was invented in 1876 by Bell. "
        "Gold is a metal. The heart is responsible for pumping blood. "
        "A pond is smaller than a lake.")
    mechs = {m for _, m, _, _ in trips}
    assert "created_in" in mechs
    assert "smaller_than" in mechs
    assert "responsible_for" in mechs
    assert "is_a" in mechs


def test_relations_canonical_causal():
    trips = relations.extract_relations(
        "Smoking leads to lung cancer. Exercise reduces stress.")
    d = {(a, m): c for a, m, c, _ in trips}
    assert ("smoking", "causes") in d
    assert ("exercise", "reduces") in d
