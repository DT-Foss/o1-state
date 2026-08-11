"""FERTIG — Tests der Grounding-Schicht (offline, ohne CLIP/Netz)."""

from __future__ import annotations

from fertig import grounding


def test_extract_quantitative():
    trips = grounding.extract_quantitative(
        "The cheetah can run at speeds up to 110 km/h. "
        "Elephants weigh up to 6,350 kg.")
    units = [u for _, u, _ in trips]
    assert any("km/h" in u for u in units)
    assert any("kg" in u for u in units)


def test_extract_quantitative_negative():
    assert grounding.extract_quantitative(
        "The cheetah is a big cat.") == []


def test_grounding_coverage():
    trips = [("a", "causes", "b", 0.5), ("a", "causes", "c", 0.5),
             ("d", "is_a", "e", 0.5)]
    anchored = {"a": True, "d": False}
    cov = grounding.grounding_coverage(trips, anchored)
    assert cov["symbols"] == 5  # a, b, c, d, e
    assert cov["grounded"] == 1
    assert abs(cov["coverage"] - 0.2) < 1e-9


def test_quant_infobar_fields():
    # "mass" in der Infobox wird als Anker erkannt
    assert "mass" in grounding._QUANT_INFOBAR
    assert "top_speed" in grounding._QUANT_INFOBAR
