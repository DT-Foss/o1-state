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


def test_extract_quantitative_thousands_comma():
    """Regression: '6,350 kg' is a thousands-grouped number (6350 kg), not
    a decimal (6.35 kg). LIVE_GROUNDING_READONLY_REVIEW.md's finding --
    the pre-existing test above used this exact string but only checked
    the unit STRING, never the parsed numeric value, so the bug (every
    comma treated as a decimal point) went undetected."""
    trips = grounding.extract_quantitative(
        "Elephants weigh up to 6,350 kg.")
    values = {u: val for _, u, val in trips}
    kg_entries = [v for u, v in values.items() if "kg" in u]
    assert kg_entries, "expected at least one kg entry"
    assert kg_entries[0] == 6350.0, (
        f"expected 6350.0 (thousands separator), got {kg_entries[0]} "
        "(comma misparsed as decimal point)"
    )


def test_parse_number_thousands_vs_decimal():
    """The parser rule (see grounding._parse_number's docstring for the
    full heuristic): a comma followed by exactly 3 digits (repeatable) is
    a thousands separator; a comma followed by 1-2 trailing digits is a
    decimal separator. English-locale web-text heuristic, not a full
    locale parser."""
    assert grounding._parse_number("6,350") == 6350.0        # thousands
    assert grounding._parse_number("1,234,567") == 1234567.0  # repeated thousands
    assert grounding._parse_number("6,35") == 6.35            # decimal (2 trailing digits)
    assert grounding._parse_number("3,14") == 3.14            # decimal (2 trailing digits)
    assert grounding._parse_number("110") == 110.0            # no comma at all


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
