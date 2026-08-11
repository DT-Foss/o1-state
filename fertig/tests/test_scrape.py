"""FERTIG — Tests des Scrape-/Gap-Moduls (offline)."""

from __future__ import annotations

from fertig import scrape, gaps
from fertig._vendor.dotcausal import CausalWriter, CausalReader


def test_extract_is_a_pattern():
    trips = scrape.extract_triplets(
        "sugar", "Sweet-tasting carbohydrates",
        "Sugar is a class of sweet-tasting soluble carbohydrates. "
        "Simple sugars include glucose.",
        {})
    assert ("sugar", "is_a", "class of sweet-tasting soluble carbohydrates",
            0.5) in trips
    assert any(t[1] == "described_as" for t in trips)


def test_extract_infobox_tier():
    trips = scrape.extract_triplets("sugar", None, "",
                                    {"energy": "1,600 kJ"})
    assert ("sugar", "energy", "1,600 kj", 0.7) in trips


def test_extract_dedupe_max_conf():
    trips = scrape.extract_triplets("sugar", None, "",
                                    {"color": "white"})
    # gleiches Triplett nur einmal
    keys = [(a, b, c) for a, b, c, _ in trips]
    assert len(keys) == len(set(keys))


def test_infobox_cleanup():
    raw = "[[White sugar|white]] <ref>x</ref> {{small|y}}"
    # die Regex-Funktion wird über extract_triplets getestet — hier nur
    # sicherstellen, dass der Parser keine Klammerreste durchlässt
    trips = scrape.extract_triplets("sugar", None, "", {"color": raw})
    assert not any("[[White" in t[2] for t in trips)


def test_world_roundtrip(tmp_path):
    w = CausalWriter(api_id="fertig")
    w.add_triplet("sugar", "causes", "cavities", confidence=0.7)
    p = tmp_path / "world.causal"
    w.save(str(p))
    r = CausalReader(str(p)).get_all_triplets()
    assert any(t["trigger"] == "sugar" and t["outcome"] == "cavities"
               for t in r)


def test_load_world_empty(tmp_path, monkeypatch):
    monkeypatch.setattr(gaps, "WORLD_GRAPH", tmp_path / "missing.causal")
    assert gaps._load_world() == []
