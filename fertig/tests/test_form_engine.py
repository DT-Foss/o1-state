"""FERTIG — Tests der Form-Engine (Fallback-Pfad ohne trainierte Gewichte)."""

from __future__ import annotations

from pathlib import Path

from fertig.form_engine import speak_with_engine, FormEngine

DATA = Path(__file__).resolve().parent.parent / "data" / "chained.causal"


def test_fallback_without_weights():
    # ohne trainierte Gewichte: deterministischer verbalize-Fallback
    r = speak_with_engine(str(DATA), "smoking")
    assert r["ok"]
    assert r["engine"] == "fallback"
    assert r["verified"]
    assert "tar buildup" in r["prose"]


def test_engine_missing_weights(tmp_path):
    # gewichte-freier Pfad -> not ready, keine Varianten
    import fertig.form_engine as fe
    e = FormEngine(model_path=str(tmp_path / "fehlt.pt"))
    assert not e.ready
    assert e.variants("Smoking causes") == []
