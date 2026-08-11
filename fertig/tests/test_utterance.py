"""FERTIG — Tests des Utterance-IR (Plan -> Prosa -> Rückverifikation)."""

from __future__ import annotations

from pathlib import Path

from fertig.utterance import speak, plan_from_graph

DATA = Path(__file__).resolve().parent.parent / "data" / "chained.causal"


def test_plan_from_graph():
    plan = plan_from_graph(str(DATA), "smoking", n=3)
    assert plan
    assert all(len(k) == 4 for k in plan)
    assert plan[0][0] == "smoking"


def test_ir_closed_circle():
    res = speak(str(DATA), "smoking", n=5)
    assert len(res.utterances) == 5
    assert res.all_verified
    assert res.verified_count == 5
    # jede Kante ist wörtlich in der Prosa belegt
    for u in res.utterances:
        assert u.obj in res.prose.lower()


def test_ir_abstains_on_missing_evidence():
    # eine Kante, die die Prosa NICHT trägt, wird verworfen
    res = speak(str(DATA), "smoking", n=5)
    # manipuliere: füge eine unbelegte Kante hinzu — sie zählt NICHT
    from fertig.utterance import Utterance
    res.utterances.append(Utterance("smoking", "causes", "cancer", 0.9,
                                    prose="Smoking causes tar buildup."))
    assert res.verified_count == 5  # die unbelegte Kante blieb unverifiziert
