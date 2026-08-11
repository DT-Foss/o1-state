"""FERTIG — Tests des Stream-Learners (offline, synthetische Streams)."""

from __future__ import annotations

import numpy as np

from fertig import stream


def _synth_wander(n=60):
    for t in range(n):
        img = np.zeros((64, 64, 3), dtype=np.float32)
        speed = 2 + int(2 * abs(np.sin(t / 15.0)))
        x0 = (t * speed) % 60
        img[:, x0:x0 + 5] = (0.8, 0.2, 0.2)
        yield img


def test_learner_detects_motion():
    l = stream.StreamLearner()
    for f in _synth_wander(60):
        l.update(f)
    st = l.state()
    # Pixel-Signal fängt Translation (Histogramme sind invariant);
    # der synthetische Stream ist rauschfrei -> pixel > 0.01 ist klar
    # oberhalb des Rausch-Levels
    assert st["pixel"] > 0.01


def test_learner_constant_memory():
    # O(1)-Beweis: Zustandsgröße wächst nicht mit der Frame-Zahl
    l = stream.StreamLearner()
    for f in _synth_wander(30):
        l.update(f)
    size_30 = sum(len(str(v)) for v in l.state().values())
    l2 = stream.StreamLearner()
    for f in _synth_wander(300):
        l2.update(f)
    size_300 = sum(len(str(v)) for v in l2.state().values())
    # gleiche Zustandsfelder, begrenzte Grammatik
    assert len(l.state()) == len(l2.state())
    assert l2.state()["memory_konstant"]


def test_learner_generates_from_grammar():
    l = stream.StreamLearner()
    for f in _synth_wander(60):
        l.update(f)
    gen = l.generate(8)
    assert len(gen) >= 1
    assert all(0 <= c < l.n_bins for c in gen)


def test_fast_signature_deterministic():
    f = np.zeros((64, 64, 3), dtype=np.float32)
    f[:, 10:20] = (1.0, 0.0, 0.0)
    assert np.array_equal(stream.fast_signature(f), stream.fast_signature(f))


def test_learner_to_video_bank_cycle(tmp_path, monkeypatch):
    # Der komplette Zyklus: Stream -> Kategorie -> erkennen -> Graph-Fakt
    from fertig import video as v
    from fertig.gaps import WORLD_GRAPH
    monkeypatch.setattr("fertig.gaps.WORLD_GRAPH", tmp_path / "w.causal")
    l = stream.StreamLearner()
    for f in _synth_wander(60):
        l.update(f)
    bank = v.VideoBank()
    bank.add_from_learner("wander", l)
    assert "wander" in bank.prototypes
    # zweiter Learner desselben Typs wird erkannt
    l2 = stream.StreamLearner()
    for f in _synth_wander(60):
        l2.update(f)
    word, d = bank.recognize_signature(l2.sequence_signature())
    assert word == "wander"
    # Graph-Fakten
    facts = l.to_graph_facts("wander")
    assert any("bewegung" in b for _, b, _, _ in facts)


def test_fast_signature_ordered():
    # andere Farben -> andere Signaturen
    a = np.zeros((64, 64, 3), dtype=np.float32)
    b = np.zeros((64, 64, 3), dtype=np.float32)
    a[:, :] = (1.0, 0.0, 0.0)
    b[:, :] = (0.0, 0.0, 1.0)
    assert not np.array_equal(stream.fast_signature(a),
                              stream.fast_signature(b))
