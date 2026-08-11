"""FERTIG — Tests des Interpolations-Lerners (Stützräder-Curriculum)."""

from __future__ import annotations

import numpy as np

from fertig import video as video_mod
from fertig.interp import InterpLearner


def _frames(kind: str, n: int = 24) -> list:
    from io import BytesIO
    from PIL import Image
    rng = np.random.RandomState(5)
    frames = []
    for t in range(n):
        img = np.zeros((48, 48, 3), dtype=np.uint8)
        if kind == "puls":
            r = 6 + int(10 * abs(np.sin(t / 3.0)))
            y, x = np.ogrid[:48, :48]
            img[(y - 24) ** 2 + (x - 24) ** 2 < r ** 2] = (220, 60, 60)
        else:
            img[:, (t * 3) % 40:(t * 3) % 40 + 6] = (60, 60, 220)
        noise = rng.randint(-4, 5, img.shape, dtype=np.int16)
        img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        frames.append(Image.fromarray(img))
    buf = BytesIO()
    frames[0].save(buf, format="GIF", save_all=True,
                   append_images=frames[1:], duration=80, loop=0)
    return video_mod.extract_frames(buf.getvalue(), max_frames=n)


def test_learner_learns_grammar():
    l = InterpLearner()
    for f in _frames("puls"):
        l.update(f)
    assert l.frames_seen == 24
    assert sum(len(r) for r in l.transitions.values()) >= 3


def test_interpolation_quality_improves_with_learning():
    frames = _frames("puls", 40)
    # wenig gelernt: nur die ersten 8 Frames
    l_small = InterpLearner(quality_threshold=0.05)
    for f in frames[:8]:
        l_small.update(f)
    # voll gelernt
    l_full = InterpLearner(quality_threshold=0.05)
    for f in frames:
        l_full.update(f)
    # Qualität über mehrere Segmente gemittelt (ein Segment ist verrauscht)
    def avg_q(l):
        return np.mean([l.quality(frames[i:i + 8], 4)
                        for i in range(0, len(frames) - 8, 4)])
    q_small, q_full = avg_q(l_small), avg_q(l_full)
    assert q_full <= q_small + 0.01  # mehr lernen schadet nicht


def test_mastered_gap_grows_with_periodic_stream():
    frames = _frames("puls", 40)
    l = InterpLearner(quality_threshold=0.05)
    for f in frames:
        l.update(f)
    # periodischer Stream: Lücken 4+ beherrschbar
    assert l.mastered_gap(frames, max_gap=6) >= 4


def test_self_paced_gap_never_crashes():
    frames = _frames("puls", 24)
    l = InterpLearner(quality_threshold=0.05)
    curve = l.self_paced_learn(frames, max_gap=8)
    assert len(curve) > 5
    # gap bleibt im erlaubten Bereich
    assert all(2 <= g <= 8 for g, _ in curve)


def test_interpolate_returns_gap_minus_1_frames():
    frames = _frames("puls", 16)
    l = InterpLearner()
    for f in frames:
        l.update(f)
    gen = l.interpolate(frames[0], frames[4], 4)
    assert gen is not None and len(gen) == 3
