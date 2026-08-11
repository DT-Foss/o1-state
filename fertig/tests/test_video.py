"""FERTIG — Tests des Video-Moduls (offline, generierte GIFs)."""

from __future__ import annotations

from io import BytesIO

import numpy as np
from PIL import Image

from fertig import video


def _make_gif(kind: str, variant: int, n_frames: int = 16) -> bytes:
    rng = np.random.RandomState(variant * 100 + len(kind))
    frames = []
    for t in range(n_frames):
        img = np.zeros((48, 48, 3), dtype=np.uint8)
        if kind == "puls":
            r = 6 + int(10 * abs(np.sin(t / 3.0)))
            y, x = np.ogrid[:48, :48]
            img[(y - 24) ** 2 + (x - 24) ** 2 < r ** 2] = (220, 60, 60)
        elif kind == "wander":
            x0 = (t * 3) % 40
            img[:, x0:x0 + 6] = (60, 60, 220)
        else:
            img[10:38, 10:38] = (60, 200, 80)
        noise = rng.randint(-4, 5, img.shape, dtype=np.int16)
        img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        frames.append(Image.fromarray(img))
    buf = BytesIO()
    frames[0].save(buf, format="GIF", save_all=True,
                   append_images=frames[1:], duration=80, loop=0)
    return buf.getvalue()


def test_understand_motion_types():
    # zwei Bewegungs-Primitive: puls (Signatur), wander (Pixel)
    u_puls = video.understand(video.extract_frames(_make_gif("puls", 0)))
    u_wander = video.understand(video.extract_frames(_make_gif("wander", 0)))
    u_stat = video.understand(video.extract_frames(_make_gif("statisch", 0)))
    assert u_puls["bewegt"]
    assert u_wander["bewegt"]
    assert not u_stat["bewegt"]


def test_understand_noise_calibration():
    # statisches Video: Signatur-Signal == Rausch-Level
    u = video.understand(video.extract_frames(_make_gif("statisch", 1)))
    assert u["signatur"] < 3 * video.SIGNATURE_NOISE
    assert u["pixel"] < 3 * video.PIXEL_NOISE


def test_video_bank_recognizes_unseen():
    bank = video.VideoBank()
    for kind in ("puls", "wander", "statisch"):
        for v in range(2):
            raw = video.extract_frames(_make_gif(kind, v))
            bank.add(kind, video.frame_signatures(raw))
    hits = 0
    for kind in ("puls", "wander", "statisch"):
        for v in range(2, 4):
            raw = video.extract_frames(_make_gif(kind, v))
            pred, _ = bank.recognize(video.frame_signatures(raw))
            hits += pred == kind
    assert hits == 6


def test_generation_uses_measured_grammar():
    raw = video.extract_frames(_make_gif("puls", 0))
    codes = video.frame_code(video.frame_signatures(raw))
    trans = video.learn_transitions(codes)
    gen = video.generate_frames(codes[0], trans, n=16)
    assert len(gen) == 16
    # jede Kante der generierten Sequenz stammt aus der gemessenen Grammatik
    for a, b in zip(gen, gen[1:]):
        assert b in trans.get(a, {}) or a in trans
    # kein Attraktor-Loop: die letzten 4 Codes sind nicht identisch
    assert len(set(gen[-4:])) > 1


def test_video_bank_save_load(tmp_path):
    bank = video.VideoBank()
    for kind in ("puls", "wander"):
        raw = video.extract_frames(_make_gif(kind, 0))
        bank.add(kind, video.frame_signatures(raw))
    p = tmp_path / "bank.json"
    bank.save(p)
    loaded = video.VideoBank().load(p)
    assert set(loaded.prototypes) == {"puls", "wander"}


def test_video_bank_unknown_threshold():
    bank = video.VideoBank(threshold=0.01)
    raw = video.extract_frames(_make_gif("puls", 0))
    bank.add("puls", video.frame_signatures(raw))
    # fremde Video-Signatur -> ehrlich 'unbekannt' über der Schwelle
    fremd_raw = video.extract_frames(_make_gif("wander", 5))
    word, d = bank.recognize_signature(
        video.sequence_signature(video.frame_signatures(fremd_raw)))
    assert word is None
