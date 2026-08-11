"""FERTIG — Tests der deterministischen Bilderkennung (offline)."""

from __future__ import annotations

from io import BytesIO

import numpy as np
from PIL import Image

from fertig import vision


def _make_image(kind: str, variant: int) -> bytes:
    rng = np.random.RandomState(variant * 100 + len(kind))
    img = np.zeros((64, 64, 3), dtype=np.uint8)
    if kind == "rot":
        img[:, :] = np.array([(140 + rng.randint(-20, 20), 30, 30)],
                             dtype=np.uint8)
        for y in range(0, 64, 8):
            img[y:y + 2, :] = np.array([(200, 60, 60)], dtype=np.uint8)
    elif kind == "blau":
        img[:, :] = np.array([(30 + rng.randint(-15, 15), 50, 150)],
                             dtype=np.uint8)
        for x in range(0, 64, 6):
            img[:, x:x + 2] = np.array([(60, 90, 220)], dtype=np.uint8)
    else:
        img[:, :] = np.array([(30, 150, 50)], dtype=np.uint8)
        for _ in range(12):
            y, x = rng.randint(0, 64, 2)
            img[y:y + 4, x:x + 4] = np.array([(20, 90, 30)], dtype=np.uint8)
    buf = BytesIO()
    Image.fromarray(img).save(buf, format="PNG")
    return buf.getvalue()


def test_signature_deterministic():
    img = _make_image("rot", 0)
    assert np.array_equal(vision.signature(img), vision.signature(img))


def test_signature_within_below_between():
    a1, a2 = _make_image("rot", 0), _make_image("rot", 1)
    b1 = _make_image("blau", 0)
    within = vision.signature_distance(vision.signature(a1),
                                       vision.signature(a2))
    between = vision.signature_distance(vision.signature(a1),
                                        vision.signature(b1))
    assert within < between


def test_cluster_unsupervised_finds_structure():
    # Harnad-Ebene: Kategorien OHNE Labels aus Pixel-Struktur
    images, labels = [], []
    for kind in ("rot", "blau", "gruen"):
        for v in range(4):
            images.append(_make_image(kind, v))
            labels.append(kind)
    clusters, centers = vision.cluster_unsupervised(images, k=3)
    assert len(clusters) == 3
    purity = vision.cluster_purity(clusters, labels)
    assert purity > 0.9


def test_category_bank_recognize():
    bank = vision.CategoryBank()
    for kind in ("rot", "blau"):
        sigs = [vision.signature(_make_image(kind, v)) for v in range(3)]
        bank.add(kind, sigs)
    word, d = bank.recognize(_make_image("rot", 7))
    assert word == "rot"
