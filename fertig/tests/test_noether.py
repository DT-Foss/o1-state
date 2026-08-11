"""FERTIG — Tests für fertig.noether (Zahlentheorie + Signal-Struktur).

Umgezogen aus test_primitives.py (Ring-Homomorphismus/Noether-Tests
gehoerten dort zu primitives.py, das jetzt die reine Relations-Schema-API
ist -- digital_root/digital_root_class/detect_periodicity/detect_parity
leben jetzt in fertig.noether, siehe dessen Modul-Docstring fuer die
Begruendung des Umzugs)."""

from __future__ import annotations

from fertig import noether


def test_digital_root_ring_homomorphism():
    # dr(a+b) = dr(dr(a)+dr(b))
    a, b = "sugar", "candy"
    assert noether.digital_root(a + b) == \
        noether.digital_root(noether.digital_root(a) +
                             noether.digital_root(b))
    # 9 statt 0
    assert noether.digital_root("ii") == 9  # i=9, 9+9=18 -> 9


def test_digital_root_class_groups():
    # Kommutativität: gleiche Buchstaben -> gleiche Klasse
    assert noether.digital_root_class("ab") == \
        noether.digital_root_class("ba")
    # Anhängen von 9er-Schritten ändert die Klasse nicht (9 ≡ 0 mod 9)
    assert noether.digital_root_class("a") == \
        noether.digital_root_class("ai")


def test_detect_periodicity():
    seq = [1.0, 2.0, 1.0, 2.0, 1.0, 2.0]
    assert noether.detect_periodicity(seq) == 2
    assert noether.detect_periodicity([1.0, 2.0, 3.0, 4.0]) is None


def test_detect_parity():
    assert noether.detect_parity([1.0, 2.0, 2.0, 1.0]) == "even"
    assert noether.detect_parity([1.0, 2.0, -2.0, -1.0]) == "odd"
    assert noether.detect_parity([1.0, 2.0, 3.0]) is None
