"""
fertig.noether — Zahlentheorie- und Signal-Struktur-Werkzeuge.

Vier Werkzeuge, ZWEI Herkunftsgruppen, EIN gemeinsamer Hinweis:

  wirkung    : detect_periodicity / detect_parity — Noether-Detektoren
               (Periodizität/Paritaet) auf reellwertigen Sequenzen. Live
               genutzt von fertig.stream/fertig.video zur Bewegungs-/
               Struktur-Analyse von Signatur-Zeitreihen.

  vortexmath : digital_root / digital_root_class — Digitalwurzel
               dr(n) = n mod 9, ein idempotenter Ring-Homomorphismus
               ℕ -> ℤ/9ℤ (dr(a+b) = dr(dr(a)+dr(b)), dr(a·b) = dr(dr(a)·dr(b))).

WICHTIGER HINWEIS (Umzug aus primitives.py, per Lead-Entscheidung nach
_codex_lab/primitive_schema_snapshot/LAB_NOTES.md's Formel-Audit):

  digital_root/digital_root_class sind ZAHLENTHEORETISCHE Werkzeuge --
  NICHT semantisch nutzen. Sie wurden zuvor in primitives.py als
  Klassifikations-FALLBACK für unbekannte Oberflaechenformen verwendet
  (digital_root_class(word) gruppierte unbekannte Mechanismen ueber ihre
  Buchstaben-Summe mod 9 in 9 "Bedeutungsklassen"). LAB_NOTES.md's
  Formel-Audit verwarf das explizit: "Digitalwurzeln sind Mod-9-Buckets.
  Die eigenen Vortex-Notizen berichten Nachteile fuer Sprache und
  schlechtere PPM-Werte; als Synonym-Tiebreaker waeren sie ein
  kollisionsreicher Hash." Die neue primitives.py (RelationFamily/
  RelationSpec) laesst unbekannte Mechanismen ehrlich unbekannt statt sie
  ueber diesen Hash zu "raten" -- siehe primitives.py's eigenen Docstring.

  digital_root selbst wird HIER erhalten und getestet (siehe
  tests/test_noether.py, Ring-Homomorphismus-Eigenschaften), aber NICHT
  mehr von primitives.py oder relations.py als semantischer Fallback
  aufgerufen. Falls ein zukuenftiger Konsument eine Zahlentheorie-Funktion
  braucht (Hashing, Checksummen, o.ae.), ist das hier der richtige Ort --
  eine semantische Klassifikations-Nutzung ist es nicht.
"""

from __future__ import annotations

from typing import List, Optional

# ---------------------------------------------------------------------------
# vortexmath: Digitalwurzel (Ring-Homomorphismus ℕ -> ℤ/9ℤ)
# ---------------------------------------------------------------------------

_LETTERS = "abcdefghijklmnopqrstuvwxyz"


def digital_root(value) -> int:
    """dr(n) = n mod 9 (9 statt 0). Idempotenter Ring-Homomorphismus:
    dr(a+b) = dr(dr(a)+dr(b)), dr(a·b) = dr(dr(a)·dr(b)).

    Reines Zahlentheorie-Werkzeug -- siehe Modul-Docstring fuer die
    Begruendung, warum es NICHT (mehr) als semantischer Klassifikations-
    Fallback verwendet wird."""
    if isinstance(value, str):
        n = sum(_LETTERS.index(c) + 1 for c in value.lower()
                if c in _LETTERS)
    else:
        n = int(value)
    r = n % 9
    return 9 if r == 0 else r


def digital_root_class(word: str) -> int:
    """Digitalwurzel-Klasse eines Worts (9 Klassen, 5 Orbit-Typen unter der
    ueblichen Vortexmath-Verdopplungsfolge). NICHT als Bedeutungs-/
    Synonym-Klassifikator verwenden -- siehe Modul-Docstring."""
    return digital_root(word)


# ---------------------------------------------------------------------------
# wirkung: Noether-Detektoren (Parität/Periodizität — Struktur-Primitive)
# ---------------------------------------------------------------------------

def detect_periodicity(seq: List[float], tol: float = 0.05) -> Optional[int]:
    """Periodizitäts-Detektor (Noether-Stil): kleinste Periode p mit
    |x[i+p] − x[i]| < tol für alle i. None wenn nicht periodisch."""
    n = len(seq)
    for p in range(1, n // 2 + 1):
        if all(abs(seq[i + p] - seq[i]) < tol for i in range(n - p)):
            return p
    return None


def detect_parity(seq: List[float], tol: float = 0.05) -> Optional[str]:
    """Paritäts-Detektor: 'even' (symmetrisch) / 'odd' (antisymmetrisch)."""
    n = len(seq)
    if n < 2:
        return None
    even = all(abs(seq[i] - seq[n - 1 - i]) < tol for i in range(n))
    if even:
        return "even"
    odd = all(abs(seq[i] + seq[n - 1 - i]) < tol for i in range(n))
    return "odd" if odd else None
