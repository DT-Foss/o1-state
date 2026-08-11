"""
Junk-Schluessel-Filter fuer trigger_key/outcome_key (Sprecher-Datenbasis v2,
Baustein: Store bleibt unberuehrt, der Filter wirkt nur bei der
Paar-Erzeugung -- keine Mutation des LiveStore).

WT-103's Markup + der freie builder_v0-Extraktor lassen vier Junk-Muster
in trigger_key/outcome_key durchsickern (gemessen auf p72_store_local,
3916 unique keys, siehe compute_junk_stats()):

  1. "="-Ueberschriften-Reste   : Wikipedia-Sektions-Marker roh im Text
                                   ("= = overview", "brick = = main").
  2. "@"-Artefakte               : WT-103's Tokenizer trennt Satzzeichen mit
                                   @ ab ("3 @.@ 5 m", "@-@ king ravana",
                                   "on @-@ site museum has").
  3. Kein alphabetisches Wort    : reine Zahlen-/Symbolfragmente, kein
                                   [a-zA-Z]-Zeichen im Schluessel.
  4. Ein-Zeichen-Schluessel      : ein einzelnes Zeichen nach Trim, zu kurz
                                   um ein Konzept zu sein.

Ein Schluessel ist Junk, wenn IRGENDEINE der vier Regeln zutrifft (Regeln
sind Teilmengen-agnostisch -- ein Schluessel mit "=" UND "@" wird nur
einmal gezaehlt, siehe is_junk_key()). Diese Funktionen sind reine
str->bool/Reports -- kein Store-Zugriff, kein Seiteneffekt."""

from __future__ import annotations

import re
from typing import Dict, List, Tuple

_HAS_ALPHA = re.compile(r"[a-zA-Z]")


def is_junk_key(key: str) -> bool:
    """True wenn key eines der vier Junk-Muster traegt."""
    k = str(key)
    if "=" in k:
        return True
    if "@" in k:
        return True
    if not _HAS_ALPHA.search(k):
        return True
    if len(k.strip()) <= 1:
        return True
    return False


def junk_reason(key: str) -> str:
    """Welche Regel zuerst greift (fuer Reports/Beispiele) -- Reihenfolge
    wie is_junk_key(), erste zutreffende Regel gewinnt."""
    k = str(key)
    if "=" in k:
        return "heading_residue (=)"
    if "@" in k:
        return "at_artifact (@)"
    if not _HAS_ALPHA.search(k):
        return "no_alpha_word"
    if len(k.strip()) <= 1:
        return "single_char"
    return "clean"


def filter_pair(pair: dict, key_fields: Tuple[str, ...] = ("trigger", "outcome")) -> bool:
    """True wenn das Paar BEHALTEN werden soll (kein Feld in key_fields ist
    Junk). Arbeitet auf den generischen Feldnamen eines Trainingspaar-Dicts
    (trigger/outcome oder trigger_key/outcome_key je nach Aufrufer) --
    prueft nur Felder, die tatsaechlich vorhanden sind."""
    for field in key_fields:
        if field in pair and is_junk_key(pair[field]):
            return False
    return True


def compute_junk_stats(keys) -> Dict:
    """Junk-Anteil + Beispiele ueber eine Iterable von Schluessel-Strings.
    Reine Messfunktion, kein Store-Zugriff -- der Aufrufer liefert die
    Schluesselmenge (z.B. alle trigger_key/outcome_key eines Stores)."""
    keys = list(keys)
    total = len(keys)
    junk_keys: List[str] = []
    clean_keys: List[str] = []
    reason_counter: Dict[str, int] = {}

    for k in keys:
        reason = junk_reason(k)
        if reason == "clean":
            clean_keys.append(k)
        else:
            junk_keys.append(k)
            reason_counter[reason] = reason_counter.get(reason, 0) + 1

    return {
        "total": total,
        "n_junk": len(junk_keys),
        "n_clean": len(clean_keys),
        "junk_ratio": len(junk_keys) / total if total else 0.0,
        "reason_counts": dict(sorted(reason_counter.items(), key=lambda kv: -kv[1])),
        "junk_examples": junk_keys[:10],
        "clean_examples": clean_keys[:10],
    }
