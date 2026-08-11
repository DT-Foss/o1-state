"""
Parameterisiertes Vokabular fuer den Sprecher (Sprecher-Datenbasis v2,
Baustein 4). length_extrap_v2.build_vocab() ist fest auf VOCAB_MAX=5000
verdrahtet (Organism-Vergleichsmessungen haengen an dieser Zahl) -- WIRD
NICHT in-place geaendert. Dieses Modul stellt stattdessen eine eigene
parameterisierte Funktion bereit, die dieselbe Frequenz-Sortierung nutzt,
aber jede Vokabulargroesse n erlaubt.

ID-SCHEMA (identisch zum bestehenden Organism-5002 + <fact>/<say>-Schema
aus graph_to_text.py/speak_train.py, nur n statt fest 5000):
    0 .. n-1        : die n haeufigsten Woerter
    n               : unk
    n+1             : mask
    n+2             : <fact>
    n+3             : <say>
    total_ids = n+4

Persistenz: ein Vokabular-File ist eine Zeile pro Wort in Frequenz-
Reihenfolge (Index = ID 0..n-1), damit speak_train.py --vocab-file ein
GENAU REPRODUZIERBARES Vokabular laden kann (kein Neuaufbau aus dem
Korpus noetig, kein Risiko einer abweichenden HF-Datasets-Version, die
eine andere Wortfrequenz-Reihenfolge liefert).
"""

from __future__ import annotations

import re
from typing import Dict, List, Tuple

_WORD_RE = re.compile(r"[a-zA-Z]+")

FACT_TOKEN = "<fact>"
SAY_TOKEN = "<say>"


def build_vocab_n(text: str, n: int) -> Tuple[List[str], Dict[str, int], int, int]:
    """Wie length_extrap_v2.build_vocab, aber mit parameterisierter Groesse
    n statt dem festen VOCAB_MAX=5000. Gleiche Tokenisierung ([a-zA-Z]+,
    lowercased), gleiche Sortierung (Frequenz absteigend, bei Gleichstand
    stabil nach erster Fundstelle -- Python's sort ist stabil, dict-
    Iterationsreihenfolge ist Einfuegereihenfolge, also deterministisch
    fuer denselben Text).

    Returns (vocab, stoi, unk_id, mask_id) -- gleiche Signatur-Form wie
    length_extrap_v2.build_vocab (dort: vocab, stoi, len(vocab), len(vocab)+1)."""
    words = _WORD_RE.findall(text.lower())
    freq: Dict[str, int] = {}
    for w in words:
        freq[w] = freq.get(w, 0) + 1
    vocab = [w for w, _ in sorted(freq.items(), key=lambda kv: -kv[1])][:n]
    stoi = {w: i for i, w in enumerate(vocab)}
    return vocab, stoi, len(vocab), len(vocab) + 1


def build_extended_vocab_n(text: str, n: int):
    """build_vocab_n + <fact>/<say>-Erweiterung, analog speak_train.py's
    build_extended_vocab (dort fest an length_extrap_v2.build_vocab
    gebunden). Returns (stoi, unk_id, mask_id, fact_id, say_id, total_ids)."""
    vocab, stoi, unk, mask = build_vocab_n(text, n)
    fact_id = mask + 1
    say_id = mask + 2
    stoi = dict(stoi)
    stoi[FACT_TOKEN] = fact_id
    stoi[SAY_TOKEN] = say_id
    total_ids = say_id + 1
    return stoi, unk, mask, fact_id, say_id, total_ids


def tokenize_n(text: str, stoi: dict, unk: int) -> List[int]:
    """Gleiche Tokenisierung wie length_extrap_v2.tokenize (identischer
    Regex, damit ein mit build_vocab_n gebautes Vokabular konsistent mit
    Text getokenized wird, egal welches n gewaehlt wurde)."""
    return [stoi.get(w, unk) for w in _WORD_RE.findall(text.lower())]


def save_vocab(path: str, vocab: List[str]) -> None:
    """Ein Wort pro Zeile, Index = ID (0..n-1). unk/mask/fact/say sind NICHT
    im File (sie sind aus n ableitbar, siehe load_vocab)."""
    with open(path, "w", encoding="utf-8") as f:
        for w in vocab:
            f.write(w + "\n")


def load_vocab(path: str):
    """Vokabular-File -> (vocab, stoi, unk_id, mask_id, fact_id, say_id,
    total_ids). Reproduziert exakt dasselbe ID-Schema wie
    build_extended_vocab_n, ohne den Korpus erneut zu scannen."""
    with open(path, encoding="utf-8") as f:
        vocab = [line.rstrip("\n") for line in f if line.rstrip("\n")]
    n = len(vocab)
    stoi = {w: i for i, w in enumerate(vocab)}
    unk_id = n
    mask_id = n + 1
    fact_id = n + 2
    say_id = n + 3
    stoi[FACT_TOKEN] = fact_id
    stoi[SAY_TOKEN] = say_id
    total_ids = say_id + 1
    return vocab, stoi, unk_id, mask_id, fact_id, say_id, total_ids


def compute_oov_stats(texts: List[str], stoi: dict, unk_id: int) -> Dict:
    """OOV-Rate ueber eine Liste von Texten gegen ein gegebenes (stoi,
    unk_id)-Paar -- die gemeinsame Messfunktion fuer den 5k-vs-20k-
    Vergleich (LM-Stream und v2-Paare, siehe Report)."""
    rates = []
    lens = []
    for t in texts:
        words = _WORD_RE.findall(t.lower())
        if not words:
            continue
        n_oov = sum(1 for w in words if stoi.get(w, unk_id) == unk_id)
        rates.append(n_oov / len(words))
        lens.append(len(words))
    if not rates:
        return {"mean_oov_rate": 0.0, "median_oov_rate": 0.0,
                "pct_texts_zero_oov": 1.0, "n_texts": 0}
    return {
        "mean_oov_rate": sum(rates) / len(rates),
        "median_oov_rate": sorted(rates)[len(rates) // 2],
        "pct_texts_zero_oov": sum(1 for r in rates if r == 0.0) / len(rates),
        "n_texts": len(rates),
    }
