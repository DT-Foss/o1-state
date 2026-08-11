"""
fertig — das fertige symbolische Sprachsystem.

Deterministische, gewicht-freie Sprach-Erzeugung aus zwei gemessenen Quellen:
  * .causal-Wissensgraphen  (Fakten exakt, Walk generiert)
  * Korpora                 (gemessene Bigramm/Trigramm-Übergänge)

Kein neuronales Netz, kein Training, keine Embeddings. Gleiche Eingabe ->
gleiche Ausgabe. Alles beruht auf den Foss-Preprints (siehe README, Abschnitt
"Grundlagen"): Kontraktion (F33-F40), Möbius-Kopplung, Ginibre-Kerne (F38),
BvN-Zerlegung (F63-F68), Berry-Phasen-Wächter (bphm).

Module:
  sampler        tau-kontrollierter Kontraktions-Sampler (Zeno, Ginibre, BvN)
  state_init     hyperboloide Symbol-Zustände (Berry-Phasen-Guard)
  bphm           Berry-Phasen-Wiederholungs-Erkennung
  pattern_bank   aus Korpora gemessene Satzform-Muster
  inference      Jaro-Winkler + 3-Pass-Ketten-Inferenz
  pipeline       .causal -> Walk -> Sprache (Fakten exakt, Form generiert)
  corpus         Korpus-Modus: gemessene Übergänge -> Fortsetzung
  mined          gesprochene Form mit gemessener Muster-Bank
"""

from __future__ import annotations

__version__ = "1.0.0"

from . import sampler, state_init, bphm, pattern_bank, inference
from . import pipeline, corpus, mined
from . import intent, tools, learn, arena, bench, grammar, code, scrape, gaps, grounding, quant, vision, video, stream, interp

__all__ = [
    "sampler", "state_init", "bphm", "pattern_bank", "inference",
    "pipeline", "corpus", "mined",
    "intent", "tools", "learn", "arena", "bench", "grammar", "code",
    "scrape", "gaps", "grounding",
    "__version__",
]
