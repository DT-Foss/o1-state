#!/usr/bin/env python3 -u
"""
DER O1-SCHREIBER -- das eigene Modell am Mund, in der Rolle, die ihm
passt: RANKEN statt frei generieren.

Freies Sampling des 6.3M-HSSLM-C erzeugt Degeneration (gemessen:
"—that,—that,—that…" — im Ergebnis-JSON dokumentiert). Die richtige
Architektur für ein kleines eigenes Modell ist deshalb CONSTRAINED:

  FAKTEN     stehen fest (Graph / Weltbuch) — unantastbar.
  SÄTZE      kommen aus der geprüften Varianten-Kaskade (form_arena) und
             der Pronomen-Maschine (diskurs, mit Selbst-Zuhör-Gate).
  DAS MODELL (fertig/hsslm, HSSLM-C mit Möbius-SSM — die o1-Familie,
             geladen aus data/hsslm_form.pt) trifft die FORM-Entscheide:
             welcher Übergang zwischen zwei Sätzen, Pronomen oder voller
             Name, Punkt oder Semikolon — per Logprob-Ranking über eine
             Whitelist. Es kann nichts erfinden, weil es nie Wörter
             erzeugt, nur wählt.
  PRÜFUNG    unsichtbar wie immer (oberflaeche.check_text): fällt sie,
             wird nicht ausgeliefert.

Ohne Gewichte: lauter Fallback (Übergangswahl per TrigramLM-Zählung —
das gewichtfreie Hausmodell), gleiche Schnittstelle. Bessere Gewichte
morgen = bessere Entscheide, null Umbau. Register: FE9.
"""

import json
import os
import sys

import numpy as np

FERTIG_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HERE = os.path.dirname(os.path.abspath(__file__))
for p in (FERTIG_ROOT, HERE):
    if p not in sys.path:
        sys.path.insert(0, p)

from fertig.bench import TrigramLM                       # read-only
from form_arena import select, CORPUS
from diskurs import order_edges, _ddet
from oberflaeche import check_text, facts_health, facts_weltbuch, GRAPH

OPENERS = ["", "Moreover, ", "Over time, ", "In turn, ", "Meanwhile, ",
           "In the end, ", "And ", "Yet "]
JOINERS = [". ", "; "]


class O1Ranker:
    """Logprob-Ranker über dem HSSLM-C (o1-Familie). Deterministisch:
    eval-Modus, keine Sampling-Pfade — nur Scores über Kandidaten."""

    def __init__(self, force_fallback: bool = False):
        self.engine = None
        self.kind = "trigram_fallback"
        if not force_fallback:
            try:
                from fertig.form_engine import FormEngine
                e = FormEngine()
                if e.ready:
                    self.engine = e
                    self.kind = "hsslm_o1"
            except Exception as ex:                   # laut, nie still
                print(f"[schreiber_o1] HSSLM nicht ladbar ({ex}) -> "
                      f"Trigram-Fallback", flush=True)
        self.lm = TrigramLM(open(CORPUS, encoding="utf-8",
                                 errors="ignore").read())

    def logprob(self, text: str) -> float:
        if self.engine is not None:
            import torch
            ids = self.engine._encode(text).to(self.engine.device)
            with torch.no_grad():
                out = self.engine.model.forward(ids)
            logits = out["logits"][0]
            lp = torch.log_softmax(logits[:-1], dim=-1)
            tgt = ids[0][1:]
            tok = lp[torch.arange(tgt.numel()), tgt]
            return float(tok.mean())
        return self.lm.sentence_logprob(text)

    def choose(self, candidates):
        """Deterministische Wahl: bester mittlerer Logprob, Tie-Break
        lexikalisch (Stabilität)."""
        scored = sorted(((self.logprob(c), c) for c in candidates),
                        key=lambda t: (-t[0], t[1]))
        return scored[0][1]


def write_with_o1(facts, ranker, lm):
    """Der Schreibvorgang: geordnete Fakten, geprüfte Basissätze, und das
    o1-Modell wählt Übergang + Pronomen + Fügung."""
    for f in facts:
        f.setdefault("conf", 0.8)
    ordered = order_edges(facts)
    pieces = []
    prev = None
    for i, e in enumerate(ordered):
        best, _, _ = select(lm, e["subj"], e["verb"], e["obj"], GRAPH)
        base = best["prose"]
        candidates = [base]
        if prev is not None and e["subj"] == prev["subj"]:
            candidates.append(f"It also {e['verb']} {_ddet(e['obj'])}.")
        if i > 0:
            base_opts = list(candidates)
            for op in OPENERS[1:]:
                for c in base_opts:
                    if not c.startswith("It also"):
                        candidates.append(op + c[0].lower() + c[1:]
                                          if op else c)
        # das Modell wählt den Satz IM KONTEXT des bisherigen Textes
        tail = " ".join(pieces)[-160:]
        chosen = ranker.choose([(tail + " " + c).strip() for c in candidates])
        chosen = chosen[len(tail):].strip() if chosen.startswith(tail) else \
            sorted(candidates)[0]
        pieces.append(chosen)
        prev = e
    return " ".join(pieces), ordered


def main(out_path=None):
    ranker = O1Ranker()
    lm = ranker.lm
    results = {"model": ranker.kind,
               "free_generation_note": "HSSLM-C frei gesampelt degeneriert "
               "('—that,—that,…') — deshalb rankt das Modell nur; es "
               "erzeugt nie Wörter, es wählt."}
    fallback = O1Ranker(force_fallback=True)
    for name, facts in (("health", facts_health()),
                        ("weltbuch", facts_weltbuch())):
        text, ordered = write_with_o1(list(facts), ranker, lm)
        ok, missing, unbacked = check_text(text, facts)
        # Deko-Kontrolle: entscheidet das Modell wirklich anders als der
        # gewichtfreie Fallback? Identische Texte hieße: Deko.
        fb_text, _ = write_with_o1(list(facts), fallback, lm)
        results[name] = {"released": bool(ok), "n_facts": len(facts),
                         "missing": missing, "unbacked": unbacked,
                         "text": text if ok else None,
                         "model_differs_from_fallback": bool(text != fb_text)}
        print(f"\n════ {'AUSGELIEFERT' if ok else 'ZURÜCKGEHALTEN'}: "
              f"'{name}' via {ranker.kind} ════", flush=True)
        print(text if ok else f"fehlt {missing} | ungedeckt {unbacked}",
              flush=True)
    out_path = out_path or os.path.join(HERE, "results", "schreiber_o1.json")
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"[schreiber_o1] -> {out_path}", flush=True)
    return results


if __name__ == "__main__":
    main()
