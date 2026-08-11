#!/usr/bin/env python3 -u
"""
DIE OBERFLÄCHE -- der Nutzer sieht nur den Text. Alles andere ist
Qualitätskontrolle und bleibt unter der Haube.

Arbeitsteilung (das Produkt-Prinzip in einem Satz: schreibt wie die
Großen, erfindet aber nichts):

  FAKTEN     kommen aus FERTIGs Graphen (gemessen bzw. gelebt).
  OBERFLÄCHE schreibt ein austauschbarer starker Schreiber -- heute die
             Demo-Texte unten (verfasst von einem Frontier-Modell im
             Rahmen dieser Session), morgen ein API-Modell oder die
             eigene HSSLM-Engine. Der Schreiber ist UNVERTRAUT.
  PRÜFUNG    läuft unsichtbar: jeder Fakt muss im Text vorkommen
             (Entitäten wörtlich, Verb-Signal im selben Satz, Ziffern
             wörtlich), und kein Kausalsatz darf etwas behaupten, das
             nicht in der Faktenbasis steht. Fällt die Prüfung, wird
             der Text NICHT ausgeliefert -- der Nutzer bekommt nie
             etwas Falsches zu sehen, aber auch nie die Prüfung selbst.

Damit ist die Rollenfrage geklärt: Templates (diskurs.py) sind der
Fallback, der immer geht; die Klasse des Outputs kommt vom besten
verfügbaren Schreiber, und FERTIG ist der Grund, warum man ihm trauen
darf, ohne es zu merken.
"""

import json
import os
import re
import sys

FERTIG_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HERE = os.path.dirname(os.path.abspath(__file__))
for p in (FERTIG_ROOT, HERE):
    if p not in sys.path:
        sys.path.insert(0, p)

from fertig.pipeline import _toks                      # read-only
from fertig.utterance import _VERB_SIGNALS             # read-only
from lifted_walk import load_fertig_graph
from weltbuch import aggregate_records, build_graph as build_weltbuch

GRAPH = os.path.join(FERTIG_ROOT, "data", "chained.causal")


# ═══════════════════════════════════════════════════════════════════════════
#  Unsichtbare Prüfung
# ═══════════════════════════════════════════════════════════════════════════
def _norm(s):
    return " " + " ".join(_toks(s)) + " "


def _signals(verb):
    sigs = list(_VERB_SIGNALS.get(verb, []))
    base = verb.replace("_", " ")
    sigs += [base]
    # Flexionsvarianten des Kopfverbs (causes/cause/caused, shifts/shift...)
    head = base.split()[0]
    for suf in ("s", "d", "ed", ""):
        if head.endswith(suf) and len(head) > len(suf) + 2:
            sigs.append(head[: len(head) - len(suf)] if suf else head)
    return sorted(set(s for s in sigs if s))


def check_text(text, facts):
    """Jeder Fakt gedeckt, kein Kausalsatz ungedeckt. Rückgabe:
    (bestanden, fehlende_fakten, ungedeckte_saetze).

    Pronomen-Auflösung mit der GETEILTEN Regel des Diskurs-Ohrs: ein
    Satz, der mit 'It (also)' beginnt, erbt das zuletzt explizit
    genannte Fakten-Subjekt — geprüft wird der aufgelöste Satz, aber
    ausgeliefert wird exakt der Text, der geprüft wurde (nie zwei
    Versionen)."""
    raw_sents = [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]
    subj_by_len = sorted({f["subj"] for f in facts}, key=len, reverse=True)
    sents, prev_subj = [], None
    for s in raw_sents:
        low = s.lower()
        if prev_subj and (low.startswith("it also ") or low.startswith("it ")):
            rest = s.split(None, 2)[2] if low.startswith("it also ") \
                else s.split(None, 1)[1]
            s = f"{prev_subj} {rest}"
        sn = _norm(s)
        # Subjekt des Satzes = das am SATZANFANG früheste Fakten-Subjekt
        # (Position, dann Länge) — nicht irgendein enthaltenes Entity;
        # 'Tar buildup causes lung damage' hat das Subjekt tar buildup,
        # auch wenn lung damage anderswo Subjekt ist.
        best_pos, best_cand = None, None
        for cand in subj_by_len:
            pos = sn.find(f" {_norm(cand).strip()} ")
            if pos >= 0 and (best_pos is None or pos < best_pos):
                best_pos, best_cand = pos, cand
        if best_cand is not None:
            prev_subj = best_cand
        sents.append(s)
    sents_n = [_norm(s) for s in sents]

    def digits_ok(sent_raw, phrase):
        toks = [t for t in phrase.lower().split()
                if any(ch.isdigit() for ch in t)]
        return all(t in sent_raw.lower().split() or t in sent_raw.lower()
                   for t in toks)

    missing = []
    for f in facts:
        subj_n, obj_n = _norm(f["subj"]).strip(), _norm(f["obj"]).strip()
        covered = False
        for raw, sn in zip(sents, sents_n):
            if (f" {subj_n} " in sn and f" {obj_n} " in sn
                    and any(sig in sn for sig in
                            (" " + s for s in _signals(f["verb"])))
                    and digits_ok(raw, f["obj"])):
                covered = True
                break
        if not covered:
            missing.append(f"{f['subj']} -[{f['verb']}]-> {f['obj']}")

    # Kein Kausalsatz ohne Deckung: ein Satz, der ein bekanntes Verb-Signal
    # trägt, muss mindestens ein bekanntes Faktenpaar (beide Entitäten)
    # enthalten.
    all_sigs = sorted({s for f in facts for s in _signals(f["verb"])})
    unbacked = []
    for raw, sn in zip(sents, sents_n):
        if not any(" " + s in sn for s in all_sigs):
            continue                        # kein Kausalsatz
        ok = any(f" {_norm(f['subj']).strip()} " in sn
                 and f" {_norm(f['obj']).strip()} " in sn for f in facts)
        if not ok:
            unbacked.append(raw)
    return (not missing and not unbacked), missing, unbacked


def facts_health():
    vocab, stoi, adj, mech = load_fertig_graph(GRAPH)
    return [{"subj": vocab[a], "verb": mech[(a, b)], "obj": vocab[b]}
            for a, nb in sorted(adj.items()) for b in sorted(nb)]


def facts_weltbuch():
    agg, n_total = aggregate_records()
    wv, ws, wa, wm, _ = build_weltbuch(agg, n_total)
    return [{"subj": wv[i], "verb": wm[(i, j)], "obj": wv[j]}
            for i, nb in sorted(wa.items()) for j in sorted(nb)]


# ═══════════════════════════════════════════════════════════════════════════
#  Die Demo-Oberflächen (externer Schreiber; unvertraut, deshalb geprüft)
# ═══════════════════════════════════════════════════════════════════════════
SURFACE_HEALTH = """\
The story begins with smoking. Smoking causes tar buildup, and smoking also \
indirectly leads to lung damage on its own. The tar buildup causes lung \
damage of its own; over time, tar buildup indirectly leads to breathlessness, \
and in the worst case tar buildup indirectly leads to cancer. Once lung \
damage sets in, that lung damage causes breathlessness directly, and lung \
damage leads to cancer as well. Breathlessness then closes the trap: \
breathlessness reduces exercise, breathlessness indirectly inhibits via \
blocking promoter health, and breathlessness indirectly promotes via \
inhibition heart disease. Yet exercise is exactly the thing worth keeping: \
exercise improves health and exercise prevents heart disease. Two quieter \
threads run alongside. Caffeine prevents sleep; poor sleep causes stress; \
and stress damages health in the end."""

SURFACE_WELTBUCH = """\
The organism has learned what its own hands do. Pressing the left key \
shifts the view 1 pixel to the left when the step is small; with more \
momentum, pressing the left key shifts the view 2 pixels to the left, and \
at full stride pressing the left key shifts the view 3 pixels to the left. \
The mirror image holds on the other side: pressing the right key shifts \
the view 1 pixel to the right, pressing the right key shifts the view \
2 pixels to the right, and pressing the right key shifts the view 3 pixels \
to the right. Every one of these sentences was lived hundreds of times \
before it was written down."""


def main(out_path=None):
    results = {}
    for name, text, facts in (("health", SURFACE_HEALTH, facts_health()),
                              ("weltbuch", SURFACE_WELTBUCH, facts_weltbuch())):
        ok, missing, unbacked = check_text(text, facts)
        results[name] = {"released": bool(ok), "n_facts": len(facts),
                         "missing": missing, "unbacked": unbacked,
                         "text": text if ok else None}
        print(f"\n════ {'AUSGELIEFERT' if ok else 'ZURÜCKGEHALTEN'}: "
              f"'{name}' ({len(facts)} Fakten"
              f"{'' if ok else f', fehlend {len(missing)}, ungedeckt {len(unbacked)}'}) ════",
              flush=True)
        if ok:
            print(text, flush=True)
        else:
            for m in missing:
                print(f"  fehlt: {m}", flush=True)
            for u in unbacked:
                print(f"  ungedeckt: {u}", flush=True)
    out_path = out_path or os.path.join(HERE, "results", "oberflaeche.json")
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    return results


if __name__ == "__main__":
    main()
