#!/usr/bin/env python3 -u
"""
FORM-ARENA + UID-REGEL + OHR-RICHTER -- "perfekt klingen" wird messbar,
und dann wird danach ausgewählt. Gewichtfrei, deterministisch, FERTIG-pur.

Die drei Stücke (Register FE3-FE5 in PREDICTIONS_ERWEITERUNG.md):

  FORM-ARENA (FE3): vier Messgrößen pro Satz, alle aus Bestands-Organen:
    fluency   mittlere Log-Wahrscheinlichkeit unter fertig.bench.TrigramLM,
              gebaut aus FERTIGs Korpus-Kanon data/faraday_candle.txt --
              Form aus Korpus GEMESSEN, kein Training.
    uid_var   Varianz der Wort-Surprisal (Uniform Information Density:
              gute Sätze verteilen Überraschung gleichmäßig -- das
              psycholinguistische Prinzip als Zahl; per-Wort-Surprisal
              re-derived aus den TrigramLM-Tabellen, identische
              Interpolations-Mathematik wie _sum_logprob).
    ir        fertig.utterance._verify_utterance -- Subjekt+Objekt
              wörtlich + Verb-Signal: der Wahrheits-Anker.
    ohr       fertig.semantic.parse_semantic muss BEIDE Kanten-Entitäten
              aus dem Satz zurückgewinnen -- das eigene Ohr als Richter
              (strenger als IR-Kanal 3, der nur das Objekt verlangt).

  UID-REGEL (FE4): pro Plan-Kante werden deterministische Formvarianten
  erzeugt (alle tragen Subjekt/Objekt wörtlich und ein Verb-Signal --
  Wahrheit ist Konstruktionsbedingung, nicht Hoffnung); Auswahl-Kaskade:
  IR-Gate (hart) -> Ohr-Gate (hart) -> minimale uid_var -> Tie-Break
  maximale fluency. Deterministisch, gleiche Eingabe -> gleiche Wahl.

  OHR-RICHTER (FE5): das Ohr-Gate muss nachweislich RICHTEN (mindestens
  eine Variante irgendwo verwerfen) -- ein Richter, der nie pfeift, ist
  Dekoration, und genau das wäre der Falsifier.

Ehrliche Messlücke, benannt: TrigramLM._ids überspringt OOV-Wörter --
Wörter außerhalb des Faraday-Vokabulars sind für das Metrum unsichtbar
(sie stehen im Satz, zählen aber nicht in fluency/uid_var). Varianten
nutzen deshalb bewusst kleines, korpus-nahes Vokabular.

Suite: alle Kanten des Health-Graphen (data/chained.causal) PLUS die
sechs gelebten Weltbuch-Kanten -- die Arena benotet also auch die Sätze
mit Quittung.
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

from fertig.bench import TrigramLM                     # read-only
from fertig.pipeline import _toks, _det                # read-only
from fertig.utterance import Utterance, _verify_utterance, _VERB_SIGNALS  # read-only

from lifted_walk import load_fertig_graph
from weltbuch import aggregate_records, build_graph as build_weltbuch

CORPUS = os.path.join(FERTIG_ROOT, "data", "faraday_candle.txt")
GRAPH = os.path.join(FERTIG_ROOT, "data", "chained.causal")


# ── Metrum ────────────────────────────────────────────────────────────────
def per_word_surprisals(lm: TrigramLM, sentence: str):
    """-log P(w|h) pro Wort, identische Interpolation wie
    TrigramLM._sum_logprob (0.6·P3 + 0.3·P2 + 0.1·P1, add-one)."""
    ids = lm._ids(_toks(sentence))
    out = []
    for i, w in enumerate(ids):
        if i >= 2:
            d = lm.trigram.get((ids[i - 2], ids[i - 1]))
            if d:
                p3 = (d.get(w, 0) + 1.0) / (sum(d.values()) + lm.V)
                b = lm.bigram.get(ids[i - 1], {})
                p2 = (b.get(w, 0) + 1.0) / (sum(b.values()) + lm.V) if b else 1.0 / lm.V
                p1 = (lm.unigram[w] + 1.0) / (lm.unigram.sum() + lm.V)
                p = 0.6 * p3 + 0.3 * p2 + 0.1 * p1
            else:
                p = 1e-9
        elif i == 1:
            b = lm.bigram.get(ids[0], {})
            p = (b.get(w, 0) + 1.0) / (sum(b.values()) + lm.V) if b else 1.0 / lm.V
        else:
            p = (lm.unigram[w] + 1.0) / (lm.unigram.sum() + lm.V)
        out.append(-float(np.log(max(p, 1e-12))))
    return out


def edge_ear(prose: str, subj: str, obj: str, verb: str) -> bool:
    """Der Ohr-Richter: ein absichtlich EINFACHER, deterministischer
    Hörer mit geteiltem Lexikon (Naming-Game-Bedingung), der die KANTE
    samt RICHTUNG aus der Oberfläche zurückgewinnen muss: erste Entität
    vor dem Verb-Signal = Subjekt, erste danach = Objekt. Formen, die
    die kausale Richtung an der Oberfläche verdrehen (z.B. Frontierung),
    fallen bei diesem Hörer durch -- genau dafür ist er Richter.

    (Ursprünglich war fertig.semantic.parse_semantic als Ohr registriert
    gedacht; der Debug fand: das ist ein Textaufgaben-Parser
    (possess/sold/half-Muster), der auf Kausalprosa leere Graphen liefert
    -- dokumentiert, Hörer hier neu und offen definiert.)"""
    s = " " + " ".join(_toks(prose)) + " "
    sig = None
    for cand in _VERB_SIGNALS.get(verb, [verb.replace("_", " ")]) + [verb]:
        pos = s.find(" " + cand.split()[0])
        if pos >= 0:
            sig = pos
            break
    if sig is None:
        return False
    subj_n = " ".join(_toks(subj))       # dieselbe Normalisierung wie der
    obj_n = " ".join(_toks(obj))         # Satz selbst (Debug-Fund Nr. 3:
    ps = s.find(" " + subj_n + " ")      # _toks normalisiert Ziffern anders
    po = s.find(" " + obj_n + " ")       # als der Roh-String)
    if ps < 0 or po < 0:
        return False
    return ps < sig < po


def score_sentence(lm, subj, verb, obj, prose, graph_path):
    surps = per_word_surprisals(lm, prose)
    u = Utterance(subj, verb, obj, 0.9, prose=prose)
    ir_ok, ir_detail = _verify_utterance(u, graph_path)
    return {"fluency": round(float(np.mean([-s for s in surps])), 4)
            if surps else float("-inf"),
            "uid_var": round(float(np.var(surps)), 4) if surps else float("inf"),
            "n_scored_words": len(surps),
            "ir": bool(ir_ok), "ohr": edge_ear(prose, subj, obj, verb)}


# ── Varianten (Wahrheit als Konstruktionsbedingung) ───────────────────────
def variants(subj, verb, obj):
    """Deterministische Formvarianten. Jede trägt Subjekt und Objekt
    wörtlich und das unveränderte Verb -- das IR-Gate ist erfüllbar, aber
    nicht garantiert (Artikel/Umstellungen können das Ohr verwirren, und
    genau das soll der Richter finden)."""
    def det(phrase):
        # Weltbuch-Entitäten tragen ihre Artikel/Gerundien schon selbst
        # ("pressing the left key", "the view 2 pixels ...") -- doppelte
        # Artikel waren der Debug-Fund Nr. 2.
        if phrase.split()[0] in ("the", "a", "an", "pressing"):
            return phrase
        return _det(phrase)

    S, O = det(subj), det(obj)
    v = verb
    return [
        ("default", f"{S.capitalize()} {v} {O}."),
        ("cleft", f"It is {S} that {v} {O}."),
        ("fronted", f"{O.capitalize()}: that is what {S} {v}."),
        ("plain_the",
         f"{('The ' + subj) if subj.split()[0] not in ('the', 'a', 'an', 'pressing') else subj.capitalize()} "
         f"{v} "
         f"{('the ' + obj) if obj.split()[0] not in ('the', 'a', 'an', 'pressing') else obj}."),
        ("consider", f"Consider {S}. It {v} {O}."),
        ("chain", f"{S.capitalize()} {v} {O} in every measured case."),
    ]


def select(lm, subj, verb, obj, graph_path):
    """Die Kaskade: IR-Gate -> Ohr-Gate -> min uid_var -> max fluency."""
    scored = []
    for name, prose in variants(subj, verb, obj):
        m = score_sentence(lm, subj, verb, obj, prose, graph_path)
        scored.append({"name": name, "prose": prose, **m})
    survivors = [s for s in scored if s["ir"]]
    killed_ir = len(scored) - len(survivors)
    after_ohr = [s for s in survivors if s["ohr"]]
    killed_ohr = len(survivors) - len(after_ohr)
    pool = after_ohr or survivors or scored     # ehrliche Degradation, geloggt
    best = sorted(pool, key=lambda s: (s["uid_var"], -s["fluency"]))[0]
    return best, scored, {"killed_ir": killed_ir, "killed_ohr": killed_ohr,
                          "degraded": not after_ohr}


# ── Suite ─────────────────────────────────────────────────────────────────
def edge_suite():
    """(subj, verb, obj, quelle) für Health-Graph + Weltbuch."""
    edges = []
    vocab, stoi, adj, mech = load_fertig_graph(GRAPH)
    for a, nbrs in sorted(adj.items()):
        for b in sorted(nbrs):
            edges.append((vocab[a], mech[(a, b)], vocab[b], "health"))
    agg, n_total = aggregate_records()
    wv, ws, wa, wm, _ = build_weltbuch(agg, n_total)
    for i, nbrs in sorted(wa.items()):
        for j in sorted(nbrs):
            edges.append((wv[i], wm[(i, j)], wv[j], "weltbuch"))
    return edges


def main(out_path=None):
    lm = TrigramLM(open(CORPUS, encoding="utf-8", errors="ignore").read())
    edges = edge_suite()
    rows, kills = [], {"killed_ir": 0, "killed_ohr": 0, "degraded": 0}
    for subj, verb, obj, src in edges:
        best, scored, k = select(lm, subj, verb, obj, GRAPH)
        default = next(s for s in scored if s["name"] == "default")
        for key in ("killed_ir", "killed_ohr"):
            kills[key] += k[key]
        kills["degraded"] += int(k["degraded"])
        rows.append({"edge": f"{subj} -[{verb}]-> {obj}", "source": src,
                     "default": default, "selected": best})
    n = len(rows)
    uid_improved = sum(1 for r in rows
                       if r["selected"]["uid_var"] <= r["default"]["uid_var"])
    flu_def = float(np.mean([r["default"]["fluency"] for r in rows]))
    flu_sel = float(np.mean([r["selected"]["fluency"] for r in rows]))
    out = {"n_edges": n, "corpus": os.path.basename(CORPUS),
           "uid_improved_or_equal": uid_improved,
           "uid_improved_frac": round(uid_improved / n, 4),
           "fluency_default_mean": round(flu_def, 4),
           "fluency_selected_mean": round(flu_sel, 4),
           "fluency_rel_change": round((flu_sel - flu_def) / abs(flu_def), 4),
           "ir_selected_pass": sum(1 for r in rows if r["selected"]["ir"]),
           "ohr_selected_pass": sum(1 for r in rows if r["selected"]["ohr"]),
           "gate_kills": kills,
           "selected_variant_histogram": {},
           "rows": rows}
    hist = {}
    for r in rows:
        hist[r["selected"]["name"]] = hist.get(r["selected"]["name"], 0) + 1
    out["selected_variant_histogram"] = hist
    out_path = out_path or os.path.join(HERE, "results", "form_arena.json")
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f"[form_arena] {n} Kanten | UID besser/gleich {uid_improved}/{n} | "
          f"fluency {flu_def:.3f} -> {flu_sel:.3f} "
          f"({out['fluency_rel_change']:+.1%}) | IR {out['ir_selected_pass']}/{n} "
          f"| Ohr {out['ohr_selected_pass']}/{n} | kills {kills} | "
          f"Wahl {hist}", flush=True)
    for r in rows[:3] + rows[-2:]:
        print(f"  [{r['source']}] {r['default']['prose'] if 'prose' in r['default'] else ''}"
              f"{r['selected']['name']}: {r['selected']['prose']}", flush=True)
    return out


if __name__ == "__main__":
    main()
