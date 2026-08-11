#!/usr/bin/env python3 -u
"""
DAS WELTBUCH GESPROCHEN -- FERTIG spricht gelebtes, bit-belegbares Wissen.

FERTIGs Identität: Fakten kommen exakt aus dem Graphen, nie erfunden;
der Walk darüber ist nur Form. Bisher stammen die Graphen aus Text-
Wissensbasen. Dieses Modul gibt FERTIG einen Graphen, dessen Kanten
GELEBT wurden: die acted-Records des o1-state-Körpers ("pressed:pan_left
| view_shift | dx-3"), als Daten-Snapshot in data/weltbuch/ (KEIN Import
aus dem o1-state-Repo -- FERTIG bleibt eigenständig; Replay läuft über
den eingefrorenen Vendor-Snapshot _vendor_o1welt.py, Commit-Pin b575b3f).

Der Anspruch "MASSIVST perfekt sprechen" wird hier wörtlich und
falsifizierbar: perfekt heißt in diesem Haus nicht flauschig, sondern
JEDER SATZ TRÄGT QUITTUNG -- Zählung der gelebten Ereignisse plus eine
konkrete Frame-Koordinate mit SHA-256, und eine Stichprobe der Quittungen
wird durch die Welt BIT-EXAKT nachgespielt, bevor die Prosa als
FREIGEGEBEN gilt (FERTIGs eigenes Muster: nie "erst generieren, dann
hoffen").

Ablauf:
  1. Records aggregieren: (Aktion, dx) -> sprechbares Triplet
     (trigger "pressing the left key", mechanism "shifts the view",
     outcome "N pixel to the left/right"), confidence aus der Häufigkeit,
     Belege = Frame-Koordinaten + Hashes.
  2. Graph in FERTIGs Form (vocab/stoi/adj/mech) bauen; Sätze über
     fertig.pipeline.verbalize-kompatible Hops erzeugen, je Satz die
     Quittung angehängt.
  3. Provenance-Gate: 5 gesampelte Belege via RichPanWorld-Replay --
     beide Frame-Hashes exakt UND dx re-detektiert, sonst VERWORFEN.

Register: erweiterung/PREDICTIONS_ERWEITERUNG.md FE2, VOR dem Scoring.
"""

import json
import os
import sys
from collections import defaultdict

import numpy as np

FERTIG_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if FERTIG_ROOT not in sys.path:
    sys.path.insert(0, FERTIG_ROOT)

HERE = os.path.dirname(os.path.abspath(__file__))
from _vendor_o1welt import RichPanWorld, frame_sha, estimate_shift  # noqa: E402

DATA = os.path.join(FERTIG_ROOT, "data", "weltbuch")
ACTION_PHRASE = {"pan_left": "pressing the left key",
                 "pan_right": "pressing the right key"}


def speakable(action_name: str, dx: int):
    """Record-Vokabular -> sprechbares Triplet. Vorzeichenkonvention der
    Records: dx > 0 heißt, der Blick wanderte nach rechts."""
    direction = "right" if dx > 0 else "left"
    n = abs(dx)
    outcome = f"the view {n} pixel{'s' if n != 1 else ''} to the {direction}"
    return ACTION_PHRASE[action_name], "shifts", outcome


def aggregate_records(records_path=None):
    records_path = records_path or os.path.join(DATA, "a_records.jsonl")
    agg = defaultdict(lambda: {"count": 0, "quotes": []})
    n_total = 0
    for line in open(records_path):
        r = json.loads(line)
        q = r["quote"]
        key = (q["action_name"], int(q["dx_measured"]))
        a = agg[key]
        a["count"] += 1
        n_total += 1
        if len(a["quotes"]) < 50:
            a["quotes"].append({"frame": q["frame"],
                                "base_seed": q["base_seed"],
                                "action": q["action"],
                                "dx": int(q["dx_measured"]),
                                "sha_pre": q["frame_sha256_pre"],
                                "sha_post": q["frame_sha256_post"]})
    return agg, n_total


def build_graph(agg, n_total):
    """FERTIG-förmiger Graph (vocab/stoi/adj/mech) aus den aggregierten
    gelebten Kanten -- Aufbauschleife wie pipeline.load_graph, re-stated."""
    vocab, stoi, adj, mech, evidence = [], {}, {}, {}, {}

    def sym(s):
        if s not in stoi:
            stoi[s] = len(vocab)
            vocab.append(s)
        return stoi[s]

    for (action, dx), a in sorted(agg.items()):
        trig, m, out = speakable(action, dx)
        i, j = sym(trig), sym(out)
        conf = min(0.95, a["count"] / max(1, n_total * 0.05))
        adj.setdefault(i, {})[j] = max(adj.get(i, {}).get(j, 0), round(conf, 3))
        mech[(i, j)] = m
        evidence[(i, j)] = a
    return vocab, stoi, adj, mech, evidence


def speak_weltbuch(vocab, adj, mech, evidence):
    """Jede gelebte Kante als Satz mit Quittung. Harte Regeln: nur Kanten
    aus adj (per Konstruktion), und KEIN Satz ohne mindestens einen
    konkreten Beleg -- ein belegloser Satz wird VERWORFEN, nicht poliert."""
    sentences, discarded = [], 0
    for i in sorted(adj):
        for j in sorted(adj[i]):
            ev = evidence.get((i, j))
            if not ev or not ev["quotes"]:
                discarded += 1
                continue
            q0 = ev["quotes"][0]
            text = (f"{vocab[i].capitalize()} {mech[(i, j)]} {vocab[j]}.")
            receipt = (f"[gelebt {ev['count']}x; z.B. frame {q0['frame']}, "
                       f"sha {q0['sha_pre'][:10]}…]")
            sentences.append({"text": text, "receipt": receipt,
                              "edge": [vocab[i], vocab[j]],
                              "count": ev["count"]})
    return sentences, discarded


def verify_receipts(agg, trace_path=None, n_samples=5, seed=7):
    """Provenance-Gate: gesampelte Belege durch die Vendor-Welt
    nachspielen. Bit-exakt (beide Hashes) UND dx re-detektiert."""
    trace_path = trace_path or os.path.join(DATA, "a_life.json")
    trace = json.load(open(trace_path))["trace"]
    quotes = [q for a in agg.values() for q in a["quotes"]]
    rng = np.random.default_rng(seed)
    picks = [quotes[int(i)] for i in
             rng.choice(len(quotes), size=min(n_samples, len(quotes)),
                        replace=False)]
    frames_cache = {}
    results = []
    for q in picks:
        key = q["base_seed"]
        if key not in frames_cache:
            frames_cache[key] = RichPanWorld.replay_frames(key, trace)
        fs = frames_cache[key]
        f0, f1 = fs[q["frame"]], fs[q["frame"] + 1]
        sha_ok = (frame_sha(f0) == q["sha_pre"]
                  and frame_sha(f1) == q["sha_post"])
        s, _, _ = estimate_shift(f0, f1)
        results.append({"frame": q["frame"], "exact": bool(sha_ok),
                        "dx_redetected": bool(s == q["dx"])})
        print(f"[weltbuch] Beleg frame {q['frame']} dx {q['dx']:+d} -> "
              f"{'BIT-EXAKT' if sha_ok else 'HASH-FEHLER'}"
              f"{'' if s == q['dx'] else ' (dx abweichend!)'}", flush=True)
    n_ok = sum(1 for r in results if r["exact"] and r["dx_redetected"])
    return {"picks": results, "exact": f"{n_ok}/{len(results)}",
            "pass": bool(results) and n_ok == len(results)}


def main(out_path=None):
    agg, n_total = aggregate_records()
    vocab, stoi, adj, mech, evidence = build_graph(agg, n_total)
    sentences, discarded = speak_weltbuch(vocab, adj, mech, evidence)
    prov = verify_receipts(agg)
    status = "FREIGEGEBEN" if prov["pass"] else "VERWORFEN"
    print(f"\n════ DAS WELTBUCH, GESPROCHEN ({status}) ════", flush=True)
    for s in sentences:
        print(f"  {s['text']}  {s['receipt']}", flush=True)
    out = {"n_records": n_total, "n_edges": len(sentences),
           "edges": sentences, "n_discarded_unbacked": discarded,
           "provenance": prov, "status": status,
           "note": "Belege via Vendor-Snapshot _vendor_o1welt.py "
                   "(o1-state Commit b575b3f) nachgespielt"}
    out_path = out_path or os.path.join(HERE, "results", "weltbuch.json")
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f"[weltbuch] {len(sentences)} Sätze, {n_total} Records, "
          f"Provenance {prov['exact']} -> {out_path}", flush=True)
    return out


if __name__ == "__main__":
    main()
