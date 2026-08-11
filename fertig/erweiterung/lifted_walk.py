#!/usr/bin/env python3 -u
"""
DER GELÜFTETE KAUSAL-WALK -- FERTIGs Spaziergang bekommt Impuls.

FERTIGs Generierung IST ein Walk über den .causal-Graphen
(fertig/pipeline.py: walk_chain -- konfidenz-gewichtetes Sampling über
Out-Kanten, Berry-Phasen-Wächter stoppt beim Kreisen). Ein memoryloser
Walk auf kleinen Graphen kreist aber FRÜH -- und FERTIGs ehrliche Antwort
darauf ist Verstummen (bphm: "wir kreisen -- stoppen"). Der Mund
verstummt also nicht, weil das Wissen endet, sondern weil der Spaziergang
sich verläuft.

Hier greift die in dieser Session unabhängig replizierte Formel
(PS-Lifted, Foss 2026; Replikation: o1-state dynamics/pslifted.py --
Karate 12.2 Runden vs. Paper 12, konstant über 40x n): NICHT-REVERSIBLER
FLUSS AUF VERDOPPELTEM ZUSTAND. Der Walk-Zustand wird (Knoten, Richtung):
die Fiedler-Koordinate des ungerichteten Skeletts orientiert den Graphen,
und der Walk behält mit p_continue seine Fluss-Richtung (bergauf bzw.
bergab in der Fiedler-Koordinate) statt zu diffundieren -- Momentum durch
die Engstellen, strukturell weniger Rückkehr, längere Ketten bevor der
Berry-Wächter abbricht.

WAS SICH NICHT ÄNDERT (FERTIGs Garantien): gelaufen werden AUSSCHLIESSLICH
existierende gerichtete Kausal-Kanten (der Lift wählt nur ANDERS unter den
echten Out-Kanten -- nie erfundene Kausalität, hart assertiert);
Konfidenz-Gewichtung bleibt (Sampling innerhalb der Fluss-Gruppe ∝ zur
Kanten-Konfidenz bei gleicher tau-Temperatur); derselbe Berry-Wächter
richtet über beide Arme; Determinismus bleibt (seeded Generator, gleiche
Eingabe -> gleiche Ausgabe).

Fairness des Vergleichs: der Baseline-Arm nutzt hier DENSELBEN lokalen
Sampler (konfidenz-gewichtet, gleiche tau-Temperatur, eigener Generator)
ohne Schichten -- die EINZIGE Differenz ist der Lift. (FERTIGs
Original-walk_chain sampelt über den globalen NumPy-RNG; für einen
gepaarten Test braucht es lokale Generatoren. Die Original-Funktion
bleibt unberührt.)

Register: erweiterung/PREDICTIONS_ERWEITERUNG.md FE1, VOR dem Scoring.
"""

import json
import os
import sys

import numpy as np

FERTIG_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if FERTIG_ROOT not in sys.path:
    sys.path.insert(0, FERTIG_ROOT)

from fertig import bphm, state_init                       # read-only
from fertig.sampler import tau_to_temperature             # read-only
from fertig._vendor.dotcausal.io import CausalReader      # read-only

PC = 0.95            # p_continue -- der P95-validierte Wert
MAX_HOPS = 40
TAU = 0.3            # FERTIGs graph-Modus-Default


def load_fertig_graph(path):
    """pipeline.load_graph's Aufbau, re-stated (8 Zeilen), mit
    verify_integrity=False: die gelieferte chained.causal trägt einen
    CRC, den der Vendor-Reader nicht reproduziert (stored 8cc5c4e8...,
    computed 03e234c4...) -- Inhalte sind plausibel (17 Triplets,
    korrekte Felder); als Bug an das FERTIG-Team gemeldet, hier bewusst
    laut dokumentiert statt still geschluckt."""
    from fertig.pipeline import _toks
    trips = CausalReader(str(path), verify_integrity=False).get_all_triplets()
    vocab, stoi, adj, mech = [], {}, {}, {}

    def sym(s):
        s = " ".join(_toks(s))
        if not s:
            return None
        if s not in stoi:
            stoi[s] = len(vocab)
            vocab.append(s)
        return stoi[s]

    for t in trips:
        a, b = sym(t.get("trigger", "")), sym(t.get("outcome", ""))
        if a is None or b is None:
            continue
        c = float(t.get("confidence", 0.5) or 0.5)
        adj.setdefault(a, {})[b] = max(adj.get(a, {}).get(b, 0), c)
        mech[(a, b)] = " ".join(_toks(t.get("mechanism", ""))) or "leads to"
    return vocab, stoi, adj, mech


def fiedler_coordinate(n: int, adj) -> np.ndarray:
    """v2 des ungerichteten Skeletts (dichte eigh -- FERTIG-Graphen sind
    klein; numpy-only, keine neue Abhängigkeit)."""
    A = np.zeros((n, n))
    for a, nbrs in adj.items():
        for b in nbrs:
            A[a, b] = 1.0
            A[b, a] = 1.0
    L = np.diag(A.sum(1)) - A
    vals, vecs = np.linalg.eigh(L)
    order = np.argsort(vals)
    return vecs[:, order[1]] if n >= 2 else np.zeros(n)


def _pick(group, adj_row, rng, tau=TAU):
    """Konfidenz-gewichtete Wahl innerhalb einer Kanten-Gruppe, bei
    FERTIGs tau-Temperatur (dieselbe Lorentz-Formel wie contraction_sample),
    aber über einen LOKALEN Generator."""
    T = max(tau_to_temperature(tau), 1e-9)
    logits = np.array([np.log(adj_row[b] + 1e-9) / T for b in group])
    p = np.exp(logits - logits.max())
    p /= p.sum()
    return int(rng.choice(group, p=p))


def baseline_chain(start, vocab, stoi, adj, SM, rng, max_hops=MAX_HOPS):
    """Reversibler Referenz-Walk: gleicher Sampler, gleicher Berry-
    Wächter, keine Schichten."""
    cur = stoi.get(start)
    if cur is None:
        return []
    hops, hist = [], []
    for _ in range(max_hops):
        nbrs = adj.get(cur, {})
        if not nbrs:
            break
        nxt = _pick(sorted(nbrs), nbrs, rng)
        hist.append(state_init.state_for_symbol(nxt, SM))
        if len(hist) >= 5 and bphm.detect_repetition(hist[-6:]):
            break
        hops.append((cur, nxt))
        cur = nxt
    return hops


def lifted_chain(start, vocab, stoi, adj, SM, v2, rng, pc=PC,
                 max_hops=MAX_HOPS):
    """Der gelüftete Walk: Zustand (Knoten, Richtung). Bergauf-Gruppe
    F = Out-Kanten mit steigender Fiedler-Koordinate, B = Rest; in
    Richtung + wird mit pc aus F (Richtung bleibt), sonst aus B
    (Richtung kippt) gewählt -- symmetrisch für −. Leere Gruppe: Impuls
    reflektiert (P95-Randregel). Jede gelaufene Kante MUSS in adj stehen."""
    cur = stoi.get(start)
    if cur is None:
        return []
    layer = 1 if rng.random() < 0.5 else -1
    hops, hist = [], []
    for _ in range(max_hops):
        nbrs = adj.get(cur, {})
        if not nbrs:
            break
        F = sorted(b for b in nbrs if (v2[b], b) > (v2[cur], cur))
        B = sorted(b for b in nbrs if (v2[b], b) <= (v2[cur], cur))
        with_flow, against = (F, B) if layer > 0 else (B, F)
        if with_flow and (not against or rng.random() < pc):
            group = with_flow
        else:
            group, layer = against, -layer
        nxt = _pick(group, nbrs, rng)
        assert nxt in adj.get(cur, {}), "erfundene Kante -- verboten"
        hist.append(state_init.state_for_symbol(nxt, SM))
        if len(hist) >= 5 and bphm.detect_repetition(hist[-6:]):
            break
        hops.append((cur, nxt))
        cur = nxt
    return hops


def benchmark(graph_path=None, seeds=range(5), out_path=None):
    graph_path = graph_path or os.path.join(FERTIG_ROOT, "data", "chained.causal")
    vocab, stoi, adj, mech = load_fertig_graph(graph_path)
    n = len(vocab)
    SM = state_init.initialize_symbol_state(n)
    v2 = fiedler_coordinate(n, adj)
    starts = sorted(vocab[a] for a in adj if adj[a])
    res = {"baseline": {"hops": [], "edges": set(), "aborts": 0, "walks": 0},
           "lifted": {"hops": [], "edges": set(), "aborts": 0, "walks": 0}}
    determinism_ok = True
    for seed in seeds:
        for si, start in enumerate(starts):
            for arm, fn, extra in (("baseline", baseline_chain, ()),
                                   ("lifted", lifted_chain, (v2,))):
                rng = np.random.default_rng(10_000 * seed + si)
                hops = fn(start, vocab, stoi, adj, SM, *extra, rng)
                rng2 = np.random.default_rng(10_000 * seed + si)
                hops2 = fn(start, vocab, stoi, adj, SM, *extra, rng2)
                determinism_ok &= (hops == hops2)
                r = res[arm]
                r["walks"] += 1
                r["hops"].append(len(hops))
                r["edges"].update(hops)
                if len(hops) < MAX_HOPS and hops and adj.get(hops[-1][1]):
                    r["aborts"] += 1     # Stopp trotz vorhandener Kanten = Berry-Abbruch
    n_edges_total = sum(len(v) for v in adj.values())
    out = {"graph": os.path.basename(str(graph_path)), "n_entities": n,
           "n_edges": n_edges_total, "n_starts": len(starts),
           "n_seeds": len(list(seeds)), "pc": PC, "tau": TAU,
           "determinism_ok": bool(determinism_ok),
           "crc_note": "chained.causal mit verify_integrity=False geladen "
                       "(CRC-Mismatch, an FERTIG-Team gemeldet)"}
    for arm in ("baseline", "lifted"):
        r = res[arm]
        out[arm] = {"mean_hops": round(float(np.mean(r["hops"])), 3),
                    "median_hops": float(np.median(r["hops"])),
                    "edge_coverage": round(len(r["edges"]) / n_edges_total, 4),
                    "abort_rate": round(r["aborts"] / r["walks"], 4)}
    out["hops_ratio"] = round(out["lifted"]["mean_hops"]
                              / max(out["baseline"]["mean_hops"], 1e-9), 3)
    out["coverage_ratio"] = round(out["lifted"]["edge_coverage"]
                                  / max(out["baseline"]["edge_coverage"], 1e-9), 3)
    if out_path:
        os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(out, f, indent=2)
    print(f"[lifted_walk] hops {out['baseline']['mean_hops']} -> "
          f"{out['lifted']['mean_hops']} (x{out['hops_ratio']}) | coverage "
          f"{out['baseline']['edge_coverage']} -> {out['lifted']['edge_coverage']} "
          f"(x{out['coverage_ratio']}) | aborts {out['baseline']['abort_rate']} -> "
          f"{out['lifted']['abort_rate']} | determinism {out['determinism_ok']}",
          flush=True)
    return out


if __name__ == "__main__":
    here = os.path.dirname(os.path.abspath(__file__))
    benchmark(out_path=os.path.join(here, "results", "lifted_walk.json"))
