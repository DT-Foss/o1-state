"""
ERWEITERUNG smoke tests -- pytest oder direkt.

  1. Graph lädt (17 Triplets, CRC-Umgehung dokumentiert), Fiedler-
     Koordinate hat n Einträge und trennt (nicht konstant).
  2. Beide Walker laufen NUR echte Kanten (hart geprüft gegen adj) und
     sind deterministisch (gleicher Seed -> identische Hop-Folge).
  3. Der Lift kehrt seltener sofort um: Anteil unmittelbarer
     Rückschritte (a->b->a) über viele Walks kleiner als Baseline --
     die strukturelle Signatur des Impulses, kein Register-Bar.
  4. speakable-Codec: Vorzeichen und Pluralformen exakt.
  5. Weltbuch-Aggregation: Zählungen summieren zu n_total, jede Kante
     trägt Belege.
  6. Vendor-Welt-Determinismus: gleiche (seed, actions) -> byte-
     identische Frames (die Grundlage des Quittungs-Replays).
"""

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
FERTIG_ROOT = os.path.dirname(HERE)
for p in (FERTIG_ROOT, HERE):
    if p not in sys.path:
        sys.path.insert(0, p)

import numpy as np

from lifted_walk import (load_fertig_graph, fiedler_coordinate,
                         baseline_chain, lifted_chain)
from weltbuch import speakable, aggregate_records, build_graph
from _vendor_o1welt import RichPanWorld
from fertig import state_init

GRAPH = os.path.join(FERTIG_ROOT, "data", "chained.causal")


def _setup():
    vocab, stoi, adj, mech = load_fertig_graph(GRAPH)
    SM = state_init.initialize_symbol_state(len(vocab))
    v2 = fiedler_coordinate(len(vocab), adj)
    return vocab, stoi, adj, mech, SM, v2


def test_graph_and_fiedler():
    vocab, stoi, adj, mech, SM, v2 = _setup()
    assert len(vocab) >= 10 and len(v2) == len(vocab)
    assert float(np.std(v2)) > 1e-6


def test_walkers_only_real_edges_and_deterministic():
    vocab, stoi, adj, mech, SM, v2 = _setup()
    start = vocab[sorted(adj)[0]]
    for fn, extra in ((baseline_chain, ()), (lifted_chain, (v2,))):
        h1 = fn(start, vocab, stoi, adj, SM, *extra,
                np.random.default_rng(3))
        h2 = fn(start, vocab, stoi, adj, SM, *extra,
                np.random.default_rng(3))
        assert h1 == h2
        for a, b in h1:
            assert b in adj.get(a, {}), "Kante nicht im Graphen!"


def test_lift_reduces_immediate_backtracking():
    vocab, stoi, adj, mech, SM, v2 = _setup()
    starts = [vocab[a] for a in sorted(adj) if adj[a]]

    def backtrack_rate(fn, extra):
        back = steps = 0
        for seed in range(6):
            for si, s in enumerate(starts):
                hops = fn(s, vocab, stoi, adj, SM, *extra,
                          np.random.default_rng(100 * seed + si))
                for (a1, b1), (a2, b2) in zip(hops, hops[1:]):
                    steps += 1
                    back += int(b2 == a1)
        return back / max(steps, 1)

    rb = backtrack_rate(baseline_chain, ())
    rl = backtrack_rate(lifted_chain, (v2,))
    assert rl <= rb + 1e-9, (rl, rb)


def test_speakable_codec():
    assert speakable("pan_left", -3) == (
        "pressing the left key", "shifts", "the view 3 pixels to the left")
    assert speakable("pan_right", 1) == (
        "pressing the right key", "shifts", "the view 1 pixel to the right")


def test_weltbuch_aggregation():
    agg, n_total = aggregate_records()
    assert n_total > 1000
    assert sum(a["count"] for a in agg.values()) == n_total
    vocab, stoi, adj, mech, evidence = build_graph(agg, n_total)
    for (i, j) in mech:
        assert evidence[(i, j)]["quotes"], "Kante ohne Beleg"


def test_vendor_world_deterministic():
    actions = [0, 1, 2, 1, 0, 2, 1, 1]
    fa = RichPanWorld.replay_frames(11, actions)
    fb = RichPanWorld.replay_frames(11, actions)
    for a, b in zip(fa, fb):
        assert a.tobytes() == b.tobytes()


def test_edge_ear_judges_direction():
    from form_arena import edge_ear
    assert edge_ear("Smoking causes tar buildup.", "smoking", "tar buildup", "causes")
    # Frontierung verdreht die Oberflächen-Richtung -> der Hörer MUSS durchfallen
    assert not edge_ear("Tar buildup: that is what smoking causes.",
                        "smoking", "tar buildup", "causes")
    # Ziffern-Normalisierung (Debug-Fund Nr. 3)
    assert edge_ear("Pressing the right key shifts the view 2 pixels to the right.",
                    "pressing the right key", "the view 2 pixels to the right",
                    "shifts")


def test_variants_no_double_articles():
    from form_arena import variants
    for name, prose in variants("pressing the left key", "shifts",
                                "the view 3 pixels to the left"):
        assert "the the" not in prose.lower() and "the pressing" not in prose.lower(), (name, prose)




def test_discourse_ear_resolves_and_rejects():
    from diskurs import ear_transcript
    edges = [{"subj": "smoking", "verb": "causes", "obj": "tar buildup", "conf": 0.9},
             {"subj": "smoking", "verb": "causes", "obj": "lung damage", "conf": 0.9}]
    ok, _ = ear_transcript(["Smoking causes tar buildup.",
                            "It also causes lung damage."], edges)
    assert ok
    # illegaler Referenzsprung: anderes Subjekt, Pronomen -> muss reissen
    edges2 = [{"subj": "smoking", "verb": "causes", "obj": "tar buildup", "conf": 0.9},
              {"subj": "stress", "verb": "damages", "obj": "health", "conf": 0.9}]
    ok2, _ = ear_transcript(["Smoking causes tar buildup.",
                             "It also damages health."], edges2)
    assert not ok2


def test_certificate_tamper_detection():
    import json
    from diskurs import verify_certificate
    cert = json.load(open("erweiterung/results/zertifikat_health.json"))
    edges = [{"subj": c["subj"], "verb": c["verb"], "obj": c["obj"]}
             for c in cert["claims"]]
    status, fails = verify_certificate(cert, edges)
    assert status == "FREIGEGEBEN", fails
    bad = json.loads(json.dumps(cert))
    bad["text"] = bad["text"].replace("causes", "prevents", 1)
    status2, _ = verify_certificate(bad, edges)
    assert status2 == "VERWORFEN"




def test_oberflaeche_released_and_guarded():
    from oberflaeche import (check_text, facts_health, SURFACE_HEALTH,
                             facts_weltbuch, SURFACE_WELTBUCH)
    ok, m, u = check_text(SURFACE_HEALTH, facts_health())
    assert ok and not m and not u
    ok, m, u = check_text(SURFACE_WELTBUCH, facts_weltbuch())
    assert ok and not m and not u
    # Lüge -> zurückhalten; Lücke -> zurückhalten
    ok, _, u = check_text(SURFACE_HEALTH + " Smoking improves health.",
                          facts_health())
    assert not ok and u
    ok, m, _ = check_text(SURFACE_HEALTH.replace("Caffeine prevents sleep; ", ""),
                          facts_health())
    assert not ok and m


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"PASS {fn.__name__}", flush=True)
    print(f"[test_erweiterung] {len(fns)}/{len(fns)} PASS", flush=True)
