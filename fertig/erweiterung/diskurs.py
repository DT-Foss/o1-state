#!/usr/bin/env python3 -u
"""
DER DISKURS-KOMPONIST + TEXT-ZERTIFIKAT -- FERTIGs Endgame-Stück:
vom benoteten Einzelsatz zum komponierten TEXT mit Beipackzettel.

Drei Organe (Register FE6-FE8):

  KOMPONIST (FE6): aus einer Menge belegter Kanten wird ein Text --
    Reihenfolge nach dem Given-New-Gesetz (jeder Satz beginnt, wo
    möglich, mit bereits Eingeführtem; Topic-Ketten zuerst), Konnektive
    deterministisch aus der Kanten-Beziehung, und die Güte wird an der
    Form-Arena gemessen, jetzt auf Text-Ebene: Diskurs-UID (Surprisal-
    Varianz über den GANZEN Text) und strukturelle Kohärenz (erreichter
    vs. theoretisch maximaler Given-New-Anteil -- das Maximum ist durch
    die Komponenten-Struktur der Kantenmenge begrenzt und wird
    berechnet, nicht behauptet).

  PRONOMEN MIT OHR-GARANTIE (FE7): Wiederholte Subjekte werden zu
    "It also ..." -- aber NUR, wenn das eigene Ohr (edge_ear_discourse,
    geteilte Auflösungsregel: Pronomen = Subjekt des Vorsatzes) die
    Kante danach noch exakt zurückgewinnt. Der Mund hört sich selbst zu,
    bevor er spricht; geblockte Pronominalisierungen werden gezählt --
    ein Gate, das nie blockt, wäre Dekoration (Falsifier).

  TEXT-ZERTIFIKAT (FE8): jeder Text liefert ein maschinenprüfbares
    Zertifikat: Claims mit Graph-Kante + Quittungen (Weltbuch: Frame +
    SHA-256), Arena-Metriken, Ohr-Transkript, SHA des Textes.
    verify_certificate() rechnet ALLES von Null nach -- inklusive
    Bit-Replay gelebter Belege durch die Vendor-Welt -- und muss
    Manipulationen erkennen (Tamper-Tests: ein geändertes Wort, ein
    geänderter Hash -> VERWORFEN).

Prosa bleibt sauber, Quittungen wandern ins Zertifikat (Beipackzettel
statt Fußnoten im Satz) -- Flüssigkeit und Wahrheit trennen sich nie,
sie stehen nur auf getrennten Seiten desselben Dokuments.
"""

import hashlib
import json
import os
import sys

import numpy as np

FERTIG_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HERE = os.path.dirname(os.path.abspath(__file__))
for p in (FERTIG_ROOT, HERE):
    if p not in sys.path:
        sys.path.insert(0, p)

from fertig.bench import TrigramLM                                  # read-only
from fertig.pipeline import _toks                                   # read-only
from fertig.utterance import _VERB_SIGNALS                          # read-only

from form_arena import per_word_surprisals, variants, edge_ear, CORPUS, GRAPH
from lifted_walk import load_fertig_graph
from weltbuch import aggregate_records, build_graph as build_weltbuch
from _vendor_o1welt import RichPanWorld, frame_sha, estimate_shift

DATA = os.path.join(FERTIG_ROOT, "data", "weltbuch")


# ═══════════════════════════════════════════════════════════════════════════
#  Komponist
# ═══════════════════════════════════════════════════════════════════════════
def order_edges(edges):
    """Given-New-Greedy: beginne mit dem konfidenzstärksten Wurzel-Subjekt;
    wähle als nächstes stets eine Kante, deren Subjekt schon eingeführt
    ist (Topic-Kette: gleiches Subjekt vor Ketten-Fortsetzung), sonst --
    unvermeidbar bei neuer Komponente -- einen frischen Anfang.
    Deterministisch (Sortierung als Tie-Break)."""
    remaining = list(edges)
    objs = {e["obj"] for e in remaining}
    # Reine Quellen (Subjekte, die nirgends Objekt sind) MÜSSEN kalt
    # starten -- der Komponist beginnt Ketten deshalb dort, damit kein
    # vermeidbarer Kaltstart verschwendet wird (Debug-Fund: der alte
    # Greedy startete beim konfidenzstärksten Satz und verschenkte einen).
    remaining.sort(key=lambda e: (e["subj"] not in objs and -1 or 0,
                                  -e["conf"], e["subj"], e["obj"]))
    ordered, introduced = [], set()
    while remaining:
        pick = None
        for cand in remaining:                       # 1) gleiches Subjekt
            if ordered and cand["subj"] == ordered[-1]["subj"]:
                pick = cand
                break
        if pick is None:                             # 2) Subjekt eingeführt
            for cand in remaining:
                if cand["subj"] in introduced:
                    pick = cand
                    break
        if pick is None:                             # 3) Kaltstart: reine Quelle zuerst
            pick = next((c for c in remaining if c["subj"] not in objs),
                        remaining[0])
        remaining.remove(pick)
        ordered.append(pick)
        introduced.add(pick["subj"])
        introduced.add(pick["obj"])
    return ordered


def coherence(ordered):
    """(erreichte, maximal mögliche) Given-New-Sätze. Das strukturelle
    Maximum: jedes REINE QUELL-Subjekt (kommt nirgends als Objekt vor)
    muss genau einmal kalt eingeführt werden; alles andere kann chainen.
    Also max = n − #reine_Quellen. (Debug-Fund: die frühere Komponenten-
    Formel überschätzte das Maximum, weil eine Komponente mehrere reine
    Quellen enthalten kann.)"""
    n = len(ordered)
    objs = {e["obj"] for e in ordered}
    pure_sources = {e["subj"] for e in ordered if e["subj"] not in objs}
    seen, coherent = set(), 0
    for e in ordered:
        if e["subj"] in seen:
            coherent += 1
        seen.add(e["subj"])
        seen.add(e["obj"])
    return coherent, n - len(pure_sources)


def best_sentence(lm, e):
    """Die FE4-Kaskade pro Kante (IR -> Ohr -> min uid_var -> fluency),
    wiederverwendet aus form_arena über die dortigen Varianten."""
    from form_arena import select
    best, _, _ = select(lm, e["subj"], e["verb"], e["obj"], GRAPH)
    return best["prose"]


def compose(lm, edges):
    """Kanten -> Text. Pronomen-Regel (geteilt mit dem Ohr): folgt ein
    Satz mit GLEICHEM Subjekt, wird 'It also <verb> <obj>.' gesprochen --
    aber nur, wenn das Diskurs-Ohr den Probetext danach noch vollständig
    zurückgewinnt (Selbst-Zuhören vor dem Sprechen). Geblockte Versuche
    werden gezählt."""
    ordered = order_edges(edges)
    sentences, blocked = [], 0
    for i, e in enumerate(ordered):
        sent = best_sentence(lm, e)
        if i > 0 and e["subj"] == ordered[i - 1]["subj"]:
            # Pronomen-Satz wird aus der KANTE gebaut, nie aus dem
            # normalisierten Satz zurückgeschnitten (Debug-Fund: _toks
            # frisst Ziffern -- 'the view 2 pixels' verlor seine 2).
            cand = f"It also {e['verb']} {_ddet(e['obj'])}."
            trial = sentences + [cand]
            ok, _ = ear_transcript(trial, ordered[: i + 1])
            if ok:
                sent = cand
            else:
                blocked += 1
        sentences.append(sent)
    text = " ".join(sentences)
    return text, sentences, ordered, blocked


def gate_probe(lm, edges):
    """Der Richter-Beweis, deterministisch konstruiert: ein Pronomen über
    einen SUBJEKTWECHSEL hinweg (illegal -- das Ohr löst 'It' zum
    Vorsatz-Subjekt auf und hört dann die falsche Kante). Das Gate MUSS
    blocken; ein Gate, das diese Probe schluckt, ist Dekoration."""
    ordered = order_edges(edges)
    switch = next((i for i in range(1, len(ordered))
                   if ordered[i]["subj"] != ordered[i - 1]["subj"]), None)
    if switch is None:
        return None
    sentences = [best_sentence(lm, e) for e in ordered[:switch]]
    e = ordered[switch]
    illegal = f"It also {e['verb']} {_ddet(e['obj'])}."
    ok, _ = ear_transcript(sentences + [illegal], ordered[: switch + 1])
    return {"probe_sentence": illegal, "blocked": (not ok)}


# ═══════════════════════════════════════════════════════════════════════════
#  Diskurs-Ohr (geteilte Auflösungsregel)
# ═══════════════════════════════════════════════════════════════════════════
def _ddet(phrase):
    from form_arena import variants as _v  # nur für dieselbe det-Logik
    if phrase.split()[0] in ("the", "a", "an", "pressing"):
        return phrase
    from fertig.pipeline import _det
    return _det(phrase)


def _digits_ok(sent, obj):
    """Ziffernfester Oberflächen-Check: alle Ziffern-Tokens des Objekts
    müssen wörtlich im Satz stehen (Debug-Fund: _toks ist ziffernblind,
    also prüft das Ohr die Ziffern auf der ROHEN Oberfläche)."""
    s = sent.lower()
    return all(d in s.split() or d in s for d in
               [t for t in obj.lower().split() if any(ch.isdigit() for ch in t)])


def ear_transcript(sentences, expected):
    """Hört den Text Satz für Satz. Geteilte Regel: 'It …' am Satzanfang
    löst zum Subjekt des VORSATZES auf; danach muss die Kante exakt
    hörbar sein (edge_ear auf dem ggf. de-pronominalisierten Satz) UND
    die Ziffern des Objekts müssen wörtlich dastehen."""
    transcript = []
    prev_subj = None
    ok_all = True
    for sent, e in zip(sentences, expected):
        low = sent.lower()
        if low.startswith("it also ") or low.startswith("it "):
            heard_subj = prev_subj
            rest = sent.split(None, 2)[2] if low.startswith("it also ") \
                else sent.split(None, 1)[1]
            reconstructed = f"{heard_subj} {rest}" if heard_subj else sent
        else:
            heard_subj = e["subj"] if (" " + " ".join(_toks(sent)) + " ").find(
                " " + " ".join(_toks(e["subj"])) + " ") >= 0 else None
            reconstructed = sent
        heard = bool(heard_subj == e["subj"]
                     and edge_ear(reconstructed, e["subj"], e["obj"], e["verb"])
                     and _digits_ok(sent, e["obj"]))
        transcript.append({"sentence": sent, "heard_subj": heard_subj,
                           "expected": [e["subj"], e["verb"], e["obj"]],
                           "exact": heard})
        ok_all &= heard
        prev_subj = heard_subj if heard_subj else prev_subj
    return ok_all, transcript


# ═══════════════════════════════════════════════════════════════════════════
#  Zertifikat
# ═══════════════════════════════════════════════════════════════════════════
def make_certificate(name, text, sentences, ordered, lm, receipts=None):
    surps = per_word_surprisals(lm, text)
    coh, coh_max = coherence(ordered)
    ok, transcript = ear_transcript(sentences, ordered)
    claims = []
    for e in ordered:
        c = {"subj": e["subj"], "verb": e["verb"], "obj": e["obj"],
             "conf": e["conf"]}
        if receipts and (e["subj"], e["obj"]) in receipts:
            r = receipts[(e["subj"], e["obj"])]
            c["receipt"] = {"count": r["count"], "quotes": r["quotes"][:3]}
        claims.append(c)
    cert = {"name": name, "text": text, "text_sha256":
            hashlib.sha256(text.encode()).hexdigest(),
            "claims": claims,
            "metrics": {"discourse_uid_var": round(float(np.var(surps)), 4),
                        "fluency": round(float(-np.mean(surps)), 4),
                        "coherence": [coh, coh_max]},
            "ear": {"all_exact": bool(ok), "transcript": transcript}}
    return cert


def verify_certificate(cert, graph_edges, trace=None, n_replay=3):
    """Rechnet ALLES nach: Text-Hash, jede Claim-Kante existiert in der
    Kantenmenge, Ohr-Transkript wird neu gehört, und (Weltbuch) bis zu
    n_replay Quittungen werden durch die Vendor-Welt BIT-EXAKT
    nachgespielt. Ein einziger Bruch -> VERWORFEN."""
    fails = []
    if hashlib.sha256(cert["text"].encode()).hexdigest() != cert["text_sha256"]:
        fails.append("text_sha256")
    keyset = {(e["subj"], e["verb"], e["obj"]) for e in graph_edges}
    for c in cert["claims"]:
        if (c["subj"], c["verb"], c["obj"]) not in keyset:
            fails.append(f"claim_not_in_graph:{c['subj']}->{c['obj']}")
    sentences = [t["sentence"] for t in cert["ear"]["transcript"]]
    ok, _ = ear_transcript(sentences, cert["claims"])
    if not ok:
        fails.append("ear_reheard_failed")
    if trace is not None:
        quotes = [q for c in cert["claims"] for q in
                  c.get("receipt", {}).get("quotes", [])][:n_replay]
        cache = {}
        for q in quotes:
            key = q["base_seed"]
            if key not in cache:
                cache[key] = RichPanWorld.replay_frames(key, trace)
            fs = cache[key]
            f0, f1 = fs[q["frame"]], fs[q["frame"] + 1]
            if frame_sha(f0) != q["sha_pre"] or frame_sha(f1) != q["sha_post"]:
                fails.append(f"receipt_hash:{q['frame']}")
            s, _, _ = estimate_shift(f0, f1)
            if s != q["dx"]:
                fails.append(f"receipt_dx:{q['frame']}")
    return ("FREIGEGEBEN" if not fails else "VERWORFEN"), fails


# ═══════════════════════════════════════════════════════════════════════════
def main(out_path=None):
    lm = TrigramLM(open(CORPUS, encoding="utf-8", errors="ignore").read())

    # Text 1: Health-Graph
    vocab, stoi, adj, mech = load_fertig_graph(GRAPH)
    health_edges = [{"subj": vocab[a], "verb": mech[(a, b)], "obj": vocab[b],
                     "conf": c} for a, nb in sorted(adj.items())
                    for b, c in sorted(nb.items())]
    # Text 2: Weltbuch (mit Quittungen)
    agg, n_total = aggregate_records()
    wv, ws, wa, wm, wev = build_weltbuch(agg, n_total)
    welt_edges = [{"subj": wv[i], "verb": wm[(i, j)], "obj": wv[j],
                   "conf": c} for i, nb in sorted(wa.items())
                  for j, c in sorted(nb.items())]
    receipts = {(wv[i], wv[j]): wev[(i, j)] for (i, j) in wev}
    trace = json.load(open(os.path.join(DATA, "a_life.json")))["trace"]

    results = {}
    for name, edges, rec, tr in (("health", health_edges, None, None),
                                 ("weltbuch", welt_edges, receipts, trace)):
        probe = gate_probe(lm, edges)
        text, sentences, ordered, blocked = compose(lm, edges)
        # naive Ordnung als UID-Vergleichsanker
        naive_text = " ".join(best_sentence(lm, e) for e in edges)
        uid_naive = float(np.var(per_word_surprisals(lm, naive_text)))
        cert = make_certificate(name, text, sentences, ordered, lm, rec)
        status, fails = verify_certificate(cert, edges, tr)
        # Tamper-Tests: ein Wort ändern; eine Quittung fälschen
        t1 = dict(cert)
        t1 = json.loads(json.dumps(cert))
        t1["text"] = cert["text"].replace("the", "thee", 1)
        s1, _ = verify_certificate(t1, edges, tr)
        t2 = json.loads(json.dumps(cert))
        s2 = "FREIGEGEBEN"
        for c in t2["claims"]:
            if "receipt" in c and c["receipt"]["quotes"]:
                c["receipt"]["quotes"][0]["sha_pre"] = "0" * 64
                s2, _ = verify_certificate(t2, edges, tr)
                break
        n_pron = sum(1 for s in sentences if s.startswith("It also"))
        coh, coh_max = cert["metrics"]["coherence"]
        results[name] = {
            "text": text, "n_sentences": len(sentences),
            "coherence": [coh, coh_max],
            "coherence_at_max": bool(coh == coh_max),
            "discourse_uid_var": cert["metrics"]["discourse_uid_var"],
            "discourse_uid_var_naive_order": round(uid_naive, 4),
            "uid_composed_le_naive": bool(
                cert["metrics"]["discourse_uid_var"] <= uid_naive + 1e-9),
            "n_pronominalized": n_pron, "n_pron_blocked": blocked,
            "gate_probe": probe,
            "ear_all_exact": cert["ear"]["all_exact"],
            "certificate_status": status, "verify_fails": fails,
            "tamper_text_detected": s1 == "VERWORFEN",
            "tamper_receipt_detected": (s2 == "VERWORFEN") if rec else None,
        }
        print(f"\n════ TEXT '{name}' [{status}] ════", flush=True)
        print(text, flush=True)
        r = results[name]
        print(f"[diskurs:{name}] Kohärenz {coh}/{coh_max} | UID "
              f"{r['discourse_uid_var']} (naiv {r['discourse_uid_var_naive_order']}) "
              f"| Pronomen {n_pron} (geblockt {blocked}) | Ohr exakt "
              f"{r['ear_all_exact']} | Tamper erkannt: Text "
              f"{r['tamper_text_detected']} Quittung {r['tamper_receipt_detected']}",
              flush=True)
        cpath = os.path.join(HERE, "results", f"zertifikat_{name}.json")
        os.makedirs(os.path.dirname(cpath), exist_ok=True)
        with open(cpath, "w") as f:
            json.dump(cert, f, indent=2, ensure_ascii=False)

    out_path = out_path or os.path.join(HERE, "results", "diskurs.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\n[diskurs] -> {out_path}", flush=True)
    return results


if __name__ == "__main__":
    main()
