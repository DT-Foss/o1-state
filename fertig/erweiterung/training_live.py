#!/usr/bin/env python3 -u
"""
LIVE-TRAINING, DIE O1-ART -- das eigene Modell lernt seine eigene Sprache,
als Stream, mit Surprise-Gate, auf CPU, warm gestartet.

Diametral zum klassischen Rezept (großer Korpus, Epochen, GPU):

  1. DIE SPRACHE IST KLEIN, ALSO IST MEISTERSCHAFT MÖGLICH. Ein
     6.3M-Modell kann "Englisch" nie; die Faktprosa-Register, die
     FERTIGs eigene Regeln erzeugen (Varianten, Übergänge, Pronomen,
     die Demo-Stil-Absätze), kann es VOLLSTÄNDIG meistern. Der Korpus
     wird deshalb GENERIERT -- aus den echten Graphen, nur wahre
     Aussagen, in tausenden Reihenfolgen und Formen (deterministisch,
     seedbar). Kein Download, kein fremder Korpus.
  2. STREAM STATT EPOCHEN, GATE STATT VOLLGAS. Ein Pass über den
     Strom, Chunk für Chunk; trainiert wird ein Chunk nur, wenn seine
     Überraschung über dem rollenden Quantil liegt (q=0.75, Ignition
     zuerst) -- das o1-state-Hausrezept (POS/P1: ~volle Qualität bei
     ~25% der Gradienten). Übersprungene Chunks kosten nur den
     No-Grad-Forward.
  3. WARM GESTARTET, NIE ÜBERSCHRIEBEN. Ausgang ist euer Checkpoint
     data/hsslm_form.pt (allgemeines Englisch-Rückgrat); das Ergebnis
     geht nach data/hsslm_form_live.pt -- das Original bleibt unberührt.
  4. GEMESSEN WIRD, WAS DER NUTZER HÖRT (FE10): degeneriert das freie
     Schreiben noch? Besteht ein FREI geschriebener Absatz die
     unsichtbare Prüfung? Vorher/Nachher, Zahlen statt Eindruck.
"""

import argparse
import json
import os
import sys
import time
from collections import deque

import numpy as np

FERTIG_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HERE = os.path.dirname(os.path.abspath(__file__))
for p in (FERTIG_ROOT, HERE):
    if p not in sys.path:
        sys.path.insert(0, p)

import torch
import torch.nn.functional as F

torch.set_num_threads(2)

from form_arena import variants
from diskurs import order_edges, _ddet
from oberflaeche import (facts_health, facts_weltbuch, SURFACE_HEALTH,
                         SURFACE_WELTBUCH)

OPENERS = ["", "Moreover, ", "Over time, ", "In turn, ", "Meanwhile, ",
           "In the end, ", "And ", "Yet ", "Once that happens, "]
LIVE_CKPT = os.path.join(FERTIG_ROOT, "data", "hsslm_form_live.pt")

# Gate-Konstanten: das o1-state-Rezept (streaming.py-Werte)
GATE_Q, GATE_WINDOW, MIN_WINDOW, IGNITION = 0.75, 200, 30, 25
SEQ = 128


def build_register_corpus(n_paragraphs: int = 2500, seed: int = 42) -> str:
    """Tausende wahre Absätze im Ziel-Register, deterministisch erzeugt:
    zufällige Teilmengen/Reihenfolgen der echten Fakten, zufällige der
    sechs geprüften Satzformen, Übergänge, Pronomen bei Subjekt-Wieder-
    holung. Dazu die zwei Demo-Stil-Absätze als Stil-Saat (x12, offen
    deklariert). NUR wahre Aussagen -- Reihenfolge und Form variieren,
    die Fakten nie."""
    rng = np.random.default_rng(seed)
    all_facts = facts_health() + facts_weltbuch()
    paras = []
    for _ in range(n_paragraphs):
        k = int(rng.integers(3, 9))
        idx = rng.choice(len(all_facts), size=min(k, len(all_facts)),
                         replace=False)
        facts = [dict(all_facts[int(i)], conf=0.8) for i in idx]
        ordered = order_edges(facts) if rng.random() < 0.7 else \
            [facts[int(j)] for j in rng.permutation(len(facts))]
        sents, prev = [], None
        for i, e in enumerate(ordered):
            forms = variants(e["subj"], e["verb"], e["obj"])
            name, prose = forms[int(rng.integers(0, len(forms)))]
            if (prev is not None and e["subj"] == prev["subj"]
                    and rng.random() < 0.5):
                prose = f"It also {e['verb']} {_ddet(e['obj'])}."
            elif i > 0 and rng.random() < 0.5 and not prose.startswith("It"):
                op = OPENERS[int(rng.integers(1, len(OPENERS)))]
                prose = op + prose[0].lower() + prose[1:]
            sents.append(prose)
            prev = e
        paras.append(" ".join(sents))
    paras += [SURFACE_HEALTH, SURFACE_WELTBUCH] * 12
    rng.shuffle(paras)
    return " ".join(paras)


def stream_train(minutes: float = 15.0, seed: int = 42,
                 out_ckpt: str = LIVE_CKPT):
    from fertig.form_engine import FormEngine
    eng = FormEngine()
    assert eng.ready, "Warmstart-Checkpoint fehlt"
    model = eng.model
    model.train()
    opt = torch.optim.Adam(model.parameters(), lr=2.5e-4)

    text = build_register_corpus(seed=seed)
    ids = torch.tensor(eng.bpe.encode(text), dtype=torch.long)
    print(f"[live] Register-Korpus: {len(text)} Zeichen, {ids.numel()} Tokens",
          flush=True)

    window = deque(maxlen=GATE_WINDOW)
    n_chunks = n_bwd = 0
    losses, pos = [], 0
    t0 = time.time()
    while time.time() - t0 < minutes * 60:
        if pos + SEQ + 1 >= ids.numel():
            pos = 0                                   # Strom läuft weiter
        x = ids[pos:pos + SEQ][None]
        y = ids[pos + 1:pos + SEQ + 1][None]
        pos += SEQ
        with torch.no_grad():
            out = model.forward(x)
            surprise = float(F.cross_entropy(
                out["logits"].reshape(-1, out["logits"].size(-1)),
                y.reshape(-1)))
        if n_chunks < IGNITION:
            gated = True
        elif len(window) >= MIN_WINDOW:
            gated = surprise > float(np.quantile(
                np.fromiter(window, dtype=np.float64), GATE_Q))
        else:
            gated = True
        if gated:
            out = model.forward(x)
            loss = F.cross_entropy(
                out["logits"].reshape(-1, out["logits"].size(-1)),
                y.reshape(-1))
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            n_bwd += 1
        window.append(surprise)
        losses.append(surprise)
        n_chunks += 1
        if n_chunks % 50 == 0:
            print(f"[live] chunk {n_chunks} | bwd {n_bwd} "
                  f"(gate {n_bwd/n_chunks:.2f}) | surprise "
                  f"{np.mean(losses[-50:]):.3f} | {time.time()-t0:.0f}s",
                  flush=True)
    model.eval()
    torch.save(model.state_dict(), out_ckpt)
    stats = {"chunks": n_chunks, "bwd": n_bwd,
             "gate_rate": round(n_bwd / max(n_chunks, 1), 4),
             "surprise_first25_mean": round(float(np.mean(losses[:25])), 4),
             "surprise_last50_mean": round(float(np.mean(losses[-50:])), 4),
             "wall_s": round(time.time() - t0, 1), "ckpt": out_ckpt}
    print(f"[live] fertig: {json.dumps(stats)}", flush=True)
    return stats


# ── Bewertung: hört man den Unterschied? ─────────────────────────────────
def free_write(engine, prompt: str, max_new: int = 70, seed: int = 0) -> str:
    torch.manual_seed(seed)
    pids = engine._encode(prompt).to(engine.device)
    with torch.no_grad():
        out = engine.model.generate(pids, max_new_tokens=max_new,
                                    use_zeno=True, use_foss_gate=False)
    return engine._decode(out[0].cpu())


def degeneration_score(text: str) -> dict:
    toks = text.split()
    if len(toks) < 8:
        return {"distinct_ratio": 0.0, "has_loop": True}
    distinct = len(set(toks)) / len(toks)
    tri = [tuple(toks[i:i + 3]) for i in range(len(toks) - 2)]
    from collections import Counter
    worst = Counter(tri).most_common(1)[0][1] if tri else 99
    return {"distinct_ratio": round(distinct, 3),
            "has_loop": bool(worst >= 3)}


def evaluate(ckpt: str = None, seeds=(0, 1, 2)) -> dict:
    from fertig.form_engine import FormEngine
    from oberflaeche import check_text
    res = {}
    for tag, path in (("vorher", None), ("nachher", ckpt or LIVE_CKPT)):
        eng = FormEngine() if path is None else FormEngine(model_path=path)
        if not eng.ready:
            res[tag] = {"error": "checkpoint fehlt"}
            continue
        samples, scores = [], []
        for s in seeds:
            txt = free_write(eng, "Smoking causes tar buildup. ", seed=s)
            samples.append(txt)
            scores.append(degeneration_score(txt))
        # frei geschriebener Fakten-Absatz -> unsichtbare Prüfung
        facts = [f for f in facts_health() if f["subj"] == "smoking"] + \
                [f for f in facts_health() if f["subj"] == "tar buildup"]
        released = 0
        free_paras = []
        for s in seeds:
            txt = free_write(eng, "The story begins with smoking. ",
                             max_new=90, seed=s)
            ok, _, _ = check_text(txt, facts)
            released += int(ok)
            free_paras.append(txt)
        res[tag] = {"distinct_ratio_mean": round(float(np.mean(
            [d["distinct_ratio"] for d in scores])), 3),
            "loops": sum(1 for d in scores if d["has_loop"]),
            "free_released": released, "samples": samples[:2],
            "free_paragraphs": free_paras[:1]}
        print(f"[eval:{tag}] distinct {res[tag]['distinct_ratio_mean']} | "
              f"loops {res[tag]['loops']}/{len(seeds)} | frei-bestanden "
              f"{released}/{len(seeds)}", flush=True)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--minutes", type=float, default=15.0)
    ap.add_argument("--eval-only", action="store_true")
    ap.add_argument("--out", default=os.path.join(HERE, "results",
                                                  "training_live.json"))
    args = ap.parse_args()
    out = {}
    if not args.eval_only:
        out["training"] = stream_train(minutes=args.minutes)
    out["evaluation"] = evaluate()
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f"[live] -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
