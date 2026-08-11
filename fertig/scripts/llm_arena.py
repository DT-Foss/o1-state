"""
llm_arena — die präregistrierte Arena: FERTIG vs. DeepSeek auf denselben
Benchmark-Samples. Gleiche Aufgaben, gleiche Metrik, deterministische Samples.

Benutzung: python3 scripts/llm_arena.py [--n 20]
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fertig.bench import (  # noqa: E402
    _read_parquet, _wikitext_lm, _snips_utterances, SNIPS_INTENTS,
    _toks, _fetch,
)

API_URL = "https://api.deepseek.com/chat/completions"
MODEL = "deepseek-chat"
BENCH = Path(__file__).resolve().parent.parent / "data" / "bench"


def ask(prompt: str, api_key: str) -> str:
    body = json.dumps({"model": MODEL, "temperature": 0,
                       "messages": [{"role": "user", "content": prompt}]}
                      ).encode()
    req = urllib.request.Request(
        API_URL, data=body,
        headers={"Content-Type": "application/json",
                 "Authorization": f"Bearer {api_key}"})
    with urllib.request.urlopen(req, timeout=60) as resp:
        out = json.loads(resp.read())
    return out["choices"][0]["message"]["content"].strip()


def arena(name: str, prompts, answers, api_key: str, verbose=False):
    hits = total = 0
    for p, a in zip(prompts, answers):
        try:
            pred = ask(p, api_key)
        except Exception as e:
            print(f"  API-Fehler: {e}")
            continue
        hits += int(pred == a)
        total += 1
        if verbose:
            print(f"  {name:12s} pred={pred!r:12s} gold={a!r:12s} "
                  f"{'OK' if pred == a else '..'}")
    return hits, total


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=20)
    args = ap.parse_args()
    api_key = os.environ.get("DEEPSEEK_API_KEY", "")
    if not api_key:
        sys.exit("DEEPSEEK_API_KEY nicht gesetzt")
    n = args.n

    print(f"=== LLM-Arena: DeepSeek ({MODEL}) auf FERTIG-Benchmarks ===\n")

    # --- HellaSwag ---
    rows = _read_parquet(BENCH / "hellaswag.parquet")[:n]
    ps, as_ = [], []
    for r in rows:
        endings = "\n".join(f"{i}. {e}" for i, e in enumerate(r["endings"]))
        ps.append(f"Wähle die wahrscheinlichste Fortsetzung (Antworte nur mit "
                  f"der Zahl 0-3).\nKontext: {r['ctx']}\n{endings}")
        as_.append(str(r["label"]))
    h, t = arena("HellaSwag", ps, as_, api_key)
    print(f"HellaSwag: DeepSeek {h}/{t} ({100*h/max(t,1):.1f}%) | "
          f"FERTIG: 26.7%")

    # --- WinoGrande ---
    rows = _read_parquet(BENCH / "winogrande.parquet")[:n]
    ps, as_ = [], []
    for r in rows:
        ps.append(f"Fülle den Blank (_) auf. Antworte nur mit 1 oder 2.\n"
                  f"Satz: {r['sentence']}\n1. {r['option1']}\n2. {r['option2']}")
        as_.append(str(r["answer"]))
    h, t = arena("WinoGrande", ps, as_, api_key)
    print(f"WinoGrande: DeepSeek {h}/{t} ({100*h/max(t,1):.1f}%) | "
          f"FERTIG: 50.9%")

    # --- BLiMP (Anapher + Inseln) ---
    pairs = []
    for sub in ("anaphor_gender_agreement", "left_branch_island_simple_question"):
        p = _fetch(
            "https://huggingface.co/datasets/nyu-mll/blimp/resolve/main/"
            f"{sub}/train-00000-of-00001.parquet", f"blimp_{sub}.parquet")
        pairs += [(r["sentence_good"], r["sentence_bad"]) for r in
                 _read_parquet(p)][: n // 2]
    ps, as_ = [], []
    for g, b in pairs:
        ps.append(f"Welcher Satz ist grammatisch korrekt? Antworte nur mit 1 "
                  f"oder 2.\n1. {g}\n2. {b}")
        as_.append("1")
    h, t = arena("BLiMP", ps, as_, api_key)
    print(f"BLiMP-Sample: DeepSeek {h}/{t} ({100*h/max(t,1):.1f}%) | "
          f"FERTIG (diese Subtasks): 24.9% / 82.6%")

    # --- SNIPS ---
    ps, as_ = [], []
    intents = sorted(SNIPS_INTENTS)
    labels = ", ".join(intents)
    for intent in intents:
        utts = _snips_utterances(intent)
        for u in utts[: n // len(intents)]:
            ps.append(f"Klassifiziere die Anfrage in genau einen Intent: "
                      f"{labels}. Antworte NUR mit dem Intent-Namen.\n\n{u}")
            as_.append(intent)
    h, t = arena("SNIPS", ps, as_, api_key)
    print(f"SNIPS: DeepSeek {h}/{t} ({100*h/max(t,1):.1f}%) | "
          f"FERTIG: 88.3%")


if __name__ == "__main__":
    main()
