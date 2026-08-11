"""
Struktur->Satz Trainingsdatensatz-Generator v2 (Sprecher-Datenbasis v2,
Baustein 3). Wiederverwendet graph_to_text.py's Rekonstruktionslogik
(curator_yield_run.iter_windows, gleiche Cadence-Mechanik) unveraendert --
DUPLIZIERT SIE NICHT -- und wendet sie auf den vollen Store an
(results/wt103_full_store_local, 20278 Segmente, 65125 Records, gebaut
mit derselben Cadence wie P72 nur chunks=200000 statt 3000, siehe
run_wt103_full_build.sh auf core).

NEU gegenueber graph_to_text.py:
  1. Junk-Schluessel-Filter (key_filter.py): Paare, deren trigger ODER
     outcome ein Junk-Muster traegt (WT-103-Markup-Reste, siehe dort),
     werden VOR dem Schreiben verworfen. Der Store selbst bleibt
     unberuehrt -- der Filter wirkt nur bei der Paar-Erzeugung.
  2. Deduplikation: mehrere Records koennen dasselbe (trigger, mechanism,
     outcome)-Triplet an verschiedenen doc_coord-Positionen erzeugen
     (z.B. wiederkehrende Wendungen im Korpus) -- dieselbe Struktur waere
     dann mehrfach im Trainingsset, ohne zusaetzlichen Lehrwert. Dedupe-
     Schluessel: (structure, text) -- erste Fundstelle in Store-
     Iterationsreihenfolge gewinnt (deterministisch, LiveStore.iter_records
     iteriert Segmente in sortierter sha-Reihenfolge).
  3. Deterministische Sortierung der Ausgabe: nach (citation.sha,
     citation.idx) -- byte-identische Datei bei jedem Lauf ueber denselben
     Store, unabhaengig von Dict-Iterationsreihenfolgen o.ae.

Ausgabe-Schema IDENTISCH zu graph_to_text_pairs.jsonl (v1):
    {"structure": str, "text": str, "citation": {"sha": str, "idx": int},
     "doc_coord": int, "trigger": str, "mechanism": str, "outcome": str}

Usage:
    OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
    HF_DATASETS_OFFLINE=1 HF_HUB_OFFLINE=1 \
    python3 graph_to_text_v2.py \
        --store results/wt103_full_store_local \
        --out hsslm/data/graph_to_text_pairs_v2.jsonl \
        --chunks 200000
"""
import argparse
import json
import os
import sys

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
os.environ.setdefault("HF_HUB_OFFLINE", "1")

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(REPO_ROOT, "src"))
sys.path.insert(0, os.path.join(REPO_ROOT, "reference"))
sys.path.insert(0, os.path.join(REPO_ROOT, "vendor", "fabel", "extract"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from graph_to_text import build_pairs, P72_CADENCE  # noqa: E402
from key_filter import filter_pair, is_junk_key  # noqa: E402

# Volle Store-Cadence: identisch zu P72_CADENCE, nur chunks unterschiedlich
# (der Store wurde mit chunks=200000 gebaut, run_wt103_full_build.sh auf
# core -- gestoppt bei n_streamed_tape_pos=94950 durch eine Zeit-Deadline,
# nicht durch Erreichen von chunks=200000; die Rekonstruktion braucht daher
# NICHT die volle chunks=200000-Laufzeit, iter_windows stoppt selbst, wenn
# alle needed_positions gefunden sind -- siehe reconstruct_tape_pos_to_text).
FULL_CADENCE = dict(P72_CADENCE)
FULL_CADENCE["chunks"] = 200000


def build_pairs_v2(store_dir: str, cadence: dict = FULL_CADENCE):
    """build_pairs() + Junk-Filter + Dedupe + deterministische Sortierung.
    Returns (pairs, report) -- report traegt die Zaehlung fuer den
    Report an team-lead (n roh, n junk-gefiltert, n dedupe-gefiltert,
    n final)."""
    raw_pairs = list(build_pairs(store_dir, cadence))
    n_raw = len(raw_pairs)

    kept_after_junk = []
    n_junk_filtered = 0
    for p in raw_pairs:
        if filter_pair(p, key_fields=("trigger", "outcome")):
            kept_after_junk.append(p)
        else:
            n_junk_filtered += 1

    seen = set()
    deduped = []
    n_dupe_filtered = 0
    for p in kept_after_junk:
        key = (p["structure"], p["text"])
        if key in seen:
            n_dupe_filtered += 1
            continue
        seen.add(key)
        deduped.append(p)

    deduped.sort(key=lambda p: (p["citation"]["sha"], p["citation"]["idx"]))

    report = {
        "n_raw": n_raw,
        "n_junk_filtered": n_junk_filtered,
        "n_dupe_filtered": n_dupe_filtered,
        "n_final": len(deduped),
    }
    return deduped, report


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--store", default=os.path.join(REPO_ROOT, "results", "wt103_full_store_local"))
    ap.add_argument("--out", default=os.path.join(os.path.dirname(__file__), "graph_to_text_pairs_v2.jsonl"))
    ap.add_argument("--chunks", type=int, default=FULL_CADENCE["chunks"])
    args = ap.parse_args()

    cadence = dict(FULL_CADENCE)
    cadence["chunks"] = args.chunks

    pairs, report = build_pairs_v2(args.store, cadence)

    with open(args.out, "w", encoding="utf-8") as f:
        for p in pairs:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")

    print("\n" + "=" * 70)
    print("GRAPH_TO_TEXT V2 REPORT")
    print("=" * 70)
    print(json.dumps(report, indent=2))
    print(f"\nwrote {len(pairs)} pairs to {args.out}")

    report_out = args.out.replace(".jsonl", "_report.json")
    with open(report_out, "w") as f:
        json.dump(report, f, indent=2)
    print(f"Report JSON: {report_out}")


if __name__ == "__main__":
    main()
