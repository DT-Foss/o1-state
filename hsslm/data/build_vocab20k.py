"""
Baut hsslm/data/vocab20k.txt (Sprecher-Datenbasis v2, Baustein 4) + misst
OOV-Raten 5k vs 20k auf LM-Stream-Text UND graph_to_text_pairs_v2.jsonl.

Nutzt DENSELBEN WT-103-Text wie speak_train.py's --vocab-file-loser Default-
Pfad (ds["train"]["text"][:20000], auf 20M Zeichen begrenzt) -- damit ein
mit diesem Skript gebautes vocab20k.txt und speak_train.py's eigener
Vokabular-Aufbau (falls --vocab-file NICHT gesetzt ist) konsistent bleiben.

Usage:
    OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
    HF_DATASETS_OFFLINE=1 HF_HUB_OFFLINE=1 \
    python3 build_vocab20k.py
"""
import json
import os
import sys

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
os.environ.setdefault("HF_HUB_OFFLINE", "1")

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(REPO_ROOT, "src"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from vocab import build_vocab_n, save_vocab, compute_oov_stats  # noqa: E402
from length_extrap_v2 import build_vocab as build_vocab_5k  # noqa: E402


def main():
    print("[build_vocab20k] loading WT-103 (same slice as speak_train.py default)...")
    from datasets import load_dataset
    ds = load_dataset("Salesforce/wikitext", "wikitext-103-raw-v1")
    text = "\n\n".join(ds["train"]["text"][:20000])[:20_000_000]
    print(f"    text chars: {len(text):,}")

    print("[build_vocab20k] building 5k vocab (length_extrap_v2.build_vocab, unveraendert)...")
    vocab5k, stoi5k, unk5k, mask5k = build_vocab_5k(text)
    print(f"    5k vocab: {len(vocab5k)} words, unk={unk5k}, mask={mask5k}")

    print("[build_vocab20k] building 20k vocab (vocab.build_vocab_n, n=20000)...")
    vocab20k, stoi20k, unk20k, mask20k = build_vocab_n(text, 20000)
    print(f"    20k vocab: {len(vocab20k)} words, unk={unk20k}, mask={mask20k}")

    out_path = os.path.join(os.path.dirname(__file__), "vocab20k.txt")
    save_vocab(out_path, vocab20k)
    print(f"[build_vocab20k] wrote {out_path} ({len(vocab20k)} lines)")

    # --- OOV messen: LM-Stream (der ganze Trainingstext als eine lange
    # Sequenz von "Saetzen" -- hier pro Zeile des Rohtexts approximiert,
    # da WT-103 keine Satzgrenzen im Rohformat traegt) UND v2-Paare
    # (structure + text getrennt gemessen, wie graph_to_text.py's
    # compute_oov_stats es fuer v1 tut).
    print("\n[build_vocab20k] measuring OOV: LM-stream text (5k vs 20k)...")
    lm_lines = [l for l in text.split("\n") if l.strip()][:20000]
    lm_oov_5k = compute_oov_stats(lm_lines, stoi5k, unk5k)
    lm_oov_20k = compute_oov_stats(lm_lines, stoi20k, unk20k)

    v2_path = os.path.join(os.path.dirname(__file__), "graph_to_text_pairs_v2.jsonl")
    v2_oov = {}
    if os.path.exists(v2_path):
        print(f"[build_vocab20k] measuring OOV: {v2_path} (5k vs 20k)...")
        pairs = [json.loads(l) for l in open(v2_path)]
        texts = [p["text"] for p in pairs]
        structures = [p["structure"] for p in pairs]
        v2_oov = {
            "text_5k": compute_oov_stats(texts, stoi5k, unk5k),
            "text_20k": compute_oov_stats(texts, stoi20k, unk20k),
            "structure_5k": compute_oov_stats(structures, stoi5k, unk5k),
            "structure_20k": compute_oov_stats(structures, stoi20k, unk20k),
            "n_pairs": len(pairs),
        }
    else:
        print(f"[build_vocab20k] {v2_path} not found yet -- skipping v2-pairs OOV "
              f"(run graph_to_text_v2.py first, or re-run this script after)")

    report = {
        "vocab_5k_size": len(vocab5k),
        "vocab_20k_size": len(vocab20k),
        "lm_stream": {"5k": lm_oov_5k, "20k": lm_oov_20k},
        "v2_pairs": v2_oov,
    }
    report_path = os.path.join(os.path.dirname(__file__), "vocab20k_oov_report.json")
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)

    print("\n" + "=" * 70)
    print("OOV REPORT")
    print("=" * 70)
    print(json.dumps(report, indent=2))
    print(f"\nReport JSON: {report_path}")


if __name__ == "__main__":
    main()
