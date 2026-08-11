"""
extract_patterns — Muster-Bank aus Rohtext minen (gewicht-frei, reines Zählen).

Benutzung:
    python3 scripts/extract_patterns.py [korpus.txt ...] -o data/faraday_bank.json
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fertig.pattern_bank import PatternBank  # noqa: E402

DEFAULT_CORPUS = Path(__file__).resolve().parent.parent / "data" / "faraday_candle.txt"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("corpora", nargs="*", default=[str(DEFAULT_CORPUS)])
    ap.add_argument("-o", "--out", default=str(DEFAULT_CORPUS.parent / "faraday_bank.json"))
    args = ap.parse_args()

    bank = PatternBank()
    for path in args.corpora:
        text = Path(path).read_text(encoding="utf-8", errors="ignore")
        bank.extract(text)
        print(f"extrahiert: {path} ({len(text)} Zeichen)")

    print(f"\nSätze          : {bank.n_sentences}")
    print(f"dist. Skelette : {len(bank.skeletons)}")
    print(f"dist. Opener   : {len(bank.openers)}")

    print("\n=== Top-Opener nach Polaritäts-Klasse (gemessen) ===")
    for cls in ("cause", "contrast", "add", "neutral"):
        inv = bank.opener_inventory(polarity=cls)[:6]
        items = ", ".join(f"'{' '.join(o['tokens'])}' x{o['count']} "
                          f"(c={o['conf']})" for o in inv)
        print(f"  {cls:9s}: {items}")

    print("\n=== Top-2-Slot-Frames (Assertions-Kandidaten) ===")
    for fr in bank.frames(n_slots=2)[:8]:
        print(f"  x{fr['count']:<4} c={fr['conf']:.2f}  {' '.join(fr['skeleton'])}")
        if fr["filler_samples"]:
            print(f"        z. B. {fr['filler_samples'][0]}")

    bank.save(args.out)
    print(f"\nBank gespeichert: {args.out}")


if __name__ == "__main__":
    main()
