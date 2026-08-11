"""
Struktur->Satz Trainingsdatensatz-Generator v3 (Sprecher-Datenbasis v2,
dringende Nachbesserung nach dem DeepSeek-Grader-Schwarm-Befund auf v2:
5189 Paare bewertet, 44% broken, dominiert von zwei Fenster-Rekonstruktions-
Defekten:

  (a) FRAGMENT  -- 87% der Fenster beginnen mitten im Satz (Token- statt
      Satzgrenzen), z.B. "2015 study suggested that..." (der Satzanfang
      "A 2015 study..." o.ae. fehlt).
  (b) MISMATCH  -- manche Fenster enthalten den Inhalt ihres eigenen
      Records nicht (Paar-Index 1 in v2: structure "many frozen | is
      caused by | ...", text = ein Mammut-Fenster ohne diese Woerter).

WURZEL: curator_yield_run.iter_windows steht unter MVP-4-CONTRACT-Schutz
("do not change without flagging the team lead" -- builder_run.py haengt
an der exakten Signatur/Yield-Form) und gibt window_source_text() NUR den
rohen Token-Slice zurueck, ohne Satzgrenzen-Kenntnis. Dieser Fix aendert
iter_windows NICHT -- er dupliziert dessen Kern-Streaming-Schleife HIER
(gleiche Klassen: HFStreamWithText, ChunkFeederWithText, Organism.
step_gated, gleiche Cadence-Parameter -> gleiche tape_pos-Sequenz), haelt
aber seg_tape im eigenen Scope, um bei Bedarf ZUSAETZLICHEN Kontext vor/
nach dem urspruenglichen Fenster zu lesen -- das ist der Unterschied zu
window_source_text, der nur den 128-Token-Slice selbst sieht.

FIX 1 -- Satzgrenzen-Snapping:
  Fenstertext wird mit +/- PAD_TOKENS Kontext-Puffer gezogen (Default 64,
  siehe SNAP_PAD_TOKENS), dann:
    - Vorwaerts-Snap auf den naechsten Satzanfang NACH dem urspruenglichen
      Fensterbeginn (WT-103-Konvention: Saetze enden auf ". "/"! "/"? ",
      mit Leerzeichen VOR der Interpunktion aus der Tokenisierung -- siehe
      _SENTENCE_END_RE).
    - Rueckwaerts-Snap des Fensterendes auf die letzte Satzgrenze VOR dem
      Puffer-Ende (kein abgeschnittener letzter Satz).
  Falls im Puffer keine Satzgrenze gefunden wird (seltene Randlage nahe
  Tape-Anfang/-Ende), faellt der Snap auf den unveraenderten Rohtext
  zurueck UND das Paar durchlaeuft trotzdem den Content-Guard (Fix 2) --
  kein stiller Qualitaetsverlust ohne Pruefung.

FIX 2 -- Content-Guard (Mismatch-Schutz):
  Content-Woerter (>3 Zeichen, [a-zA-Z]+) aus trigger+outcome muessen zu
  >=50% im (gesnappten) Fenster vorkommen (case-insensitive Teilstring-
  Match je Wort). Unterschreitet ein Paar die Schwelle, wird es VERWORFEN
  (gezaehlt, nicht stillschweigend behalten) -- kein Fenster-Verschieben-
  Versuch, weil der eigentliche Fund (Mismatch) heisst: die tape_pos des
  Records und die tape_pos, die iter_windows/diese Funktion fuer denselben
  chunk-Index sieht, sind bereits divergiert (z.B. durch eine store-seitig
  andere Builder-Version) -- ein lokal verschobenes Fenster wuerde nicht
  zuverlaessig den richtigen Inhalt treffen, ein Verwurf ist die ehrliche
  Antwort.

Alles andere (Filter, Dedupe, deterministische Sortierung, Schema) bleibt
identisch zu graph_to_text_v2.py -- v3 importiert dessen filter_pair-
Wiederverwendung aus key_filter.py, dupliziert NICHT die Junk-Filter-Logik.

Usage:
    OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
    HF_DATASETS_OFFLINE=1 HF_HUB_OFFLINE=1 \
    python3 graph_to_text_v3.py \
        --store results/wt103_full_store_local \
        --out hsslm/data/graph_to_text_pairs_v3.jsonl \
        --chunks 96000
"""
import argparse
import json
import os
import re
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

from graph_to_text import linearize_structure, P72_CADENCE  # noqa: E402
from key_filter import filter_pair  # noqa: E402

FULL_CADENCE = dict(P72_CADENCE)
FULL_CADENCE["chunks"] = 200000

# Puffer-Groesse (Tokens) auf jeder Seite des urspruenglichen Fensters, in
# der nach einer Satzgrenze gesucht wird. 64 = die Haelfte von
# window_tokens=128 -- grosszuegig genug fuer die meisten Saetze, ohne den
# joined-Text pro Position unnoetig aufzublasen (Vielfaches der 51128
# benoetigten Positionen).
SNAP_PAD_TOKENS = 64

# WT-103-Tokenisierung: Satzzeichen ist von Nachbar-Woertern durch
# Leerzeichen getrennt ("word . Next" statt "word. Next"). Eine
# Satzgrenze ist ". "/"! "/"? " gefolgt von einem Grossbuchstaben oder
# Anfuehrungszeichen (haelt Abkuerzungen wie "U. S." nicht zuverlaessig
# auseinander -- das ist eine bewusste Vereinfachung, keine vollstaendige
# Satzsegmentierung; der Content-Guard faengt grobe Fehltreffer ab).
_SENTENCE_END_RE = re.compile(r'[.!?]\s+(?=[A-Z"“])')

_WORD_RE = re.compile(r"[a-zA-Z]+")


def snap_to_sentence_boundaries(padded_text: str, orig_start_char: int,
                                 orig_end_char: int) -> str:
    """padded_text traegt orig_start_char..orig_end_char als Koordinaten
    des URSPRUENGLICHEN (nicht gesnappten) Fensters innerhalb von
    padded_text. Gibt den auf Satzgrenzen gesnappten Teilstring zurueck:
    Beginn = erste Satzgrenze NACH orig_start_char (oder Textanfang, falls
    keine gefunden), Ende = letzte Satzgrenze VOR orig_end_char (oder
    Textende, falls keine gefunden)."""
    # Vorwaerts-Snap: erste Grenze, deren Ende-Position >= orig_start_char
    # liegt (die naechste Satzgrenze NACH dem urspruenglichen Beginn).
    # Kein Treffer -> Fallback auf orig_start_char selbst (NICHT Puffer-
    # anfang 0 -- ein Snap ohne gefundene Grenze darf den Fensterbereich
    # nicht auf den ganzen, viel groesseren Puffer aufblasen).
    fwd_match = None
    for m in _SENTENCE_END_RE.finditer(padded_text):
        if m.end() >= orig_start_char:
            fwd_match = m
            break
    start = fwd_match.end() if fwd_match is not None else orig_start_char

    # Rueckwaerts-Snap: letzte Grenze, deren Ende-Position <= orig_end_char
    # liegt UND die NACH dem gewaehlten start liegt (die letzte Satzgrenze
    # VOR/AM urspruenglichen Ende, die einen ANDEREN Satz abschliesst als
    # den, dessen Anfang start markiert -- sonst wuerde derselbe einzelne
    # Match, der noch VOR orig_start_char begann, faelschlich auch als
    # Ende-Grenze zaehlen und Fenster kollabieren lassen, wenn im Puffer
    # nur eine einzige Satzgrenze liegt). Kein Treffer -> Fallback auf
    # orig_end_char selbst (gleiche Begruendung wie beim Vorwaerts-Snap).
    bwd_match = None
    for m in _SENTENCE_END_RE.finditer(padded_text):
        if m.start() < start:
            continue
        if m.end() <= orig_end_char:
            bwd_match = m
        else:
            break
    end = bwd_match.start() + 1 if bwd_match is not None else orig_end_char

    if start >= end:
        # Snap kollabierte -- Rueckfall auf den unveraenderten Rohbereich;
        # Content-Guard prueft trotzdem, kein stiller Qualitaetsverlust.
        return padded_text[orig_start_char:orig_end_char].strip()
    return padded_text[start:end].strip()


def content_guard_ok(trigger: str, outcome: str, text: str, threshold: float = 0.5) -> bool:
    """>=50% der Content-Woerter (>3 Zeichen) aus trigger+outcome muessen
    im (gesnappten) Fenster vorkommen (case-insensitive)."""
    words = [w for w in _WORD_RE.findall(f"{trigger} {outcome}".lower()) if len(w) > 3]
    if not words:
        return True  # nichts zu pruefen (z.B. nur kurze Woerter) -- nicht verwerfen
    text_lower = text.lower()
    n_hit = sum(1 for w in words if w in text_lower)
    return (n_hit / len(words)) >= threshold


def reconstruct_snapped(cadence: dict, needed_positions: set, pad_tokens: int = SNAP_PAD_TOKENS):
    """Eigene Streaming-Schleife (dupliziert curator_yield_run.iter_windows's
    Kern-Logik ABSICHTLICH, statt es zu importieren -- iter_windows selbst
    bleibt unter MVP-4-Contract unveraendert), die seg_tape im eigenen Scope
    haelt, um bei jeder benoetigten tape_pos einen groesseren Kontext-Puffer
    lesen zu koennen als window_source_text erlaubt.

    Returns {tape_pos: snapped_text}."""
    import curator_yield_run as cyr
    import portable_organism as po
    import torch

    stream_cls = cyr.HFStreamWithText

    po.D_MODEL, po.BATCH, po.CHUNK = cadence["d_model"], cadence["batch"], cadence["chunk_size"]
    po.GATE_Q, po.GATE_WINDOW, po.MIN_WINDOW, po.IGNITION_CHUNKS = \
        cadence["q"], cadence["window"], cadence["min_window"], cadence["ignition_chunks"]
    vocab, stoi, unk, mask, val_ids = po.get_vocab()

    torch.manual_seed(cadence["seed"])
    org = po.Organism("curator", len(vocab), mask, seed=cadence["seed"])
    stream = stream_cls(cadence["substrate"], stoi, unk)
    feeder = cyr.ChunkFeederWithText(stream, po.BATCH, po.CHUNK)

    window_tokens = cadence["window_tokens"]
    remaining = set(needed_positions)
    found = {}

    seg_tape = []
    for ci in range(1, cadence["chunks"] + 1):
        x, y, lane0_segs = feeder.next_xy()
        s, gated, nll = org.step_gated(x, y)
        tape_pos = len(seg_tape)
        seg_tape.extend(lane0_segs)

        if tape_pos in remaining:
            pad_start = max(0, tape_pos - pad_tokens)
            pad_end = min(len(seg_tape), tape_pos + window_tokens + pad_tokens)
            # seg_tape can still be shorter than pad_end right after this
            # chunk's extend() -- clip is already applied via min() above,
            # and any further growth happens on LATER chunks, which only
            # gives snapping MORE right-context on a later lookup. Since a
            # given tape_pos is resolved the first time it's seen (this
            # chunk), the pad_end here is the tape's current true length.
            padded_text = "".join(seg_tape[pad_start:pad_end])
            orig_start_char = len("".join(seg_tape[pad_start:tape_pos]))
            orig_end_char = len("".join(seg_tape[pad_start:min(tape_pos + window_tokens, len(seg_tape))]))
            snapped = snap_to_sentence_boundaries(padded_text, orig_start_char, orig_end_char)
            found[tape_pos] = snapped
            remaining.discard(tape_pos)
        if not remaining:
            break
    return found


def build_pairs_v3(store_dir: str, cadence: dict = FULL_CADENCE):
    """Wie graph_to_text_v2.build_pairs_v2, aber mit satzgrenzen-gesnapptem
    Fenstertext + Content-Guard statt roher Token-Slices. Returns (pairs,
    report) -- report zaehlt zusaetzlich n_content_guard_rejected."""
    sys.path.insert(0, os.path.join(REPO_ROOT, "src", "livecausal"))
    from store import LiveStore

    store = LiveStore(store_dir)
    all_records = list(store.iter_records())
    needed_positions = {rec["doc_coord"] for _, _, rec in all_records}

    print(f"[graph_to_text_v3] store records: {len(all_records)}  "
          f"unique doc_coord positions needed: {len(needed_positions)}")
    print(f"[graph_to_text_v3] reconstructing SNAPPED window_text "
          f"(cadence: {cadence}, pad_tokens={SNAP_PAD_TOKENS})...")

    tape_pos_to_text = reconstruct_snapped(cadence, needed_positions)
    n_found = len(tape_pos_to_text)
    n_missing = len(needed_positions) - n_found
    print(f"[graph_to_text_v3] reconstructed {n_found}/{len(needed_positions)} positions "
          f"({n_missing} not covered by chunks={cadence['chunks']})")

    raw_pairs = []
    n_skipped_missing = 0
    for sha, idx, rec in all_records:
        doc_coord = rec["doc_coord"]
        text = tape_pos_to_text.get(doc_coord)
        if text is None:
            n_skipped_missing += 1
            continue
        structure = linearize_structure(rec["trigger_key"], rec["mechanism"], rec["outcome_key"])
        raw_pairs.append({
            "structure": structure,
            "text": text,
            "citation": {"sha": sha, "idx": idx},
            "doc_coord": doc_coord,
            "trigger": rec["trigger"],
            "mechanism": rec["mechanism"],
            "outcome": rec["outcome"],
        })
    n_raw = len(raw_pairs)
    print(f"[graph_to_text_v3] {n_raw} pairs with reconstructed text "
          f"({n_skipped_missing} skipped: doc_coord outside coverage)")

    # --- Content-Guard: verwirft Mismatch-Faelle -----------------------
    kept_after_guard = []
    n_content_guard_rejected = 0
    for p in raw_pairs:
        if content_guard_ok(p["trigger"], p["outcome"], p["text"]):
            kept_after_guard.append(p)
        else:
            n_content_guard_rejected += 1

    # --- Junk-Schluessel-Filter (identisch zu v2) -----------------------
    kept_after_junk = []
    n_junk_filtered = 0
    for p in kept_after_guard:
        if filter_pair(p, key_fields=("trigger", "outcome")):
            kept_after_junk.append(p)
        else:
            n_junk_filtered += 1

    # --- Dedupe (identisch zu v2) ----------------------------------------
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
        "n_skipped_missing_coverage": n_skipped_missing,
        "n_content_guard_rejected": n_content_guard_rejected,
        "n_junk_filtered": n_junk_filtered,
        "n_dupe_filtered": n_dupe_filtered,
        "n_final": len(deduped),
    }
    return deduped, report


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--store", default=os.path.join(REPO_ROOT, "results", "wt103_full_store_local"))
    ap.add_argument("--out", default=os.path.join(os.path.dirname(__file__), "graph_to_text_pairs_v3.jsonl"))
    ap.add_argument("--chunks", type=int, default=FULL_CADENCE["chunks"])
    args = ap.parse_args()

    cadence = dict(FULL_CADENCE)
    cadence["chunks"] = args.chunks

    pairs, report = build_pairs_v3(args.store, cadence)

    with open(args.out, "w", encoding="utf-8") as f:
        for p in pairs:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")

    print("\n" + "=" * 70)
    print("GRAPH_TO_TEXT V3 REPORT")
    print("=" * 70)
    print(json.dumps(report, indent=2))
    print(f"\nwrote {len(pairs)} pairs to {args.out}")

    report_out = args.out.replace(".jsonl", "_report.json")
    with open(report_out, "w") as f:
        json.dump(report, f, indent=2)
    print(f"Report JSON: {report_out}")


if __name__ == "__main__":
    main()
