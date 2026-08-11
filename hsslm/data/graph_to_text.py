"""
Struktur->Satz Trainingsdatensatz-Generator (Baustein 1, Sprache-Haupt-Track).

Liest den LiveStore (src/livecausal/store.py) unter results/p72_store_local
und emittiert (structure, text) Trainingspaare fuer HSSLM:

    structure: linearisierte Struktur, z.B.
        "<fact> growth | is caused by | the mumbai harbour <say>"
        (trigger_key | mechanism | outcome_key, mit den ZWEI Struktur-
        Markern <fact>/<say>)
    text: der Originalsatz (verbatim WT-103-Text), aus dem der Builder das
        Triplet extrahiert hat

WICHTIGER BEFUND (dokumentiert, nicht verschwiegen): der LiveStore-Record
selbst enthaelt KEIN 'text'-Feld mit dem Originalsatz -- store.py/
builder_run.py speichern nur trigger/mechanism/outcome/trigger_key/
outcome_key/doc_coord/evidence_count/use_count/meta. Der Originalsatz
(window_text) wird im Builder-Loop zwar berechnet (builder_run.py:574,
curator_yield_run.iter_windows Zeile 250), aber NIE in den Record
geschrieben -- nur doc_coord (die Tape-Startposition) bleibt als
Herkunftskoordinate.

Der Text wird deshalb hier REKONSTRUIERT: curator_yield_run.iter_windows
ist deterministisch (fester seed=42, Organism.step_gated hat keine
Zufallsquelle ausser dem Seed) und mit der EXAKTEN Cadence aus dem
p72-Register (results/p72_run1.json: seed=42, source=wt103, chunks=3000,
window_tokens=128, d_model=128, batch=8, chunk_size=64, q=0.75, window=500,
min_window=100, ignition_chunks=100) reproduziert sie exakt dieselbe
tape_pos -> window_text Zuordnung wie der urspruengliche p72-Build. Verifiziert
gegen 2 echte Records (doc_coord=12160: trigger="growth" + outcome="the
Mumbai harbour" erscheinen wortwoertlich im rekonstruierten Fenstertext;
doc_coord=13760: Fenstertext beginnt exakt an der erwarteten Stelle).

FORMAT-ENTSCHEIDUNG (Struktur-Tokens im Vokabular):
Das Organism-Vokabular (length_extrap_v2.build_vocab) hat KEINEN
eingebauten Mechanismus fuer Spezial-Tokens -- es sind die 5000
haeufigsten Woerter des Trainingstexts plus unk (id=5000) und mask
(id=5001), macht 5002 IDs total.

Abwaegung:
  (a) Vokabular um 2-4 IDs ERWEITERN (<fact>, <say>, ggf. weitere) ->
      total_ids = 5002 + n_new. Sauber: keine Kollision mit echten
      Woertern, kein semantisch aufgeladenes Wort wird zweckentfremdet,
      LM-Head bleibt weight-tied (LMHead nimmt nur vocab_size entgegen,
      siehe hsslm/neural/lm_head.py -- eine groessere Embedding-Tabelle
      aendert daran nichts strukturell). Nachteil: die neuen IDs sind bei
      Trainingsbeginn UNTRAINIERT (Embeddings zufaellig init), brauchen
      also selbst ein paar Gradientenschritte um sinnvolle Repraesentation
      zu bekommen.
  (b) Reservierte SELTENE Woerter als Marker missbrauchen (z.B. die beiden
      seltensten Vokabeleintraege am Rand von VOCAB_MAX). Nachteil: nicht
      wirklich reserviert -- falls das Wort im echten WT-103-Text an
      anderer Stelle vorkommt (auch selten heisst nicht: nie), ueberlagern
      sich Bedeutung und Struktur-Signal; unklar, sprachlich unsauber,
      und macht die Trainingsdaten fuer einen menschlichen/programmatischen
      Leser missverstaendlich.

ENTSCHEIDUNG: (a), Vokabular-Erweiterung um genau 2 IDs (<fact>=5002,
<say>=5003, total_ids=5004). Begruendung: sauberer, keine Doppeldeutigkeit,
Kosten (ein paar Gradientenschritte zum Warmlaufen zweier Embeddings) sind
bei einem d256-Modell mit tausenden Trainingsschritten vernachlaessigbar
gegen die semantische Unsauberkeit von (b). Reserviert bewusst NUR 2 Marker
(nicht 4) -- <fact> markiert Strukturanfang, <say> den Uebergang
Struktur->Zielsatz; trigger/mechanism/outcome werden mit dem literalen
Trennzeichen " | " (Leerzeichen-Pipe-Leerzeichen) getrennt statt mit
weiteren Spezial-Tokens, da " | " selbst aus bereits-vokabularisierten
Zeichen/Woertern besteht (kein Vokabular-Wachstum noetig) und die
Struktur eindeutig macht (drei Segmente durch das feste Trennzeichen).

Output: JSONL, ein Paar pro Zeile:
    {"structure": str, "text": str, "citation": {"sha": str, "idx": int},
     "doc_coord": int, "trigger": str, "mechanism": str, "outcome": str}

Usage:
    OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
    HF_DATASETS_OFFLINE=1 HF_HUB_OFFLINE=1 \
    python3 graph_to_text.py --store results/p72_store_local \
        --out hsslm/data/graph_to_text_pairs.jsonl
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

# Struktur-Marker (Vokabular-Erweiterung, siehe Docstring-Entscheidung oben).
FACT_TOKEN = "<fact>"
SAY_TOKEN = "<say>"
STRUCT_SEP = " | "

# Exakte p72-Cadence (aus results/p72_run1.json, unveraendert reproduziert).
P72_CADENCE = dict(
    substrate="wt103",
    chunks=3000,
    window_tokens=128,
    d_model=128,
    batch=8,
    chunk_size=64,
    seed=42,
    q=0.75,
    window=500,
    min_window=100,
    ignition_chunks=100,
)


def linearize_structure(trigger_key: str, mechanism: str, outcome_key: str) -> str:
    """<fact> trigger_key | mechanism | outcome_key <say>"""
    body = STRUCT_SEP.join([trigger_key, mechanism, outcome_key])
    return f"{FACT_TOKEN} {body} {SAY_TOKEN}"


def reconstruct_tape_pos_to_text(cadence: dict, needed_positions: set) -> dict:
    """Run curator_yield_run.iter_windows ONCE with the exact p72 cadence and
    collect window_text for every tape_pos actually referenced by the store's
    records (doc_coord). Deterministic (fixed seed) -- reproduces exactly the
    tape_pos -> window_text mapping the original p72 build saw.

    Returns {tape_pos: window_text} for whichever needed_positions were
    actually yielded within `chunks` chunks (some late-store doc_coords may
    exceed what `chunks` covers if the store was built with a different
    windows-per-segment cutoff; report coverage, don't assume completeness).
    """
    import curator_yield_run as cyr

    found = {}
    remaining = set(needed_positions)
    gen = cyr.iter_windows(**cadence)
    for tape_pos, text, surprise, gated in gen:
        if tape_pos in remaining:
            found[tape_pos] = text
            remaining.discard(tape_pos)
        if not remaining:
            break
    return found


def build_pairs(store_dir: str, cadence: dict = P72_CADENCE):
    """Yields dicts: structure, text, citation, doc_coord, trigger,
    mechanism, outcome -- one per store record whose window_text could be
    reconstructed. Records whose doc_coord fell outside the reconstruction
    run's coverage are skipped (reported in the summary, not silently
    dropped from the count)."""
    sys.path.insert(0, os.path.join(REPO_ROOT, "src", "livecausal"))
    from store import LiveStore

    store = LiveStore(store_dir)
    all_records = list(store.iter_records())
    needed_positions = {rec["doc_coord"] for _, _, rec in all_records}

    print(f"[graph_to_text] store records: {len(all_records)}  "
          f"unique doc_coord positions needed: {len(needed_positions)}")
    print(f"[graph_to_text] reconstructing window_text via curator_yield_run.iter_windows "
          f"(cadence: {cadence})...")

    tape_pos_to_text = reconstruct_tape_pos_to_text(cadence, needed_positions)
    n_found = len(tape_pos_to_text)
    n_missing = len(needed_positions) - n_found
    print(f"[graph_to_text] reconstructed {n_found}/{len(needed_positions)} positions "
          f"({n_missing} not covered by chunks={cadence['chunks']})")

    n_emitted, n_skipped = 0, 0
    for sha, idx, rec in all_records:
        doc_coord = rec["doc_coord"]
        text = tape_pos_to_text.get(doc_coord)
        if text is None:
            n_skipped += 1
            continue
        structure = linearize_structure(rec["trigger_key"], rec["mechanism"], rec["outcome_key"])
        n_emitted += 1
        yield {
            "structure": structure,
            "text": text,
            "citation": {"sha": sha, "idx": idx},
            "doc_coord": doc_coord,
            "trigger": rec["trigger"],
            "mechanism": rec["mechanism"],
            "outcome": rec["outcome"],
        }
    print(f"[graph_to_text] emitted {n_emitted} pairs, skipped {n_skipped} "
          f"(doc_coord outside reconstruction coverage)")


def compute_oov_stats(pairs, stoi, unk_id):
    """OOV rate against the 5000-word Organism vocabulary: what fraction of
    TEXT tokens (the harder side -- raw WT-103 prose) fall outside the
    vocabulary vs STRUCTURE tokens (trigger/mechanism/outcome keys, which
    the vocabulary was itself partly built from the same corpus, so should
    have somewhat better coverage)."""
    import re
    word_re = re.compile(r"[a-zA-Z]+")

    def oov_rate(s):
        words = word_re.findall(s.lower())
        if not words:
            return 0.0, 0
        n_oov = sum(1 for w in words if stoi.get(w, unk_id) == unk_id)
        return n_oov / len(words), len(words)

    text_oov_rates, text_lens = [], []
    struct_oov_rates, struct_lens = [], []
    for p in pairs:
        r, n = oov_rate(p["text"])
        text_oov_rates.append(r)
        text_lens.append(n)
        r2, n2 = oov_rate(p["structure"])
        struct_oov_rates.append(r2)
        struct_lens.append(n2)

    def summarize(rates, lens):
        if not rates:
            return {}
        return {
            "mean_oov_rate": sum(rates) / len(rates),
            "median_oov_rate": sorted(rates)[len(rates) // 2],
            "pct_pairs_zero_oov": sum(1 for r in rates if r == 0.0) / len(rates),
            "mean_word_len": sum(lens) / len(lens),
            "min_word_len": min(lens),
            "max_word_len": max(lens),
        }

    return {
        "text": summarize(text_oov_rates, text_lens),
        "structure": summarize(struct_oov_rates, struct_lens),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--store", default=os.path.join(REPO_ROOT, "results", "p72_store_local"))
    ap.add_argument("--out", default=os.path.join(os.path.dirname(__file__), "graph_to_text_pairs.jsonl"))
    ap.add_argument("--chunks", type=int, default=P72_CADENCE["chunks"],
                     help="override reconstruction run length (default: exact p72 cadence)")
    args = ap.parse_args()

    cadence = dict(P72_CADENCE)
    cadence["chunks"] = args.chunks

    pairs = list(build_pairs(args.store, cadence))

    with open(args.out, "w", encoding="utf-8") as f:
        for p in pairs:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")
    print(f"[graph_to_text] wrote {len(pairs)} pairs to {args.out}")

    # OOV stats against the Organism vocabulary (built from WT-103 itself,
    # so this measures whether these SPECIFIC extracted-triplet pairs stay
    # in-vocabulary, not a general WT-103 OOV rate).
    from length_extrap_v2 import build_vocab

    print("\n[graph_to_text] building Organism vocabulary for OOV measurement "
          "(same text source, length_extrap_v2.build_vocab)...")
    from curator_yield_run import HFStreamWithText  # noqa: F401  (network guard check below)
    import portable_organism as po

    vocab, stoi, unk, mask, val_ids = po.get_vocab()
    print(f"    vocab words: {len(vocab)}  unk={unk}  mask={mask}")

    stats = compute_oov_stats(pairs, stoi, unk)

    print("\n" + "=" * 70)
    print("STATISTIK")
    print("=" * 70)
    print(f"n Paare: {len(pairs)}")
    if pairs:
        text_lens = [len(p["text"]) for p in pairs]
        struct_lens = [len(p["structure"]) for p in pairs]
        print(f"text char-len: min={min(text_lens)} max={max(text_lens)} "
              f"mean={sum(text_lens)/len(text_lens):.1f}")
        print(f"structure char-len: min={min(struct_lens)} max={max(struct_lens)} "
              f"mean={sum(struct_lens)/len(struct_lens):.1f}")
    print(f"\nOOV gegen 5000er-Organism-Vokabular:")
    print(f"  TEXT (WT-103-Prosa, Zielsatz):")
    for k, v in stats["text"].items():
        print(f"    {k}: {v:.4f}" if isinstance(v, float) else f"    {k}: {v}")
    print(f"  STRUCTURE (trigger_key | mechanism | outcome_key):")
    for k, v in stats["structure"].items():
        print(f"    {k}: {v:.4f}" if isinstance(v, float) else f"    {k}: {v}")

    stats_out = args.out.replace(".jsonl", "_stats.json")
    with open(stats_out, "w") as f:
        json.dump({
            "n_pairs": len(pairs),
            "cadence": cadence,
            "oov_stats": stats,
            "struct_tokens": {"fact": FACT_TOKEN, "say": SAY_TOKEN, "sep": STRUCT_SEP},
        }, f, indent=2)
    print(f"\nStats JSON: {stats_out}")


if __name__ == "__main__":
    main()
