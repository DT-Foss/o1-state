"""
Lehrer v3: die IR-verifizierte FERTIG-Form-Engine als Satz-Generator fuer
den HSSLM-Sprecher (Sprache-Haupt-Track, Anschluss an fertig_teacher.py [v2]
und graph_to_text_v3.py [Rohtext-Spur]).

fertig_teacher.py (v2) verbalisiert jeden LiveStore-Record mit einem
handgeschriebenen Konnektor-Register (Konnektor-Wahl nach RelationFamily,
siehe dort). teacher_v3 ersetzt das durch ein TRAINIERTES Modell
(fertig.hsslm.model.HSSLMC, "HSSLM-C", 6.3M, Mamba/S6-Familie, aus dem
FERTIG-Resync) UND behaelt trotzdem die Wahrheitsgarantie: jede generierte
Variante wird gegen den Plan (trigger, mechanism, outcome) rueckverifiziert
(Utterance-IR-Muster aus fertig.form_engine.speak_with_engine, hier direkt
auf unsere LiveStore-Records angewendet statt auf einen .causal-Graph-Walk)
-- nur FREIGEGEBENE (verified=True) Varianten werden ins Paar-Schema
uebernommen.

PIPELINE PRO RECORD:
  1. PLAN   : (trigger, mechanism, outcome) direkt aus dem Record -- kein
              .causal-Graph noetig, das ist der Unterschied zu
              fertig.form_engine.speak_with_engine (dort: Graph-Walk).
  2. PROMPT : deterministischer Basissatz "Subjekt Verb Objekt." (identisch
              zum Prompt-Aufbau in speak_with_engine, siehe dessen
              verbalize()-Aufruf) -- der Prompt selbst ist bereits ein
              gueltiger, belegter Satz; die Form-Engine soll ihn FLUESSIGER
              umformulieren, nicht neu erfinden.
  3. VARIANTEN: FormEngine.variants(prompt, n=3) -- HSSLM-C generiert n
              Fortsetzungen, deterministisch (fester Seed PRO Variante,
              seed=0..n-1, intern in FormEngine.variants via
              torch.manual_seed(seed) VOR jeder Generierung -- siehe dort).
  4. IR-VERIFIKATION: eine Variante gilt als FREIGEGEBEN, wenn sowohl
              trigger ALS AUCH outcome (Content-Woerter, >3 Zeichen,
              case-insensitive) im generierten Text vorkommen -- das ist
              dieselbe Grundregel wie speak_with_engine's
              `all(u.obj.lower() in v.lower() for u in res.utterances)`,
              hier auf BEIDE Record-Enden (nicht nur das Objekt) angewendet,
              weil unser Plan kein Graph-Walk mit mehreren Utterances ist,
              sondern ein einzelnes Triplet -- Subjekt UND Objekt muessen
              belegt sein, sonst ist die Variante keine Aussage ueber
              diesen Record mehr.
  5. AUSWAHL: unter den freigegebenen Varianten gewinnt die KUERZESTE
              (deterministische Regel, dokumentiert -- kuerzer heisst hier
              NICHT "schlechter": die Form-Engine neigt dazu, freigegebene
              Kurzsaetze praezise zu halten, waehrend laengere Fortsetzungen
              mehr Gelegenheit haben, vom Plan abzudriften, ohne dass der
              simple Contains-Check das noch fassen wuerde). Bei
              Gleichstand: alphabetisch (stabiler Tie-Break, keine
              versteckte Praeferenz).

SCHEMA: identisch zu fertig_teacher.py (v2) --
    {structure, text, citation, doc_coord, trigger, mechanism, outcome}
plus "family" (wie v2) und "n_variants_verified" (wie viele der n
generierten Varianten die IR-Pruefung bestanden -- die Rohzahl hinter der
Freigabe-Quote im Report).

DETERMINISMUS: FormEngine.variants() selbst ist deterministisch (fester
Seed 0..n-1 PRO Aufruf, torch.manual_seed innen). Records ohne JEDE
freigegebene Variante werden NICHT ins Paar-Schema aufgenommen (gezaehlt,
nicht stillschweigend mit einer unbelegten Variante gefuellt) -- die
33%-Verwurfsrate aus dem manuellen Live-Test ist die Erwartung, die
tatsaechliche Zahl auf unseren Records ist der Kern des Reports.

Usage:
    OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
    python3 teacher_v3.py --store results/p72_store_local \
        --out hsslm/data/teacher_v3_pairs_sample.jsonl --limit 500
"""
import argparse
import json
import os
import re
import sys

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(REPO_ROOT, "src"))
sys.path.insert(0, os.path.join(REPO_ROOT, "fertig"))  # for `fertig` package (repo/fertig/fertig/*)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from graph_to_text import linearize_structure  # noqa: E402
from fertig_teacher import _humanize  # noqa: E402 -- reuse the @-@ / whitespace cleanup, no duplication
from fertig import primitives  # noqa: E402

_WORD_RE = re.compile(r"[a-zA-Z]+")


def _det(noun: str) -> str:
    """Wie fertig.pipeline._det: kein Artikel doppelt einsetzen, wenn das
    Nomen selbst schon einen traegt (grobe Heuristik, gleiche Regel wie die
    Form-Engine-eigene Prompt-Erzeugung -- Konsistenz zum Trainingskorpus-Stil)."""
    n = noun.strip()
    if not n:
        return n
    first = n.split()[0].lower()
    if first in ("the", "a", "an", "this", "that", "these", "those"):
        return n
    return n


def build_prompt(trigger: str, mechanism: str, outcome: str) -> str:
    """Deterministischer Basissatz -- identisch im Stil zu
    fertig.form_engine.speak_with_engine's verbalize()-basiertem Prompt
    (erster Hop: 'Subjekt Verb Objekt.')."""
    trigger = _humanize(trigger)
    outcome = _humanize(outcome)
    mechanism = _humanize(mechanism)
    subj = _det(trigger)
    subj = subj[0].upper() + subj[1:] if subj else subj
    obj = _det(outcome)
    return f"{subj} {mechanism} {obj}.".strip()


def content_words(text: str, min_len: int = 4):
    return [w for w in _WORD_RE.findall(text.lower()) if len(w) >= min_len]


def verify_variant(trigger: str, outcome: str, variant_text: str) -> bool:
    """IR-Verifikation: trigger UND outcome muessen als Content-Woerter
    (>=4 Zeichen) im generierten Text wiederzufinden sein -- dieselbe
    Grundregel wie speak_with_engine's Objekt-Containment-Check, hier auf
    beide Record-Enden angewendet (siehe Modul-Docstring Schritt 4)."""
    text_lower = variant_text.lower()
    trig_words = content_words(trigger)
    out_words = content_words(outcome)
    if not trig_words or not out_words:
        return False  # nichts Pruefbares -- kein Beleg moeglich, kein Pass
    trig_ok = any(w in text_lower for w in trig_words)
    out_ok = any(w in text_lower for w in out_words)
    return trig_ok and out_ok


def is_prompt_echo_only(prompt: str, variant_text: str) -> bool:
    """Diagnose-Metrik (NICHT Teil der IR-Verifikation selbst -- die bleibt
    exakt die simple Containment-Regel aus speak_with_engine, siehe
    verify_variant). Misst separat: enthaelt die Variante nach dem
    woertlichen Prompt-Praefix noch WEITEREN Content, oder ist sie nur der
    Prompt selbst plus Rauschen ohne neue Content-Woerter?

    Grund fuer diese zusaetzliche Messung: der Prompt selbst ist bereits
    ein vollstaendiger, belegter Satz (Subjekt Verb Objekt.) -- er enthaelt
    trigger UND outcome per Konstruktion. Eine Form-Engine, die den Prompt
    nur echot und danach Kauderwelsch anhaengt, wuerde verify_variant()
    TRIVIAL bestehen, ohne irgendeinen Formulierungs-Mehrwert zu liefern.
    Diese Metrik faengt genau das (gemessen, nicht gefixt -- siehe Report:
    'wenn die Engine strukturell scheitert, ist das ein Report-Ergebnis')."""
    if not variant_text.lower().startswith(prompt.lower().rstrip(".").rstrip()):
        return False  # kein Echo -- die Variante wich schon frueh vom Prompt ab
    remainder = variant_text[len(prompt):]
    remainder_words = set(content_words(remainder))
    prompt_words = set(content_words(prompt))
    new_words = remainder_words - prompt_words
    return len(new_words) == 0


def choose_best(verified_texts):
    """Deterministische Auswahl unter freigegebenen Varianten: kuerzeste
    zuerst, alphabetisch als Tie-Break (siehe Modul-Docstring Schritt 5)."""
    return sorted(verified_texts, key=lambda t: (len(t), t))[0]


def process_record(engine, rec: dict, n: int = 3):
    """Ein LiveStore-Record -> dict mit text/family/n_variants_verified,
    oder None wenn keine Variante freigegeben wurde (Record wird verworfen,
    siehe Modul-Docstring 'Determinismus')."""
    trigger = _humanize(rec["trigger"])
    outcome = _humanize(rec["outcome"])
    mechanism = _humanize(rec["mechanism"])

    canon = primitives.canonicalize_mechanism(mechanism)
    family_label = primitives.RELATIONS[canon].family.value if canon else "unknown"

    prompt = build_prompt(trigger, mechanism, outcome)
    variants = engine.variants(prompt, n=n)

    verified_texts = [v for v in variants if verify_variant(trigger, outcome, v)]
    rejected = [v for v in variants if v not in verified_texts]

    if not verified_texts:
        return None, {"family": family_label, "prompt": prompt,
                       "rejected": rejected, "trigger": trigger, "outcome": outcome}

    best = choose_best(verified_texts)
    return {
        "text": best,
        "family": family_label,
        "n_variants_verified": len(verified_texts),
        "n_variants_total": len(variants),
        "prompt_echo_only": is_prompt_echo_only(prompt, best),
    }, None


def build_teacher_v3_pairs(store_dir: str, limit: int = None, n_variants: int = 3):
    """Yields (pair_dict_or_None, reject_info_or_None) tuples -- ein Eintrag
    pro Store-Record. pair_dict ist None wenn der Record verworfen wurde
    (reject_info traegt dann family/prompt/rejected-Varianten fuer den
    Report). Beide sind niemals gleichzeitig gesetzt."""
    sys.path.insert(0, os.path.join(REPO_ROOT, "src", "livecausal"))
    from store import LiveStore
    from fertig.form_engine import FormEngine

    engine = FormEngine()
    if not engine.ready:
        raise RuntimeError(
            "FormEngine not ready -- hsslm_form.pt missing? "
            "(expected under fertig/data/hsslm_form.pt)"
        )

    store = LiveStore(store_dir)
    all_records = list(store.iter_records())
    if limit is not None:
        all_records = all_records[:limit]

    n_verified = n_rejected = 0
    for sha, idx, rec in all_records:
        result, reject_info = process_record(engine, rec, n=n_variants)
        if result is None:
            n_rejected += 1
            yield None, {**reject_info, "citation": {"sha": sha, "idx": idx},
                          "doc_coord": rec["doc_coord"]}
            continue
        n_verified += 1
        structure = linearize_structure(rec["trigger_key"], rec["mechanism"], rec["outcome_key"])
        pair = {
            "structure": structure,
            "text": result["text"],
            "citation": {"sha": sha, "idx": idx},
            "doc_coord": rec["doc_coord"],
            "trigger": rec["trigger"],
            "mechanism": rec["mechanism"],
            "outcome": rec["outcome"],
            "family": result["family"],
            "n_variants_verified": result["n_variants_verified"],
            "n_variants_total": result["n_variants_total"],
            "prompt_echo_only": result["prompt_echo_only"],
        }
        yield pair, None
    print(f"[teacher_v3] {n_verified} verified, {n_rejected} rejected "
          f"(of {len(all_records)} records)", file=sys.stderr)


def compute_report_stats(pairs, rejects):
    from collections import Counter

    n_total = len(pairs) + len(rejects)
    fam_verified = Counter(p["family"] for p in pairs)
    fam_rejected = Counter(r["family"] for r in rejects)
    fam_all = set(fam_verified) | set(fam_rejected)

    per_family = {}
    for fam in fam_all:
        v = fam_verified.get(fam, 0)
        r = fam_rejected.get(fam, 0)
        per_family[fam] = {
            "verified": v, "rejected": r, "total": v + r,
            "acceptance_rate": v / (v + r) if (v + r) else 0.0,
        }

    lens = [len(p["text"]) for p in pairs]
    word_lens = [len(p["text"].split()) for p in pairs]
    distinct_texts = len(set(p["text"] for p in pairs))
    n_echo_only = sum(1 for p in pairs if p.get("prompt_echo_only"))

    return {
        "n_total": n_total,
        "n_verified": len(pairs),
        "n_rejected": len(rejects),
        "acceptance_rate": len(pairs) / n_total if n_total else 0.0,
        "per_family": per_family,
        "char_len": {
            "min": min(lens) if lens else 0, "max": max(lens) if lens else 0,
            "mean": sum(lens) / len(lens) if lens else 0,
        },
        "word_len": {
            "min": min(word_lens) if word_lens else 0,
            "max": max(word_lens) if word_lens else 0,
            "mean": sum(word_lens) / len(word_lens) if word_lens else 0,
        },
        "distinct_texts": distinct_texts,
        "distinct_ratio": distinct_texts / len(pairs) if pairs else 0.0,
        "n_prompt_echo_only": n_echo_only,
        "prompt_echo_only_ratio": n_echo_only / len(pairs) if pairs else 0.0,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--store", default=os.path.join(REPO_ROOT, "results", "p72_store_local"))
    ap.add_argument("--out", default=os.path.join(os.path.dirname(__file__), "teacher_v3_pairs_sample.jsonl"))
    ap.add_argument("--limit", type=int, default=500)
    ap.add_argument("--n-variants", type=int, default=3)
    args = ap.parse_args()

    pairs, rejects = [], []
    for pair, reject in build_teacher_v3_pairs(args.store, limit=args.limit, n_variants=args.n_variants):
        if pair is not None:
            pairs.append(pair)
        else:
            rejects.append(reject)

    with open(args.out, "w", encoding="utf-8") as f:
        for p in pairs:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")
    print(f"[teacher_v3] wrote {len(pairs)} pairs to {args.out}")

    rejects_out = args.out.replace(".jsonl", "_rejected.jsonl")
    with open(rejects_out, "w", encoding="utf-8") as f:
        for r in rejects:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"[teacher_v3] wrote {len(rejects)} rejects to {rejects_out}")

    stats = compute_report_stats(pairs, rejects)
    print("\n" + "=" * 70)
    print("STATISTIK")
    print("=" * 70)
    print(json.dumps(stats, indent=2))

    stats_out = args.out.replace(".jsonl", "_stats.json")
    with open(stats_out, "w") as f:
        json.dump(stats, f, indent=2)
    print(f"\nStats JSON: {stats_out}")


if __name__ == "__main__":
    main()
