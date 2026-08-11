"""
FERTIG als Lehrer fuer den HSSLM-Sprecher (Baustein 2, Sprache-Haupt-Track).

graph_to_text.py rekonstruiert den ROHEN WT-103-Fenstertext hinter jedem
LiveStore-Record als Zielsatz -- verrauscht, Markup-Reste, keine Kontrolle
ueber Satzform. Dieses Modul ersetzt den Zielsatz durch eine DETERMINISTISCHE
FERTIG-Verbalisierung desselben Kausal-Records: (trigger, mechanism, outcome)
+ Citation werden ueber fertig.primitives' typisierte Register-API (RelationFamily)
in einen fluessigen, neutralen englischen Lehrersatz uebersetzt.

SCHEMA (identisch zu graph_to_text.py, kompatibel mit demselben Trainings-
Loader): {structure, text, citation, doc_coord, trigger, mechanism, outcome}
    structure : "<fact> trigger_key | mechanism | outcome_key <say>"
                (dieselbe linearize_structure()-Funktion, wiederverwendet)
    text      : die FERTIG-Verbalisierung statt des rohen Textfensters.

WICHTIGER BEFUND (gemessen, nicht verschwiegen): der p72_store_local wurde
mit einem AELTEREN, freieren Kausal-Extraktor gebaut (builder_v0) -- seine
Mechanismus-Strings ("is caused by", "destroyed", "killed", "expanded", ...)
sind ueberwiegend NICHT im neuen typisierten primitives.RELATIONS-Register
(das auf kanonische Praesens-Einwort-Verben wie "causes"/"reduces" zielt).
Gemessen an den 2047 Records des Stores: nur 190/2047 (9.3%) canonicalizen
auf eine bekannte RelationFamily (172 causal, 18 function) -- 1857/2047
(90.7%) bleiben "unknown" nach primitives.canonicalize_mechanism().

ENTSCHEIDUNG: kein Records-Verwurf. Ein Filter auf nur die 9.3% typisierten
Records wuerde 90% der Store-Daten fuer den Sprecher wegwerfen -- das ist
kein sinnvoller Kompromiss fuer einen Lehrer-Datensatz, dessen einziger
Job ist, dem Sprecher FLUESSIGERE Saetze als das rohe WT-103-Fenster zu
zeigen (nicht: nur perfekt schema-konforme Saetze zu zeigen). Stattdessen
zwei Verbalisierungs-Pfade:
  (a) TYPISIERT (canonicalize_mechanism() != None): Konnektor + Satzform
      nach RelationFamily gewaehlt (kausal/temporal/vergleichend/neutral,
      siehe _FAMILY_CONNECTORS unten) -- die "Konnektor-Vielfalt aus
      fertig", wie beauftragt.
  (b) UNKNOWN-FALLBACK: das rohe Mechanismus-Wort wird direkt als Verb
      eingesetzt (trigger + mechanism + outcome, exakt wie
      bridge_livegraph.verbalize_live_walk()'s Fallback-Zweig), mit
      DERSELBEN Opener-Rotation wie der kausale Pfad (die meisten unknown-
      Mechanismen SIND lose kausale/assoziative Aussagen aus rohem Text --
      "destroyed", "killed", "expanded" transportieren Kausalitaet, sind
      nur nicht in der kanonischen Registry). Jeder Satz traegt trotzdem
      seine Familie im Record (family: "unknown" vs. der RelationFamily-
      Name) -- die Trainingsdaten verschweigen die Typisierungsluecke
      nicht, sie machen sie sichtbar und messbar (siehe Report-Metriken).

DETERMINISMUS: kein Zufall im Verbalisierungspfad selbst (Konnektor-Wahl
ist eine reine Funktion von Record-Index i und Familie, kein RNG-Draw) --
gleicher Store + gleiche Iterationsreihenfolge (LiveStore.iter_records())
ergibt byte-identische Ausgabe. Ein --seed-Flag existiert fuer API-
Konsistenz mit graph_to_text.py und fuer zukuenftige stochastische
Konnektor-Varianz, wird aber von der aktuellen (deterministischen)
Opener-Rotation nicht konsumiert -- siehe Determinismus-Test.

Usage:
    OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
    python3 fertig_teacher.py --store results/p72_store_local \
        --out hsslm/data/fertig_teacher_pairs_sample.jsonl --limit 500
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

# hsslm depends on fertig, never the other way around (lead's explicit
# dependency direction) -- fertig is imported here, fertig never imports
# anything from hsslm/.
from fertig import primitives  # noqa: E402

from graph_to_text import linearize_structure, FACT_TOKEN, SAY_TOKEN, STRUCT_SEP  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


# ---------------------------------------------------------------------------
# Konnektor-Register nach RelationFamily (die "Konnektor-Vielfalt aus fertig")
# ---------------------------------------------------------------------------
# Jede Familie bekommt eine kleine Rotation von Oeffnern/Satzformen, analog
# zu pipeline.py's _CAUSE_OPENERS/_CONTRAST, aber ueber die TYPISIERTE
# RelationFamily statt ueber ein hartcodiertes POS/NEG-Verb-Set. i % len(...)
# waehlt deterministisch nach Record-Index -- keine Zufallsquelle.

_CAUSAL_OPENERS = ["As a result,", "Therefore,", "Consequently,", "This means"]
_TEMPORAL_OPENERS = ["Then,", "Afterward,", "Following that,", "Next,"]
_CONTRAST_OPENERS = ["However,", "On the other hand,", "Yet,"]
_ASSOC_OPENERS = ["Additionally,", "In this context,", "Relatedly,"]

# Familien, in denen der Kausal-Opener-Rhythmus zusaetzlich zum Grund-Satz
# eingesetzt wird (ansonsten steht der Grund-Satz allein, keine Konnektor-
# Verzierung -- z.B. bei einem einzelnen Vergleich "X is smaller than Y"
# klingt ein vorangestelltes "Therefore," falsch).
_OPENER_FAMILIES = {
    primitives.RelationFamily.CAUSAL: _CAUSAL_OPENERS,
    primitives.RelationFamily.TEMPORAL: _TEMPORAL_OPENERS,
    primitives.RelationFamily.ASSOCIATION: _ASSOC_OPENERS,
}

# Negations-Familie-Verben (Polaritaet -- fuer den kausalen Pfad, ob der
# Konnektor-Ton kontrastiv statt additiv klingen sollte).
_NEG_CANONICAL = {"prevents", "reduces", "decreases"}


_WT103_HYPHEN_ARTIFACT = re.compile(r"\s*@-@\s*")


def _humanize(key: str) -> str:
    """trigger_key/outcome_key -> lesbarer Satzbaustein. Zwei rein
    orthografische Bereinigungen, KEINE semantische Aenderung -- die
    Entitaet bleibt exakt das, was der Extraktor als Record gespeichert hat:
      1. WT-103's bekanntes Tokenisierungsartefakt @-@ (Bindestrich wurde
         beim Korpus-Tokenizer abgetrennt, z.B. "co @-@ operate") -> "-",
         gemessen in 169/2047 (8.3%) der p72_store_local-Records.
      2. Mehrfach-Whitespace -> ein Leerzeichen."""
    s = _WT103_HYPHEN_ARTIFACT.sub("-", str(key))
    return re.sub(r"\s+", " ", s).strip()


_PLURAL_TRIGGER_RE = re.compile(r"(?<!s)s$|(?<!s)ies$")
_SINGULAR_S_EXCEPTIONS = re.compile(r"(ss|us|is)$")


def _agrees_plural(trigger: str) -> bool:
    """Grobe (nicht linguistisch praezise) Subjekt-Numerus-Heuristik fuer
    den 'is/are caused by'-Sonderfall: 205/2047 (10.0%) der p72_store_local-
    Records tragen diesen fest an Singular gebundenen Mechanismus-String,
    aber trigger kann Plural sein ("its predecessors is caused by story").
    Regel: trigger endet auf -s/-ies (ohne -ss/-us/-is-Ausnahmen) -> Plural.
    Kein POS-Tagger, kein Anspruch auf Vollstaendigkeit -- dokumentierte
    Heuristik, deterministisch, faengt den haeufigsten Fehlerfall."""
    last = trigger.strip().split()[-1] if trigger.strip() else ""
    if _SINGULAR_S_EXCEPTIONS.search(last):
        return False
    return bool(_PLURAL_TRIGGER_RE.search(last))


def verbalize_record(rec: dict, index: int) -> tuple:
    """Ein LiveStore-Record -> (text, family_label).

    family_label ist entweder ein primitives.RelationFamily.value String
    (z.B. "causal") oder "unknown" -- immer im Output-Record sichtbar,
    siehe Modul-Docstring "die Trainingsdaten verschweigen die Typisierungs-
    luecke nicht"."""
    trigger = _humanize(rec["trigger"])
    outcome = _humanize(rec["outcome"])
    mechanism = _humanize(rec["mechanism"])

    if mechanism == "is caused by" and _agrees_plural(trigger):
        mechanism = "are caused by"

    canon = primitives.canonicalize_mechanism(mechanism)

    if canon is not None:
        spec = primitives.RELATIONS[canon]
        family = spec.family
        family_label = family.value
        verb_phrase = canon.replace("_", " ")
    else:
        family = None
        family_label = "unknown"
        verb_phrase = mechanism

    subject = trigger[0].upper() + trigger[1:] if trigger else trigger
    base_clause = f"{subject} {verb_phrase} {outcome}".strip()
    base_clause = re.sub(r"\s+", " ", base_clause)

    openers = _OPENER_FAMILIES.get(family) if family is not None else _CAUSAL_OPENERS
    if openers:
        opener = openers[index % len(openers)]
        if family == primitives.RelationFamily.CAUSAL and verb_phrase.split()[-1:] and \
                verb_phrase.split()[-1] in _NEG_CANONICAL and index % 3 == 0:
            # gelegentliche kontrastive Variante bei negativer Kausalpolaritaet
            opener = _CONTRAST_OPENERS[index % len(_CONTRAST_OPENERS)]
        clause_body = base_clause[0].lower() + base_clause[1:] if base_clause else base_clause
        text = f"{opener} {clause_body}."
    else:
        text = f"{base_clause}."

    return text, family_label


def build_teacher_pairs(store_dir: str, limit: int = None):
    """Yields dicts: structure, text, citation, doc_coord, trigger,
    mechanism, outcome, family -- ein Paar pro Store-Record (kein
    Coverage-Verlust wie bei graph_to_text.py's Tape-Rekonstruktion,
    da hier kein rohes Fenster gebraucht wird -- die Verbalisierung
    ist eine reine Funktion des Records selbst)."""
    sys.path.insert(0, os.path.join(REPO_ROOT, "src", "livecausal"))
    from store import LiveStore  # src/livecausal/store.py

    store = LiveStore(store_dir)
    all_records = list(store.iter_records())
    if limit is not None:
        all_records = all_records[:limit]

    n_emitted = 0
    for i, (sha, idx, rec) in enumerate(all_records):
        structure = linearize_structure(rec["trigger_key"], rec["mechanism"], rec["outcome_key"])
        text, family_label = verbalize_record(rec, i)
        n_emitted += 1
        yield {
            "structure": structure,
            "text": text,
            "citation": {"sha": sha, "idx": idx},
            "doc_coord": rec["doc_coord"],
            "trigger": rec["trigger"],
            "mechanism": rec["mechanism"],
            "outcome": rec["outcome"],
            "family": family_label,
        }
    print(f"[fertig_teacher] emitted {n_emitted} pairs from {len(all_records)} records")


def compute_report_stats(pairs):
    """Konnektor-Verteilung, Laengenverteilung, Familie-Verteilung --
    fuer den Report an team-lead."""
    from collections import Counter

    all_openers = (_CAUSAL_OPENERS + _TEMPORAL_OPENERS + _CONTRAST_OPENERS +
                   _ASSOC_OPENERS)
    family_counter = Counter(p["family"] for p in pairs)
    opener_counter = Counter()
    for p in pairs:
        text = p["text"]
        matched = next((o for o in all_openers if text.startswith(o + " ")), None)
        opener_counter[matched or "(kein Opener)"] += 1

    lens = [len(p["text"]) for p in pairs]
    word_lens = [len(p["text"].split()) for p in pairs]

    return {
        "n_pairs": len(pairs),
        "family_distribution": dict(family_counter.most_common()),
        "opener_distribution": dict(opener_counter.most_common()),
        "char_len": {
            "min": min(lens) if lens else 0,
            "max": max(lens) if lens else 0,
            "mean": sum(lens) / len(lens) if lens else 0,
        },
        "word_len": {
            "min": min(word_lens) if word_lens else 0,
            "max": max(word_lens) if word_lens else 0,
            "mean": sum(word_lens) / len(word_lens) if word_lens else 0,
        },
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--store", default=os.path.join(REPO_ROOT, "results", "p72_store_local"))
    ap.add_argument("--out", default=os.path.join(os.path.dirname(__file__), "fertig_teacher_pairs_sample.jsonl"))
    ap.add_argument("--limit", type=int, default=500)
    ap.add_argument("--seed", type=int, default=42,
                     help="reserviert fuer kuenftige stochastische Konnektor-"
                          "Varianz; die aktuelle Opener-Rotation ist deterministisch "
                          "nach Record-Index und konsumiert diesen Seed nicht.")
    args = ap.parse_args()

    pairs = list(build_teacher_pairs(args.store, limit=args.limit))

    with open(args.out, "w", encoding="utf-8") as f:
        for p in pairs:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")
    print(f"[fertig_teacher] wrote {len(pairs)} pairs to {args.out}")

    stats = compute_report_stats(pairs)
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
