"""
Converter: FOSS-KI's data/conceptnet_en_500k.json -> LIVE-CAUSAL LiveStore
segments (Phase 4, revival-probe Task 14).

Same target schema as convert_knowledge_full.py (trigger_key/outcome_key/
mechanism), but ConceptNet's 31 distinct relation types are NOT a uniform
fit for that schema the way knowledge_full.json's already-curated
(subject, relation, object) triplets were -- ConceptNet mixes genuine
world-fact relations (IsA, Causes, AtLocation, UsedFor, ...) with pure
LEXICAL relations between word-forms (Synonym, DerivedFrom, FormOf, ...)
that encode "these two strings are related as words," not "this entity
has this property/relationship." Converting the lexical relations into
mechanism-labeled causal-graph edges the same way as a fact would
misrepresent word-form kinship as world knowledge and would flood the
graph: DerivedFrom + Synonym + SimilarTo alone are 258,286 of the
500,000 records (51.7%) -- more than half the file, almost entirely
inflection/etymology/thesaurus data, not facts a "capital of France?"-
shaped question would ever need.

================================================================
RELATION MAPPING -- every one of the 31 relation types, decided and
documented, not silently dropped
================================================================
Measured distribution (full file, 500,000 records):

  145,348  DerivedFrom       EXCLUDED (lexical: etymology/word-derivation)
   91,276  Synonym           EXCLUDED (lexical: same-meaning word pairs)
   78,284  IsA               INCLUDED as mechanism="is_a"
   25,662  AtLocation        INCLUDED as mechanism="at_location"
   22,677  CapableOf         INCLUDED as mechanism="capable_of"
   22,190  RelatedTo         EXCLUDED (too unspecific to be a mechanism --
                              ConceptNet's own catch-all/weakest-typed
                              relation; "related to" names no actual
                              relationship an inference chain could reason
                              over, unlike every other relation here)
   21,272  SimilarTo         EXCLUDED (lexical: near-synonym pairs)
   16,801  Causes            INCLUDED as mechanism="causes" -- the single
                              most directly load-bearing relation for this
                              schema's trigger/outcome/mechanism shape
   13,951  Antonym           EXCLUDED (lexical: opposite-word pairs, not
                              a world relationship between the referents)
   12,702  MannerOf          INCLUDED as mechanism="manner_of" (a real,
                              if weak, "way of doing" relation between
                              two actions)
    9,380  PartOf            INCLUDED as mechanism="part_of"
    8,448  HasContext        EXCLUDED (lexical/pragmatic: usage-domain
                              tagging, e.g. "menorah HasContext judaism"
                              -- a topic label for the WORD, not a fact
                              about the entity)
    7,540  FormOf            EXCLUDED (lexical: inflected-form pairs,
                              e.g. plural/spelling variants)
    5,323  UsedFor           INCLUDED as mechanism="used_for"
    4,688  CausesDesire      INCLUDED as mechanism="causes_desire"
    3,520  HasPrerequisite   INCLUDED as mechanism="has_prerequisite"
    3,423  HasSubevent       INCLUDED as mechanism="has_subevent"
    2,173  DefinedAs         INCLUDED as mechanism="defined_as" -- despite
                              looking lexical, sampled records are
                              genuinely factual ("mercury DefinedAs
                              first planet from sun"), not word-pairs
    1,111  MotivatedByGoal   INCLUDED as mechanism="motivated_by_goal"
      830  HasProperty       INCLUDED as mechanism="has_property"
      589  HasA              INCLUDED as mechanism="has_a"
      508  Desires           INCLUDED as mechanism="desires"
      405  Entails           INCLUDED as mechanism="entails"
      390  NotDesires        INCLUDED as mechanism="not_desires"
      333  ReceivesAction    INCLUDED as mechanism="receives_action"
      288  HasLastSubevent   INCLUDED as mechanism="has_last_subevent"
      263  CreatedBy         INCLUDED as mechanism="created_by"
      262  HasFirstSubevent  INCLUDED as mechanism="has_first_subevent"
      256  DistinctFrom      EXCLUDED (lexical: "these are different
                              words/concepts" -- the antonym-shaped sibling
                              of DistinctFrom's own sampled records, e.g.
                              "dog DistinctFrom cat," name no positive
                              relationship, only an absence of one)
      105  MadeOf            INCLUDED as mechanism="made_of"
        2  LocatedNear       INCLUDED as mechanism="located_near"

INCLUDED total: 189,719 records (37.9% of the file).
EXCLUDED total: 310,281 records (62.1% of the file) -- the majority of
ConceptNet-en-500k is lexical/word-form data by volume, not world facts;
this is a real, measured property of the source file, not a converter
artifact.

Grain of salt on the INCLUDED/EXCLUDED line for RelatedTo and HasContext:
both are genuinely borderline (a future call could go the other way) --
flagged here explicitly rather than presented as an obviously-correct
cut. DefinedAs's inclusion despite its lexical-sounding name is the one
place this mapping goes against a name-based first impression, backed by
sampling three real records before deciding (see docstring above).

Every other mapping decision follows the same rule this project has used
throughout: sample the actual data before excluding or including a
category, never go by the relation name alone.
"""
import json
import os
import sys
import time

sys.path.insert(0, os.path.expanduser("~"))
from livecausal_bridge import LiveStore  # noqa: E402

SEGMENT_SIZE = 50
SOURCE_TAG = "foss-ki:conceptnet_en_500k.json"

INCLUDED_RELATIONS = {
    "IsA": "is_a",
    "AtLocation": "at_location",
    "CapableOf": "capable_of",
    "Causes": "causes",
    "MannerOf": "manner_of",
    "PartOf": "part_of",
    "UsedFor": "used_for",
    "CausesDesire": "causes_desire",
    "HasPrerequisite": "has_prerequisite",
    "HasSubevent": "has_subevent",
    "DefinedAs": "defined_as",
    "MotivatedByGoal": "motivated_by_goal",
    "HasProperty": "has_property",
    "HasA": "has_a",
    "Desires": "desires",
    "Entails": "entails",
    "NotDesires": "not_desires",
    "ReceivesAction": "receives_action",
    "HasLastSubevent": "has_last_subevent",
    "CreatedBy": "created_by",
    "HasFirstSubevent": "has_first_subevent",
    "MadeOf": "made_of",
    "LocatedNear": "located_near",
}

EXCLUDED_RELATIONS = {
    "DerivedFrom", "Synonym", "SimilarTo", "RelatedTo", "Antonym",
    "HasContext", "FormOf", "DistinctFrom",
}


def _normalize(s):
    return s.strip().lower()


def load_conceptnet(path):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data["relations"], data.get("version"), data.get("total")


def relation_to_record(rel, source_index):
    r, s, o, w = rel["r"], rel["s"], rel["o"], rel.get("w", 1.0)
    mechanism = INCLUDED_RELATIONS[r]
    return {
        "trigger": s,
        "mechanism": mechanism,
        "outcome": o,
        "trigger_key": _normalize(s),
        "outcome_key": _normalize(o),
        "doc_coord": "conceptnet_en_500k:{}".format(source_index),
        "evidence_count": 1,
        "use_count": 0,
        "meta": {
            "source": SOURCE_TAG,
            "source_index": source_index,
            "conceptnet_relation": r,
            "conceptnet_weight": w,
        },
    }


def convert(conceptnet_path, store_dir, segment_size=SEGMENT_SIZE, dry_run=False,
            limit=None):
    t_load_start = time.perf_counter()
    relations, version, total = load_conceptnet(conceptnet_path)
    t_load = time.perf_counter() - t_load_start

    if total is not None and len(relations) != total:
        raise ValueError(
            "conceptnet file's 'total' field ({}) does not match actual "
            "relations list length ({}) -- refusing to convert against a "
            "self-inconsistent source file.".format(total, len(relations))
        )

    if limit is not None:
        relations = relations[:limit]

    included = [
        (i, rel) for i, rel in enumerate(relations)
        if rel["r"] in INCLUDED_RELATIONS
    ]
    n_excluded = len(relations) - len(included)

    store = LiveStore(store_dir)
    segment_shas = []
    n_records = 0

    t_convert_start = time.perf_counter()
    batch = []
    for source_index, rel in included:
        batch.append(relation_to_record(rel, source_index))
        if len(batch) >= segment_size:
            n_records += len(batch)
            if not dry_run:
                sha = store.append_segment(batch)
                segment_shas.append(sha)
            batch = []
    if batch:
        n_records += len(batch)
        if not dry_run:
            sha = store.append_segment(batch)
            segment_shas.append(sha)
    t_convert = time.perf_counter() - t_convert_start

    return {
        "source_path": conceptnet_path,
        "source_version": version,
        "source_total_records": len(relations),
        "n_included_records": len(included),
        "n_excluded_records": n_excluded,
        "included_relation_types": sorted(INCLUDED_RELATIONS.keys()),
        "excluded_relation_types": sorted(EXCLUDED_RELATIONS),
        "segment_size": segment_size,
        "n_segments": (len(included) + segment_size - 1) // segment_size if included else 0,
        "n_records_converted": n_records,
        "n_segments_written": len(segment_shas),
        "store_dir": store_dir,
        "dry_run": dry_run,
        "load_seconds": round(t_load, 3),
        "convert_seconds": round(t_convert, 3),
        "records_per_second": round(n_records / t_convert, 1) if t_convert > 0 else None,
    }


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--source", default=os.path.expanduser(
        "~/mac_offload/desktop/foss-ki/data/conceptnet_en_500k.json"))
    ap.add_argument("--store-dir", required=True)
    ap.add_argument("--segment-size", type=int, default=SEGMENT_SIZE)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--limit", type=int, default=None,
                     help="Only process the first N relations (for smoke tests).")
    args = ap.parse_args()

    result = convert(args.source, args.store_dir, args.segment_size, args.dry_run,
                      args.limit)
    print(json.dumps(result, indent=2))
