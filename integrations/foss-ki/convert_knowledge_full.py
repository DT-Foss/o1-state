"""
Converter: FOSS-KI's data/knowledge_full.json -> LIVE-CAUSAL LiveStore segments.

Maps FOSS-KI's flat (subject, relation, object) KB triplets onto the
livecausal_bridge record schema (trigger_key/outcome_key/mechanism), and
seals them into content-addressed segments via LiveStore.append_segment.

Schema mapping (documented, not guessed):
  trigger_key   = subject   (lowercased/stripped, matching FOSS-KI's own
                              KnowledgeStore._normalize discipline so a
                              later cross-query against the raw KB using
                              the same normalization stays consistent)
  outcome_key   = object    (same normalization)
  mechanism     = relation  (kept verbatim, NOT normalized -- mechanism is
                              free text in the livecausal schema, and
                              FOSS-KI's relation strings like "capital",
                              "author", "is_a" are exactly the kind of
                              short mechanism label the schema expects)
  trigger       = subject   (original casing preserved, for display)
  outcome       = object    (original casing preserved, for display)
  doc_coord     = "knowledge_full:<line_index>" -- a synthetic per-triplet
                  coordinate. knowledge_full.json has no real document
                  structure (it's a flat generated list, not extracted
                  from a corpus), so there is no genuine document-id to
                  carry. Using the source-list index as a string-prefixed
                  doc_coord means evidence.py's _evidence_key_for_record
                  will (if ever run over this data) resolve every triplet
                  as its OWN distinct evidence source -- honest, since
                  each triplet in this bootstrap file genuinely is an
                  independently-authored fact, not a sentence extracted
                  from shared surrounding context the way a corpus
                  extraction's doc_coord int would be.
  evidence_count = 1   -- seed value; the real evidence_count this schema
                  expects is a LEDGER fold (evidence.py), not a static
                  field baked into the record at conversion time. Setting
                  1 here is a placeholder so the field is present and
                  type-correct; do not read it as authoritative. Use
                  evidence.py's EvidenceLedger.evidence_count() against
                  the running store for the real, foldable number.
  use_count     = 0   -- seed value, same rationale (use.ledger is the
                  authoritative source once the adapter starts serving
                  live queries).
  meta          = {"source": "foss-ki:knowledge_full.json", "kb_version": 2}

Segmentation: 50 records per segment (per the assignment), in source
order, so a single `git blame`-style provenance question ("which segment
holds the France/capital/Paris fact?") is answerable by dividing the
source index by 50 -- and so the cut/append demo has enough granularity
to cut exactly one segment without dragging in unrelated facts.
"""
import json
import os
import sys

sys.path.insert(0, os.path.expanduser("~"))
from livecausal_bridge import LiveStore  # noqa: E402

SEGMENT_SIZE = 50
SOURCE_TAG = "foss-ki:knowledge_full.json"
KB_VERSION = 2


def _normalize(s):
    return s.strip().lower()


def load_triplets(path):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data["triplets"], data.get("version"), data.get("count")


def triplet_to_record(triplet, source_index):
    s, r, o = triplet[0], triplet[1], triplet[2]
    return {
        "trigger": s,
        "mechanism": r,
        "outcome": o,
        "trigger_key": _normalize(s),
        "outcome_key": _normalize(o),
        "doc_coord": "knowledge_full:{}".format(source_index),
        "evidence_count": 1,
        "use_count": 0,
        "meta": {"source": SOURCE_TAG, "kb_version": KB_VERSION, "source_index": source_index},
    }


def convert(knowledge_full_path, store_dir, segment_size=SEGMENT_SIZE, dry_run=False):
    triplets, version, count = load_triplets(knowledge_full_path)
    if count is not None and len(triplets) != count:
        raise ValueError(
            "knowledge_full.json count field ({}) does not match actual "
            "triplet list length ({}) -- refusing to convert against a "
            "self-inconsistent source file.".format(count, len(triplets))
        )

    store = LiveStore(store_dir)
    segment_shas = []
    n_records = 0

    for batch_start in range(0, len(triplets), segment_size):
        batch = triplets[batch_start:batch_start + segment_size]
        records = [
            triplet_to_record(t, batch_start + i)
            for i, t in enumerate(batch)
        ]
        n_records += len(records)
        if dry_run:
            continue
        sha = store.append_segment(records)
        segment_shas.append(sha)

    return {
        "source_path": knowledge_full_path,
        "source_version": version,
        "source_count": len(triplets),
        "segment_size": segment_size,
        "n_segments": (len(triplets) + segment_size - 1) // segment_size,
        "n_records_converted": n_records,
        "segment_shas": segment_shas,
        "store_dir": store_dir,
        "dry_run": dry_run,
    }


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--source", default=os.path.expanduser(
        "~/mac_offload/desktop/foss-ki/data/knowledge_full.json"))
    ap.add_argument("--store-dir", required=True)
    ap.add_argument("--segment-size", type=int, default=SEGMENT_SIZE)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    result = convert(args.source, args.store_dir, args.segment_size, args.dry_run)
    print(json.dumps(result, indent=2))
