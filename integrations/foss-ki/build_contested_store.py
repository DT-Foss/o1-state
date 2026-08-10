"""
Builds a genuine SS2-conflict store for Task 16's contested/dominance demo:
same edge_key (trigger_key=coffee, outcome_key=alertness), TWO different
mechanisms asserted -- exactly evidence.py's own conflict definition
(module docstring: "two records citing the same (from_key, to_key) pair
with DIFFERENT mechanisms collide on the same edge_key"). No new conflict
logic invented -- this is data setup, not a conflict-detection algorithm.
"""
import sys

sys.path.insert(0, "/root")
from livecausal_bridge import LiveStore  # noqa: E402

store = LiveStore("/root/fosski-venv/contested_store")

# evidence.py's _evidence_key_for_record falls back to "segment provenance"
# (evidence_key = citing segment's sha, "at most one evidence unit per
# segment for this edge, regardless of how many times the edge recurs
# inside it") whenever doc_coord isn't a list/tuple or an int/float -- a
# STRING doc_coord like "study:1" hits exactly that fallback. So three
# "causes" records packed into ONE segment would fold to evidence_count=1,
# not 3 -- each independent source needs its OWN segment for
# evidence_count to actually count sources, not records. This is a real,
# traced property of evidence.py's fallback resolution, not a workaround.
causes_shas = []
for i in range(1, 4):
    rec = [{"trigger": "Coffee", "mechanism": "causes", "outcome": "alertness",
            "trigger_key": "coffee", "outcome_key": "alertness",
            "doc_coord": f"study:{i}", "evidence_count": 1, "use_count": 0, "meta": {}}]
    sha = store.append_segment(rec)
    causes_shas.append(sha)
print(f"causes segments (3 independent sources, 1 each): {causes_shas}")

# One more segment asserting "correlates_with" instead -- same edge_key,
# different (weaker) mechanism claim.
sha_b = store.append_segment([
    {"trigger": "Coffee", "mechanism": "correlates_with", "outcome": "alertness",
     "trigger_key": "coffee", "outcome_key": "alertness",
     "doc_coord": "study:4", "evidence_count": 1, "use_count": 0, "meta": {}},
])
print("segment (correlates_with, 1 source):", sha_b)
