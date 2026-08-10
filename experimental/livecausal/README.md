# experimental/livecausal/ — LIVE-CAUSAL bridge, revival-probe Phase 2

Source-of-record copies of the LiveCausalAdapter MVP built during the
FOSS-KI revival probe (2026-08-10). These files are committed here for
history/review; the files ACTUALLY IMPORTED at runtime live outside this
repo, at `~/fosski-venv/adapter/live_causal_adapter.py` and
`~/fosski-venv/convert_knowledge_full.py` — kept out-of-repo deliberately,
since they depend on `~/livecausal_bridge/` (store.py/infer.py/evidence.py/
canon.py, the o1-state team's bridge modules), which is not part of this
repository and was placed on this machine for this probe only.

If promoting this out of "experimental probe" status: sync these files
back into `~/fosski-venv/` (or vice versa — decide which copy is
authoritative) and update `repl.py`'s `_maybe_load_live_causal_adapter()`
path resolution if the adapter's location changes.

## Files

- `convert_knowledge_full.py` — one-shot converter, `data/knowledge_full.json`
  (FOSS-KI's flat subject/relation/object KB, 4,855 triplets) into
  LiveStore segments (50 records/segment, 98 segments). Schema mapping and
  every placeholder value (evidence_count=1 seed, doc_coord synthesis) is
  documented in the module docstring — read it before trusting any
  confidence number derived downstream.

- `live_causal_adapter.py` — `LiveCausalAdapter`, a narrow drop-in for the
  subset of `core.knowledge.KnowledgeStore`'s interface `repl.py` actually
  calls (`query`, `find_by_entity`, plus a materialized `.facts` list and
  a `store_fact`/`store_facts` write path repl.py's ~87 direct-facts-access
  call sites and learn-from-statement path need). Confidence derivation
  from `evidence_count` is a documented, explicitly-not-principled linear
  scale — see the module docstring's "CONFIDENCE DERIVATION" section for
  the exact formula and why it was chosen.

- `demo_cut_append.py` — the Phase 2 point-6 proof script. Boots ONE
  `FossKIRepl(live_causal_store=...)`, asks "what is the capital of
  France?" three times (baseline / after cutting the citing segments /
  after re-appending them), with a control question in between to prove
  the rest of the KB is untouched. All 6 automated checks PASS. Honest
  caveat baked into the script's own output (not hidden): repl.py's
  independent ConceptNet/CommonSense layer also knows "France capital
  Paris" and is NOT converted into this adapter's store, so the FULL
  repl.py answer in step (ii) can still say Paris via that other route —
  the cut is proven at the adapter/query() level directly, which is the
  level this deliverable actually controls.

## Activating the switch

```python
from repl import FossKIRepl
repl = FossKIRepl(live_causal_store="/path/to/a/converted/store")
# or: FOSSKI_LIVECAUSAL_STORE=/path/to/store python repl.py
```

Default (no argument, no env var) is unchanged: the original in-memory
`KnowledgeStore` + brain-snapshot boot path, byte-identical to every
`FossKIRepl()` call before this switch existed.
