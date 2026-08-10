# experimental/livecausal/ — LIVE-CAUSAL bridge, revival-probe Phase 2+3

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

See `DEMO.md` for the 10-line pitch.

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

- `demo_cut_append.py` — the proof script (Phase 3, hardened). Boots ONE
  `FossKIRepl(live_causal_store=..., knowledge_only=True)`, asks "what is
  the capital of France?" three times (baseline / after cutting the
  citing segments / after re-appending them), with a control question in
  between to prove the rest of the KB is untouched. All 7 automated
  checks PASS, on the FULL `repl.process()` answer (not just an isolated
  adapter query) — see `repl.py`'s `knowledge_only` flag below for how.

## `knowledge_only` mode (Phase 3)

Phase 2's version of this demo proved the cut/append cycle at the
`LiveCausalAdapter`'s own `query()` level, but noted honestly that the
FULL `repl.process()` answer in step (ii) could still say "Paris" via
FOSS-KI's independent ConceptNet/CommonSense layer — a real, separate
knowledge source never converted into this adapter's store.

`repl.py`'s `knowledge_only` constructor kwarg / `FOSSKI_KNOWLEDGE_ONLY=1`
env var (default `False`, byte-identical behavior when unset) disables,
for the FACTS path only:

  - the ConceptNet/CommonSense fast-path AND fallback queries
  - the CBR case-library fallback (`answer_open_question`, both the
    "why"-question and general fallback call sites)
  - MultiHop reasoning (which resolves through CommonSense, not the
    adapter, whenever the brain-snapshot boot path — the only place that
    ever points `multi_hop.kb` at `self.knowledge` — is skipped, which it
    always is under `live_causal_store`)
  - Web search (the ultimate independent fact source; would trivially
    defeat a forgetting demo)

**Left ON, deliberately:** `_solve_reasoning` (physics formulas,
sequences, analogies), `self.formulas`, and `self.reasoning`
(`ReasoningEngine`, constructed with `knowledge_store=self.knowledge` —
it already answers FROM the adapter, not around it). None of these are a
facts bypass; disabling them would prove nothing about forgetting and
would only make the demo's reasoning/math worse for no reason.

## Activating the switches

```python
from repl import FossKIRepl
repl = FossKIRepl(live_causal_store="/path/to/a/converted/store")
# or: FOSSKI_LIVECAUSAL_STORE=/path/to/store python repl.py

# for a forgetting demo where the FULL repl.process() answer (not just
# the adapter's own query()) must honestly reflect a cut:
repl = FossKIRepl(live_causal_store="/path/to/store", knowledge_only=True)
# or: FOSSKI_LIVECAUSAL_STORE=/path/to/store FOSSKI_KNOWLEDGE_ONLY=1 python repl.py
```

Default (no arguments, no env vars) is unchanged: the original in-memory
`KnowledgeStore` + brain-snapshot boot path, with every fact source
active, byte-identical to every `FossKIRepl()` call before either switch
existed.
