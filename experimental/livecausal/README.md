# experimental/livecausal/ — LIVE-CAUSAL bridge, revival-probe Phase 2+3+4+5

Source-of-record copies of the LiveCausalAdapter MVP built during the
FOSS-KI revival probe (2026-08-10). These files are committed here for
history/review; the files ACTUALLY IMPORTED at runtime live outside this
repo, at `~/fosski-venv/adapter/live_causal_adapter.py` and
`~/fosski-venv/convert_knowledge_full.py` / `~/fosski-venv/convert_conceptnet.py`
— kept out-of-repo deliberately, since they depend on `~/livecausal_bridge/`
(store.py/infer.py/evidence.py/canon.py, the o1-state team's bridge
modules), which is not part of this repository and was placed on this
machine for this probe only.

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

- `convert_conceptnet.py` (Phase 4) — converts `data/conceptnet_en_500k.json`
  (500,000 ConceptNet-en-5.7.0 assertions) into LiveStore segments. Every
  one of ConceptNet's 31 relation types is mapped explicitly in the module
  docstring — 189,719 records (37.9%) included as world-fact mechanisms
  (`is_a`, `causes`, `at_location`, `used_for`, ...), 310,281 (62.1%)
  excluded as lexical word-form data (`Synonym`, `DerivedFrom`, `FormOf`,
  `SimilarTo`, `Antonym`, `HasContext`, `FormOf`, `DistinctFrom`,
  `RelatedTo`) that names a relationship between STRINGS, not a fact about
  the entities they refer to. Every relation's inclusion/exclusion is
  backed by sampling real records, not going by the relation name alone
  (see the docstring's note on `DefinedAs`, which looks lexical but
  sampled as genuinely factual).

- `live_causal_adapter.py` — `LiveCausalAdapter`, a narrow drop-in for the
  subset of `core.knowledge.KnowledgeStore`'s interface `repl.py` actually
  calls (`query`, `find_by_entity`, plus a materialized `.facts` list and
  a `store_fact`/`store_facts` write path repl.py's ~87 direct-facts-access
  call sites and learn-from-statement path need). Confidence derivation
  from `evidence_count` is a documented, explicitly-not-principled linear
  scale — see the module docstring's "CONFIDENCE DERIVATION" section for
  the exact formula and why it was chosen. Phase 4 adds
  `find_segments_citing(subject, obj=None)` — every segment sha citing a
  base or inferred edge from `subject` (optionally filtered to a specific
  `obj`); see the "Multi-source forgetting" section below for why this
  exists and what it does NOT automatically handle.

- `demo_cut_append.py` — the proof script (Phase 4, multi-source). Boots
  ONE `FossKIRepl(live_causal_store=..., knowledge_only=True)` against a
  store built from BOTH `knowledge_full.json` AND a ConceptNet slice, and
  shows: cutting only the forward-direction citations is honestly NOT
  enough (the reverse-direction ConceptNet citation survives); cutting
  BOTH directions via `find_segments_citing` in each direction IS enough
  — the FULL `repl.process()` answer becomes an honest "I don't have
  information about that topic."; re-appending restores everything. 9
  automated checks, all PASS.

- `demo_e2e_loop.py` / `marie_curie_article.txt` (Phase 5, Task 15) — the
  full-stack proof: `builder_run.py` (`/root/o1lab/src/livecausal/`, the
  o1-state team's organism+fabel builder loop) streams a fixed local text
  file (`marie_curie_article.txt`, sentence-per-line, no web scraping),
  fabel's deterministic `curator_yield_run.extract_validated` (no LLM
  anywhere) validates causal triplets, folds them into the SAME
  `LiveStore`/`LiveGraph` a `FossKIRepl(live_causal_store=...)` mounts
  immediately afterward in the SAME process — no separate conversion
  step between building and answering. 8 automated checks, all PASS. See
  `DEMO.md`'s "DEMO 2" section for the full walkthrough and the honest
  scope note on what iteration this required (see below).

## `builder_run.py` integration (Phase 5)

`builder_run.py` and its `curator_yield_run.py` extraction dependency
live in `/root/o1lab/src/` (the o1-state team's own repository on this
machine, alongside the production q-sweeps this probe must never
disturb) — NOT copied into this repo or `~/fosski-venv/`, since they are
substantial, actively-developed modules with their own dependency tree
(`portable_organism.py` → `streaming_train.py` →
`moebius_scan_transformer_selective.py` → `moebius_attention.py`, the
last only importable with `/root/o1lab/reference/` added to
`PYTHONPATH` — a real gap in `/root/o1lab/src/`'s own layout, found by
tracing the actual `ModuleNotFoundError`, not guessed). `demo_e2e_loop.py`
invokes `builder_run.py` as a subprocess exactly as a human operator
would (`python3 /root/o1lab/src/livecausal/builder_run.py --text-file
... --store-dir ...`), confining ALL of its output to
`~/fosski-venv/e2e_loop_store` and `~/fosski-venv/e2e_loop_build*` — it
never writes into `/root/o1lab/results/` (the default `--store-dir`/
`--out-prefix`, always overridden explicitly by this demo).

`repl.py`'s `_direct_kb_lookup` gained one small addition for this
phase: a "what does X cause?" / "what causes X?" lookup against
`self.knowledge.facts` (mechanism `causes`), returning a full sentence
rather than the bare outcome — see the module-level comment at that call
site, and `DEMO.md`'s honest scope note, for exactly why and how this
was traced (not assumed) to close a real gap between fabel's causal-
mechanism triplet output and `core/router.py`'s pre-existing attribute-
shaped query patterns (`capital`/`known_for`/`location`/`born`/...,
none of which cover `causes`).

## `knowledge_only` mode (Phase 3, corrected in Phase 4)

Phase 2's version of this demo proved the cut/append cycle at the
`LiveCausalAdapter`'s own `query()` level, but noted that the FULL
`repl.process()` answer in step (ii) could still say "Paris" via FOSS-KI's
independent ConceptNet/CommonSense layer. Phase 3 built the
`knowledge_only` switch to close that gap.

**Phase 4 correction:** tracing the exact source of that leak (not just
inferring it) found it was NOT ConceptNet — `core.commonsense.CommonSenseEngine`,
queried directly and isolated, returns `{'found': False}` for
"capital of France" whether backed by `conceptnet_en_500k.json` or the
prebuilt `conceptnet_index.pkl`. The actual leak was
`load_bootstrap_to_engine(self.commonsense)`, called unconditionally in
`repl.py`'s `live_causal_store` boot branch — it loads `data/knowledge_full.json`
**directly into `self.commonsense`** via `engine.add_fact()`, the exact
same source file `convert_knowledge_full.py` converts into the adapter's
store. Cutting the adapter's copy left this second, un-cuttable copy of
the identical fact sitting in `self.commonsense` the whole time. This is
now also gated by `knowledge_only` (skipped when true) — see `repl.py`'s
comment at that call site for the full trace.

`repl.py`'s `knowledge_only` constructor kwarg / `FOSSKI_KNOWLEDGE_ONLY=1`
env var (default `False`, byte-identical behavior when unset) disables,
for the FACTS path only:

  - `load_bootstrap_to_engine(self.commonsense)` (Phase 4 fix — the real
    knowledge_full.json duplication bug described above)
  - the ConceptNet/CommonSense fast-path AND fallback queries
  - the CBR case-library fallback (`answer_open_question`, both the
    "why"-question and general fallback call sites)
  - MultiHop reasoning (which resolves through CommonSense, not the
    adapter, whenever the brain-snapshot boot path — the only place that
    ever points `multi_hop.kb` at `self.knowledge` — is skipped, which it
    always is under `live_causal_store`)
  - the Foss Pipeline (Reservoir/Hopfield autoregressive generation)
  - Web search (the ultimate independent fact source; would trivially
    defeat a forgetting demo)

**Left ON, deliberately:** `_solve_reasoning` (physics formulas,
sequences, analogies), `self.formulas`, and `self.reasoning`
(`ReasoningEngine`, constructed with `knowledge_store=self.knowledge` —
it already answers FROM the adapter, not around it). None of these are a
facts bypass; disabling them would prove nothing about forgetting and
would only make the demo's reasoning/math worse for no reason.

## Multi-source forgetting (Phase 4)

With both `knowledge_full.json` and ConceptNet converted into the same
store, a real-world relationship can be asserted by MORE THAN ONE source,
and — critically — in DIFFERENT DIRECTIONS: `knowledge_full.json` encodes
`("France", "capital", "Paris")` (`trigger_key=france, outcome_key=paris`),
while ConceptNet encodes `("paris", "AtLocation", "france")`
(`trigger_key=paris, outcome_key=france` — the REVERSE direction, a
different mechanism entirely). `find_segments_citing(subject, obj)`
finds every segment citing subject→obj — but a caller building a real
"forget this relationship" feature must call it in BOTH directions
(`find_segments_citing("france", "paris")` AND
`find_segments_citing("paris", "france")`) and union the results, because
this method does not itself know that two edges in opposite directions
describe "the same" real-world fact — that judgment is the caller's, this
method only answers "which segments cite this exact directed edge."
Measured directly in `demo_cut_append.py`'s step (ii-partial): cutting
only the forward direction leaves the reverse ConceptNet citation fully
queryable.

## Known scaling limit — full ConceptNet does NOT mount safely yet (Phase 4)

Converting the FULL 189,719 included ConceptNet records (500,000 raw,
mapped per the table in `convert_conceptnet.py`'s docstring) into segments
works fine — 22 seconds, 3,795 segments, no memory issue (writing
segments is O(1) per segment, no graph structure involved).

**Mounting that store as a `LiveGraph`/`LiveCausalAdapter` does not.**
`LiveGraph._rebuild_from_store()`'s `_batch_transitive_closure` (the
same routine measured non-scale-invariant in P74, MAX_DEPTH=5) OOM-killed
a mount attempt at 21.3GB resident memory on this machine (30GB total,
kernel log timestamp 2026-08-10T18:56:23 UTC, PID 1670063). Measured
scaling curve (own subprocess per size, `ulimit -v 8000000` safety cap,
each run's peak RSS via `resource.getrusage`):

| base edges | inferred edges | mount time | peak RSS |
|---|---|---|---|
| 3,278 | 11,325 (3.5×) | 0.19s | 37 MB |
| 11,047 | 305,835 (27.7×) | 6.0s | 490 MB |
| 19,690 | 2,101,020 (106.7×) | 43.3s | 3,389 MB |
| 25,076 | 4,379,273 (174.6×) | 92.6s | 6,945 MB |

The growth is clearly superlinear in both time and memory well before the
full 189,719-record ConceptNet conversion (~90k+ base edges after
dedup) — extrapolating this curve puts the full mount far past 30GB.
This is the same P74 density-is-not-scale-invariant phenomenon, now
concretely quantified in RSS/time on real ConceptNet data rather than
synthetic benchmark data. **Not fixed here per the assignment's own
instruction** ("dokumentieren, nicht selbst fixen") — this is exactly the
canon-persistence need the canon-organ track is already addressing (Task
#12, canon-map persistence landed 2026-08-10, warm mount 17.8s→1.81s on
the pre-existing store — but that fix targets `canon=True`'s specific
cache path, not `_batch_transitive_closure`'s base cost, which is what
OOM'd here under `canon=False`, the adapter's default).

**Practical consequence for this demo:** `demo_cut_append.py` and the
"ConceptNet is live" claim use a **15,000-source-relation ConceptNet
slice** (11,217 included records after mapping, 225 segments, 11,047
base edges, 305,835 inferred edges) — proven safe at 6.4s mount / 489MB
RSS, well inside the measured-safe range above. This is a real,
honestly-scoped subset, not the full 500K file — promoting to the full
file requires the mount-time fix this document flags, not a bigger
machine alone (the growth is superlinear, not just large).

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
