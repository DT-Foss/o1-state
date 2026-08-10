# EVIDENCE CALCULUS — DRAFT (for Lead review)

**Status: DRAFT. Not binding. Written as MVP-7 design groundwork, no code
changes made. Options are laid out with trade-offs; each section ends with
a recommendation.**

Scope: how the live graph counts evidence for a base edge, resolves
contradictory triplets, and lets `use_count` grow — all without breaking
the content-addressing that `src/livecausal/store.py` and
`analysis/LIVE_CAUSAL_SPEC.md` (SS1-SS2) establish (segments are sealed,
sha256-named, append-only; a record's bytes never change after sealing).

The one constraint every option below is checked against: **a sealed
segment's bytes must never change.** Any mechanism that wants to mutate
`evidence_count` or `use_count` on an already-sealed record is disallowed
by construction — those counters cannot live inside the record once it is
part of a hashed segment. That reframes all three problems as the same
problem: where does mutable state live, next to immutable base data,
staying just as deterministic and stranger-verifiable (P60 pattern)?

---

## 1 — Evidence counting: what is an independent source, and where does
the count live

### 1a. What counts as independent evidence for a base edge

A base edge is the (trigger_key, mechanism, outcome_key) triple. Two
triplet records support the "same" edge when their keys and mechanism
match after normalization; each such record is a candidate evidence
instance. The question is when two instances count as independent
sources rather than one restated fact.

**Option A — doc_coord distance threshold.** Two records for the same
edge count as independent evidence if their `doc_coord` values are more
than some window W apart (e.g. different documents entirely, or >N tokens
apart within one document). Rationale: text near each other in a stream
is likely one restated claim (a sentence and its paraphrase), not two
observations.
- Pro: directly measures "did the extractor see this claim twice in one
  breath," which is the actual redundancy risk.
- Con: W is an arbitrary knob; cross-document vs. within-document
  needs different W; doc_coord is `int|list` per the schema (P55 rule)
  so "distance" needs a type-aware definition (flat int diff for text,
  structural distance for the (lane_seed, episode, frame, offset) form).

**Option B — segment provenance.** Two records count as independent
evidence if they were curated into different segments (i.e., different
builder passes / different gate-curation events). Within one segment, all
matching records collapse to one evidence unit; across segments, each
distinct segment contributes at most one evidence unit per edge regardless
of how many times that edge appears inside it.
- Pro: no arbitrary distance constant; reuses a boundary the store
  already has (segment = one sealed batch = one curation event, SS1).
  Directly aligned with the builder loop (SS4): each Fold step is a
  natural "independent look."
- Con: coarser — a huge segment spanning many documents only ever
  contributes 1 evidence unit for a repeatedly-seen edge, even if that
  segment genuinely saw the claim in three unrelated sources.

**Option C — doc identity (whatever "document" means at ingestion:
stream id, file id, episode id).** Independent evidence = distinct source
document, full stop; doc_coord's leading component (the P55 coordinate
already carries this) is compared for equality, not distance.
- Pro: matches the intuitive meaning of "independent source" directly;
  no tuning constant.
- Con: needs the coordinate scheme to expose a stable document-id prefix
  uniformly across the int and list forms of doc_coord — that's a schema
  commitment, not just a counting rule.

**Recommendation: C, with B as the fallback when a document id isn't
resolvable from doc_coord.** Distance thresholds (A) invent a constant we
cannot justify from first principles and that will need re-tuning per
corpus; segment provenance (B) is free but too coarse as the sole rule
once segments get big. Document identity is the actual semantic target —
"did N different sources say this" — and doc_coord already carries what's
needed for it under the P55 stranger-coordinate rule; B becomes the
degenerate case for coordinate schemes where no doc-id prefix exists
(e.g. a raw offset stream). Concretely: `evidence_key = doc identity
extracted from doc_coord`, distinct `evidence_key` values seen supporting
the same base edge = the evidence count.

### 1b. The content-addressing tension: idempotent re-append vs. evidence
accumulation

Re-appending byte-identical records produces the same sha (store.py's
current, correct behavior) and is a no-op on the manifest. But if the
extractor re-derives the same edge from the same source twice (e.g. two
separate builder runs both see the same document), that's redundant
evidence, not new evidence — the store already treats it as a no-op,
which is right for the SEGMENT layer. The tension is real only if we
wanted evidence to accumulate ON the record itself, which the content-
addressing model forbids.

**Option A — Evidence Ledger, separate from segments (recommended).** A
new file, `evidence.ledger` (or `.jsonl`, append-only, one line per
observation), living beside `manifest.json` in the store directory. Each
line: `{"edge_key": ..., "evidence_key": ..., "segment": sha, "idx": i}`
— i.e., "this segment+index observed this edge from this source." The
ledger is itself append-only and hashable (canonical bytes = same JSONL
discipline as segments; a running sha256 or a periodic sealed checkpoint
gives it the same tamper-evidence property). `evidence_count` for an edge
is COMPUTED, not stored: `len(distinct evidence_key for edge_key across
the ledger)`. No mutation of any sealed segment, ever — the ledger only
ever grows, and the count is a pure fold over it.
- Pro: base segments stay exactly as content-addressed as they are today;
  zero schema change to store.py; the count is always re-derivable by a
  stranger from the ledger alone (P60 pattern) and cross-checkable
  against the segments it cites.
  Because it's keyed by (segment, idx) it can be dropped in lockstep with
  drop_segments — ledger lines citing a dropped segment become dead and
  are filtered out on read, exactly like inferred-edge invalidation (SS2).
- Con: a second file to keep in sync; a ledger line can reference a
  segment that later gets dropped (must filter live, not clean eagerly —
  same "invalidation over rewrite" approach the spec already uses for
  inference).

**Option B — evidence segments (a distinct sealed segment type,
append-only observations rather than append-only base triplets).**
Structurally identical to Option A but the observation-log itself gets
sealed into hash-named `.evseg` files rather than one flat ledger file.
- Pro: uniform mental model (everything in the store is "a sealed
  segment of something"); reuses append_segment machinery almost as-is.
- Con: over-engineered for what is fundamentally a single append-only
  log; splitting the evidence log into content-addressed chunks buys
  nothing (nobody drops "half the evidence log" the way one drops a
  base segment) and adds a second manifest-like bookkeeping surface for
  no operational benefit.

**Recommendation: Option A (flat evidence ledger).** It is the minimal
mechanism that satisfies "never mutate a sealed record" while staying
exactly as auditable — sealed segments answer "what triplets exist,"
the ledger answers "how many independent times was each edge seen," and
both are pure functions of append-only, hashable logs. Segmenting the
ledger (Option B) adds bookkeeping without adding falsifiability.

---

## 2 — Deterministic conflict rule

Contradiction = two base edges sharing (trigger_key, outcome_key) [same
node pair] with incompatible mechanism or direction (A causes B vs. B
causes A vs. A causes NOT-B — mechanism-level negation needs its own
normalized vocabulary, out of scope here; assume a boolean "conflicts"
predicate over two mechanism strings is available from the extractor).

### 2a. Coexistence vs. suppression

**Option A — always coexist, annotate.** Both edges live in the graph
permanently; a `contested: true` flag (or a pointer to the conflicting
edge_key) is attached at read time by consulting the ledger, never
written into the sealed record. Downstream consumers (inference, the
organism's consult step) see both and the evidence tally for each.
- Pro: maximally honest — the graph never silently drops a minority
  claim; matches "knowledge lives in files, not frozen weights" (SPEC
  thesis) — suppression is a kind of premature freezing.
- Con: consumers must be conflict-aware; without a downstream policy,
  "coexist" just defers the decision.

**Option B — dominance suppression.** Once one side's evidence count
exceeds the other's by a fixed ratio or margin (threshold T), the
minority edge is marked `superseded` (still present, still verifiable,
but excluded from default reads/inference) rather than deleted.
- Pro: gives consumers a usable default answer without deleting history.
- Con: T is another arbitrary constant; a threshold invites edge cases
  (evidence counts 5 vs. 6 flips the "winner" on one more source) that
  need a stability rule (hysteresis?) to avoid the graph's answer
  flapping as evidence trickles in.

**Recommendation: A for storage, with a well-defined dominance function
computed at READ time (not baked into the store) as the default policy
for consumers that want a single answer.** Never delete or hard-suppress
a minority claim at write time — that's an inference-time policy
decision, and per SS2 inference is already the delta-maintained, re-
derivable layer. Concretely: define `dominance_ratio(edge_key) =
evidence_count(winning side) / evidence_count(losing side)`, computed
from the ledger, with the RATIO exposed to consumers rather than
resolved by the store into a boolean. This keeps the store an evidence
substrate, not a truth arbiter.

### 2b. The threshold, if consumers want one

For the one consumer that does want a binary "dominant edge" answer
(e.g. the organism's consult step, SS4.4, needs a decidable graph miss
vs. graph hit): propose **T = 2x evidence count AND >=2 independent
evidence_keys on the winning side** (the count-of-2 floor kills the
degenerate "1 vs. 0" false-dominance case). Both numbers are placeholders
for the Lead to set or for P71/whatever measurement registers dominance
behavior empirically — flagged here as a knob, not asserted as correct.

### 2c. How inference treats contested edges

Per SS2, inferred edges are derived from base edges and carry citation of
their derivation. Recommendation: **delta-inference propagates through
an edge unconditionally (dominance-agnostic) but tags the inferred edge's
contested status as inherited from its base edges** — i.e., an inferred
edge is contested if ANY edge in its derivation chain is contested. This
keeps inference a pure, deterministic function of base edges (no implicit
threshold check silently pruning propagation, which would make inference
policy-dependent and harder to audit) while still letting a downstream
consumer filter contested inferred edges the same way it filters
contested base edges. Propagating only past-dominance-threshold edges
was considered and rejected: it would make the inferred graph's shape
depend on a tunable constant, breaking "any inferred edge is re-derivable
by a stranger from the cited base edges alone" (SS2) unless the threshold
value itself is pinned into the derivation record — extra bookkeeping for
a filter that read-time consumers can already apply themselves.

---

## 3 — Use-reinforcement (`use_count`)

Same root tension as SS1: `use_count` on a sealed record cannot be
mutated in place. Same shape of answer.

**Recommendation: a second append-only ledger, `use.ledger`, edge-key
addressed** (could be the same physical ledger file as the evidence
ledger with a `"kind": "evidence"|"use"` discriminator, or a separate
file — separate file recommended for read-load isolation, since use
events will be far higher-frequency than evidence events once the
organism is consulting the graph live, SS4.4). Each successful consult
that resolves against edge_key appends `{"edge_key": ..., "ts_logical":
n}` where `ts_logical` is a monotonic logical counter (a build/run
sequence number owned by the caller), NOT a wall-clock timestamp — SS1's
"no timestamps in the serialization" rule for segments should extend
here: wall-clock time is not reproducible or stranger-verifiable, a
logical sequence number is. `use_count(edge_key) = count of lines in
use.ledger for that key`, computed at read time exactly like
`evidence_count`.

This makes "the graph learns from being used" (MVP-5's framing) a pure
consequence of an append-only, replayable log — a stranger can recompute
every use_count from use.ledger alone, and use.ledger's own canonical-
bytes + running hash gives it the same tamper-evidence as a segment.

One open design question flagged, not resolved here: should `use_count`
feed BACK into dominance (SS2.1) — i.e., does a heavily-consulted edge
that keeps resolving correctly count as its own evidence? That conflates
"cited by sources" with "confirmed by usage," which are different
epistemic claims (the former is about the world, the latter is about the
graph's own track record). Recommend keeping them as two separate
numbers exposed to consumers (`evidence_count`, `use_count`) rather than
merging them into one score — merging is a modeling choice with real
consequences (self-reinforcing errors: a wrong edge used often would rank
itself up) and belongs in a later, explicitly-registered prediction, not
folded silently into the calculus.

---

## Summary — the 2-3 decisions that matter most

1. **Evidence counting = distinct source-document identity extracted
   from doc_coord** (fallback: segment provenance where no document id
   is resolvable), stored as a separate append-only **evidence ledger**
   keyed by (edge_key, evidence_key, segment, idx) — never mutates sealed
   segments, count is a computed fold, dead ledger lines are filtered
   (not rewritten) when their segment is dropped.

2. **Conflicts always coexist in storage; dominance is a read-time ratio
   over ledger-derived evidence counts, never a write-time deletion or
   suppression.** Inference propagates through contested edges
   unconditionally and tags inherited contested status, keeping the
   inferred graph a deterministic, stranger-re-derivable function of the
   base edges — filtering by dominance is left to consumers.

3. **`use_count` gets the same append-only-ledger treatment as evidence**
   (separate file, logical sequence numbers not wall-clock time), and is
   kept as a distinct number from `evidence_count` rather than merged
   into one score — merging is flagged as a real future decision, not
   made here.

All three keep the store's core invariant intact: sealed segment bytes
never change; every derived number (evidence_count, use_count, dominance
ratio, contested flag) is a pure, replayable fold over append-only logs
that cite base segments by (sha, idx), so any party can recompute any of
them from scratch — the P60 stranger-verification pattern extended from
entries and inference derivations to the evidence layer itself.
