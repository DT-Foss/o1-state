# LIVE-CAUSAL — the knowledge file becomes a living graph

**Author: David Tom Foss · Public technical disclosure (prior art as of commit
date). Design specification for `.causal` v2 ("live-causal") and the
organism-as-builder coupling. Everything here is disclosed in its broadest
form; measured anchors are cited where they exist.**

## The thesis

Knowledge should live in files, not in frozen weights. The end architecture:
a small, generic, constant-memory reader (the organism) plus a set of
**live** `.causal` knowledge graphs — append-able, truncate-able,
incrementally inferenced, per-entry verifiable — which are the system's
editable, composable, hot-loadable "weights". The organism is the BUILDER:
its surprise gate curates what enters the graph (measured: the gate selects
first-ever content at 1.51× / 7× on new types and 0.69× redundancy, P58;
its file's transfer value is novelty-graded, P64/P67), the deterministic
extractor (fabel) turns curated text into validated triplets at
gigabytes-in-seconds throughput, and delta-inference folds them into the
graph without ever rebuilding it.

## 1 — Segments: append-only, sealed, hashed

The v2 file is a MANIFEST plus an ordered sequence of SEGMENTS. Each segment
is an append-only batch of base triplets, sealed with sha256 over its
canonical serialization (the P47/P55 file discipline applied per segment).
Every base triplet carries stranger-verifiable coordinates (doc_coord or the
(lane_seed, episode, frame, offset) form — the standing DECISIONS rule).
Growth = appending a segment. Truncation = dropping segments from the tail
(or any subset, see §3). The manifest records the segment order and hashes:
the whole graph state is content-addressed at every step of its life.

## 2 — Delta-inference: the graph never rebuilds

Materialized inference (transitive chains, direction propagation, fuzzy
key-matching — the v1 build-time closure) becomes INCREMENTAL:

- **Append (semi-naive delta propagation).** A new base edge A→B generates
  exactly the inferred edges reachable through it: ancestors(A) ×
  descendants(B), depth-capped as in v1 (hsslm's 5-hop closure). Cost scales
  with the delta's neighborhood, not with graph size. This is the semi-naive
  evaluation of datalog, applied to the `.causal` closure rules.
- **Every inferred edge carries its derivation** — the list of base edges
  (by segment + index) that produced it. Inference stays deterministic and
  auditable: any inferred edge is re-derivable by a stranger from the cited
  base edges alone (the P60 verification pattern lifted from entries to
  inferences).
- **Delete/truncate (derivation invalidation).** Dropping a segment
  invalidates exactly the inferred edges whose derivation cites it
  (counting/DRed-style maintenance in the standard case; the append-only
  segment structure makes the common case — tail truncation — a pure index
  scan). No re-inference of the surviving graph.

## 3 — Composition: graphs as loadable weights

Because segments are content-addressed and inference is delta-maintained,
graphs COMPOSE at file level: mount several graphs (the v1 fabel/brain
multi-mount path), merge by appending segment sets, subtract by dropping
them, fork by copying a manifest prefix. A reader can be dosed with any
combination (measured for span files: dosed replay stores entry-level keyed
content, P55; diffuse and keyed roles split by harvest policy, P64/P67).
Knowledge becomes an ARTIFACT ECONOMY: files with provenance, hashes, and
per-edge verifiability — the opposite of weight-baked knowledge.

## 4 — The organism as builder (the distiller bridge)

The coupling loop, each stage measured or registered:

1. **Curate**: the gate marks stream windows worth extracting (novelty
   filter, P58; frontier-graded value, P67). Redundant stream regions are
   never extracted — the filter is the dedup.
2. **Extract**: the deterministic multi-pass extractor (fabel; 14-step FOSS
   Gate) turns curated windows into validated triplets. Registered next:
   does gate-curation raise validated-triplet yield and novel-entity rate
   per token over random windows (P70)?
3. **Fold**: delta-inference appends the new segment; the graph stays
   inferenced at all times.
4. **Consult**: the organism reads the graph back in flight (F4 closed
   loop, reminded reads); graph MISSES on high-surprise queries are gap
   signals — the gap-driven growth loop (fabel v1's symbolic loop),
   neuralized: what the organism cannot answer becomes what the builder
   fetches next.
5. **Verify**: any party replays any entry or derivation from coordinates
   (P60: cross-ISA 19/20 bit-exact, the one divergence localized to a ULP
   and answered with the integer-quantization rule).

## 5 — What this is not (yet)

The interaction/intent layer (a minimal language module over the reader +
graphs) is deliberately LAST: it is the best-explored, least-scarce part of
the stack. The scarce parts — the perpetual constant-cost reader, the
curation laws, the live graph with per-edge provenance — are the engine,
and they are what this repository measures.

## Measured anchors cited

P47/P52/P55 (file discipline, tappable storage), P58 (novelty filter),
P60 (stranger verification), P64+P67 (the filter buys the frontier),
F4 (two-system law), F7 (portable organism); fabel v1: deterministic
extraction + build-time closure + multi-mount reading (vendor/fabel,
PAPERS.md).
