# LIVE-CAUSAL — MVP Dossier (DRAFT)

> **Status: DRAFT.** This is a first pass for internal Lead review, not a
> distributed document. Every number below has an artifact path; nothing
> here is rounded past what the source JSON states, and every falsifier
> that fired is reported at the same strength as every pass. Machine
> classes are named by role (an ARM development machine, two x86 runners),
> never by hostname.

---

## 1. The system, in five sentences

Knowledge should live in a file, not be baked into frozen weights: a
small, constant-memory reader (the organism) streams a corpus and, on its
own surprise signal, curates which windows are worth extracting. A
deterministic multi-pass extractor (fabel, a 14-step validation gate — no
LLM anywhere) turns curated text into validated causal triplets, and a
delta-inference engine folds each new batch into a live `.causal` graph
without ever rebuilding it from scratch. The graph is content-addressed
and append-only — every base edge and every inferred chain carries an
exact citation back to the segment and record that produced it, so a
stranger on different hardware can re-derive any edge from the file alone
and never has to trust the engine that built it. Truncation (dropping a
segment) invalidates exactly the edges whose derivation cites it, with
zero re-inference of the surviving graph. This dossier is the record of
that loop actually running end to end, once, on a real corpus, with every
number pre-registered before it was measured.

---

## 2. The demo

The four-command transcript below is the same scenario proven in
`src/livecausal/test_demo.py` (2/2 tests green) and shown live via
`src/livecausal/demo.py`. Each command is a thin wrapper over the modules
scored in Section 3 below — no new logic lives in the CLI itself.

```
$ python3 -m livecausal.demo build --text-file korpus.txt --store demo_store \
    --window-tokens 40 --max-windows 80
[builder] windows=80 (gated=11) triplets=80 (from_gated=11) segments=16 \
    base_edges=16 inferred_edges=70 wall=2.9s
==========================================================================
BUILD SUMMARY — demo_store
--------------------------------------------------------------------------
  tokens streamed (approx, windows x window_tokens) : 3,200
  windows (gated / total)                           : 11 / 80
  validated triplets (from gated / total)           : 11 / 80
  segments sealed                                   : 16
  base edges                                        : 16
  inferred edges                                    : 70
  wall clock                                        : 2.91s
```
*The organism streams the corpus; its own surprise gate decides what it
learns from (11 of 80 windows here), while the extractor runs on every
window — the gate curates storage, not extraction (P70, Section 3).*

```
$ python3 -m livecausal.demo query --store demo_store --key "smoking"
  [BASE] smoking -> lung cancer
  evidence_count : 19
  use_count      : 0
  contested      : False
    [15b5c9ece340...:0] [15b5c9ece340...:2] [15b5c9ece340...:3] ...

  [INFERRED] smoking -> reduced life expectancy  (depth=2)  [52 derivations, showing 1]
  evidence_count : 0
  use_count      : 0
  contested      : False
    smoking -> lung cancer -> reduced life expectancy
        [15b5c9ece340...:0] smoking -> lung cancer  (smoking causes lung cancer)
        [25e27852d2d6...:2] lung cancer -> reduced life expectancy  (lung cancer leads to reduced life expectancy)
```
*The query surfaces both a directly-extracted edge and a two-hop chain the
delta-inference engine derived from it, each citation traceable to the
exact segment and record that produced it.*

```
$ python3 -m livecausal.demo cut --store demo_store --segment af36a49c90ae...
  base edges                 : 16 -> 16
  inferred edges             : 70 -> 60
  inferred edges invalidated : 10
no re-inference of the surviving graph — this is the live property: cut
removes exactly what it cites, nothing else.
```
*Dropping one segment invalidates exactly the ten inferred chains whose
derivation cited it — the rest of the graph, including an unrelated causal
chain in the same corpus, is untouched.*

```
$ python3 -m livecausal.demo verify --store demo_store --n 10
  sampled   : 10
  verified  : 10/10 (pass=True)
  consensus : 10/10 (pass=True)
VERDICT: PASS
```
*A stranger re-derives ten sampled edges — base and inferred — from
nothing but the cited raw records, never trusting the engine's own cache;
two independent replays agree bit-for-bit.*

---

## 3. The numbers

Claim → measurement → artifact. Every clause below is a pre-registered
prediction in `analysis/PREDICTIONS.md`, scored after the fact against its
own pre-committed bar.

| # | Claim | Measurement | Artifact |
|---|---|---|---|
| P71 | The live graph never rebuilds | 40 segments: incremental and full-rebuild closures sha256-identical (`c6a96567…`). 16× data growth (5→80 segments): append time grows 1.68× (bar ≤4×; median 7.9ms → 13.3ms), delta-yield holds constant at 190 new edges per append regardless of graph size. Truncation: 5 repetitions, **0 closure computations** on drop, 5/5 bit-equal to a full batch rebuild. | `results/livecausal_p71.json` |
| P72 | The builder builds, end to end | Two full runs, real WT-103 text, spaCy-strict extractor path pinned: **2,046 base records**, **644 segments**, wall clock **281.8s** (under five minutes) on a 4-core x86 runner. Second run reproduces all 644 segment SHAs bit-identically — zero full rebuilds. Direction-3 stranger audit: **30/30 edges verified, 30/30 two-mount consensus**. Gated fraction **0.2053**, inside the q=0.75 dial band [0.20, 0.30]. | `results/p72_run1.json`, `results/p72_compare.json`, `results/p72_verify.json` |
| P73 | Consult-back on the real graph | Machinery sound and deterministic: 0 ledger violations, 14/14 use citations resolve against live segments, two runs byte-identical including the use.ledger sha256. Value honestly negative: coverage **0.0162** (40 of 2,471 spikes answerable by exact key) and the real-path arm does **not** beat the random arm (mean Δ −0.0040 vs −0.0023) — both numbers localize to entity canonicalization, agreeing with P72's write-side finding. | `results/livecausal_consult_run1.json`, `results/livecausal_consult_run2.json` |
| P70 | Does the gate's curation raise validated-triplet yield over random windows? | **No — the falsifier fires.** Surprise-selected windows: 18.13 validated triplets/kilotoken. Seeded-random windows: 18.60. Yield ratio **0.974** (bar 1.3× — fail). Entity-novelty ratio **1.03** (bar 1.5× — fail). Double-pass extraction stays byte-identical (pass). **Policy consequence, taken straight from the falsifier:** extraction runs ungated on every window; the surprise gate curates what the organism *learns from*, not what the curator *may extract* — a real result folded directly into the builder's design (Section 2's `windows (gated / total): 11 / 80` line is this policy, live). | `results/curator_yield.json` |
| P60 | Cross-ISA stranger verification | Forward (ARM→x86): **10/10** bit-exact, 10/10 consensus. Reverse (x86→ARM): **9/10** — the falsifier fires on one entry (lane 4205, episode 0, frame 14: a single token, one quantization bin apart), localized to a 1-ULP float block-mean reduction-order difference between ISAs and answered with a normative fix (integer-exact token quantization). Net: **19/20** sampled entries bit-exact across both directions. | `results/stranger_verify.json`, `results/stranger_verify_arm.json` |
| P55 | The frozen file answers by own key | After dosed replay of a sha256-frozen file, the reader completes the file's own entries **0.264 nats better** than its no-file twin (bar 0.05, 5.3× over) — keyed recall, not diffuse fertilization. | `results/keyed_file.json` |
| P58 | Surprise is a novelty filter, not a difficulty filter | Gate-selected windows carry **1.51× the first-ever content** of seeded-random windows (7.0× on brand-new token types), at 0.69× the redundancy — holds against a second random seed. | `results/surprise_filter.json` |
| P67 | The filter's value is novelty-graded, not uniform | One producer, two harvest policies: the random-harvested file wins average-heldout fertilization ~1.8×, but its advantage lives entirely in familiar content and shrinks monotonically as the eval gets newer (Δ(S−R): −0.053 → −0.027 → **+0.001**, crossing to favor the surprise-harvested file in the top-novelty tercile). | `results/novelty_transfer.json` |
| P59 | Age does not shift the gate's rate; experience is measurable immunity | A 7.44-billion-token life forked read-only and compared to a fresh 50M-token twin: gate rate Δ0.5pp against a ±2pp bar (149× the lived experience, invariance holds). Under a WT-103 shock the veteran forgets **32% less** (+0.0845 vs +0.1243) and recovers to *below* its pre-shock loss — reader-side, this means the organism does not need to relearn what it has genuinely lived through. | `results/aged_brain.json` |

---

## 4. What's missing, honestly

**The measured constraint: entity canonicalization.** P72's clause (a)
split: **2,046 base records** cleared the bar (≥1,000) decisively, but the
graph produced only **76 inferred (multi-hop) edges** against the
registered bar of **200 — a clear fail.** The cause is precisely
localized, not vague: 2,047 extracted triplets collapsed to 2,046
*distinct* trigger/outcome key pairs — near-zero key collisions. Real
prose almost never repeats an exact trigger/outcome string twice, so
chains that are semantically the same but worded differently never join
in the graph's exact-string key space. This is not a bug in the
delta-inference engine (P71's closure math
is proven exact); it is a missing layer, named twice before this run and
now measured as the binding constraint on graph connectivity: entity
canonicalization, or the deferred fuzzy-matching pass v1's inference
engine already documents as flagged-not-implemented scope. This is the
next organ to build, not a caveat to hedge around.

**Consult-back is built, measured, and blocked on the same organ.** The
spec's fifth stage — the organism reading the graph back in flight — ran
against the real P72 store (P73, two runs): the machinery is proven
sound (0 ledger-discipline violations, 14/14 citations resolve, both
runs byte-identical down to the use.ledger sha), but its VALUE is
measured as blocked on canonicalization from the read side too. Coverage
0.0162 — of 2,471 surprise spikes on a live WT-103 stream, only 40 gap
words found any edge in the exact-string key space — and at that
coverage the matched paths are semantically unrelated to the
continuation, so the real-vs-random injection arm shows no separation
(mean Δ −0.0040 vs −0.0023, noise scale; the pre-registered falsifier,
reported at full strength). Write side (P72: 76/200 inferred edges) and
read side (P73: 1.6% coverage) now agree on the single binding
constraint. One organ, two measured numbers waiting on it.

**Scale is measured at 10×, and it taught us something (P74).** The
natural falsifier ran: a 30,000-chunk build (6,401 segments, 20,648
records — precisely 10× P72) replayed through its own history twice.
The mechanics scale cleanly — truncation stays zero-closure at any
density (10/10 samples), closure accounting is exact across all 6,401
appends, and both replays are structurally byte-identical. The
discovery: graph density is not scale-invariant. Exact-key collisions
accumulate superlinearly with stream length (76 inferred edges at 1× →
5,408 at 10×, density 0.037 → 0.262), and append cost tracks that
density — the pre-committed guard fired, so the sparse-regime bar is
void and the curve reports descriptively (last/first fifth ≈ 11–13×,
absolute cost still 94ms median at the frontier; the full 10× history
replays from raw segments in ~8.5 minutes). The dense-regime bar
re-registers in density-normalized form now that canonicalization
deliberately raises density.

---

## 5. The methodology (the moat)

Every number in Section 3 was a numbered, immutable prediction in
`analysis/PREDICTIONS.md` *before* the run that scored it — committed to
the record before the data existed, never edited after the fact. The
register holds **72 predictions through P73**, and the latest full
mechanical audit reads **0 mismatches** (`results/scorer_audit.json`,
2026-08-10); `src/score_predictions_v2.py`
mechanically re-parses the entire file, loads every cited `results/*.json`
artifact, and compares the register's own claimed numbers against the
artifact's actual values — MATCH, MISMATCH, or UNPARSEABLE, always
reported, never smoothed, exit code 0 regardless of outcome ("it reports,
it does not gate"). A prediction that fails its own pre-committed bar
stays in the register at full strength: P70's yield-ratio falsifier and
P60's one-ULP mismatch are recorded exactly like P71's and P72's passes,
each with the artifact that produced it and, where the falsifier localized
a cause, the fix that followed. This is what makes the numbers above
checkable by someone who was not in the room: the file, the coordinates,
and the re-derivation are the whole proof.
