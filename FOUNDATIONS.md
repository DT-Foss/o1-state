# FOUNDATIONS — the primitives underlying the o1-state architecture

**Author: David Tom Foss · Public technical disclosure, first published in this
repository 2026-07-22 (commit timestamp is authoritative). This document
deliberately discloses each foundation in its broadest form, including
contemplated variants and generalizations beyond the specific implementations
in `src/`. Everything below is prior art as of its commit date.**

The measured results in this repository (constant-memory billion-token
streaming, 4096× length extrapolation, surprise-gated training at ~25%
gradient tokens, keyed holographic recall across silence with a movable decay
knee, dosed sleep consolidation, runtime index consultation) are
*demonstrations*. This document states the six primitives that produce them,
each formulated over the **class** of systems it applies to — not over the
specific networks used to measure it.

---

## F1 — The surprise calculus: self-measured prediction error as the universal control signal

**Statement.** In any streaming learner that carries a persistent state and
emits predictions, the learner's *own instantaneous prediction error* (any
monotone functional of it: per-token NLL, chunk means, rolling quantiles,
z-scores, ratios against a reference model, ensembles thereof) suffices as
the control signal for every plasticity and memory decision the system makes:

- **when to learn** — gradient updates gated on the error signal clearing a
  data-dependent threshold (rolling quantile, fixed bar, adaptive
  homeostat), applied at any granularity (token, span, chunk, batch, layer,
  parameter group);
- **what to remember** — spans, keys, or abstractions selected for storage
  in any external memory when the signal spikes;
- **when to consult** — retrieval from any external store triggered by the
  signal, with the retrieved content folded back into the stream (as tokens,
  as state perturbations, or as auxiliary inputs) with or without a gradient;
- **when and how much to sleep** — offline replay of stored material dosed
  by measured marginal benefit (a dividend monitor), throttled to zero when
  the benefit is spent;
- **how curious to be** — the gating threshold itself regulated by the
  statistics of the signal (homeostatic target rates, non-stationarity
  detectors), including the selection *between* input sources by per-source
  signal statistics.

This applies to any bounded-state sequence model (linear SSMs and gated
variants, RNNs, hybrid attention/SSM stacks), any training regime (from
scratch, continued, fine-tuned), and any stream (text, code, sensor data,
multimodal token streams). Measured instantiations: `src/pos_run.py` (gating,
~25% gradient tokens at ≈0.97–1.00 of full-gradient learning),
`src/pos_index.py` (storage + recurrence probes), `src/closed_loop.py`
(consultation), `src/pos_sleep.py` / `src/pos_sleep_cycles.py` (dosed
replay; the dividend life-curve), `src/active_sourcing.py` (source
selection).

**The calculus is locked as a composition (P54,
`results/chimera_v1.json`):** one process making every plasticity and
memory decision from its own surprise beats every single organ and every
ablation on all three continual axes raw (forgetting 0.381 vs 1.306
against the fixed-schedule replicate arm, plasticity 0.672 vs 0.631,
recovery −0.040 — better than pre-shock) at 41% fewer gradient tokens,
and degrades sublinearly with exposure (0.259→0.381 across 6.7×) where
the fixed-schedule arm degrades near-linearly (0.189→1.306): composition
is the stabilizer. Two further measured characterizations sharpen the
calculus. The signal selects **novelty, not difficulty** (P58,
`results/surprise_filter.json`): surprise-selected windows carry 1.51×
the first-ever content of seeded-random windows (7.0× on first-ever
token types) at 0.69× the redundancy, seed-robust — the gate is a
knowledge filter. And the dose is a **dial, not an emergent** (P56,
`results/gate_law_width_curve_q08.json`): the cumulative gate rate
tracks the gating quantile alone (rate ≈ 1−q; five q=0.8 runs across an
8× width range land within 0.13pp), while at fixed dose the gated arm's
improvement ratio crosses 1.0 — at d=1024 on one fifth of the gradient
tokens (1.0092), and on the token axis at 909.7M tokens (1.0091):
selection wins the long game on both axes.

**Contemplated variants disclosed here:** per-layer and per-head gating;
surprise signals computed against an exponential-moving-average teacher copy;
gating of optimizer moments separately from gradients; threshold schedules
tied to wall-clock duty cycles ("waking hours"); multi-signal calculi
combining surprise with uncertainty (entropy) and disagreement (ensemble
variance); and — disclosed in full breadth — the **multi-horizon extension**:
a ladder of prediction heads over horizons H ∈ {1, …, unbounded}, each
DEPOSITING a prediction about the stream's future (token statistics, summary
signatures, its own future surprise, or any functional of the future
segment), holding it in any persistence mechanism (the carried state, a
write-once channel, the external index) until the future arrives, scoring it
against the realized stream, and feeding the per-horizon error back as (i) a
gating signal at its own timescale, (ii) a storage/consultation trigger,
(iii) a regime-change early-warning (long-horizon error rises before
short-horizon error under distribution shift), and (iv) a training signal
for the depositing head — so that the present's model of the future improves
from the future's own arrival. This applies to any number of horizons, any
number of parallel input streams, and any composition with the other
foundations (a deposited prediction is content in the F3 sense; a scored
prediction→outcome pair is index material in the F4 sense; the whole ladder
is family-generic in the F5 sense). A living stream is the only setting in
which this loop closes at deployment time — a batch-trained model never
experiences its own future.

## F2 — The exactness license: bounded contraction makes streaming training exact and decouples training layout from deployment layout

**Statement.** For any recurrent operator whose state dynamics are a
contraction with an effectively bounded receptive field (measurably: the
gradient of the output at time t with respect to inputs at t−k decays below
numerical relevance for k beyond a small horizon r), the following are exact,
not approximate:

1. **Detach-carry streaming training**: truncated BPTT with the state carried
   and detached at chunk boundaries reproduces full-window BPTT gradients
   whenever the chunk exceeds r *and a warmup overlap of order r is
   recomputed per chunk* — cosine 1.000000000000 at overlap 16, relative
   error ~5e-7, across every operator and chunk size swept
   (`results/f2_equivalence_sweep.json`, 4 operator configurations × 6
   (chunk, overlap) points).

   The overlap is load-bearing and this is the sweep's sharpest finding:
   it dominates chunk size. With overlap dropped to 0 the relative error
   rises by four orders of magnitude (5e-3 to 2e-1) while cosine falls to
   0.9762 in the worst cell (complex scan, chunk 16). The exactness license
   is therefore a statement about *overlap ≳ r*, not about chunk length —
   a long chunk with no warmup is measurably worse than a short chunk with
   one.
2. **Layout decoupling**: full-sequence forward with zero initial state and
   chunked-carried forward are the same operator — and here the result is
   stronger than "to float precision": with pure detach-carry the logits
   agree to **exactly 0.0** at every operator and every chunk size down to
   16 (same artifact). Measured on the selective scalar scan, the complex
   holographic scan under both of its read paths, and a phase-off control
   that isolates the scan from the binding. Therefore the *training*
   computation graph
   (full-sequence, arbitrarily long, gradient reaching every write) and the
   *deployment* computation (chunked, O(chunk) memory, unbounded length) may
   be chosen independently — train however the gradient needs, deploy
   however memory requires. This license is what turns gap-curriculum
   training into deployable streaming skills (`src/holo_gap_knee.py`,
   `src/holo_mag_read.py`).

Applies to any member of the affine-scan family (`src/ssm_family_reduction.py`
reduces Mamba/S6, S5, LRU to one operator at ~1e-15) and to any stacked
combination with pointwise/feedforward layers.

## F3 — Phase–magnitude separation in bound complex states: content is written, persistence decays, and the two never mix

**Statement.** In any bounded recurrence that stores associations as
complex-valued (or otherwise rotational) accumulations `S_t = γ_t·S_{t-1} +
a_t·e^{iφ(x_t)}` with real decay γ, the stored *content* (the phase — the
key binding) is invariant during input silence, while only the *magnitude*
(the persistence) decays as ∏γ. Consequences, each measured or in
measurement:

- recall over silence exhibits a **knee, not a slope** — flat until the
  magnitude crosses the readout floor, at G* ≈ ln(margin)/(1−γ);
- the knee is **movable by any lever that raises effective γ on
  non-informative inputs** (learned input-gating, curriculum, explicit
  γ-bias initialization of a subset of channels) — measured arc: knee 32→512
  in one day (`analysis/HOLO_STREAM_VERDICT.md`);
- the readout floor is **independently movable by magnitude-invariant
  reads** (normalizing |S| before de-rotation, or reading pure phase),
  because the content is intact by construction (`src/holo_mag_read.py`);
- capacity (how many bindings) and persistence (how long) are **independent
  axes**: pair count attacks phase SNR (~1/√P), gap length attacks only
  magnitude.

Disclosed variants: multi-slot and multi-head phase banks; per-channel
γ-kickstart at any bias point; phase-only readouts; renormalization applied
in-state at controlled intervals (a "magnitude refresh") rather than at
read; binding angles derived from learned key projections, from token
embeddings directly, or from external key registries.

## F4 — The two-system law: sharp gated readouts force compounding into an external index

**Statement.** Bounded states read through saturating gates exhibit a sharp
capacity cliff (measured: fidelity 0.99→0.65 across load K/D≈1, slope 1.32
vs 0.57 for linear reads, `src/gssm_potentiation.py`) — above capacity,
stored structure is deleted, not gracefully degraded. Therefore any system
that must *accumulate* unbounded knowledge over an unbounded stream divides
by construction into: (i) a bounded state carrying few live bindings with a
sharp read, and (ii) an external, growable index (symbolic graph, span
store, key-value registry — any persistence layer) written and consulted
under the surprise calculus (F1). Runtime consultation without any gradient
measurably lifts performance far beyond state capacity, dose-dependently: at
P=16, G=32 the state alone reads 0.150, a half-dose reminder 0.340, a full
reminder 0.512, against chance 0.0625 and a random-reminder control at 0.112
(`results/holo_hybrid.json`). Stated with its own limit: that sweep's
pre-registered acceptance bar was hybrid ≥ 0.9 at P=16, and 0.512 does not
clear it — consultation lifts the read far above the state's capacity
without restoring full fidelity at high load.

Training *with* in-stream consultation closes the gap at low load and moves
it at high load, but does not remove it: reminders are then read at
**0.998 / 0.995 at P=2** (vs 0.715 / 0.843 read-time-only) and **0.487 /
0.505 at P=16** (vs 0.450 / 0.512) — near-perfect where the state is not
contended, essentially unchanged where it is
(`results/holo_reminded.json`; its own P18 bar of ≥0.85 at P=16 is likewise
not cleared). The reading across both artifacts: **the ceiling at
high load is state interference, not reading.** That is precisely why
compounding has to leave the state — which is this foundation's claim. Disclosed variants: shared indices across multiple
independent organisms (collective memory); freshness-weighted and
dividend-monitored replay from the index; index entries as reminders
injected in-stream at any position; consultation policies trained end-to-end.

The external store also compounds **offline**: a frozen span store
harvested under the calculus (a knowledge file, sha256-fixed) is not
merely fertile but **tappable** (P55, `results/keyed_file.json`) —
after dosed replay a reader completes the file's own entries from their
first halves 0.264 nats better than its no-file twin (5.3× the
pre-registered bar), with specificity +0.218 over never-harvested,
length-matched spans from the same stream region: dosed replay alone
stores entry-level, keyed content, no index-mediated read required at
this dose. Which content *earns* a file is itself measured under the
calculus (P58; the composed filter-vs-file experiment is registered as
P64).

## F5 — Operating modes are family-generic: the calculus attaches to the operator class, not to one architecture

**Statement.** Because the linear-SSM family reduces to a single affine scan
operator (machine-precision reductions in `src/ssm_family_reduction.py`),
every operating mode above (F1's gating/storage/consultation/sleep, F2's
streaming exactness, F3's phase memory where the state is complex) is
defined on the *family*, not on any single parametrization. A
surprise-gated Mamba, a sleeping S5, an LRU with a bolted-on phase bank and
an external index are instances of the same disclosed system.
(`src/pos_family_transfer.py` measures the transfer directly.)

## F6 — Train short, deploy unbounded: shift-equivariance plus the exactness license remove every length wall

**Statement.** An operator with no absolute-position term (NoPE; the only
index-dependence is through lags) is in-distribution at every sequence
length; combined with F2, training at tiny horizons (T=32, gap≤12) yields
deployment at unbounded horizons. Measured, on a model trained at T=32:

| eval length | × training length | PPL ratio | peak RSS |
|---|---|---|---|
| 131,072 | 4,096× | ×0.896 | 1.93 GB |
| 1,048,576 | 32,768× | ×0.886 | 2.09 GB |
| 4,194,304 | 131,072× | ×0.825 | 2.13 GB |
| **16,777,216** | **524,288×** | **×0.803** | **2.48 GB** |

(`results/scale_to_a_million.json`, chunked+carried per F2.) Perplexity does
not merely hold — it *improves* monotonically with length, while memory grows
by 0.5 GB across four orders of magnitude. The position-free variant is what
carries this: at 256× the same comparison gives NoPE ×0.973 against ×4.23 for
the position-bearing Selective arm and ×11.25 for Pure
(`results/length_extrap_v2_extreme.json`) — the wall is the position term,
not the length. Extended independently by the lifetime organism: **7.4B+
tokens streamed through one 1.7M-parameter life** at RSS 0.69–0.83 GB the
entire way — the stretch from 0.87B past 7B in a single OS process over
~12 days, with the one stall (an upstream HF hang) self-healed from the
run's own atomic checkpoint, 51,200 tokens replayed, ~3 minutes lost
(`results/lifetime_7b_curve.json`, series in `lifetime_7b_series.json`;
the 1B crossing preserved in `lifetime_billion_status.json`). Keyed recall
stays flat across 8 detached chunk boundaries. Length, in this architecture class, is a
wall-clock quantity, never a memory or validity quantity.

## F7 — The portable organism: the living state is a small, serializable, migratable, shardable, seedable asset

**Statement.** In this architecture class the complete living system — weights,
optimizer moments, carried state, gating windows, span store, index — is a
small serializable artifact (measured: ~53 MB at the reference scale), and
every operation a distributed deployment needs is either measured or a
composition of measured primitives:

- **live migration** — checkpoint/resume is exact (crash-resume with tail
  trim, `src/pos_run.py`); a transplanted state heals against any weights of
  the same lineage within ~256 tokens (P23) while stored content survives the
  move at recall 1.0 (P26); migration across machines and across CPU
  architectures (ARM↔x86) is therefore a bounded-cost, no-downtime operation;
- **forking and seeding** — a running organism can be forked live (the twin
  experiment, P5: the fork's transient is small and decays), and because the
  artifact is small, organisms can be distributed, mirrored, and seeded like
  files — N replicas from one lineage;
- **organ-level sharding ("each holds a slice")** — unlike dense
  architectures whose parallelism couples through high-bandwidth activations,
  this organism's organs couple through the surprise calculus: spans,
  reminders, prediction→outcome records, and weight deltas — kilobytes.
  The index can live on one machine (shared across organisms, measured:
  P31), the sleep organ on another (replaying from the shared store),
  wake-streams on others; loss of any replica is compensated by the
  remaining ones, and a rejoining replica catches up by snapshot + the
  measured ~256-token heal;
- **offline mode** — when the stream disappears, the organism idles (state
  persists through silence: the carrier measurements) or sleeps (dosed
  replay from its own store, dividend-monitored — measured), and resumes on
  reconnect; connectivity is a duty-cycle input, not a liveness requirement.

Disclosed variants: layer- and organ-level partitions in any mixture;
delta-based weight synchronization between replicas at any cadence;
majority/quorum reads over replica ensembles; heterogeneous fleets (replicas
at different d_model via the growth operator, P24/P27); index-only seeding
(a new organism bootstrapped from a lineage's span store and index alone);
and — disclosed in full breadth — **stop-free streaming migration**: the
source organism never pauses; a snapshot flows to the target while the
source keeps living, the target resumes and REPLAYS the source's
subsequently-consumed input (deterministic resume — measured to six decimals
across CPU architectures — makes the catch-up provably exact), and cut-over
happens at parity with zero downtime; iterated at any cadence this becomes
CONTINUOUS REPLICATION, in which the organism exists as the stream of its
own deltas rather than as a file at any location — never 100% transferred,
never final, always alive — with the surprise calculus itself dosing the
replication bandwidth (only gated chunks produce deltas).

**The replication rate condition (measured, P50).** Catch-up replay was
long carried as "≈2× live cost" (P39). Decomposed at production cadence
(8/64/128, warmup-robust medians, `results/replay_law_prod.json`), the
factor dissolves: **replay compute is 1.03× live** from a local token
cache (1.10× re-pulling the stream), bit-equal outcomes in every cell.
The remaining catch-up cost is a FIXED overhead per catch-up — process
cold-start + snapshot load (~6s) plus a step-shaped stream fast-forward
cost (~9–10s once the skip distance crosses a shard threshold; NOT
proportional to the life replayed, `results/replay_law.json`). The
historical 2× was an instrument artifact on top: multi-core CPU-clock
inflation of 13.6–15.5× wall persists on the 16-core reference machine
against every thread pin, so CPU-time ratios there compare fictions.
Consequence, stated as the rate condition: a replica following a live
source converges to a **fixed equilibrium sync-debt**
T∞ = rate_A · fix / (1 − rate_A · c_replay) — chunks of lag set by the
fixed costs alone, independent of cycle length — rather than diverging
under a 2× compute wall. Continuous replication is rate-feasible on
equal hardware; the fixed costs, not the compute, are the design target
(cache the tokens, keep the process warm, amortize the fast-forward).

**The two replication regimes (measured, cross-machine staging).** Run
against a LIVING source over a real network, replication has exactly two
modes, both now carrying numbers. (1) **Chase** — the follower recomputes
the source's stream: feasible iff rate_source · cost_follower < 1. On an
unequal pairing (ARM source at 200 chunks/s vs an x86 follower at
11.8 ms/chunk under nice) the product is 2.35 and the standing debt
DIVERGES (measured live: 10,000 → 19,000 chunks in one cycle), with the
debt recursion confirmed in-run to 3% (predicted 19,580, measured
19,000). (2) **Snapshot-sync** — the follower adopts the source's state
directly: one ~53 MB transfer erased a 45,000-chunk deficit instantly;
the transfer is the settlement, not the compute. A fleet design follows:
chase only on equal-or-faster hardware (the condition is computable in
advance from two rates), snapshot-sync everywhere else — and the
deterministic exact-resume property (measured at 50M scale) is what
makes both modes lossless.

---

## Interactions (disclosed as a system)

The six foundations compose into a single organism — one process that
streams unboundedly (F6) at exact constant memory (F2), decides every
plasticity and memory action from its own surprise (F1), carries live keyed
bindings through silence in phase (F3), accumulates unbounded knowledge in
an external index it writes and consults in flight (F4), and does all of
this identically across the SSM family (F5). The composition is no longer
a design note: CHIMERA — one process, every organ driven by the one
surprise signal at production cadence — is measured and locked
(`results/chimera_v1.json`, P54 4/4): it beats every single organ and
every ablation on all three continual axes at 41% fewer gradient tokens,
and its degradation with exposure is sublinear where the fixed-schedule
alternative is near-linear. Composition is not overhead on the
foundations; it is the stabilizer that makes them add.

*Every claim above is either measured in this repository (file references
inline) or explicitly disclosed here as a contemplated variant. Measured
numbers carry their reproduction scripts; the registered-prediction ledger
(`analysis/PREDICTIONS.md`, scored by `src/score_predictions.py`) documents
which quantitative expectations survived contact with the data — including
the ones that did not.*
