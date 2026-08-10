# o1-state — the O(1)-state organism

**An O(1)-state sequence model that consumes an unbounded stream at constant memory, decides from its own surprise when to learn, carries keyed holographic memory through silence, consolidates in dosed sleep, and consults an external knowledge index at runtime.**

**Author:** David Tom Foss · **First disclosed:** 2026-06-26, extended continuously (commit timestamps are the record) · **License:** Apache-2.0

> This README is a **timestamped public disclosure** (prior art). Every claim below is a
> number we measured, with the exact script that reproduces it. The dates, the code, and the
> result JSONs in this repository are the record. The seven primitives underlying everything
> here are formalized in their broadest form in **[FOUNDATIONS.md](FOUNDATIONS.md)** — read
> that first for the claims; read on for the evidence. The prediction ledger
> ([analysis/PREDICTIONS.md](analysis/PREDICTIONS.md)) runs to P66, most already scored —
> confirmations and falsifications at the same strength, mechanically re-checkable via
> `src/score_predictions_v2.py`.
>
> **o1-state builds on [GSSM](https://github.com/DT-Foss/gssm)** and inherits its full commit
> history. GSSM is the architecture — the bounded reproducing-kernel SSM operator, the
> length-invariant NoPE-selective recurrence, the `O(log T)` parallel scan. o1-state is what
> that architecture becomes when you let it live: constant-memory streaming (training and
> inference), surprise-gated plasticity, holographic memory carried through silence, sleep
> consolidation, and a runtime-consulted knowledge index.

---

## The seven foundations

Formal statements, each over the *class* of systems it applies to, in
[FOUNDATIONS.md](FOUNDATIONS.md). One line each, with the measured anchor:

| # | Primitive | Measured anchor |
|---|---|---|
| F1 | **The surprise calculus** — the learner's own prediction error, at every horizon: a ladder of deposited predictions about the future, scored when the future arrives, controls when to learn, store, consult, sleep | POS: ~25% gradient tokens ≈ 100% of full-gradient learning, crossing ABOVE 1.0 at 909M tokens and at d=1024 (§9); locked as a composition (§17); the rate is a dial, the gate a measured novelty filter (§9, §18); span store, sleep dosing, runtime consultation (§11–12) |
| F2 | **The exactness license** — bounded contraction makes detach-carry streaming training *exact* and decouples training layout from deployment layout | grad-cosine 1.0000, max-abs-delta 0.0 (§4); full-seq ≡ chunked+carried to float precision (§10); any state×weight mismatch heals in 256 tokens (§14) |
| F3 | **Phase–magnitude separation** — in complex bound states, content (phase) is written and never evolves; persistence (magnitude) decays; the two never mix | zero-drive phase invariance \|Δφ\| ≈ 1e-8; knee moved 32→2176+ via the disclosed clamp+refresh pair, an interaction of both axes (§10) |
| F4 | **The two-system law** — sharp gated readouts have a capacity cliff, so unbounded accumulation belongs to an external index the stream writes and consults | cliff slope 1.32 vs 0.57 (§8); hybrid recall at P=16: base 0.10–0.20 → 0.41–0.62 (§12); reminded reads ~1.0 (§12); the frozen file answers by key (§18) |
| F5 | **Family-generic operating modes** — every mode above attaches to the affine-scan operator class (Mamba/S6, S5, LRU), not to one architecture | family reduction ~1e-15 (§1); POS-on-S6 at 0.98× of POS-on-GSSM, GSSM ahead 0.156 nats head-to-head (§13) |
| F6 | **Train short, deploy unbounded** — no absolute position + exactness ⇒ tiny training horizons, unbounded deployment | ×0.98 PPL at 4096×; 1B-token eval at flat 4.36 GB and a 7.4B+-token training LIFE at 0.69–0.83 GB (§4, §19); recall flat across 8 detached boundaries (§10) |
| F7 | **The portable organism** — the living system is a ~53 MB serializable asset; organs couple through kilobytes (spans, reminders, deltas), not activations: migratable, forkable, shardable, seedable, offline-capable | live ARM→x86 mid-stream migration behaviorally identical to six decimals (§15); stranger verification across ISAs, 19/20 bit-exact with the one divergence pinned to a ULP (§16); a 7.4B-token life forked and measured (§19); fork cost measured (twin, §9) |

---

## Thesis

A mind does not need to hold its whole history to keep thinking. **Two systems and a law.**

**The living now — an O(1) stream.** The GSSM gated recurrence `zₜ = γₜ·zₜ₋₁ + aₜ` consumes
an *unbounded* token stream at **constant memory**. Memory is decoupled from length: the
corpus stops being an object and becomes an iterator — C4, a web-scraper, the whole internet,
all at constant RAM. The state stays awake without input (idle-persistence), forgets in a
controlled way (`γₜ`), and reads sharply through a nonlinear gate.

**Knowledge — a growing external index.** A `.causal` knowledge graph (deterministic
multi-pass inference) holds what the stream has learned and grows as it runs. When the state
hits high surprise, it queries the graph and folds the retrieved association back into the
stream **without a gradient step**.

**The law — a threshold.** Between isolated knowledge and one connected mass there is a
critical point; below it reinforcement is local, above it capability compounds. We measure it
three ways. The dynamical threshold lives in the index; the sharp gated read lives in the
state.

Underneath, **GSSM-Selective is the general affine reproducing-kernel operator of the
linear-SSM family** (Mamba/S6, S5, LRU are parametric special cases of one prefix-scan
operator). That kernel structure — bounded state, `O(log T)` scan, KV-cache-free inference,
shift-equivariance in time — is exactly what makes the O(1) stream possible. The mathematics
(Möbius coupling, doubly-stochastic spectra, non-reversible lifted Markov chains) is archived
with permanent DOIs — see [PAPERS.md](PAPERS.md).

---

## The five verified contributions (the GSSM core)

Each line is the headline measured number and the script that reproduces it. All runs:
PyTorch 2.9.1, offline, Apple Mac (M-series) CPU/MPS.

### 1 — RKHS / kernel unification: one operator, three switches

GSSM ⊃ {Mamba/S6, S5, LRU} as switch-restrictions of a single dtype-agnostic affine
operator. The parallel ⊗-scan reproduces the sequential recurrence for every family
member to machine precision:

| Family member | State algebra | `A_t` input-dep? | Drive `B_t` | max abs err (seq vs ⊗-scan) |
|---|---|---|---|---|
| GSSM-Selective | real scalar ∈(0,1) | yes | `α_t·log(1−v̄_t²)` (nonlinear) | **4.44e-16** |
| Mamba / S6 | real diagonal ∈(0,1)ᴺ | yes | `Δ_t·B̄·u_t` (linear, input-scaled) | **8.88e-16** |
| S5 | complex diagonal `exp(ΔΛ)` | no (LTI) | `Δ·B·u_t` | **1.26e-15** |
| LRU | complex diagonal `e^{−ν+iθ}` | no (LTI) | `B·u_t` | **8.88e-16** |

**Real max 8.88e-16 (Mamba/S6 row; GSSM-Selective 4.44e-16), complex max 1.26e-15 — the whole
family reduces to ~1e-15.**
→ `src/ssm_family_reduction.py`

And the LTI restriction is literally the geometric kernel: freezing the gates to
time-constants makes the layer's temporal operator the geometric Toeplitz kernel *by
construction*; the BPTT-trained read map matches the closed-form kernel `z = K·a` to
**3.55e-15 at d=512** (width-invariant: 1.78e-15 @ d128, 1.78e-15 @ d256), with per-channel
read scale ≈ 1.0 (no extra readout). A genuinely selective control departs from any single
geometric kernel by 4.87e-2 — a **control/match ratio of 1.37e13** that proves the match is
structural, not a coincidence.
→ `src/constant_gate_kernel_match_width.py`

### 2 — Parallel scan: `O(log T)` doubling scan, exact to the loop

A Hillis–Steele doubling prefix scan over the affine operator, wired into the actual model
forward and backward. Forward and **gradient** are identical to the sequential reference loop:

- **fp64:** forward max abs err **1.67e-16**, gradient max abs err **3.55e-15** (per-param ≤2.7e-15).
- fp32 (training dtype): forward 1.49e-7, gradient 1.91e-6 — below the 1e-5 gate.
- Training loss curves (sequential vs parallel) coincide to 4.8e-7 over 12 steps.
- Scan depth is logarithmic: T=128→7, T=512→9, T=1024→10, T=2048→11, T=4096→12.

On MPS the doubling scan beats the sequential loop **4–7×** (median wall-time, up to 7.2× at
T=4096) while passing the correctness gate at every T. Blelloch's lower asymptotic work does
*not* translate to wall-time — its `index_copy` scatter makes it 5.4–21.5× slower than
doubling, so doubling is the shipped default. The dispatcher routes GPU/MPS → doubling,
CPU → sequential loop (parallel loses on CPU, 0.2–0.8×), with zero edits to the frozen
reference layer.
→ `src/parallel_scan_integration.py`, `src/scan_dispatch.py` (+ `src/test_scan_dispatch.py`)

### 3 — Holographic recall: breaking the scalar-recall wall

The proven wall: a bounded *scalar* state with a **key-agnostic** write cannot do exact
associative recall. On MQAR (5 seeds, len-256 eval, chance 1.56%), Selective and the
holographic-write-OFF ablation both sit at **~1.6%** — the wall, confirmed.

The lever: a **key-conditioned holographic complex write**. Per channel carry a complex
leaky accumulator `S_t = γ_t·S_{t-1} + u_t·e^{iφ_t}` with key angle `φ_t = π·tanh(W_key x_t)`
(token *identity*, not time), read at a query by de-rotation `Re(S_t·e^{−iφ_q})`. Matched
keys rotate coherently onto the real axis; mismatched keys average toward zero. This is the
complex analogue of attention's outer-product KV binding.

| Arm | MQAR recall (mean ± std, 5 seeds) |
|---|---|
| Attention (validity gate) | **0.994** |
| Selective (scalar baseline) | 0.017 |
| Holographic write OFF (== Selective) | 0.017 |
| **Holographic write ON (key-conditioned)** | **0.089 ± 0.019** |

**Key-conditioned holographic write: 1.6% → 8.9% ± 1.9%, +7.2 pp**, clearing both chance
(1.56%) and the noise band (3.72 pp), with the attention validity gate at 0.994 (so the
GSSM numbers are valid, not a broken harness).
→ `src/holographic_gssm.py`, `src/holographic_mqar_run.py`

**What this is.** A bounded scalar-state recurrence performing content-addressable associative
recall — a capability the standard reading says bounded-state models structurally cannot have.
The mechanism is **key-conditioning of the write** (the second-order, outer-product interaction):
each value is written at a key-specific phase and read back by query de-rotation. This is the
complex analogue of attention's KV binding, in `O(1)`-per-step state with no KV-cache.

The figure is the recall of a **single bounded channel** holding 8 key–value pairs at once,
and it is interference-bound, not capacity-bound: with fewer pairs in superposition recall rises
sharply — **25.8% at 2 pairs** — following the classic HRR/VSA `~1/√N` holographic-memory law
(`src/crosstalk_smoking_gun.py`). Full research log of the recall investigation (every experiment,
measured effect, and what it taught us) — ongoing — in
[analysis/RESEARCH_LOG.md](analysis/RESEARCH_LOG.md).

**Mechanism — polyphase quadrature-bank read cancels mismatched-key crosstalk.** Reading the
bounded complex state with `n ≥ 3` equally-spaced phase rotations turns the `O(√N)` interference
floor into an `O(1)` subtractable DC pedestal, via the constant-power identity
`Σ_k cos²(x + 2πk/n) = n/2` for `n ≥ 3` (identity verified in `src/holographic_z3.py` to ~1e-15).
The single de-rotation read is exactly the `n = 2` case the identity proves *cannot* cancel; an
`n ≥ 3` polyphase read decoheres non-matching keys while preserving the matched key. The
read-combine verdict for this bank is in
[analysis/Z3_COMBINE_VERDICT.md](analysis/Z3_COMBINE_VERDICT.md).
→ `src/holographic_z3.py`, [analysis/Z3_COMBINE_VERDICT.md](analysis/Z3_COMBINE_VERDICT.md)

**Theory — per-channel state rank is the associative-recall capacity coordinate.** The scalar
selective state is rank-1; the bounded-state arm is a fixed `(n_heads, d_k, d_v) = (4, 32, 32)`
memory, `O(1)` in `T` and vocab, with attention (rank-`K`) reaching test recall **1.0** as the
validity gate (`results/deltanet_mqar.json`). The ladder that lifts the scalar floor runs by
read rank: scalar rank-1 → complex/phase rank-2 (`src/phase_gssm.py`) → bounded fast-weight
rank-`D` (`src/deltanet_gssm.py`) → attention rank-`K`. Lifting per-channel state rank lifts the
associative-recall ceiling. The rank-1 floor is measured (pure-Selective recall 0.1406 at train
length 64, 0.1445 at test 256) and matches the theorem's prediction, inverting to an effective
binding rank `D_eff ≈ 1` — the limit is a theorem about the operator class, not a training artifact
([analysis/RANK1_CAPACITY_THEOREM.md](analysis/RANK1_CAPACITY_THEOREM.md)).
→ `src/deltanet_gssm.py`, `src/phase_gssm.py`, `results/deltanet_mqar.json`,
[analysis/RANK1_CAPACITY_THEOREM.md](analysis/RANK1_CAPACITY_THEOREM.md)

### 4 — Length invariance: train at T=32, run to T=8192 (256×), perplexity unchanged

![Length invariance: NoPE-Selective is the only architecture that stays flat to 256×](plots/length_invariance.png)

The structural payoff of a bounded state, and a **clean causal ablation**. Train at sequence
length **T=32**, evaluate out to **256× that length (T=8192)** by re-tiling the validation corpus
— same model, same weights, no fine-tuning. The position-free GSSM-Selective (NoPE) holds a
**perfectly horizontal** perplexity curve across the whole span. The *identical* architecture
*with* a sinusoidal positional encoding breaks. The only difference is the PE.

All four arms, same harness, same data, trained at T=32 (×N = PPL relative to T=32):

| eval length | extrap. | **Selective-NoPE** | Selective + PE | Pure (no gate) | Transformer |
|---|---|---|---|---|---|
| T=32 (train) | 1× | 165 (×1.00) | 169 (×1.00) | 231 (×1.00) | 226 (×1.00) |
| T=1024 | 32× | 155 (×0.94) | 196 (×1.16) | 2855 (×12.3) | 307 (×1.36) |
| T=2048 | 64× | 160 (×0.97) | 305 (×1.81) | 2810 (×12.2) | 341 (×1.51) |
| T=4096 | 128× | 159 (×0.96) | 473 (×2.80) | 2774 (×12.0) | **crashes** |
| **T=8192** | **256×** | **160 (×0.97)** | 714 (×4.23) | 2603 (×11.3) | **crashes** |

**NoPE-Selective is the only flat line in the field: ×0.97 at 256× the training length** (153–165
PPL the whole way, slightly *better* at long T). Every other arm breaks:
- **Selective + PE** — identical to NoPE except for the positional encoding — degrades monotonically
  to ×4.23. Same weights up to the PE, so this isolates the cause: **the PE is what breaks at unseen
  lengths; removing it removes the break entirely.**
- **Pure** (bounded, but without the selective gate) explodes to ×12 — the *selective* gate is what
  makes the bounded state hold; a bounded state alone is not enough.
- **Transformer** degrades ×1.5 and then **cannot execute at all past T=2048**: its fixed sinusoidal
  PE buffer (`max_len`) throws a tensor-size error at T=4096. Position-coding ties a model to a
  maximum length *by construction* — the same failure that crashes Selective+PE without a larger
  buffer. NoPE has no such ceiling; it ran clean to T=8192.

So the result is not merely "GSSM beats a Transformer at length" — it is that **`selective gate` +
`no positional encoding` is the unique combination that stays length-invariant**, and the two
ingredients are both necessary (Pure breaks without the gate; Selective+PE breaks with the PE).

**How far does it go? We pushed it to the wall.** Using the `O(1)` recurrent forward (the
deployment path), the same NoPE model trained at T=32 was evaluated up the length ladder until the
machine stopped it:

![Scaling to the wall: flat PPL to 4096× the training length](plots/scale_to_the_wall.png)

| eval length | extrap. | PPL | ratio |
|---|---|---|---|
| T=8,192 | 256× | 149.9 | ×0.92 |
| T=32,768 | 1024× | 158.3 | ×0.97 |
| T=65,536 | 2048× | 156.8 | ×0.96 |
| **T=131,072** | **4096×** | 160.8 | **×0.98** |

**PPL stays flat (×0.98) at 4096× the training length.** Re-run on WikiText-103 (4M tokens) the
curve is identical — ×0.98 flat through T=131,072 — and a *naive* whole-sequence eval then hits a
**memory** wall at T=262,144 (the activation tensors exceed RAM). But that wall is an
*implementation* artifact, not an architecture limit — and we remove it.

**No length wall: flat PPL to 16.7M tokens at constant memory.** The bounded contraction receptive
field `r` (~5–8 tokens; [analysis/STREAMING_THESIS.md](analysis/STREAMING_THESIS.md)) is the
primitive that turns an unbounded sequence into `O(chunk)` memory. Because the operator only "sees"
the last `r` tokens, any chunking with a left-context overlap **>** `r` reproduces the whole-sequence
computation exactly, while scoring only the new region. So an arbitrarily long sequence is evaluated
by *chunked streaming* — a sliding window with overlap ≫ `r`. Memory is then `O(chunk)`, not `O(T)`,
and length is limited only by *time*. Falsifier: shrink the overlap below `r` and the chunked result
diverges from the whole-sequence result.

![No length wall: flat PPL to 16.7M tokens at constant memory](plots/scale_to_a_million.png)

| effective length | extrap. | PPL | ratio | peak RSS |
|---|---|---|---|---|
| 1,048,576 | 32,768× | 225.9 | ×0.89 | 2.1 GB |
| 4,194,304 | 131,072× | 210.3 | ×0.82 | 2.1 GB |
| **16,777,216** | **524,288×** | 204.8 | **×0.80** | **2.5 GB** |

**16.7 million tokens — 524,288× the training length — at a constant 2.5 GB, and the perplexity
*improves* the whole way (×0.80).** Only one chunk-sized state is ever materialized; the overlap ≫ `r`
makes the chunked stream score the same tokens the whole-sequence pass would, and `scale_to_a_million.py`
validates this directly — the chunked PPL equals the whole-sequence PPL (×1.00) where both fit (T=8192),
and the batched eval is exact, `ppl_batched / ppl_single = 1.00000` on identical scored tokens. The
exactness is independently confirmed on the *training* side: truncated-BPTT with a carried, overlapped
state reproduces full-window BPTT to max-abs-delta **0.0**, grad-cosine **1.0** (`results/streaming_check.json`).
Length is no longer RAM-bounded; with more wall-clock time the same
`O(1)`-state forward streams to a billion tokens and beyond — anyone can push it higher with more
compute. The improving PPL is the model *integrating* causal context across the distance as a noise
filter, not merely "not crashing." (All safety-guarded; the machine stayed >80% free throughout.)
→ `src/scale_to_the_wall.py`, `src/scale_to_a_million.py`, `results/scale_to_a_million.json`

**Doubly `O(1)`: the corpus is just an iterator — flat PPL to 1 BILLION tokens at constant memory.**
The million-token run holds the corpus in a list. The next step removes that too — stream the corpus
*lazily* (HuggingFace `streaming=True`, documents tokenized on the fly into a rolling buffer) and run
the same chunked, now *batched*, eval. Neither the corpus nor the activations are ever materialized in
full, so **effective sequence length is limited only by wall-clock time — never by RAM.**

![No length wall: flat PPL to 1 billion tokens at constant memory](plots/scale_to_a_billion.png)

The same `O(1)`-state model trained at T=32 streamed **1,000,013,824 tokens of C4** — 31,250,432× the
training length — at **constant 4.36 GB** (final PPL 247.5), checkpointing every 50M tokens. Across
all 20 checkpoints the running PPL moved **1.3 points** (247.08–248.38, a 0.52% band) and the peak RSS
moved **0.08 GB** (4.28–4.36). Two flat lines across a billion tokens; 153 minutes on one Mac mini that
never approached its 16 GB. The run holds chunked windows (chunk 8192, overlap 128 ≫ the receptive
field) as the only state ever materialized — streaming is constant-memory, not an unbounded
approximation. The committed exactness guarantee is the training-side equivalence (truncated-BPTT carry
vs full-window BPTT, max-abs-delta 0.0, grad-cosine 1.0; `results/streaming_check.json`).

| effective length | extrap. | corpus | PPL | peak RSS |
|---|---|---|---|---|
| 100,000,000 | 3,125,000× | C4 streamed | 247.6 | 4.3 GB |
| 500,000,000 | 15,625,000× | C4 streamed | 247.5 | 4.4 GB |
| **1,000,013,824** | **31,250,432×** | **C4 streamed** | **247.5** | **4.36 GB** |

Because the corpus enters only as an iterator, C4 is interchangeable with any token stream: a web
scraper, a live feed, the whole internet. That is the real claim, of which every length number here is
evidence: **constant-memory consumption of an unbounded stream** — a model that does not load a context
but *consumes a stream*. The thesis and its consequence are in
[analysis/STREAMING_THESIS.md](analysis/STREAMING_THESIS.md). Confirm the run without re-running it:
`python src/verify_billion.py` checks every claim against the committed JSON (exit 0 = all pass); the
raw run log is committed verbatim.
→ `src/scale_to_a_billion.py`, `src/plot_billion.py`, `src/verify_billion.py`,
`results/scale_to_a_billion.json`, `results/scale_to_a_billion.run.log`

**Living-stream: constant-memory TRAINING + a state that lives through silence.** The billion-token
result is *eval*. The same persistent state also makes *training* O(1), and lets the model remember
across a pause in the input — two things a turn-based, KV-cache model structurally cannot do.

**The 7-billion-token life (running).** A 1,713,673-parameter organism has now *learned online*
through **more than 7.4 billion streamed tokens** in one continuous life — no dataset stored, no
epochs, no growing context — at a process memory that never left the band **0.69–0.83 GB**. The
stretch from 0.87B past 7B ran in a *single OS process* for ~12 days without any intervention; the
one stall in the life (an upstream HF-stream hang) was self-healed from the run's own atomic
checkpoint, resuming 51,200 tokens back with ~3 minutes of stream lost. The cost of having lived
70,000 books is zero memory. (Stated plainly: at 1.7M parameters the loss EMA is at its capacity
floor, 4.25 → 4.17 over the logged stretch — this artifact measures the *constancy of resources
over experience*, not continued learning.) The life is now also an experimental substrate: its
checkpoint was forked read-only at 7,442,664,960 tokens and measured against a fresh twin — the
age axis, §19.

![The 7-billion-token life: constant memory across the whole of experience](plots/lifetime_7b.png)

→ `results/lifetime_7b_curve.json`, `results/lifetime_7b_series.json`,
`results/lifetime_billion_status.json` (the 1B crossing), `src/plot_lifetime_7b.py`

![Living-stream: constant-memory training and a bit carried through an input gap](plots/living_stream.png)

- **(A) Constant-memory streaming training.** Train from scratch on streamed C4, carrying the
  per-layer state `Z` across chunks and cutting the graph with `.detach()` (truncated BPTT). Held-out
  loss (WT-2 val, never streamed) falls **8.69 → 5.22** over 3M streamed tokens at a committed **peak
  RSS 0.822 GB**. The truncation is *exact*, not approximate: grad-cosine vs full-window BPTT =
  **1.0000**, max-abs-delta **0.0** (`results/streaming_check.json`) — the ~5-8-token receptive field
  throws away no gradient. The `.detach()` carry is the primitive, and the falsifier is measured:
  remove the boundary detach and the no-detach control grows RSS **0.774 → 1.815 GB** over 60k tokens
  as the autograd graph accumulates (`results/streaming_nodetach.json`). The `.detach()` carry is
  exactly what makes training O(1).
- **(D) Idle-persistence.** A 1-bit beacon task — `[beacon][G filler tokens, no beacon][probe]` —
  trained with a gap curriculum. The bit is recalled **perfectly through a 256-token input gap**; the
  decisive control, **zeroing the carried state at the gap**, collapses recall to chance. So the answer
  rode the persistent state across the silence, not local context.
- **(E) The mechanism — a learned write-once-freeze memory register (the carrier / bit-vault).**
  The head-mean γ suggested only short memory (τ≈2) — but that average *hides* the carrier; only the
  per-channel decomposition reveals it. The *carrier* channel (the one whose state correlates with the
  bit, corr −1.00) runs at **γ = 0.9999 (τ≈1000)** with its input gate **shut in the gap (α≈0.005)**
  and **open at the beacon (α≈0.52)**: write-once, freeze, read-on-cue. The class-separation margin
  holds **96.7 %** across 256 tokens. Layer 0 is fast/local (γ≈0.60); Layer 1 holds the carrier — a
  learned division of labour. The falsifier is measured: zeroing the carrier channel collapses idle
  recall (carried **1.0 → 0.505** at gap 256, i.e. to chance) (`results/carrier_probe.json`,
  `results/idle_persistence.json`).

The thesis and the next attacks (adversarial non-ignorable fillers, source hot-swap) are in
[analysis/LIVING_STREAM_THESIS.md](analysis/LIVING_STREAM_THESIS.md).
→ `src/streaming_train.py` (`--train` / `--idle` / `--carrier` / `--check`), `src/plot_living_stream.py`,
`results/streaming_train.json`, `results/idle_persistence.json`, `results/carrier_probe.json`

**Seed-robustness (n=5).** The whole ablation is deterministic across seeds. Over 5 seeds
{1,7,42,123,2024} at 256× (T=8192): Selective-NoPE = **×0.93 ± 0.00** (std rounds to zero — every
seed lands on the same flat line), while Selective+PE = **×7.05 ± 2.34** (breaks on every seed). The
length-invariance is not a lucky run; it is a structural constant.
→ `src/length_seed_robustness.py`, `results/length_seed_robustness_d128.json`

Why it works — and this is **provable, not just measured**: unroll the recurrence and the state
is `z_t = Σ_k α_k·Γ_{k→t}·φ(v̄_k)` with `Γ_{k→t}=∏_{j=k+1..t} γ_j`. Every factor depends on token
*content*; the only index-dependent factor `Γ_{k→t}` depends on `t` and `k` **only through the lag
`t−k`, never through the absolute coordinate `t`**. There is no `g(t)` term — the operator is
shift-equivariant in time *by construction* (the temporal kernel is Toeplitz, Pillar P2). The
contraction `τ<1` keeps the receptive field at ≈5–8 tokens, far inside the T=32 window, so nothing
new appears at T=8192. The smoking gun: NoPE's learned gates are **frozen across 256×**
(γ_mean 0.2252→0.2251, four sig-figs) — the operator is literally in-distribution at every length.
A positional encoding is the *sole* injection of absolute `t`; removing it removes the only length-
dependent term (with PE, the gates visibly drift 0.231→0.356 to compensate, and break). The state
stays `O(1)` in memory at every length; attention pays `O(T)` cache and `O(T²)` compute and must
learn a positional code that fails out of distribution. **This is the axis where a bounded state
wins by construction** — not by more parameters or data, the only lever the large labs have here.
Full derivation, falsifier, and code audit in
[analysis/LENGTH_INVARIANCE_THEORY.md](analysis/LENGTH_INVARIANCE_THEORY.md).
→ `src/length_extrap_v2.py`, `results/length_extrap_v2_extreme.json`

### 5 — Capability boundary: a task GSSM solves at lengths where attention cannot run

Length-invariance is not only a perplexity property — it is a **capability**. On a long-range
state-tracking task (a single register: sparse writes overwrite it, sparse queries read the
most-recent value; the answer can sit arbitrarily far back), train at T=64 and evaluate out to
T=8192 = 128×:

![Capability boundary: NoPE-GSSM holds 100% to 128× while the Transformer degrades then crashes](plots/capability_flipflop.png)

| eval length | extrap. | **NoPE-GSSM** | Transformer (same size) |
|---|---|---|---|
| T=64 (train) | 1× | **100%** | 100% (validity gate ✓) |
| T=256 | 4× | **100%** | 46% |
| T=1024 | 16× | **100%** | 23% |
| T=2048 | 32× | **100%** | **forward pass crashes** |
| T=4096 | 64× | **100%** | **crashes** |
| **T=8192** | **128×** | **100%** | **crashes** |

**NoPE-GSSM holds a perfect 100% across 128× extrapolation.** The same-size Transformer solves the
task at the training length (validity gate: 99.6% — the harness is fair, not rigged), then degrades
to near-chance as positions go out of distribution, and from T=2048 its forward pass **cannot
execute at all** (fixed PE buffer). This is not a perplexity delta — it is a clean *can / cannot*
boundary: the bounded state tracks the register through arbitrary length at `O(1)` memory; attention
both loses the thread and then hits its structural length ceiling. The task is single-thread
state-tracking, exactly the regime where long context is needed and attention fails. (Multi-key
recall is a different instrument with its own characterized ~9% ceiling, Contribution 3 — a
thermometer is not a barometer.)
→ `src/longcontext_tasks.py`, `src/longcontext_run.py`, `results/longcontext_flipflop.json`

The same boundary holds on **needle-in-a-haystack key–value retrieval** — a key:value pair buried
at a random position in a growing filler sequence, recalled at the end. Train at T=64, evaluate to
T=8192 (128×): NoPE-GSSM holds **100% recall at every length**, including the longest gaps; the
same-size Transformer matches to T=1024 and then **cannot run past T=2048** (the same fixed-PE
ceiling). Two independent long-range tasks, one verdict — the bounded O(1) state retrieves across
distance where attention structurally cannot.
→ `src/bench_needle.py`, `results/bench_needle.json`

---

## The O1 contributions (beyond the GSSM core)

What the architecture becomes when the O(1) state runs as a living stream coupled to an
external memory, governed by a measured threshold.

### 6 — Runtime retrieval: the index feeds back into the stream

The O(1) state is paired with an external `.causal` knowledge index, and **surprise is the
retrieval trigger**. (i) Per-token surprise spikes are a gap detector: index-resolvable terms
spike sharply (avg surprise **15.313**) while common words stay low (avg **0.59**), so the
state's own surprise separates what to look up from what it already carries
(`results/pathfinding_bridge.json`). (ii) On a detected gap the pre-gap state is forked, the
retrieved path is injected back **as tokens through the same O(1) state**, and follow-on surprise
drops **without any gradient update** — measured against a with/without control from the *same*
pre-gap state (mean reduction **+0.0256**, helped **27 of 40** probes,
`results/closed_loop.json`). Falsifier: injecting a random non-retrieved path does not lower
follow-on surprise. The living stream consults its external memory in flight — a model that does
not just consume tokens but *looks things up* as it reads.
→ `src/closed_loop.py`, `src/pathfinding_bridge.py`, `src/attic.py`, `results/closed_loop.json`,
`results/pathfinding_bridge.json`

### 7 — A measured capacity threshold, three ways

Between isolated knowledge and one connected mass there is a critical point. We measure it as
a real phase transition, not a metaphor.

**Structural (knowledge graph).** Percolation susceptibility χ **increases with system size N**
— [5.1, 6.6, 18.1, 17.9] over N = [2k, 5k, 10k, 20k] — the finite-size-scaling signature a
smooth crossover cannot produce, PMI-driven, with critical mean degree ⟨k⟩ ≈ 1.
→ `src/percolation_hard.py`, `results/percolation_hard.json`, `plots/night_percolation.png`

**Dynamical (knowledge graph).** With the edge set frozen, reinforcing only the *traversed*
paths raises connected capability **super-linearly** (C: 0.04 → 0.66, logistic with mid-range
inflection). Reinforcing random pairs or a degree-preserving shuffled graph does nothing
(+0.00) across three seeds — it is the structure that compounds, not the act of bumping
weights.
→ `src/reinforcement_loop.py`, `results/reinforcement_loop.json`

### 8 — The threshold in the neural readout: it lives in the gate

In the actual GSSM recurrence, the gated **m·tanh** readout exhibits a **sharp capacity cliff
at load K/D ≈ 1** (max slope **1.32 per unit load**, fall concentrated in a narrow band), where
a linear least-squares readout on the same state only rolls off smoothly (**0.57 per unit
load**, no cliff). At load 1.0 (K=64) the gated read drops to fidelity **0.652** while the
linear read on the same state still holds **0.990** — the cliff is in the gate, not the state.
This is a design law that forces the two-system split: dynamical potentiation requires
recoverable latent structure, which the bounded state does not retain above capacity (present ⇒
clean, over-capacity ⇒ deleted, with no recoverable latent regime). So the compounding belongs
to the external index while the bounded state provides the sharp gated read — the state/index
split is derived from a measured gate property, not assumed. The threshold is a gating phenomenon.
→ `src/gssm_potentiation.py`, `results/gssm_potentiation.json`, `plots/bridge_gssm_threshold.png`

And **one substrate, many readings.** A single bounded **D = 64** state holding **K = 32**
superposed key–value facts is read out at **mean recovery 1.00 with mean crosstalk 0.035**,
against a random-operator null of 0.14 and an operator-distinctness check of 0.035 (the operators
are distinct, not degenerate). Recoverable information scales with the **number of read operators
applied, not with the state dimension**: store one substrate plus N cheap linear-system operators
instead of N states. The mechanism is that information lives in the data×operator interaction, not
in the data alone.
→ `src/operator_readout.py`, `results/operator_readout.json` (K=32, D=64, 2000 trials)

---

## The organism results (the o1-state layer)

The contributions above are the substrate. The results below make it an *organism*: one
process that decides from its own surprise when to learn, what to remember, and when to
sleep — measured under a pre-registered, auto-scored ledger
([analysis/PREDICTIONS.md](analysis/PREDICTIONS.md), scored by `src/score_predictions_v2.py`;
predictions committed before the data, falsifications kept in the record).

### 9 — Plasticity on surprise (POS): ~25% of the gradients, ~100% of the learning

Three arms, one C4 stream, identical from-scratch init (seed 42): A1 forward-only (frozen
control), A2 full-gradient, A3 **surprise-gated** — backward only when the chunk's own NLL
clears a rolling quantile (q=0.75, window 500) of its own history. Final, 909.7M streamed
tokens over 40h, 16/16 integrity checks: **A3 BEAT the full-gradient arm — ratio 1.0091 of
A2's held-out improvement at 25.17% of the gradient tokens** (A3 4.7430 vs A2 4.7782 from
8.6588). The registered prediction's own embarrassment threshold ("ratio > 1.0 would be a
bigger result than the thesis itself") fired. The gate needs no oracle, no second model, no
labels — the learner's own surprise is the signal (F1). The run also carried a **twin fork**
at T+24h (A3's weights into a fresh process, zero carried state): the restart proved FREE —
surprise excess 0.0029 (10–50× below the registered band), identical post-fork gate rates,
converged one chunk after the fork, and the twin finished ahead (4.7365). Full verdict:
[analysis/POS_THESIS.md](analysis/POS_THESIS.md).

**The width curve — four points, and the correction that cleaned it (P42 + MS-H + P56).**
The original three-point curve was published with a note claiming one identical q=0.75 recipe
across widths. The machine-written run configs say otherwise: the d=512 run was launched at
q=0.8 (`pos_run.py`'s default; the narrower runs had the flag set explicitly). The seed-43
hardening runs and the d=1024 run inherited the same default — and thereby completed, by
accident, a 2×4 q-width grid that separates the two variables cleanly:

| d_model | seed | q | gate fires | improvement ratio | per M grad-token |
|---|---|---|---|---|---|
| 128 | 42/43 | 0.75 | 24.7% | 0.9729 / 0.9807 | 0.2969 / 0.3009 |
| 256 | 42/43 | 0.75 | 24.7% | 0.9953 / 0.9951 | 0.2937 / 0.2936 |
| 128 | 43 | 0.80 | 19.9% | 0.9708 | 0.3696 |
| 256 | 42/43 | 0.80 | 19.9% | 0.9841 / 0.9892 | 0.3611 / 0.3629 |
| 512 | 42/43 | 0.80 | 19.8% | 0.9777 / 0.9758 | 0.3547 / 0.3467 |
| 1024 | 42/43 | 0.80 | 19.8% | **1.0092 / 1.0304** | 0.3305 / 0.3278 |

**The gate rate is a dial, not an emergent.** Seven q=0.8 runs across an 8× width range and
two seeds land inside 0.1978–0.1994 (0.16pp scatter); all four q=0.75 runs land inside
0.2466–0.2478. Both q levels are seed-hard, and no rate anywhere shows more than 0.06pp of
seed delta. The 4.9pp "fall at d=512" was the q flip — there never was a width→rate
effect. The earlier ignition-depth mechanism reading of that gap is **retracted as a width
mechanism**: it modeled a config difference as physics (its own forensics said as much — the
fraction of window entries above the *q75* threshold was 0.230 at both d=256 and d=512, which
is exactly what one sees when d=512 is gating on q80). The per-chunk ignition traces remain
valid data about each config's ignition dynamics.

**What the clean grid shows instead — selection crosses the firehose at d=1024, on both
seeds.** At fixed q=0.8 and matched dose (~19.8%, 9.89–9.97M gradient tokens), the
improvement ratio reads 0.9708 → 0.9841/0.9892 → 0.9777/0.9758 (the d=512 valley holds on
both d=256 seeds) → **1.0092 / 1.0304** (mean 1.0198): the gated arm ends up to 0.096 nats
*better* than the full-gradient arm on one fifth of the gradient tokens, and the second
seed lands above the first. The d128→d256 rise replicates at both q levels. Per-token
efficiency at fixed dose falls monotonically with width (0.3696 → 0.3278) — the old
"selection gets cheaper with width" read compared q=0.75 doses against a q=0.8 dose; a
higher quantile doses fewer, more selective tokens at any width. Ratio seed-deltas grow as
width shrinks (d512 0.0019, d256 0.0051, d128 0.0078) while the RATES never move more than
0.06pp with seed — the rate is the hard invariant. One caveat carried openly: all points
share the 50M anchor, and A2 degrades with width at fixed tokens (4.93 → 5.53), so the
anchor sits earlier in a wider model's training.

**The repeat at 50M scale has now run, and it resolves the tension.** Eleven days after the
original, in a different process, with one mid-run stream reconnect, the seed-42 d=512 run
reproduced **all 97,657 chunks identically on every deterministic field** — chunk index,
gate decision, all three surprise streams at print precision, threshold (wall-clock time is
the only field that moves, as it must) — with final heldouts to the last
digit (A2 5.0898, A3 5.1701, cumulative rate 0.1984). Same-seed variance at 50M is **zero**:
the rate is a deterministic function of (seed, width, recipe, stream), not a frozen coin-flip.
The two phenomena coexist cleanly — the 2k-chunk forks were measured in *parallel* forensics
cells (co-load), while the undisturbed production path is deterministic end-to-end, including
exact stream re-instantiation through a reconnect (138,532 documents, same order): provenance
at production scale. **And the seed axis is closed: seed 43 at d=512/50M lands at 0.1986
against seed 42's 0.1984** — the same rate to the third decimal, with the full profile
reproducing (ratio 0.9758 vs 0.9777, efficiency 0.3467 vs 0.3547 per million gradient
tokens). The determinism results stand untouched by the q correction: the 50M gate rate is
deterministic per (seed, q, width, recipe, stream) and seed-robust — and per the grid above,
the width and seed dependence is nil at fixed q. The early lottery is a transient, not a
fate — 2k-chunk rates fluctuate, the lifetime cumulative rate converges to its q's line.
Memory stays flat throughout (0.41 GB at d=256, 1.73 GB at d=512, zero stream reconnects over
138,532 documents at d=256).
→ `results/ignition_forensics.json`, `results/gate_rate_width_probe.json`
→ `results/gate_law_width_curve.json`, `results/gate_law_width_curve_q08.json` (the corrected grid),
`results/pos_d512_status.json`, `results/pos_d256_status.json`, `results/pos_d1024_status.json`
→ `src/pos_run.py`, `src/pos_index.py`, `src/pos_analyze.py`, `src/verify_pos.py`,
`analysis/DECISIONS.md`, `results/pos_*`

### 10 — Keyed holographic recall on the carried stream: a movable knee, and why it moves

The MQAR line (Contribution 3) goes streaming: the complex holographic write runs as a
**stateful layer carried across detach boundaries** (full-seq ≡ chunked+carried < 1.2e-6 —
the F2 license), trained at tiny gaps, evaluated far beyond them:

- **Keyed recall survives silence across chunk boundaries**: P=2 recall 0.79–0.83 through
  G=32 (+25–30 pp over the magnitude-only ablation), P=1 at 1.00 through G=32, and the
  zeroed-at-gap null at chance in every cell of every version.
- **The knee is movable, theory-led** (G* ≈ ln-margin/(1−γ)): train-short-eval-long moved it
  to 32, full-sequence training to 256, the magnitude-normalized read to **512** (seeds 0/1;
  a 4-seed replication puts the M3-recipe knee at 128–512 — the knee's absolute position is
  seed-dependent, the intervention effects are what replicate), and the eval-time
  clamp+refresh pair to **2176 mean / 4096 end-of-range** (same caveat: seed0 4096, seed1 256) —
  a 68×+ shift in two days, each step predicted before it was run. The closing mechanism:
  the filler write is a *double agent* (phase pollutant AND magnitude feeder), and the two
  cures **interact** — cleaning the phase alone starves the magnitude, feeding the
  magnitude alone rescales pollution; only together does the knee jump.
- **The φ-drift falsifier locks the law**: under zero drive the phase is invariant to
  |Δφ| ≈ 1e-8 (machine precision) — content is *written, never evolved*. The deployment
  measurement then reinterprets the far field: real fillers actively pollute the phase
  (α(x_filler) > 0, magnitude grows 35×), so the falloff past the knee is **pollution, not
  decay** — which is why the magnitude-normalized read stops at 512, and α-shut on fillers
  (the measured write-once-freeze register, Contribution 4E) is the next lever.
- The wrong fixes are documented at full strength too: patience curricula *cannot* repair
  multi-chunk training — truncated BPTT is structurally gradient-blind to the write
  (measured twice) — which is exactly why F2's train-anyhow/deploy-chunked decoupling is
  load-bearing.

Full arc with every version, prediction, and falsification:
[analysis/HOLO_STREAM_VERDICT.md](analysis/HOLO_STREAM_VERDICT.md),
[analysis/HOLO_CARRIER_THEORY.md](analysis/HOLO_CARRIER_THEORY.md).
→ `src/holo_stream_recall.py`, `src/holo_gap_knee.py`, `src/holo_mag_read.py`,
`src/phi_drift_probe.py`, `results/holo_*.json`, `results/phi_drift.json`

### 11 — Sleep: dosed replay of self-selected surprises, with a measured life curve

The organism stores its own high-surprise spans (F1) and replays them offline. Measured:

- **Replay of own surprise spans beats fresh data** at equal gradient budget (paired
  +0.078 nats early), and beats the model's own sampled "dreams" — self-generated data is
  self-confirmation (the generator finds its own dreams *less* surprising than the world,
  NLL 4.71 vs 4.77), so **storage stays**. The dream generator is killed by evidence, and
  the kill is a result.
- **Consolidation, not relearning**: the stored spans are no longer surprising to the mature
  snapshot, yet replaying them still wins — sleep stabilizes what was learned.
- **A dose law and a life curve**: overdosed replay overfits (the budget must be
  volume-coupled); the dividend is strong early (+0.033 @10M tokens), positive mid-training,
  spent at maturity — so the sleep organ runs on a dividend monitor, not a fixed cadence.

→ `src/pos_sleep.py`, `src/pos_sleep_cycles.py`, `src/pos_dream.py`, `results/pos_sleep*`

### 12 — The two-system law, closed loop: index + state, reminded reads at ~1.0

Above the state's capacity cliff (Contribution 8), the external index carries the load —
measured end-to-end on recall. At P=16 pairs (far above state capacity; state alone
0.10–0.20 across cells), runtime injection of index content lifts recall to **0.41–0.62**
— a consistent 2–4× lift, dose-dependent, with the random-injection control at floor. Training *with* stochastic consultation makes the read
near-perfect: **0.99–1.00 at P=2** (vs 0.72–0.84 un-reminded) — the model learns to *be
reminded* (F4). Held-out-key controls kill the lookup-table explanation: binding generalizes
to never-trained keys.
→ `src/holo_index_hybrid.py`, `src/holo_reminded.py`, `src/holo_heldout_keys.py`

### 13 — Family transfer: the operating mode belongs to the operator class

The POS recipe run identically on GSSM-Selective and on a parameter-matched (scan ratio
1.0016) Mamba/S6 configuration of the same codebase — same stream, same seed, same gate, 6M
tokens per arm: **POS-ratio S6/GSSM = 0.98** (GSSM 0.952 at 22.6% gradient chunks, S6 0.934
at 23.1%) — the gating benefit is a property of the *family* (F5), not of one architecture.
And the head-to-head at full gradient lands as a lead, not a tie: **GSSM-Selective beats the
S6 configuration by 0.156 nats** (5.18 vs 5.33 held-out) at identical pipeline, tokens, and
seed. Equivalence gates: full-seq ≡ chunked+carried at max-abs-delta 0.0 for both
architectures. Registered as P22, confirmed on both parts.
→ `src/pos_family_transfer.py`, `src/ssm_family_reduction.py`, `results/pos_family*.json`

### 14 — The deployment primitives: growth, update, and the two-timescale law

What it takes to *operate* a living stream — measured, each half of the law with its own
falsifier:

- **Growth without restart (the brain surgery).** Function- and state-preserving widening
  (channel duplication d64→d128, carried Z migrated) on a live C4 stream: surgery
  equivalence **6.7e-6 on the trained model** (stateless and chunked), no post-surgery
  transient (0.046 nats, conservatively measured *including* an optimizer reset), and
  **growth beats restarting a fresh d128 by 0.127 nats** at the same post-surgery token
  count. Migrating the Adam moments through the same duplication map (gradient transforms
  derived numerically; commutation `grow∘step == step∘grow` holds on all 29 parameter
  tensors) removes 87% of the remaining capacity deficit in early measurement.
- **Weight updates are a non-event on the fast path.** Cross-matrix on the living POS
  organism's snapshot archive (weights from 359M tokens × states from 128M/240M/359M/
  zero/shuffled, forward-only): every arm converges to the native trajectory **within 4
  chunks (256 tokens)**; the cold start is not worse than the carried state — the
  fast-path state rebuilds inside one chunk (the ~5–8-token receptive field, F2 measured
  from the other side). **No lock-in, no compatibility risk, nothing to migrate.**
- **The content survives in the slow channel.** The stored beacon bit (the write-once-
  freeze carrier, Contribution 4E) survives the widening surgery at **recall 1.000** and
  survives a weight swap — written under W(T1), read under W(T2), 2× training distance —
  at **recall 1.000 through a 512-token gap**, while zeroing the state collapses to
  chance. The encoding is *redundant* (position + magnitude): the carrier's channel
  address can even relocate across training and the read still lands.

Together: **the two-timescale law of operation** — the fast path forgets by design (so
model updates and model growth are safe, live operations), and everything worth keeping
lives in the slow carrier and the external index, which is exactly the layer the
organism's own machinery (F1, F4) manages. Registered as P23/P24/P26/P27; the
falsifications (P23a, P24c-at-1.2M) are kept in the ledger at full strength.
→ `src/hot_swap_growth.py`, `src/state_weight_swap.py`, `src/beacon_swap.py`,
`results/hot_swap_growth*.json`, `results/state_weight_swap.json`, `results/beacon_swap.json`

### 15 — Collective memory, and the organism as a portable asset

- **One organism's surprises are another's immunity (P31, all four checks).** Organism A
  streams a C4+code mix and stores its surprise spans; organism B, hit by a code shock,
  replays A's shared spans and forgets only **0.67×** of what it forgets with its own spans —
  at *better* plasticity — while token-shuffled A-spans are nearly as useless as no replay
  (content, not regularization). Collective memory across O(1) individuals is measured.
- **Live migration across CPU architectures is a free operation (P38a).** A running organism
  checkpointed mid-stream on Apple Silicon and resumed on an x86 server ends **behaviorally
  identical to six decimals** with its never-migrated twin (heldout 6.182391 == 6.182391,
  every gate decision matched); only the BLAS bit-digest differs, and it does not propagate.
  Locally, checkpoint→new-process→resume is bit-identical line-by-line. The full state —
  weights, optimizer, carried Z, gating windows, store, stream position, RNG — is one
  atomic ~53 MB artifact.
- **The same parity holds across two live machines on a real network (Möbius staging).**
  Organism A streams continuously on the ARM machine and is never paused; organism B rises on an
  x86 server from A's snapshot over real `scp`/`ssh` and chases A's stream position. At the
  one chunk count where both organisms stood (560), the heldout delta is **0.0 — exact**
  (ARM 6.199778 vs x86 6.199778), reproduced across two independent stagings. Parity is
  scored *only* at equal chunk counts: B on 16 server cores outruns A on the ARM machine and
  overtakes it, and a comparison at unequal stream positions is not a parity check at all.
  Multi-cycle parity therefore needs rate-matched machines — the single-cycle number is the
  measured one.
- **Replication has exactly two regimes, both measured against a living source.** The old
  "replay costs ≈2× live" was decomposed and retired (cache-replay compute is **1.032× live**
  at production cadence; the 2× was CPU-clock inflation × a fixed stream fast-forward step ×
  toy-cadence environment). What remains is a clean law: **chase** replication (recompute the
  source's stream) is feasible iff rate_source·cost_follower < 1 — on an unequal pairing the
  product read 2.35 and the standing debt diverged live (10,000 → 19,000 chunks in one cycle,
  the debt recursion confirmed in-run to 3%); **snapshot-sync** collapses any debt instantly —
  one ~53 MB transfer erased a 45,000-chunk deficit, and a follower replayed an entire 45k-chunk
  life over the real network in 458 s. Fleet rule, computable in advance from two rates: chase
  on equal-or-faster hardware, snapshot-sync everywhere else.
→ `src/replay_law_run.py`, `results/replay_law.json`, `results/replay_law_prod.json`,
`results/moebius_rate_check4.json`
- **Where the phase pays rent is now a map, not an anecdote (P32).** A 16-cell
  P_max × d_model sweep (corner lr-controls) kills both one-parameter laws (ratio and
  product) and shows two rent regions — the scarce corner (P_max ≤ 16, d ≤ 64: +11 to
  +32 pp) and a capacity-return region at mid-P_max × large d — separated by a valley.
  The real-text graft result (§ below) is *explained* by this map: it sits in the valley.
→ `src/pos_shared_index.py`, `src/portable_organism.py`, `src/holo_rent_map.py`,
`scripts/moebius_stage.py`, `results/pos_shared_index.json`, `results/holo_rent_map.json`,
`results/moebius_parity.json`

### 16 — The organism gets a body: pixels, worlds, and a real game engine

The unchanged stack — same reader, same gate, same harvest — runs on rendered worlds by
mapping frames to token chunks. Three lives, and the boundary between what transfers and
what does not is now measured:

- **A deterministic world under a fixed policy contains no underivable novelty (P48).**
  Eight wall-following walkers in a shared maze, egocentric views as 256-token frames:
  the reader learns the world to a 61% NLL drop at flat memory, harvests 901 knowledge
  entries, and every sampled entry replays **bit-exact** from (seed, walker, step) — 5/5,
  twice. But the gate goes quiet: after ignition a competent predictor silences every
  transition the policy can produce. The registered transfer clause died on this, as
  registered.
- **Token-sparse novelty is invisible to averaging gates (P49).** A sealed room opens
  mid-life onto colors never streamed before — and at first sight the novelty is 1–3 cells
  of a 225-cell frame: the lane-MEAN gate reads 0.82× (quieter than steady), while the
  token-level harvest sees the room clearly (1,384 entries by end of life). Same event,
  two instruments — embodied gates need foveal resolution, not panoramic means. And the
  architecture clause held: provenance replays exactly **through the world mutation**
  (5/5, with the unseal replayed in the coordinates).
- **A third-party game engine, and the file carries the world (P52).** A fresh reader
  lives 24,000 chunks in VizDoom's `my_way_home` (real 3D, first-person, software we did
  not write), eight bodies as eight headless engine instances. The engine state survives
  `new_episode` — a fired falsifier proved per-episode reseeding does not isolate, so the
  coordinate design was rebuilt (fresh engine per episode, one factory for life and
  replay) — after which **provenance reads 5/5 bit-exact across episodes {0,1,4,9,12}**,
  and the harvested file (24,660 entries, sha256-frozen) **transfers: a fresh reader dosed
  with the file beats its no-file twin by 0.056 NLL on a route neither ever saw**
  (bar 0.02). The learning-depth clause fell short as registered (13.5% drop vs the 30%
  bar at flat 0.011 GB memory — the 12-level token map exhausts quickly; the registered
  falsifier names the token map, not the claim, as the next attack), and the own-vs-fresh
  route instrument was confounded by unmatched frame populations — named, not patched.

- **A stranger on different silicon can check the file (P60).** The community-review
  mechanism, run as a measurement: the ARM-harvested VizDoom file verifies **10/10 entries
  bit-exact on an x86 machine from coordinates alone**; the reverse direction reads 9/10 —
  and the single divergence localizes to ONE token at ONE quantization bin (gray level 2
  vs 1): the float block-mean's 1-ULP reduction-order difference between the two ISAs'
  vector units, caught at a bin boundary by pure recomputation. Within-ISA consensus is
  10/10 on both sides. The fix is a spec sentence, recorded as a standing rule: token
  quantization is integer arithmetic (`(block_sum × levels) // (n_pixels × 256)`),
  ISA-invariant by construction. A verification protocol that finds a one-ULP float
  boundary in someone else's silicon is exactly the review the file format needs.

The refinery loop — harvest → freeze → hash → replay-exact → measurably useful to a
stranger — now holds in text (P47), through a game engine we did not write (P52), and
across instruction-set architectures with the one measured divergence pinned to a named,
fixable float operation (P60).
→ `src/pixel_body_run.py`, `src/vizdoom_run.py`, `src/stranger_verify_run.py`,
`results/pixel_body.json`, `results/pixel_p49.json`, `results/vizdoom_life.json`,
`results/vizdoom_knowledge.jsonl`, `results/stranger_verify.json`,
`results/stranger_verify_arm.json`

### 17 — CHIMERA at decidable exposure: composition is the stabilizer

The composed organism — gate, store, reminder, dosed sleep with dividend monitor, one
process, every plasticity and memory decision from its own surprise — re-ran the shock
protocol at 1,000-chunk phases (6.7× the original exposure, cadence recorded):

| axis | chimera | gate+sleep only | full-gradient | no_monitor |
|---|---|---|---|---|
| forgetting | **+0.381** | +1.306 | +1.836 | +0.619 |
| plasticity | **+0.672** | +0.631 | +0.854 | +0.623 |
| recovery (residual) | **−0.040** | +0.599 | +0.037 | +0.424 |

Chimera beats the best single-organ regime **raw on all three axes at 41% fewer gradient
tokens** (426k vs 718k) — and recovers to *better than pre-shock*. The ablations attribute
in-run: removing the dividend monitor collapses recovery to +0.424; the reminder organ,
undecidable at the original exposure (2 fires), becomes decidable at 12 fires and earns its
place on the plasticity axis (+0.028). The finding that outranks the table: with 6.7× the
exposure, the single-organ regime degrades near-linearly (forgetting 0.189→1.306) while the
composition degrades sublinearly (0.259→0.381) — **the organs stabilize each other; the
whole is the reason the parts keep working**.

**And the curve over the exposure decade (P57).** Re-run at 5,000- and 20,000-chunk phases:
the composition's forgetting fits **log-log slope 0.295** over the four exposure points
(150 → 20,000 chunks: 0.259 → 0.381 → 0.752 → 1.027) — sublinearity is a four-point law,
not a two-point accident. The predicted near-linear contrast arm instead *saturates*
(forgetting has a ceiling and the fixed-schedule arm hits it early, sitting 21–72% above
chimera throughout), the full-gradient arm forgets most of all (1.954 at 4× chimera's
7.8M-gradient-token budget), and **recovery survives the decade**: chimera's residual is
+0.018 at 20,000-chunk phases (returning to pre-shock at 20× the original exposure) against
the no-monitor ablation's +0.651. One honest subtraction, kept at full strength: the
reminder organ's edge vanishes at 20k — its fixed dose does not scale with phase length,
named as that organ's next attack.
→ `src/chimera.py`, `results/chimera_v1.json`, `results/chimera_curve_5k.json`,
`results/chimera_curve_20k.json`, `analysis/CHIMERA_SPEC.md`

### 18 — The file answers by key — and the filter buys memory, not fertilizer

Three results close the loop from gate to file to retrieval:

**Surprise points at novelty, not difficulty (P58).** On WT-103, windows selected by the
gate carry **1.51× the first-ever content** of seeded-random windows — **7.0× on brand-new
token types** — at 0.69× the redundancy, and both verdicts hold against a second random
seed. The instrument is fully deterministic (a running registry of every type and bigram
the stream has shown so far). The gate is a knowledge filter: it lands on the stream's
first encounters.

**The frozen file is tappable (P55).** P47/P52 proved diffuse transfer; P55 asks the
targeted question. After dosed replay of a sha256-frozen file, the reader completes the
file's OWN entries from their first halves **0.264 nats better** than its no-file twin
(bar 0.05), with **+0.218 specificity** over never-harvested, length-matched spans from
the same stream region (bar 0.02). Dosed replay alone stores entry-level, keyed content —
the file is not just fertile, it answers by key. (Run 1 was an instrument null — the probe
filter demanded spans the harvest geometry cannot produce — documented and fixed the same
hour; the producer harvest reproduced bit-identically across all three runs of the recipe.)

**And the two file roles split by harvest policy (P64 + P67).** One producer, two harvest
policies, exactly matched span counts, the producers reproduced bit-identically across five
runs and the verdicts checked on two seeds: the random-harvested file wins *average*-heldout
fertilization ~1.8× on both seeds — and the novelty-stratified eval (P67, the P58 registry
instrument applied to the eval) shows exactly why. Its advantage lives entirely in familiar
content (3.5× in the lowest-novelty tercile), shrinks strictly monotonically as the eval
gets newer (Δ(S−R) −0.053 → −0.027 → +0.001), and the surprise file crosses ahead in the
top novelty tercile. Keyed storage of own entries is content-agnostic — both files clear
the bar decisively on both seeds; a one-seed S-advantage did not replicate and is
withdrawn. **The filter buys the frontier**: novelty-graded transfer plus entry-level
recall — the right harvest for the `.causal` knowledge path, while average-perplexity
fertilizer is harvest-policy-free.
→ `src/surprise_filter_run.py`, `src/keyed_file_run.py`, `src/filter_file_run.py`,
`src/novelty_transfer_run.py`, `results/surprise_filter.json`, `results/keyed_file.json`,
`results/filter_file.json`, `results/filter_file_s43.json`, `results/novelty_transfer.json`

### 19 — The age axis: a 7.4-billion-token life, forked and measured

Determinism makes the project's most expensive asset an experimental substrate: the
uninterrupted multi-billion-token life was forked read-only at 7,442,664,960 tokens
(optimizer state included; the living run untouched beyond one checkpoint copy) and
measured against a fresh 50M-token twin of identical configuration (P59):

- **Age does not shift the rate.** Veteran 29.70% vs young 29.20% on the same fresh-gate
  2,000-chunk probe — Δ0.5pp against a ±2pp bar, at 149× the lived experience. With the
  q-grid this is the dial law's fourth invariance: q sets the gate rate; width, seed, and
  age do not.
- **Experience is immunity.** Under the WT-103 shock the veteran forgets 32% less
  (+0.0845 vs +0.1243) and recovers to BELOW its pre-shock loss (−0.0086) — entering the
  shock 0.167 nats ahead on C4.
- **The plasticity price is real and small.** The veteran learns the shock domain at
  0.744× the young twin's rate — outside the registered 25% band by 0.6pp, so the
  pre-committed follow-up fired: the plasticity-decay curve across six same-recipe ages
  (0.05B–7.44B; four mid-life snapshots were already on disk) is registered as P66 and
  running.
→ `src/aged_brain_run.py`, `src/age_ladder_run.py`, `results/aged_brain.json`

### The method: pre-registered, auto-scored, falsifications kept

Every run above was preceded by a numbered prediction in
[analysis/PREDICTIONS.md](analysis/PREDICTIONS.md) (immutable P-numbers, committed before the
data existed). `src/score_predictions_v2.py` re-scores the whole register mechanically —
at its last full audit: 59 predictions, 75 clause checks, 0 mismatches against the
hand-scored record — and three machine-readability rules are standing policy
(pass-booleans beside raw numbers, artifact paths in scored headers, one verdict word per
clause). Falsified predictions stay in the record with the number that killed them —
the falsify-then-confirm arc of the phase-rent question, the dream generator's kill, the
twice-measured gradient-blindness of multi-chunk training, and the width-rate story
corrected by the register's own provenance trail (the q flip) are documented at the same
strength as the confirmations. Learning-rate controls guard every "X beats Y" claim.

---

## The knowledge index — `.causal` / fabel

The external memory O1 consults (Contribution 6) is a real, peer-reviewed engine, vendored in
`vendor/fabel`. It turns text into a queryable causal knowledge graph **deterministically — no
LLM in the inference loop, zero hallucination by construction, fully air-gappable.** Causal
structure is *measured* in text (causal connectives are finite, patterned, and findable), so a
rule set extracts it and a binary format serves it. This is what makes the index trustworthy
enough to feed back into the stream without a gradient: every retrieved edge is traceable to a
source.

**The pipeline, layer by layer:**

1. **Extraction.** A multi-pass extractor turns a corpus into candidate `(trigger → mechanism →
   outcome)` triplets — semantic chunking, domain detection, quantification.
2. **Validation — the 14-step FOSS Gate.** Fourteen deterministic predicates accept or reject
   each triplet. It is **100% byte-level deterministic across 150 repeated extractions** and
   **model-agnostic** (Qwen-8B / Gemma-2B / Llama-3B all reach perfect consistency despite 9×
   extraction-rate variation — determinism is a property of the validation architecture, not the
   model), at 88% precision on DocRED. *(ICECET 2026.)*
3. **The `.causal` format (dotcausal).** Validated triplets are written to a binary knowledge-graph
   format with **embedded inference**: transitive chains, direction propagation, and fuzzy
   key-matching are **materialized at build time**, so the reader serves inferred edges instantly —
   the graph stays inferenced, no per-load recompute. A richer closure engine (`hsslm`) extends
   this to 5-hop transitive paths, ~18× more reachable edges than the base reader.
4. **Autonomous growth — the gap-driven loop.** The engine detects what it *cannot* yet answer
   (gaps), queries for it, validates and ingests the result, and repeats:
   `Gₖ₊₁ = Gₖ ∪ {τ ∈ Extract(Retrieve(Q(g))) : V(τ), g ∈ TopGaps(Gₖ)}`. The graph spans its own
   domains. This is the symbolic ancestor of O1's neural runtime retrieval.
5. **Conversation / readout.** `fabel.py` / `brain.py` mount one or many graphs and answer from
   them, every answer traceable to a source — the speak path of the index.

For O1, the coupling is **neural**: surprise in the O(1) state *is* a gap signal, and the
retrieved path is folded back into the stream (Contribution 6). The night build also ships a
purpose-built neural index (`src/gssm_causal.py`) — **a single-substrate, multi-channel
co-occurrence graph read by a field-of-view reader.** It is *one* graph carrying superposed
fwd / bwd / near / far channels (the one-substrate-many-states compression applied at *build*
time, not N separate graphs), and its reader acquires the whole neighbourhood *cone* around a
query — an FOV / implicit-learning operator rather than a single-point lookup. A single-graph
multi-channel superposed store read by view-operators is a distinct compression mechanism for
retrieval indices.
→ `src/gssm_causal.py`

**Active sourcing — a self-curated stream (mechanism disclosed, speedup not yet measured).**
Because memory is `O(1)` and the corpus is just an iterator, the model can choose its *next*
source by its own per-source surprise — passive-fixed, passive-random, and active modes, the
active mode pulling from the highest-surprise source. This is the neural gap-driven discovery
loop applied to *data selection* at constant memory. The mechanism and harness are disclosed
here; no committed measured speedup ships, so the disclosure is to the mechanism.
→ `src/active_sourcing.py`

**This engine is not a side project.** It is the core of **nine peer-reviewed papers** accepted at
four 2026 IEEE conferences — including the **IEEE-NANO flagship in Nanjing** — across causal
knowledge extraction, post-quantum cryptanalysis, nuclear knowledge graphs, and a full
**bit-for-bit-validated security assessment of IBM z/OS mainframe infrastructure** (50 findings,
responsible disclosure to IBM PSIRT). The complete list, with venues and DOIs, is in
[PAPERS.md](PAPERS.md). Software: [dotcausal.com](https://dotcausal.com) ·
[github.com/dotcausal/dotcausal](https://github.com/dotcausal/dotcausal).

---

## Reproduce

```bash
# Python 3.12 (tested on 3.12.7), PyTorch 2.9.1, CPU or Apple MPS. Fully offline.
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt        # torch>=2.9, numpy, matplotlib
```

```bash
# Contribution 1 — SSM-family reduction to ~1e-15 (exit 0 on success)
python src/ssm_family_reduction.py
# → results/ssm_family_reduction_results.json   (real 8.88e-16, complex 1.26e-15)

# Contribution 1 — constant-gate == geometric Toeplitz kernel, up to d=512
python src/constant_gate_kernel_match_width.py
# → results/constant_gate_kernel_match_width_results.json   (3.55e-15 @ d512)

# Contribution 2 — parallel scan: forward+grad identity + MPS timing
python src/parallel_scan_integration.py
# → results/parallel_scan_integration_results.json   (fp64 grad 3.55e-15; 4–7× MPS)
python src/test_scan_dispatch.py        # deployment dispatcher, exact to reference

# Contribution 3 — holographic key-conditioned write, 5-seed MQAR
python src/holographic_mqar_run.py
# → results/holographic_mqar.json   (holo_on 8.9% vs floor 1.6%, +7.2pp, gate 0.994)
```

Supporting / plateau-diagnostic runs (all under `src/` → `results/`):
`holographic_qk_run.py` (separate-QK control), `holographic_capacity_run.py` (channel sweep),
`holographic_readout_shootout.py` (readout ablation), `holographic_crosstalk_diag.py`,
`phase_mqar_run.py` (the additive-phase negative this corrects), `mqar.py` (task harness),
`rank_sweep.py` (the relational-rank negative below).

**A kept negative — relational rank (`results/rank_sweep.json`).** Across K = 2…32 the
attention control solves the task at every rank (recall 0.9898 → 0.9997), so the harness is
sound and the task is solvable. Both state arms collapse to chance from K = 4 on (phase
0.046 → 0.015, scalar 0.018 → 0.014, against chance 0.0156), and the cliff ratio
phase/scalar is **1.0** against a pre-registered ≥ 2 — phase buys nothing over scalar here.
The rank check itself is reported as *insufficient data* rather than as a pass or a fail: no
K clears 3× chance for phase with the required ≥ 2 ignited seeds, so no cell was eligible to
score.

---

## Repository layout

```
o1-state/
├── README.md / FOUNDATIONS.md / PAPERS.md
├── reference/               the architecture (frozen reference modules)
│   └── moebius_scan_transformer_selective.py     ← the Selective GSSM layer
├── src/                     experiments + runnable verifications
│   ├── ssm_family_reduction.py, constant_gate_kernel_match*.py   kernel unification (§1)
│   ├── parallel_scan*.py, scan_dispatch.py                       the O(log T) scan (§2)
│   ├── holographic_gssm.py + holographic_*_run.py                key-conditioned recall (§3)
│   ├── length_extrap_v2.py, scale_to_a_million.py, scale_to_a_billion.py   length/stream (§4)
│   ├── streaming_train.py, longcontext_run.py                    living-stream + capability (§4/§5)
│   ├── closed_loop.py, pathfinding_bridge.py, attic.py           runtime retrieval (§6)
│   ├── percolation_hard.py, reinforcement_loop.py                capacity threshold (§7)
│   ├── gssm_potentiation.py, operator_readout.py                 threshold in the readout (§8)
│   ├── pos_run.py, pos_index.py, pos_analyze.py, verify_pos.py   plasticity on surprise (§9)
│   ├── holo_stream_recall.py, holo_gap_knee.py, holo_mag_read.py, phi_drift_probe.py   the knee arc (§10)
│   ├── pos_sleep*.py, pos_dream.py                               sleep + dreams (§11)
│   ├── holo_index_hybrid.py, holo_reminded.py, holo_heldout_keys.py   two-system closed loop (§12)
│   ├── pos_family_transfer.py                                    family transfer (§13)
│   ├── hot_swap_growth.py, state_weight_swap.py, beacon_swap.py  deployment primitives (§14)
│   ├── portable_organism.py, pos_shared_index.py, replay_law_run.py   portability + replication (§15)
│   ├── pixel_body_run.py, vizdoom_run.py, stranger_verify_run.py embodiment + review (§16)
│   ├── chimera.py                                                the composed organism (§17)
│   ├── knowledge_file_run.py, keyed_file_run.py, surprise_filter_run.py, filter_file_run.py   the knowledge file (§18)
│   ├── aged_brain_run.py, age_ladder_run.py                      the age axis (§19)
│   └── score_predictions_v2.py                                   the auto-falsifier
├── vendor/fabel/            the .causal deterministic knowledge engine (the index)
├── analysis/                theory, pre-registered predictions, research logs
├── results/                 measured JSON + logs — the evidence
└── plots/                   figures
```

`reference/` = architecture · `src/` = experiments · `vendor/fabel/` = knowledge index
· `analysis/` = theory · `results/` = measured JSON · `plots/` = figures.

---

## Status

Two weeks of continuous disclosure, published at the moment of discovery. The GSSM core (C1–C5) already does it:
the whole linear-SSM family collapses to one affine operator at machine precision (~1e-15), the
constant-gate restriction *is* the geometric Toeplitz kernel to 3.55e-15 at d=512, the parallel
scan is gradient-identical to the loop in fp64 and 4–7× faster on MPS, a key-conditioned
holographic write gives a bounded scalar state content-addressable recall at 5.7× its floor, and
— the structural headline — the position-free variant holds **flat perplexity across 524,288×
length extrapolation** (train T=32, eval to a single 16.7-million-token sequence at constant
2.5 GB, perplexity *improving* the whole way), **streams a billion eval tokens at a flat
4.36 GB — and has LIVED past 7.4 billion training tokens in one continuous process at
0.69–0.83 GB**.
Out of the box, with no years-long tuning, the operator already **matches** years-tuned SOTA
perplexity at the WikiText-2 data ceiling (135 PPL) — and on its own axes it does not compete, it
stands alone: flat perplexity to 524,288× length and a billion-token stream at constant memory,
which no attention model can do at any tuning budget.

On top of that, O1 adds the living-stream layer: **constant-memory training** (truncated-BPTT
exact to gradient cosine 1.0000), a **state that survives a 256-token silence**, a **runtime
`.causal` index** the stream consults without a gradient, and a **measured capacity threshold**
that is a sharp cliff in the gated readout (slope 1.32 vs 0.57 for a linear read).

And the o1-state layer makes it an organism: **surprise-gated plasticity** that matches
full-gradient learning at 25% of the gradient tokens on a live 40-hour stream, **keyed
holographic recall carried through silence** with a knee moved 32→512 in one theory-led day
and a machine-precision φ-invariance law behind it, **dosed sleep consolidation** with a
measured life curve, **index-reminded reads at ~1.0** above the state's capacity cliff, and
the whole operating mode **transferred to the Mamba/S6 configuration at 0.98×** (with
GSSM-Selective ahead 0.156 nats head-to-head at parameter parity) — seven
foundations, formalized in [FOUNDATIONS.md](FOUNDATIONS.md), every step pre-registered in
[analysis/PREDICTIONS.md](analysis/PREDICTIONS.md) before its data existed. And the organism
is *social* and *portable*: collective memory across individuals is measured (P31), and a
living run migrates across CPU architectures mid-stream with behavior identical to six
decimals (P38a) — the complete state is one ~53 MB artifact. Staged across two live machines
on a real network, at the one stream position where both organisms stood the heldout delta is
**0.0, exact**.

The latest wave sharpened the laws and opened two axes. The gate rate is a **dial, not an
emergent**: rate ≈ 1−q, invariant across an 8× width range, across seeds, and across a
149× difference in lived experience — a config-default flip that had masqueraded as a
width effect was caught by the register's own provenance trail and corrected through every
document. At fixed dose the gated arm **crosses full-gradient training at d=1024** (ratio
1.0092 on one fifth of the gradient tokens), echoing the token-axis crossing at 909.7M
tokens. The frozen knowledge file is **tappable** (keyed recall +0.264, specificity
+0.218), the gate is a measured **novelty filter** (1.51× first-ever content, 7× on new
types), and the two file roles split cleanly by harvest policy — the filter buys the
frontier: its file's transfer value is novelty-graded and crosses ahead on the newest
content, while random harvest wins only on the familiar. The **age axis** is open: experience is immunity (32% less forgetting,
recovery below pre-shock), at a small measured plasticity price (0.744×) and zero rate
shift. And the file format's review mechanism works across instruction sets: 19/20
sampled entries bit-exact between ARM and x86, the single divergence pinned to a one-ULP
float boundary and answered with an integer-exact quantization rule.

Every number here is reproducible from the scripts in `src/`. The kernel reductions are exact
identities; the recall result is 5-seed with the attention validity gate at 0.994; the threshold
controls (random/shuffle) are null across 3 seeds.

---

## License

Apache License 2.0. See `LICENSE`.

## Citation

```bibtex
@misc{foss2026o1state,
  author = {Foss, David Tom},
  title  = {{o1-state: The O(1)-State Organism --- Constant-Memory Streaming,
            Surprise-Gated Plasticity, Holographic Carry, and a Runtime
            Knowledge Index}},
  year   = {2026},
  note   = {Public research disclosure (prior art), github.com/DT-Foss/o1-state.
            Builds on GSSM (github.com/DT-Foss/gssm). A bounded
            reproducing-kernel SSM that consumes an unbounded stream at constant
            memory, gates its own plasticity on self-measured surprise, carries
            keyed holographic memory through silence, consolidates in dosed
            sleep, and consults an external .causal index at runtime — and
            migrates live across CPU architectures with identical behavior.
            Seven underlying primitives formalized in FOUNDATIONS.md.}
}
```
