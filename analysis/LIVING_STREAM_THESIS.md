# Living-Stream: constant-memory training + a state that lives through silence

**Disclosed:** 2026-06-26 · PolyForm Noncommercial 1.0.0. The training-side dual of the O(1) eval flag, plus a measured
idle-persistence mechanism. Three pillars, each with its decisive control, on the *same* NoPE
GSSM that streamed a billion tokens.

---

## The headline

**A GSSM-Selective (NoPE) trains on an unbounded token stream at flat memory — held-out loss
falls while RSS stays constant — and the same persistent per-layer state carries a planted bit
through an input gap, held by a dedicated channel the model grows at γ≈1: a learned bit-vault
that opens its input gate to write and shuts it to hold.**

This is what a turn-based, KV-cache model structurally cannot do: keep learning, and keep
*remembering*, across an unbounded stream at constant memory — including across pauses in the
input.

---

## Pillar A — constant-memory streaming TRAINING (the dual of the eval flag)

Train from scratch on streamed C4, carrying the per-layer state `Z` across chunks and cutting the
graph with `.detach()` (truncated BPTT). Held-out loss (WT-2 val, never in the stream) falls
**8.685 → 5.22** over 3M streamed tokens at **flat RSS ~0.8 GB**.

Two controls make it mean the mechanic:
- **Grad-exactness:** truncated-BPTT-with-carried-state vs full-window BPTT gives **cosine 1.0000**
  (with overlap warmup). The ~5-8-token receptive field means the truncation throws away *no*
  gradient — constant-memory training is not an approximation here, it is exact.
- **No-detach control:** keep the state attached and the autograd graph grows every step — RSS
  climbs **0.77 → 1.81 GB** over 56k tokens (unbounded). The `.detach()` carry is *exactly* what
  makes training O(1); remove it and there is no constant memory.

→ `src/streaming_train.py --train` · `results/streaming_train.json`

## Pillar D — idle-persistence: the state lives through the silence

A 1-bit, 2-way beacon task: `[beacon β_k][G filler tokens, no beacon][probe ?]` → recover `k` at
the probe. Trained with a gap curriculum (start G=4, grow to 128 — the ignition lesson: a large
gap from scratch never fires). Then test recall vs gap length, with the **decisive null**:

| gap G | carried state | state zeroed at gap | no-beacon control |
|---|---|---|---|
| 8 | **1.000** | 0.50 | 0.50 |
| 64 | **1.000** | 0.46 | 0.46 |
| 256 | **1.000** | 0.50 | 0.50 |

**The bit survives a 256-token input gap perfectly. Zero the carried state at the gap start and
recall collapses to chance.** So the answer rode the persistent state across the silence — not
local or post-gap context. (1 bit is deliberately inside the bounded-scalar regime; multi-key
recall is capped ~13% and is not claimed here.)

→ `src/streaming_train.py --idle` · `results/idle_persistence.json`

## Pillar E — the mechanism: a learned bit-vault (γ≈1, input-gate shut)

How does the bit survive when the head-mean γ is only ~0.6 (τ≈3)? The mean collapsed the channel
axis. Find the *carrier* channel — the one whose state correlates with the label — and read its
own gates:

- **Carrier = Layer 1, head 2, channel 15, corr −1.00** with the bit (perfectly identified).
- **γ_carrier = 0.9999 (τ≈1000)** — this single channel is a near-lossless integrator. The
  head-mean hid it among 15 short-memory channels.
- **α_carrier in the gap = 0.005, at the beacon = 0.52** — the input gate is **shut** during the
  filler tokens (nothing leaks in) and **open** at the beacon (the bit is written). `a_rms` in the
  gap = 0.000.
- **Class-separation margin holds 96.7%** across the 256-token gap (6.07 → 5.87). No functional
  decay.
- Layer 0's carrier has corr only +0.14, γ≈0.6 — a *local* layer. The model learned a **division
  of labour**: Layer 0 processes locally, Layer 1 holds a dedicated long-memory register.

So it is *both* of the candidate mechanisms at once: γ≈1 (never forgets) and input gated out in
the gap (frozen, not decaying), opening to write at the beacon. A bit-vault the model grew on its
own. Scope: tested on ignorable fillers; the next adversarial probe is beacon-like fillers.

→ `src/streaming_train.py --carrier` · `results/carrier_probe.json`

---

## Method note: how the carrier was found

The first γ-spectrum measurement (head-mean) suggested only short-memory heads (τ≈2) and *seemed*
to contradict the perfect 256-token recall. That was a measurement artifact: averaging over the
channel axis hid the one γ≈1 carrier. Naming the contradiction and building the per-channel carrier
probe turned a confusing negative into the real, stronger finding — the model builds a dedicated
long-memory channel. The mean was wrong; the mechanism is clean.

## Scope and next attacks

- **Is:** constant-memory *training* on a stream (exact, not approximate); a persistent state that
  carries a bit through an input gap with a decisive zeroing control; a mechanistic account (a γ≈1
  input-gated carrier channel).
- **Isn't (yet):** multi-bit / multi-key memory through gaps (bounded-scalar regime); robustness to
  adversarial non-ignorable fillers; a full continual-learning study across many source switches.
  These are the named next attacks, not hidden gaps.
