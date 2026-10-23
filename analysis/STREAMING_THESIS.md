# Constant-memory consumption of an unbounded stream

**Timestamped public disclosure (prior art).** Author: David Tom Foss · 2026 · PolyForm Noncommercial 1.0.0.

This note states a thesis and the mechanism that makes it true. The mechanism is proven and
reproducible (script + numbers below). The thesis is the consequence that mechanism forces.

---

## The thesis

**Turn-based AI is episodic because its memory is tied to context length. A bounded-state
model breaks that tie — and makes continuous, non-episodic processing of an unbounded stream
architecturally possible.**

Every attention model has `O(length)` memory. So "the context" is necessarily a finite object
with edges: you load it, the model answers, the state is discarded, the next turn starts from
zero. The model does not live between turns. It is handed still frames and responds to each one
in isolation.

A bounded `O(1)` state does not extend the context — it removes the context's *objecthood*.
Constant memory under an unbounded stream means there is no edge, no "context full," no reset.
The state is not a container you fill; it is a flow that never stops. The difference is a photo
versus an eye. The forgetting gate `γ_t` is controlled forgetting — the exact primitive a
system needs to integrate a stream without drowning in it.

What this enables — a model that processes the world as a continuous stream of impressions at
constant memory, rather than as a sequence of discrete prompts — is the implication. We are
building the evidence for it one flag at a time. The billion-token run is flag one.

---

## The mechanism (proven)

Two things are decoupled at once — **doubly `O(1)`**:

1. **The corpus is never materialized.** It is streamed lazily (HuggingFace `streaming=True`):
   documents arrive one at a time and are tokenized into a rolling buffer. The corpus is just an
   *iterator*. C4 is one source — swap it for any token stream (a web scraper, a live feed, the
   whole internet) and nothing about the memory profile changes.

2. **The activations are never materialized for the full sequence.** Because the NoPE-Selective
   receptive field is ~5–8 tokens (it is a contraction; see
   [LENGTH_INVARIANCE_THEORY.md](LENGTH_INVARIANCE_THEORY.md)), an arbitrarily long sequence is
   evaluated by a sliding window of `chunk` tokens with a left-context overlap ≫ the receptive
   field, scoring only the new region. Memory is `O(chunk)`, not `O(T)`.

So neither the corpus nor the activations are ever held in full. **Effective sequence length is
limited only by wall-clock time — never by RAM.** The state's memory footprint is the same at
token 1 and at token 1,000,000,000.

The chunked / batched eval is exact: batching `b` windows through one forward pass gives
**identical** perplexity to scoring them one at a time — verified `ppl_batched / ppl_single =
1.00000`, same scored tokens. Batching changes throughput, nothing else.

---

## The evidence so far

Same NoPE-Selective model, trained at **T=32**, run at constant memory:

| effective length | extrap. | source | peak RSS | PPL behaviour |
|---|---|---|---|---|
| 16,777,216 | 524,288× | WT-103, chunked | 2.5 GB | flat / improving (×0.80) |
| **1,000,013,824** | **31,250,432×** | **C4 streamed**, chunked + batched | **4.36 GB (constant)** | **flat (×1.00)** |

![No length wall: flat PPL to 1 billion tokens at constant memory](../plots/scale_to_a_billion.png)

The billion-token row is the C4 streaming run: corpus streamed lazily, eval chunked and batched,
checkpoints every 50M tokens, **20 checkpoints across the full billion**. Over the entire span the
**running PPL moved 1.3 points** (247.08 → 248.38, a 0.52% band) and the **peak RSS moved 0.08 GB**
(4.28 → 4.36 GB). The memory line is flat from the first checkpoint to the last — this is the point
of the run, more than any single perplexity value. (PPL here is on unique C4 text, so its absolute
value differs from the WT-2/WT-103 numbers; the claim is the **flatness and the constant memory
across a billion tokens**, which is what an `O(1)`-state model uniquely gives.) The whole run took
153 minutes on a single Apple Mac mini; the machine never approached its 16 GB.

Anyone can confirm this against the committed result without re-running the 1B stream:
`python src/verify_billion.py` reads the JSON and checks every claim (≥1B tokens, flat memory, flat
PPL, huge sample) — exit 0 only if all pass. The raw run log (training curve + every checkpoint) is
committed verbatim as `results/scale_to_a_billion.run.log`.

→ `src/scale_to_a_billion.py` · `results/scale_to_a_billion.json` ·
`results/scale_to_a_billion.run.log` · `src/verify_billion.py` · `src/plot_billion.py`

---

## Why this is the headline, not a footnote

Every length number in this repository — 256×, 4096×, 16.7M, 1B — is **evidence**. The thesis is
the abstract claim that dominates all of them: *constant-memory consumption of an unbounded
stream.* Every other long-context system ties memory to sequence length somewhere (attention's
KV-cache is the obvious case). Here both axes are severed: the corpus never materializes, the
activations never materialize. A bounded-state model can, in principle, consume an infinite live
data stream at constant RAM.

That is a different category of system. Not "a model with a big context window" — **a model that
consumes a stream.** This note is the timestamped record of that claim and the code that makes it
real.
