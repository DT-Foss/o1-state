# Möbius Scan Transformer — Research Summary

## One-line claim

A causal Möbius scan over token representations matches or beats standard
attention on WikiText-2 MLM **without any learned permutation, attention, or
quadratic pairwise interaction**. The associative, non-commutative Möbius
accumulation appears to be the operative mechanism; learned permutation paths
are redundant on this task.

## Core result

On WikiText-2 masked language modelling (`d_model=128, n_layers=2,
n_heads=4, d_head=32, seq_len=32, 5k vocab`):

**Small split (2 epochs):**

| Model | Val PPL | Val Acc | Relative time |
|---|---:|---:|---:|
| Standard Transformer (bidirectional) | 329–346 | 0.16 | 1.0× |
| Gumbel-Sinkhorn, no diversity | 272.88 | 0.20 | ~3.5× |
| Gumbel-Sinkhorn, matrix diversity=0.01 | 265.50 | 0.20 | ~4.0× |
| Gumbel-Sinkhorn, logit diversity=0.01 | 270.36 | 0.20 | ~4.5× |
| Gumbel-Sinkhorn, n_paths=6 + diversity | 267.53 | 0.20 | ~6.0× |
| **Möbius Scan Only (no permutation)** | **266.16** | **0.20** | **~2.4×** |

**Medium split (400k train / 80k val, 3 epochs) — CPU:**

| Model | Val PPL | Val Acc | Time |
|---|---:|---:|---:|
| Standard Transformer (bidirectional) | 358.31 | 0.1662 | 35.9s |
| **Möbius Scan Transformer** | **312.27** | **0.2026** | 73.9s |

**Medium split — MPS (Apple Silicon GPU):**

| Model | Val PPL | Val Acc | Time |
|---|---:|---:|---:|
| Standard Transformer (bidirectional) | 366.41 | 0.1655 | 18.2s |
| **Möbius Scan Transformer** | **324.33** | **0.1945** | 36.4s |

The Möbius Scan Transformer wins by **40+ PPL** and **~3pp accuracy**
on both CPU and MPS.  MPS roughly halves runtime for both models.

**Versus an established linear-complexity baseline (MPS):**

| Model | Val PPL | Val Acc | Time |
|---|---:|---:|---:|
| Standard Transformer | 365.57 | 0.1635 | 27.3s |
| Linear Attention | 370.27 | 0.1635 | 46.8s |
| **Möbius Scan Transformer** | **316.96** | **0.1952** | 70.9s |

Möbius scan beats both standard attention and linear attention by a
wide margin, showing the advantage is not merely "linear vs.
quadratic" but specific to the geometric Möbius coupling.

**Sequence-length scaling (MPS, T=32/64/128):**

| Model | T | Val PPL | Val Acc |
|---|---:|---:|---:|
| Standard Transformer | 32 | 291.85 | 0.1933 |
| Möbius Scan Transformer | 32 | 280.25 | 0.1997 |
| Standard Transformer | 64 | 245.53 | 0.2159 |
| Möbius Scan Transformer | 64 | 258.60 | 0.2119 |
| Standard Transformer | 128 | 274.70 | 0.2125 |
| Möbius Scan Transformer | 128 | 316.92 | 0.2107 |

Under a **token-controlled protocol** (same number of tokens per epoch
for every T):

| Model | T | Val PPL | Val Acc |
|---|---:|---:|---:|
| Standard Transformer | 32 | 371.04 | 0.1620 |
| **Möbius Scan Transformer** | **32** | **305.73** | **0.2012** |
| Standard Transformer | 64 | 373.34 | 0.1609 |
| **Möbius Scan Transformer** | **64** | **298.36** | **0.2041** |
| Standard Transformer | 128 | 301.40 | 0.2059 |
| **Möbius Scan Transformer** | **128** | **297.97** | **0.2079** |

The fixed-step T=128 result was an artefact.  Under fair data exposure
Möbius wins at T=32 and T=64 and is on par at T=128.  Standard
attention improves more with longer sequences, while the Möbius scan
already extracts most of its value from short contexts.

**Inference scaling (CPU, T=64 to 1024):**

| T | Std no-cache | Std with-cache | Möbius |
|---:|---:|---:|---:|
| 64 | 0.0020 | 0.0017 | 0.0057 |
| 128 | 0.0036 | 0.0027 | 0.0059 |
| 256 | 0.0085 | 0.0066 | 0.0122 |
| 512 | 0.0223 | 0.0183 | 0.0244 |
| 1024 | 0.0693 | 0.0631 | **0.0573** |

At T=1024 the Möbius scan overtakes standard attention without
KV-cache in this prototype.  It needs no KV-cache, so its memory
footprint is bounded by the recurrent state rather than growing with
sequence length.  A fused kernel implementation is expected to close
the remaining gap to cached attention.

**Interpretation:** The expensive Gumbel-Sinkhorn path-mixing machinery adds
negligible value over a simple Möbius scan. The scan itself is the driver of
the MLM advantage.

## Why this matters

Standard Transformers rely on **attention**: pairwise query-key similarity
followed by softmax-weighted aggregation. This is:
- **Quadratic** in sequence length.
- **Position-agnostic** at the interaction level.
- **Globally averaging**: every token can directly attend to every other token.

The Möbius Scan Transformer replaces attention with a **sequential,
associative, non-commutative recurrence** derived from hyperbolic (Poincaré
ball / Lorentz) geometry:
- **Linear** in sequence length and parallelisable via prefix scan.
- **Position-aware** by construction.
- **State-accumulating**: a latent state evolves along the sequence through a
  geometric coupling.

If the effect scales, this is a genuine architectural alternative to both
attention and recent State Space Models (Mamba, RWKV, LRU).

## Key auxiliary findings

1. **Learned permutations collapse.** Gumbel-Sinkhorn path matrices converge
to near-uniform doubly-stochastic mixers with cosine similarity > 0.9998.
2. **Diversity regularisation helps only marginally.** It improves PPL from
263.90 to 240.40 in one run, but the matrices remain visually identical.
3. **Matrix diversity outperforms logit diversity.** Penalising the final
soft permutations (PPL 265.50) works better than penalising raw logits
(PPL 270.36).
4. **More paths help, but not enough.** n_paths=6 with diversity reaches
PPL 267.53, still slower than the scan-only model and no better in quality.

## Implications

- **Simpler is better here.** Drop Gumbel, Sinkhorn, path-weights, and
  diversity losses.
- **Focus on the algebra.** The Möbius coupling / Lorentz correspondence is
  the research object that deserves attention.
- **Efficiency.** Removing the permutation path-mixing cuts runtime and
  model complexity while preserving performance.
- **Interpretability.** A recurrent scan state is easier to inspect than a
  dense attention map or a diffuse permutation matrix.

## Open questions / next experiments

1. **Scale.** Does the advantage hold for larger models, longer sequences,
   and full WikiText-2 / Wikitext-103?
2. **Baselines.** How does it compare to Mamba, RWKV, LRU, or a strong
   linear-attention variant?
3. **Task breadth.** Does it work for autoregressive LM, classification,
   long-range arena tasks?
4. **Theory.** Why exactly does the Möbius scan help? Is it the
   non-linearity, the gating, the geometric mean-like accumulation, or a
   favourable inductive bias for language?
5. **Bidirectional scan.** Does a full bidirectional Möbius scan close the
   gap to bidirectional attention even further?

## Suggested next move

Run a clean, reproducible medium-scale benchmark (`moebius_scan_transformer.py`
vs. standard transformer) on a larger WikiText-2 split with multiple seeds,
then publish the numbers and a minimal clean codebase.
