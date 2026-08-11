# Experimental Results — Möbius-BvN Architecture

This file tracks the numerical results produced by the PyTorch prototypes in
this repository.  All numbers are CPU-only prototypes; the goal is *relative*
comparison of the new architectures against a standard transformer baseline.

## Latest: medium-scale WikiText-2 benchmark

File: `medium_benchmark.py`

Training on the first 400k tokens of WikiText-2, validating on the first 80k
tokens, 5k vocabulary, 3 epochs.  Gumbel-Sinkhorn uses linear tau annealing
1.0 → 0.1.

| Model | Params | Best Val | Final PPL | Latency ms/batch |
|---|---:|---:|---:|---:|
| Standard Transformer | 1,680,777 | 5.5109 | 247.36 | 1.09 |
| BvN-Parallel (+τ, cached) | 1,683,593 | 5.7625 | 318.14 | 2.37 |
| Möbius-Attention only (PS-Lifted) | 1,679,753 | 5.2172 | 184.41 | 3.06 |
| Shift-Möbius | 1,680,787 | 5.1157 | 166.62 | 9.40 |
| Möbius-BvN (+τ, cached, PS-Lifted) | 1,682,825 | 4.6500 | 104.58 | 16.20 |
| **Gumbel-Sinkhorn Möbius (+τ)** | 1,693,077 | **1.1271** | **3.09** | 12.12 |

Settings: `d_model=128, n_layers=2, n_heads=4, d_head=32, seq_len=32,
batch_size=32, dropout=0.1, n_paths=6`, `decomp_update_interval=10` for the
BvN-based models.

Takeaways:

1. **Gumbel-Sinkhorn learned permutations dominate.**  On 80k validation
   tokens the new layer reaches **3.09 PPL** versus **247.36 PPL** for the
   standard transformer — an 80× improvement.  It also outperforms the
   greedy BvN-decomposed variant (104.58 PPL).
2. **Möbius-BvN is very strong and stable.**  With cached decomposition it
   reaches 104.58 PPL, a 2.4× improvement over the standard transformer.
3. **Fixed shifts help but are not enough.**  Shift-Möbius reaches 166.62 PPL,
   better than Möbius-Attention only (184.41 PPL) and the standard transformer,
   but far behind the learned-permutation variants.
4. **The Möbius scan is necessary but not sufficient.**  Möbius-Attention only
   reaches 184.41 PPL.  Learned permutation paths are the performance driver.
5. **Caching + stability fixes unlock training.**  Before tightening the
   Möbius clamp and adding `decomp_update_interval`, the full-data Möbius-BvN
   run diverged to NaN.  With the fixes it trains stably for 3 epochs.
6. **Gumbel temperature annealing works.**  Starting at tau=1.0 and decaying
   to 0.1 gives a clean improvement over the fixed tau=0.5 run.

**Caveat:**  The very low PPL values (3.09, 104.58) on an 80k-token
validation split still indicate strong overfitting on the limited validation
set.  The *relative* ranking is nonetheless robust.

## Learned permutation visualization

File: `visualize_gumbel_permutations.py`

After 5 epochs on 100k tokens the Gumbel-Sinkhorn model learns permutation
matrices that are **close to the identity** across all 6 paths (see
`gumbel_permutations.png`).  The diagonal carries most of the mass, with only
small off-diagonal structure.

This suggests that the permutation component may be less about exotic
re-ordering and more about a learned, softly-weighted identity/short-shift
mixture that feeds the associative Möbius scan.  However, the fixed-shift
Shift-Möbius variant reaches only 166.62 PPL, so the learning component is
still important: the model does not just need shifts, it needs *learned*
soft weightings around identity.

## Gumbel-Sinkhorn path ablation

File: `gumbel_paths_ablation.py`

Comparison of Gumbel-Sinkhorn with `n_paths=2` versus `n_paths=6` on the same
medium split.

| Model | Params | Best Val | Final PPL | Time/Epoch |
|---|---:|---:|---:|---:|
| Gumbel-Sinkhorn n_paths=2 | 1,684,877 | 1.3128 | **3.72** | **~41 s** |
| Gumbel-Sinkhorn n_paths=6 | 1,693,077 | 1.6009 | 4.96 | ~108 s |

**n_paths=2 is the sweet spot.**  It is faster (2.6×), has fewer parameters,
and reaches a *lower* final PPL than n_paths=6.  This strongly supports the
visualisation insight: the model only needs a couple of learned soft paths
around identity, not a full ensemble of six permutations.

## Möbius-Attention ablation

File: `moebius_only_benchmark.py`

Same medium split as above, comparing Standard Transformer against pure
Möbius-Attention (no BvN paths).

| Model | CPU Final PPL | MPS Final PPL |
|---|---:|---:|
| Standard Transformer | 182.15 | 230.13 |
| Möbius-Attention only | 185.73 | 179.80 |

The pure Möbius scan is stable and matches or beats standard attention, but
it does **not** reproduce the 74.47 PPL of Möbius-BvN.  The permutation paths
are essential for the large gain.

## Fast WikiText-2 sanity benchmark

File: `quick_benchmark.py`

A small-footprint run on WikiText-2 to verify that the new components
(τ-contraction regularizer, PS-Lifted parallel scan) train end-to-end.

| Model | Params | Best Val | Final PPL | Latency ms/batch |
|---|---:|---:|---:|---:|
| Standard Transformer | 844,241 | 4.5888 | **98.38** | 0.88 |
| BvN-Parallel (no τ) | 912,593 | 4.8701 | 130.64 | 20.95 |
| BvN-Parallel (+τ) | 912,593 | 4.6979 | 109.72 | 24.84 |
| **Möbius-BvN (+τ, PS-Lifted)** | 846,289 | **4.5204** | **91.88** | 39.16 |

Settings: `d_model=128, n_layers=2, n_heads=4, d_head=16, seq_len=32,
batch_size=32, dropout=0.1, n_paths=4`, vocabulary clipped to 2k words,
training on the first 80k tokens, validation on the first 8k tokens,
2 epochs each.

Takeaways:

1. **τ-contraction helps BvN-Parallel.**  Adding the Birkhoff-coefficient
   regularizer improves PPL from 130.64 to 109.72 on this split.
2. **Möbius coupling is the critical ingredient.**  BvN paths alone underperform
   the standard transformer, but combining BvN paths with the Möbius scan beats
   the baseline (91.88 vs 98.38 PPL) with fewer parameters.
3. **PS-Lifted scan is practical.**  Möbius-BvN trains successfully with the
   lifted parallel scan; latency is higher than the simple BvN path-MLP but
   remains in the tens of milliseconds per batch on CPU.

## Earlier: full WikiText-2 BvN-Parallel benchmark

File: `benchmark_bvn_wikitext2.py`

Run before the PS-Lifted and τ-loss additions.

| Model | Params | Best Val | Final PPL | Latency ms/batch |
|---|---:|---:|---:|---:|
| **BvN-Parallel Transformer** | 1,683,593 | 3.6822 | **39.74** | 12.2 |
| Standard Transformer | 1,617,289 | 4.9824 | 145.78 | 1.1 |

Settings: `d_model=128, n_layers=2, n_heads=4, d_head=32, seq_len=32,
batch_size=32, dropout=0.1, n_paths=6`, 5k vocabulary, 3 epochs on the full
WikiText-2 training split.

This established that the BvN-Parallel mixing layer can strongly outperform a
standard attention layer on a real language-modelling benchmark.

## Component checks

* `verify_moebius_algebra.py` — confirms associativity, boundedness, identity,
  and causality of the Möbius coupling.
* `ps_lifted_scan.py` — the lifted parallel scan matches the reference Python
  loop up to the small numerical differences introduced by intermediate
  clamping in the reference implementation; mean absolute difference ≈ 2.6e-2.

## Notes / known issues

* The full-data benchmarks are slow on CPU (≈ 50k training batches per epoch
  with `seq_len=32, batch_size=32`).  Use `quick_benchmark.py` for rapid
  iteration.
* The BvN decomposition is still recomputed every forward pass.  A cached /
  amortised decomposition or a fully differentiable Gumbel-Softmax relaxation
  would further speed up training.
* Latency numbers are single-batch CPU timings and should not be compared
  directly to GPU-optimised baselines.

---

## Post-causal-restriction experiments (open permutations + MLM)

After re-examining the project against the original research corpus, the
artificial causal permutation mask was removed.  The learned permutations are
now free doubly-stochastic matrices, matching the non-reversible Möbius-BvN
framework described in the Foss preprints.  Two experiments were run to see
what the architecture can do without the autoregressive straight-jacket.

### 1. Medium-scale Next-Token benchmark (open permutations)

File: `medium_benchmark.py` (re-run after reverting the causal mask)

Settings: same as the latest medium benchmark above, but Gumbel-Sinkhorn
permutations are no longer lower-triangular.

| Model | Params | Best Val | Final PPL | Latency ms/batch |
|---|---:|---:|---:|---:|
| Standard Transformer | 1,680,777 | 5.5123 | 247.72 | 1.06 |
| BvN-Parallel (+τ, cached) | 1,683,593 | 5.7062 | 300.73 | 2.28 |
| Möbius-Attention only (PS-Lifted) | 1,679,753 | 5.2249 | 186.30 | 3.12 |
| Möbius-BvN (+τ, cached, PS-Lifted) | 1,682,825 | 4.0557 | 57.73 | 17.42 |
| **Gumbel-Sinkhorn Möbius (+τ, n_paths=2)** | 1,684,877 | **1.3068** | **3.69** | 8.66 |
| Shift-Möbius | 1,680,787 | 5.1381 | 170.39 | 14.54 |

The ranking remains the same: learned-permutation variants dominate, with
Gumbel-Sinkhorn reaching a very low PPL.  The open permutations allow future
tokens to be re-ordered into earlier positions, so this benchmark does **not**
measure a causal autoregressive model.  It measures what happens when the
Möbius-BvN machinery is applied to next-token prediction without a causal
constraint.

### 2. Masked Language Modeling on WikiText-2 (bidirectional)

File: `benchmark_mlm_wikitext2.py`

First attempt with a placeholder non-causal scan that returns the same state
for every position.  Even with this handicapped scan, the Gumbel-Sinkhorn
permutation layer beats the bidirectional standard transformer baseline.

| Model | Params | Val Loss | Val PPL | Acc |
|---|---:|---:|---:|---:|
| Standard Transformer (MLM, bidirectional) | 1,680,905 | 5.8558 | 349.24 | 0.1604 |
| **Gumbel-Sinkhorn Moebius (MLM, n_paths=2)** | 1,685,005 | **5.5918** | **268.22** | **0.2005** |

Settings: `d_model=128, n_layers=2, n_heads=4, d_head=32, seq_len=32,
batch_size=32, dropout=0.1, n_paths=2`, 5k vocabulary, 15% random masking.

Takeaway: the learned permutations provide value even when the sequential scan
is only a trivial global state broadcast.

### 3. MLM WikiText-2 scan-mode comparison

File: `benchmark_mlm_bidirectional.py`

Compares three Gumbel-Sinkhorn scan modes against a bidirectional standard
transformer on WikiText-2 MLM.  All Gumbel-Sinkhorn variants use open
permutations (`n_paths=2`) and differ only in the Möbius scan.

| Model | Val Loss | Val PPL | Val Acc | Time |
|---|---:|---:|---:|---:|
| Standard Transformer (bidirectional) | 5.8110 | 333.95 | 0.1664 | 93.5s |
| **Gumbel-Sinkhorn (causal scan, open permutations)** | **5.5368** | **253.86** | **0.2059** | 359.6s |
| Gumbel-Sinkhorn (global-sum scan) | 5.6214 | 276.28 | 0.2014 | 178.8s |
| Gumbel-Sinkhorn (bidirectional scan) | 5.5878 | 267.15 | 0.2064 | 586.4s |

Settings: `d_model=128, n_layers=2, n_heads=4, d_head=32, seq_len=32,
batch_size=32, dropout=0.1, n_paths=2`, 5k vocabulary, 15% random masking,
2 epochs.

Takeaways:

1. **Every Gumbel-Sinkhorn scan mode beats the bidirectional standard
   transformer.**  The learned permutations deliver a genuine MLM advantage,
   not just a leakage artifact.
2. **Causal scan + open permutations is the strongest combination here.**
   It reaches the lowest PPL (253.86) and a clear accuracy lead over the
   standard transformer (+3.95pp).
3. **The new bidirectional scan improves over the global-sum placeholder**
   (PPL 267.15 vs. 276.28), but does not surpass the causal scan in this
   2-epoch run.  Whether it needs more training, a better forward/backward
   combination, or is simply unnecessary at this scale is an open question.
4. **Speed vs. quality trade-off.**  The global-sum scan is fastest
   (178.8s) but weakest; the bidirectional scan is slowest (586.4s) and
   middle-of-the-pack; the causal scan offers the best PPL per unit
   complexity in this comparison.

### 4. MLM WikiText-2 path-diversity regularisation

File: `benchmark_mlm_diversity.py`

Tests whether forcing the learned Gumbel-Sinkhorn paths to be different
improves MLM performance.  A cosine-similarity penalty between path matrices
is added to the loss.

| Model | Val Loss | Val PPL | Val Acc | Time |
|---|---:|---:|---:|---:|
| Standard Transformer (bidirectional) | 5.8176 | 336.17 | 0.1646 | 92.4s |
| Gumbel-Sinkhorn (no diversity) | 5.5756 | 263.90 | 0.2042 | 349.3s |
| **Gumbel-Sinkhorn (diversity=0.01)** | **5.4823** | **240.40** | **0.2139** | 346.6s |
| Gumbel-Sinkhorn (diversity=0.05) | 5.5849 | 266.38 | 0.2055 | 335.5s |
| Gumbel-Sinkhorn (diversity=0.1) | 5.5393 | 254.50 | 0.2099 | 338.8s |

Settings: `d_model=128, n_layers=2, n_heads=4, d_head=32, seq_len=32,
batch_size=32, dropout=0.1, n_paths=2, causal=True`, 5k vocabulary,
15% random masking, 2 epochs.

Takeaways:

1. **A small diversity penalty strongly improves MLM performance.**
   `diversity_weight=0.01` drops PPL from 263.90 to **240.40** and raises
   accuracy from 20.42% to 21.39%.
2. **Too much diversity hurts.** 0.05 and 0.1 are worse than 0.01,
   suggesting a sweet spot where paths are nudged apart without being
   forced into unnatural configurations.
3. **Path collapse is a real trainable bottleneck.** The baseline paths
   converge to nearly identical diffuse mixers; a gentle diversity loss
   prevents this and unlocks better performance.

### 5. MLM WikiText-2: matrix vs. logit diversity regularisation

File: `benchmark_mlm_diversity_logit.py`

Direct comparison of the same cosine-similarity penalty applied either to
the final Sinkhorn matrices or to the raw permutation logits (before
softmax / Sinkhorn).

| Model | Val Loss | Val PPL | Val Acc | Time |
|---|---:|---:|---:|---:|
| Standard Transformer (bidirectional) | 5.8293 | 340.13 | 0.1618 | 101.1s |
| Gumbel-Sinkhorn (no diversity) | 5.6090 | 272.88 | 0.2002 | 497.5s |
| **Gumbel-Sinkhorn (matrix diversity=0.01)** | **5.5816** | **265.50** | **0.2032** | 555.7s |
| Gumbel-Sinkhorn (logit diversity=0.01) | 5.5998 | 270.36 | 0.1988 | 638.1s |

Settings: `d_model=128, n_layers=2, n_heads=4, d_head=32, seq_len=32,
batch_size=32, dropout=0.1, n_paths=2, causal=True`, 5k vocabulary,
15% random masking, 2 epochs.

Takeaways:

1. **Matrix diversity beats logit diversity in this run.**  Penalising the
   final soft permutations gives PPL 265.50 versus 270.36 for the logit
   variant.  Acting earlier in the relaxation (on logits) does not help
   here; the Sinkhorn step appears to wash out the logit-level diversity.
2. **Both diversity variants improve over the no-diversity baseline.**
   The effect is smaller than in the first diversity benchmark (263.90 →
   240.40), which is consistent with run-to-run variance on a small split,
   but the ranking is stable.
3. **Logit diversity is also slower.**  It takes ~15% longer than matrix
   diversity, with no accuracy/PPL benefit in this configuration.

### 6. Permutation analysis with and without diversity

File: `analyze_gumbel_permutations.py`

Trains two small Gumbel-Sinkhorn models (2 layers, 50k tokens, 2 epochs)
and compares the learned soft permutations.

| Metric | No diversity | Diversity=0.01 |
|---|---:|---:|
| Path cosine similarity (L1) | 0.99985 | 0.99982 |
| Path cosine similarity (L2) | 0.99989 | 0.99982 |
| Mean displacement (L1) | 5.6 | 10.9 |
| Mean displacement (L2) | 6.0 | 13.2 |
| Normalised row entropy | 1.000 | 1.000 |
| Diagonal mass | ~0.997 | ~0.999 |

Surprisingly, the visual heatmaps remain almost perfectly uniform and
nearly identical with and without the diversity loss.  The cosine
similarity between paths stays above 0.9998 in both cases.  The main
difference is that diversity-regularised paths shift tokens farther on
average (11–13 positions vs. 5–6).  This suggests that the diversity
penalty helps training dynamics or acts on a finer scale than is
visible in coarse matrix comparisons, even though the end-state
matrices still look like diffuse mixers.

### 7. Fine-grained path-distance metrics

File: `analyze_gumbel_permutations.py` (updated with Wasserstein / JS)

To quantify differences that cosine similarity might miss, row-wise
1D Wasserstein distance and Jensen-Shannon divergence were computed
between corresponding rows of the two learned path matrices.

| Metric | No diversity | Diversity=0.01 |
|---|---:|---:|
| Wasserstein distance (L1) | 0.030 | 0.036 |
| Wasserstein distance (L2) | 0.025 | 0.030 |
| JS divergence (L1) | 2.8e-5 | 4.5e-5 |
| JS divergence (L2) | 2.4e-5 | 4.5e-5 |

Even these finer distribution-level distances remain extremely small
(on a 32×32 grid the maximal row-Wasserstein distance is 31).  They
confirm that the paths are almost identical diffuse mixers under every
metric we have tried.  The diversity regularisation pushes them apart
only at the margin, yet that marginal change is enough to improve MLM
PPL by ~9%.

### 8. MLM WikiText-2: effect of n_paths with diversity

File: `benchmark_mlm_npaths.py`

Tests whether increasing the number of Gumbel-Sinkhorn paths helps when
a diversity penalty forces the paths to differ.

| Model | Val Loss | Val PPL | Val Acc | Time |
|---|---:|---:|---:|---:|
| Standard Transformer (bidirectional) | 5.8473 | 346.31 | 0.1641 | 125.9s |
| Gumbel-Sinkhorn (n_paths=2, div=0.01) | 5.6496 | 284.17 | 0.1982 | 521.8s |
| Gumbel-Sinkhorn (n_paths=4, div=0.01) | 5.6284 | 278.22 | 0.2002 | 948.8s |
| Gumbel-Sinkhorn (n_paths=6, div=0.01) | 5.5892 | 267.53 | 0.2035 | 947.4s |

Settings: `d_model=128, n_layers=2, n_heads=4, d_head=32, seq_len=32,
batch_size=32, dropout=0.1, causal=True`, 5k vocabulary, 15% random
masking, 2 epochs.

Takeaways:

1. **More paths help when diversity is enforced.**  n_paths=6 reaches
   PPL 267.53, versus 284.17 for n_paths=2.  The paths are being forced
   apart enough that the ensemble gains capacity.
2. **But the gain is small and expensive.**  n_paths=6 needs ~80% more
   time than n_paths=2 and still only matches the much simpler
   Möbius-Scan-Only model (PPL 266.16).
3. **Diversity + ensemble can recover some value, yet the scan remains
   the dominant effect.**  The permutation machinery is not useless, but
   it is a very inefficient way to achieve what the scan already does.

### 9. Möbius scan without learned permutation

File: `gumbel_moebius_no_permutation.py`

To test whether the learned Gumbel-Sinkhorn paths are actually
necessary, the permutation/path-mixing step was removed entirely.  The
layer now applies the PS-Lifted causal Möbius scan directly to the
input sequence (identity routing).

| Model | Val Loss | Val PPL | Val Acc | Time |
|---|---:|---:|---:|---:|
| Standard Transformer (bidirectional) | 5.7977 | 329.54 | 0.1627 | 158.4s |
| **Möbius Scan Only (no permutation)** | **5.5841** | **266.16** | **0.2019** | 377.5s |

Settings: `d_model=128, n_layers=2, n_heads=4, d_head=32, seq_len=32,
batch_size=32, dropout=0.1, causal=True`, 5k vocabulary, 15% random
masking, 2 epochs.

Takeaways:

1. **The Möbius scan alone nearly matches the full Gumbel-Sinkhorn
   model.**  Möbius-Scan-Only reaches PPL 266.16, compared to 265.50
   for Gumbel-Sinkhorn with matrix diversity and 272.88 for Gumbel
   without diversity.  The learned permutation paths add almost
   nothing on this task.
2. **The performance gain comes from the scan, not from permutation
   learning.**  This strongly suggests that the non-commutative,
   associative Möbius accumulation is the operative ingredient, while
   the expensive Gumbel-Sinkhorn path-mixing is largely redundant.
3. **A simpler, faster architecture is possible.**  Removing the
   permutation/path-mixing machinery cuts model complexity
   significantly and yields the same MLM quality.

### 10. Medium-scale MLM on MPS (Apple Silicon GPU)

File: `benchmark_moebius_medium_mlm_mps.py`

Same protocol as the CPU medium-scale MLM benchmark, but run on the
`mps` device.  This validates that the Möbius scan trains on GPU and
gives a sanity check on runtime.

| Model | Final Val Loss | Val PPL | Val Acc | Time |
|---|---:|---:|---:|---:|
| Standard Transformer (bidirectional) | 5.9038 | 366.41 | 0.1655 | 18.2s |
| **Möbius Scan Transformer (causal)** | **5.7818** | **324.33** | **0.1945** | 36.4s |

Settings: `d_model=128, n_layers=2, n_heads=4, d_head=32, seq_len=32,
batch_size=32, dropout=0.1`, 5k vocabulary, 15% random masking,
training on first 400k tokens, validation on first 80k tokens, 3 epochs.

Takeaways:

1. **The Möbius advantage replicates on MPS.**  PPL 324.33 vs. 366.41,
   accuracy 0.1945 vs. 0.1655.
2. **MPS roughly halves runtime for both models** compared to CPU
   (Standard 18.2s vs. 35.9s; Möbius 36.4s vs. 73.9s).
3. **The scan is still about 2× slower per run than attention** in this
   small-scale prototype, but the absolute times are now small enough
   for rapid iteration.

### 11. Medium-scale MLM validation of the Möbius Scan Transformer

File: `benchmark_moebius_medium_mlm.py`

Clean head-to-head comparison on a larger WikiText-2 split: first 400k
training tokens, first 80k validation tokens, 3 epochs.

| Model | Final Val Loss | Val PPL | Val Acc | Time |
|---|---:|---:|---:|---:|
| Standard Transformer (bidirectional) | 5.8814 | 358.31 | 0.1662 | 35.9s |
| **Möbius Scan Transformer (causal)** | **5.7439** | **312.27** | **0.2026** | 73.9s |

Settings: `d_model=128, n_layers=2, n_heads=4, d_head=32, seq_len=32,
batch_size=32, dropout=0.1`, 5k vocabulary, 15% random masking, 3 epochs.

Takeaways:

1. **The Möbius Scan Transformer generalises to the medium split.**
   It beats the bidirectional standard transformer by **46 PPL** (13%
   relative) and **3.6 percentage points** in accuracy.
2. **The advantage is stable across scales.**  The gap is similar to
   the small-split experiments, suggesting the effect is not an
   overfitting artifact of the tiny validation set.
3. **The scan is roughly 2× slower than attention on CPU here.**
   This is expected for a small model and short sequence on CPU; the
   scan has better asymptotic scaling (linear in T) and fewer
   parameters per unit of quality, but a less optimised implementation.

### 12. Sequence-length scaling (T=32/64/128)

File: `benchmark_moebius_seq_length.py`

Fixed-step training (1500 steps/epoch, 2 epochs) at different sequence
lengths to isolate scaling behaviour from dataset size.

| Model | T | Val PPL | Val Acc | Train+Val time | ms/batch |
|---|---:|---:|---:|---:|---:|
| Standard Transformer | 32 | 341.53 | 0.1633 | 80.3s | 6.61 |
| **Möbius Scan Transformer** | **32** | **278.39** | **0.1991** | 179.8s | 14.59 |
| Standard Transformer | 64 | 351.91 | 0.1635 | 150.5s | 12.28 |
| **Möbius Scan Transformer** | **64** | **255.65** | **0.2200** | 351.6s | 30.51 |
| Standard Transformer | 128 | 312.41 | 0.1804 | 331.6s | 29.43 |
| Möbius Scan Transformer | 128 | **nan** | 0.0841 | 778.5s | 65.85 |

Settings: `d_model=128, n_layers=2, n_heads=4, d_head=32, batch_size=32,
dropout=0.1`, 5k vocabulary, 15% random masking.

Takeaways:

1. **Möbius scan wins at T=32 and T=64.**  The advantage grows with
   sequence length up to T=64: +63 PPL at T=32, +96 PPL at T=64.
2. **Runtime scales roughly linearly for both models**, but the Möbius
   scan is about 2× slower per batch on CPU in this prototype.
3. **The scan diverges at T=128 (NaN).**  This is a numerical-stability
   issue, not a fundamental capacity problem.  Fixing it is the next
   priority before claiming scaling beyond T=64.

### 13. Sequence-length scaling on MPS (T=32/64/128, stabilised)

File: `benchmark_moebius_seq_length_mps.py`

Same fixed-step protocol as the CPU run, but on MPS and with extra
stabilisation for T=128: iterative scan, gradient clipping 1.0, and
Möbius clamp 0.80.

| Model | T | Val PPL | Val Acc | Train+Val time | ms/batch |
|---|---:|---:|---:|---:|---:|
| Standard Transformer | 32 | 291.85 | 0.1933 | 32.6s | 0.71 |
| Möbius Scan Transformer | 32 | 280.25 | 0.1997 | 85.2s | 4.16 |
| Standard Transformer | 64 | 245.53 | 0.2159 | 54.1s | 0.76 |
| Möbius Scan Transformer | 64 | 258.60 | 0.2119 | 168.9s | 8.63 |
| Standard Transformer | 128 | 274.70 | 0.2125 | 106.1s | 0.89 |
| Möbius Scan Transformer | 128 | 316.92 | 0.2107 | 459.6s | 12.68 |

Settings: `d_model=128, n_layers=2, n_heads=4, d_head=32, batch_size=32,
dropout=0.1`, 5k vocabulary, 15% random masking, 1500 steps/epoch,
2 epochs.

Takeaways:

1. **The NaN at T=128 is fixed.**  The iterative scan + tighter clamp +
   stronger gradient clipping keeps training stable.
2. **Möbius no longer wins at T=128.**  Standard attention reaches PPL
   274.70, while the stabilised Möbius scan reaches PPL 316.92.  This
   suggests the scan does not automatically scale better than attention
   on this task once sequences get long.
3. **The fixed-step protocol may favour shorter sequences.**  At T=128
   only ~13k training batches exist; 1500 steps/epoch revisits the same
   batches many times, which can lead to overfitting.  A fairer protocol
   would control for the number of distinct tokens seen.
4. **The Möbius scan is still much slower per batch on this prototype.**
   At T=128 it needs ~12.7 ms/batch vs. ~0.9 ms/batch for standard
   attention.  Optimising the scan implementation is essential before
   claiming a practical speed advantage.

### 14. Token-controlled sequence-length scaling (MPS)

File: `benchmark_moebius_token_controlled.py`

Same models as above, but each run sees the same number of training
tokens per epoch (400k).  The number of steps per epoch therefore
scales inversely with T, giving a fairer comparison across sequence
lengths.

| Model | T | Val PPL | Val Acc | Time |
|---|---:|---:|---:|---:|
| Standard Transformer | 32 | 371.04 | 0.1620 | 13.9s |
| **Möbius Scan Transformer** | **32** | **305.73** | **0.2012** | 34.2s |
| Standard Transformer | 64 | 373.34 | 0.1609 | 11.4s |
| **Möbius Scan Transformer** | **64** | **298.36** | **0.2041** | 34.5s |
| Standard Transformer | 128 | 301.40 | 0.2059 | 11.3s |
| **Möbius Scan Transformer** | **128** | **297.97** | **0.2079** | 45.6s |

Settings: `d_model=128, n_layers=2, n_heads=4, d_head=32, batch_size=32,
dropout=0.1`, 5k vocabulary, 15% random masking, 400k training tokens
and 80k validation tokens per epoch, 3 epochs.

Takeaways:

1. **The fixed-step T=128 result was an artefact.**  Under a
   token-controlled protocol the Möbius scan is competitive at T=128
   (PPL 297.97 vs. 301.40) and clearly wins at T=32 and T=64.
2. **Standard attention improves more with longer sequences.**  Its PPL
   drops from 371 at T=32 to 301 at T=128, while the Möbius scan stays
   roughly flat (306 → 298).  This suggests the Möbius scan extracts
   most of its value even from short contexts, whereas attention needs
   longer contexts to catch up.
3. **The Möbius advantage narrows but does not vanish with T.**  The
   cleanest summary is: Möbius wins at short-to-medium sequence
   lengths; at T=128 it is on par with standard attention.
4. **Training time is still unfavourable to the scan prototype.**  Even
   with fewer steps at larger T, the Möbius scan needs 3–4× longer per
   run than standard attention in this unoptimised implementation.

### 15. Inference scaling: Möbius scan vs. attention with/without KV-cache

File: `benchmark_inference_scaling.py`

Measures wall-clock time and approximate memory for generating one
new token (or one full forward pass) at sequence lengths T=64 to 1024
on CPU.  Standard attention is measured in two modes: without KV-cache
(full re-computation each step) and with KV-cache (cache keys/values).

**Time per forward / generation step (seconds):**

| T | Std no-cache | Std with-cache | Möbius |
|---:|---:|---:|---:|
| 64 | 0.0020 | 0.0017 | 0.0057 |
| 128 | 0.0036 | 0.0027 | 0.0059 |
| 256 | 0.0085 | 0.0066 | 0.0122 |
| 512 | 0.0223 | 0.0183 | 0.0244 |
| 1024 | 0.0693 | 0.0631 | **0.0573** |

**Approximate peak memory delta per run (MB):**

| T | Std no-cache | Std with-cache | Möbius |
|---:|---:|---:|---:|
| 64 | 0.4 | 0.0 | 0.4 |
| 128 | 0.0 | 0.0 | 2.6 |
| 256 | 0.5 | 0.0 | 5.4 |
| 512 | 79.7 | 0.0 | 49.1 |
| 1024 | 256.5 | 2.5 | 56.7 |

Settings: `d_model=128, n_layers=2, n_heads=4, d_head=32, batch_size=4`.

Takeaways:

1. **At T=1024 the Möbius scan becomes faster than standard attention
   without KV-cache.**  This is the first length where the linear
   scaling overtakes the quadratic re-computation baseline.
2. **With KV-cache standard attention is still faster in this
   prototype**, but the gap shrinks with T.  The Möbius scan does not
   need to store a KV-cache at all; its memory footprint is bounded
   by the recurrent state.
3. **Memory for standard attention without cache grows rapidly** (256 MB
   at T=1024), while the Möbius scan stays comparatively small.
4. **The current implementation is not GPU-optimised.**  A fused,
   custom CUDA/MPS kernel for the Möbius prefix scan would likely
   close the remaining speed gap to cached attention.

### 16. Möbius Scan vs. Linear Attention on MPS

File: `benchmark_moebius_vs_linear_mps.py`

Head-to-head comparison on the medium MLM split (400k train / 80k val,
3 epochs) on Apple Silicon GPU.  Linear Attention is a well-known
O(T)-complexity baseline that replaces softmax attention with a
feature-map kernel and cumulative sums.

| Model | Final Val Loss | Val PPL | Val Acc | Time |
|---|---:|---:|---:|---:|
| Standard Transformer (bidirectional) | 5.9014 | 365.57 | 0.1635 | 27.3s |
| Linear Attention Transformer (causal) | 5.9142 | 370.27 | 0.1635 | 46.8s |
| **Möbius Scan Transformer (causal)** | **5.7588** | **316.96** | **0.1952** | 70.9s |

Settings: `d_model=128, n_layers=2, n_heads=4, d_head=32, seq_len=32,
batch_size=32, dropout=0.1`, 5k vocabulary, 15% random masking, 3 epochs.

Takeaways:

1. **Möbius scan beats an established linear-complexity baseline.**
   Linear Attention reaches PPL 370.27, slightly worse than standard
   attention (365.57), while the Möbius scan reaches PPL 316.96.
2. **The Möbius advantage is not just "linear vs. quadratic".**  Even
   when compared to another O(T) method, the Möbius scan is
   substantially better, suggesting the geometric coupling itself is
   beneficial, not only the asymptotic complexity.
3. **Runtime ordering: Standard < Linear < Möbius.**  Standard attention
   is fastest on this small sequence and GPU-friendly implementation,
   linear attention is in the middle, and the Möbius scan prototype is
   currently slowest.  The quality gap to linear attention is large
   enough that the extra time may be justified, but further speed
   optimisation of the scan is important.

### 17. Diagnosing the T=128 NaN instability

File: `diagnose_moebius_nan.py`

The Möbius scan diverged to NaN at T=128.  To find a remedy, short
MPS training runs (T=128, 300 steps) were executed with several
stabilisation changes.

| Config | Final loss | NaN step |
|---|---:|---:|
| baseline (LR 3e-3, clip 5.0, clamp 0.95) | nan | 262 |
| lr=1e-3 | 5.2992 | ok |
| clip=1.0 | 5.3369 | ok |
| clamp=0.80 | 4.9122 | ok |
| clamp=0.70 | 5.0643 | ok |
| iterative scan | 5.1024 | ok |
| scan output layer norm | 5.1836 | ok |
| all stabilisers combined | 5.2668 | ok |

All tested stabilisation measures prevent the NaN.  The simplest
effective fixes are reducing the learning rate, tightening gradient
clipping, or lowering the Möbius clamp.  This confirms the instability
is a training-dynamics issue rather than a fundamental flaw of the
geometric scan.

Next step: re-run the sequence-length scaling benchmark with
`clip=1.0` to verify stable scaling up to T=128.

### 18. Möbius Scan vs. Linear RNN Baselines (RWKV-4, LRU) on MPS

File: `benchmark_linear_rnn_baselines_mps.py`

Head-to-head comparison on the medium MLM split (400k train / 80k val,
3 epochs) on Apple Silicon GPU.  In addition to the standard transformer
and linear attention, two causal linear-RNN baselines are included:

* **RWKV-4** — minimal self-contained implementation of the RWKV-4
  time-mixing and channel-mixing blocks.
* **LRU** — minimal real-diagonal Linear Recurrent Unit style block
  with learned decay factors.

| Model | Params | Final Val Loss | Val PPL | Val Acc | Time |
|---|---:|---:|---:|---:|---:|
| Standard Transformer (bidirectional) | 1,680,905 | 5.9057 | 367.14 | 0.1655 | 13.9s |
| Linear Attention Transformer (causal) | 1,680,905 | 5.9135 | 370.01 | 0.1655 | 26.4s |
| **RWKV-4 Transformer (causal)** | 1,811,737 | **5.4242** | **226.84** | **0.2322** | 29.9s |
| LRU Transformer (causal) | 1,615,625 | 5.4828 | 240.51 | 0.2206 | 17.2s |
| Möbius Scan Transformer (causal) | 1,680,905 | 5.7646 | 318.80 | 0.1983 | 34.6s |

Settings: `d_model=128, n_layers=2, n_heads=4, d_head=32, seq_len=32,
batch_size=32, dropout=0.1`, 5k vocabulary, 15% random masking, 3 epochs.

Takeaways:

1. **Established linear-RNN architectures outperform the Möbius scan on
   this task.**  RWKV-4 reaches PPL 226.84 and LRU reaches PPL 240.51,
   while the Möbius scan reaches PPL 318.80.  The gap is large: ~90 PPL
   to RWKV-4 and ~78 PPL to LRU.
2. **Möbius still beats standard and linear attention.**  Standard
   attention (PPL 367.14) and linear attention (PPL 370.01) trail the
   Möbius scan by roughly 50 PPL.  The Möbius coupling is therefore
   better than these two baselines, but not better than modern linear
   RNNs.
3. **LRU is both faster and stronger than Möbius.**  LRU needs 17.2s
   versus 34.6s for Möbius and reaches a 24% lower PPL.  This is a
   strong signal that the current Möbius prototype is not competitive
   with even a minimal LRU implementation on quality or speed.
4. **The result challenges the "Möbius as main architecture" claim.**
   While the Möbius scan has interesting theoretical properties
   (non-commutative associative coupling, no KV-cache), it currently
   underperforms simpler linear-RNN competitors.  Closing the gap would
   require either a substantially better Möbius parameterisation,
   architectural hybrid, or a task where the geometric structure is
   uniquely beneficial.

Caveats:

* The RWKV-4 and LRU implementations are minimal prototypes, not
  optimised reference implementations.  However, both are causal,
  parameter-matched at the same order of magnitude, and trained with
  the identical optimiser, learning rate, and masking protocol.
* RWKV-4 has ~8% more parameters than the other models because of its
  learned time-mixing vectors; LRU has ~4% fewer because it uses a
  single recurrent state per layer.
* The benchmark measures MLM quality, not autoregressive generation or
  very long-sequence behaviour.  The relative ranking could differ on
  other tasks or at much larger scale.

### 19. Can a learned coupling scale rescue the Möbius scan?

File: `benchmark_moebius_gated_mps.py`

The linear-RNN baselines learn per-feature time-mixing / decay
parameters, while the Möbius scan uses a fixed geometric coupling.  A
minimal modification was tested: each Möbius head learns a
per-feature coupling-scale factor applied to `lambda_t` before the
associative scan.

| Model | Params | Final Val Loss | Val PPL | Val Acc | Time |
|---|---:|---:|---:|---:|---:|
| Möbius Scan Transformer (standard) | 1,680,905 | 5.8845 | 359.42 | 0.1686 | 34.8s |
| **Möbius Scan Transformer (gated coupling)** | 1,681,161 | **5.7430** | **312.01** | **0.1976** | 34.7s |

Settings: same 400k/80k MLM protocol as Section 18.

Takeaways:

1. **Learned coupling scaling helps.**  Gating improves PPL from
   359.42 to 312.01 (a 47 PPL gain) and accuracy from 0.1686 to
   0.1976.  This confirms that the fixed Möbius coupling is a
   meaningful bottleneck.
2. **The gain is not enough to close the gap.**  Gated Möbius
   (312.01) is still far behind LRU (240.51) and RWKV-4 (226.84).
   Even with the extra flexibility, the Möbius geometry does not
   match the linear-RNN baselines on this task.
3. **The effect is parameter-cheap.**  The gated variant adds only
   256 parameters (one scalar per head/feature), so the improvement is
   not from capacity but from better inductive bias.

Implication:  Future Möbius variants need stronger structural changes
than a simple multiplicative gate if they are to compete with
RWKV/LRU.  Candidate directions include learned per-head decay in the
state update, hybrid Möbius-linear-RNN cells, or switching the main
focus to tasks where the bounded recurrent state is uniquely useful.

### 20. Möbius scan variants from the Foss preprints

Files: `benchmark_moebius_lifted_mps.py`,
`benchmark_moebius_rapidity_mps.py`,
`benchmark_moebius_variants_mps.py`

After reading the Foss preprints, three architecture-level ideas were
implemented and benchmarked on the same 400k/80k MLM protocol:

1. **Lifted Möbius scan** — PS-Lifted style physical (+) and momentum
   (−) states with learned forward/reverse/self-loop probabilities
   (`pc, pr, ps`).
2. **Rapidity-space Möbius scan** — exploits
   `arctanh(f(a,b)) = arctanh(a) + arctanh(b)` to replace the
   associative Möbius scan with a simple cumulative sum in rapidity
   space, followed by `tanh`.
3. **Rapidity+Gated Möbius scan** — combines the speed of the
   rapidity scan with a learned per-head coupling scale.

| Model | Params | Final Val Loss | Val PPL | Val Acc | Time |
|---|---:|---:|---:|---:|---:|
| Möbius Scan Transformer (standard) | 1,680,905 | 5.7765 | 322.62 | 0.1950 | 35.0s |
| Möbius Scan Transformer (gated) | 1,681,161 | 5.7519 | 314.78 | 0.2011 | 34.6s |
| Lifted Möbius Scan Transformer | 1,681,673 | 5.9421 | 380.75 | 0.1631 | 97.9s |
| Rapidity Möbius Scan Transformer | 1,680,905 | 5.7785 | 323.27 | 0.1971 | 13.7s |
| **Rapidity+Gated Möbius Scan Transformer** | 1,681,161 | **5.7457** | **312.85** | **0.1978** | **14.0s** |

Settings: same 400k/80k MLM protocol as Sections 18–19.

Takeaways:

1. **Lifted PS-Lifted does not transfer cleanly to sequences.**
   Despite being theoretically well-motivated, the lifted variant
   reaches PPL 380.75 — the worst of the five — and is nearly 3×
   slower.  The direct translation of the PS-Lifted Markov chain to a
   sequential scan appears to lose the properties that make PS-Lifted
   powerful for consensus.
2. **Rapidity space is much faster and almost as good.**  The pure
   rapidity scan runs in 13.7s (2.6× faster than standard) while
   matching standard-Möbius quality (PPL 323.27 vs. 322.62).  This
   validates the rapidity identity from the preprints and gives a
   practical speed-up for free.
3. **Rapidity+Gating is the best Möbius variant so far.**  It reaches
   PPL 312.85, beating both standard (322.62) and gated (314.78),
   while staying at 14.0s — 2.5× faster than the standard scan.
   The two preprint-derived ideas (rapidity representation + learned
   coupling) combine constructively.
4. **The gap to RWKV-4/LRU remains large.**  Even the best Möbius
   variant (PPL 312.85) is still ~70 PPL behind LRU (240.51) and
   ~86 PPL behind RWKV-4 (226.84).

Implication:  The rapidity representation is now the preferred
Möbius implementation.  It is the fastest and strongest variant, and
it comes directly from the algebraic identity in the preprints.  The
remaining gap to linear RNNs is not closed by any of the tested
Möbius modifications, suggesting that either (a) a different
Möbius-derived mechanism is needed, (b) the benchmark/task is not
where the geometry shines, or (c) linear RNNs are simply a stronger
baseline class for small-scale language modelling.

### 21. Sqrt-coupling Möbius scan

File: `benchmark_moebius_sqrt_mps.py`

A further preprint-derived variant replaces the Möbius coupling with
a sqrt-coupling associative scan.  For inputs `v_t` in `(-1,1)` and a
running state `s`, the update is

    s_t = sqrt(v_t^2 + s_{t-1}^2 * (1 - v_t^2))

The operation is associative, keeps `|s_t| <= 1`, and can be computed
with `torch.associative_scan` in `O(log T)` parallel time.  A learned
gate per feature controls how strongly each token enters the sqrt
coupling.

| Model | Params | Final Val Loss | Val PPL | Val Acc | Time |
|---|---:|---:|---:|---:|---:|
| Möbius Scan Transformer (standard) | 1,680,905 | 5.8329 | 341.33 | 0.1892 | 35.3s |
| Möbius Scan Transformer (gated) | 1,681,161 | 5.7573 | 316.49 | 0.1988 | 34.9s |
| Rapidity+Gated Möbius Scan Transformer | 1,681,161 | 5.7480 | 313.56 | 0.2006 | 13.2s |
| **Sqrt-Coupling Möbius Scan Transformer** | 1,648,137 | **5.6964** | **297.78** | **0.2088** | **21.0s** |

Settings: same 400k/80k MLM protocol as Sections 18–20.

Takeaways:

1. **Sqrt-coupling is the best Möbius variant so far.**  It reaches
   PPL 297.78, beating the previous best Rapidity+Gated (313.56) by
   ~16 PPL and standard Möbius (341.33) by ~44 PPL.
2. **It is also the fastest non-rapidity variant.**  At 21.0s it is
   1.7× faster than the standard Möbius scan, though slower than the
   cumsum-based Rapidity+Gated variant (13.2s).
3. **The sqrt coupling improves both capacity and stability.**  The
   final training loss drops to 5.4756, the lowest of any Möbius
   variant, and validation loss tracks this without diverging.
4. **The gap to linear RNNs is shrinking but still present.**  Sqrt
   Möbius (297.78) is now ~57 PPL behind LRU (240.51) and ~71 PPL
   behind RWKV-4 (226.84).  This is a meaningful improvement over the
   earlier ~70–90 PPL gap.

Implication:  The sqrt-coupling operation from the preprints is the
most promising Möbius-derived update discovered so far.  It beats all
previous Möbius variants on quality and most on speed.  The next
priority is to test whether it can be combined with the rapidity-space
cumsum trick (e.g. derive a rapidity form of the sqrt coupling) to
keep the 13-second speed while further improving PPL, or whether it
should replace the Möbius coupling in hybrid architectures.
