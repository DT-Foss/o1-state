# FOSS-KI v3.0 Research Findings
## 6 Opus Agents, 7 Verzeichnisse, 52 Findings — 2026-03-17

---

## Architektur-Einsichten

### 1. Capacity Problem Is Already Solved (Quantum Agent)
- Reservoir = Feature Extractor (Holevo: 12 Bit/State max), NICHT Speicher
- Nach 5 Mixing-Steps: <1 Bit effektiv (T13: capacity decays as 4.3×exp(-0.5t))
- KB (137K Facts, Dict O(1)) = primärer Speicher — korrekt
- Hopfield Bank (2000 Patterns, 512d) = Assoziativ-Cache — korrekt
- Reservoir Memory Capacity: MC ≈ 0.258 × dim ≈ 1057 Tokens, Cliff bei ~160

### 2. Consensus auf Strings statt Vektoren = Depth-Bottleneck (Brain Agent)
- Transformer-Tiefe = kontinuierliche Vektor-Integration über Layers
- FOSS-KI macht diskrete KB-Lookups → Consensus auf Strings
- Fix: Vector-Space Consensus = Gossip-Runden auf 2048d Vektoren
- Jede Gossip-Runde ≈ ein Transformer-Layer Information-Integration

### 3. CASI Gate ist KAPUTT (CASI Agent)
- `casi_gate.py` importiert Crypto-CASI (byte-level: bit_correlation, xor_distribution, parity_chain)
- Braucht 1D-CASI (7 statistische Strategien: runs, DW, crossing_radius, adj_corr, sign_change, radial_trend, quartile_shift)
- Crypto-CASI auf quantisierte Float-Similarities = misst NICHTS
- Fix: 80 LOC pure NumPy, AUC=0.988 für Halluzinations-Detektion (CF248)

---

## Formeln

### F1: 1D-CASI (7 Strategien)
```
CASI(x) = Σsᵢ(x) / E[Σsᵢ(π(x))]
```
- Strategien: runs_test, durbin_watson, crossing_radius, adjacent_correlation, sign_change_rate, radial_trend, quartile_shift
- CASI=1.0 = structureless (exchangeable), CASI>1.5 = structure detected
- Source: plasma_cosmology/loss_function/task186_early_stopping.py lines 33-101

### F2: Intrinsic Dimension Scaling Law
```
ID_needed = 2 × log₂(K) + 1.2
```
- K = number of distinct answer classes, R²=0.919
- 2000 Hopfield patterns → ID_needed ≈ 23d (aktuell 512d = 22× over-provisioned)
- Source: gottformel/experiments/breadcrumb_4_id_complexity_law/run.py

### F3: GCV Ridge Alpha (Cross-Validation-Free)
```
α_opt = (D/N) × MSE_residual
```
- D = feature dimension (4096), N = training samples
- Marchenko-Pastur: α = gamma × σ²_noise
- Source: Formelbot audit, RMT theory

### F4: Consensus Convergence Formula
```
k = ceil(5 × log(1/ε) / spectral_gap)
```
- DS Markov chains mix in 2-5 steps (T25/T26: gap ~ n^{-5/12})
- PS-Lifted speedup 26-115× → k ≈ 3-10 iterations
- Aktuell: max_steps=5000 = massive overkill

### F5: Confidence = bits_wasted
```
bits_wasted = max(0, log₂(CASI(source_confidences)))
final_conf = consensus_conf × (1 - bits_wasted / max_bits)
```
- Source: plasma_cosmology/de_finetti/task204_sufficiency.py (CF207)

### F6: MERA Rank Decay
```
rank ~ 2n × exp(-0.31t)
```
- After 5 steps: rank ≈ 4096 × 0.21 = 860d effective
- Readout komprimierbar auf 860d (4.8× compression)
- Source: pralelluniversumquantencomputing/T12, T113

### F7: Sinkhorn Normalization (Doubly Stochastic)
```python
def sinkhorn(W, iters=5):
    M = abs(W)
    for _ in range(iters):
        M = M / (M.sum(dim=1, keepdim=True) + 1e-8)  # row
        M = M / (M.sum(dim=0, keepdim=True) + 1e-8)  # col
    return M
```
- 1 Iteration = 86% des Effekts (BC11)
- 130× besserer Gradient-Flow bei Depth 16
- 1.65× schnellere Kompression pro Layer
- Source: gottformel/experiments/breadcrumb_1_sinkhorn_attention/run.py

### F8: Optimal p_c = 0.65
- Bestätigt durch 3 unabhängige Kriterien:
  - Maximum Entropy Production (T111)
  - Fisher Information minimum sensitivity (T107)
  - Maximum Spectral Gap (T09/T46)
- Aktuell: consensus.py nutzt p_c=0.80 — FALSCH

### F9: Effective Rank (Shannon Entropy der Singulärwerte)
```python
def effective_rank(sigma):
    s = sigma / sigma.sum()
    entropy = -sum(s * log(s))
    return exp(entropy)
```
- Diagnostiziert echte Kapazität vs nominale

### F10: TwoNN Intrinsic Dimension Estimator
```python
def twonn_id(X, n_sample=500):
    dist = cdist(X, X); dist[diag] = inf
    r1, r2 = sort(dist)[:, 0], sort(dist)[:, 1]
    return 1.0 / mean(log(r2 / r1))
```
- Diagnostiziert Reservoir-Degeneration: ID<5=collapsed, ID>100=noise

### F11: KB Compression Lower Bound
- Shannon minimum: 137K × 17.1 Bit = 293KB
- Sparse triplet structure: (1317×279) mit 4855 nonzero → 11-15KB
- Aktuell: 2MB JSON = 100× overhead

### F12: Reservoir as Erasure Code
- Z₂ structure = [[2n,1,d]] self-correcting erasure code
- Toleriert 50-80% Node-Erasure, Recovery in 3 extra Rounds
- Aggressive Quantisierung (4-bit) ohne Accuracy-Loss möglich

---

## DOOM Findings (Creative Cross-Pollination)

### D1: Situation Memory — Anti-Halluzination + Generalization
- Source: doom/foss-v2/modules/situation_memory.py
- Modern Hopfield mit (query, reasoning_path, quality_score) Triplets
- Attractor distance = "hab ich noch nie gesehen" → "I don't know" statt Halluzination
- 150 LOC NumPy, circular buffer, EMA update

### D2: Cerebellum Reasoning Loop — Iterative Verfeinerung
- Source: doom/foss-v2/modules/cerebellum.py
- Predict → Compare (CASI) → Correct → Re-inject → Repeat (3-5×)
- Reservoir Echo State = verschiedene Trajektorien bei Re-Injection
- "Thinking" ohne Transformer-Layers

### D3: Experience Pool — Online Learning ohne Gradients
- Source: doom/foss-v2/modules/experience_pool.py
- Circular buffer (max 500), dedupliziert, älteste pruned
- Nach erfolgreicher Inferenz (CASI > threshold): (query_emb, chain, answer_emb, CASI) speichern
- Vor nächster Inferenz: top-3 relevante Experiences als Priming

### D4: Multi-Timescale Reservoir — Fast/Medium/Slow Bands
- Source: doom/foss-v2/ARCHITECTURE.md
- Fast (512 nodes, leak=0.5): Token-Features
- Medium (1024 nodes, leak=0.3): Satz-Semantik
- Slow (512 nodes, leak=0.1): Kontext-Akkumulation
- Z₂ zwischen Fast/Slow = Mismatch-Detektor

### D5: Online Ridge + Grokking
- Source: doom/foss-v2/ARCHITECTURE.md lines 596-618
```python
XtX *= 0.999; XtY *= 0.999  # exponential forgetting
XtX += outer(state, state); XtY += outer(state, target)
W_out = solve(XtX, XtY)
```
- Lernen während Inferenz, kein Backprop
- Sinkhorn auf Readout-Weights könnte Grokking triggern

### D6: Sleep Consolidation
- SVD-Truncation (NREM = Rauschen vergessen)
- Random Perturbation (REM = neue Assoziationen)
- Stability Check (Reawakening)
- Source: gottformel/experiments/H6_sleep_wake_cycles/synthetic_sleep.py

### D7: Wang-Landau Knowledge Explorer
- Source: gottformel/experiments/H11_dark_optimization/wang_landau.py
- Flat-Histogram Sampling: seltene Facts genauso oft wie häufige
- Für kreative/hypothetische Fragen: non-obvious Connections finden

---

## Brain Architecture Findings

### B1: MLP Stores Facts, Attention Routes (Tenney 2019)
- Layer 10: syntactic/relational (Delta=54)
- Layer 18: semantic/factual (Delta=242)
- extract_mlp_facts.py: nur 198 von 1300+ Subjects extrahiert

### B2: Predictive Coding = 100× Effizienz
- Brain filtert 95-99% des Inputs (nur Prediction Errors weiter)
- Transformer: 0% Filtering (alles bei O(n²) verarbeitet)
- Z₂ Novelty IST der Prediction-Error-Signal → formalisieren

### B3: Relational Binder fehlt
- Transformer: spezielle Attention Heads binden "France" + "capital"
- FOSS-KI: String-Matching → brüchig
- Fix: Bilinear Form `score(s,r,o) = s^T × W_r × o`, ein W_r pro Relationstyp

### B4: Sleep Pruning = Brain prunes 45% der Synapsen
- Hopfield Bank akkumuliert Patterns ohne Pruning
- Fix: Bottom 30-40% der Patterns mit niedrigem Recall entfernen

### B5: Overparameterization → Robustness
- Brain: 2763× overparameterized → 95% Performance bei 10% Damage
- FOSS-KI Reservoir: 4× overparameterized → zu niedrig
- Empfehlung: Reservoir auf 8192-16384 Nodes

---

## Quantum/Info Theory Findings

### Q1: KB Compression = 50-300KB statt 2MB
- Binary Triplet Format: (subject_id: 11 Bit, relation_id: 9 Bit, object_id: variable) = ~25-30 Bit/Fact
- 137K × 30 Bit = 513KB, mit Entropy Coding ~50KB

### Q2: Z₂ = KL Momentum (24-74% der Info in early Steps)
- state_pos - state_neg IST die Momentum-Komponente des KL
- Aktuell nur als Scalar genutzt (mean abs)
- Full KL Decomposition = kalibriertes Confidence-Signal

### Q3: MERA Readout Compression (4.8×)
- Nach 5 Mixing-Steps: effektiver Rank ≈ 860
- SVD-Truncation der Readout-Matrix von 4096 auf 860d

### Q4: p_c = 0.65 optimal (3× bestätigt)
- Entropy Production peak, Fisher Information, Spectral Gap — alle bei 0.65

---

## Formelbot Findings

### FB1: 1-Step Universality Theorem
- Eine einzelne Row-Column-Normalisierung → universelle spektrale Statistik
- NND input-unabhängig nach 1 Step (8 Verteilungen getestet)

### FB2: Softmax Attention Sink Compounds 1049% über 4 Layers
- Pro Layer: 7.4% Column-Sum-Deviation
- Über 4 Layers: 1049% → erster Token bekommt 7× mehr Gewicht
- Fix: 1 Sinkhorn-Iteration auf Attention Weights

### FB3: Mobius Attention (3 Parameter statt Q·K·V)
```
f(x) = (α×x + β) / (γ×x + 1)
```
- Source: formelbot-audit/engine/math_core.py

### FB4: DS-MCMC Sampler — 10× besser als Metropolis-Hastings
- TV=0.071 vs 0.800 auf multimodalen Verteilungen
- Für zukünftige Sampling-basierte Generation

---

## Implementation Priority

### Phase 1: Quick Wins (1-2 LOC each)
1. p_c = 0.65 in consensus.py
2. Sinkhorn auf Hopfield Attention (2 LOC)
3. Consensus max_steps → formula-based

### Phase 2: CASI Fix (80 LOC)
4. 1D-CASI mit 7 Strategien in casi_gate.py
5. CASI Confidence Penalty (bits_wasted)
6. CASI Consensus Convergence

### Phase 3: Hopfield Optimization
7. SVD Keys 512d → 32d
8. GCV Ridge Alpha
9. Expand MLP Facts (198 → 1300+)

### Phase 4: Iterative Reasoning (DOOM)
10. Situation Memory
11. Experience Pool
12. Cerebellum Loop (Predict → CASI → Correct)

### Phase 5: Deep Architecture
13. DS Layer Stack (Sinkhorn depth)
14. Multi-Timescale Reservoir
15. Vector-Space Consensus
