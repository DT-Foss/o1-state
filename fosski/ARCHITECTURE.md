# FOSS-KI Architecture — Complete System Design

**Version**: 1.0 | **Date**: 2026-03 | **Author**: David Foss | **Status**: Production Handoff

This document describes every component, data flow, and design decision in FOSS-KI. A new engineer should be able to understand the entire system without reading code.

## Executive Summary

FOSS-KI is a non-transformer AI engine that performs language understanding and generation via six coordinated layers:

1. **Reservoir** — Echo State Network (fixed-weight recurrent processor)
2. **Memory** — Hopfield patterns for associative retrieval
3. **Consensus** — Foss-accelerated gossip protocol for multi-component agreement
4. **Evolution** — Graph topology learning (not used at inference)
5. **Causal** — Pearl's do-calculus for interventional reasoning
6. **Spikes** — Z₂ parity signals for novelty detection

At inference, components run in parallel, consensus combines their answers, and a CASI gate monitors generation quality. Zero transformer layers execute at runtime.

### Key Facts

| Property | Value |
|----------|-------|
| Trainable components | Reservoir readout (ridge regression) only |
| Training samples | 16,748 (multi-hop, cross-fact, related-fact rollout) |
| Inference speed | ~200ms per query (CPU) |
| Memory footprint | ~2GB (embeddings + readout) |
| Non-transformer architecture | Yes (completely) |
| Anti-hallucination mechanism | Hopfield attractor distance |
| Main dependency | Qwen3-1.7B (for embeddings only, no inference-time forward pass) |

---

## System Overview

### 6-Layer Architecture

```
┌─────────────────────────────────────────────┐
│ Layer 0: Input Processing                   │
│ Tokenize → Qwen3 512d embeddings (SVD proj) │
└──────────────────┬──────────────────────────┘
                   │
┌──────────────────┴──────────────────────────┐
│ Layer 1: Reservoir (Echo State Network)      │
│ 2048 nodes, Foss Barbell topology, Z₂ copy  │
│ Fixed weights (only readout trained)        │
└──────────────────┬──────────────────────────┘
                   │
    ┌──────────────┼──────────────┐
    │              │              │
┌───┴────┐   ┌────┴────┐   ┌────┴────┐
│ Layer 2│   │ Layer 2  │   │ Layer 2 │
│Reservoir   Hopfield   KB  Causal
│Memory   Memory Store  Facts Reasoning
└───┬────┘   └────┬────┘   └────┬────┘
    │             │             │
    └─────────────┼─────────────┘
                  │
          ┌───────┴────────┐
          │                │
    ┌─────┴─────┐    ┌────┴────┐
    │ Layer 4:  │    │ Layer 5: │
    │Consensus  │    │Spikes    │
    │ (Gossip)  │    │(Z₂ parity│
    │ PS-Lifted │    │novelty)  │
    │Barbell    │    └────┬────┘
    └─────┬─────┘         │
          │               │
    ┌─────┴───────────────┘
    │
┌───┴────────────────────┐
│ CASI Gate              │
│ Stop if structure      │
│ collapses              │
└───┬────────────────────┘
    │
┌───┴────────────────┐
│ Response           │
│ Naturalized answer │
└────────────────────┘
```

---

## Component Deep Dive

### Layer 1: Reservoir (Echo State Network)

**File**: `/Users/bhkmie/Desktop/foss-ki/core/reservoir.py` (245 lines)

**What It Does**: Fixed-weight recurrent processor that accepts a sequence of embeddings and produces a compressed state representation. No training of the recurrent weights — only the readout layer is trained via ridge regression.

**Architecture**:
- **Nodes**: 2048 (even number for Z₂ parity structure)
- **Topology**: Foss Barbell — two 3-cliques with 1 bridge (embedded in 2048-node grid)
- **Input dimension**: 512 (Qwen3 SVD-projected embeddings)
- **State dimension**: 2048 (one Z₂ layer, but used as single state for efficiency)
- **Activation**: tanh (hyperbolic tangent)
- **Leak rate**: α = 0.3 (controls plasticity; higher = more responsive to new input)
- **Spectral radius scaling**: 0.95 (ensures Echo State Property — bounded activity)

**Z₂ Structure**:
```python
# Internal representation
state_pos: (1024,)  # forward-biased view
state_neg: (1024,)  # backward-biased view
# Parity signal (computed on demand)
parity = state_pos - state_neg  # detects agreement/disagreement
```

**Step Equation**:
```
state(t+1) = (1-α)·state(t) + α·tanh(T·state(t) + W_in·u(t))
where:
  T: transition matrix (scaled to |λ_max| = 0.95)
  W_in: input weights (random, fixed, drawn from N(0, 0.1²))
  u(t): input at time t (512d embedding)
  α: leak rate (0.3)
```

**Readout Training**:
- Ridge regression on states: `W_out = (S^T·S + λI)^{-1}·S^T·Y`
- Feature augmentation: `[state, state²]` (quadratic features for expressiveness)
- Training samples: 16,748 (from teacher-forced rollout)
- Ridge α = 1e-6 (minimal regularization)
- Output dimension: 512 (predict embeddings)

**Key Innovation**: No backpropagation through the recurrent weights. The Echo State Property (spectral radius < 1) guarantees stable dynamics without explicit training. This is Jaeger (2001) applied to Foss topology.

---

### Layer 2a: Memory — Hopfield Template Bank

**File**: `/Users/bhkmie/Desktop/foss-ki/core/hopfield_bank.py` (221 lines)

**What It Does**: Modern Hopfield Network (exponential capacity) storing KB facts + ConceptNet relations as learnable patterns. Retrieval via softmax attention (O(N) matmul, not iterative dynamics).

**Capacity**:
- Patterns: 1989 (200 KB + 1789 ConceptNet)
- Dimension: 512 (Qwen3 embeddings)
- Theoretical capacity: O(d^(d-1)) = O(2^4608) for d=512 (far beyond our use)
- Practical capacity tested: 1000+ patterns with >85% accuracy (Ramsauer et al. 2020)

**Retrieval** (softmax attention):
```python
retrieve(query_words, top_k=3, β=8.0):
    query = mean([emb_store.encode(w) for w in query_words])
    scores = β * (K^T @ query)           # (N,) inner products
    weights = softmax(scores)             # (N,) normalized [0,1]
    top_k_idx = argpartition(weights, -k)
    return [(values[i], weights[i]) for i in top_k_idx]
```

**β (Inverse Temperature)**: β = 8.0 (sharp attention, near-exact matching)

---

### Layer 2b: Memory — Residual Hopfield

**File**: `/Users/bhkmie/Desktop/foss-ki/core/residual_hopfield.py` (194 lines)

**What It Does**: Stores pre-extracted transformer hidden states (Qwen3 layer 10 and 18) as Hopfield patterns. Provides "stolen depth" — transformer-like semantic representations without running the model at inference.

**Data Source**:
```
qwen3_residuals.npz:
  word__layer_10: (2048,) hidden state after 10 transformer layers
  word__layer_18: (2048,) hidden state after 18 transformer layers
  (1877 words total, ~30MB compressed)
```

**Benefit**: A 512d query gets projected back to 2048d and matched against transformer-contextualized patterns, providing semantic richness without forward passes.

---

### Layer 2c: Memory — Knowledge Store

**File**: `/Users/bhkmie/Desktop/foss-ki/core/knowledge.py` (1032 lines)

**What It Does**: Fact store with three-tier lookup: dict-based exact matching, entity-based fast lookup, and Hopfield fuzzy matching. Anti-hallucination via attractor distance measurement.

**Data**:
- Triplets: 4,855 (S, R, O)
- Unique subjects: 1,317
- Unique relations: 279
- Encoder: GloVe 100d (pre-trained Wikipedia embeddings)
- Hopfield β = 64.0 (sharp matching)

**Three-Tier Lookup**:

1. **Tier 1: Dict Exact Match** (O(1)):
```python
_forward_index: (subject.lower(), relation.lower()) → [fact_idx, ...]
query(subject="france", relation="capital"):
    key = (subject.lower(), relation.lower())
    return facts[_forward_index.get(key)[0]]  # Instant lookup
```

2. **Tier 2: Entity Lookup** (O(k), k = facts per entity):
```python
_entity_subject_index: subject.lower() → [fact_idx, ...]
query(subject="france"):
    indices = _entity_subject_index.get(subject.lower())
    return [facts[i] for i in indices]
```

3. **Tier 3: Hopfield Fuzzy** (O(N), only on dict miss):
```python
Vectorized cosine similarity:
  query_sr = concat([encode(subject), encode(relation)])
  sims = (SR_matrix @ query_sr) / (SR_norms * ||query_sr||)
  best_idx = argmax(sims)
  confidence = sims[best_idx]
```

**Confidence Thresholds** (adaptive via Gumbel extreme value distribution):
- HIGH: similarity > high_threshold
- MEDIUM: low_threshold < similarity ≤ high_threshold
- REJECTED: similarity ≤ low_threshold
- UNKNOWN: no pattern matched

---

### Layer 2d: Causal DAG

**File**: `/Users/bhkmie/Desktop/foss-ki/core/causal_dag.py` (278 lines)

**What It Does**: Directed acyclic graph for interventional reasoning. Implements Pearl's do-calculus at simplified level: forward propagation, backward propagation, interventions, and common cause detection.

**Data**:
- Nodes: 20,042 (from KB + ConceptNet)
- Causal edges: 27,331
- Edge types: causes, leads_to, enables, produces, creates, results_in, triggers, generates, makes, needs, requires, depends_on, caused_by, prevents, blocks, inhibits, stops

**Key Operations**:
1. **Forward propagation**: What does X cause?
2. **Backward propagation**: What causes X?
3. **Interventional**: Pearl's do-calculus (remove incoming edges, propagate effects)
4. **Common cause detection**: Find confounders

---

### Layer 3: Consensus (Foss-Gossip)

**File**: `/Users/bhkmie/Desktop/foss-ki/core/consensus.py` (226 lines)

**What It Does**: Combines outputs from multiple sources via consensus protocol on barbell graph topology. PS-Lifted acceleration provides 26× speedup.

**Topology** (6 nodes, barbell):
```
    [0]     [3]
     |  \  /  |
    [1]   ×   [4]   3+3 cliques, 1 bridge
     |  /  \  |
    [2]     [5]
```

**Cluster Assignment**:
- Cluster 1 (pattern-based): reservoir, Hopfield, autoregressive
- Cluster 2 (knowledge-based): KB, multi-hop, causal DAG

**Speedup Mechanism**: PS-Lifted transition matrix has spectral gap ~ gap_base^0.32, giving 26× speedup on barbell (proven in T391)

---

### Layer 5: Spikes — Z₂ Novelty Detection

**What It Does**: Measures divergence between forward-biased (pos) and backward-biased (neg) reservoir states. High divergence = novel/surprising input.

**Mechanism**:
```python
novelty = mean(|state_pos - state_neg|)  # ∈ [0, 1]
```

**Interpretation**:
- novelty ≈ 0.05: common token
- novelty ≈ 0.20: moderately novel
- novelty ≈ 0.50: very unusual input

---

### Input Layer: Embeddings

**File**: `/Users/bhkmie/Desktop/foss-ki/core/embeddings.py` (340 lines)

**Raw Space**: (151936 tokens, 2048d) ≈ 1.2GB
**Projected Space**: (151936 tokens, 512d) via SVD, 38.6% variance retained
**SVD Matrix**: (512, 2048) projection, 4MB

---

## Data Flow Walkthrough

### Query Pipeline: "capital of France?"

```
1. INPUT PARSING
   Clean: "capital of france"
   Stop words removed: ["capital", "france"]

2. ATTENTION PHASE (3 sources)
   a) Extracted Attention (Q·K^T from Qwen3)
   b) Ricci Curvature (geometric importance)
   c) Z₂ Novelty (reservoir pos/neg divergence)

3. PARALLEL RETRIEVAL (3+ sources)
   a) RESERVOIR ESN
      Reset → Step 2 embeddings → 5 mix steps
      Predict via ridge readout → "Paris" (sim=0.87)

   b) HOPFIELD BANK
      Softmax attention over 1989 patterns
      Top match: "Paris" (score=0.92)

   c) KNOWLEDGE STORE
      Tier 1 dict: exact (S,R) match
      Tier 2 entity: find all facts with subject
      Tier 3 Hopfield: fuzzy SR-vector match
      Return: "Paris" (confidence=1.0)

   d) CAUSAL DAG
      Not applicable for "capital of" (not what-if)

   e) MULTI-HOP
      Not WHY/HOW question, skip

4. CONSENSUS COMBINATION
   Sources: [(reservoir, Paris, 0.87), (hopfield, Paris, 0.92), (kb, Paris, 1.0)]
   Barbell topology (6 nodes):
      Cluster 1 (patterns): 1.79 total conf
      Cluster 2 (knowledge): 1.0 total conf
   PS-Lifted consensus → speedup 26×
   Final confidence: 0.99

5. Z₂ NOVELTY GATING
   novelty < 0.5 → no penalty

6. NATURALIZATION
   Template: "The capital of France is Paris."
```

---

## Training Pipeline

### Data Preparation (16,748 samples)

1. **Multi-word objects** (30%):
   ```
   (france, capital, paris) → train on all intermediate tokens
   ```

2. **Cross-fact circular** (30%):
   ```
   Link facts sharing entities
   ```

3. **Related-fact rollout** (40%):
   ```
   Autoregressive continuation
   ```

**Training**:
```python
Ridge regression: W_out = (S^T S + λI)^{-1} S^T Y
λ = 1e-6 (minimal regularization)
```

---

## Data Files (in `/Users/bhkmie/Desktop/foss-ki/data/`)

| File | Size | Used By |
|------|------|---------|
| `qwen3_1.7b_embeddings.npy` | 1.2G | Reservoir, ICL, Hopfield |
| `qwen3_svd_V512.npy` | 4M | Pipeline projection |
| `qwen3_residuals.npz` | 30M | Residual Hopfield |
| `knowledge_full.json` | 2M | Knowledge Store |
| `conceptnet_en_500k.json` | 32M | CommonSense, Causal |
| `glove.6B.100d.txt` | 370M | Knowledge Store encoder |
| `reservoir_readout.npz` | 30M | Reservoir.predict() |

---

## Generation Pipeline

### Autoregressive Sequence: `generate_sequence("capital france", max_tokens=12)`

```
1. Encode input → reservoir state
2. Mixing steps
3. Loop (max_tokens):
   a. Predict next embedding
   b. Find nearest tokens (filter garbage)
   c. Z₂ novelty check
   d. CASI gate (monitor structure)
   e. Feed back (autoregressive)
4. Output: [(token, similarity, novelty), ...]
```

---

## In-Context Learning (ICL)

**Method 1: Translation Vector**
```python
delta = mean([embed(Paris) - embed(france), embed(Berlin) - embed(germany)])
spain_emb + delta → nearest neighbors → [Madrid, Barcelona, ...]
```

**Method 2: Sherman-Morrison** (rank-1 updates)
```python
Snapshot W_out → update with demos → predict → restore
```

---

## Novel Components

### 1. Foss Gap Theorem
Spectral gap of PS-Lifted Markov chains scales as gap_lifted ~ gap_base^0.32 on barbell topologies, giving 26× speedup. Used in consensus protocol.

### 2. Z₂ Parity Spikes
Novel approach: divergence between forward-biased and backward-biased recurrent states for novelty detection.

### 3. Anti-Hallucination via Attractor Distance
Use Hopfield attractor dynamics. Distance = ||attractor - query|| / ||query||. High distance → low confidence.

### 4. CASI Gate for Generation
Monitor CASI score (Compression-based Analytic Structural Index) of token similarity sequence. IEEE-validated (ICECET 2026, Paper #1142).

### 5. Ridge Readout (No Backprop)
Echo State Property guarantees stability without training recurrent weights. Only readout is trained via ridge regression.

---

## Performance Characteristics

### Inference Speed
- Tokenize + embed: 10ms
- Attention (3 sources): 15ms
- Reservoir (5 steps): 20ms
- Hopfield retrieve: 30ms
- KB query: <1ms (Tier 1-2), 50ms (Tier 3)
- Consensus: 25ms
- **Total: ~200ms** (CPU, parallel where possible)

### Memory Footprint
- Qwen3 embeddings: 1.2GB
- SVD projection: 4MB
- Reservoir readout: 30MB
- Hopfield bank: ~50MB
- Knowledge Store: ~5MB
- ConceptNet: 32MB
- **Total: ~1.3GB**

---

## Known Limitations

1. **Semantic drift in long sequences**: >12 tokens, quality degrades
2. **Limited common-sense reasoning**: No weighted causal propagation
3. **Dependency on pre-trained embeddings**: Qwen3 frozen
4. **No multi-token input parsing**: Treats word order weakly
5. **Hopfield capacity ceiling**: 2000 patterns is less than large KBs
6. **CASI gate requires live-casi library**: Fallback to variance check

---

## Testing & Validation

- Unit tests: Each component (reservoir, Hopfield, KB, causal, consensus)
- Integration tests: Full query → answer, generation, ICL
- Benchmarks: FB15k-237, BaBI Tasks 1-20, ConceptNet evaluation
- Regression tests: Monthly on reference queries

---

## Maintenance & Extension

### Add New Knowledge Source
1. Create module with `retrieve(question)` method
2. Add to `foss_pipeline.py` query() method
3. Update consensus topology if needed

### Retrain Reservoir
```bash
python repl.py  # Auto-generates samples, trains ridge
```

### Extract New Residual Patterns
```bash
python extract_residuals.py --words word_list.txt --layers 10,18
```

---

## References

- **Foss Gap Theorem**: T391 (26× speedup on barbell)
- **Echo State Networks**: Jaeger (2001), Maass (2002)
- **Modern Hopfield**: Ramsauer et al. (2020)
- **CASI Metric**: ICECET 2026 Paper #1142
- **Knowledge Graphs**: Wikidata, ConceptNet

---

## File Structure

```
/Users/bhkmie/Desktop/foss-ki/
├── core/
│   ├── foss_pipeline.py      # Central orchestration (957 lines)
│   ├── reservoir.py          # Echo State Network (245 lines)
│   ├── icl.py                # Translation Vector + Sherman-Morrison (320 lines)
│   ├── causal_dag.py         # Pearl's do-calculus (278 lines)
│   ├── casi_gate.py          # Generation quality gate (123 lines)
│   ├── hopfield_bank.py      # Modern Hopfield patterns (221 lines)
│   ├── residual_hopfield.py  # Pre-contextualized residuals (194 lines)
│   ├── consensus.py          # Foss Barbell gossip (226 lines)
│   ├── embeddings.py         # Qwen3 512d projections (340 lines)
│   ├── knowledge.py          # Anti-hallucination fact store (1032 lines)
│   ├── multi_hop.py          # Chaining for complex questions
│   ├── commonsense.py        # ConceptNet integration
│   └── [other modules]
├── repl.py                   # Interactive interface (2823 lines)
├── data/
│   ├── qwen3_1.7b_embeddings.npy
│   ├── qwen3_svd_V512.npy
│   ├── qwen3_residuals.npz
│   ├── knowledge_full.json
│   ├── conceptnet_en_500k.json
│   ├── glove.6B.100d.txt
│   └── external_benchmarks/
├── tests/
├── README.md
├── CLAUDE.md
└── ARCHITECTURE.md
```

---

**End of Document**

Last Updated: 2026-03-17 | For questions: refer to CLAUDE.md or README.md
