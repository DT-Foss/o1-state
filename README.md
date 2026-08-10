# FOSS-KI — Non-Transformer AI Engine

A complete language understanding and generation system built from reservoir computing, Hopfield memory, and Foss Gap Theorem mathematics. **Zero transformer layers at inference time.**

## What It Does

- Answers factual questions ("capital of France?" → "Paris")
- Generates semantically coherent sequences (autoregressive rollout)
- Learns in-context from examples (Translation Vector + Sherman-Morrison ICL)
- Performs causal/interventional reasoning ("what if X disappeared?")
- Self-monitors generation quality (CASI gate)

## Architecture

```
Input → Tokenize → Qwen3 512d Embeddings (SVD Top-512 projection)
                          ↓
               ┌──────────┴──────────┐
               │   Reservoir ESN     │  2048 nodes, Foss Barbell topology
               │   Z₂ Parity Copy   │  pos/neg state → novelty detection
               │   Ridge Readout     │  only trainable component (16748 samples)
               └──────────┬──────────┘
                          ↓
         ┌────────────────┼────────────────┐
         │                │                │
   Pattern Sources   Knowledge Sources   Causal
   ├─ Reservoir      ├─ KB (4855 facts)  ├─ CausalDAG
   ├─ Hopfield Bank  ├─ MultiHop Chain   │  (20K nodes)
   └─ AR Rollout     └─ ConceptNet       └─ do-calculus
         │                │                │
         └────────┬───────┘                │
                  ↓                        │
        Foss Consensus Ensemble ←──────────┘
        (3+3 Barbell, PS-Lifted 26× speedup)
                  ↓
           CASI Gate → Stop if structure collapses
                  ↓
              Response
```

### Key Components

| Component | File | Lines | What |
|-----------|------|-------|------|
| **Foss Pipeline** | `core/foss_pipeline.py` | 957 | Central orchestration |
| **Reservoir ESN** | `core/reservoir.py` | 245 | Fixed-weight recurrent processor |
| **ICL Engine** | `core/icl.py` | 320 | Translation Vector + Sherman-Morrison |
| **Causal DAG** | `core/causal_dag.py` | 278 | Pearl's do-calculus |
| **CASI Gate** | `core/casi_gate.py` | 123 | Generation quality monitor |
| **Hopfield Bank** | `core/hopfield_bank.py` | 221 | Modern Hopfield (β=8.0, 1989 patterns) |
| **Residual Hopfield** | `core/residual_hopfield.py` | 194 | Stolen Qwen3 Layer 10+18 contexts |
| **Consensus** | `core/consensus.py` | 226 | Foss Barbell gossip protocol |
| **REPL** | `repl.py` | 2823 | Interactive interface |

### Data Requirements (not tracked in git)

These files must be generated before first run:

| File | Size | Generator | What |
|------|------|-----------|------|
| `qwen3_1.7b_embeddings.npy` | 1.2G | `repl.py` (auto) | Raw token embeddings |
| `qwen3_svd_V512.npy` | 4M | `repl.py` (auto) | SVD projection matrix |
| `qwen3_residuals.npz` | ~30M | `extract_residuals.py` | 1877 pre-contextualized vectors |
| `qwen3_mlp_facts.npz` | ~2M | `extract_mlp_facts.py` | 198 MLP fact vectors |
| `knowledge_full.json` | ~2M | `repl.py` (auto) | 4855 KB triplets |
| `conceptnet_en_500k.json` | 32M | `core/conceptnet_loader.py` | ConceptNet subset |
| `reservoir_readout.npz` | 30M | `repl.py` (auto) | Trained readout weights |
| `layer_profile.npz` | <1M | `profile_layers.py` | Layer importance map |

## Quick Start

```bash
# 1. Install dependencies
pip install mlx mlx-lm numpy

# 2. Start REPL (auto-downloads Qwen3-1.7B on first run, auto-trains reservoir)
python repl.py

# 3. Ask questions
> capital of France?
> who wrote Hamlet?
> /icl france:Paris germany:Berlin spain:? → Madrid
> /generate capital france
> /causal what if the sun disappeared?
```

## What Makes This Different

This is NOT a stripped-down transformer. Every component uses different mathematics:

- **Reservoir**: Fixed random weights, leaky integration, no backprop
- **Hopfield**: Energy-based associative memory, softmax attention
- **Consensus**: Gossip protocol on barbell graph, spectral gap optimization
- **ICL**: Linear algebra (translation vectors, rank-1 matrix updates)
- **Causal**: DAG propagation, Pearl's do-calculus
- **CASI**: IEEE-validated compression metric (ICECET 2026, Paper #1142)

The Foss Gap Theorem (spectral gap on PS-Lifted Markov chains) provides the mathematical foundation for consensus convergence: 26× speedup on barbell topologies.

## Research Foundation

| Result | Speedup | Paper |
|--------|---------|-------|
| PS-Lifted Consensus on Barbell | 26× | T391 |
| Classical > Quantum Mixing | O(m^0.38) vs O(m^1.01) | T385 |
| Near-Optimal Convergence | 33× | T382 |
| Metastability Breaking | 20,085× | T201 |
| Topology Change Detection | AUC=1.000 | T376 |

## License

Private. All rights reserved.

## Author

David Tom Foss — david@foss.com.de
