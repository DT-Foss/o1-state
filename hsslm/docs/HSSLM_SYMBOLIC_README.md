# HSSLM-S: Symbolic Language Module

**Zero neural components. Zero training. Pure deterministic mathematics.**

A language processing module built entirely from closed-form mathematical formulas — no PyTorch, no TensorFlow, no embeddings, no gradient descent. Every operation is deterministic: same input always produces same output.

## Architecture

```
Input Text → Tokenize → Symbol States (Ginibre-init) 
    → Möbius-SSM State Transitions → Z2 Topological Lift
    → 3-Pass Deterministic Inference → Weak Signal Amplification
    → Foss Gate Quality Filter → Contraction Sampler
    → Zeno/Anti-Zeno Scheduling → Output Token
```

## What's Inside (14 modules, 7,462 lines)

### Core Engine (3,086 lines)
| Module | Formulas | Description |
|--------|----------|-------------|
| `moebius_core.py` | F4–F24 | Möbius coupling, Lorentz factor, state transitions, Sinkhorn |
| `z2_lift.py` | F28–F47 | PS-Lifted Z2 doubling, quantum potential, Foss index |
| `inference.py` | F29–F56 | 3-pass deterministic inference, Jaro-Winkler, confidence |
| `sampler.py` | F33–F68 | Contraction sampler, BvN path integral, zeno scheduling |
| `foss_gate.py` | F54 | 14-step deterministic quality filter |
| `state_init.py` | F9–F15 | Ginibre kernel initialization, symbol state matrices |

### Novel Features (866 lines)
| Module | Score | Description |
|--------|-------|-------------|
| `egta.py` | **12.0** | Entropy-Gradient Tau Advection — self-tuning creativity |
| `csqg.py` | **7.9** | Cheeger Spectral Quality Gate — spectral output validation |
| `bphm.py` | **4.5** | Berry Phase Holographic Memory — repetition detection |

### Mac M4 + Parallel (3,510 lines)
| Module | Description |
|--------|-------------|
| `mac_optimize.py` | SIMD, memory mapping, CPU affinity, benchmarks |
| `parallel_engine.py` | Lock-free ring buffer, shared state, parallel generation |
| `backpressure.py` | Hysteresis-based flow control |
| `worker_pool.py` | Circuit breaker, crash recovery |

## Key Formulas Implemented

| ID | Formula | Source Paper |
|----|---------|-------------|
| F4 | `f(λ,v) = (λ+v)/(1+λv)` | Markov→Minkowski |
| F5 | `g(λ) = (1-λ²)^(-1/2)` | Lorentz factor |
| F17–F24 | Möbius-SSM state transitions | Contractive SSM |
| F28–F32 | PS-Lifted Z2 decomposition | Non-Reversibility |
| F33–F40 | Contraction sampler with τ control | Collapse is Contraction |
| F49–F56 | 3-pass deterministic inference | .causal Format |
| F63–F68 | BvN path integral decomposition | Gossip Consensus |

## Quick Start

```python
import numpy as np
from hsslm_s.moebius_core import moebius_couple, period_function
from hsslm_s.z2_lift import z2_decompose, compute_foss_index
from hsslm_s.inference import jaro_winkler, moebius_confidence
from hsslm_s.sampler import tau_to_temperature, contraction_sample
from hsslm_s.state_init import initialize_symbol_state
from hsslm_s.egta import egta_update, EGTAScheduler
from hsslm_s.csqg import compute_quality_gate
from hsslm_s.bphm import compute_berry_phase

# Möbius coupling
f = moebius_couple(0.5, 0.3)  # 0.6957

# Z2 decomposition
states = initialize_symbol_state(100, 32)
h = states[:10]
phys, mom = z2_decompose(h)
F = compute_foss_index(phys, mom)  # ~0.75

# Deterministic inference
sim = jaro_winkler("cellular", "cellular protection")  # 0.927
conf = moebius_confidence(0.9, 0.85)  # 0.9915

# Tau-controlled generation
temp = tau_to_temperature(0.65)  # 0.316

# Self-tuning tau (EGTA)
weights = np.array([0.3, 0.2, 0.15, 0.1, 0.25])
tau_new, _ = egta_update(0.65, weights)

# Spectral quality check
Q = compute_quality_gate(states[:10])  # [0, 1]

# Berry phase memory
state_list = [states[i] for i in range(5)]
phase = compute_berry_phase(state_list)
```

## Mac mini M4 Optimization

```python
from hsslm_s.mac_optimize import (
    moebius_simd, load_state_mmap, 
    set_performance_cores, benchmark_moebius
)
from hsslm_s.parallel_engine import ParallelHSSLMS

# SIMD-optimized operations
result = moebius_simd(lam_array, v_array)

# Memory-mapped state (instant loading)
states = load_state_mmap("model.state")

# Run on P-cores only
set_performance_cores()

# Parallel generation with different τ
engine = ParallelHSSLMS(n_workers=4)
streams = engine.generate(prompt, taus=[0.5, 0.65, 0.8])
```

## Design Documents

| Document | Content |
|----------|---------|
| `symbolic_core_design.md` | 68 formulas, 22 algorithms, 77 constants |
| `mac_m4_optimization.md` | SIMD, ANE, mmap, CPU affinity, benchmarks |
| `parallel_architecture.md` | Lock-free, worker pools, Amdahl analysis |
| `novel_approaches.md` | 6 novel features with scoring and pseudocode |

## Requirements

- Python 3.10+
- NumPy 1.24+
- 100MB RAM (typical)
- No GPU required
- Optional: Apple Silicon for M4 optimizations

## License

MIT
