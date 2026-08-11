# SPEC.md — HSSLM-C: Causal/Contraction Extensions to HSSLM

## Overview

Integrate David Tom Foss's mathematical frameworks into HSSLM to achieve:
- **30-40% parameter reduction** (contraction efficiency)
- **O(1) per-layer convergence** (PS-Lifted non-reversibility)
- **Deterministic inference chains** (.causal 3-pass engine)
- **Faster training convergence** (Ginibre initialization)
- **Quality-gated generation** (Foss Gate token filter)

## New Modules

### 1. `hsslm/moebius_ssm.py` — Möbius Contractive SSM Core

Replaces the standard SelectiveSSM with a Möbius-coupled contractive variant.

#### Key Formulas (from Foss papers)
- Möbius coupling: `f(λ, v) = (λ + v) / (1 + λv)`
- Period function: `g(λ) = (1 - λ²)^(-1/2)`
- Contraction coefficient: `τ = sup_{x≠y} d_TV(Wx, Wy) / d_TV(x, y)`

#### Classes

```python
class MoebiusCoupling:
    """Möbius coupling function f(λ,v) = (λ+v)/(1+λv)"""
    @staticmethod
    def forward(lam: Tensor, v: Tensor) -> Tensor:
        return (lam + v) / (1 + lam * v)
    
    @staticmethod
    def period(lam: Tensor) -> Tensor:
        """g(λ) = (1-λ²)^(-1/2)"""
        return (1 - lam.pow(2)).clamp(min=1e-6).pow(-0.5)
    
    @staticmethod
    def lorentz_factor(lam: Tensor, v: Tensor) -> Tensor:
        """γ(λ,v) = g(f(λ,v)) / g(λ) = (1+λv)/√(1-v²)"""
        return (1 + lam * v) / (1 - v.pow(2)).clamp(min=1e-6).sqrt()


class ContractiveSSM(nn.Module):
    """
    Contractive State Space Module with Möbius coupling.
    
    Key difference from SelectiveSSM:
    - Uses Möbius coupling for state transitions
    - Enforces contraction coefficient τ < 1
    - Period function g(λ) as intrinsic timescale
    
    Parameters (REDUCED from SelectiveSSM):
    - d_inner: 256 (was 512) → 50% reduction via contraction efficiency
    - d_state: 16 (same)
    - dt_rank: 8 (same)
    
    Architecture:
        x → [linear proj] → (λ, v) → [Möbius coupling] → h → [output]
                             ↓
                    τ-constraint: τ < 1 enforced via spectral clamping
    """
    def __init__(self, d_inner: int = 256, d_state: int = 16, dt_rank: int = 8, tau_max: float = 0.95):
        # tau_max: maximum contraction coefficient (τ < 1)
        
    def forward(self, x: Tensor, state: Optional[Tensor] = None) -> Tuple[Tensor, Tensor]:
        # 1. Project x to (lambda, v) pairs
        # 2. Apply Möbius coupling: h' = f(lambda, v) * h
        # 3. Enforce τ < tau_max via spectral clamping
        # 4. Apply period function for timescale gating
        # 5. Output projection


class PSLiftedBlock(nn.Module):
    """
    PS-Lifted variant with Z2 state-space doubling.
    
    Each physical state has forward (+) and backward (-) copy.
    Fiedler-vector-like orientation creates non-reversible flow.
    
    Architecture:
        x → [split ±] → x_+, x_- → [ContractiveSSM] → h_+, h_- → [combine] → h
        
    Physical projection: h = h_+ + h_- (token probabilities)
    Momentum projection: m = h_+ - h_- (driving field)
    """
    def __init__(self, d_model: int = 256, d_state: int = 16, pc: float = 0.65, ps: float = 0.003):
        # pc: forward probability (from PS-Lifted)
        # ps: self-loop probability
        
    def forward(self, x: Tensor, state: Optional[Tensor] = None) -> Tuple[Tensor, Tensor]:
        # 1. Split into + and - channels
        # 2. Forward flow with probability pc
        # 3. Backward flow with probability pr = 1 - pc - ps
        # 4. Self-loop with probability ps
        # 5. Combine: physical = + + -, momentum = + - -


class MoebiusStateSpaceCore(nn.Module):
    """Stack of PSLiftedBlocks replacing StateSpaceCore."""
    def __init__(self, n_layers: int = 4, d_model: int = 256, d_state: int = 16, tau_max: float = 0.95):
        # n_layers REDUCED from 6 to 4 (contraction = fewer layers needed)
        # d_inner REDUCED from 512 to 256
```

### 2. `hsslm/causal_inference.py` — Deterministic Transitive Inference Engine

Embedded 3-pass inference engine (from .causal format).

```python
class CausalInferenceEngine:
    """
    Three-pass deterministic inference for token relationships.
    
    Pass 1: Exact token matching (chain: A→B, B→C ⇒ A→C)
    Pass 2: Semantic direction propagation (+/-/0)
    Pass 3: Fuzzy token matching (similarity ≥ 0.85)
    
    Confidence: Möbius addition f(c1,c2) = (c1+c2)/(1+c1*c2)
    Quality filter: threshold 0.30 removes contradictions
    """
    
    def __init__(self, vocab_size: int = 16384, threshold: float = 0.30):
        
    def pass1_exact_matching(self, tokens: List[int]) -> Dict[Tuple, float]:
        """Find exact token chains. O(n×r) complexity."""
        
    def pass2_semantic_direction(self, chains: Dict) -> Dict[Tuple, float]:
        """Propagate direction: (+)+(+)→(+), (-)+(-)→(+), (+)+(-)→(-)"""
        
    def pass3_fuzzy_matching(self, tokens: List[int], threshold: float = 0.85) -> Dict[Tuple, float]:
        """Jaro-Winkler fuzzy matching for token similarity."""
        
    def moebius_confidence(self, c1: float, c2: float) -> float:
        """f(c1,c2) = (c1+c2)/(1+c1*c2). Prevents confidence decay at hubs."""
        return (c1 + c2) / (1 + c1 * c2)
    
    def transitive_closure(self, tokens: List[int]) -> Dict[Tuple, float]:
        """Full 3-pass deterministic inference with provenance tracking."""
        # Returns: {chain: confidence, ...} with provenance for each


class WeakSignalAmplifier(nn.Module):
    """
    Amplify weak token signals through transitive inference.
    
    3 tokens → 21+ tokens (7x amplification)
    Applied to embeddings before SSM processing.
    """
    def __init__(self, inference_engine: CausalInferenceEngine, d_model: int = 256):
        
    def forward(self, embeddings: Tensor, token_ids: Tensor) -> Tensor:
        # 1. Extract token co-occurrence patterns
        # 2. Run transitive closure
        # 3. Amplify embeddings with inferred relationships
        # 4. Return amplified embeddings
```

### 3. `hsslm/ginibre_init.py` — Ginibre Kernel Initialization

```python
class GinibreInitializer:
    """
    Initialize weight matrices using Ginibre kernel statistics.
    
    Target: ⟨s²⟩ = 1.08747... (2D NND second moment)
    Cubic repulsion: β = 3 for weight distribution
    Sinkhorn renormalization for stability
    """
    GINIBRE_KERNEL_VALUE = 1.08746866652609
    
    @staticmethod
    def initialize_weight(shape: Tuple[int, ...], asymmetry: float = 0.5) -> Tensor:
        """
        Initialize with 2D spectral properties.
        
        1. Generate random matrix with asymmetry parameter ε
        2. Apply Sinkhorn renormalization (doubly stochastic projection)
        3. Scale to target spectral statistics
        """
        
    @staticmethod
    def sinkhorn_renormalize(W: Tensor, n_iter: int = 10) -> Tensor:
        """Project W to doubly stochastic manifold via Sinkhorn."""
        # Alternating row/column normalization
        # Converges in ~10 iterations
        
    @staticmethod
    def verify_spectral_stats(W: Tensor) -> float:
        """Measure ⟨s²⟩ of weight matrix eigenvalue distribution."""
        return s2_measured


def ginibre_init_(tensor: Tensor, asymmetry: float = 0.5):
    """In-place initialization following Ginibre kernel statistics."""
```

### 4. `hsslm/foss_gate.py` — Foss Gate Token Quality Filter

```python
class FossGate:
    """
    14-step deterministic quality filter for generated tokens.
    
    From .causal format quality assurance pipeline.
    """
    
    def __init__(self, vocab_size: int = 16384, contamination_markers: Optional[List] = None):
        
    def validate(self, token_id: int, context: List[int]) -> Tuple[bool, str]:
        """
        Run full 14-step validation on a proposed token.
        
        Returns: (passed, reason_if_failed)
        """
        
    def step1_3_field_validation(self, token_id: int) -> bool:
        """P1-P3: Required fields, length limits, vocab bounds."""
        
    def step4_5_tautology_detection(self, token_id: int, context: List[int]) -> bool:
        """P4-P5: Exact and semantic duplicate detection."""
        
    def step6_9_quality_signals(self, token_id: int, context: List[int]) -> bool:
        """P6-P9: Causal language patterns, mechanism quality."""
        
    def step10_evidence_validation(self, token_id: int, context: List[int]) -> bool:
        """P10: Reject unsupported claims."""
        
    def step11_quantification_check(self, token_id: int, context: List[int]) -> bool:
        """P11: Verify quantification in source."""
        
    def step12_13_artifact_detection(self, token_id: int) -> bool:
        """P12-P13: Encoding errors, format artifacts."""
        
    def step14_contamination_filter(self, token_id: int) -> bool:
        """P14: Few-shot leakage prevention."""
```

### 5. `hsslm/contraction_inference.py` — Contraction-Based Generation

```python
class ContractionSampler:
    """
    Token sampling controlled by contraction coefficient τ.
    
    τ → 1: More diverse/creative (near quantum phase transition)
    τ → 0: More deterministic/focused (strong contraction)
    
    Zeno effect: frequent sampling slows diversity
    Anti-Zeno: optimal sampling interval speeds coherence
    """
    
    def __init__(self, tau_default: float = 0.65, zeno_interval: int = 5):
        # tau_default from Foss's measured τ = 0.508, rounded up for creativity
        # zeno_interval k* ≈ 5 for anti-Zeno speedup (41% faster)
        
    def sample(self, logits: Tensor, tau: Optional[float] = None) -> int:
        """
        Sample token with τ-controlled temperature.
        
        temperature = g(tau) = (1 - tau²)^(-1/2) - 1
        tau=0.5 → temp ≈ 0.15 (focused)
        tau=0.8 → temp ≈ 0.67 (creative)
        tau=0.95 → temp ≈ 2.20 (very creative)
        """
        
    def zeno_schedule(self, step: int) -> float:
        """
        Adaptive τ based on Zeno/anti-Zeno schedule.
        
        step % k* == 0: τ = tau_default (anti-Zeno speedup)
        step % k* != 0: τ = tau_default * 0.8 (Zeno slowdown)
        
        Net effect: optimal coherence at minimal cost.
        """


class Z2TopologicalLift(nn.Module):
    """
    Z2 topological lift for the full model.
    
    Doubles latent space: physical + momentum projections.
    Quantum potential Q_i = m_i / x_i guides generation.
    Topological protection: 80% parameter tolerance.
    """
    
    def __init__(self, d_model: int = 256):
        
    def forward(self, h: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        """
        Returns: (physical, momentum, quantum_potential)
        """
        # physical = (h_+ + h_-) / 2
        # momentum = (h_+ - h_-) / 2
        # Q = momentum / (physical + eps)
```

## Modified Modules

### `hsslm/model.py` — HSSLM-C Full Model

Replace StateSpaceCore with MoebiusStateSpaceCore:
- n_layers: 6 → 4 (2 fewer layers via contraction efficiency)
- d_inner: 512 → 256 (50% reduction)
- Add CausalInferenceEngine
- Add WeakSignalAmplifier  
- Add FossGate
- Add ContractionSampler
- Add Z2TopologicalLift

Target parameter reduction: 8.6M → ~5-6M

## Data Flow

```
Input Tokens
    |
    v
[HierarchicalTokenizer] → token_ids + boundaries
    |
    v
[HierarchicalEmbedding] → embeddings
    |
    v
[WeakSignalAmplifier] → amplified embeddings (deterministic inference)
    |
    v
[Z2TopologicalLift] → physical + momentum projections
    |
    v
[MoebiusStateSpaceCore x4] → contractive SSM processing
    |                           (Möbius coupling + τ < 1)
    |
    v
[HierarchicalComposer] → word/phrase/sentence/discourse
    |
    v
[LMHead] → logits
    |
    v
[ContractionSampler τ=0.65] → sampled token
    |
    v
[FossGate] → quality validation → output token
```

## Integration Points

1. **MoebiusStateSpaceCore** replaces StateSpaceCore in model.py
2. **CausalInferenceEngine** integrated into tokenizer + embedding pipeline
3. **GinibreInitializer** used for ALL weight initialization
4. **FossGate** applied at end of generation loop
5. **ContractionSampler** replaces standard temperature sampling
6. **Z2TopologicalLift** applied after embedding, before SSM core

## Training Changes

1. Weight init: Use `ginibre_init_` for all nn.Linear and nn.Embedding
2. Optimizer: Same (AdamW, lr=6e-4)
3. Loss: Same (next-token CE + auxiliary hierarchical losses)
4. Convergence: Expected ~35K steps (vs 50K) due to better initialization

## Testing Requirements

Each module must pass:
1. Shape tests: input/output dimensions match
2. Parameter count: track reduction targets
3. Numerical stability: no NaN/Inf with τ constraints
4. Integration: works with rest of HSSLM pipeline
