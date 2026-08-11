# SPEC.md - Hierarchical State-Space Language Module (HSSLM)

## Overview

- **Purpose**: Minimal non-transformer language model with explicit multi-level linguistic processing
- **Architecture**: Selective State Space Model (Mamba-style S6) with learned hierarchical composition
- **Target**: ~7.3M parameters, CPU/GPU runnable, pure PyTorch (no custom CUDA)
- **Language**: English (BPE-tokenized)
- **License**: MIT

### Key Design Decisions
1. **Pure PyTorch** - No mamba_ssm, no FlashAttention, no custom kernels. Every line is inspectable.
2. **Modular hierarchy** - Hierarchical composers can be disabled; model works as flat LM.
3. **Weight tying** - Input/output embeddings shared to save 4.2M parameters.
4. **Hierarchical supervision** - Auxiliary losses at word/phrase/sentence/discourse levels.

---

## 1. Module Structure

### 1.1 Tokenizer Module

**File**: `hsslm/tokenizer.py`

```python
import re
from typing import List, Tuple, Dict, Optional
import torch


class HierarchicalTokenizer:
    """
    BPE-based tokenizer with hierarchical boundary detection.
    
    Wraps a pre-trained BPE tokenizer and adds linguistic boundary markers
    for the hierarchical composer.
    """
    
    # Class constants
    VOCAB_SIZE: int = 16384
    PAD_TOKEN_ID: int = 0
    UNK_TOKEN_ID: int = 1
    BOS_TOKEN_ID: int = 2
    EOS_TOKEN_ID: int = 3
    
    def __init__(self, vocab_file: Optional[str] = None) -> None:
        """
        Initialize tokenizer.
        
        Args:
            vocab_file: Path to BPE vocabulary file. If None, uses simple
                       character-level fallback for testing.
        """
        ...
    
    def encode(
        self,
        text: str,
        add_bos: bool = True,
        add_eos: bool = False,
        max_length: int = 2048
    ) -> Dict[str, torch.Tensor]:
        """
        Encode text to token IDs with boundary information.
        
        Args:
            text: Raw input string
            add_bos: Prepend BOS token
            add_eos: Append EOS token
            max_length: Truncate to this length
            
        Returns:
            Dict with keys:
                - "input_ids": (L,) token ID tensor
                - "word_boundaries": (W, 2) word span tensor [start_idx, end_idx]
                - "sentence_boundaries": (S, 2) sentence span tensor
                - "attention_mask": (L,) binary mask tensor
        """
        ...
    
    def decode(
        self,
        token_ids: torch.Tensor,
        skip_special: bool = True
    ) -> str:
        """
        Decode token IDs back to string.
        
        Args:
            token_ids: (L,) or (B, L) tensor of token IDs
            skip_special: Remove BOS/EOS/PAD tokens
            
        Returns:
            Decoded string (or list of strings if batch)
        """
        ...
    
    def get_morpheme_hints(
        self,
        token_ids: torch.Tensor
    ) -> torch.Tensor:
        """
        Get morpheme boundary hints for each token.
        
        BPE tokens often align with morpheme boundaries. This method
        returns a score indicating how likely each token boundary is
        also a morpheme boundary.
        
        Args:
            token_ids: (L,) tensor
            
        Returns:
            (L-1,) float tensor in [0, 1], score per inter-token boundary
        """
        ...
    
    def batch_encode(
        self,
        texts: List[str],
        padding: bool = True,
        max_length: int = 2048
    ) -> Dict[str, torch.Tensor]:
        """
        Batch encode multiple texts.
        
        Args:
            texts: List of raw strings
            padding: Pad to max length in batch
            max_length: Max sequence length
            
        Returns:
            Dict with batched tensors:
                - "input_ids": (B, L)
                - "attention_mask": (B, L)
                - "word_boundaries": list of (W, 2) tensors per sample
                - "sentence_boundaries": list of (S, 2) tensors per sample
        """
        ...
```

### 1.2 Embedding Module

**File**: `hsslm/embedding.py`

```python
import torch
import torch.nn as nn
from typing import Dict


class HierarchicalEmbedding(nn.Module):
    """
    Token embedding layer with positional encoding.
    
    Simple learned embeddings - character-level decomposition is handled
    by the BPE tokenizer (subword units naturally capture morpheme-like
    structure).
    """
    
    def __init__(
        self,
        vocab_size: int = 16384,
        d_model: int = 256,
        max_seq_len: int = 2048,
        dropout: float = 0.1
    ) -> None:
        """
        Args:
            vocab_size: Token vocabulary size (16384)
            d_model: Embedding dimension (256)
            max_seq_len: Maximum sequence length for position embeddings
            dropout: Dropout rate
        """
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.max_seq_len = max_seq_len
        
        self.token_embedding: nn.Embedding = ...      # (16384, 256) = 4,194,304 params
        self.position_embedding: nn.Embedding = ...   # (2048, 256) = 524,288 params
        self.norm: RMSNorm = ...                        # 256 params
        self.dropout: nn.Dropout = ...
    
    def forward(
        self,
        input_ids: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            input_ids: (B, L) token IDs
            
        Returns:
            (B, L, 256) contextualized embeddings
        """
        ...
    
    def get_embedding_table(self) -> torch.Tensor:
        """Return the token embedding weight matrix for weight tying."""
        ...
```

### 1.3 Core SSM/Recurrent Engine

**File**: `hsslm/core_engine.py`

This is the heart of the model. A pure PyTorch implementation of the Mamba selective SSM block.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization."""
    
    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps
        self.weight: nn.Parameter = nn.Parameter(torch.ones(dim))  # (dim,)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (*, dim) input
        Returns:
            (*, dim) RMS-normalized output
        """
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight


class SelectiveSSM(nn.Module):
    """
    Selective State Space Model (S6/Mamba-style).
    
    Implements the selective scan mechanism in pure PyTorch.
    Uses recurrence for both training and inference (no convolution mode)
    for simplicity. This is slightly slower for training but functionally
    equivalent and much simpler to implement.
    
    State equations (continuous):
        h'(t) = A * h(t) + B * x(t)
        y(t) = C * h(t) + D * x(t)
    
    Discretized (zero-order hold):
        h[t] = discretize(A) * h[t-1] + discretize(B) * x[t]
        y[t] = C * h[t] + D * x[t]
    
    Selectivity: B, C, Delta are input-dependent (projected from x).
    """
    
    def __init__(
        self,
        d_inner: int = 512,
        d_state: int = 16,
        dt_rank: int = 8,
        bias: bool = False
    ) -> None:
        """
        Args:
            d_inner: Expanded dimension (d_model * expand = 256 * 2 = 512)
            d_state: SSM state dimension per channel (16)
            dt_rank: Rank for Delta (step size) projection (8)
            bias: Whether to use bias in linear layers
        """
        super().__init__()
        self.d_inner = d_inner
        self.d_state = d_state
        self.dt_rank = dt_rank
        
        # x_proj: project input to dt, B, C
        self.x_proj: nn.Linear = nn.Linear(d_inner, dt_rank + 2 * d_state, bias=False)
        # 512 * 40 = 20,480 params
        
        # dt_proj: project rank-rank dt to full dimension
        self.dt_proj: nn.Linear = nn.Linear(dt_rank, d_inner, bias=True)
        # 8 * 512 + 512 = 4,608 params
        
        # A_log: learned parameter for state transition (initialized as HiPPO)
        self.A_log: nn.Parameter = nn.Parameter(torch.log(torch.arange(1, d_state + 1)).repeat(d_inner, 1))
        # 512 * 16 = 8,192 params
        
        # D: skip connection parameter
        self.D: nn.Parameter = nn.Parameter(torch.ones(d_inner))
        # 512 params
    
    def forward(
        self,
        x: torch.Tensor,
        state: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Selective SSM forward pass (recurrent mode).
        
        Args:
            x: (B, L, d_inner) input sequence
            state: Optional (B, d_inner, d_state) previous state for inference
            
        Returns:
            - y: (B, L, d_inner) output sequence
            - state: (B, d_inner, d_state) final state
        """
        # 1. Project input to Delta, B, C
        x_proj_out = self.x_proj(x)  # (B, L, dt_rank + 2*d_state)
        delta, B, C = torch.split(
            x_proj_out,
            [self.dt_rank, self.d_state, self.d_state],
            dim=-1
        )
        # delta: (B, L, dt_rank)
        # B: (B, L, d_state)
        # C: (B, L, d_state)
        
        # 2. Project delta to full dimension and apply softplus
        delta = F.softplus(self.dt_proj(delta))  # (B, L, d_inner)
        
        # 3. Compute discretized A (input-dependent via delta)
        A = -torch.exp(self.A_log.float())  # (d_inner, d_state)
        # A_discrete: (B, L, d_inner, d_state)
        A_discrete = torch.exp(A.unsqueeze(0).unsqueeze(0) * delta.unsqueeze(-1))
        
        # 4. Discretize B
        B_discrete = B.unsqueeze(2) * delta.unsqueeze(-1)  # (B, L, 1, d_state)
        
        # 5. Recurrent scan
        B_batch, L, _ = x.shape
        h = torch.zeros(B_batch, self.d_inner, self.d_state, device=x.device, dtype=x.dtype) if state is None else state
        
        ys = []
        for t in range(L):
            h = A_discrete[:, t] * h + B_discrete[:, t] * x[:, t].unsqueeze(-1)
            # h: (B, d_inner, d_state)
            y = torch.sum(h * C[:, t].unsqueeze(1), dim=-1)  # (B, d_inner)
            y = y + self.D * x[:, t]  # skip connection
            ys.append(y)
        
        y = torch.stack(ys, dim=1)  # (B, L, d_inner)
        return y, h


class MambaBlock(nn.Module):
    """
    Complete Mamba block: projection -> conv -> SSM -> gating -> output.
    
    Architecture:
        x ----->[in_proj]-----> x_branch -->[conv+SiLU]-->[SSM]-->*-->[out_proj]--> + --> y
          |                       |                                      ^           ^
          |                       +---> z_branch -----------------------+           |
          |                                                                         |
          +------------------------------------------------------------------------>+
        (residual connection around entire block)
    """
    
    def __init__(
        self,
        d_model: int = 256,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        dt_rank: int = 8,
        bias: bool = False
    ) -> None:
        """
        Args:
            d_model: Model dimension (256)
            d_state: SSM state dimension (16)
            d_conv: Local convolution kernel size (4)
            expand: Expansion factor for inner dimension (2)
            dt_rank: Delta projection rank (8)
            bias: Use bias in linear layers
        """
        super().__init__()
        self.d_model = d_model
        self.d_inner = d_model * expand  # 512
        
        self.norm: RMSNorm = RMSNorm(d_model)  # 256 params
        
        self.in_proj: nn.Linear = nn.Linear(d_model, self.d_inner * 2, bias=bias)
        # 256 * 1024 = 262,144 params (x2 for x_branch and z_branch)
        
        self.conv1d: nn.Conv1d = nn.Conv1d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            kernel_size=d_conv,
            padding=d_conv - 1,
            groups=self.d_inner,
            bias=True
        )
        # 512 * 4 + 512 = 2,560 params (depthwise separable)
        
        self.ssm: SelectiveSSM = SelectiveSSM(
            d_inner=self.d_inner,
            d_state=d_state,
            dt_rank=dt_rank,
            bias=bias
        )
        # 20,480 + 4,608 + 8,192 + 512 = 33,792 params
        
        self.out_proj: nn.Linear = nn.Linear(self.d_inner, d_model, bias=bias)
        # 512 * 256 = 131,072 params
    
    def forward(
        self,
        x: torch.Tensor,
        state: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: (B, L, d_model) input
            state: Optional previous SSM state
            
        Returns:
            - output: (B, L, d_model)
            - state: Final SSM state for recurrent inference
        """
        # Residual connection
        residual = x
        
        # RMSNorm
        x = self.norm(x)
        
        # Input projection to two branches
        xz = self.in_proj(x)  # (B, L, d_inner * 2)
        x_branch, z_branch = xz.chunk(2, dim=-1)  # each (B, L, d_inner)
        
        # Short convolution on x_branch
        x_conv = self.conv1d(x_branch.transpose(1, 2))  # (B, d_inner, L + pad)
        x_conv = x_conv[:, :, :x_branch.size(1)]  # crop padding
        x_conv = x_conv.transpose(1, 2)  # (B, L, d_inner)
        x_conv = F.silu(x_conv)
        
        # Selective SSM
        x_ssm, state = self.ssm(x_conv, state)  # (B, L, d_inner)
        
        # Gating: multiply with z_branch
        x_gated = x_ssm * F.silu(z_branch)  # (B, L, d_inner)
        
        # Output projection
        output = self.out_proj(x_gated)  # (B, L, d_model)
        
        # Residual
        return output + residual, state


class StateSpaceCore(nn.Module):
    """
    Stack of MambaBlocks forming the core sequence processing engine.
    """
    
    def __init__(
        self,
        n_layers: int = 6,
        d_model: int = 256,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        dt_rank: int = 8
    ) -> None:
        """
        Args:
            n_layers: Number of Mamba blocks (6)
            d_model: Model dimension (256)
            d_state: SSM state dimension (16)
            d_conv: Convolution kernel size (4)
            expand: Expansion factor (2)
            dt_rank: Delta rank (8)
        """
        super().__init__()
        self.n_layers = n_layers
        self.d_model = d_model
        self.d_state = d_state
        self.d_inner = d_model * expand
        
        self.layers: nn.ModuleList = nn.ModuleList([
            MambaBlock(d_model, d_state, d_conv, expand, dt_rank)
            for _ in range(n_layers)
        ])
        
        self.final_norm: RMSNorm = RMSNorm(d_model)
    
    def forward(
        self,
        x: torch.Tensor,
        states: Optional[list] = None
    ) -> Tuple[torch.Tensor, list]:
        """
        Args:
            x: (B, L, d_model) embedded input
            states: Optional list of layer states for recurrent inference
            
        Returns:
            - output: (B, L, d_model) processed sequence
            - states: List of final states per layer
        """
        new_states = []
        for i, layer in enumerate(self.layers):
            state = states[i] if states is not None else None
            x, new_state = layer(x, state)
            new_states.append(new_state)
        
        x = self.final_norm(x)
        return x, new_states
```

### 1.4 Hierarchical Composer

**File**: `hsslm/hierarchy.py`

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Optional, Dict


class CompositionLayer(nn.Module):
    """
    Generic 2-layer MLP composition module with residual connection.
    Transforms pooled representations at each linguistic level.
    """
    
    def __init__(self, d_model: int = 256, dropout: float = 0.1) -> None:
        super().__init__()
        self.norm = RMSNorm(d_model)
        self.linear1 = nn.Linear(d_model, d_model * 2)
        self.linear2 = nn.Linear(d_model * 2, d_model)
        self.dropout = nn.Dropout(dropout)
        # Total: 256*512 + 512 + 512*256 + 256 = 262,912 params
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, N, d_model) pooled representations
        Returns:
            (B, N, d_model) composed representations
        """
        residual = x
        x = self.norm(x)
        x = self.linear1(x)
        x = F.silu(x)
        x = self.dropout(x)
        x = self.linear2(x)
        return x + residual


class WordComposer(nn.Module):
    """
    Compose token-level representations into word-level representations.
    
    Uses word boundary information from tokenizer to mean-pool
    token states within each word span.
    """
    
    def __init__(self, d_model: int = 256) -> None:
        super().__init__()
        self.composition = CompositionLayer(d_model)
    
    def forward(
        self,
        token_states: torch.Tensor,
        word_boundaries: List[torch.Tensor]
    ) -> Tuple[List[torch.Tensor], torch.Tensor]:
        """
        Args:
            token_states: (B, L, d_model) token-level hidden states
            word_boundaries: List of (W, 2) tensors per batch item [start, end]
            
        Returns:
            - word_states: List of (W, d_model) tensors per batch item
            - word_pooled: (B, W_max, d_model) padded tensor (for batching)
        """
        ...


class PhraseComposer(nn.Module):
    """
    Compose word-level representations into phrase-level representations.
    
    Uses local attention-weighted aggregation over 3-word windows.
    """
    
    def __init__(self, d_model: int = 256, window_size: int = 3) -> None:
        super().__init__()
        self.window_size = window_size
        self.query = nn.Linear(d_model, 1)  # attention scoring
        self.composition = CompositionLayer(d_model)
    
    def forward(
        self,
        word_states: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            word_states: (B, W, d_model) word-level states
            
        Returns:
            (B, P, d_model) phrase-level states (P <= W)
        """
        ...


class SentenceComposer(nn.Module):
    """
    Compose phrase-level representations into sentence-level representations.
    
    Max-pools over all phrases within each sentence, then MLP transform.
    """
    
    def __init__(self, d_model: int = 256) -> None:
        super().__init__()
        self.composition = CompositionLayer(d_model)
    
    def forward(
        self,
        phrase_states: torch.Tensor,
        sentence_boundaries: List[torch.Tensor]
    ) -> torch.Tensor:
        """
        Args:
            phrase_states: (B, P, d_model) phrase-level states
            sentence_boundaries: List of (S, 2) tensors per batch [start, end]
            
        Returns:
            (B, S, d_model) sentence-level states
        """
        ...


class DiscourseComposer(nn.Module):
    """
    Compose sentence-level representations into discourse-level representations.
    
    Uses a learned gated recurrence (mini-SSM) to accumulate sentence
    states across the document, modeling discourse coherence.
    """
    
    def __init__(self, d_model: int = 256) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(d_model * 2, d_model)  # gating function
        self.composition = CompositionLayer(d_model)
        self.discourse_state = None  # running state
    
    def forward(
        self,
        sentence_states: torch.Tensor,
        reset_state: bool = False
    ) -> torch.Tensor:
        """
        Args:
            sentence_states: (B, S, d_model) sentence-level states
            reset_state: If True, reset discourse state (new document)
            
        Returns:
            (B, S, d_model) discourse-enriched sentence representations
        """
        ...
    
    def reset(self) -> None:
        """Reset internal discourse state. Call at document boundaries."""
        self.discourse_state = None


class HierarchicalComposer(nn.Module):
    """
    Full hierarchical composition pipeline.
    
    Takes token-level SSM outputs and builds multi-level representations.
    Can be disabled (forward passes through with minimal overhead).
    """
    
    def __init__(
        self,
        d_model: int = 256,
        enabled: bool = True
    ) -> None:
        super().__init__()
        self.enabled = enabled
        self.word_composer = WordComposer(d_model)
        self.phrase_composer = PhraseComposer(d_model)
        self.sentence_composer = SentenceComposer(d_model)
        self.discourse_composer = DiscourseComposer(d_model)
        # Total: 4 * 262,912 = ~1,051,648 params
    
    def forward(
        self,
        token_states: torch.Tensor,
        boundaries: Dict[str, List[torch.Tensor]],
        return_all_levels: bool = False
    ) -> Dict[str, torch.Tensor]:
        """
        Run full hierarchical composition.
        
        Args:
            token_states: (B, L, d_model) from SSM core
            boundaries: Dict with "word" and "sentence" boundary tensors
            return_all_levels: If True, return all intermediate representations
            
        Returns:
            Dict with:
                - "token": (B, L, d_model) original token states
                - "word": (B, W, d_model) word-level (if enabled)
                - "phrase": (B, P, d_model) phrase-level (if enabled)
                - "sentence": (B, S, d_model) sentence-level (if enabled)
                - "discourse": (B, S, d_model) discourse-level (if enabled)
        """
        ...
```

### 1.5 Language Modeling Head

**File**: `hsslm/lm_head.py`

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class LMHead(nn.Module):
    """
    Language modeling head for next-token prediction.
    
    Uses weight tying with input embeddings to save parameters.
    """
    
    def __init__(
        self,
        d_model: int = 256,
        vocab_size: int = 16384,
        embedding_weight: Optional[nn.Parameter] = None
    ) -> None:
        """
        Args:
            d_model: Model dimension (256)
            vocab_size: Vocabulary size (16384)
            embedding_weight: Shared embedding weight for tying
        """
        super().__init__()
        if embedding_weight is not None:
            self.weight = embedding_weight  # tied
            self.out_proj = None
        else:
            self.out_proj = nn.Linear(d_model, vocab_size, bias=False)
    
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Args:
            hidden_states: (B, L, d_model) from model core
            
        Returns:
            (B, L, vocab_size) logits
        """
        if self.out_proj is None:
            return F.linear(hidden_states, self.weight)
        return self.out_proj(hidden_states)


class AuxiliaryHeads(nn.Module):
    """
    Auxiliary prediction heads for hierarchical supervision.
    
    These are used during training only (to guide the model to learn
    meaningful hierarchical representations). During inference, they
    can optionally provide linguistic analysis.
    """
    
    def __init__(self, d_model: int = 256) -> None:
        super().__init__()
        # Word-level POS tagging (17 Universal POS tags)
        self.pos_head = nn.Linear(d_model, 17)
        
        # Phrase boundary detection (binary)
        self.phrase_boundary_head = nn.Linear(d_model, 2)
        
        # Sentence relation prediction (8 discourse relations)
        self.sentence_relation_head = nn.Linear(d_model, 8)
        
        # Discourse coherence scoring (scalar)
        self.coherence_head = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.SiLU(),
            nn.Linear(d_model, 1)
        )
    
    def forward(
        self,
        hierarchical_outputs: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            hierarchical_outputs: Dict from HierarchicalComposer
            
        Returns:
            Dict with auxiliary predictions:
                - "pos_logits": (B, W, 17)
                - "phrase_boundary_logits": (B, P, 2)
                - "sentence_relation_logits": (B, S, 8)
                - "coherence_scores": (B, S-1, 1)
        """
        ...
```

### 1.6 Full Model

**File**: `hsslm/model.py`

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple, List

from .embedding import HierarchicalEmbedding
from .core_engine import StateSpaceCore
from .hierarchy import HierarchicalComposer
from .lm_head import LMHead, AuxiliaryHeads


class HSSLM(nn.Module):
    """
    Hierarchical State-Space Language Module.
    
    A minimal non-transformer language model with explicit hierarchical
    linguistic processing. Based on selective state space models (Mamba-style)
    with learned composition modules for multi-scale representation.
    
    Architecture:
        Input tokens -> Embedding -> [SSM layers x6] -> Hierarchical Composer -> LM Head
                                                     |-> Auxiliary Heads (training)
    
    Parameters: ~7.3M
    Complexity: O(n) in sequence length
    """
    
    def __init__(self, config: Optional[Dict] = None) -> None:
        """
        Args:
            config: Configuration dict. Defaults below:
                - vocab_size: 16384
                - d_model: 256
                - n_layers: 6
                - d_state: 16
                - d_conv: 4
                - expand: 2
                - dt_rank: 8
                - max_seq_len: 2048
                - dropout: 0.1
                - hierarchical: True (enable/disable composers)
                - aux_loss_weight: 0.1
        """
        super().__init__()
        config = config or {}
        
        self.vocab_size = config.get('vocab_size', 16384)
        self.d_model = config.get('d_model', 256)
        self.n_layers = config.get('n_layers', 6)
        self.d_state = config.get('d_state', 16)
        self.d_conv = config.get('d_conv', 4)
        self.expand = config.get('expand', 2)
        self.dt_rank = config.get('dt_rank', 8)
        self.max_seq_len = config.get('max_seq_len', 2048)
        self.dropout = config.get('dropout', 0.1)
        self.hierarchical = config.get('hierarchical', True)
        self.aux_loss_weight = config.get('aux_loss_weight', 0.1)
        
        # 1. Embedding module
        self.embedding = HierarchicalEmbedding(
            vocab_size=self.vocab_size,
            d_model=self.d_model,
            max_seq_len=self.max_seq_len,
            dropout=self.dropout
        )
        
        # 2. Core SSM engine (6 layers)
        self.core = StateSpaceCore(
            n_layers=self.n_layers,
            d_model=self.d_model,
            d_state=self.d_state,
            d_conv=self.d_conv,
            expand=self.expand,
            dt_rank=self.dt_rank
        )
        
        # 3. Hierarchical composer
        self.composer = HierarchicalComposer(
            d_model=self.d_model,
            enabled=self.hierarchical
        )
        
        # 4. LM head (weight-tied to embeddings)
        self.lm_head = LMHead(
            d_model=self.d_model,
            vocab_size=self.vocab_size,
            embedding_weight=self.embedding.token_embedding.weight
        )
        
        # 5. Auxiliary heads (training only)
        self.aux_heads = AuxiliaryHeads(d_model=self.d_model)
    
    def forward(
        self,
        input_ids: torch.Tensor,
        boundaries: Optional[Dict[str, List[torch.Tensor]]] = None,
        labels: Optional[torch.Tensor] = None,
        return_hierarchy: bool = False,
        states: Optional[list] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass.
        
        Args:
            input_ids: (B, L) token IDs
            boundaries: Optional word/sentence boundaries from tokenizer
            labels: Optional (B, L) target IDs for loss computation
            return_hierarchy: Return hierarchical representations
            states: Optional list of layer states for recurrent inference
            
        Returns:
            Dict with:
                - "logits": (B, L, vocab_size) next-token logits
                - "loss": scalar loss (if labels provided)
                - "hierarchy": hierarchical representations (if requested)
                - "states": final layer states (for recurrent generation)
        """
        ...
    
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 100,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        eos_token_id: Optional[int] = None
    ) -> torch.Tensor:
        """
        Auto-regressive text generation.
        
        Uses recurrent mode for efficient inference (O(1) per step).
        
        Args:
            input_ids: (B, L) prompt token IDs
            max_new_tokens: Number of tokens to generate
            temperature: Sampling temperature (1.0 = no change)
            top_k: If set, sample from top-k tokens only
            top_p: If set, use nucleus sampling
            eos_token_id: Stop generation on this token
            
        Returns:
            (B, L + max_new_tokens) generated token IDs
        """
        ...
    
    def analyze(
        self,
        text: str,
        tokenizer: 'HierarchicalTokenizer'
    ) -> Dict[str, any]:
        """
        Run hierarchical linguistic analysis on input text.
        
        Returns representations at all linguistic levels.
        
        Args:
            text: Input text string
            tokenizer: Tokenizer instance
            
        Returns:
            Dict with:
                - "tokens": List of token strings
                - "words": List of word representations
                - "phrases": List of phrase representations  
                - "sentences": List of sentence representations
                - "discourse": Discourse-level representation
        """
        ...
    
    def get_parameter_count(self) -> Dict[str, int]:
        """Return parameter count breakdown by component."""
        return {
            'embedding': sum(p.numel() for p in self.embedding.parameters()),
            'core': sum(p.numel() for p in self.core.parameters()),
            'composer': sum(p.numel() for p in self.composer.parameters()),
            'lm_head': sum(p.numel() for p in self.lm_head.parameters()),
            'aux_heads': sum(p.numel() for p in self.aux_heads.parameters()),
            'total': sum(p.numel() for p in self.parameters()),
        }
```

---

## 2. Data Flow

```
TEXT INPUT (e.g., "The cat sat on the mat.")
    |
    v
+---------------------------------------------+
|  HierarchicalTokenizer.encode()             |
|  - BPE tokenization                         |
|  - Word boundary detection (whitespace)     |
|  - Sentence boundary detection (.!? )       |
+---------------------------------------------+
    |
    v  input_ids (B, L) + boundaries dict
+---------------------------------------------+
|  HierarchicalEmbedding                      |
|  - Token embedding lookup (16384 x 256)     |
|  - Position embedding (2048 x 256)          |
|  - RMSNorm + Dropout                        |
+---------------------------------------------+
    |
    v  embeddings (B, L, 256)
+---------------------------------------------+
|  StateSpaceCore (6 MambaBlocks)             |
|                                             |
|  for i in range(6):                         |
|    x = MambaBlock(x)                        |
|      - RMSNorm                              |
|      - in_proj: 256 -> 1024                 |
|      - conv1d: kernel=4                     |
|      - SelectiveSSM: state=16, dt_rank=8    |
|      - gating (SiLU)                        |
|      - out_proj: 512 -> 256                 |
|      - residual add                         |
|  x = final RMSNorm(x)                       |
+---------------------------------------------+
    |
    v  token_states (B, L, 256)
+---------------------------------------------+
|  HierarchicalComposer (optional)            |
|                                             |
|  WordComposer                               |
|    - Mean-pool tokens per word span         |
|    - MLP composition (256 -> 512 -> 256)    |
|    -> word_states (B, W, 256)               |
|                                             |
|  PhraseComposer                             |
|    - Local attention over 3-word window     |
|    - MLP composition                        |
|    -> phrase_states (B, P, 256)             |
|                                             |
|  SentenceComposer                           |
|    - Max-pool phrases per sentence          |
|    - MLP composition                        |
|    -> sentence_states (B, S, 256)           |
|                                             |
|  DiscourseComposer                          |
|    - Gated recurrence across sentences      |
|    - MLP composition                        |
|    -> discourse_states (B, S, 256)          |
+---------------------------------------------+
    |                           |
    v                           v
+-------------+     +-------------------------+
|  LMHead     |     |  AuxiliaryHeads (train) |
|  weight-tied|     |  - POS tagging          |
|  256 -> 16K |     |  - Phrase boundaries    |
+-------------+     |  - Sentence relations   |
    |               |  - Coherence scoring    |
    v               +-------------------------+
 logits (B, L, 16K)
    |
    v
+---------------------------------------------+
|  Loss Computation (training only)           |
|  L_total = L_lm + 0.1*(L_word + L_phrase + |
|                        L_sentence + L_disc) |
+---------------------------------------------+
```

---

## 3. Interfaces

### 3.1 Complete Function Signatures

```python
# ============================================================================
# config.py - Model Configuration
# ============================================================================

@dataclass
class HSSLMConfig:
    """Complete configuration for HSSLM model."""
    vocab_size: int = 16384
    d_model: int = 256
    n_layers: int = 6
    d_state: int = 16
    d_conv: int = 4
    expand: int = 2
    dt_rank: int = 8
    max_seq_len: int = 2048
    dropout: float = 0.1
    hierarchical: bool = True
    aux_loss_weight: float = 0.1
    
    # Training
    learning_rate: float = 6e-4
    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.95
    grad_clip: float = 1.0
    warmup_steps: int = 2000
    max_steps: int = 50000
    batch_size: int = 32
    seq_length: int = 2048
    
    # Inference
    temperature: float = 1.0
    top_k: Optional[int] = 50
    top_p: Optional[float] = 0.9


# ============================================================================
# tokenizer.py - HierarchicalTokenizer
# ============================================================================

def encode(self, text: str, add_bos: bool = True, add_eos: bool = False,
           max_length: int = 2048) -> Dict[str, torch.Tensor]: ...

def decode(self, token_ids: torch.Tensor, skip_special: bool = True) -> str: ...

def get_morpheme_hints(self, token_ids: torch.Tensor) -> torch.Tensor: ...

def batch_encode(self, texts: List[str], padding: bool = True,
                 max_length: int = 2048) -> Dict[str, torch.Tensor]: ...


# ============================================================================
# embedding.py - HierarchicalEmbedding
# ============================================================================

def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
    """Returns: (B, L, 256)"""

def get_embedding_table(self) -> torch.Tensor:
    """Returns: (vocab_size, 256) weight matrix"""


# ============================================================================
# core_engine.py - StateSpaceCore
# ============================================================================

def forward(self, x: torch.Tensor, states: Optional[list] = None
           ) -> Tuple[torch.Tensor, list]:
    """Input: (B, L, 256) -> Output: (B, L, 256), [states]"""


# MambaBlock
def forward(self, x: torch.Tensor, state: Optional[torch.Tensor] = None
           ) -> Tuple[torch.Tensor, torch.Tensor]:
    """Input: (B, L, 256) -> Output: (B, L, 256), state"""


# SelectiveSSM
def forward(self, x: torch.Tensor, state: Optional[torch.Tensor] = None
           ) -> Tuple[torch.Tensor, torch.Tensor]:
    """Input: (B, L, 512) -> Output: (B, L, 512), state"""


# ============================================================================
# hierarchy.py - HierarchicalComposer
# ============================================================================

def forward(self, token_states: torch.Tensor,
            boundaries: Dict[str, List[torch.Tensor]],
            return_all_levels: bool = False
           ) -> Dict[str, torch.Tensor]:
    """
    Input:
        token_states: (B, L, 256)
        boundaries: {"word": [...], "sentence": [...]}
    Output:
        {
            "token": (B, L, 256),
            "word": (B, W, 256),
            "phrase": (B, P, 256),
            "sentence": (B, S, 256),
            "discourse": (B, S, 256)
        }
    """


# ============================================================================
# lm_head.py - LMHead + AuxiliaryHeads
# ============================================================================

# LMHead
def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
    """Input: (B, L, 256) -> Output: (B, L, 16384) logits"""

# AuxiliaryHeads
def forward(self, hierarchical_outputs: Dict[str, torch.Tensor]
           ) -> Dict[str, torch.Tensor]:
    """
    Output:
        {
            "pos_logits": (B, W, 17),
            "phrase_boundary_logits": (B, P, 2),
            "sentence_relation_logits": (B, S, 8),
            "coherence_scores": (B, S-1, 1)
        }
    """


# ============================================================================
# model.py - HSSLM (Full Model)
# ============================================================================

def forward(self, input_ids: torch.Tensor,
            boundaries: Optional[Dict] = None,
            labels: Optional[torch.Tensor] = None,
            return_hierarchy: bool = False,
            states: Optional[list] = None
           ) -> Dict[str, torch.Tensor]:
    """
    Output:
        {
            "logits": (B, L, 16384),
            "loss": scalar (if labels),
            "hierarchy": Dict (if return_hierarchy),
            "states": list (for recurrent generation)
        }
    """

def generate(self, input_ids: torch.Tensor, max_new_tokens: int = 100,
             temperature: float = 1.0, top_k: Optional[int] = None,
             top_p: Optional[float] = None,
             eos_token_id: Optional[int] = None) -> torch.Tensor:
    """Returns: (B, L + max_new_tokens) generated IDs"""

def analyze(self, text: str, tokenizer: HierarchicalTokenizer) -> Dict[str, any]: ...

def get_parameter_count(self) -> Dict[str, int]: ...
```

---

## 4. File Layout

```
project/
|   README.md                    # Project documentation
|   requirements.txt             # Python dependencies
|   setup.py                     # Package setup
|
+---hsslm/                      # Main package
|   |   __init__.py             # Package exports
|   |   config.py               # HSSLMConfig dataclass
|   |   tokenizer.py            # HierarchicalTokenizer
|   |   embedding.py            # HierarchicalEmbedding
|   |   core_engine.py          # RMSNorm, SelectiveSSM, MambaBlock, StateSpaceCore
|   |   hierarchy.py            # CompositionLayer, *Composer, HierarchicalComposer
|   |   lm_head.py              # LMHead, AuxiliaryHeads
|   |   model.py                # HSSLM full model
|   |
|   +---utils/                  # Utility functions
|   |       __init__.py
|   |       training.py         # Training loop helpers
|   |       generation.py       # Sampling utilities
|
+---scripts/
|   |   train.py                # Main training script
|   |   inference.py            # Interactive inference CLI
|   |   evaluate.py             # Evaluation on benchmarks
|   |   demo.py                 # Quick demo script
|
+---tests/
    |   test_tokenizer.py
    |   test_model.py
    |   test_hierarchy.py
    |   test_generation.py
```

---

## 5. Training Protocol

### 5.1 Dataset

| Stage | Dataset | Tokens | Purpose |
|---|---|---|---|
| Warmup | OpenWebText subset | 100M | Stabilize SSM dynamics |
| Main | SlimPajama | 2-3B | Primary training |
| Fine-tune | Target domain | Varies | Task adaptation |

**Minimum viable training**: 100M tokens (achieves basic coherence)
**Recommended training**: 2-3B tokens (Chinchilla-optimal for 7M params)

### 5.2 Training Loop

```python
# Pseudocode
def train_step(batch):
    # Forward
    outputs = model(
        input_ids=batch['input_ids'],
        boundaries=batch['boundaries'],
        labels=batch['labels']
    )
    
    loss = outputs['loss']
    
    # Backward
    loss.backward()
    
    # Gradient clip (CRITICAL for SSM stability)
    torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
    
    optimizer.step()
    scheduler.step()
    optimizer.zero_grad()
    
    return loss.item()
```

### 5.3 Optimizer Configuration

```python
optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=6e-4,
    betas=(0.9, 0.95),
    weight_decay=0.1
)

scheduler = get_cosine_schedule_with_warmup(
    optimizer,
    num_warmup_steps=2000,
    num_training_steps=50000,
    min_lr_ratio=0.1  # decays to 6e-5
)
```

### 5.4 Loss Function

```python
def compute_loss(outputs, labels):
    # Primary: next-token cross-entropy
    lm_loss = F.cross_entropy(
        outputs['logits'].view(-1, vocab_size),
        labels.view(-1),
        ignore_index=PAD_TOKEN_ID
    )
    
    if model.hierarchical and 'aux' in outputs:
        aux = outputs['aux']
        aux_loss = (
            0.1 * aux.get('pos_loss', 0) +
            0.05 * aux.get('phrase_boundary_loss', 0) +
            0.05 * aux.get('sentence_relation_loss', 0) +
            0.02 * aux.get('coherence_loss', 0)
        )
        return lm_loss + model.aux_loss_weight * aux_loss
    
    return lm_loss
```

### 5.5 Checkpointing Strategy

- Save every 5,000 steps
- Keep top-3 checkpoints by validation loss
- Final checkpoint is best validation checkpoint, not last

---

## 6. Inference Protocol

### 6.1 Generation

```python
def generate(model, tokenizer, prompt, max_new_tokens=100):
    # Encode
    inputs = tokenizer.encode(prompt, add_bos=True)
    input_ids = inputs['input_ids'].unsqueeze(0)  # (1, L)
    
    # Generate autoregressively
    model.eval()
    with torch.no_grad():
        output_ids = model.generate(
            input_ids,
            max_new_tokens=max_new_tokens,
            temperature=0.8,
            top_k=50,
            top_p=0.9,
            eos_token_id=tokenizer.EOS_TOKEN_ID
        )
    
    # Decode
    return tokenizer.decode(output_ids[0])
```

### 6.2 Hierarchical Analysis

```python
def analyze(model, tokenizer, text):
    model.eval()
    with torch.no_grad():
        analysis = model.analyze(text, tokenizer)
    
    # Returns:
    # {
    #   'tokens': ['The', 'cat', 'sat', ...],
    #   'words': [{'text': 'The', 'vector': tensor(...)}, ...],
    #   'phrases': [{'span': (0, 2), 'vector': tensor(...)}, ...],
    #   'sentences': [{'text': 'The cat sat.', 'vector': tensor(...)}],
    #   'discourse': tensor(...)  # document-level vector
    # }
```

### 6.3 Streaming Generation (Recurrent Mode)

For long-form generation, use the recurrent state to avoid reprocessing:

```python
# First, process prompt
outputs = model(input_ids, states=None)
states = outputs['states']

# Then generate token-by-token, carrying state forward
for _ in range(max_new_tokens):
    outputs = model(next_input.unsqueeze(1), states=states)
    logits = outputs['logits']
    states = outputs['states']
    next_token = sample(logits, temperature, top_k, top_p)
```

This is O(1) per step in memory (state is fixed size) vs O(n) for Transformers.

---

## 7. Performance Targets

### 7.1 Training Speed (A100 GPU)

| Batch Size | Seq Length | Tokens/sec | Time per 1B tokens |
|---|---|---|---|
| 32 | 2048 | ~120,000 | ~2.3 hours |

### 7.2 Inference Speed

| Model | 7.3M HSSLM | 124M GPT-2 | 130M Mamba |
|---|---|---|---|
| Tokens/sec (batch=1) | ~850 | ~120 | ~1,200* |
| Memory (seq=8K) | ~45 MB | ~180 MB | ~50 MB* |

*Mamba with optimized CUDA kernel. Our pure PyTorch implementation will be ~30% slower but still competitive.

### 7.3 Quality Targets (Perplexity on OpenWebText validation)

| Training Tokens | Target PPL | Baseline (GPT-2 small) |
|---|---|---|
| 100M | ~35-40 | ~25 (trained on 10B) |
| 1B | ~25-30 | ~25 (trained on 10B) |
| 3B | ~18-22 | ~25 (trained on 10B) |

Note: Our model has 1/17th the parameters of GPT-2 small. PPL in the 20-35 range is competitive for this parameter count.

---

## 8. Implementation Notes

### 8.1 SSM Stability

- Always use **bf16 or fp32** for SSM parameters (not fp16)
- Gradient clipping at 1.0 is **mandatory**
- Initialize Delta bias to small positive values (0.01-0.1)
- If NaN occurs: reduce learning rate by 2x, increase warmup

### 8.2 Hierarchical Boundary Detection

Word boundaries come from the tokenizer (whitespace-delimited BPE tokens).
Sentence boundaries detected by `.` `!` `?` token IDs.
For more accuracy, integrate with a proper sentence segmenter (e.g., NLTK).

### 8.3 Memory Optimization

- Gradient checkpointing can reduce memory by ~40% at ~15% speed cost
- For inference on CPU, use `torch.inference_mode()` and float32
- State vectors are (B, d_inner, d_state) = (B, 512, 16) = 32KB per layer - negligible

### 8.4 Extending the Model

To add a new linguistic level:
1. Create new `XxxComposer` class inheriting from `nn.Module`
2. Add to `HierarchicalComposer.__init__`
3. Add auxiliary head if training signal needed
4. Update `forward()` to call new composer

---

*SPEC version: 1.0*
*Target implementation: Pure PyTorch, no external dependencies beyond torch/tiktoken*
*Estimated implementation effort: 2-3 hours (core) + 2 hours (training/infrastructure)*
