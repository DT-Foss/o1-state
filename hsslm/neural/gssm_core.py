"""
GSSM-SELECTIVE core: Geometric State Space Model, as an HSSLM core variant.

Alternative to core_engine.py's Mamba-style S6 core. Same external contract
(forward(x, states) -> (x, new_states), init_states(batch_size, device)) so
HSSLM.forward can select between them via config.core = 's6' | 'gssm'
without touching embedding/composer/lm_head/streaming.

Source: David Tom Foss, "Geometric State Space Models: Bounded Hyperbolic
Recurrence for Language Modeling" (SYMBOLISCH/hsslm_s/PREPRINT.md, read-only
source). Breakthrough commit 3bf577e "DeepSeek uebernimmt: Log-Komplement
Rapiditaet" in that repo's git history; SELECTIVE variant lands at 614ac56
("T=128: SELECTIVE schlaegt ALLE — Mamba, LRU, RWKV-4").

Core identity (Theorem 1, preprint):
    1 - f_S(s,v)^2 = (1-v^2)(1-s^2)     [sqrt-coupling is additive in log(1-s^2)]

Recurrence (Theorem 2, preprint -- the [0,1] boundedness guarantee):
    z_0 = 0
    z_t = gamma(x_t) * z_{t-1} + alpha(x_t) * log(1 - v_t^2)
    s_t = sqrt(1 - exp(z_t))                         in [0,1] for ALL t, ALWAYS

This module is NOT a copy-paste of moebius_scan_transformer_selective.py --
it reimplements the same math (verified identical formulas, cross-checked
against the source) but wired to HSSLM's own state-tuple convention (see
core_engine.py's MambaBlock: state = (recurrent_state, extra_context)) so a
streaming caller (HSSLMStreamer) that already knows how to detach-carry
core_engine.py's states does the same here without special-casing.

This block has NO local convolution (unlike MambaBlock's depthwise conv1d),
so there is no conv-buffer half of the state tuple -- state is just the
recurrent z (one tensor per layer), wrapped in a 1-tuple for structural
symmetry with core_engine.py's (ssm_state, conv_buffer) 2-tuple. A caller
iterating layer states generically (detach_state_tree in streaming.py)
handles both shapes since it recurses over whatever nesting it finds.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, List

LOG_COMPLEMENT_CLAMP = 0.999
EPS = 1e-6


def _assert_bounded_state(s: torch.Tensor, where: str) -> None:
    """Theorem 2 runtime check: s_t must be in [0,1] for ALL t, ALWAYS --
    the preprint's guarantee is unconditional (holds for any input, any
    gate values, no clipping/normalization required). This assertion is
    cheap (one min/max reduction) and documents the guarantee at its new
    integration point in HSSLM, not just in the original prototype repo."""
    with torch.no_grad():
        s_min = s.min().item()
        s_max = s.max().item()
    assert -1e-4 <= s_min and s_max <= 1.0 + 1e-4, (
        f"GSSM boundedness violation at {where}: state range "
        f"[{s_min:.6f}, {s_max:.6f}] outside the proven [0,1] bound "
        f"(Theorem 2). This should be structurally impossible -- indicates "
        f"a bug in the log-complement recurrence, not a training instability."
    )


def sequential_linear_scan(
    a: torch.Tensor, gamma: torch.Tensor, z0: Optional[torch.Tensor] = None
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Stateful scan for z_t = gamma_t * z_{t-1} + a_t.

    a, gamma: (B, T, H, D). z0: (B, H, D) or None (zeros).
    Returns (Z_seq: (B, T, H, D), Z_final: (B, H, D)).

    O(T) sequential loop -- matches the preprint's own current
    implementation (Section 4.4/6.3 notes a parallel Blelloch scan, O(log T),
    as future work; not implemented here either, kept faithful to source).
    """
    B, T, H, D = a.shape
    Z = torch.zeros(B, H, D, device=a.device, dtype=a.dtype) if z0 is None else z0
    out = []
    for t in range(T):
        Z = gamma[:, t] * Z + a[:, t]
        out.append(Z)
    return torch.stack(out, dim=1), Z


class GSSMSelectiveScan(nn.Module):
    """Data-dependent gated scan in log-complement space (GSSM-SELECTIVE).

    Four signals per token:
        v_t     = tanh(W_v x_t)        token velocity in (-1, 1)
        gate_t  = sigmoid(W_gate x_t)  value gate
        gamma_t = sigmoid(W_gamma x_t) forget gate
        alpha_t = sigmoid(W_alpha x_t) input gate

    a_t = alpha_t * log(1 - (gate_t * v_t)^2)
    z_t = gamma_t * z_{t-1} + a_t
    s_t = sqrt(1 - exp(z_t))   -- bounded in [0,1] by Theorem 2, asserted below.
    """

    def __init__(
        self,
        d_model: int,
        d_head: int = 32,
        n_heads: int = 4,
        dropout: float = 0.0,
        check_bounds: bool = True,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.d_head = d_head
        self.n_heads = n_heads
        self.check_bounds = check_bounds
        self.dropout = nn.Dropout(dropout) if dropout > 0 else None

        total_dim = n_heads * d_head
        self.W_v = nn.Linear(d_model, total_dim, bias=False)
        self.W_gate = nn.Linear(d_model, total_dim, bias=False)
        self.W_gamma = nn.Linear(d_model, total_dim, bias=False)
        self.W_alpha = nn.Linear(d_model, total_dim, bias=False)
        self.W_out = nn.Linear(total_dim, d_model, bias=False)

        # Preprint init recipe: gamma/alpha start near-open (gain=0.1, sigmoid
        # ~0.5) so the model begins near a uniform cumsum and learns
        # selectivity from there; v/gate/out use a larger gain (0.6).
        for module in [self.W_gamma, self.W_alpha]:
            for p in module.parameters():
                if p.dim() >= 2:
                    nn.init.xavier_uniform_(p, gain=0.1)
        for module in [self.W_v, self.W_gate, self.W_out]:
            for p in module.parameters():
                if p.dim() >= 2:
                    nn.init.xavier_uniform_(p, gain=0.6)

    def forward(
        self, x: torch.Tensor, z0: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: (B, L, d_model) input.
            z0: Optional (B, n_heads, d_head) carried recurrent state.

        Returns:
            - output: (B, L, d_model).
            - z_final: (B, n_heads, d_head) final recurrent state, for the
              next chunk (detach yourself for streaming carry).
        """
        B, L, D = x.shape

        v = torch.tanh(self.W_v(x))
        gate = torch.sigmoid(self.W_gate(x))
        gamma = torch.sigmoid(self.W_gamma(x))
        alpha = torch.sigmoid(self.W_alpha(x))

        v_gated = v * gate
        if self.dropout is not None:
            v_gated = self.dropout(v_gated)

        v_gated = v_gated.view(B, L, self.n_heads, self.d_head)
        gamma = gamma.view(B, L, self.n_heads, self.d_head)
        alpha = alpha.view(B, L, self.n_heads, self.d_head)

        w = torch.clamp(v_gated * v_gated, max=LOG_COMPLEMENT_CLAMP)
        z_in = torch.log(1.0 - w + EPS)
        a = alpha * z_in

        Z_seq, z_final = sequential_linear_scan(a, gamma, z0)

        s_sq = torch.clamp(1.0 - torch.exp(Z_seq), min=0.0)
        state = torch.sqrt(s_sq + EPS)

        if self.check_bounds:
            _assert_bounded_state(state, where="GSSMSelectiveScan.forward")

        state = state.view(B, L, self.n_heads * self.d_head)
        return self.W_out(state), z_final


class RMSNorm(nn.Module):
    """Same RMSNorm as core_engine.py, duplicated here so gssm_core.py has
    no import dependency on the S6 core (the two cores stay independently
    swappable)."""

    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        norm = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return norm * self.weight


class GSSMBlock(nn.Module):
    """Complete GSSM block: RMSNorm -> scan -> residual -> RMSNorm -> FFN -> residual.

    Mirrors MambaBlock's residual/norm placement (RMSNorm pre-scan, not the
    preprint's own post-norm LayerNorm placement) so GSSM and S6 blocks are
    structurally comparable inside HSSLM's stack, not just numerically
    different recurrences bolted onto different scaffolding.
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int = 4,
        d_head: int = 32,
        dropout: float = 0.0,
        check_bounds: bool = True,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.norm = RMSNorm(d_model)
        self.scan = GSSMSelectiveScan(
            d_model, d_head=d_head, n_heads=n_heads, dropout=dropout,
            check_bounds=check_bounds,
        )
        self.norm2 = RMSNorm(d_model)
        ffn_dim = 4 * d_model
        self.ffn = nn.Sequential(
            nn.Linear(d_model, ffn_dim),
            nn.GELU(),
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
            nn.Linear(ffn_dim, d_model),
        )

    def forward(
        self, x: torch.Tensor, state: Optional[Tuple[Optional[torch.Tensor]]] = None
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor]]:
        """
        Args:
            x: (B, L, d_model) input.
            state: Optional 1-tuple (z0,) carried recurrent state from a
                previous chunk (1-tuple for structural symmetry with
                core_engine.py's (ssm_state, conv_buffer) 2-tuple; this
                block has no local convolution so there's nothing to put in
                a second slot).

        Returns:
            - output: (B, L, d_model).
            - state: (z_final,) 1-tuple for the next chunk.
        """
        z0 = state[0] if state is not None else None

        residual = x
        h = self.norm(x)
        scan_out, z_final = self.scan(h, z0)
        x = residual + scan_out

        residual = x
        h = self.norm2(x)
        x = residual + self.ffn(h)

        return x, (z_final,)


class GSSMCore(nn.Module):
    """Stack of GSSMBlocks -- the GSSM-SELECTIVE analogue of core_engine.py's
    StateSpaceCore. Same forward/init_states contract, so HSSLM.forward can
    swap this in for the S6 StateSpaceCore via config.core."""

    def __init__(
        self,
        n_layers: int = 6,
        d_model: int = 256,
        n_heads: int = 4,
        d_head: Optional[int] = None,
        dropout: float = 0.0,
        check_bounds: bool = True,
    ) -> None:
        super().__init__()
        self.n_layers = n_layers
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_head if d_head is not None else max(1, d_model // n_heads)

        self.layers = nn.ModuleList([
            GSSMBlock(
                d_model=d_model, n_heads=n_heads, d_head=self.d_head,
                dropout=dropout, check_bounds=check_bounds,
            )
            for _ in range(n_layers)
        ])
        self.final_norm = RMSNorm(d_model)

    def forward(
        self,
        x: torch.Tensor,
        states: Optional[List[Tuple[Optional[torch.Tensor]]]] = None,
    ) -> Tuple[torch.Tensor, List[Tuple[torch.Tensor]]]:
        """
        Args:
            x: (B, L, d_model) embedded input.
            states: Optional list of per-layer (z,) tuples.

        Returns:
            - output: (B, L, d_model) processed sequence.
            - states: List of final (z,) tuples per layer.
        """
        if states is None:
            states = [None] * self.n_layers

        new_states = []
        for i, layer in enumerate(self.layers):
            x, new_state = layer(x, states[i])
            new_states.append(new_state)

        x = self.final_norm(x)
        return x, new_states

    def init_states(
        self, batch_size: int, device: torch.device
    ) -> List[Tuple[torch.Tensor]]:
        """Initialize empty (zero) states for recurrent/streaming inference.

        Returns:
            List of (z,) 1-tuples, z: (B, n_heads, d_head) zeros, one per
            layer.
        """
        return [
            (torch.zeros(batch_size, self.n_heads, self.d_head, device=device),)
            for _ in range(self.n_layers)
        ]
