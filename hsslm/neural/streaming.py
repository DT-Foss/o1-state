"""
Streaming/gated-training adapter for HSSLM, in the style of
src/portable_organism.py's Organism.step_gated (NOT importing or modifying
that file -- this is a parallel adapter for HSSLM's own state shape).

Ports the R2-style rolling-quantile surprise gate:
  - Forward the chunk WITHOUT gradients to measure surprise (mean NLL).
  - During an "ignition" warmup window, always train (gated=True).
  - After warmup, once the rolling window has enough samples, gate on
    whether this chunk's surprise exceeds the GATE_Q quantile of the
    recent window -- i.e. only backprop on chunks that are more surprising
    than most recent chunks (spike-triggered learning).
  - On a gated step: real forward+backward+opt.step(), states detached and
    carried forward (SSM ssm_state + conv_buffer per layer, from the
    streaming fix in core_engine.py, PLUS discourse_state if hierarchical).
  - On a non-gated step: states still carried forward (from the no-grad
    forward), just no backward/opt.step().

HSSLM-specific differences from portable_organism.Organism:
  - HSSLM's states are a List[Tuple[ssm_state, conv_buffer]] per layer
    (see core_engine.py), not a flat list of tensors -- detach_state_tree
    below handles the nesting.
  - position_offset must advance by chunk_size each step (the position-
    embedding streaming fix in embedding.py) -- portable_organism's
    StreamingNoPELM has no positional embedding at all (NoPE), so this
    concern doesn't exist there; it's new here.
  - discourse_state (hierarchical mode only) is carried and detached
    alongside states, per the hierarchy.py fix.
"""

import math
from collections import deque
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

# Same defaults as src/portable_organism.py's module-level constants,
# duplicated here (not imported) since these are streaming-adapter policy,
# not shared architecture code.
GATE_Q = 0.75
GATE_WINDOW = 200
MIN_WINDOW = 30
IGNITION_CHUNKS = 15


def detach_state_tree(states):
    """Detach an arbitrarily nested list/tuple-of-tensors state structure.

    HSSLM's per-layer state is (ssm_state, conv_buffer) tuples inside a
    list, so a flat [s.detach() for s in states] (portable_organism's
    pattern) is not enough -- this recurses.
    """
    if states is None:
        return None
    if isinstance(states, torch.Tensor):
        return states.detach()
    if isinstance(states, (list, tuple)):
        cls = type(states)
        return cls(detach_state_tree(s) for s in states)
    return states


class HSSLMStreamer:
    """Gated streaming trainer for an HSSLM model instance.

    Mirrors Organism.step_gated's contract (surprise, gated, nll) but
    carries HSSLM's richer state (SSM states + conv buffers + position
    offset + optional discourse state) across chunks.
    """

    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        vocab_size: int,
        pad_token_id: int = 0,
        gate_q: float = GATE_Q,
        gate_window: int = GATE_WINDOW,
        min_window: int = MIN_WINDOW,
        ignition_chunks: int = IGNITION_CHUNKS,
        grad_clip: float = 1.0,
    ) -> None:
        self.model = model
        self.opt = optimizer
        self.vocab_size = vocab_size
        self.pad_token_id = pad_token_id
        self.gate_q = gate_q
        self.min_window = min_window
        self.ignition_chunks = ignition_chunks
        self.grad_clip = grad_clip

        self.window = deque(maxlen=gate_window)
        self.states: Optional[List] = None
        self.discourse_state: Optional[torch.Tensor] = None
        self.position: int = 0  # running absolute token count
        self.n_chunks: int = 0
        self.n_bwd: int = 0
        self.grad_tokens: int = 0

    def step_gated(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        boundaries: Optional[Dict[str, List[torch.Tensor]]] = None,
        ignition: bool = False,
    ) -> Tuple[float, bool, torch.Tensor]:
        """One gated training chunk.

        Args:
            x: (B, L) input token IDs for this chunk.
            y: (B, L) target token IDs (next-token labels) for this chunk.
            boundaries: Optional word/sentence boundaries, LOCAL to this
                chunk (hierarchical mode only; caller re-derives per chunk,
                see hierarchy.py's chunking note).
            ignition: Force a gated (trained) step regardless of the window.

        Returns:
            (surprise, gated, nll) -- same shape as Organism.step_gated:
                surprise: float, mean NLL of this chunk (ungated forward).
                gated: bool, whether a real backward+opt.step() happened.
                nll: (B, L) per-token NLL tensor from the ungated forward.
        """
        B, L = x.shape

        # 1. Ungated forward: measure surprise, no grad.
        with torch.no_grad():
            out_ng = self.model(
                x,
                boundaries=boundaries,
                states=self.states,
                position_offset=self.position,
                discourse_state=self.discourse_state,
            )
            logits_ng = out_ng["logits"]
            nll = F.cross_entropy(
                logits_ng.reshape(-1, self.vocab_size),
                y.reshape(-1),
                ignore_index=self.pad_token_id,
                reduction="none",
            ).view(B, L)
        surprise = float(nll.mean())

        # 2. Gate decision (R2-style rolling quantile, portable_organism's
        # exact recipe).
        if ignition or self.n_chunks < self.ignition_chunks:
            gated = True
        elif len(self.window) >= self.min_window:
            import numpy as np
            thresh = float(np.quantile(
                np.fromiter(self.window, dtype=np.float64), self.gate_q
            ))
            gated = surprise > thresh
        else:
            gated = True

        # 3. Conditional real step.
        if gated:
            out = self.model(
                x,
                boundaries=boundaries,
                labels=y,
                states=self.states,
                position_offset=self.position,
                discourse_state=self.discourse_state,
            )
            loss = out["loss"]
            self.opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
            self.opt.step()
            self.states = detach_state_tree(out["states"])
            ds = out.get("discourse_state")
            self.discourse_state = ds.detach() if ds is not None else None
            self.grad_tokens += x.numel()
            self.n_bwd += 1
        else:
            self.states = detach_state_tree(out_ng["states"])
            ds = out_ng.get("discourse_state")
            self.discourse_state = ds.detach() if ds is not None else None

        self.window.append(surprise)
        self.n_chunks += 1
        self.position += L
        return surprise, gated, nll

    def reset_stream(self) -> None:
        """Reset streaming state (states, discourse, position) -- e.g. at a
        document boundary. Does NOT reset the gate window (that's a
        surprise-history statistic, not a stream-position statistic)."""
        self.states = None
        self.discourse_state = None
        self.position = 0

    def stats(self) -> Dict[str, float]:
        return {
            "n_chunks": self.n_chunks,
            "n_bwd": self.n_bwd,
            "gate_rate": self.n_bwd / max(1, self.n_chunks),
            "grad_tokens": self.grad_tokens,
            "window_len": len(self.window),
        }
