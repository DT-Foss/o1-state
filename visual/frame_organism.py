"""
FRAME ORGANISM -- Phase 1 of the visual organism: a frame-stream predictor
built on our own architecture (GSSMCore, hsslm/neural/gssm_core.py),
inheriting the three laws the language organism already proved:

  F1  surprise gating       -- rolling-quantile gate, ported from
                                hsslm/neural/streaming.py's HSSLMStreamer
                                (same recipe, re-implemented here as its own
                                small class since the state shape differs:
                                frames are float tensors, not token IDs, so
                                there is no cross_entropy/vocab_size/pad
                                machinery to carry over -- just the gate
                                arithmetic itself).
  F2  exact chunk streaming -- detach-carry between chunks, one gradient
                                step per gated chunk. Proven at this new
                                integration point by test_chunking_f2.py.
  F5  GSSM family            -- GSSMCore imported (not copied) from
                                hsslm/neural/gssm_core.py; this file only
                                adds the frame-specific encoder/decoder
                                shell around it.

Model shape: frame (4096 float, 64x64 flattened) -> Linear encoder ->
d_model=256 -> GSSMCore -> Linear decoder -> 4096 -> Sigmoid -> predicted
next frame (64x64, [0,1]).

Loss: L1 between predicted frame and the true next frame. Surprise = that
same per-chunk mean L1 (a scalar) -- the gate signal, same role NLL played
for the language organism.
"""

import os
import sys
import json
import time
from collections import deque
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from hsslm.neural.gssm_core import GSSMCore

FRAME_SIZE = 64
FRAME_DIM = FRAME_SIZE * FRAME_SIZE

# Gate defaults -- same numbers as HSSLMStreamer / portable_organism.py's
# R2-style rolling-quantile recipe (streaming.py's GATE_Q/GATE_WINDOW/
# MIN_WINDOW/IGNITION_CHUNKS), reused here because this IS that recipe,
# just re-implemented against a frame-shaped surprise signal instead of a
# token-NLL one.
GATE_Q = 0.75
GATE_WINDOW = 200
MIN_WINDOW = 30
IGNITION_CHUNKS = 15
CHUNK_FRAMES = 16


class FramePredictor(nn.Module):
    """Linear encoder -> GSSMCore -> Linear decoder -> Sigmoid.

    forward(x, states) mirrors GSSMCore's own contract (x, states) ->
    (pred, new_states) so the streaming wrapper below can detach-carry
    states exactly like HSSLMStreamer does for the language core -- the
    encoder/decoder are both stateless per-frame ops (no recurrence of
    their own), so ALL sequential memory lives in GSSMCore's z-state.

    residual=True (Lead's design-sharpening after the Phase-1 smoke result):
    the decoder predicts a DELTA, and the output is
    clamp(input_frame + delta, 0, 1) instead of sigmoid(decoder(h)) directly.
    Rationale (measured, not assumed): the copy-last-frame baseline (L1
    0.023) dominates this task's trivial structure -- with direct sigmoid
    output the model has to relearn "reproduce the input" from scratch AND
    it collapsed instead to the dataset's bimodal marginal mean (L1 0.214,
    worse than either trivial baseline). With residual output, doing
    NOTHING (zero delta) already equals copy-last, so training only has to
    capture the actual change (the moving shapes) rather than reconstruct
    static content it already has for free.
    """

    def __init__(self, d_model: int = 256, n_layers: int = 4, n_heads: int = 4,
                 frame_dim: int = FRAME_DIM, residual: bool = False):
        super().__init__()
        self.d_model = d_model
        self.frame_dim = frame_dim
        self.residual = residual
        self.encoder = nn.Linear(frame_dim, d_model)
        self.core = GSSMCore(n_layers=n_layers, d_model=d_model, n_heads=n_heads,
                              check_bounds=True)
        self.decoder = nn.Linear(d_model, frame_dim)
        if residual:
            # Delta head starts near-zero so the model begins AT the
            # copy-last baseline (zero delta) and has to earn any deviation
            # from it, rather than starting at a random offset.
            nn.init.zeros_(self.decoder.weight)
            nn.init.zeros_(self.decoder.bias)

    def forward(
        self, frames: torch.Tensor, states: Optional[List] = None,
    ) -> Tuple[torch.Tensor, List]:
        """
        Args:
            frames: (B, L, frame_dim) flattened frames in [0, 1].
            states: optional GSSMCore per-layer state list (carried).

        Returns:
            pred: (B, L, frame_dim) predicted NEXT frame at each position
                  (pred[:, t] predicts frames[:, t+1] -- caller supplies the
                  shifted target).
            new_states: GSSMCore states after this chunk (detach yourself
                        for streaming carry).
        """
        h = self.encoder(frames)
        h, new_states = self.core(h, states)
        if self.residual:
            delta = torch.tanh(self.decoder(h))  # bounded delta, in (-1, 1)
            pred = torch.clamp(frames + delta, 0.0, 1.0)
        else:
            pred = torch.sigmoid(self.decoder(h))
        return pred, new_states

    def init_states(self, batch_size: int, device: torch.device) -> List:
        return self.core.init_states(batch_size, device)


def detach_state_tree(states):
    """Same recursive detach as hsslm/neural/streaming.py's
    detach_state_tree -- duplicated here (not imported) because this file
    has no other dependency on streaming.py and the function is three
    lines; GSSMCore's states are List[(z,)] which this handles identically."""
    if states is None:
        return None
    if isinstance(states, torch.Tensor):
        return states.detach()
    if isinstance(states, (list, tuple)):
        cls = type(states)
        return cls(detach_state_tree(s) for s in states)
    return states


class FrameStreamer:
    """Gated chunk-streaming trainer for a FramePredictor, mirroring
    hsslm/neural/streaming.py's HSSLMStreamer contract (surprise, gated) but
    over frame chunks instead of token chunks:

      1. Ungated forward over the chunk (no grad) -> per-chunk mean L1
         ("surprise").
      2. Ignition warmup (always gated for the first `ignition_chunks`
         chunks) then rolling-quantile gate: train only if this chunk's
         surprise exceeds the GATE_Q quantile of the recent window.
      3. On a gated chunk: real forward+backward+opt.step(), states
         detached and carried to the next chunk.
      4. On a non-gated chunk: states still carried forward (from the
         no-grad forward), no backward/opt.step().

    One gradient step per gated chunk (not per frame) -- matches
    portable_organism.py's Organism.step_gated cadence.
    """

    def __init__(
        self,
        model: FramePredictor,
        optimizer: torch.optim.Optimizer,
        device: torch.device,
        gate_q: float = GATE_Q,
        gate_window: int = GATE_WINDOW,
        min_window: int = MIN_WINDOW,
        ignition_chunks: int = IGNITION_CHUNKS,
        grad_clip: float = 1.0,
        gate_enabled: bool = True,
    ):
        self.model = model
        self.opt = optimizer
        self.device = device
        self.gate_q = gate_q
        self.min_window = min_window
        self.ignition_chunks = ignition_chunks
        self.grad_clip = grad_clip
        # gate_enabled=False -- ablation arm (Lead's arm B): every chunk is
        # trained, isolating whether the gate itself (not just total frame
        # budget) is limiting how fast the model moves off the collapsed
        # output in a small-budget regime.
        self.gate_enabled = gate_enabled

        self.window = deque(maxlen=gate_window)
        self.states: Optional[List] = None
        self.n_chunks = 0
        self.n_bwd = 0

    def step_gated(
        self, x: torch.Tensor, y: torch.Tensor, ignition: bool = False,
    ) -> Tuple[float, bool, torch.Tensor]:
        """One gated training chunk.

        Args:
            x: (B, L, frame_dim) input frames.
            y: (B, L, frame_dim) target frames (x shifted by one).
            ignition: force a gated step regardless of the window.

        Returns:
            (surprise, gated, per_frame_l1) -- per_frame_l1: (B, L) mean
            L1 per frame (for span/spike inspection, mirrors NLL's role).
        """
        with torch.no_grad():
            pred_ng, st_ng = self.model(x, self.states)
            per_frame_l1_ng = (pred_ng - y).abs().mean(dim=-1)  # (B, L)
        surprise = float(per_frame_l1_ng.mean())

        if not self.gate_enabled:
            gated = True
        elif ignition or self.n_chunks < self.ignition_chunks:
            gated = True
        elif len(self.window) >= self.min_window:
            thresh = float(np.quantile(
                np.fromiter(self.window, dtype=np.float64), self.gate_q
            ))
            gated = surprise > thresh
        else:
            gated = True

        if gated:
            pred, st = self.model(x, self.states)
            loss = (pred - y).abs().mean()
            self.opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
            self.opt.step()
            self.states = detach_state_tree(st)
            self.n_bwd += 1
            with torch.no_grad():
                per_frame_l1 = (pred.detach() - y).abs().mean(dim=-1)
        else:
            self.states = detach_state_tree(st_ng)
            per_frame_l1 = per_frame_l1_ng

        self.window.append(surprise)
        self.n_chunks += 1
        return surprise, gated, per_frame_l1

    def stats(self) -> Dict[str, float]:
        return {
            "n_chunks": self.n_chunks,
            "n_bwd": self.n_bwd,
            "gate_rate": self.n_bwd / max(1, self.n_chunks),
            "window_len": len(self.window),
        }


# ═══════════════════════════════════════════════════════════════════════════
#  Checkpoint I/O -- save_snapshot-style: atomic tmpfile + os.replace, same
#  pattern as src/portable_organism.py's save_snapshot.
# ═══════════════════════════════════════════════════════════════════════════
def save_snapshot(path: str, model: FramePredictor, opt: torch.optim.Optimizer,
                   streamer: FrameStreamer, frame_idx: int, source_seed: int,
                   wall_s: float, extra: Optional[dict] = None) -> None:
    ck = {
        "model": model.state_dict(),
        "opt": opt.state_dict(),
        "states": None if streamer.states is None else detach_state_tree(streamer.states),
        "window": list(streamer.window),
        "n_chunks": streamer.n_chunks,
        "n_bwd": streamer.n_bwd,
        "frame_idx": frame_idx,
        "source_seed": source_seed,
        "torch_rng": torch.get_rng_state(),
        "np_rng_state": np.random.get_state(),
        "wall_s": wall_s,
        "config": {
            "d_model": model.d_model, "frame_dim": model.frame_dim,
            "chunk_frames": CHUNK_FRAMES, "gate_q": streamer.gate_q,
        },
        "extra": extra or {},
    }
    d = os.path.dirname(path) or "."
    os.makedirs(d, exist_ok=True)
    tmp = f"{path}.tmp{os.getpid()}.pt"
    torch.save(ck, tmp)
    os.replace(tmp, path)


def write_status(path: str, status: dict) -> None:
    """pos_run-style status heartbeat: tmpfile + os.replace, so a reader
    never observes a partially-written status.json."""
    d = os.path.dirname(path) or "."
    os.makedirs(d, exist_ok=True)
    tmp = f"{path}.tmp{os.getpid()}"
    with open(tmp, "w") as f:
        json.dump(status, f, indent=2)
    os.replace(tmp, path)


def append_metric(path: str, rec: dict) -> None:
    d = os.path.dirname(path) or "."
    os.makedirs(d, exist_ok=True)
    with open(path, "a") as f:
        f.write(json.dumps(rec) + "\n")
        f.flush()
        os.fsync(f.fileno())
