"""
BODY ORGANISM -- action-conditioned frame prediction + curiosity as policy.

Closes the gap the visual track left open by construction: frame_organism's
FramePredictor answers "what will I see?", this module's
ActionConditionedPredictor answers "what will I see IF I DO THIS?" -- the
action enters the encoder as a second input, so the model's forward pass IS
a counterfactual machine: same state, different action, different predicted
next frame. That is the half of the Harnad loop the architecture inventory
(2026-08-11) named as completely missing: symbols/percepts binding to the
consequences of OWN actions.

Inheritance discipline (same as frame_organism.py's own relationship to
hsslm/neural/streaming.py): GSSMCore is IMPORTED (never copied) from
hsslm/neural/gssm_core.py; the gate constants and detach_state_tree are
IMPORTED read-only from visual/frame_organism.py; the gate arithmetic is
re-implemented here as its own small class because the step signature
differs (x, actions, y -- three tensors, not two). No file outside body/ is
modified.

Three parts:

  ActionConditionedPredictor -- frame encoder + action embedding -> GSSMCore
      -> zero-init delta head. Two measured lessons inherited:
      (1) residual delta output (frame_organism's Phase-1 finding: direct
          sigmoid collapsed to the marginal mean; zero-init delta starts AT
          the copy-last baseline and only has to learn the CHANGE);
      (2) the action embedding is zero-init too, so at step 0 the model is
          exactly a FramePredictor that ignores its action input -- the
          action channel has to EARN its influence through gradient, which
          makes "did the action arrive in the model?" a falsifiable
          question (counterfactual separation / hit-rate, see run_body.py)
          instead of an architectural assumption.

  BodyStreamer -- the F1/F2 recipe (rolling-quantile surprise gate, detach-
      carry, one grad step per gated chunk) at the new signature. Constants
      come from frame_organism so a recipe drift there is a loud import
      error here, not a silent divergence.

  LearningProgressPolicy -- curiosity as the motive: per-action learning
      progress (how fast is my prediction of THIS action's consequences
      improving?), Oudeyer-style, computed from the SAME per-frame L1 the
      gate already uses. The surprise machinery stops being only a filter
      (what to learn from) and becomes the motive (what to do). NOT
      prediction-error-seeking (that policy runs into the noisy-TV problem:
      it would stare at unlearnable noise); learning PROGRESS fades on both
      the already-mastered and the unlearnable, which is the correct
      developmental shape. Honest scope note: this is a context-free bandit
      over actions (day-1 body). State-conditional progress estimates are
      the registered next step, not smuggled in.
"""

import os
import sys
import json
from collections import deque
from typing import Dict, List, Optional, Tuple

import numpy as np

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for p in (REPO_ROOT, os.path.join(REPO_ROOT, "visual")):
    if p not in sys.path:
        sys.path.insert(0, p)

import torch
import torch.nn as nn

from hsslm.neural.gssm_core import GSSMCore                      # imported, not copied
from frame_organism import (                                     # read-only library use
    GATE_Q, GATE_WINDOW, MIN_WINDOW, IGNITION_CHUNKS, CHUNK_FRAMES,
    FRAME_SIZE, FRAME_DIM, detach_state_tree, write_status, append_metric,
)


class ActionConditionedPredictor(nn.Module):
    """(frame_t, action_t) -> predicted frame_{t+1}.

    h_t = encoder(frame_t) + act_embed(action_t)  ->  GSSMCore  ->
    delta = tanh(decoder(h))  ->  pred = clamp(frame_t + delta, 0, 1).

    Zero-init on BOTH the delta head and the action embedding (rationale in
    the module docstring). d_model/n_layers/n_heads defaults match
    frame_organism's FramePredictor so results are comparable across the
    spectator/actor pair at equal capacity.
    """

    def __init__(self, n_actions: int, d_model: int = 256, n_layers: int = 4,
                 n_heads: int = 4, frame_dim: int = FRAME_DIM):
        super().__init__()
        self.d_model = d_model
        self.frame_dim = frame_dim
        self.n_actions = n_actions
        self.encoder = nn.Linear(frame_dim, d_model)
        self.act_embed = nn.Embedding(n_actions, d_model)
        nn.init.zeros_(self.act_embed.weight)
        self.core = GSSMCore(n_layers=n_layers, d_model=d_model,
                             n_heads=n_heads, check_bounds=True)
        self.decoder = nn.Linear(d_model, frame_dim)
        nn.init.zeros_(self.decoder.weight)
        nn.init.zeros_(self.decoder.bias)

    def forward(self, frames: torch.Tensor, actions: torch.Tensor,
                states: Optional[List] = None) -> Tuple[torch.Tensor, List]:
        """frames: (B, L, frame_dim) in [0,1]; actions: (B, L) int64 --
        actions[:, t] is the action TAKEN at frame t, the prediction at t is
        for frame t+1 under that action. Returns (pred, new_states), same
        carry contract as FramePredictor/GSSMCore."""
        h = self.encoder(frames) + self.act_embed(actions)
        h, new_states = self.core(h, states)
        delta = torch.tanh(self.decoder(h))
        pred = torch.clamp(frames + delta, 0.0, 1.0)
        return pred, new_states

    def init_states(self, batch_size: int, device: torch.device) -> List:
        return self.core.init_states(batch_size, device)

    @torch.no_grad()
    def counterfactual(self, frame: torch.Tensor,
                       states: Optional[List]) -> torch.Tensor:
        """From ONE state, the predicted next frame under EVERY action:
        frame (1, 1, frame_dim), carried states -> (n_actions, frame_dim).

        Runs each action as its own 1-step forward from the SAME (cloned)
        state -- the passed `states` object is never advanced or mutated
        (detach_state_tree on a fresh clone per action), so probing
        counterfactuals mid-stream cannot corrupt the live carry. This is
        the interrogation "was sehe ich, WENN ich das tue" as an actual
        forward pass, and the basis of the action-separation / hit-rate
        instruments in run_body.py."""
        outs = []
        for a in range(self.n_actions):
            st = None if states is None else detach_state_tree(
                [tuple(t.clone() for t in layer) for layer in states])
            act = torch.full((1, 1), a, dtype=torch.long, device=frame.device)
            pred, _ = self.forward(frame, act, st)
            outs.append(pred[0, 0])
        return torch.stack(outs, dim=0)


class BodyStreamer:
    """Surprise-gated chunk streaming for the acting body -- the F1/F2
    recipe at the (x, actions, y) signature. Constants imported from
    frame_organism (GATE_Q/GATE_WINDOW/MIN_WINDOW/IGNITION_CHUNKS), the
    arithmetic re-stated because the forward takes actions; the per-chunk
    flow is line-for-line the same shape as FrameStreamer.step_gated."""

    def __init__(self, model: ActionConditionedPredictor,
                 optimizer: torch.optim.Optimizer, device: torch.device,
                 gate_q: float = GATE_Q, gate_window: int = GATE_WINDOW,
                 min_window: int = MIN_WINDOW,
                 ignition_chunks: int = IGNITION_CHUNKS,
                 grad_clip: float = 1.0, gate_enabled: bool = True):
        self.model = model
        self.opt = optimizer
        self.device = device
        self.gate_q = gate_q
        self.min_window = min_window
        self.ignition_chunks = ignition_chunks
        self.grad_clip = grad_clip
        self.gate_enabled = gate_enabled
        self.window = deque(maxlen=gate_window)
        self.states: Optional[List] = None
        self.n_chunks = 0
        self.n_bwd = 0

    def step_gated(self, x: torch.Tensor, actions: torch.Tensor,
                   y: torch.Tensor) -> Tuple[float, bool, torch.Tensor]:
        """x, y: (B, L, frame_dim); actions: (B, L) int64.
        Returns (surprise, gated, per_frame_l1 (B, L))."""
        with torch.no_grad():
            pred_ng, st_ng = self.model(x, actions, self.states)
            per_frame_l1_ng = (pred_ng - y).abs().mean(dim=-1)
        surprise = float(per_frame_l1_ng.mean())

        if not self.gate_enabled:
            gated = True
        elif self.n_chunks < self.ignition_chunks:
            gated = True
        elif len(self.window) >= self.min_window:
            thresh = float(np.quantile(
                np.fromiter(self.window, dtype=np.float64), self.gate_q))
            gated = surprise > thresh
        else:
            gated = True

        if gated:
            pred, st = self.model(x, actions, self.states)
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
        return {"n_chunks": self.n_chunks, "n_bwd": self.n_bwd,
                "gate_rate": self.n_bwd / max(1, self.n_chunks),
                "window_len": len(self.window)}


class LearningProgressPolicy:
    """Choose the action whose consequences the model is currently LEARNING
    FASTEST to predict.

    Per action a: a deque of the last `window` per-frame L1 errors observed
    on frames where a was taken (fed by the run loop from the streamer's
    per_frame_l1 -- the same scalar the gate uses; nothing new is measured).
    Learning progress LP(a) = mean(older half) - mean(newer half): positive
    while predictions of a's consequences are improving, ~0 both for the
    mastered and for the unlearnable. Policy = softmax over z-scored LP
    with temperature `tau`, mixed with an `eps` uniform floor (the
    estimators must keep receiving samples for every action, or a dead
    action could never announce that it became interesting again). During
    the first `ignition_steps` acted frames: uniform random -- LP over
    ignition noise is not signal, same reason the gate has an ignition
    phase. Actions with fewer than `min_samples` observations get the
    current MAX score (optimistic init: untried == maximally interesting).

    Everything is observable: probs()/lp_estimates() are logged per chunk
    by run_body.py, so "did curiosity develop?" is a curve in the metrics
    file, not an anecdote. Seeded RNG; choices are reproducible given the
    error stream -- but note the error stream depends on the weights, so
    run-level provenance rests on the recorded action trace
    (action_sources.py docstring), never on re-running the policy.
    """

    def __init__(self, n_actions: int, window: int = 60, eps: float = 0.10,
                 tau: float = 1.0, min_samples: int = 8,
                 ignition_steps: int = 256, seed: int = 0):
        self.n_actions = n_actions
        self.errs = [deque(maxlen=window) for _ in range(n_actions)]
        self.counts = [0] * n_actions
        self.eps = eps
        self.tau = tau
        self.min_samples = min_samples
        self.ignition_steps = ignition_steps
        self.rng = np.random.default_rng(seed)
        self.steps = 0

    def update(self, action: int, error: float) -> None:
        self.errs[action].append(float(error))
        self.counts[action] += 1

    def lp_estimates(self) -> List[Optional[float]]:
        out = []
        for a in range(self.n_actions):
            e = list(self.errs[a])
            if len(e) < self.min_samples:
                out.append(None)
                continue
            half = len(e) // 2
            out.append(float(np.mean(e[:half]) - np.mean(e[half:])))
        return out

    def probs(self) -> np.ndarray:
        lp = self.lp_estimates()
        known = [v for v in lp if v is not None]
        if not known:
            return np.full(self.n_actions, 1.0 / self.n_actions)
        optimistic = max(known)
        scores = np.array([optimistic if v is None else v for v in lp],
                          dtype=np.float64)
        sd = scores.std()
        z = (scores - scores.mean()) / sd if sd > 1e-12 else np.zeros_like(scores)
        ex = np.exp(z / max(self.tau, 1e-6))
        soft = ex / ex.sum()
        return (1 - self.eps) * soft + self.eps / self.n_actions

    def choose(self) -> int:
        self.steps += 1
        if self.steps <= self.ignition_steps:
            return int(self.rng.integers(0, self.n_actions))
        return int(self.rng.choice(self.n_actions, p=self.probs()))


class RandomPolicy:
    """The P52-shaped control arm: seeded uniform choice, no motive. Same
    interface as LearningProgressPolicy so run_body.py swaps them with one
    flag and the comparison is policy-only by construction."""

    def __init__(self, n_actions: int, seed: int = 0, **_ignored):
        self.n_actions = n_actions
        self.rng = np.random.default_rng(seed)
        self.steps = 0

    def update(self, action: int, error: float) -> None:
        pass

    def lp_estimates(self):
        return [None] * self.n_actions

    def probs(self) -> np.ndarray:
        return np.full(self.n_actions, 1.0 / self.n_actions)

    def choose(self) -> int:
        self.steps += 1
        return int(self.rng.integers(0, self.n_actions))


# ═══════════════════════════════════════════════════════════════════════════
#  Snapshot I/O -- frame_organism's save_snapshot shape, restated because
#  the config block differs (n_actions, policy state travels along).
# ═══════════════════════════════════════════════════════════════════════════
def save_body_snapshot(path: str, model: ActionConditionedPredictor,
                       opt: torch.optim.Optimizer, streamer: BodyStreamer,
                       frame_idx: int, source_seed: int, wall_s: float,
                       extra: Optional[dict] = None) -> None:
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
        "wall_s": wall_s,
        "config": {"d_model": model.d_model, "frame_dim": model.frame_dim,
                   "n_actions": model.n_actions, "chunk_frames": CHUNK_FRAMES,
                   "gate_q": streamer.gate_q},
        "extra": extra or {},
    }
    d = os.path.dirname(path) or "."
    os.makedirs(d, exist_ok=True)
    tmp = f"{path}.tmp{os.getpid()}.pt"
    torch.save(ck, tmp)
    os.replace(tmp, path)
