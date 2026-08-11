"""
THE SPEAKER -- says what happened, from vision alone, taught only by its
own body's acted records.

Input per step t: the frame f_t and the seen change (f_{t+1} - f_t). NOT
the action -- deliberately. If the pressed action were an input, the
action-word head would be an echo chamber; from vision alone it must be a
REPORT of what visibly happened, which makes two claims measurable:

  - on consequence steps the world's action<->direction bijection makes
    the action word recoverable from sight (P91a expects it learned);
  - on silent steps (hold, border-clamped pans) the press is invisible in
    principle, so accuracy there has a ceiling at the marginal prior --
    grounded language reaches exactly as far as visible evidence (P91d).

Architecture: two linear encoders (frame, delta) summed -> GSSMCore d128
(imported, never copied -- same core as body and vision) -> three heads
(direction 3, magnitude 4, action 3). Supervision comes ONLY from the
organism's own acted-event records (SILENCE elsewhere) -- no corpus, no
external labels; the language is born from the body's own life.

Training is a single streaming pass (life is a stream, no epochs -- house
convention), ungated: the gate is the BODY's economy; whether the speaker
needs one is a later question and is not smuggled in here.

Compositional holdout: steps whose true sentence equals HOLDOUT_SENTENCE
(right, 2) are masked out of EVERY head's loss (forward still runs, state
still carries) and evaluated separately -- systematicity as a measurement.
"""

import os
import sys
from typing import Dict, List, Optional, Tuple

import numpy as np

VOICE_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(VOICE_DIR)
for p in (REPO_ROOT, os.path.join(REPO_ROOT, "visual"),
          os.path.join(REPO_ROOT, "body")):
    if p not in sys.path:
        sys.path.insert(0, p)

import torch
import torch.nn as nn
import torch.nn.functional as F

from hsslm.neural.gssm_core import GSSMCore          # imported, not copied
from frame_organism import FRAME_DIM, detach_state_tree  # read-only imports

from vocab import N_DIR, N_MAG, N_ACT

HOLDOUT_SENTENCE = (2, 2)   # (dir=right, mag=2) -- masked from training loss


class SpeakerNet(nn.Module):
    def __init__(self, d_model: int = 128, n_layers: int = 2, n_heads: int = 4,
                 frame_dim: int = FRAME_DIM, delta_scale: float = 4.0):
        super().__init__()
        self.d_model = d_model
        # delta_scale: |f_{t+1}-f_t| lives an order of magnitude below |f_t|
        # (a mag-1 shift moves a few texture columns; the frame is full
        # range). One fixed gain balances the two encoders' input ranges --
        # a constant, not a learned or tuned-per-run knob.
        self.delta_scale = delta_scale
        self.enc_frame = nn.Linear(frame_dim, d_model)
        self.enc_delta = nn.Linear(frame_dim, d_model)
        self.core = GSSMCore(n_layers=n_layers, d_model=d_model,
                             n_heads=n_heads, check_bounds=True)
        self.head_dir = nn.Linear(d_model, N_DIR)
        self.head_mag = nn.Linear(d_model, N_MAG)
        self.head_act = nn.Linear(d_model, N_ACT)

    def forward(self, frames: torch.Tensor, deltas: torch.Tensor,
                states: Optional[List] = None):
        """frames, deltas: (B, L, frame_dim). Returns ((logits_dir,
        logits_mag, logits_act), new_states)."""
        h = self.enc_frame(frames) + self.enc_delta(deltas * self.delta_scale)
        h, st = self.core(h, states)
        return (self.head_dir(h), self.head_mag(h), self.head_act(h)), st


def train_speaker(frames: np.ndarray, labels: Tuple[np.ndarray, ...],
                  train_upto: int, chunk: int = 24, lr: float = 3e-4,
                  passes: int = 6, seed: int = 0,
                  log_every: int = 50) -> Tuple[SpeakerNet, list]:
    """Sleep-replay training over the life's first `train_upto` steps:
    `passes` re-streams of the same recorded life, states reset between
    passes. This is deliberately NOT single-pass streaming -- the speaker
    learns from its RECORDED life the way the organism's sleep already
    replays its knowledge file (P47 precedent); the debug smoke measured
    that one pass over a short life leaves the heads at chance, and
    claiming 'language is born' from an untrained mouth would be theater.

    frames: (n_steps+1, H, W) -- the lived frame sequence (f_0..f_n).
    labels: (dir, mag, act) int64 arrays of length n_steps, aligned so
    labels[t] describes the transition f_t -> f_{t+1}.
    Steps with sentence == HOLDOUT_SENTENCE are loss-masked (see module
    docstring). Returns (model, loss_trace)."""
    torch.manual_seed(seed)
    model = SpeakerNet()
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    d_lab, m_lab, a_lab = labels
    n = train_upto
    flat = frames.reshape(frames.shape[0], -1).astype(np.float32)
    trace = []
    starts = [(p, c0) for p in range(passes) for c0 in range(0, n - chunk, chunk)]
    states = None
    last_pass = 0
    for p, c0 in starts:
        if p != last_pass:
            states, last_pass = None, p   # a new pass starts a fresh stream
        sl = slice(c0, c0 + chunk)
        x = torch.from_numpy(flat[c0:c0 + chunk])[None]                    # (1,K,D)
        dl = torch.from_numpy(flat[c0 + 1:c0 + chunk + 1] - flat[c0:c0 + chunk])[None]
        (ld, lm, la), st = model(x, dl, states)
        td = torch.from_numpy(d_lab[sl])[None]
        tm = torch.from_numpy(m_lab[sl])[None]
        ta = torch.from_numpy(a_lab[sl])[None]
        mask = ~((td == HOLDOUT_SENTENCE[0]) & (tm == HOLDOUT_SENTENCE[1]))
        mask_f = mask.float()
        denom = mask_f.sum().clamp(min=1.0)

        def _ce(logits, target):
            ce = F.cross_entropy(logits.reshape(-1, logits.shape[-1]),
                                 target.reshape(-1), reduction="none")
            return (ce * mask_f.reshape(-1)).sum() / denom

        loss = _ce(ld, td) + _ce(lm, tm) + _ce(la, ta)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        states = detach_state_tree(st)
        trace.append(float(loss.detach()))
        if log_every and (len(trace) % log_every == 0):
            print(f"[speaker] chunk {len(trace)} loss {float(loss):.4f}", flush=True)
    return model, trace


@torch.no_grad()
def utter_life(model: SpeakerNet, frames: np.ndarray, chunk: int = 32):
    """The speaker narrates a whole life: argmax (dir, mag, act) per step,
    states carried across chunks from zero. Returns three int64 arrays of
    length n_steps = len(frames) - 1."""
    flat = frames.reshape(frames.shape[0], -1).astype(np.float32)
    n = flat.shape[0] - 1
    out_d, out_m, out_a = [], [], []
    states = None
    for c0 in range(0, n, chunk):
        c1 = min(c0 + chunk, n)
        x = torch.from_numpy(flat[c0:c1])[None]
        dl = torch.from_numpy(flat[c0 + 1:c1 + 1] - flat[c0:c1])[None]
        (ld, lm, la), states = model(x, dl, states)
        out_d.append(ld.argmax(-1)[0].numpy())
        out_m.append(lm.argmax(-1)[0].numpy())
        out_a.append(la.argmax(-1)[0].numpy())
    return (np.concatenate(out_d), np.concatenate(out_m), np.concatenate(out_a))


def eval_speaker(pred, labels, heldout_from: int) -> Dict:
    """Accuracy instruments on the held-out tail (steps >= heldout_from):
    dir/mag/act accuracy on consequence steps, act accuracy on silent
    steps vs the marginal prior (P91d), and sentence accuracy on the
    compositional-holdout steps WHEREVER they occur (they were never in
    any loss). Nothing here is a bar -- the bars live in the register."""
    pd, pm, pa = pred
    td, tm, ta = labels
    ho = slice(heldout_from, len(td))
    cons = td[ho] != 0
    sil = ~cons
    res = {}
    res["n_heldout"] = int(len(td) - heldout_from)
    res["n_consequence"] = int(cons.sum())
    if cons.any():
        res["dir_acc"] = float((pd[ho][cons] == td[ho][cons]).mean())
        res["mag_acc"] = float((pm[ho][cons] == tm[ho][cons]).mean())
        res["act_acc_consequence"] = float((pa[ho][cons] == ta[ho][cons]).mean())
        res["sentence_acc"] = float(((pd[ho][cons] == td[ho][cons])
                                     & (pm[ho][cons] == tm[ho][cons])).mean())
    if sil.any():
        res["act_acc_silent"] = float((pa[ho][sil] == ta[ho][sil]).mean())
        vals, counts = np.unique(ta[ho][sil], return_counts=True)
        res["act_prior_silent"] = float(counts.max() / counts.sum())
        res["silence_said"] = float((pd[ho][sil] == 0).mean())
    comp = (td == HOLDOUT_SENTENCE[0]) & (tm == HOLDOUT_SENTENCE[1])
    res["n_comp_holdout"] = int(comp.sum())
    if comp.any():
        res["comp_sentence_acc"] = float(((pd[comp] == td[comp])
                                          & (pm[comp] == tm[comp])).mean())
    return res


@torch.no_grad()
def intervention_flip(model: SpeakerNet, frames: np.ndarray, n_probes: int = 60,
                      mag: int = 2, seed: int = 3) -> Dict:
    """intervention_necessity in miniature: same state f_t, FORCED
    consequence left vs right (delta synthesized by the same edge-clamp
    shift imagination uses -- the world op away from borders), the
    direction word must flip. Single-step from zero state (the probe asks
    the perception, not the carry -- noted, not hidden)."""
    from imagination import shift_frame
    rng = np.random.default_rng(seed)
    idx = rng.choice(frames.shape[0] - 1, size=min(n_probes, frames.shape[0] - 1),
                     replace=False)
    flips = 0
    for t in idx:
        f = frames[t].astype(np.float32)
        said = []
        for dx in (-mag, +mag):
            delta = (shift_frame(f, dx) - f).reshape(1, 1, -1)
            x = torch.from_numpy(f.reshape(1, 1, -1))
            (ld, _, _), _ = model(x, torch.from_numpy(delta), None)
            said.append(int(ld.argmax(-1)[0, -1]))
        flips += int(said[0] == 1 and said[1] == 2)   # left said, then right said
    return {"n_probes": int(len(idx)), "flip_rate": float(flips / len(idx))}
