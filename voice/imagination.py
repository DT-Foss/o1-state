"""
IMAGINATION -- learning from hearsay, honestly labeled as imagination.

A fresh body B never lived A's life. What it gets is UTTERANCES -- three
words per event. This module turns a heard sentence into pseudo-experience:
take one of B's OWN currently-seen frames, apply the DESCRIBED consequence
(horizontal shift by the said direction and magnitude), and hand the pair
(frame, described_action) -> imagined_next_frame to B's body model as
training material. That is what instruction does for humans: you imagine
what you were told, and the imagining trains you before reality does.

Honesty notes, load-bearing:
  - shift_frame uses EDGE REPLICATION for the columns the shift exposes.
    The real world reveals NEW content there (unknowable to B) -- so the
    imagined pair is a correct lesson about the action->shift mapping and
    a slightly wrong lesson about frame content at one edge. This gap is
    named here and in the register; if imagination pretraining were to
    HURT probe L1 while helping the counterfactual instruments, this edge
    lie is the first suspect.
  - The pseudo-pairs are chunked as if sequential, but they are not a real
    trajectory; the carry across an imagined chunk is fictional. Precedent:
    P47's sleep-replay of out-of-context spans (half its benefit was
    content, half regularization -- the scrambled arm exists to make the
    same decomposition possible here).
  - Imagination consumes ZERO real interaction budget: every arm gets the
    same number of REAL acted frames; listening happens offline. That is
    the point being tested -- culture is cheap, experience is expensive.
"""

import os
import sys
from typing import List, Tuple

import numpy as np

VOICE_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(VOICE_DIR)
for p in (REPO_ROOT, os.path.join(REPO_ROOT, "visual"),
          os.path.join(REPO_ROOT, "body")):
    if p not in sys.path:
        sys.path.insert(0, p)

import torch

from vocab import dx_from_sentence, action_from_direction


def shift_frame(f: np.ndarray, dx: int) -> np.ndarray:
    """Apply the described consequence: content moves so that
    out[:, x] = f[:, x + dx] (the body-track sign convention), edge
    columns replicated where the shift exposes unseen space. dx == 0
    returns a copy."""
    out = np.empty_like(f)
    if dx == 0:
        out[:] = f
    elif dx > 0:
        out[:, :-dx] = f[:, dx:]
        out[:, -dx:] = f[:, [-1]]
    else:
        out[:, -dx:] = f[:, :dx]
        out[:, :-dx] = f[:, [0]]
    return out


def utterances_to_pairs(utterances: List[Tuple[int, int]],
                        own_frames: np.ndarray, cap: int,
                        seed: int = 0):
    """Heard sentences -> imagined training pairs.

    utterances: list of (dir_idx, mag_idx) IN LIFE ORDER (silences allowed,
    they are skipped -- you cannot imagine 'nothing happened' into a
    lesson about consequences). own_frames: (M, H, W) frames B collected
    by sitting still and watching its own world. cap: hard ceiling on
    pairs, IDENTICAL across arms so no arm buys extra gradient.

    Returns (x, a, y): float32 (N, D), int64 (N,), float32 (N, D)."""
    rng = np.random.default_rng(seed)
    xs, acts, ys = [], [], []
    order = rng.permutation(len(own_frames))
    k = 0
    for (d, m) in utterances:
        dx = dx_from_sentence(d, m)
        if dx == 0:
            continue
        f = own_frames[order[k % len(own_frames)]].astype(np.float32)
        k += 1
        xs.append(f.reshape(-1))
        acts.append(action_from_direction(d))
        ys.append(shift_frame(f, dx).reshape(-1))
        if len(xs) >= cap:
            break
    if not xs:
        return (np.zeros((0, own_frames.shape[1] * own_frames.shape[2]), np.float32),
                np.zeros((0,), np.int64), np.zeros((0, 0), np.float32))
    return (np.stack(xs).astype(np.float32), np.array(acts, dtype=np.int64),
            np.stack(ys).astype(np.float32))


def imagination_pretrain(model, opt, pairs, chunk: int = 16,
                         grad_clip: float = 1.0) -> int:
    """Ungated pretraining pass over imagined pairs, chunked (fictional
    carry -- see module docstring). Returns number of chunks trained.
    Uses the model's own forward contract (frames, actions, states) so the
    SAME body class (body/body_organism.ActionConditionedPredictor) learns
    from imagination exactly as it would from life."""
    from frame_organism import detach_state_tree  # read-only import
    x, a, y = pairs
    n = x.shape[0]
    n_chunks = 0
    states = None
    for c0 in range(0, n - 1, chunk):
        c1 = min(c0 + chunk, n)
        xb = torch.from_numpy(x[c0:c1])[None]
        ab = torch.from_numpy(a[c0:c1])[None]
        yb = torch.from_numpy(y[c0:c1])[None]
        pred, st = model(xb, ab, states)
        loss = (pred - yb).abs().mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        opt.step()
        states = detach_state_tree(st)
        n_chunks += 1
    return n_chunks


def scramble_utterances(utterances, seed: int = 0):
    """The poison arm's systematic misinformation: direction words swapped
    (left<->right), magnitudes permuted by a fixed derangement-ish map
    (1->3, 2->1, 3->2). Silences stay silences. Deterministic -- the lie
    is consistent, which is the harder poison (random noise would average
    out; a consistent wrong theory should actively mis-train the action
    channel)."""
    out = []
    mag_map = {0: 0, 1: 3, 2: 1, 3: 2}
    dir_map = {0: 0, 1: 2, 2: 1}
    for (d, m) in utterances:
        out.append((dir_map[d], mag_map[m]))
    return out
