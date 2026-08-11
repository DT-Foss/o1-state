"""
VOICE VOCABULARY -- a proto-language born from acted causal records, plus
the slightly richer world that gives it something to say.

The body track's acted records ("pressed:pan_left | view_shift | dx-3")
already ARE proto-sentences with ground truth. This module makes that
literal: a closed, compositional vocabulary of three slots

    DIRECTION  in {left, right, none}     -- sign of the seen consequence
    MAGNITUDE  in {0, 1, 2, 3}            -- |dx| of the seen consequence
    ACTION     in {pan_left, pan_right, hold}  -- what the speaker CLAIMS
                                              was done

and a DETERMINISTIC codec between records and sentences. Slots, not one
fused symbol, because composition is the falsifiable part: a speaker that
never saw (right, 2) in training but says it correctly by combining
(right,*) and (*, 2) has systematicity -- registered, not assumed
(PREDICTIONS_VOICE.md P91b).

Semantics convention (inherited from body/causal_records.py's estimator):
dx > 0 means the view moved RIGHT (content matched at f0[:, x+dx]), which
in the acted synthetic world is caused by PAN_RIGHT. So direction-word ==
action-direction is a world BIJECTION on consequence steps -- and exactly
NOT inferable on silent steps (HOLD and border-clamped pans look
identical: nothing moved). That asymmetry is the epistemics claim P91d:
grounded language reaches exactly as far as visible evidence.

RichPanSource: body/'s ActedSyntheticSource, subclassed READ-ONLY (no body
file modified), with one change -- pan magnitude cycles deterministically
1,2,3 by step index, so MAGNITUDE has something to discriminate (the base
world's fixed step 3 would make the mag slot a constant). Magnitude is a
pure function of frame_idx, which is itself a pure function of the trace
length, so the (seed, trace) -> frames provenance contract of the base
class survives untouched (test_voice.py re-verifies byte-identity).
"""

import os
import sys

import numpy as np

VOICE_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(VOICE_DIR)
for p in (REPO_ROOT, os.path.join(REPO_ROOT, "visual"),
          os.path.join(REPO_ROOT, "body"), os.path.join(REPO_ROOT, "src")):
    if p not in sys.path:
        sys.path.insert(0, p)

from action_sources import ActedSyntheticSource  # read-only import from body/

DIRECTIONS = ["none", "left", "right"]           # index 0 reserved for silence
MAGNITUDES = [0, 1, 2, 3]
ACTIONS = ["pan_left", "pan_right", "hold"]      # body world's action names, lowercased
N_DIR, N_MAG, N_ACT = len(DIRECTIONS), len(MAGNITUDES), len(ACTIONS)

SILENCE = (0, 0)  # (dir_idx, mag_idx) of "nothing happened"


def sentence_from_dx(dx: int):
    """(dir_idx, mag_idx) describing a measured consequence dx. dx == 0 ->
    SILENCE. |dx| is clamped into the vocabulary (the world produces at
    most 3; a bigger dx would be a world change and should fail loudly in
    tests, not silently saturate here)."""
    if dx == 0:
        return SILENCE
    assert 1 <= abs(dx) <= 3, f"dx {dx} outside the vocabulary's world"
    return (2 if dx > 0 else 1, abs(dx))


def dx_from_sentence(dir_idx: int, mag_idx: int) -> int:
    """Inverse codec. SILENCE -> 0."""
    if dir_idx == 0 or mag_idx == 0:
        return 0
    return mag_idx if dir_idx == 2 else -mag_idx


def action_from_direction(dir_idx: int) -> int:
    """The world bijection on consequence steps: content moved right =>
    pan_right was pressed. Returns an ACTIONS index; SILENCE -> hold (the
    listener's best guess for 'nothing seen')."""
    return {0: 2, 1: 0, 2: 1}[dir_idx]


def sentence_from_record(rec: dict):
    """Acted-event record (body/causal_records.py schema) -> (dir, mag,
    act) label triple. Pure function of the record's own fields."""
    dx = rec["quote"]["dx_measured"]
    d, m = sentence_from_dx(dx)
    act = {"pan_left": 0, "pan_right": 1, "hold": 2}[rec["quote"]["action_name"]]
    return (d, m, act)


def render_sentence(dir_idx: int, mag_idx: int, act_idx=None) -> str:
    """Human-readable form for logs/GIF captions: 'right 2 (pan_right)'."""
    if (dir_idx, mag_idx) == SILENCE:
        return "—"
    s = f"{DIRECTIONS[dir_idx]} {MAGNITUDES[mag_idx]}"
    return s if act_idx is None else s + f" ({ACTIONS[act_idx]})"


class RichPanSource(ActedSyntheticSource):
    """body/'s acted world with cycling pan magnitude 1,2,3 (see module
    docstring). Everything else -- landmarks, self-moving shapes, action
    set, trace recording, dx_truth bookkeeping, replay contract -- is
    inherited unchanged; replay_frames() works via cls() and therefore
    replays THIS world when called on THIS class."""

    def act(self, a: int) -> None:
        assert 0 <= a < self.N_ACTIONS, f"action {a} out of range"
        step = (self.frame_idx % 3) + 1          # deterministic 1,2,3 cycle
        self.trace.append(int(a))
        old = self.view_x
        if a == 0:
            self.view_x = max(0, self.view_x - step)
        elif a == 1:
            self.view_x = min(self.world_w - self.view, self.view_x + step)
        self.dx_truth.append(self.view_x - old)
        self._step_shapes()
        self.frame_idx += 1


def labels_for_life(n_steps: int, records: list):
    """Per-step sentence labels for a lived life: records where the
    extractor spoke, SILENCE everywhere else. Returns three int64 arrays
    (dir, mag, act) of length n_steps. The action slot on silent steps is
    labeled with the RECORDED world action where the caller supplies it --
    here we DON'T have it from records (no record fired), so silent steps
    get act = hold-index as the 'nothing seen' default and the caller that
    knows the true trace may overwrite (run_voice.py does, from the action
    trace, so the speaker is honestly ASKED to say what was pressed even
    where it cannot know -- that ceiling is the P91d measurement)."""
    d = np.zeros(n_steps, dtype=np.int64)
    m = np.zeros(n_steps, dtype=np.int64)
    a = np.full(n_steps, 2, dtype=np.int64)
    for r in records:
        t = r["quote"]["frame"]
        if 0 <= t < n_steps:
            dd, mm, aa = sentence_from_record(r)
            d[t], m[t], a[t] = dd, mm, aa
    return d, m, a
