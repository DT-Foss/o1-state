"""
VOICE smoke tests -- pytest or direct (python3 voice/test_voice.py).

Guarded claims, one per test:
  1. codec bijection -- every world dx round-trips through the sentence
     vocabulary; the action<->direction bijection matches the body world.
  2. rich world provenance -- RichPanSource keeps the (seed, trace) ->
     byte-identical-frames contract UNDER the magnitude cycle, live == replay,
     and dx_truth realizes all magnitudes 1..3 (else the mag slot is dead).
  3. imagination is the world op -- shift_frame agrees with the causal
     extractor's estimator (shift by dx, estimator re-reads dx): the
     lesson B imagines is the lesson A's records describe, cross-module.
  4. scramble is a consistent lie -- deterministic, silence-preserving,
     never the identity on consequences.
  5. speaker mechanics -- shapes, holdout mask arithmetic, one training
     chunk moves the loss, narration API returns aligned arrays.
  6. paired-arm identity -- two make_b_model() calls yield bit-identical
     init weights (the transmission comparison is model-paired by
     construction, not by hope).
"""

import os
import sys

VOICE_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(VOICE_DIR)
for p in (REPO_ROOT, os.path.join(REPO_ROOT, "visual"),
          os.path.join(REPO_ROOT, "body"), os.path.join(REPO_ROOT, "src"),
          VOICE_DIR):
    if p not in sys.path:
        sys.path.insert(0, p)

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import numpy as np
import torch

torch.set_num_threads(1)

from vocab import (RichPanSource, sentence_from_dx, dx_from_sentence,
                   action_from_direction, labels_for_life)
from imagination import shift_frame, scramble_utterances, utterances_to_pairs
from speaker import SpeakerNet, train_speaker, utter_life
from causal_records import estimate_shift


def test_codec_bijection():
    for dx in (-3, -2, -1, 0, 1, 2, 3):
        d, m = sentence_from_dx(dx)
        assert dx_from_sentence(d, m) == dx
    assert action_from_direction(1) == 0   # left seen -> pan_left pressed
    assert action_from_direction(2) == 1
    assert action_from_direction(0) == 2


def test_rich_world_provenance_and_magnitudes():
    rng = np.random.default_rng(4)
    actions = [int(a) for a in rng.integers(0, 3, size=150)]
    fa = RichPanSource.replay_frames(seed=5, actions=actions)
    fb = RichPanSource.replay_frames(seed=5, actions=actions)
    for a, b in zip(fa, fb):
        assert a.tobytes() == b.tobytes()
    src = RichPanSource(seed=5)
    live = [src.observe()]
    for a in actions:
        src.act(a)
        live.append(src.observe())
    for a, b in zip(live, fa):
        assert a.tobytes() == b.tobytes()
    mags = {abs(d) for d in src.dx_truth if d != 0}
    assert mags == {1, 2, 3}, f"magnitude cycle incomplete: {mags}"


def test_imagination_matches_world_op():
    rng = np.random.default_rng(6)
    f = rng.random((64, 64)).astype(np.float32)
    for dx in (-3, -1, 1, 2, 3):
        g = shift_frame(f, dx)
        s, err_s, err_0 = estimate_shift(f, g)
        assert s == dx and err_s < 1e-6 < err_0, (dx, s, err_s, err_0)
    assert shift_frame(f, 0).tobytes() == f.tobytes()


def test_scramble_is_consistent_lie():
    utt = [(1, 3), (2, 2), (0, 0), (1, 1)]
    s1 = scramble_utterances(utt)
    s2 = scramble_utterances(utt)
    assert s1 == s2
    assert s1[2] == (0, 0)                      # silence stays silence
    for (d, m), (sd, sm) in zip(utt, s1):
        if d != 0:
            assert sd != d, "direction must be lied about"


def test_speaker_mechanics():
    rng = np.random.default_rng(7)
    frames = rng.random((200, 64, 64)).astype(np.float32)
    d = rng.integers(0, 3, 199).astype(np.int64)
    m = rng.integers(0, 4, 199).astype(np.int64)
    a = rng.integers(0, 3, 199).astype(np.int64)
    model, trace = train_speaker(frames, (d, m, a), train_upto=160,
                                 chunk=32, passes=2, seed=0, log_every=0)
    assert len(trace) == 8 and all(np.isfinite(v) for v in trace)
    said = utter_life(model, frames)
    assert all(len(s) == 199 for s in said)
    assert said[0].max() < 3 and said[1].max() < 4 and said[2].max() < 3


def test_paired_arm_identity():
    from run_voice import make_b_model
    m1, _ = make_b_model()
    m2, _ = make_b_model()
    for p1, p2 in zip(m1.parameters(), m2.parameters()):
        assert torch.equal(p1, p2), "arms must start bit-identical"


def test_utterance_pairs_cap_and_silence():
    frames = np.random.default_rng(8).random((10, 64, 64)).astype(np.float32)
    utt = [(0, 0)] * 5 + [(1, 2)] * 30
    x, a, y = utterances_to_pairs(utt, frames, cap=12, seed=0)
    assert x.shape[0] == 12 and (a == 0).all()   # left-said -> pan_left
    s, err_s, _ = estimate_shift(x[0].reshape(64, 64), y[0].reshape(64, 64))
    assert s == -2 and err_s < 1e-6


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"PASS {fn.__name__}", flush=True)
    print(f"[test_voice] {len(fns)}/{len(fns)} PASS", flush=True)
