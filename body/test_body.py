"""
BODY smoke tests -- runnable via pytest OR directly (python3 body/test_body.py).

What is covered and WHY (each test guards a load-bearing claim, not a line):
  1. acted-world determinism -- the provenance rule for learned policies is
     "replay = seed + recorded trace"; if two worlds under the same trace
     diverge byte-wise, every acted record's bit-exactness claim is dead.
  2. copy-last start -- the zero-init discipline (delta head AND action
     embedding) must make step-0 predictions EQUAL the input frame; if not,
     the 'action channel earns its influence' framing is wrong from line 1.
  3. counterfactual purity -- probing "what if" must not advance or mutate
     the live carried state (states are cloned per action); a mutating
     probe would silently corrupt F2's exactness.
  4. streamer mechanics -- chunk stepping trains, carries state, counts.
  5. shift estimator ground truth -- a synthetically shifted frame must be
     recovered at the exact dx with the documented sign convention.
  6. record round-trip -- a small acted run yields records that replay
     BIT-EXACT through verify_records; the store seals and re-verifies.
  7. learning-progress policy -- an action whose error stream improves must
     end up preferred over flat ones post-ignition; ignition is uniform.

No vizdoom dependency here: everything runs on the synthetic acted world
(the always-available arm), matching the repo's convention that the smoke
layer never needs optional packages.
"""

import os
import sys

BODY_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(BODY_DIR)
for p in (REPO_ROOT, os.path.join(REPO_ROOT, "visual"),
          os.path.join(REPO_ROOT, "src"), BODY_DIR):
    if p not in sys.path:
        sys.path.insert(0, p)

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import numpy as np
import torch

torch.set_num_threads(1)

from action_sources import ActedSyntheticSource
from body_organism import (
    ActionConditionedPredictor, BodyStreamer, LearningProgressPolicy,
)
from causal_records import (
    ActedRecordExtractor, estimate_shift, frame_sha, seal_records,
    verify_records,
)
from livecausal.store import LiveStore


def test_acted_world_determinism():
    rng = np.random.default_rng(3)
    actions = [int(a) for a in rng.integers(0, 3, size=120)]
    f_a = ActedSyntheticSource.replay_frames(seed=9, actions=actions)
    f_b = ActedSyntheticSource.replay_frames(seed=9, actions=actions)
    assert len(f_a) == len(actions) + 1
    for a, b in zip(f_a, f_b):
        assert a.tobytes() == b.tobytes()
    # and a LIVE run equals its own replay -- the actual provenance path
    src = ActedSyntheticSource(seed=9)
    live = [src.observe()]
    for a in actions:
        src.act(a)
        live.append(src.observe())
    for a, b in zip(live, f_a):
        assert a.tobytes() == b.tobytes()


def test_zero_init_starts_at_copy_last():
    torch.manual_seed(0)
    model = ActionConditionedPredictor(n_actions=3)
    x = torch.rand(1, 4, model.frame_dim)
    acts = torch.tensor([[0, 1, 2, 1]])
    pred, states = model(x, acts)
    assert torch.allclose(pred, x), "zero-init delta head must start AT copy-last"
    assert len(states) == model.core.n_layers


def test_counterfactual_does_not_mutate_states():
    torch.manual_seed(0)
    model = ActionConditionedPredictor(n_actions=3)
    x = torch.rand(1, 8, model.frame_dim)
    acts = torch.randint(0, 3, (1, 8))
    _, states = model(x, acts)
    snap = [tuple(t.clone() for t in layer) for layer in states]
    cf = model.counterfactual(torch.rand(1, 1, model.frame_dim), states)
    assert cf.shape == (3, model.frame_dim)
    assert float(cf.min()) >= 0.0 and float(cf.max()) <= 1.0
    for lay, lay_snap in zip(states, snap):
        for t, t_snap in zip(lay, lay_snap):
            assert torch.equal(t, t_snap), "counterfactual mutated the live state"


def test_streamer_runs_and_carries():
    torch.manual_seed(1)
    model = ActionConditionedPredictor(n_actions=3, d_model=64, n_layers=2)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    st = BodyStreamer(model, opt, torch.device("cpu"))
    for _ in range(3):
        x = torch.rand(1, 6, model.frame_dim)
        acts = torch.randint(0, 3, (1, 6))
        y = torch.rand(1, 6, model.frame_dim)
        s, gated, pfl = st.step_gated(x, acts, y)
        assert np.isfinite(s) and pfl.shape == (1, 6)
    assert st.n_chunks == 3 and st.n_bwd == 3  # all ignition -> all gated
    assert st.states is not None and not st.states[0][0].requires_grad


def test_shift_estimator_recovers_known_shift():
    rng = np.random.default_rng(5)
    world = rng.random((64, 80), dtype=np.float64).astype(np.float32)
    # viewport at x=5 then x=8 (moved RIGHT by 3) -> convention dx = +3
    f0, f1 = world[:, 5:5 + 64], world[:, 8:8 + 64]
    s, err_s, err_0 = estimate_shift(f0, f1)
    assert s == 3 and err_s < 1e-6 and err_0 > err_s


def test_acted_records_roundtrip(tmp_path=None):
    src = ActedSyntheticSource(seed=11)
    rng = np.random.default_rng(2)
    ex = ActedRecordExtractor("acted_synthetic", base_seed=11,
                              action_names=src.ACTION_NAMES,
                              policy_name="random")
    prev = src.observe()
    for _ in range(80):
        ep0, fi0 = src.episode, src.frame_idx
        a = int(rng.integers(0, 3))
        src.act(a)
        cur = src.observe()
        ex.offer(prev, a, cur, ep0, fi0, crossed_boundary=False)
        prev = cur
    assert len(ex.records) > 0, "80 steps with pans must yield acted records"
    # every record's measured dx must agree with world ground truth
    for r in ex.records:
        q = r["quote"]
        assert q["dx_measured"] == src.dx_truth[q["frame"]], (
            f"extractor dx {q['dx_measured']} != ground truth "
            f"{src.dx_truth[q['frame']]} at frame {q['frame']}")
    prov = verify_records(ex.records, src.trace, n_samples=4, seed=1)
    assert prov["pass"], f"provenance failed: {prov}"
    # seal + store integrity
    import tempfile
    d = tmp_path or tempfile.mkdtemp(prefix="body_store_test_")
    sha = seal_records(ex.records, str(d), None)
    assert LiveStore(str(d)).verify(), "sealed segment failed store verify"
    assert sha in LiveStore(str(d)).segments()


def test_learning_progress_policy_prefers_improving_action():
    pol = LearningProgressPolicy(3, window=40, eps=0.1, tau=0.5,
                                 min_samples=8, ignition_steps=10, seed=0)
    # ignition: uniform
    assert np.allclose(pol.probs(), 1 / 3)
    for i in range(40):
        pol.update(0, 0.5 - i * 0.01)   # improving -> positive LP
        pol.update(1, 0.5)              # flat mastered/unlearnable
        pol.update(2, 0.5 + (i % 2) * 0.001)  # flat noise
    probs = pol.probs()
    lp = pol.lp_estimates()
    assert lp[0] > 0.05 and abs(lp[1]) < 0.01
    assert probs[0] == max(probs), f"improving action not preferred: {probs}"
    assert min(probs) >= 0.1 / 3 - 1e-9  # eps floor keeps every action alive
    for _ in range(20):
        assert 0 <= pol.choose() < 3


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"PASS {fn.__name__}", flush=True)
    print(f"[test_body] {len(fns)}/{len(fns)} PASS", flush=True)
