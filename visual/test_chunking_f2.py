"""
F2 PROOF at the visual-organism integration point: exact chunk streaming.

Claim (F1/F2 in FOUNDATIONS.md, now checked for the frame-encoder + GSSMCore
+ frame-decoder stack): running a long frame sequence as one-piece forward
(no chunk boundaries, one call, full state carried internally by the scan)
must produce the IDENTICAL predictions as running it in K-frame chunks with
states detached and carried between chunks -- because detaching a tensor
changes nothing about its VALUE, only whether gradients flow through it, and
GSSMCore's recurrence only ever reads the carried z-state's value, never its
grad-graph.

Two sub-claims, checked separately (per the task spec):
  (a) GSSMCore itself: chunked-with-detach vs one-piece must be EXACTLY
      equal (max|delta| == 0.0) -- pure float64-free arithmetic, no
      nondeterministic op crosses the boundary.
  (b) Full stack (encoder -> GSSMCore -> decoder): chunked vs one-piece must
      match at rounding level (max|delta| at or near float32 eps), since the
      encoder/decoder are stateless per-frame linear ops that introduce no
      cross-chunk dependency of their own -- any nonzero delta here can only
      come from float32 op-order effects in the Linear layers' matmuls, not
      from a state-carrying bug.

Run: python3 visual/test_chunking_f2.py
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import torch

torch.set_num_threads(1)

from hsslm.neural.gssm_core import GSSMCore
from frame_organism import FramePredictor, detach_state_tree, FRAME_DIM


def run_chunked(forward_fn, x, chunk_size, states=None):
    """Runs forward_fn over x in chunks of chunk_size along dim=1, detaching
    and carrying state between chunks. Returns the concatenated output."""
    B, L, D = x.shape
    outs = []
    for lo in range(0, L, chunk_size):
        hi = min(lo + chunk_size, L)
        out, states = forward_fn(x[:, lo:hi], states)
        outs.append(out)
        states = detach_state_tree(states)
    return torch.cat(outs, dim=1)


def test_gssm_core_exact():
    """(a) GSSMCore alone: chunked-with-detach vs one-piece, expect EXACT
    equality (max|delta| == 0.0)."""
    torch.manual_seed(42)
    core = GSSMCore(n_layers=3, d_model=64, n_heads=4, check_bounds=True)
    core.eval()

    B, L, D = 2, 96, 64
    x = torch.randn(B, L, D)

    with torch.no_grad():
        out_full, _ = core(x, None)
        out_chunked = run_chunked(lambda xc, st: core(xc, st), x, chunk_size=16)

    delta = (out_full - out_chunked).abs()
    max_delta = float(delta.max())
    print(f"[F2a] GSSMCore chunked vs one-piece: max|delta| = {max_delta:.3e} "
          f"(mean {float(delta.mean()):.3e})")
    assert max_delta == 0.0, f"GSSMCore chunking is NOT exact: max|delta|={max_delta}"
    print("[F2a] PASS -- exact zero delta, as expected (Theorem 2's recurrence "
          "only reads carried state VALUES, detach doesn't touch them).")


def test_full_stack_rounding():
    """(b) Full frame stack (encoder -> GSSMCore -> decoder): chunked vs
    one-piece, expect match at rounding level (encoder/decoder are
    stateless, so any nonzero delta is pure float32 op-order noise, not a
    state-carrying bug)."""
    torch.manual_seed(7)
    model = FramePredictor(d_model=128, n_layers=3, n_heads=4, frame_dim=FRAME_DIM)
    model.eval()

    B, L = 2, 96
    frames = torch.rand(B, L, FRAME_DIM)

    with torch.no_grad():
        pred_full, _ = model(frames, None)
        pred_chunked = run_chunked(lambda fc, st: model(fc, st), frames, chunk_size=16)

    delta = (pred_full - pred_chunked).abs()
    max_delta = float(delta.max())
    print(f"[F2b] Full stack (encoder+GSSMCore+decoder) chunked vs one-piece: "
          f"max|delta| = {max_delta:.3e} (mean {float(delta.mean()):.3e})")
    # float32 rounding-level tolerance -- generous but still tight (predicted
    # values live in [0,1] via sigmoid, so 1e-5 absolute is a real rounding
    # bound, not a loophole).
    assert max_delta < 1e-5, f"Full-stack chunking delta too large: {max_delta}"
    print("[F2b] PASS -- rounding-level match (stateless encoder/decoder introduce "
          "no cross-chunk dependency; residual delta is float32 matmul op-order noise).")


if __name__ == "__main__":
    print("=" * 70)
    print("F2 PROOF: chunked streaming with detach-carry == one-piece forward")
    print("=" * 70)
    test_gssm_core_exact()
    print()
    test_full_stack_rounding()
    print()
    print("F2 PROOF: ALL CHECKS PASSED")
