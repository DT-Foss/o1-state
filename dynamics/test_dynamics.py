"""
DYNAMICS smoke tests -- pytest oder direkt (python3 dynamics/test_dynamics.py).

  1. Lifted-W-Gesundheit: row-stochastisch, Masse-Erhaltung unter W^T
     (1e-9), beide Schichten erreichbar.
  2. Konsens-Korrektheit auf Winz-Graph: ALLE drei Methoden konvergieren
     zum exakten Mittel (die Replikationsfrage ist WIE SCHNELL, nie OB --
     eine Methode, die woanders hinläuft, wäre ein Bug, kein Befund).
  3. PS-Lifted schlägt Uniform auf dem Pfad-Graphen (der klassische
     Bottleneck-Fall der Lifted-Chain-Literatur) -- schwache Vorbedingung
     der Replikation, kein Register-Bar.
  4. RapidityAdam: Schrittnorm strikt < lr (Lorentz-Grenze), Rapidität
     akkumuliert, ein Schritt senkt den Loss eines Quadratik-Problems.
  5. lorentz_lr: Peak bei t=0 == peak_lr, Ende << Peak, monoton fallend.
"""

import os
import sys

DYN_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(DYN_DIR)
for p in (REPO_ROOT, DYN_DIR):
    if p not in sys.path:
        sys.path.insert(0, p)

os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np
import scipy.sparse as sp
import torch

torch.set_num_threads(1)

from pslifted import (build_lifted_W, pslifted_rounds, reversible_rounds,
                      fiedler_vector)
from rapidity import RapidityAdam, lorentz_lr


def _path_graph(n):
    rows = list(range(n - 1)) + list(range(1, n))
    cols = list(range(1, n)) + list(range(n - 1))
    return sp.csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(n, n))


def test_lifted_w_health():
    A = _path_graph(12)
    W = build_lifted_W(A)
    rs = np.asarray(W.sum(axis=1)).ravel()
    assert np.allclose(rs, 1.0, atol=1e-12)
    s = np.zeros(24)
    s[:12] = np.random.default_rng(0).random(12)
    tot = s.sum()
    WT = W.T.tocsr()
    for _ in range(50):
        s = WT @ s
    assert abs(s.sum() - tot) < 1e-9
    assert s[12:].sum() > 0, "minus layer never reached -- lift is dead"


def test_all_methods_reach_the_mean():
    A = _path_graph(8)
    x = np.random.default_rng(1).random(8)
    for method in ("uniform", "mh"):
        r = reversible_rounds(A, x, method, tol=1e-3)
        assert r < 20000
    r = pslifted_rounds(A, x, tol=1e-3)
    assert r < 20000


def test_pslifted_beats_uniform_on_path():
    A = _path_graph(40)
    rng = np.random.default_rng(2)
    ps_r, un_r = [], []
    for _ in range(3):
        x = rng.random(40)
        ps_r.append(pslifted_rounds(A, x))
        un_r.append(reversible_rounds(A, x, "uniform"))
    assert np.mean(ps_r) < np.mean(un_r), (ps_r, un_r)


def test_fiedler_orients():
    A = _path_graph(10)
    v = fiedler_vector(A)
    # Pfad-Graph: v2 ist monoton entlang des Pfads (bis auf Vorzeichen)
    d = np.diff(v)
    assert (d > 0).all() or (d < 0).all()


def test_rapidity_adam_bounded_and_learns():
    torch.manual_seed(0)
    p = torch.nn.Parameter(torch.tensor([5.0, -3.0]))
    opt = RapidityAdam([p], lr=0.1)
    prev = p.detach().clone()
    loss0 = float((p ** 2).sum())
    for _ in range(60):
        opt.zero_grad()
        loss = (p ** 2).sum()
        loss.backward()
        opt.step()
        step = (p.detach() - prev).abs().max()
        assert step < 0.1 * (1 + 1e-5), "Lorentz-Grenze verletzt: |Schritt| > lr"
        prev = p.detach().clone()
    assert float((p ** 2).sum()) < loss0 * 0.5
    w = opt.state[p]["w"]
    assert w.abs().max() > 0


def test_lorentz_schedule_shape():
    total = 375
    lrs = [lorentz_lr(t, total, peak_lr=3e-4) for t in range(total + 1)]
    assert abs(lrs[0] - 3e-4) < 1e-12
    assert lrs[-1] < 0.12 * lrs[0]
    assert all(a >= b - 1e-15 for a, b in zip(lrs, lrs[1:]))


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"PASS {fn.__name__}", flush=True)
    print(f"[test_dynamics] {len(fns)}/{len(fns)} PASS", flush=True)
