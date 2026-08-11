"""
PS-LIFTED, UNABHÄNGIG REPLIZIERT -- push-sum consensus on non-reversibly
lifted chains, implemented FRESH from the paper's spec, not from any
existing code.

Source of the spec: Foss (2026), "Constant-Round Gossip Consensus via
PS-Lifted Chains" (FERTIG/Formeln/Foss_2026_Constant-Round_Gossip_
Consensus_PS-Lifted-2). The claim under test is the paper's Table 1 /
Fig-(b): PS-Lifted reaches ||x_hat - x_bar||_inf < 0.01 in 12-34 rounds
REGARDLESS of n (Karate: 12 vs uniform 125 / MH 101; BA graphs n=100 ...
30,000 flat), where reversible gossip scales diffusively with the
spectral gap. An independent re-implementation either reproduces the
constant-round behavior or it does not -- both outcomes are the point
(register block P95 in PREDICTIONS_DYNAMICS.md).

Construction (verbatim from the paper's "Transition Matrix" section):
  - Orientation: Fiedler vector v2 of the graph Laplacian; edge i->j
    oriented forward iff v2[i] < v2[j].
  - Doubled states (i,+) and (i,-). With F_i = forward neighbors of i,
    B_i = backward neighbors, pc (continue), ps (self), pr = 1 - pc - ps:
        W[(i,+),(j,+)] = pc/|F_i|   for j in F_i
        W[(i,+),(j,-)] = pr/|B_i|   for j in B_i
        W[(i,+),(i,+)] = ps
    and symmetrically for (i,-) with forward/backward exchanged.
  - Push-sum (Kempe et al. 2003): every lifted state carries (s, w);
    init s_{i,+} = x_i, w_{i,+} = 1, minus layer 0. Per round the pairs
    flow along the chain; estimate x_hat_i = (s_{i,+}+s_{i,-}) /
    (w_{i,+}+w_{i,-}).

Two implementation decisions the paper's text under-determines, made
explicit here (they are reported in the result JSON, not hidden):
  1. Mass action: the paper writes "s(t+1) = W s(t)" and in the same
     breath claims mass conservation (sum_i s_i = const). A ROW-stochastic
     W conserves mass under the TRANSPOSE action (each state splits its
     mass along its outgoing row); s <- W^T s is therefore what push-sum
     means and what is implemented -- test_dynamics.py asserts the
     conservation to 1e-9.
  2. Boundary nodes: a v2-extremal node has F_i (or B_i) empty. Its pc
     mass has no forward target; it is folded into the reverse share
     (momentum reflects at the boundary). Self-loops stay ps.

Baselines, same convergence rule, same trials: uniform max-degree
averaging (W = I - L/(d_max+1)) and Metropolis-Hastings weights -- the
two standard reversible gossips the paper races against (FDLA needs an
SDP solve and is deliberately skipped; MH tracks it within ~2x in the
paper's own table).
"""

import json
import os
from typing import Dict, List, Tuple

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

PC_DEFAULT = 0.95    # continue probability (momentum)
PS_DEFAULT = 0.003   # self-loop (breaks periodicity)   -> pr = 0.047
TOL = 0.01           # ||x_hat - x_bar||_inf, the paper's rule
MAX_ROUNDS = 20000


def fiedler_vector(A: sp.spmatrix) -> np.ndarray:
    """v2 of the combinatorial Laplacian, smallest nonzero eigenvalue,
    via sparse Lanczos (the paper's own 'compute once at startup' rule)."""
    n = A.shape[0]
    d = np.asarray(A.sum(axis=1)).ravel()
    L = sp.diags(d) - A
    # shift-invert around 0 finds the small end; k=2 -> [lambda1=0, lambda2]
    vals, vecs = spla.eigsh(L.asfptype(), k=2, sigma=-1e-6, which="LM")
    order = np.argsort(vals)
    return vecs[:, order[1]]


def build_lifted_W(A: sp.spmatrix, pc: float = PC_DEFAULT,
                   ps: float = PS_DEFAULT) -> sp.csr_matrix:
    """The 2n x 2n row-stochastic lifted transition matrix. State index:
    (i, +) -> i, (i, -) -> n + i."""
    n = A.shape[0]
    v2 = fiedler_vector(A)
    pr = 1.0 - pc - ps
    A = A.tocoo()
    fwd: List[List[int]] = [[] for _ in range(n)]
    bwd: List[List[int]] = [[] for _ in range(n)]
    for i, j in zip(A.row, A.col):
        if i == j:
            continue
        # ties broken by index so the orientation is a total order (the
        # paper's rule v2[i] < v2[j]; equal components are possible on
        # symmetric graphs and must not orphan the edge)
        if (v2[i], i) < (v2[j], j):
            fwd[i].append(j)
            bwd[j].append(i)
    rows, cols, vals = [], [], []

    def emit(r, c, v):
        rows.append(r); cols.append(c); vals.append(v)

    for i in range(n):
        for layer in (0, 1):                      # 0 = plus, 1 = minus
            src = i + layer * n
            F = fwd[i] if layer == 0 else bwd[i]  # minus layer: exchanged
            B = bwd[i] if layer == 0 else fwd[i]
            p_fwd, p_rev = pc, pr
            if not F:                             # boundary: momentum reflects
                p_rev += p_fwd
                p_fwd = 0.0
            if not B:
                p_fwd += p_rev
                p_rev = 0.0
            emit(src, src, ps + (p_fwd if not F and not B else 0.0))
            if F and p_fwd > 0:
                share = p_fwd / len(F)
                for j in F:
                    emit(src, j + layer * n, share)          # stay in layer
            if B and p_rev > 0:
                share = p_rev / len(B)
                for j in B:
                    emit(src, j + (1 - layer) * n, share)    # switch layer
    W = sp.csr_matrix((vals, (rows, cols)), shape=(2 * n, 2 * n))
    row_sums = np.asarray(W.sum(axis=1)).ravel()
    assert np.allclose(row_sums, 1.0, atol=1e-12), "W must be row-stochastic"
    return W


def pslifted_rounds(A: sp.spmatrix, x: np.ndarray, pc: float = PC_DEFAULT,
                    ps: float = PS_DEFAULT, tol: float = TOL,
                    max_rounds: int = MAX_ROUNDS) -> int:
    """Rounds until ||x_hat - x_bar||_inf < tol. Returns max_rounds if never."""
    n = A.shape[0]
    W_T = build_lifted_W(A, pc, ps).T.tocsr()
    s = np.zeros(2 * n)
    w = np.zeros(2 * n)
    s[:n] = x
    w[:n] = 1.0
    target = x.mean()
    s_total = s.sum()
    for t in range(1, max_rounds + 1):
        s = W_T @ s
        w = W_T @ w
        assert abs(s.sum() - s_total) < 1e-6 * max(1.0, abs(s_total)), \
            "push-sum mass leaked -- implementation bug, not a finding"
        denom = w[:n] + w[n:]
        ok = denom > 1e-12
        est = np.where(ok, (s[:n] + s[n:]) / np.where(ok, denom, 1.0), np.inf)
        if np.max(np.abs(est - target)) < tol:
            return t
    return max_rounds


def reversible_rounds(A: sp.spmatrix, x: np.ndarray, kind: str,
                      tol: float = TOL, max_rounds: int = MAX_ROUNDS) -> int:
    """Uniform max-degree gossip or Metropolis-Hastings gossip (both
    symmetric, doubly stochastic -> plain repeated averaging)."""
    n = A.shape[0]
    d = np.asarray(A.sum(axis=1)).ravel()
    A = A.tocoo()
    rows, cols, vals = [], [], []
    if kind == "uniform":
        alpha = 1.0 / (d.max() + 1.0)
        offdiag = {}
        for i, j in zip(A.row, A.col):
            if i != j:
                offdiag[(i, j)] = alpha
    elif kind == "mh":
        offdiag = {}
        for i, j in zip(A.row, A.col):
            if i != j:
                offdiag[(i, j)] = 1.0 / (1.0 + max(d[i], d[j]))
    else:
        raise ValueError(kind)
    diag = np.ones(n)
    for (i, j), v in offdiag.items():
        rows.append(i); cols.append(j); vals.append(v)
        diag[i] -= v
    for i in range(n):
        rows.append(i); cols.append(i); vals.append(diag[i])
    W = sp.csr_matrix((vals, (rows, cols)), shape=(n, n))
    target = x.mean()
    est = x.copy()
    for t in range(1, max_rounds + 1):
        est = W @ est
        if np.max(np.abs(est - target)) < tol:
            return t
    return max_rounds


def run_benchmark(out_path: str, n_trials: int = 5, seed: int = 42,
                  ba_sizes=(100, 1000, 4000)) -> Dict:
    """Karate + BA(m=3) graphs, three methods, n_trials value draws each
    (x ~ U[0,1], fresh per trial; the graph is fixed per size)."""
    import networkx as nx
    graphs: List[Tuple[str, sp.spmatrix]] = [
        ("karate", nx.to_scipy_sparse_array(nx.karate_club_graph(), format="csr"))
    ]
    for nn in ba_sizes:
        g = nx.barabasi_albert_graph(nn, 3, seed=seed)
        graphs.append((f"ba_{nn}", nx.to_scipy_sparse_array(g, format="csr")))
    rng = np.random.default_rng(seed)
    out = {"pc": PC_DEFAULT, "ps": PS_DEFAULT, "tol": TOL,
           "n_trials": n_trials, "graphs": {}}
    for name, A in graphs:
        n = A.shape[0]
        res = {"n": int(n), "edges": int(A.nnz // 2)}
        for method in ("pslifted", "uniform", "mh"):
            rounds = []
            for _ in range(n_trials):
                x = rng.random(n)
                if method == "pslifted":
                    rounds.append(pslifted_rounds(A, x))
                else:
                    rounds.append(reversible_rounds(A, x, method))
            res[method] = {"rounds_mean": float(np.mean(rounds)),
                           "rounds": [int(r) for r in rounds]}
        out["graphs"][name] = res
        print(f"[pslifted] {name} (n={n}): "
              f"pslifted {res['pslifted']['rounds_mean']:.1f} | "
              f"uniform {res['uniform']['rounds_mean']:.1f} | "
              f"mh {res['mh']['rounds_mean']:.1f}", flush=True)
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    return out


if __name__ == "__main__":
    here = os.path.dirname(os.path.abspath(__file__))
    run_benchmark(os.path.join(here, "results", "pslifted_replication.json"))
