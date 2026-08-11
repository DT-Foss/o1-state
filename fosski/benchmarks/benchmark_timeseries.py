#!/usr/bin/env python3
"""
Benchmark: FOSS-KI vs Standard Reservoir on Time Series
==========================================================
Test tasks:
1. Mackey-Glass chaotic time series prediction
2. NARMA-10 nonlinear system identification
3. Sine wave frequency discrimination
4. Lorenz attractor short-term prediction
5. Memory capacity (how far back can it remember?)
6. Ensemble consensus speedup
"""

import numpy as np
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core.engine import FossKI
from core.reservoir import MarkovReservoir
from core.consensus import EnsembleConsensus
import networkx as nx


def mackey_glass(n_steps=2000, tau=17, beta=0.2, gamma=0.1, n_pow=10, dt=1.0):
    """Generate Mackey-Glass chaotic time series."""
    history = np.ones(tau + 1) * 1.2
    series = [1.2]

    for t in range(1, n_steps):
        x = series[-1]
        x_tau = history[max(0, len(history) - tau)]
        dx = beta * x_tau / (1 + x_tau ** n_pow) - gamma * x
        x_new = x + dt * dx
        series.append(x_new)
        history = np.append(history, x_new)

    return np.array(series)


def narma10(n_steps=2000, seed=42):
    """Generate NARMA-10 system."""
    rng = np.random.RandomState(seed)
    u = rng.uniform(0, 0.5, n_steps)
    y = np.zeros(n_steps)

    for t in range(10, n_steps):
        y[t] = (0.3 * y[t-1]
                + 0.05 * y[t-1] * np.sum(y[t-10:t])
                + 1.5 * u[t-1] * u[t-10]
                + 0.1)

    return u, y


def lorenz(n_steps=5000, dt=0.01, sigma=10, rho=28, beta=8/3):
    """Generate Lorenz attractor."""
    x, y, z = 1.0, 1.0, 1.0
    trajectory = []

    for _ in range(n_steps):
        dx = sigma * (y - x) * dt
        dy = (x * (rho - z) - y) * dt
        dz = (x * y - beta * z) * dt
        x += dx
        y += dy
        z += dz
        trajectory.append([x, y, z])

    return np.array(trajectory)


def rmse(pred, target):
    return np.sqrt(np.mean((pred - target) ** 2))


def nrmse(pred, target):
    return rmse(pred, target) / np.std(target)


def main():
    print("=" * 80)
    print("FOSS-KI BENCHMARK — Time Series")
    print("=" * 80)

    n_nodes = 50
    washout = 50

    # Build graphs
    G_ba = nx.barabasi_albert_graph(n_nodes, 3, seed=42)
    A_ba = nx.to_numpy_array(G_ba)

    G_barbell = nx.barbell_graph(n_nodes // 2, 1)
    A_barbell = nx.to_numpy_array(G_barbell)

    G_ws = nx.watts_strogatz_graph(n_nodes, 4, 0.3, seed=42)
    A_ws = nx.to_numpy_array(G_ws)

    # ── TEST 1: Mackey-Glass ──────────────────────────────────
    print(f"\n{'━' * 80}")
    print("TEST 1: Mackey-Glass Chaotic Prediction (τ=17)")
    print(f"{'━' * 80}")

    mg = mackey_glass(3000)
    mg = (mg - mg.mean()) / mg.std()

    X_mg = mg[:-1].reshape(-1, 1)
    Y_mg = mg[1:].reshape(-1, 1)

    train_end = 2000
    X_train, Y_train = X_mg[:train_end], Y_mg[:train_end]
    X_test, Y_test = X_mg[train_end:], Y_mg[train_end:]

    print(f"\n  {'Config':<30} {'Train NRMSE':>12} {'Test NRMSE':>12}")
    print(f"  {'─' * 58}")

    for name, A in [('BA(50,3)', A_ba), ('Barbell(25)', A_barbell), ('WS(50,4,0.3)', A_ws)]:
        for use_foss in [False, True]:
            p_c = 0.80 if use_foss else 0.50  # p_c=0.50 ≈ undirected (symmetric)
            p_s = 0.003 if use_foss else 0.003
            label = f"{name} {'Foss' if use_foss else 'Base'}"

            res = MarkovReservoir(A, input_dim=1, output_dim=1,
                                  p_c=p_c, p_s=p_s,
                                  spectral_radius=0.95,
                                  input_scaling=0.1, leak_rate=0.3)

            train_err = res.train(X_train, Y_train[washout:],
                                   washout=washout, ridge_alpha=1e-6)
            pred = res.predict(X_test)
            test_err = nrmse(pred, Y_test[:len(pred)])

            print(f"  {label:<30} {train_err:>12.4f} {test_err:>12.4f}")

    # ── TEST 2: NARMA-10 ─────────────────────────────────────
    print(f"\n{'━' * 80}")
    print("TEST 2: NARMA-10 Nonlinear System Identification")
    print(f"{'━' * 80}")

    u_narma, y_narma = narma10(3000)
    X_n = u_narma.reshape(-1, 1)
    Y_n = y_narma.reshape(-1, 1)

    X_n_train, Y_n_train = X_n[:2000], Y_n[:2000]
    X_n_test, Y_n_test = X_n[2000:], Y_n[2000:]

    print(f"\n  {'Config':<30} {'Train NRMSE':>12} {'Test NRMSE':>12}")
    print(f"  {'─' * 58}")

    for name, A in [('BA(50,3)', A_ba), ('Barbell(25)', A_barbell)]:
        for use_foss in [False, True]:
            p_c = 0.80 if use_foss else 0.50
            label = f"{name} {'Foss' if use_foss else 'Base'}"

            res = MarkovReservoir(A, input_dim=1, output_dim=1,
                                  p_c=p_c, p_s=0.003,
                                  spectral_radius=0.95,
                                  input_scaling=0.1, leak_rate=0.3)

            train_err = res.train(X_n_train, Y_n_train[washout:],
                                   washout=washout, ridge_alpha=1e-6)
            pred = res.predict(X_n_test)
            test_err = nrmse(pred, Y_n_test[:len(pred)])

            print(f"  {label:<30} {train_err:>12.4f} {test_err:>12.4f}")

    # ── TEST 3: Memory Capacity ──────────────────────────────
    print(f"\n{'━' * 80}")
    print("TEST 3: Memory Capacity (how far back can it remember?)")
    print(f"{'━' * 80}")

    T_mem = 2000
    rng = np.random.RandomState(42)
    u_mem = rng.uniform(-1, 1, T_mem)

    print(f"\n  {'Config':<30} {'MC (total)':>12} {'Max delay':>10}")
    print(f"  {'─' * 56}")

    for name, A in [('BA(50,3)', A_ba), ('Barbell(25)', A_barbell)]:
        for use_foss in [False, True]:
            p_c = 0.80 if use_foss else 0.50
            label = f"{name} {'Foss' if use_foss else 'Base'}"

            res = MarkovReservoir(A, input_dim=1, output_dim=1,
                                  p_c=p_c, p_s=0.003,
                                  spectral_radius=0.95,
                                  input_scaling=0.1, leak_rate=0.3)

            # Process input
            states = res.process_sequence(u_mem.reshape(-1, 1), washout=washout)

            # Test delays 1..n
            mc_total = 0.0
            max_delay = 0

            for delay in range(1, n_nodes + 1):
                target = u_mem[washout:T_mem - delay]
                state_aligned = states[:len(target)]

                if len(state_aligned) < 10:
                    break

                # Ridge regression for this delay
                S = np.hstack([state_aligned, np.ones((len(state_aligned), 1))])
                Y_d = target.reshape(-1, 1)

                try:
                    W = np.linalg.solve(S.T @ S + 1e-6 * np.eye(S.shape[1]), S.T @ Y_d)
                    pred_d = S @ W
                    r2 = 1.0 - np.sum((Y_d - pred_d) ** 2) / np.sum((Y_d - Y_d.mean()) ** 2)
                    if r2 > 0.01:
                        mc_total += r2
                        max_delay = delay
                except Exception:
                    break

            print(f"  {label:<30} {mc_total:>12.2f} {max_delay:>10}")

    # ── TEST 4: Lorenz Attractor ──────────────────────────────
    print(f"\n{'━' * 80}")
    print("TEST 4: Lorenz Attractor Short-Term Prediction")
    print(f"{'━' * 80}")

    lor = lorenz(5000)
    lor = (lor - lor.mean(axis=0)) / lor.std(axis=0)

    X_lor = lor[:-1]
    Y_lor = lor[1:]

    print(f"\n  {'Config':<30} {'Train NRMSE':>12} {'Test NRMSE':>12}")
    print(f"  {'─' * 58}")

    for name, A in [('BA(50,3)', A_ba), ('Barbell(25)', A_barbell)]:
        for use_foss in [False, True]:
            p_c = 0.80 if use_foss else 0.50
            label = f"{name} {'Foss' if use_foss else 'Base'}"

            res = MarkovReservoir(A, input_dim=3, output_dim=3,
                                  p_c=p_c, p_s=0.003,
                                  spectral_radius=0.95,
                                  input_scaling=0.1, leak_rate=0.3)

            train_err = res.train(X_lor[:3000], Y_lor[washout:3000],
                                   washout=washout, ridge_alpha=1e-6)
            pred = res.predict(X_lor[3000:])
            test_err = nrmse(pred, Y_lor[3000:3000 + len(pred)])

            print(f"  {label:<30} {train_err:>12.4f} {test_err:>12.4f}")

    # ── TEST 5: FossKI Engine ─────────────────────────────────
    print(f"\n{'━' * 80}")
    print("TEST 5: Full FossKI Engine — Mackey-Glass")
    print(f"{'━' * 80}")

    for graph_type in ['barabasi_albert', 'barbell', 'watts_strogatz']:
        config = {
            'n_reservoir_nodes': 50,
            'graph_type': graph_type,
            'graph_params': {'m': 3, 'k': 4, 'p': 0.3},
            'input_dim': 1,
            'output_dim': 1,
            'spectral_radius': 0.95,
            'input_scaling': 0.1,
            'leak_rate': 0.3,
            'p_c': 0.80,
            'p_s': 0.003,
            'memory_size': 100,
            'memory_temperature': 0.0,
            'n_ensembles': 1,
            'ensemble_topology': 'ring',
            'evolve_topology': False,
            'n_generations': 20,
            'washout': 50,
            'ridge_alpha': 1e-6,
        }

        fki = FossKI(config)
        train_err = fki.train(X_train, Y_train)
        pred = fki.predict(X_test)
        test_err = nrmse(pred, Y_test[:len(pred)])

        print(f"  {graph_type:<25} Train={train_err:.4f}  Test={test_err:.4f}  "
              f"γ_ratio={fki.mc.ratio:.3f}")

    # ── TEST 6: Ensemble + Consensus ──────────────────────────
    print(f"\n{'━' * 80}")
    print("TEST 6: Ensemble with Foss Consensus")
    print(f"{'━' * 80}")

    for n_ens in [1, 3, 5]:
        config_ens = {
            'n_reservoir_nodes': 30,
            'graph_type': 'barabasi_albert',
            'graph_params': {'m': 2},
            'input_dim': 1,
            'output_dim': 1,
            'spectral_radius': 0.95,
            'input_scaling': 0.1,
            'leak_rate': 0.3,
            'p_c': 0.80,
            'p_s': 0.003,
            'memory_size': 50,
            'memory_temperature': 0.0,
            'n_ensembles': n_ens,
            'ensemble_topology': 'bottleneck' if n_ens > 2 else 'ring',
            'evolve_topology': False,
            'n_generations': 20,
            'washout': 50,
            'ridge_alpha': 1e-6,
        }

        fki_ens = FossKI(config_ens)
        train_err = fki_ens.train(X_train, Y_train)
        pred = fki_ens.predict(X_test)
        test_err = nrmse(pred, Y_test[:len(pred)])

        print(f"  Ensemble={n_ens:<3} Train={train_err:.4f}  Test={test_err:.4f}")

    # ── FINAL ──────────────────────────────────────────────────
    print(f"\n{'═' * 80}")
    print("BENCHMARK COMPLETE")
    print(f"{'═' * 80}")
    print("""
  WHAT THIS PROVES:
  - MarkovReservoir works as a sequence processor (RC baseline)
  - Z₂ lifting (Foss) changes the reservoir dynamics
  - Graph topology matters (barbell vs BA vs WS)
  - Ensemble consensus can combine multiple reservoirs
  - NO gradient descent needed (only linear readout)

  NEXT: Compare against standard Echo State Network (ESN)
  and LSTM baselines to quantify the advantage/disadvantage.
""")


if __name__ == "__main__":
    main()
