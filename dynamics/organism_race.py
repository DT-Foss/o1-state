#!/usr/bin/env python3 -u
"""
ORGANISM RACE -- die Formel-Dynamiken dort antreten lassen, wo o1-state
wirklich lebt: im Stream-Regime (ein Leben, ein Pass, Surprise-Gate,
getragener GSSM-Zustand), gezielt auf die zwei GEMESSENEN Wunden des
body-Tracks:

  Wunde 1 (P87b, falsifiziert): copy-last wurde bei 6000 Frames in roher
  L1 nicht geschlagen -- der Aktionskanal zahlt sich in der Loss noch
  nicht aus.
  Wunde 2 (P88/P92-Instrumente): die Aktionskanal-Ignition ist langsam;
  Hit-Rate über Chance kommt spät.

Vier Arme, hart gepaart -- identische Welt (ActedSyntheticSource, Seed 42,
dieselbe vorgezogene Random-Aktionsfolge: der gemessen schnellste Lehrer
aus P88), identische Init-Gewichte, identisches Gate, identisches Budget;
NUR die Update-Dynamik unterscheidet sich:

  adam              Adam @ 3e-4                (exakt der body-Lauf: Referenz)
  rapidity          RapidityAdam @ 3e-4        (Möbius-Momentum, |Schritt|<lr)
  lorentz           Adam + tau-Lorentz-LR      (Peak = 3e-4, Kühlung 0.95->0.5)
  rapidity_lorentz  beides

Bewertung ausschließlich auf der FESTEN Probe-Route in separater Welt
(voice/-Design, P89c-Confound tot): probe_l1 unter der geskripteten
Aktion, Counterfactual-Hit mit Tie-Guard, und als Nullpunkt der
copy-last der Route selbst (eine Konstante der Route, einmal berechnet).
Register: dynamics/PREDICTIONS_DYNAMICS.md P96, VOR den Läufen.
"""

import os
import sys
import json
import time
import argparse

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

DYN_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(DYN_DIR)
for p in (REPO_ROOT, os.path.join(REPO_ROOT, "visual"),
          os.path.join(REPO_ROOT, "body"), os.path.join(REPO_ROOT, "voice"),
          DYN_DIR):
    if p not in sys.path:
        sys.path.insert(0, p)

import numpy as np
import torch

torch.set_num_threads(1)
try:
    torch.set_num_interop_threads(1)
except RuntimeError:
    pass

from action_sources import ActedSyntheticSource              # read-only (body/)
from body_organism import ActionConditionedPredictor, BodyStreamer
from run_voice import probe_eval                              # read-only (voice/)
from rapidity import RapidityAdam, lorentz_lr

MODEL_SEED = 4242


def simulate(seed: int, actions):
    return np.stack(ActedSyntheticSource.replay_frames(seed, actions), axis=0)


def make_model():
    torch.manual_seed(MODEL_SEED)
    return ActionConditionedPredictor(n_actions=3)


def main():
    ap = argparse.ArgumentParser(description="Formel-Dynamiken im Organismus-Regime")
    ap.add_argument("--frames", type=int, default=6000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--probe-every", type=int, default=500)
    ap.add_argument("--probe-len", type=int, default=96)
    ap.add_argument("--out", default=os.path.join(DYN_DIR, "results",
                                                  "organism_race.json"))
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed + 1)
    life_actions = [int(a) for a in rng.integers(0, 3, size=args.frames)]
    life_frames = simulate(args.seed, life_actions)

    rngp = np.random.default_rng(9009)
    p_actions = [int(a) for a in rngp.integers(0, 3, size=args.probe_len)]
    p_frames = simulate(999, p_actions)
    copylast_probe = float(np.mean(np.abs(
        p_frames[1:len(p_actions) + 1] - p_frames[:len(p_actions)])))

    K = 16
    n_chunks_total = (args.frames - K) // K + 1
    arms = ["adam", "rapidity", "lorentz", "rapidity_lorentz"]
    results = {}
    for arm in arms:
        t0 = time.time()
        model = make_model()
        if arm in ("rapidity", "rapidity_lorentz"):
            opt = RapidityAdam(model.parameters(), lr=args.lr)
        else:
            opt = torch.optim.Adam(model.parameters(), lr=args.lr)
        streamer = BodyStreamer(model, opt, torch.device("cpu"))
        checkpoints = []
        l1, hit = probe_eval(model, p_frames, p_actions)
        checkpoints.append({"frame": 0, "probe_l1": round(l1, 6),
                            "probe_hit": round(hit, 4)})
        life_l1 = []
        ci = 0
        for c0 in range(0, args.frames - K, K):
            if arm in ("lorentz", "rapidity_lorentz"):
                lr_now = lorentz_lr(ci, n_chunks_total, peak_lr=args.lr)
                for g in opt.param_groups:
                    g["lr"] = lr_now
            x = torch.from_numpy(
                life_frames[c0:c0 + K].reshape(K, -1).astype(np.float32))[None]
            y = torch.from_numpy(
                life_frames[c0 + 1:c0 + K + 1].reshape(K, -1).astype(np.float32))[None]
            acts = torch.tensor(life_actions[c0:c0 + K], dtype=torch.long)[None]
            s, _, _ = streamer.step_gated(x, acts, y)
            life_l1.append(s)
            ci += 1
            fr = c0 + K
            if fr % args.probe_every < K and fr >= args.probe_every:
                l1, hit = probe_eval(model, p_frames, p_actions)
                checkpoints.append({"frame": fr, "probe_l1": round(l1, 6),
                                    "probe_hit": round(hit, 4)})
        late = life_l1[int(len(life_l1) * 0.8):]
        results[arm] = {
            "checkpoints": checkpoints,
            "hit_auc": round(float(np.mean([c["probe_hit"] for c in checkpoints])), 4),
            "final_probe_l1": checkpoints[-1]["probe_l1"],
            "final_probe_hit": checkpoints[-1]["probe_hit"],
            "life_l1_late": round(float(np.mean(late)), 6),
            "gate": streamer.stats(),
            "wall_s": round(time.time() - t0, 1),
        }
        r = results[arm]
        print(f"[race:{arm}] hit {checkpoints[0]['probe_hit']:.3f} -> "
              f"{r['final_probe_hit']:.3f} (AUC {r['hit_auc']:.3f}) | "
              f"probe L1 {r['final_probe_l1']:.4f} vs copy-last {copylast_probe:.4f} | "
              f"life late {r['life_l1_late']:.4f} | {r['wall_s']}s", flush=True)

    out = {"frames": args.frames, "seed": args.seed, "lr": args.lr,
           "model_seed": MODEL_SEED, "probe_len": args.probe_len,
           "copylast_probe_l1": round(copylast_probe, 6), "arms": results}
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    print(f"[race] -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
