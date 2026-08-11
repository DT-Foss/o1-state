#!/usr/bin/env python3 -u
"""
RUN BODY -- the acting organism's life loop: perceive, CHOOSE, act, predict,
be surprised, learn -- and write what its own hands did into causal records.

Per acted frame:
    f_t = source.observe()
    a_t = policy.choose()            <- the new organ. Nothing in visual/
    source.act(a_t)                     ever chose; this line closes the
    f_{t+1} = source.observe()          Harnad loop's acting half.
Chunks of (f_t, a_t) -> f_{t+1} go through BodyStreamer (F1 surprise gate /
F2 detach-carry, constants imported from frame_organism). The per-frame L1
the gate already computes is fed back per-action into the
LearningProgressPolicy -- surprise stops being only a filter and becomes
the motive.

Instruments written every run (all falsifiable, nothing interpretive):
  - metrics.jsonl, one record per chunk: l1, copy_last_l1 (the baseline the
    model must beat WHERE ITS OWN ACTIONS move the view), gate stats,
    counterfactual separation (mean pairwise L1 between predicted next
    frames under different actions from the SAME state -- 0 means the
    action input is ignored), counterfactual hit (does the taken action's
    prediction match reality better than the untaken ones -- chance 1/A),
    policy probs + learning-progress estimates + action counts.
  - GIFs: (1) life GIF -- real / |diff| / predicted rows plus an action
    strip showing WHICH action produced each transition; (2) counterfactual
    GIF -- per probe state, the predicted next frame under EVERY action
    side by side with reality, taken action marked. "Was sehe ich, WENN ich
    das tue" as pixels, degeneracy visible immediately.
  - acted-event causal records (causal_records.py) sealed into a LiveStore
    under body/records/ + JSONL mirror, action trace saved, and a
    provenance gate replaying sampled records BIT-EXACTLY before the run
    reports success.
  - status heartbeat + snapshot + summary JSON, house atomic-write style.

Usage (smoke):
  OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \\
    python3 body/run_body.py --source acted_synthetic --frames 3000 \\
    --policy curiosity --seed 42 --out-prefix body/results/body_smoke

Boundaries honored: writes only under body/ by default; thread clamps; no
servers; no file outside body/ modified.
"""

import os
import sys
import json
import time
import argparse

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

BODY_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(BODY_DIR)
for p in (REPO_ROOT, os.path.join(REPO_ROOT, "visual"),
          os.path.join(REPO_ROOT, "src"), BODY_DIR):
    if p not in sys.path:
        sys.path.insert(0, p)

import numpy as np
import torch

torch.set_num_threads(1)
try:
    torch.set_num_interop_threads(1)
except RuntimeError:
    pass

from frame_organism import FRAME_SIZE, FRAME_DIM, CHUNK_FRAMES, write_status, append_metric
from action_sources import make_acted_source
from body_organism import (
    ActionConditionedPredictor, BodyStreamer, LearningProgressPolicy,
    RandomPolicy, save_body_snapshot,
)
from causal_records import ActedRecordExtractor, seal_records, verify_records


def frame_to_tensor(frame: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(frame.reshape(-1)).float()


# ═══════════════════════════════════════════════════════════════════════════
#  GIF writers -- train_visual.make_gif's panel discipline, extended with
#  the action dimension (strip / counterfactual columns).
# ═══════════════════════════════════════════════════════════════════════════
def _action_strip(action: int, n_actions: int, width: int, h: int = 6) -> np.ndarray:
    strip = np.full((h, width), 25, dtype=np.uint8)
    cell = width // n_actions
    x0 = action * cell
    strip[1:h - 1, x0 + 1:x0 + cell - 1] = 255
    return strip


def make_life_gif(real_frames, pred_frames, actions, n_actions, out_path,
                  size=FRAME_SIZE, scale=3, fps=10):
    """Rows: real / |real-pred| / pred / action strip. The strip shows the
    action that PRODUCED the transition into this frame -- so 'the view
    jumped left AND the left cell is lit' is checkable frame by frame."""
    from PIL import Image
    assert len(real_frames) == len(pred_frames) == len(actions)
    strip_h = 6
    panel_h = size * 3 + 4 + strip_h + 2
    frames_out = []
    for i in range(len(real_frames)):
        real = (np.clip(real_frames[i], 0, 1) * 255).astype(np.uint8)
        pred = (np.clip(pred_frames[i], 0, 1) * 255).astype(np.uint8)
        diff = (np.clip(np.abs(real_frames[i] - pred_frames[i]), 0, 1) * 255).astype(np.uint8)
        canvas = np.full((panel_h, size), 40, dtype=np.uint8)
        canvas[0:size] = real
        canvas[size + 2:size * 2 + 2] = diff
        canvas[size * 2 + 4:size * 3 + 4] = pred
        canvas[size * 3 + 6:size * 3 + 6 + strip_h] = _action_strip(
            actions[i], n_actions, size, strip_h)
        img = Image.fromarray(canvas, mode="L").resize(
            (size * scale, panel_h * scale), Image.NEAREST)
        frames_out.append(img)
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    frames_out[0].save(out_path, save_all=True, append_images=frames_out[1:],
                       duration=int(1000 / fps), loop=0)
    return out_path


def make_counterfactual_gif(probes, n_actions, out_path, size=FRAME_SIZE,
                            scale=3, fps=4):
    """probes: list of (cf_stack (A, D) np, real_next (H, W) np, taken int).
    Columns: predicted-next under action 0..A-1, then REAL next. The taken
    action's column gets a bright top bar. If the A prediction columns are
    identical, the action has not arrived in the model -- visible at a
    glance, which is the point."""
    from PIL import Image
    frames_out = []
    n_cols = n_actions + 1
    W = n_cols * size + (n_cols - 1) * 2
    H = size + 4
    for cf, real_next, taken in probes:
        canvas = np.full((H, W), 40, dtype=np.uint8)
        for a in range(n_actions):
            img = (np.clip(cf[a].reshape(size, size), 0, 1) * 255).astype(np.uint8)
            x0 = a * (size + 2)
            canvas[4:, x0:x0 + size] = img
            canvas[0:3, x0:x0 + size] = 255 if a == taken else 90
        x0 = n_actions * (size + 2)
        canvas[4:, x0:x0 + size] = (np.clip(real_next, 0, 1) * 255).astype(np.uint8)
        canvas[0:3, x0:x0 + size] = 180
        img = Image.fromarray(canvas, mode="L").resize(
            (W * scale, H * scale), Image.NEAREST)
        frames_out.append(img)
    if not frames_out:
        return None
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    frames_out[0].save(out_path, save_all=True, append_images=frames_out[1:],
                       duration=int(1000 / fps), loop=0)
    return out_path


def main():
    ap = argparse.ArgumentParser(description="Body organism: act, predict, learn, record.")
    ap.add_argument("--source", choices=["acted_synthetic", "vizdoom"],
                    default="acted_synthetic")
    ap.add_argument("--frames", type=int, default=3000)
    ap.add_argument("--policy", choices=["curiosity", "random"], default="curiosity")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--q", type=float, default=0.75)
    ap.add_argument("--d-model", type=int, default=256)
    ap.add_argument("--n-layers", type=int, default=4)
    ap.add_argument("--n-heads", type=int, default=4)
    ap.add_argument("--chunk-frames", type=int, default=CHUNK_FRAMES)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--no-gate", action="store_true")
    ap.add_argument("--gif-every", type=int, default=500)
    ap.add_argument("--gif-window", type=int, default=64)
    ap.add_argument("--cf-frames", type=int, default=24,
                    help="probe states per counterfactual GIF")
    ap.add_argument("--lp-window", type=int, default=60)
    ap.add_argument("--lp-eps", type=float, default=0.10)
    ap.add_argument("--lp-tau", type=float, default=1.0)
    ap.add_argument("--lp-ignition", type=int, default=256)
    ap.add_argument("--out-prefix", type=str,
                    default=os.path.join(BODY_DIR, "results", "body_smoke"))
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device("cpu")
    source = make_acted_source(args.source, args.seed)
    A = source.N_ACTIONS

    model = ActionConditionedPredictor(
        n_actions=A, d_model=args.d_model, n_layers=args.n_layers,
        n_heads=args.n_heads, frame_dim=FRAME_DIM).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    streamer = BodyStreamer(model, opt, device, gate_q=args.q,
                            gate_enabled=not args.no_gate)
    if args.policy == "curiosity":
        policy = LearningProgressPolicy(
            A, window=args.lp_window, eps=args.lp_eps, tau=args.lp_tau,
            ignition_steps=args.lp_ignition, seed=args.seed + 1)
    else:
        policy = RandomPolicy(A, seed=args.seed + 1)

    extractor = ActedRecordExtractor(
        env_name=args.source, base_seed=args.seed,
        action_names=source.ACTION_NAMES, policy_name=args.policy)

    out_prefix = args.out_prefix
    os.makedirs(os.path.dirname(out_prefix) or ".", exist_ok=True)
    status_path = f"{out_prefix}_status.json"
    metrics_path = f"{out_prefix}_metrics.jsonl"
    ckpt_path = f"{out_prefix}_ckpt.pt"
    trace_path = f"{out_prefix}_actions.json"
    records_jsonl = f"{out_prefix}_acted_records.jsonl"
    summary_path = f"{out_prefix}_summary.json"
    store_dir = os.path.join(BODY_DIR, "records",
                             os.path.basename(out_prefix) + "_store")
    gif_dir = os.path.join(BODY_DIR, "gifs")

    print(f"[body] source={args.source} A={A} ({source.ACTION_NAMES}) "
          f"policy={args.policy} frames={args.frames} q={args.q} "
          f"seed={args.seed} gate={not args.no_gate}", flush=True)

    K = args.chunk_frames
    action_counts = [0] * A
    l1_history, sep_history, hit_history = [], [], []
    gif_real, gif_pred, gif_act = [], [], []
    gif_recording, next_gif_at = False, args.gif_every
    cf_probes, gifs_written = [], []
    n_boundaries = 0
    frame_idx = 0
    t0 = time.time()

    prev = source.observe()
    while frame_idx < args.frames:
        xs, acts, ys, metas = [], [], [], []
        for _ in range(K):
            ep0, fi0 = source.episode, source.frame_idx
            a = policy.choose()
            source.act(a)
            cur = source.observe()
            boundary = source.episode != ep0
            n_boundaries += int(boundary)
            action_counts[a] += 1
            xs.append(prev)
            acts.append(a)
            ys.append(cur)
            metas.append((ep0, fi0, boundary, prev, cur))
            prev = cur
        x = torch.stack([frame_to_tensor(f) for f in xs]).unsqueeze(0).to(device)
        act_t = torch.tensor(acts, dtype=torch.long).unsqueeze(0).to(device)
        y = torch.stack([frame_to_tensor(f) for f in ys]).unsqueeze(0).to(device)

        # counterfactual probe from the PRE-chunk state on the chunk's first
        # frame: exact state alignment, three 1-step forwards, cheap.
        pre_states = streamer.states
        cf = model.counterfactual(x[:, :1], pre_states).cpu().numpy()
        pair_d = [float(np.abs(cf[i] - cf[j]).mean())
                  for i in range(A) for j in range(i + 1, A)]
        separation = float(np.mean(pair_d))
        cf_errs = [float(np.abs(cf[a] - y[0, 0].cpu().numpy()).mean())
                   for a in range(A)]
        hit = int(int(np.argmin(cf_errs)) == acts[0])
        sep_history.append(separation)
        hit_history.append(hit)
        if len(cf_probes) < args.cf_frames or frame_idx >= next_gif_at - K:
            cf_probes.append((cf.copy(), ys[0].copy(), acts[0]))
            cf_probes = cf_probes[-args.cf_frames:]

        s, gated, per_frame_l1 = streamer.step_gated(x, act_t, y)
        for i in range(K):
            policy.update(acts[i], float(per_frame_l1[0, i]))
        with torch.no_grad():
            pred_chunk, _ = model(x, act_t, pre_states)
        pred_np = pred_chunk.squeeze(0).cpu().numpy().reshape(K, FRAME_SIZE, FRAME_SIZE)

        for i, (ep0, fi0, boundary, f0, f1) in enumerate(metas):
            extractor.offer(f0, acts[i], f1, ep0, fi0, boundary)

        with torch.no_grad():
            copy_last_l1 = float((x - y).abs().mean())
        l1_history.append(s)
        frame_idx += K

        if not gif_recording and frame_idx >= next_gif_at - K:
            gif_recording = True
        if gif_recording:
            for i in range(K):
                gif_real.append(ys[i])
                gif_pred.append(pred_np[i])
                gif_act.append(acts[i])

        append_metric(metrics_path, {
            "frame_idx": frame_idx, "l1": round(s, 6),
            "copy_last_l1": round(copy_last_l1, 6),
            "gate_rate": round(streamer.stats()["gate_rate"], 4),
            "gated": int(gated),
            "cf_separation": round(separation, 6), "cf_hit": hit,
            "probs": [round(float(p), 4) for p in policy.probs()],
            "lp": [None if v is None else round(v, 6)
                   for v in policy.lp_estimates()],
            "action_counts": list(action_counts),
        })

        if gif_recording and len(gif_real) >= args.gif_window:
            tag = f"{args.source}_{args.policy}_seed{args.seed}_f{frame_idx}"
            p1 = make_life_gif(gif_real[:args.gif_window],
                               gif_pred[:args.gif_window],
                               gif_act[:args.gif_window], A,
                               os.path.join(gif_dir, f"body_{tag}.gif"))
            p2 = make_counterfactual_gif(
                list(cf_probes), A,
                os.path.join(gif_dir, f"body_cf_{tag}.gif"))
            gifs_written += [p1] + ([p2] if p2 else [])
            print(f"[body] GIFs at frame {frame_idx}: {p1}"
                  + (f" | {p2}" if p2 else ""), flush=True)
            gif_real, gif_pred, gif_act = [], [], []
            gif_recording = False
            next_gif_at += args.gif_every

        if frame_idx % (K * 12) == 0 or frame_idx >= args.frames:
            write_status(status_path, {
                "frame_idx": frame_idx, "target_frames": args.frames,
                "n_chunks": streamer.n_chunks, "n_bwd": streamer.n_bwd,
                "gate_rate": round(streamer.stats()["gate_rate"], 4),
                "last_l1": round(s, 6),
                "cf_separation": round(separation, 6),
                "hit_rate_recent": round(float(np.mean(hit_history[-50:])), 4),
                "action_counts": list(action_counts),
                "n_records": len(extractor.records),
                "wall_s": round(time.time() - t0, 1),
                "source": args.source, "policy": args.policy,
            })

    source.close()

    # ── action trace (the provenance carrier under a learned policy) ──────
    trace = source.trace
    with open(trace_path, "w") as f:
        json.dump({"env": args.source, "base_seed": args.seed,
                   "policy": args.policy,
                   "trace": trace if isinstance(trace, list)
                   else {str(k): v for k, v in trace.items()}}, f)

    # ── seal + verify acted records ───────────────────────────────────────
    seg_sha = seal_records(extractor.records, store_dir, records_jsonl)
    verify_trace = trace if isinstance(trace, list) else {
        int(k): v for k, v in trace.items()}
    prov = verify_records(extractor.records, verify_trace, n_samples=5,
                          seed=args.seed)

    save_body_snapshot(ckpt_path, model, opt, streamer, frame_idx, args.seed,
                       time.time() - t0,
                       extra={"policy": args.policy,
                              "action_counts": action_counts})

    # ── summary ───────────────────────────────────────────────────────────
    n_ch = len(l1_history)
    late = slice(max(0, int(n_ch * 0.8)), n_ch)
    early_probe = slice(0, max(1, min(20, n_ch)))
    summary = {
        "source": args.source, "policy": args.policy, "seed": args.seed,
        "frames": frame_idx, "n_chunks": n_ch,
        "gate": streamer.stats(),
        "l1_first10": [round(v, 5) for v in l1_history[:10]],
        "l1_last10": [round(v, 5) for v in l1_history[-10:]],
        "l1_late_mean": round(float(np.mean(l1_history[late])), 6),
        "cf_separation_early": round(float(np.mean(sep_history[early_probe])), 6),
        "cf_separation_late": round(float(np.mean(sep_history[late])), 6),
        "cf_hit_rate_late": round(float(np.mean(hit_history[late])), 4),
        "cf_hit_chance": round(1.0 / A, 4),
        "action_counts": action_counts,
        "final_probs": [round(float(p), 4) for p in policy.probs()],
        "final_lp": [None if v is None else round(v, 6)
                     for v in policy.lp_estimates()],
        "episode_boundaries": n_boundaries,
        "acted_records": extractor.stats(),
        "records_segment_sha": seg_sha,
        "provenance": prov,
        "wall_s": round(time.time() - t0, 1),
        "gifs": gifs_written,
    }
    with open(summary_path + f".tmp{os.getpid()}", "w") as f:
        json.dump(summary, f, indent=2)
    os.replace(summary_path + f".tmp{os.getpid()}", summary_path)

    print("=" * 70)
    print(f"[body] DONE {frame_idx} frames | l1 late {summary['l1_late_mean']} "
          f"| sep {summary['cf_separation_early']} -> {summary['cf_separation_late']} "
          f"| hit {summary['cf_hit_rate_late']} (chance {summary['cf_hit_chance']}) "
          f"| records {len(extractor.records)} (prov {prov['exact']}) "
          f"| actions {action_counts}", flush=True)
    print(f"[body] summary: {summary_path}")


if __name__ == "__main__":
    main()
