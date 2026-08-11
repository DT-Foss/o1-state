#!/usr/bin/env python3
"""
TRAIN VISUAL -- CLI entry point for the visual organism, Phase 1.

Streams frames from a source (SyntheticSource always; VizDoomSource if
vizdoom is importable and initializes headless), chunks them (CHUNK_FRAMES
per chunk), trains a FramePredictor via FrameStreamer's surprise-gated
chunk loop (F1/F2 inherited from hsslm/neural/streaming.py's pattern), and
writes:

  - a checkpoint (save_snapshot-style, atomic tmpfile + os.replace)
  - a status.json heartbeat (pos_run-style)
  - a metrics.jsonl (one record per frame chunk: frame_idx, l1, gate_rate,
    gated)
  - every 500 frames, a GIF: top row real frames, bottom row this model's
    one-step predictions for the SAME time window, plus a difference-image
    panel -- so degeneracy (a predictor that just outputs a gray blur) is
    visible immediately, not just implied by the L1 number.

Usage:
  OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \\
    python3 visual/train_visual.py --source synthetic --frames 3000 --q 0.75 \\
    --seed 42 --out-prefix results/visual_smoke
"""

import os
import sys
import json
import time
import argparse

# Thread clamps BEFORE torch import -- this machine runs a saturating SAT
# fleet alongside everything else (o1-maschine-last-threads convention);
# MPS is a separate device queue and is unaffected by these CPU clamps.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch

torch.set_num_threads(1)
try:
    torch.set_num_interop_threads(1)
except RuntimeError:
    pass

from frame_sources import SyntheticSource, VizDoomSource, FRAME_SIZE
from frame_organism import (
    FramePredictor, FrameStreamer, CHUNK_FRAMES, save_snapshot, write_status,
    append_metric,
)

FRAME_DIM = FRAME_SIZE * FRAME_SIZE


def pick_device(requested: str = "auto") -> torch.device:
    if requested == "cpu":
        return torch.device("cpu")
    if requested == "mps":
        return torch.device("mps")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def make_source(name: str, seed: int):
    if name == "synthetic":
        return SyntheticSource(seed=seed)
    if name == "vizdoom":
        if not VizDoomSource.available():
            print("[train_visual] VizDoomSource requested but NOT AVAILABLE "
                  "(vizdoom import or headless init failed) -- falling back "
                  "is NOT performed automatically; exiting.", flush=True)
            sys.exit(1)
        return VizDoomSource(seed=seed)
    raise ValueError(f"unknown source: {name}")


def frame_to_tensor(frame: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(frame.reshape(-1)).float()


def make_gif(real_frames, pred_frames, out_path, size=FRAME_SIZE, scale=3, fps=10):
    """real_frames, pred_frames: list of (size, size) float32 arrays in
    [0, 1], same length, same time window. Writes a 3-row GIF: real frames
    on top, this model's one-step predictions of that same window on the
    bottom, and |real - pred| as a difference panel in the middle-bottom
    third -- so blur/degeneracy is visible directly, not just inferred from
    a number.
    """
    from PIL import Image

    assert len(real_frames) == len(pred_frames)
    n = len(real_frames)
    panel_h = size * 3 + 4  # real / diff / pred, with 2px separators
    frames_out = []
    for i in range(n):
        real = (np.clip(real_frames[i], 0, 1) * 255).astype(np.uint8)
        pred = (np.clip(pred_frames[i], 0, 1) * 255).astype(np.uint8)
        diff = (np.clip(np.abs(real_frames[i] - pred_frames[i]), 0, 1) * 255).astype(np.uint8)

        canvas = np.full((panel_h, size), 40, dtype=np.uint8)  # dark gray separators
        canvas[0:size, :] = real
        canvas[size + 2:size * 2 + 2, :] = diff
        canvas[size * 2 + 4:size * 3 + 4, :] = pred

        img = Image.fromarray(canvas, mode="L").resize(
            (size * scale, panel_h * scale), Image.NEAREST
        )
        frames_out.append(img)

    d = os.path.dirname(out_path) or "."
    os.makedirs(d, exist_ok=True)
    duration_ms = int(1000 / fps)
    frames_out[0].save(
        out_path, save_all=True, append_images=frames_out[1:],
        duration=duration_ms, loop=0,
    )
    return out_path


def main():
    ap = argparse.ArgumentParser(description="Visual organism Phase 1: frame-stream predictor.")
    ap.add_argument("--source", choices=["synthetic", "vizdoom"], default="synthetic")
    ap.add_argument("--frames", type=int, default=3000)
    ap.add_argument("--q", type=float, default=0.75, help="gate quantile (GATE_Q)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out-prefix", type=str, default="results/visual_smoke")
    ap.add_argument("--d-model", type=int, default=256)
    ap.add_argument("--n-layers", type=int, default=4)
    ap.add_argument("--n-heads", type=int, default=4)
    ap.add_argument("--chunk-frames", type=int, default=CHUNK_FRAMES)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--gif-every", type=int, default=500)
    ap.add_argument("--gif-window", type=int, default=80, help="frames per GIF (~8s at 10fps)")
    ap.add_argument("--device", choices=["auto", "cpu", "mps"], default="auto")
    ap.add_argument("--residual", action="store_true",
                     help="decoder predicts a delta; output = clamp(input + delta, 0, 1) "
                          "instead of sigmoid(decoder(h)) directly")
    ap.add_argument("--no-gate", action="store_true",
                     help="disable the surprise gate -- every chunk is trained (ablation arm)")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    device = pick_device(args.device)
    print(f"[train_visual] device={device} source={args.source} frames={args.frames} "
          f"q={args.q} seed={args.seed} residual={args.residual} gate={not args.no_gate}",
          flush=True)

    source = make_source(args.source, args.seed)

    model = FramePredictor(d_model=args.d_model, n_layers=args.n_layers,
                            n_heads=args.n_heads, frame_dim=FRAME_DIM,
                            residual=args.residual).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    streamer = FrameStreamer(model, opt, device, gate_q=args.q,
                              gate_enabled=not args.no_gate)

    out_prefix = args.out_prefix
    status_path = f"{out_prefix}_status.json"
    metrics_path = f"{out_prefix}_metrics.jsonl"
    ckpt_path = f"{out_prefix}_ckpt.pt"
    gif_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "gifs")
    os.makedirs(gif_dir, exist_ok=True)

    K = args.chunk_frames
    buf = []  # pending raw frames (np arrays) for building the next chunk
    gif_buf_real = []  # window of real frames being assembled for the next GIF
    gif_buf_pred = []  # window of predicted frames (aligned to real)
    gif_recording = False  # True while accumulating a --gif-window window
    gifs_written = []
    next_gif_at = args.gif_every  # frame_idx at which to START the next window

    l1_history = []
    frame_idx = 0
    t0 = time.time()

    def build_chunk_xy(frame_list):
        # frame_list has K+1 raw frames -> x = frames[:K], y = frames[1:K+1]
        xs = torch.stack([frame_to_tensor(f) for f in frame_list[:-1]], dim=0)
        ys = torch.stack([frame_to_tensor(f) for f in frame_list[1:]], dim=0)
        return xs.unsqueeze(0).to(device), ys.unsqueeze(0).to(device)  # (1, K, D)

    while frame_idx < args.frames:
        # Fill buffer to K+1 frames (K inputs, K targets via 1-frame shift).
        while len(buf) < K + 1 and frame_idx + len(buf) < args.frames + 1:
            buf.append(source.next_frame())
        if len(buf) < 2:
            break  # not enough frames left for even a 1-step chunk

        chunk_frames = buf[: min(K + 1, len(buf))]
        n_pred = len(chunk_frames) - 1
        x, y = build_chunk_xy(chunk_frames)

        # step_gated doesn't return prediction frames (its contract mirrors
        # HSSLMStreamer's, which returns nll not logits), so for GIF fidelity
        # we recompute a no-grad forward from the PRE-chunk state snapshot,
        # taken before step_gated advances streamer.states. Negligible extra
        # cost at this scale.
        pre_chunk_states = streamer.states
        s, gated, per_frame_l1 = streamer.step_gated(x, y)
        with torch.no_grad():
            pred_chunk, _ = model(x, pre_chunk_states)

        pred_np = pred_chunk.squeeze(0).cpu().numpy().reshape(n_pred, FRAME_SIZE, FRAME_SIZE)
        real_np = np.stack(chunk_frames[1:], axis=0)  # targets, aligned with pred

        # GIF cadence: every --gif-every frames, record ONE --gif-window-long
        # window (not every window-length span) -- so a 3000-frame run with
        # gif-every=500/gif-window=80 writes ~6 GIFs, not ~37.
        if not gif_recording and frame_idx >= next_gif_at - n_pred:
            gif_recording = True
        if gif_recording:
            for i in range(n_pred):
                gif_buf_real.append(real_np[i])
                gif_buf_pred.append(pred_np[i])

        # copy-last baseline for THIS chunk, mirroring what the model was
        # judged against -- x[:, t] predicting y[:, t] with zero work, so
        # every arm's metrics.jsonl carries its own baseline comparison
        # point, not just a number quoted from a separate script.
        with torch.no_grad():
            copy_last_l1 = float((x - y).abs().mean())

        l1_history.append(s)
        frame_idx += n_pred
        buf = buf[n_pred:]

        append_metric(metrics_path, {
            "frame_idx": frame_idx, "l1": round(s, 6),
            "copy_last_l1": round(copy_last_l1, 6),
            "gate_rate": round(streamer.stats()["gate_rate"], 4),
            "gated": int(gated),
        })

        if gif_recording and len(gif_buf_real) >= args.gif_window:
            gif_path = os.path.join(gif_dir, f"visual_{args.source}_seed{args.seed}_f{frame_idx}.gif")
            make_gif(gif_buf_real[: args.gif_window], gif_buf_pred[: args.gif_window], gif_path)
            gifs_written.append(gif_path)
            print(f"[train_visual] wrote GIF at frame {frame_idx}: {gif_path}", flush=True)
            gif_buf_real = []
            gif_buf_pred = []
            gif_recording = False
            next_gif_at += args.gif_every

        if frame_idx % 200 == 0 or frame_idx >= args.frames:
            write_status(status_path, {
                "frame_idx": frame_idx, "target_frames": args.frames,
                "n_chunks": streamer.n_chunks, "n_bwd": streamer.n_bwd,
                "gate_rate": round(streamer.stats()["gate_rate"], 4),
                "last_l1": round(s, 6), "wall_s": round(time.time() - t0, 1),
                "device": str(device), "source": args.source,
            })

    save_snapshot(ckpt_path, model, opt, streamer, frame_idx, args.seed,
                  time.time() - t0)

    wall_s = time.time() - t0
    first10 = l1_history[:10]
    last10 = l1_history[-10:]
    warmup = args.gif_window if len(l1_history) > 40 else 0
    post_warmup_rate = None
    print("=" * 70)
    print(f"[train_visual] DONE: {frame_idx} frames, {streamer.n_chunks} chunks, "
          f"{streamer.n_bwd} gated (gate_rate={streamer.stats()['gate_rate']:.3f}), "
          f"wall={wall_s:.1f}s, device={device}")
    print(f"[train_visual] L1 first10: {[round(v,4) for v in first10]}")
    print(f"[train_visual] L1 last10:  {[round(v,4) for v in last10]}")
    print(f"[train_visual] GIFs written: {gifs_written}")
    print(f"[train_visual] checkpoint: {ckpt_path}")
    print(f"[train_visual] status: {status_path}")
    print(f"[train_visual] metrics: {metrics_path}")


if __name__ == "__main__":
    main()
