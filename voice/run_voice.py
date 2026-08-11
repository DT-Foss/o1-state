#!/usr/bin/env python3 -u
"""
RUN VOICE -- the full circle: A lives and acts; A's records teach a speaker
to SAY what happened from vision alone; a fresh body B HEARS the sentences,
imagines them onto its own view, and gets measurably better at the world it
then lives in. Language as the carrier of embodied knowledge between
bodies, end to end, with provenance at every joint.

Stages (default --stage all, artifacts persist between stages):

  life     A = RichPanSource(seed A) under a seeded RANDOM walk (the
           measured lesson from P88: random exposure teaches the action
           channel fastest -- A's job is to generate honest experience,
           not to be clever). The model-free extractor
           (body/causal_records.py, read-only) turns acted steps into
           records; the FULL trace is saved (provenance under any policy).
           No body model is trained here -- nothing in the transmission
           claim needs A to have learned anything; A needs only to have
           LIVED, and the records need only to be true (P90a discipline).

  speak    Labels = A's own records (SILENCE elsewhere; action slot from
           the true trace everywhere -- the speaker is honestly asked to
           name the press even where it cannot see it, P91d). One
           streaming pass, compositional holdout (right,2) loss-masked.
           Instruments: held-out accuracies, composition, intervention
           flip, epistemics ceiling. Writes the speaker's own narration of
           the whole life (utterances incl. errors) for the hear_learned
           arm, and a narrated-life GIF.

  transmit The paired design: ONE simulated B-life (world + predrawn
           random actions identical across arms -- arms differ ONLY in
           what they heard while sitting still for the first
           LISTEN_FRAMES holds):
             silent        heard nothing
             hear_codec    ground-truth sentences of A's records
             hear_learned  the trained speaker's actual narration,
                           errors included
             scrambled     consistently lied-to (left<->right swapped,
                           magnitudes permuted) -- the poison control
           Hearing arms convert sentences to imagined pairs
           (imagination.py) and pretrain the SAME body class
           (body/body_organism.ActionConditionedPredictor, identical init
           weights across arms) before life; then everyone lives the SAME
           4000 real frames through the gated BodyStreamer. Probes on a
           FIXED scripted route in a SEPARATE probe world at every
           checkpoint -- no arm is graded on a test it chose (the P89c
           selection confound, closed).

Boundaries: writes only under voice/; body/, visual/, src/ imported
read-only; thread clamps; no servers.
"""

import os
import sys
import json
import time
import argparse

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

VOICE_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(VOICE_DIR)
for p in (REPO_ROOT, os.path.join(REPO_ROOT, "visual"),
          os.path.join(REPO_ROOT, "body"), os.path.join(REPO_ROOT, "src"),
          VOICE_DIR):
    if p not in sys.path:
        sys.path.insert(0, p)

import numpy as np
import torch

torch.set_num_threads(1)
try:
    torch.set_num_interop_threads(1)
except RuntimeError:
    pass

from frame_organism import FRAME_SIZE, write_status                 # read-only
from body_organism import ActionConditionedPredictor, BodyStreamer  # read-only
from causal_records import (ActedRecordExtractor, seal_records, frame_sha,
                            estimate_shift)
from vocab import (RichPanSource, labels_for_life, sentence_from_dx,
                   render_sentence, ACTIONS)
from speaker import (train_speaker, utter_life, eval_speaker,
                     intervention_flip, HOLDOUT_SENTENCE)
from imagination import (utterances_to_pairs, imagination_pretrain,
                         scramble_utterances)

RESULTS = os.path.join(VOICE_DIR, "results")
LISTEN_FRAMES = 120
IMAG_CAP = 1500          # imagined pairs, identical for every hearing arm
B_MODEL_SEED = 1234      # identical init weights across arms, by construction


# ─────────────────────────────────────────────────────────────────────────
def live_world(seed: int, actions) -> np.ndarray:
    """(n+1, H, W) frames of a RichPanSource life under a given action
    list -- the same replay primitive provenance uses, reused as the
    simulator so life and replay are identical by construction."""
    return np.stack(RichPanSource.replay_frames(seed, actions), axis=0)


def verify_rich_records(records, actions, n_samples=5, seed=7):
    """The provenance gate for the RICH world. body/'s verify_records
    replays its own two envs by name; these records live in voice/'s
    subclassed world, so the replay MUST use RichPanSource -- the debug
    smoke caught exactly this class mismatch as 0/5 hash fails. Same
    checks, same strictness: both frame hashes bit-exact AND dx
    re-detected, via the same frame_sha/estimate_shift primitives."""
    if not records:
        return {"picks": [], "exact": "0/0", "pass": False}
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(records), size=min(n_samples, len(records)),
                     replace=False)
    frames_cache = {}
    picks = []
    for i in idx:
        q = records[int(i)]["quote"]
        fr = q["frame"]
        key = q["base_seed"]
        if key not in frames_cache:
            frames_cache[key] = RichPanSource.replay_frames(key, actions)
        fs = frames_cache[key]
        sha_ok = (frame_sha(fs[fr]) == q["frame_sha256_pre"]
                  and frame_sha(fs[fr + 1]) == q["frame_sha256_post"])
        s, _, _ = estimate_shift(fs[fr], fs[fr + 1])
        picks.append({"frame": fr, "exact": bool(sha_ok),
                      "dx_redetected": bool(s == q["dx_measured"])})
        print(f"[provenance] frame {fr} {q['action_name']} dx "
              f"{q['dx_measured']:+d} -> "
              f"{'BIT-EXACT' if sha_ok else 'HASH MISMATCH'}", flush=True)
    n_ok = sum(1 for p in picks if p["exact"] and p["dx_redetected"])
    return {"picks": picks, "exact": f"{n_ok}/{len(picks)}",
            "pass": bool(picks) and n_ok == len(picks)}


def stage_life(args):
    rng = np.random.default_rng(args.seed + 1)
    actions = [int(a) for a in rng.integers(0, 3, size=args.frames)]
    frames = live_world(args.seed, actions)
    ex = ActedRecordExtractor("voice_rich_pan", base_seed=args.seed,
                              action_names=RichPanSource.ACTION_NAMES,
                              policy_name="random")
    for t, a in enumerate(actions):
        ex.offer(frames[t], a, frames[t + 1], 0, t, crossed_boundary=False)
    prov = verify_rich_records(ex.records, actions, n_samples=5, seed=args.seed)
    os.makedirs(RESULTS, exist_ok=True)
    with open(os.path.join(RESULTS, "a_life.json"), "w") as f:
        json.dump({"seed": args.seed, "frames": args.frames,
                   "trace": actions, "n_records": len(ex.records),
                   "provenance": prov}, f)
    with open(os.path.join(RESULTS, "a_records.jsonl"), "w") as f:
        for r in ex.records:
            f.write(json.dumps(r, sort_keys=True) + "\n")
    seg = seal_records(ex.records,
                       os.path.join(VOICE_DIR, "records", "a_life_store"), None)
    print(f"[life] {args.frames} steps, {len(ex.records)} records, "
          f"provenance {prov['exact']}, segment {str(seg)[:12]}…", flush=True)
    return actions, frames, ex.records


def _load_life():
    d = json.load(open(os.path.join(RESULTS, "a_life.json")))
    records = [json.loads(l) for l in open(os.path.join(RESULTS, "a_records.jsonl"))]
    frames = live_world(d["seed"], d["trace"])
    return d["trace"], frames, records


def narrate_gif(frames, said, truth, out_path, n=48, start=0, scale=3, fps=6):
    """Top: the lived frame. Below: 'said: … / truth: …' -- the speaker's
    narration against the record's word, frame by frame."""
    from PIL import Image, ImageDraw
    H = FRAME_SIZE * scale
    W = FRAME_SIZE * scale
    imgs = []
    for t in range(start, min(start + n, len(said[0]))):
        fr = (np.clip(frames[t + 1], 0, 1) * 255).astype(np.uint8)
        img = Image.fromarray(fr, "L").resize((W, H), Image.NEAREST).convert("L")
        canvas = Image.new("L", (W, H + 30), 20)
        canvas.paste(img, (0, 0))
        dr = ImageDraw.Draw(canvas)
        dr.text((3, H + 2), "said: " + render_sentence(said[0][t], said[1][t], said[2][t]),
                fill=255)
        dr.text((3, H + 16), "trth: " + render_sentence(truth[0][t], truth[1][t], truth[2][t]),
                fill=180)
        imgs.append(canvas)
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    imgs[0].save(out_path, save_all=True, append_images=imgs[1:],
                 duration=int(1000 / fps), loop=0)
    return out_path


def stage_speak(args, life=None):
    actions, frames, records = life if life else _load_life()
    n = len(actions)
    d_lab, m_lab, _ = labels_for_life(n, records)
    a_lab = np.array(actions, dtype=np.int64)  # true press everywhere (P91d)
    train_upto = int(n * 0.85)
    model, loss_trace = train_speaker(frames, (d_lab, m_lab, a_lab),
                                      train_upto, seed=args.seed)
    said = utter_life(model, frames)
    ev = eval_speaker(said, (d_lab, m_lab, a_lab), heldout_from=train_upto)
    flip = intervention_flip(model, frames[train_upto:], n_probes=60,
                             mag=2, seed=args.seed)
    gif = narrate_gif(frames, said, (d_lab, m_lab, a_lab),
                      os.path.join(VOICE_DIR, "gifs", "voice_narrated_life.gif"),
                      start=train_upto)
    torch.save(model.state_dict(), os.path.join(RESULTS, "speaker_ckpt.pt"))
    np.savez_compressed(os.path.join(RESULTS, "speaker_said.npz"),
                        dir=said[0], mag=said[1], act=said[2])
    out = {"train_upto": train_upto, "loss_first5": [round(v, 4) for v in loss_trace[:5]],
           "loss_last5": [round(v, 4) for v in loss_trace[-5:]],
           "eval": ev, "intervention": flip, "holdout_sentence": HOLDOUT_SENTENCE,
           "narrate_gif": gif}
    with open(os.path.join(RESULTS, "speaker_summary.json"), "w") as f:
        json.dump(out, f, indent=2)
    print(f"[speak] heldout dir {ev.get('dir_acc'):.3f} mag {ev.get('mag_acc'):.3f} "
          f"| comp {ev.get('comp_sentence_acc')} (n={ev.get('n_comp_holdout')}) "
          f"| act cons {ev.get('act_acc_consequence'):.3f} vs silent "
          f"{ev.get('act_acc_silent'):.3f} (prior {ev.get('act_prior_silent'):.3f}) "
          f"| flip {flip['flip_rate']:.2f}", flush=True)
    return model, said, (d_lab, m_lab, a_lab)


# ─────────────────────────────────────────────────────────────────────────
@torch.no_grad()
def probe_eval(model, pframes, pactions):
    """Fixed-route probe: stepwise carry; per step the counterfactual over
    all actions vs the real next frame (hit), and the L1 of the prediction
    under the ACTUAL scripted action. Same route, same world, every arm,
    every checkpoint."""
    from frame_organism import detach_state_tree
    states = None
    hits, l1s = [], []
    for t, a in enumerate(pactions):
        x = torch.from_numpy(pframes[t].reshape(1, 1, -1).astype(np.float32))
        y = pframes[t + 1].reshape(-1)
        cf = model.counterfactual(x, states).numpy()
        errs = [float(np.abs(cf[k] - y).mean()) for k in range(cf.shape[0])]
        # tie guard: a model whose counterfactuals don't differ has NO
        # preference -- that is a miss, not a lucky argmin (zero-init
        # models would otherwise score freq(action 0) at frame 0).
        if max(errs) - min(errs) < 1e-9:
            hits.append(0)
        else:
            hits.append(int(int(np.argmin(errs)) == a))
        l1s.append(errs[a])
        _, states = model(x, torch.tensor([[a]], dtype=torch.long), states)
        states = detach_state_tree(states)
    return float(np.mean(l1s)), float(np.mean(hits))


def make_b_model():
    torch.manual_seed(B_MODEL_SEED)          # identical init across arms
    model = ActionConditionedPredictor(n_actions=3)
    opt = torch.optim.Adam(model.parameters(), lr=3e-4)
    return model, opt


def stage_transmit(args, said=None):
    actions_a, frames_a, records = _load_life()
    if said is None:
        z = np.load(os.path.join(RESULTS, "speaker_said.npz"))
        said = (z["dir"], z["mag"], z["act"])

    # utterance streams (dir, mag) in life order
    utt_codec = [sentence_from_dx(r["quote"]["dx_measured"]) for r in records]
    utt_learned = [(int(d), int(m)) for d, m in zip(said[0], said[1])]
    utt_scrambled = scramble_utterances(utt_codec, seed=args.seed)

    # ONE simulated B-life reused by every arm (paired design)
    rngb = np.random.default_rng(args.b_seed + 1)
    b_actions = [2] * LISTEN_FRAMES + [int(a) for a in
                 rngb.integers(0, 3, size=args.b_frames)]
    b_frames = live_world(args.b_seed, b_actions)
    listen_frames = b_frames[:LISTEN_FRAMES]

    # fixed probe route in a SEPARATE world
    rngp = np.random.default_rng(9009)
    p_actions = [int(a) for a in rngp.integers(0, 3, size=args.probe_len)]
    p_frames = live_world(999, p_actions)

    arms = {"silent": None, "hear_codec": utt_codec,
            "hear_learned": utt_learned, "scrambled": utt_scrambled}
    results = {}
    for arm, utt in arms.items():
        t0 = time.time()
        model, opt = make_b_model()
        n_imag = 0
        if utt is not None:
            pairs = utterances_to_pairs(utt, listen_frames, cap=IMAG_CAP,
                                        seed=args.seed)
            n_imag = imagination_pretrain(model, opt, pairs)
        streamer = BodyStreamer(model, opt, torch.device("cpu"))
        checkpoints = []
        l1, hit = probe_eval(model, p_frames, p_actions)
        checkpoints.append({"frame": 0, "probe_l1": round(l1, 6),
                            "probe_hit": round(hit, 4)})
        K = 16
        life_a = b_actions[LISTEN_FRAMES:]
        life_f = b_frames[LISTEN_FRAMES:]
        for c0 in range(0, len(life_a) - K, K):
            x = torch.from_numpy(
                life_f[c0:c0 + K].reshape(K, -1).astype(np.float32))[None]
            y = torch.from_numpy(
                life_f[c0 + 1:c0 + K + 1].reshape(K, -1).astype(np.float32))[None]
            acts = torch.tensor(life_a[c0:c0 + K], dtype=torch.long)[None]
            streamer.step_gated(x, acts, y)
            fr = c0 + K
            if fr % args.probe_every < K and fr >= args.probe_every:
                l1, hit = probe_eval(model, p_frames, p_actions)
                checkpoints.append({"frame": fr, "probe_l1": round(l1, 6),
                                    "probe_hit": round(hit, 4)})
        ign = next((c["frame"] for c in checkpoints if c["probe_hit"] >= 0.45), None)
        results[arm] = {"n_imagined_chunks": n_imag,
                        "checkpoints": checkpoints,
                        "ignition_frame_hit045": ign,
                        "gate": streamer.stats(),
                        "wall_s": round(time.time() - t0, 1)}
        print(f"[transmit:{arm}] imag {n_imag} chunks | probe hit "
              f"{checkpoints[0]['probe_hit']:.3f} -> {checkpoints[-1]['probe_hit']:.3f} "
              f"| L1 {checkpoints[0]['probe_l1']:.4f} -> {checkpoints[-1]['probe_l1']:.4f} "
              f"| ignition {ign} | {results[arm]['wall_s']}s", flush=True)
        write_status(os.path.join(RESULTS, "transmit_status.json"),
                     {"done_arms": list(results.keys())})

    out = {"b_seed": args.b_seed, "b_frames": args.b_frames,
           "listen_frames": LISTEN_FRAMES, "imag_cap": IMAG_CAP,
           "probe_world_seed": 999, "probe_len": args.probe_len,
           "arms": results}
    with open(os.path.join(RESULTS, "transmission_summary.json"), "w") as f:
        json.dump(out, f, indent=2)
    print(f"[transmit] -> {os.path.join(RESULTS, 'transmission_summary.json')}",
          flush=True)
    return out


def main():
    ap = argparse.ArgumentParser(description="voice/: the world book, spoken.")
    ap.add_argument("--stage", choices=["life", "speak", "transmit", "all"],
                    default="all")
    ap.add_argument("--seed", type=int, default=42)        # A's world
    ap.add_argument("--frames", type=int, default=8000)    # A's life
    ap.add_argument("--b-seed", type=int, default=777)     # B's world
    ap.add_argument("--b-frames", type=int, default=4000)  # B's real life
    ap.add_argument("--probe-every", type=int, default=500)
    ap.add_argument("--probe-len", type=int, default=96)
    args = ap.parse_args()

    life = said = None
    if args.stage in ("life", "all"):
        life = stage_life(args)
    if args.stage in ("speak", "all"):
        _, said, _ = stage_speak(args, life)
    if args.stage in ("transmit", "all"):
        stage_transmit(args, said)


if __name__ == "__main__":
    main()
