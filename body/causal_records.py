"""
CAUSAL RECORDS FROM ACTED EVENTS -- Stufe 2: the eye writes into the world
book, and what it writes is something it DID.

The living causal file so far holds read-events: text triplets whose quote
is a stream coordinate. This module extracts ACTED events -- "pressed left |
view shifted | dx=-3" -- where trigger is the organism's own action, the
mechanism is a measured visual consequence, and the quote is a frame
coordinate pair plus content hashes. Records are shaped in the livecausal
key vocabulary (trigger_key / mechanism / outcome_key, lowercase strings --
see src/livecausal/evidence.py's identity note: base-edge identity is
(trigger_key, outcome_key), mechanism annotates) and sealed via the
UNTOUCHED segment store (src/livecausal/store.py, imported read-only) into
a store directory UNDER body/ -- drafts, never merged into any existing
store by this code. A plain-JSONL mirror sits next to the store for humans.

Consequence detection is deliberately the dumbest thing that is honest: a
global horizontal shift estimator. estimate_shift(f0, f1) finds the column
offset s in [-max_dx, +max_dx] minimizing mean |f1(x) - f0(x + s)| over the
overlap, and reports (s, err_at_s, err_at_0). A consequence is claimed only
if the best shift explains the frame pair MUCH better than "nothing moved"
(err_at_s < rel_improve * err_at_0) AND |s| >= 1. This catches exactly the
pan/strafe consequences both acted worlds actually have; it does NOT
pretend to a general theory of visual causation. Episode-straddling pairs
are skipped (a world reset is not a consequence of the action). ATTACK/HOLD
consequences (muzzle flash, nothing) fall below the shift detector by
design -- absence of a record is the honest output there for now.

Provenance rule (the P48/P52 discipline, extended to policies): because a
learned policy's action sequence is not derivable from any seed, every
record's quote carries (env, base_seed, episode, frame, action) AND the
sha256 of both frames; the run saves the full action trace alongside. To
verify: replay the world under the recorded trace (action_sources.py's
replay_*), hash the two frames at the recorded coordinate, compare, and
re-run the estimator. verify_records() does exactly that -- bit-exact or
FAIL, no fuzzy matching.
"""

import hashlib
import json
import os
import sys
from typing import List, Optional, Tuple

import numpy as np

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for p in (REPO_ROOT, os.path.join(REPO_ROOT, "src")):
    if p not in sys.path:
        sys.path.insert(0, p)

from livecausal.store import LiveStore  # read-only import; store dir lives under body/

RECORD_VERSION = 1


def frame_sha(frame: np.ndarray) -> str:
    """sha256 over the frame's float32 bytes -- the bit-exactness anchor.
    Frames are float32 [0,1] by the acted-source contract; tobytes() of the
    C-contiguous array is the canonical serialization."""
    return hashlib.sha256(
        np.ascontiguousarray(frame, dtype=np.float32).tobytes()).hexdigest()


def estimate_shift(f0: np.ndarray, f1: np.ndarray,
                   max_dx: int = 5) -> Tuple[int, float, float]:
    """Best global horizontal shift s: f1[:, x] ~= f0[:, x + s].

    Returns (s, err_at_s, err_at_0); errors are mean L1 over the shifted
    overlap. Sign convention matches ActedSyntheticSource: viewport moved
    RIGHT by d  ->  s = +d. Column-crop comparison, no wraparound."""
    best_s, best_err, err0 = 0, None, None
    W = f0.shape[1]
    for s in range(-max_dx, max_dx + 1):
        if s >= 0:
            a, b = f0[:, s:], f1[:, :W - s] if s > 0 else f1
        else:
            a, b = f0[:, :W + s], f1[:, -s:]
        err = float(np.mean(np.abs(a - b)))
        if s == 0:
            err0 = err
        if best_err is None or err < best_err:
            best_s, best_err = s, err
    return best_s, best_err, err0


class ActedRecordExtractor:
    """Watches (frame, action, next_frame) tuples from the run loop and
    accumulates acted-event records. Nothing here touches the model -- the
    extractor reads the same stream the trainer reads."""

    def __init__(self, env_name: str, base_seed: int, action_names: List[str],
                 rel_improve: float = 0.6, max_dx: int = 5,
                 policy_name: str = "curiosity"):
        self.env = env_name
        self.base_seed = base_seed
        self.action_names = action_names
        self.rel_improve = rel_improve
        self.max_dx = max_dx
        self.policy_name = policy_name
        self.records: List[dict] = []
        self.n_seen = 0
        self.n_skipped_boundary = 0

    def offer(self, f0: np.ndarray, action: int, f1: np.ndarray,
              episode: int, frame_idx: int,
              crossed_boundary: bool) -> Optional[dict]:
        """One acted step. frame_idx is f0's index within its episode (the
        action recorded at trace[episode][frame_idx] produced f1). Returns
        the record if a consequence was detected, else None."""
        self.n_seen += 1
        if crossed_boundary:
            self.n_skipped_boundary += 1
            return None
        s, err_s, err_0 = estimate_shift(f0, f1, self.max_dx)
        if s == 0 or err_0 <= 1e-9 or err_s >= self.rel_improve * err_0:
            return None
        name = self.action_names[action].lower()
        rec = {
            "record_version": RECORD_VERSION,
            "kind": "acted_event",
            "trigger_key": f"pressed:{name}",
            "mechanism": "view_shift",
            "outcome_key": f"view_shift:dx{s:+d}",
            "quote": {
                "env": self.env,
                "base_seed": self.base_seed,
                "episode": int(episode),
                "frame": int(frame_idx),
                "action": int(action),
                "action_name": name,
                "dx_measured": int(s),
                "err_at_dx": round(err_s, 6),
                "err_at_0": round(err_0, 6),
                "frame_sha256_pre": frame_sha(f0),
                "frame_sha256_post": frame_sha(f1),
            },
            "policy": self.policy_name,
        }
        self.records.append(rec)
        return rec

    def stats(self) -> dict:
        return {"n_seen": self.n_seen, "n_records": len(self.records),
                "n_skipped_boundary": self.n_skipped_boundary}


def seal_records(records: List[dict], store_dir: str,
                 jsonl_mirror: Optional[str] = None) -> Optional[str]:
    """Seal the records as one content-addressed segment in a LiveStore
    under body/ (drafts -- merging into any main store is a human/Lead
    decision, not this function's). Returns the segment sha, or None for
    an empty record list (an empty segment would be a hash of nothing)."""
    if jsonl_mirror:
        d = os.path.dirname(jsonl_mirror) or "."
        os.makedirs(d, exist_ok=True)
        with open(jsonl_mirror, "w") as f:
            for r in records:
                f.write(json.dumps(r, sort_keys=True, ensure_ascii=False) + "\n")
    if not records:
        return None
    store = LiveStore(store_dir)
    return store.append_segment(records)


def verify_records(records: List[dict], trace, n_samples: int = 5,
                   seed: int = 7, world_kwargs: Optional[dict] = None) -> dict:
    """The provenance gate: sample records, REPLAY the world under the
    recorded action trace, and require (a) both frame hashes match
    bit-exactly and (b) the estimator recomputes the same dx. trace is
    the run's recorded action structure: a flat list (synthetic) or
    {episode: [actions]} (vizdoom). Returns {"picks": [...], "exact": "k/n",
    "pass": bool}; prints one line per pick, P52-style."""
    if not records:
        return {"picks": [], "exact": "0/0", "pass": False,
                "note": "no records to verify"}
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(records), size=min(n_samples, len(records)),
                     replace=False)
    picks = []
    for i in idx:
        rec = records[int(i)]
        q = rec["quote"]
        env, ep, fr = q["env"], q["episode"], q["frame"]
        if env == "acted_synthetic":
            from action_sources import ActedSyntheticSource
            actions = trace if isinstance(trace, list) else trace[ep]
            frames = ActedSyntheticSource.replay_frames(
                q["base_seed"], actions[:fr + 1], **(world_kwargs or {}))
            f0, f1 = frames[fr], frames[fr + 1]
        elif env == "vizdoom":
            from action_sources import ActedVizDoomSource
            actions = trace[ep] if isinstance(trace, dict) else trace
            frames = ActedVizDoomSource.replay_episode(
                q["base_seed"], ep, actions[:fr + 1])
            if len(frames) < fr + 2:
                picks.append({"episode": ep, "frame": fr, "exact": False,
                              "why": "replay ended early"})
                continue
            f0, f1 = frames[fr], frames[fr + 1]
        else:
            picks.append({"episode": ep, "frame": fr, "exact": False,
                          "why": f"unknown env {env}"})
            continue
        sha_ok = (frame_sha(f0) == q["frame_sha256_pre"]
                  and frame_sha(f1) == q["frame_sha256_post"])
        s, _, _ = estimate_shift(f0, f1)
        dx_ok = (s == q["dx_measured"])
        picks.append({"episode": ep, "frame": fr, "exact": bool(sha_ok),
                      "dx_redetected": bool(dx_ok)})
        print(f"[provenance] ep {ep} frame {fr} action {q['action_name']} "
              f"dx {q['dx_measured']:+d} -> "
              f"{'BIT-EXACT' if sha_ok else 'HASH MISMATCH'}"
              f"{'' if dx_ok else ' (dx drifted!)'}", flush=True)
    n_exact = sum(1 for p in picks if p["exact"] and p.get("dx_redetected"))
    return {"picks": picks, "exact": f"{n_exact}/{len(picks)}",
            "pass": bool(picks) and n_exact == len(picks)}
