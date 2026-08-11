"""
VENDOR-SNAPSHOT der o1-state Pan-Welt -- NUR für Provenance-Replay.

FERTIG ist ein eigenständiges Projekt; damit das Weltbuch (gelebte
acted-Records als sprechbarer Kausalgraph) seine Belege BIT-EXAKT
nachspielen kann, ohne das o1-state-Repo zu importieren, liegt hier ein
eingefrorener, minimaler Snapshot der Welt-Logik:

  Quelle: o1-state, Branch voice-moonshot-2026-08-11, Commit b575b3f --
  body/action_sources.py (ActedSyntheticSource) + voice/vocab.py
  (RichPanSource: zyklische Pan-Magnitude 1,2,3) + body/causal_records.py
  (frame_sha, estimate_shift). Float32-Operationen in identischer
  Reihenfolge -- nur so stimmen die SHA-256-Hashes der Records überein.

Dieser Snapshot wird NIE weiterentwickelt; ändert sich die Welt drüben,
gehört hier ein NEUER Snapshot mit neuem Commit-Pin hin (Vendor-Regel
wie fertig/_vendor/dotcausal).
"""

import hashlib

import numpy as np

FRAME_SIZE = 64


class RichPanWorld:
    """Byte-treuer Snapshot: 64x128-Welt (statische Landmarken + 2
    bewegte Kreise), 64er-Viewport, Aktionen PAN_LEFT/PAN_RIGHT/HOLD,
    Pan-Magnitude zyklisch (frame_idx % 3) + 1."""

    N_ACTIONS = 3
    ACTION_NAMES = ["PAN_LEFT", "PAN_RIGHT", "HOLD"]

    def __init__(self, seed: int = 0, world_w: int = 128, view: int = FRAME_SIZE):
        self.seed = seed
        self.world_w = world_w
        self.view = view
        self.h = view
        rng = np.random.default_rng(seed)
        self.static = np.zeros((self.h, world_w), dtype=np.float32)
        for _ in range(6):
            w = int(rng.integers(6, 18))
            hh = int(rng.integers(6, 18))
            x = int(rng.integers(0, world_w - w))
            y = int(rng.integers(0, self.h - hh))
            self.static[y:y + hh, x:x + w] = rng.uniform(0.15, 0.45)
        for _ in range(5):
            x = int(rng.integers(0, world_w - 2))
            self.static[:, x:x + 2] = np.maximum(
                self.static[:, x:x + 2], rng.uniform(0.7, 0.95))
        self.shapes = []
        for _ in range(2):
            r = rng.uniform(4.0, 7.0)
            self.shapes.append({
                "x": rng.uniform(r, world_w - r),
                "y": rng.uniform(r, self.h - r),
                "r": r,
                "vx": rng.uniform(0.8, 1.8) * (1 if rng.random() < 0.5 else -1),
                "vy": rng.uniform(0.8, 1.8) * (1 if rng.random() < 0.5 else -1),
                "brightness": rng.uniform(0.85, 1.0),
            })
        self.view_x = (world_w - view) // 2
        self.frame_idx = 0

    def _step_shapes(self):
        for s in self.shapes:
            s["x"] += s["vx"]
            s["y"] += s["vy"]
            r = s["r"]
            if s["x"] - r < 0:
                s["x"], s["vx"] = r, abs(s["vx"])
            elif s["x"] + r > self.world_w:
                s["x"], s["vx"] = self.world_w - r, -abs(s["vx"])
            if s["y"] - r < 0:
                s["y"], s["vy"] = r, abs(s["vy"])
            elif s["y"] + r > self.h:
                s["y"], s["vy"] = self.h - r, -abs(s["vy"])

    def observe(self) -> np.ndarray:
        canvas = self.static.copy()
        yy, xx = np.mgrid[0:self.h, 0:self.world_w].astype(np.float32)
        for s in self.shapes:
            mask = (xx - s["x"]) ** 2 + (yy - s["y"]) ** 2 <= s["r"] ** 2
            canvas[mask] = s["brightness"]
        return canvas[:, self.view_x:self.view_x + self.view].copy()

    def act(self, a: int) -> None:
        step = (self.frame_idx % 3) + 1          # RichPan: zyklische Magnitude
        if a == 0:
            self.view_x = max(0, self.view_x - step)
        elif a == 1:
            self.view_x = min(self.world_w - self.view, self.view_x + step)
        self._step_shapes()
        self.frame_idx += 1

    @classmethod
    def replay_frames(cls, seed: int, actions):
        src = cls(seed=seed)
        frames = [src.observe()]
        for a in actions:
            src.act(int(a))
            frames.append(src.observe())
        return frames


def frame_sha(frame: np.ndarray) -> str:
    return hashlib.sha256(
        np.ascontiguousarray(frame, dtype=np.float32).tobytes()).hexdigest()


def estimate_shift(f0: np.ndarray, f1: np.ndarray, max_dx: int = 5):
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
