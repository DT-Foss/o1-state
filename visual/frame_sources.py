"""
FRAME SOURCES -- one interface, multiple frame streams, for the visual
organism (Phase 1 of the visual organism track, see NEXT.md /
FOUNDATIONS.md's F1/F2/F5 primitives that this file's consumer,
frame_organism.py, inherits).

Contract: every source exposes next_frame() -> np.ndarray, shape (64, 64),
dtype float32, values in [0, 1], grayscale. Callers pull frames one at a
time (no batching here -- frame_organism.py's chunking is the caller's
concern, not the source's).

  SyntheticSource   -- 2-3 bouncing shapes with occasional occlusion,
                       deterministic per seed. Always available (numpy only).
  VizDoomSource      -- real game frames via the vizdoom Python package,
                       ONLY constructed if vizdoom is importable AND a
                       DoomGame initializes successfully headless. Callers
                       must check VizDoomSource.available() first (see
                       train_visual.py) -- the class itself raises if you
                       try to build one on a machine without vizdoom rather
                       than silently falling back, so a caller that skips
                       the availability check gets a loud error, not a
                       quiet wrong-source substitution.
"""

import numpy as np

FRAME_SIZE = 64


class SyntheticSource:
    """2-3 moving shapes (circle/rectangle) bouncing off walls on a 64x64
    grayscale canvas, with occasional occlusion (one shape passing in front
    of / merging with another) so a memoryless per-frame predictor cannot
    solve it -- predicting through an occlusion requires carrying the
    occluded shape's velocity in state across the frames where it isn't
    directly visible, which is exactly the pressure the GSSM core's carried
    z-state is meant to absorb.

    Deterministic given a seed: same seed -> byte-identical frame sequence,
    forever (np.random.default_rng(seed) is fully self-contained, no global
    RNG state touched).
    """

    def __init__(self, seed: int = 0, n_shapes: int = 3, size: int = FRAME_SIZE):
        self.size = size
        self.rng = np.random.default_rng(seed)
        self.shapes = []
        for i in range(n_shapes):
            kind = "circle" if i % 2 == 0 else "rect"
            r = self.rng.uniform(4.0, 8.0)
            x = self.rng.uniform(r, size - r)
            y = self.rng.uniform(r, size - r)
            angle = self.rng.uniform(0, 2 * np.pi)
            speed = self.rng.uniform(1.0, 2.2)
            vx, vy = speed * np.cos(angle), speed * np.sin(angle)
            # Occlusion order: shapes drawn in list order, later shapes paint
            # over earlier ones where they overlap -- so which shape "wins"
            # an occlusion is fixed by draw order, not by depth simulation.
            brightness = self.rng.uniform(0.55, 1.0)
            self.shapes.append({
                "kind": kind, "x": x, "y": y, "r": r,
                "vx": vx, "vy": vy, "brightness": brightness,
            })
        self.frame_idx = 0

    def _step_shape(self, s):
        s["x"] += s["vx"]
        s["y"] += s["vy"]
        r = s["r"]
        if s["x"] - r < 0:
            s["x"] = r
            s["vx"] = abs(s["vx"])
        elif s["x"] + r > self.size:
            s["x"] = self.size - r
            s["vx"] = -abs(s["vx"])
        if s["y"] - r < 0:
            s["y"] = r
            s["vy"] = abs(s["vy"])
        elif s["y"] + r > self.size:
            s["y"] = self.size - r
            s["vy"] = -abs(s["vy"])

    def _render(self) -> np.ndarray:
        canvas = np.zeros((self.size, self.size), dtype=np.float32)
        yy, xx = np.mgrid[0:self.size, 0:self.size].astype(np.float32)
        for s in self.shapes:
            if s["kind"] == "circle":
                mask = (xx - s["x"]) ** 2 + (yy - s["y"]) ** 2 <= s["r"] ** 2
            else:  # rect, half-width == r (square footprint, simple + cheap)
                r = s["r"]
                mask = (np.abs(xx - s["x"]) <= r) & (np.abs(yy - s["y"]) <= r)
            canvas[mask] = s["brightness"]  # later shapes overwrite -> occlusion
        return canvas

    def next_frame(self) -> np.ndarray:
        frame = self._render()
        for s in self.shapes:
            self._step_shape(s)
        self.frame_idx += 1
        return frame


class VizDoomSource:
    """Real game frames from the vizdoom `basic.cfg` scenario (simplest
    built-in scenario: one room, one monster, movement + attack), grayscale,
    downsampled to 64x64. Deterministic random actions per seed -- the
    action sequence is drawn from a seeded RNG, not vizdoom's own internal
    randomness, so the SAME seed replays the SAME action trace (vizdoom's
    engine-internal state given a fixed action trace is itself deterministic
    for this scenario at this API level).

    Episodes auto-restart (new_episode()) when the game reports
    is_episode_finished(), so next_frame() never raises mid-stream -- the
    frame source is an infinite stream, same contract as SyntheticSource.
    """

    @staticmethod
    def available() -> bool:
        """Cheap availability probe: import + headless init + teardown of a
        throwaway DoomGame. Never raises -- returns False on any failure
        (import error, missing IWAD, headless init failure, ...)."""
        try:
            import vizdoom as vzd
            import os
            game = vzd.DoomGame()
            game.load_config(os.path.join(vzd.scenarios_path, "basic.cfg"))
            game.set_window_visible(False)
            game.set_screen_format(vzd.ScreenFormat.GRAY8)
            game.set_screen_resolution(vzd.ScreenResolution.RES_160X120)
            game.init()
            game.close()
            return True
        except Exception:
            return False

    def __init__(self, seed: int = 0, size: int = FRAME_SIZE):
        import vizdoom as vzd
        import os
        self.size = size
        self.rng = np.random.default_rng(seed)
        self.vzd = vzd

        self.game = vzd.DoomGame()
        self.game.load_config(os.path.join(vzd.scenarios_path, "basic.cfg"))
        self.game.set_window_visible(False)
        self.game.set_screen_format(vzd.ScreenFormat.GRAY8)
        self.game.set_screen_resolution(vzd.ScreenResolution.RES_160X120)
        self.game.init()

        self.n_actions = self.game.get_available_buttons_size()
        self.game.new_episode()
        self.frame_idx = 0

    def _random_action(self):
        # One-hot action over the scenario's button set (basic.cfg: 3
        # buttons -- MOVE_LEFT, MOVE_RIGHT, ATTACK). Seeded choice, not
        # vizdoom's own RNG.
        idx = int(self.rng.integers(0, self.n_actions))
        action = [0] * self.n_actions
        action[idx] = 1
        return action

    def _downsample(self, buf: np.ndarray) -> np.ndarray:
        # buf: (120, 160) uint8 GRAY8. Block-mean downsample to (64, 64) via
        # nearest-neighbor index mapping (simple, deterministic, no extra
        # dependency) -- good enough at this resolution for Phase 1.
        h, w = buf.shape
        ys = (np.linspace(0, h - 1, self.size)).astype(np.int64)
        xs = (np.linspace(0, w - 1, self.size)).astype(np.int64)
        small = buf[np.ix_(ys, xs)]
        return small.astype(np.float32) / 255.0

    def next_frame(self) -> np.ndarray:
        if self.game.is_episode_finished():
            self.game.new_episode()
        state = self.game.get_state()
        frame = self._downsample(state.screen_buffer)
        self.game.make_action(self._random_action())
        self.frame_idx += 1
        return frame

    def close(self):
        self.game.close()
