"""
ACTED FRAME SOURCES -- worlds the organism ACTS in, not just watches.

This is body/'s counterpart to visual/frame_sources.py, with one contract
change that IS the whole point of the body track: the caller chooses the
action. visual's sources hide their action choice inside next_frame()
(SyntheticSource has none; VizDoomSource draws seeded random actions
internally) -- so every consumer of that interface is structurally a
spectator. Here the interface splits into

    observe() -> np.ndarray (64, 64) float32 [0, 1]   current frame,
                                                       does NOT advance time
    act(a: int) -> None                                apply action a,
                                                       advance one step

and the world only moves when act() is called. n_actions / ACTION_NAMES
describe the discrete action set. Both sources also expose `.episode` and
`.frame_idx` (synthetic: episode is always 0) so the run loop can detect
episode boundaries, and both record the FULL action trace -- because under
a learned policy the action sequence is no longer derivable from the seed
(the P48/P52 provenance trick "re-simulate from (seed, step)" breaks the
moment the policy has weights). Provenance therefore travels as
(seed, episode, recorded action trace): replay = fresh world + same seed +
same actions -> bit-identical frames. That is the same discipline P52
verified through the vizdoom engine, extended to cover the policy.

  ActedSyntheticSource -- a 64x128 world canvas (static landmark texture +
                          2 self-moving shapes) seen through a 64-wide
                          viewport the organism pans. Own-action consequence
                          (view shifts) and world dynamics (shapes move
                          anyway) are cleanly separable BY CONSTRUCTION,
                          which is what makes the Stufe-2 causal-record
                          extraction testable against ground truth.
                          Always available (numpy only).
  ActedVizDoomSource    -- vizdoom basic.cfg (MOVE_LEFT, MOVE_RIGHT,
                          ATTACK), GRAY8 160x120 -> 64x64, frame-skip 4.
                          Fresh engine per episode with
                          set_seed(base*10000+episode) -- adopted verbatim
                          from src/vizdoom_run.py's make_game(), whose
                          docstring records the measured reason (set_seed on
                          a RUNNING engine yields a different episode than
                          on a fresh one). Availability probe reused
                          read-only from visual/frame_sources.py.

No file outside body/ is modified by importing or running this module.
"""

import os
import sys

import numpy as np

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for p in (REPO_ROOT, os.path.join(REPO_ROOT, "visual")):
    if p not in sys.path:
        sys.path.insert(0, p)

import frame_sources as fs  # read-only library import (visual/frame_sources.py)

FRAME_SIZE = fs.FRAME_SIZE  # 64 -- inherit, don't restate


class ActedSyntheticSource:
    """A pannable window onto a deterministic little world.

    World: 64 rows x `world_w` (default 128) columns. Static content is a
    landmark texture (vertical stripes + rectangles of varying brightness,
    drawn once from the seed) so that a horizontal pan has a visible,
    localizable consequence EVERYWHERE in the world -- there is no blank
    region where acting would be observationally silent. On top, 2 circles
    bounce with fixed seeded velocities: world dynamics that happen whether
    or not the organism acts, so a body model must separate "what I caused"
    from "what happens anyway" -- the minimal sensorimotor split.

    Actions (PAN_STEP = 3 px):
        0 PAN_LEFT   view_x -= 3   -> content appears shifted by +3
                                      (f1(x) == f0(x-3)... sign convention
                                      below, see estimate_shift's contract
                                      in causal_records.py)
        1 PAN_RIGHT  view_x += 3   -> content appears shifted the other way
        2 HOLD       nothing       -> only world dynamics move

    Sign convention used across body/: `dx` is the column offset s that
    best explains f1 from f0 via f1[:, x] ~= f0[:, x + s]. Viewport moving
    RIGHT by 3 (PAN_RIGHT) means f1(x) = world(x + view_x + 3) = f0(x + 3),
    so PAN_RIGHT -> dx = +3 and PAN_LEFT -> dx = -3. At the world border the
    pan clamps and the consequence honestly disappears (pressed left,
    nothing happened) -- the extractor is expected NOT to emit a record
    there, and the ground-truth helper reports dx = 0 for clamped steps.

    Determinism: frame_t is a pure function of (seed, action trace up to t).
    replay_frames() reconstructs the exact frame sequence of a run from the
    recorded trace -- byte-identical (same float32 ops in the same order).
    """

    N_ACTIONS = 3
    ACTION_NAMES = ["PAN_LEFT", "PAN_RIGHT", "HOLD"]
    PAN_STEP = 3

    def __init__(self, seed: int = 0, world_w: int = 128, view: int = FRAME_SIZE):
        self.seed = seed
        self.world_w = world_w
        self.view = view
        self.h = view  # world height == view height; pan is horizontal only
        rng = np.random.default_rng(seed)

        # -- static landmark texture ------------------------------------
        self.static = np.zeros((self.h, world_w), dtype=np.float32)
        for _ in range(6):  # dim rectangles
            w = int(rng.integers(6, 18))
            hh = int(rng.integers(6, 18))
            x = int(rng.integers(0, world_w - w))
            y = int(rng.integers(0, self.h - hh))
            self.static[y:y + hh, x:x + w] = rng.uniform(0.15, 0.45)
        for _ in range(5):  # bright vertical landmark stripes
            x = int(rng.integers(0, world_w - 2))
            self.static[:, x:x + 2] = np.maximum(
                self.static[:, x:x + 2], rng.uniform(0.7, 0.95))

        # -- self-moving shapes (world dynamics, action-independent) ----
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
        self.episode = 0  # synthetic world has one endless episode
        self.trace: list[int] = []          # full recorded action trace
        self.dx_truth: list[int] = []       # ground-truth dx per acted step
                                            # (post-clamp; 0 for HOLD/clamped)

    # -- world stepping --------------------------------------------------
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

    def _render_world(self) -> np.ndarray:
        canvas = self.static.copy()
        yy, xx = np.mgrid[0:self.h, 0:self.world_w].astype(np.float32)
        for s in self.shapes:
            mask = (xx - s["x"]) ** 2 + (yy - s["y"]) ** 2 <= s["r"] ** 2
            canvas[mask] = s["brightness"]
        return canvas

    # -- the acted contract ----------------------------------------------
    def observe(self) -> np.ndarray:
        world = self._render_world()
        return world[:, self.view_x:self.view_x + self.view].copy()

    def act(self, a: int) -> None:
        assert 0 <= a < self.N_ACTIONS, f"action {a} out of range"
        self.trace.append(int(a))
        old = self.view_x
        if a == 0:
            self.view_x = max(0, self.view_x - self.PAN_STEP)
        elif a == 1:
            self.view_x = min(self.world_w - self.view, self.view_x + self.PAN_STEP)
        # dx sign per the convention above: view moved right by d -> dx=+d
        self.dx_truth.append(self.view_x - old)
        self._step_shapes()
        self.frame_idx += 1

    def close(self):
        pass

    # -- provenance ------------------------------------------------------
    @classmethod
    def replay_frames(cls, seed: int, actions, world_w: int = 128,
                      view: int = FRAME_SIZE):
        """Re-run the world under a recorded action trace. Returns the list
        of frames exactly as a live run observed them: frames[0] is the
        pre-action observe(), frames[t] the observe() after the t-th act().
        len(frames) == len(actions) + 1. Byte-identical to the live run
        (pure float32 numpy, same op order, no global RNG)."""
        src = cls(seed=seed, world_w=world_w, view=view)
        frames = [src.observe()]
        for a in actions:
            src.act(int(a))
            frames.append(src.observe())
        return frames


class ActedVizDoomSource:
    """vizdoom basic.cfg with the action chosen by the CALLER -- the same
    engine P52 verified bit-deterministic under (seed, action trace), now
    with the trace coming from a policy instead of a seeded RNG.

    Episode addressing copies src/vizdoom_run.py's measured lesson verbatim:
    a fresh DoomGame per episode, seeded set_seed(base_seed*10000+episode),
    because set_seed on a running engine does NOT reproduce the episode a
    fresh engine gives (vizdoom_run.py make_game docstring, measured
    2026-08-05). Frame-skip 4 (P52's value) so one action has a visible
    consequence at 64x64.

    The recorded trace is a dict {episode: [action, ...]} plus the aligned
    per-episode frame count -- replay_episode() rebuilds any episode's frame
    sequence bit-exactly from it.

    Episode boundaries: observe() transparently advances to a fresh episode
    when the current one is finished; the caller detects this via
    `.episode` changing between observes. Frame pairs that straddle a
    boundary are real stream (the trainer may keep them) but are NOT
    consequences of the action -- causal_records.py skips them.
    """

    ACTION_NAMES = ["MOVE_LEFT", "MOVE_RIGHT", "ATTACK"]  # basic.cfg buttons
    N_ACTIONS = 3

    @staticmethod
    def available() -> bool:
        return fs.VizDoomSource.available()  # read-only reuse of the probe

    def __init__(self, base_seed: int = 0, size: int = FRAME_SIZE,
                 frame_skip: int = 4):
        self.base_seed = base_seed
        self.size = size
        self.skip = frame_skip
        self.episode = 0
        self.frame_idx = 0
        self.trace: dict[int, list[int]] = {0: []}
        self.game = self._make_game(self._episode_seed(0))
        n = self.game.get_available_buttons_size()
        assert n == self.N_ACTIONS, (
            f"basic.cfg exposes {n} buttons, expected {self.N_ACTIONS} "
            f"({self.ACTION_NAMES}) -- scenario drifted, refuse to guess")

    def _episode_seed(self, episode: int) -> int:
        return self.base_seed * 10000 + episode

    @staticmethod
    def _make_game(seed: int):
        """Fresh engine at a seed -- P52's factory, re-stated here because
        it is the load-bearing provenance primitive: life and replay MUST
        build engines the same way to be identical by construction."""
        import vizdoom as vzd
        g = vzd.DoomGame()
        g.load_config(os.path.join(vzd.scenarios_path, "basic.cfg"))
        g.set_window_visible(False)
        g.set_screen_format(vzd.ScreenFormat.GRAY8)
        g.set_screen_resolution(vzd.ScreenResolution.RES_160X120)
        g.set_seed(seed)
        g.init()
        g.new_episode()
        return g

    @staticmethod
    def _downsample(buf: np.ndarray, size: int = FRAME_SIZE) -> np.ndarray:
        # Same nearest-neighbor index mapping as visual/frame_sources.py's
        # VizDoomSource._downsample (re-stated: it is an instance method
        # there; 4 lines, and replay must use the identical mapping).
        h, w = buf.shape
        ys = (np.linspace(0, h - 1, size)).astype(np.int64)
        xs = (np.linspace(0, w - 1, size)).astype(np.int64)
        return buf[np.ix_(ys, xs)].astype(np.float32) / 255.0

    def _advance_episode(self):
        self.game.close()
        self.episode += 1
        self.frame_idx = 0
        self.trace[self.episode] = []
        self.game = self._make_game(self._episode_seed(self.episode))

    def observe(self) -> np.ndarray:
        if self.game.is_episode_finished():
            self._advance_episode()
        st = self.game.get_state()
        if st is None:
            self._advance_episode()
            st = self.game.get_state()
        return self._downsample(st.screen_buffer, self.size)

    def act(self, a: int) -> None:
        assert 0 <= a < self.N_ACTIONS, f"action {a} out of range"
        self.trace[self.episode].append(int(a))
        onehot = [0] * self.N_ACTIONS
        onehot[a] = 1
        self.game.make_action(onehot, self.skip)
        self.frame_idx += 1

    def close(self):
        self.game.close()

    @classmethod
    def replay_episode(cls, base_seed: int, episode: int, actions,
                       size: int = FRAME_SIZE, frame_skip: int = 4):
        """Fresh engine at the episode's seed, recorded actions stepped in
        order. Returns the frames exactly as the live run observed them
        WITHIN this episode: frames[i] is the observe() before the i-th
        recorded action (and one trailing frame if the episode had not
        ended). Stops early if the engine ends the episode -- identical to
        what the live path experienced, by construction."""
        g = cls._make_game(base_seed * 10000 + episode)
        frames = []
        try:
            for a in actions:
                if g.is_episode_finished():
                    break
                st = g.get_state()
                if st is None:
                    break
                frames.append(cls._downsample(st.screen_buffer, size))
                onehot = [0] * cls.N_ACTIONS
                onehot[int(a)] = 1
                g.make_action(onehot, frame_skip)
            if not g.is_episode_finished() and g.get_state() is not None:
                frames.append(cls._downsample(g.get_state().screen_buffer, size))
        finally:
            g.close()
        return frames


def make_acted_source(name: str, seed: int):
    """Mirror of visual/train_visual.py's make_source discipline: vizdoom
    unavailability is a LOUD exit, never a silent substitution."""
    if name == "acted_synthetic":
        return ActedSyntheticSource(seed=seed)
    if name == "vizdoom":
        if not ActedVizDoomSource.available():
            print("[body] ActedVizDoomSource requested but NOT AVAILABLE "
                  "(vizdoom import or headless init failed) -- refusing to "
                  "fall back silently; exiting.", flush=True)
            sys.exit(1)
        return ActedVizDoomSource(base_seed=seed)
    raise ValueError(f"unknown acted source: {name}")
