"""
GROUNDING BRIDGE -- Phase 4 first step: our FramePredictor encoder as a
feature extractor BEHIND the grounding_kernel's learner protocol, instead of
binder.py's hand-built _observation_features.

Design (per Lead's brief): the grounding_kernel package (visual/grounding/)
is NOT touched -- no line in grounding_kernel/*.py changes. This module lives
outside it and only imports its public contracts (GroundingEvidence,
Observation, Action, Transition) and its benchmark runner. Frame
normalization -- Observation.pixels is (H, W, 3) uint8 RGB; our FramePredictor
trained on 64x64 grayscale float32 [0,1] -- happens entirely IN this file's
adapter, never inside the kernel.

Three arms compared through the SAME run_benchmark() call, same episodes/seed:
  (i)   binder     -- grounding_kernel.binder.SensorimotorBinder, unmodified.
                      Reference: the kernel's own hand-feature learner.
  (ii)  bridge_untrained -- EncoderBridgeBinder wrapping a FRESH random-init
                      FramePredictor encoder (no training at all). Control:
                      does architecture alone (GSSMCore's bounded recurrence,
                      the encoder's projection) carry ANY binding-relevant
                      structure, before a single gradient step?
  (iii) bridge_trained -- EncoderBridgeBinder wrapping the arm_b checkpoint
                      (residual+nogate, the best Phase-2 L1) from
                      results/visual_arm_b_residual_nogate_ckpt.pt. The actual
                      question: does the LEARNED frame-prediction
                      representation carry structure a symbol binder can use?

EncoderBridgeBinder itself is a small, self-contained prototype classifier
(nearest-centroid-with-abstention, the same conceptual shape as
SensorimotorBinder's fit/predict_token but built from scratch here rather
than importing and monkeypatching binder.py's internals) so that swapping the
feature function is the ONLY variable between arms (ii) and (iii), and
neither borrows any code path from the untouched reference binder used in
arm (i).
"""

import os
import sys
import json
from collections.abc import Hashable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from math import sqrt
from types import MappingProxyType
from typing import Any

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import numpy as np
import torch

torch.set_num_threads(1)
try:
    torch.set_num_interop_threads(1)
except RuntimeError:
    pass

_VISUAL_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_VISUAL_DIR)
_GROUNDING_DIR = os.path.join(_VISUAL_DIR, "grounding")
for _p in (_VISUAL_DIR, _GROUNDING_DIR, _REPO_ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from frame_organism import FramePredictor, FRAME_SIZE  # visual/frame_organism.py
from grounding_kernel.contracts import Action, Observation, Transition  # noqa: E402
from grounding_kernel.benchmark import (  # noqa: E402
    BENCHMARK_ID,
    BenchmarkResult,
    GroundingEvidence,
    run_benchmark,
)


def _summarize(result: BenchmarkResult, *, learner: str) -> dict:
    """Same shape as benchmark.py's private _result_summary (not imported --
    that name is underscore-private and this module stays independent of
    benchmark.py's internal helpers, only its public run_benchmark/
    BenchmarkResult/GroundingEvidence surface)."""

    return {
        "benchmark_id": BENCHMARK_ID,
        "learner": learner,
        "passed": result.passed,
        "score": result.certificate.score,
        "scope_hash": result.certificate.scope.scope_hash,
        "certificate_hash": result.certificate.certificate_hash,
        "axes": {
            axis.name: {
                "estimate": axis.performance.estimate,
                "lower_bound_95": axis.performance.lower_bound,
                "coverage": axis.coverage.estimate,
                "coverage_lower_bound_95": axis.coverage.lower_bound,
                "passed": axis.passed,
            }
            for axis in result.certificate.axes
        },
        "controls": {
            control.name: {
                "estimate": control.performance.estimate,
                "coverage": control.coverage.estimate,
                "rejected_as_grounder": control.rejected_as_grounder,
            }
            for control in result.ledger.controls
        },
    }


# ═══════════════════════════════════════════════════════════════════════════
#  Frame normalization: Observation.pixels (H, W, 3) uint8 RGB -> our
#  FramePredictor's expected 64x64 grayscale float32 [0, 1]. This is the ONLY
#  place pixel format is translated -- the kernel's Observation stays exactly
#  as contracts.py defines it; the encoder never sees anything but its usual
#  4096-float input.
# ═══════════════════════════════════════════════════════════════════════════
def _observation_to_encoder_input(observation: Observation) -> torch.Tensor:
    pixels = observation.pixels.astype(np.float32) / 255.0  # (H, W, 3) in [0,1]
    gray = pixels.mean(axis=2)  # simple luminance-agnostic average, no cv2 dependency
    h, w = gray.shape
    if (h, w) != (FRAME_SIZE, FRAME_SIZE):
        ys = np.linspace(0, h - 1, FRAME_SIZE).astype(np.int64)
        xs = np.linspace(0, w - 1, FRAME_SIZE).astype(np.int64)
        gray = gray[np.ix_(ys, xs)]
    return torch.from_numpy(gray.reshape(-1).copy())  # (4096,) float32


class _EncoderFeatureFn:
    """Wraps a FramePredictor's encoder (+ one GSSMCore step) as a
    fixed-length numpy feature function over (before, action, after) triples.

    The encoder is used STATELESS per call (states=None each time) -- a
    single-frame push through encoder+GSSMCore+RMSNorm, not a carried stream
    -- because the grounding_kernel's evidence records are independent
    transitions in randomized order (fit() receives a shuffled Sequence[
    GroundingEvidence], not a temporal stream), so there is no legitimate
    "previous chunk" to carry state from between unrelated training records.
    This still exercises the FULL trained stack (encoder Linear + GSSMCore's
    bounded recurrence from a zero initial state), just not its streaming
    memory -- an honest limitation, named here rather than papered over.

    Feature vector = [enc(before) ; enc(after) ; enc(after)-enc(before) ;
                       action_features] -- mirrors binder.py's own
    before/after/delta/action structure (README's own recipe), just with a
    LEARNED enc() instead of _observation_features()'s hand-built quantile/
    histogram/geometry pipeline.

    projection_dim (optional): if set, a FIXED (seeded, not fit on training
    data) random Gaussian projection matrix maps the raw 803-dim vector down
    to this many dimensions before it reaches the prototype learner. Random
    projection was chosen over PCA for the dimension-vs-signal diagnostic
    specifically because it is parameter-free and independent of the
    12-example-per-token training split -- PCA fit on that same tiny split
    would reintroduce the exact data-scarcity confound the experiment is
    trying to isolate (Lead's brief). The projection matrix is built once at
    construction time and reused for every observation.
    """

    def __init__(self, model: FramePredictor, device: torch.device,
                 projection_dim: int | None = None, projection_seed: int = 0):
        self.model = model
        self.device = device
        self.model.eval()
        self.d_model = model.d_model
        self.projection_dim = projection_dim
        self._projection: np.ndarray | None = None
        if projection_dim is not None:
            rng = np.random.default_rng(projection_seed)
            raw_dim = model.d_model * 3 + 31 + 2 + 2  # before+after+delta + action features
            # Gaussian random projection, columns normalized to unit L2 norm
            # so distances after projection stay on a comparable scale to
            # the unprojected feature space (no rescaling surprises in the
            # threshold/margin logic below).
            matrix = rng.standard_normal((raw_dim, projection_dim))
            matrix /= np.linalg.norm(matrix, axis=0, keepdims=True)
            self._projection = matrix

    @torch.no_grad()
    def _encode(self, observation: Observation) -> np.ndarray:
        x = _observation_to_encoder_input(observation).to(self.device)
        x = x.unsqueeze(0).unsqueeze(0)  # (1, 1, 4096)
        h = self.model.encoder(x)
        h, _ = self.model.core(h, None)  # stateless: zero initial state each call
        return h.squeeze(0).squeeze(0).cpu().numpy().astype(np.float64)

    @staticmethod
    def _action_features(action: Action) -> np.ndarray:
        # Same small numeric action encoding as binder.py's _action_features
        # (categorical code hash + tanh-squashed target/vector), reimplemented
        # here at a fixed small width instead of imported, since binder.py's
        # version is private (leading underscore) and this keeps the bridge
        # fully independent of binder.py's internals.
        bins = 31
        digest = abs(hash(("action", action.code))) % bins
        categorical = np.zeros(bins, dtype=np.float64)
        categorical[digest] = 1.0
        target = np.tanh(np.asarray(action.target, dtype=np.float64) / 16.0)
        vector = np.tanh(np.asarray(action.vector, dtype=np.float64) / 16.0)
        return np.concatenate((categorical, target, vector))

    def __call__(self, before: Observation, action: Action, after: Observation) -> np.ndarray:
        enc_before = self._encode(before)
        enc_after = self._encode(after)
        delta = enc_after - enc_before
        action_feat = self._action_features(action)
        raw = np.concatenate((enc_before, enc_after, delta, action_feat))
        if self._projection is not None:
            return raw @ self._projection
        return raw


# ═══════════════════════════════════════════════════════════════════════════
#  EncoderBridgeBinder -- self-contained nearest-centroid-with-abstention
#  learner, structurally analogous to SensorimotorBinder's fit/predict_token
#  contract but built from scratch (no import of binder.py's private
#  helpers) so the ONLY shared code path between arms (ii)/(iii) and the
#  reference arm (i) is the public GroundZeroLearner protocol itself.
# ═══════════════════════════════════════════════════════════════════════════
@dataclass(frozen=True, slots=True)
class _Prototype:
    center: np.ndarray
    scale: np.ndarray
    threshold: float
    count: int

    def distance(self, vector: np.ndarray) -> float:
        standardized = (vector - self.center) / self.scale
        return sqrt(float(np.mean(standardized * standardized)))


class EncoderBridgeBinder:
    """fit(experiences) / predict_token(...) / supports_token(...) over a
    FramePredictor-encoder feature space. min_margin and quantile-based
    threshold calibration mirror binder.py's own abstention policy (same
    conceptual recipe: reject if outside the calibrated radius, or if two
    tokens are within min_margin of each other) -- reimplemented here rather
    than imported, per the "kernel untouched" constraint.
    """

    def __init__(self, feature_fn: _EncoderFeatureFn, *, alpha: float = 0.1,
                 min_margin: float = 0.08, minimum_radius: float = 0.35):
        self.feature_fn = feature_fn
        self.alpha = alpha
        self.min_margin = min_margin
        self.minimum_radius = minimum_radius
        self._prototypes: dict[Hashable, _Prototype] = {}
        self._fitted = False

    def _vector(self, evidence: GroundingEvidence) -> np.ndarray:
        return self.feature_fn(evidence.before, evidence.action, evidence.after)

    def fit(self, experiences: Sequence[GroundingEvidence]) -> "EncoderBridgeBinder":
        grouped: dict[int, list[np.ndarray]] = {}
        for record in experiences:
            if record.task_feedback is False:
                continue
            grouped.setdefault(int(record.token), []).append(self._vector(record))
        if not grouped:
            raise ValueError("at least one positive training experience is required")
        all_vectors = np.vstack([v for values in grouped.values() for v in values])
        if all_vectors.shape[0] > 1:
            scale = np.std(all_vectors, axis=0, ddof=1)
        else:
            scale = np.zeros(all_vectors.shape[1], dtype=np.float64)
        positive_scales = scale[scale > 1e-9]
        fallback = max(0.05, float(np.median(positive_scales)) * 0.25 if positive_scales.size else 0.05)
        scale = np.where(scale > 1e-9, scale, fallback)

        prototypes: dict[Hashable, _Prototype] = {}
        for token, values in grouped.items():
            stacked = np.vstack(values)
            center = np.mean(stacked, axis=0)
            if len(values) > 1:
                distances = []
                total = np.sum(stacked, axis=0)
                for v in values:
                    loo_center = (total - v) / (len(values) - 1)
                    standardized = (v - loo_center) / scale
                    distances.append(sqrt(float(np.mean(standardized * standardized))))
                rank = min(len(distances) - 1, max(0, int(np.ceil((len(distances) + 1) * (1.0 - self.alpha))) - 1))
                threshold = max(self.minimum_radius, sorted(distances)[rank])
            else:
                threshold = self.minimum_radius
            prototypes[token] = _Prototype(center, scale, threshold, len(values))
        self._prototypes = prototypes
        self._fitted = True
        return self

    @property
    def manifest(self) -> Mapping[str, Any]:
        return MappingProxyType({
            "name": "encoder-bridge-binder",
            "version": 1,
            "feature_source": "FramePredictor.encoder + GSSMCore (stateless per-call)",
            "d_model": self.feature_fn.d_model,
            "oracle_access": False,
        })

    def predict_token(
        self,
        evidence_or_before: Any,
        action: Any | None = None,
        after: Any | None = None,
        candidates: Sequence[Hashable] | None = None,
    ) -> Hashable | None:
        if not self._fitted:
            raise RuntimeError("bridge is not fitted")
        transition = evidence_or_before
        if action is not None or after is not None:
            transition = Transition(evidence_or_before, action, after, 0)
        vector = self.feature_fn(transition.before, transition.action, transition.after)
        tokens = tuple(candidates) if candidates is not None else tuple(self._prototypes)
        tokens = tuple(t for t in tokens if t in self._prototypes)
        if not tokens:
            return None
        distances = {t: self._prototypes[t].distance(vector) for t in tokens}
        ordered = sorted(distances, key=lambda t: (distances[t], repr(t)))
        best = ordered[0]
        distance = distances[best]
        second = distances[ordered[1]] if len(ordered) > 1 else float("inf")
        margin = 1.0 if second == float("inf") else max(0.0, (second - distance) / max(second, 1e-12))
        if distance > self._prototypes[best].threshold:
            return None  # outside calibrated radius -- abstain
        if len(ordered) > 1 and margin < self.min_margin:
            return None  # indistinguishable prototypes -- abstain
        return best

    def supports_token(self, transition: Any, token: Hashable) -> bool | None:
        if not self._fitted or token not in self._prototypes:
            return None
        vector = self.feature_fn(transition.before, transition.action, transition.after)
        distance = self._prototypes[token].distance(vector)
        prediction = self.predict_token(transition, candidates=tuple(self._prototypes))
        if prediction is not None:
            return prediction == token
        if distance > self._prototypes[token].threshold:
            return False
        return None


# ═══════════════════════════════════════════════════════════════════════════
#  Model construction for the two bridge arms.
# ═══════════════════════════════════════════════════════════════════════════
def _pick_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _build_untrained_model(device: torch.device, seed: int) -> FramePredictor:
    torch.manual_seed(seed)
    model = FramePredictor(d_model=256, n_layers=4, n_heads=4, frame_dim=FRAME_SIZE * FRAME_SIZE,
                            residual=True).to(device)
    return model


def _build_trained_model(device: torch.device, ckpt_path: str) -> FramePredictor:
    ck = torch.load(ckpt_path, weights_only=False, map_location=device)
    cfg = ck["config"]
    model = FramePredictor(d_model=cfg["d_model"], n_layers=4, n_heads=4,
                            frame_dim=cfg["frame_dim"], residual=True).to(device)
    model.load_state_dict(ck["model"])
    return model


def make_bridge_factory(model: FramePredictor, device: torch.device,
                         projection_dim: int | None = None, projection_seed: int = 0):
    """Returns a LearnerFactory (int -> EncoderBridgeBinder). run_benchmark
    calls this THREE times per arm (base/remapped/shuffled splits) -- each
    call must return a FRESH learner instance sharing the same (frozen,
    untrained-during-benchmark) encoder weights AND the same fixed projection
    matrix, so neither the encoder nor the projection is ever fit -- only the
    prototype centroids on top of them are, per split."""

    feature_fn = _EncoderFeatureFn(model, device, projection_dim=projection_dim,
                                    projection_seed=projection_seed)

    def factory(_seed: int) -> EncoderBridgeBinder:
        return EncoderBridgeBinder(feature_fn)

    return factory


# ═══════════════════════════════════════════════════════════════════════════
#  Four-arm runner (Lead's controlled design after the dimensionality
#  diagnosis): binder reference, bridge_untrained+projection,
#  bridge_trained+projection, bridge_trained WITHOUT projection (the prior
#  arm, carried along in the same run for direct coverage).
# ═══════════════════════════════════════════════════════════════════════════
PROJECTION_DIM = 48
PROJECTION_SEED = 1234  # fixed, arbitrary -- documented here as the single source of truth


def run_four_arms(seed: int = 3, episodes: int = 24, ckpt_path: str | None = None,
                   projection_dim: int = PROJECTION_DIM) -> dict:
    from grounding_kernel.binder import SensorimotorBinder

    device = _pick_device()
    print(f"[grounding_bridge] device={device} seed={seed} episodes={episodes} "
          f"projection_dim={projection_dim}", flush=True)

    ckpt_path = ckpt_path or os.path.join(
        _REPO_ROOT, "results", "visual_arm_b_residual_nogate_ckpt.pt"
    )

    arms: dict[str, dict] = {}

    print("[grounding_bridge] arm (1) binder -- reference, unmodified", flush=True)
    result_binder = run_benchmark(
        seed=seed, episodes=episodes,
        learner_factory=lambda _seed: SensorimotorBinder(),
    )
    arms["binder"] = _summarize(result_binder, learner="binder")

    print(f"[grounding_bridge] arm (2) bridge_untrained_proj{projection_dim} -- "
          f"fresh random-init encoder, {projection_dim}-dim random projection", flush=True)
    untrained_model = _build_untrained_model(device, seed=seed)
    result_untrained_proj = run_benchmark(
        seed=seed, episodes=episodes,
        learner_factory=make_bridge_factory(untrained_model, device,
                                             projection_dim=projection_dim,
                                             projection_seed=PROJECTION_SEED),
    )
    arms[f"bridge_untrained_proj{projection_dim}"] = _summarize(
        result_untrained_proj, learner=f"bridge_untrained_proj{projection_dim}")

    print(f"[grounding_bridge] arm (3) bridge_trained_proj{projection_dim} -- "
          f"{ckpt_path}, {projection_dim}-dim random projection", flush=True)
    trained_model = _build_trained_model(device, ckpt_path)
    result_trained_proj = run_benchmark(
        seed=seed, episodes=episodes,
        learner_factory=make_bridge_factory(trained_model, device,
                                             projection_dim=projection_dim,
                                             projection_seed=PROJECTION_SEED),
    )
    arms[f"bridge_trained_proj{projection_dim}"] = _summarize(
        result_trained_proj, learner=f"bridge_trained_proj{projection_dim}")

    print("[grounding_bridge] arm (4) bridge_trained_raw -- control, no projection "
          "(the prior arm, carried along in this same run for direct coverage)", flush=True)
    result_trained_raw = run_benchmark(
        seed=seed, episodes=episodes,
        learner_factory=make_bridge_factory(trained_model, device, projection_dim=None),
    )
    arms["bridge_trained_raw"] = _summarize(result_trained_raw, learner="bridge_trained_raw")

    return arms


def _axis_table(arms: dict) -> str:
    axis_names = None
    lines = []
    for arm_name, result in arms.items():
        axes = result["axes"]
        if axis_names is None:
            axis_names = list(axes.keys())
    header = f"{'axis':32s}" + "".join(f"{name:>22s}" for name in arms)
    lines.append(header)
    lines.append("-" * len(header))
    for axis in axis_names:
        row = f"{axis:32s}"
        for arm_name, result in arms.items():
            entry = result["axes"][axis]
            cell = f"cov={entry['coverage']:.3f} lcb={entry['lower_bound_95']:.3f} {'PASS' if entry['passed'] else 'FAIL'}"
            row += f"{cell:>22s}"
        lines.append(row)
    lines.append("")
    lines.append(f"{'static_pixels_no_action (control)':32s}" +
                 "".join(f"{arms[a]['controls']['static_pixels_no_action']['estimate']:>22.4f}" for a in arms))
    lines.append(f"{'rejected_as_grounder':32s}" +
                 "".join(f"{str(arms[a]['controls']['static_pixels_no_action']['rejected_as_grounder']):>22s}" for a in arms))
    lines.append(f"{'overall passed':32s}" +
                 "".join(f"{str(arms[a]['passed']):>22s}" for a in arms))
    lines.append(f"{'score':32s}" +
                 "".join(f"{arms[a]['score']:>22.4f}" for a in arms))
    return "\n".join(lines)


def _distance_diagnosis(model: FramePredictor, device: torch.device, seed: int, episodes: int,
                         projection_dim: int, projection_seed: int = PROJECTION_SEED) -> str:
    """Same diagnosis as the raw-feature investigation that found the
    dimensionality wall (Leave-one-out distances to own prototype vs.
    calibrated threshold, then eval-time distances across all token
    prototypes) -- rerun here at the PROJECTED dimension, so we can see
    directly whether the radii now discriminate."""

    sys.path.insert(0, _GROUNDING_DIR)
    from grounding_kernel.benchmark import build_ground_zero_dataset

    feature_fn = _EncoderFeatureFn(model, device, projection_dim=projection_dim,
                                    projection_seed=projection_seed)
    dataset = build_ground_zero_dataset(seed=seed, episodes=episodes)
    learner = EncoderBridgeBinder(feature_fn)
    learner.fit(dataset.train)

    lines = [f"--- distance diagnosis at projection_dim={projection_dim} ---"]
    tok0 = dataset.identifiable_tokens[0]
    tok0_recs = [r for r in dataset.train if r.token == tok0]
    proto = learner._prototypes[tok0]
    lines.append(f"token {tok0}: {len(tok0_recs)} training records, calibrated threshold={proto.threshold:.4f}")
    loo_distances = []
    for r in tok0_recs:
        v = learner._vector(r)
        loo_distances.append(proto.distance(v))
    lines.append(f"  leave-one-out distances to own prototype: "
                 f"{[round(d, 4) for d in sorted(loo_distances)]}")

    lines.append("")
    lines.append("eval-time distances on the first 5 token_cases (all candidate prototypes):")
    n_within_threshold = 0
    n_cases = 0
    for i, case in enumerate(dataset.token_cases[:5]):
        vec = learner.feature_fn(case.transition.before, case.transition.action, case.transition.after)
        dists = {t: learner._prototypes[t].distance(vec)
                  for t in dataset.identifiable_tokens if t in learner._prototypes}
        pred = learner.predict_token(case.transition, candidates=dataset.identifiable_tokens)
        lines.append(f"  case {i}: expected={case.expected} pred={pred} "
                     f"distances={ {t: round(d, 4) for t, d in dists.items()} }")

    all_cases_pred = [learner.predict_token(c.transition, candidates=dataset.identifiable_tokens)
                       for c in dataset.token_cases]
    n_accepted = sum(1 for p in all_cases_pred if p is not None)
    lines.append("")
    lines.append(f"acceptance rate over all {len(dataset.token_cases)} token_cases: "
                 f"{n_accepted}/{len(dataset.token_cases)} = {n_accepted/len(dataset.token_cases):.3f}")
    return "\n".join(lines)


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="Grounding bridge: four-arm binder comparison.")
    ap.add_argument("--seed", type=int, default=3)
    ap.add_argument("--episodes", type=int, default=24)
    ap.add_argument("--ckpt", type=str, default=None)
    ap.add_argument("--projection-dim", type=int, default=PROJECTION_DIM)
    ap.add_argument("--out", type=str, default=None)
    args = ap.parse_args()

    arms = run_four_arms(seed=args.seed, episodes=args.episodes, ckpt_path=args.ckpt,
                          projection_dim=args.projection_dim)
    print()
    print("=" * 100)
    print("FOUR-ARM GROUNDING BENCHMARK COMPARISON")
    print("=" * 100)
    print(_axis_table(arms))

    print()
    print("=" * 100)
    print("DISTANCE DIAGNOSIS -- bridge_trained at the projected dimension")
    print("=" * 100)
    device = _pick_device()
    ckpt_path = args.ckpt or os.path.join(_REPO_ROOT, "results", "visual_arm_b_residual_nogate_ckpt.pt")
    trained_model = _build_trained_model(device, ckpt_path)
    print(_distance_diagnosis(trained_model, device, args.seed, args.episodes, args.projection_dim))

    if args.out:
        with open(args.out, "w") as f:
            json.dump(arms, f, indent=2)
        print(f"\n[grounding_bridge] full results written: {args.out}")
