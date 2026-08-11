"""Learner-side binding of opaque tokens to ordered public episodes.

The binder consumes only duck-typed records containing an opaque token, an
episode and generic signed support feedback.  In full mode it builds compact prototypes
from raw ordered RGB effects around public image-space action targets.  It
never imports evaluator enums, decoders or hidden state, and it stores no full
frame or episode lookup table.  Query traces need only ``before/action/after``;
legacy opaque outcome codes are ignored by the full v1 path.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Hashable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from hashlib import sha256
from types import MappingProxyType
from typing import Literal

import numpy as np


BinderMode = Literal[
    "full",
    "no_sensor",
    "action_only",
    "action_target_only",
    "target_only",
    "action_outcome_only",
]
UNKNOWN: None = None
_FEATURE_VERSION = "ordered-rgb-intervention-v3-matched-action-target-controls"


def _stable_key(value: Hashable) -> tuple[str, str]:
    return (type(value).__qualname__, repr(value))


def _stable_digest(value: object) -> str:
    return sha256(repr(value).encode("utf-8")).hexdigest()


def _field(value: object, name: str) -> object:
    if isinstance(value, Mapping):
        if name not in value:
            raise AttributeError(f"missing public field: {name}")
        return value[name]
    if not hasattr(value, name):
        raise AttributeError(f"missing public field: {name}")
    return getattr(value, name)


def _optional_field(value: object, name: str, default: object) -> object:
    if isinstance(value, Mapping):
        return value.get(name, default)
    return getattr(value, name, default)


def _has_field(value: object, name: str) -> bool:
    if isinstance(value, Mapping):
        return name in value
    return hasattr(value, name)


def _token(value: object) -> Hashable:
    try:
        hash(value)
    except TypeError as error:
        raise TypeError("opaque tokens must be hashable") from error
    return value  # type: ignore[return-value]


def _strict_code(value: object, field: str) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise TypeError(f"{field} must be an integer")
    return int(value)


def _pair(value: object, field: str) -> tuple[int, int]:
    if not isinstance(value, (tuple, list)) or len(value) != 2:
        raise TypeError(f"{field} must be an integer pair")
    return (_strict_code(value[0], field), _strict_code(value[1], field))


def _transitions(episode: object) -> tuple[object, ...]:
    return tuple(_field(episode, "transitions"))  # type: ignore[arg-type]


def _training_label(record: object) -> bool:
    """Normalize legacy Boolean truth or v1 generic signed support feedback."""

    has_legacy = _has_field(record, "task_feedback")
    has_support = _has_field(record, "support_feedback")
    if has_legacy == has_support:
        raise AttributeError(
            "training records require exactly one of task_feedback or support_feedback"
        )
    if has_legacy:
        raw = _field(record, "task_feedback")
        if not isinstance(raw, (bool, np.bool_)):
            raise TypeError("task_feedback must be Boolean")
        return bool(raw)
    raw = _field(record, "support_feedback")
    if isinstance(raw, (bool, np.bool_)) or not isinstance(
        raw, (int, float, np.integer, np.floating)
    ):
        raise TypeError("support_feedback must be a finite signed number")
    feedback = float(raw)
    if not np.isfinite(feedback) or not -1.0 <= feedback <= 1.0:
        raise ValueError("support_feedback must be finite and lie in [-1, 1]")
    if feedback == 0.0:
        raise ValueError("support_feedback must be non-zero supervision")
    return feedback > 0.0


def _action_parts(transition: object) -> tuple[int, tuple[int, int], tuple[int, int]]:
    action = _field(transition, "action")
    code = _strict_code(_field(action, "code"), "action code")
    target = _pair(_field(action, "target"), "action target")
    vector = _pair(_optional_field(action, "vector", (0, 0)), "action vector")
    return (code, target, vector)


def _pixels(observation: object) -> np.ndarray:
    array = np.asarray(_field(observation, "pixels"))
    if array.ndim != 3 or array.shape[2] != 3:
        raise ValueError("public observations must be RGB arrays")
    if not np.issubdtype(array.dtype, np.number):
        raise TypeError("public RGB arrays must be numeric")
    result = np.asarray(array, dtype=np.float64)
    if not np.all(np.isfinite(result)):
        raise ValueError("public RGB arrays must be finite")
    if np.issubdtype(array.dtype, np.integer):
        maximum = float(np.iinfo(array.dtype).max)
        if maximum > 1.0:
            result = result / maximum
    return result


def _foreground(frame: np.ndarray) -> np.ndarray:
    channels = frame.shape[2]
    border = np.concatenate(
        (
            frame[0, :, :channels],
            frame[-1, :, :channels],
            frame[:, 0, :channels],
            frame[:, -1, :channels],
        ),
        axis=0,
    )
    background = np.median(border, axis=0)
    distance = np.linalg.norm(frame[:, :, :channels] - background, axis=2)
    # Renderer backgrounds can contain low-amplitude texture.  A fixed
    # normalized threshold rejects it while retaining generated objects.
    return distance > 0.08


def _resample_patch(mask: np.ndarray, target: tuple[int, int], *, bins: int = 17) -> np.ndarray:
    # The ProcessWorld structures sit 36 pixels apart around a neutral probe.
    # A 17-pixel receptive field retains the held-out context relation at
    # distance 15 while excluding the unrelated central probe at distance 18;
    # this blocks absolute left/right layout from dominating local effects.
    radius = 17
    x, y = target
    padded = np.pad(mask.astype(np.float64), radius, mode="constant")
    x += radius
    y += radius
    patch = padded[y - radius : y + radius, x - radius : x + radius]
    edges = np.linspace(0, patch.shape[0], bins + 1, dtype=np.int64)
    result = np.empty((bins, bins), dtype=np.float64)
    for row in range(bins):
        for column in range(bins):
            cell = patch[edges[row] : edges[row + 1], edges[column] : edges[column + 1]]
            result[row, column] = float(np.mean(cell)) if cell.size else 0.0
    return result.reshape(-1)


def _component_count(mask: np.ndarray) -> int:
    pending = np.array(mask, dtype=np.bool_, copy=True)
    components = 0
    height, width = pending.shape
    while np.any(pending):
        first = np.argwhere(pending)[0]
        stack = [(int(first[0]), int(first[1]))]
        pending[stack[0]] = False
        components += 1
        while stack:
            y, x = stack.pop()
            for ny, nx in ((y - 1, x), (y + 1, x), (y, x - 1), (y, x + 1)):
                if 0 <= ny < height and 0 <= nx < width and pending[ny, nx]:
                    pending[ny, nx] = False
                    stack.append((ny, nx))
    return components


def _effect_features(transition: object, target: tuple[int, int]) -> np.ndarray:
    before = _pixels(_field(transition, "before"))
    after = _pixels(_field(transition, "after"))
    if before.shape != after.shape:
        raise ValueError("transition frames must have the same shape")
    changed = np.any(np.abs(before - after) > (0.5 / 255.0), axis=2)
    count = int(np.count_nonzero(changed))
    height, width = changed.shape
    if count:
        ys, xs = np.nonzero(changed)
        geometry = np.array(
            (
                1.0,
                count / changed.size,
                (int(np.max(xs)) - int(np.min(xs)) + 1) / width,
                (int(np.max(ys)) - int(np.min(ys)) + 1) / height,
                min(_component_count(changed), 20) / 20.0,
            ),
            dtype=np.float64,
        )
    else:
        geometry = np.zeros(5, dtype=np.float64)
    before_mask = _foreground(before)
    after_mask = _foreground(after)
    before_patch = _resample_patch(before_mask, target)
    after_patch = _resample_patch(after_mask, target)
    change_patch = _resample_patch(changed, target)
    return np.concatenate((geometry, before_patch, after_patch, change_patch))


def action_target_transcript(
    episode: object,
) -> tuple[tuple[int, tuple[int, int], tuple[int, int], int], ...]:
    """Return the complete outcome-, feedback- and pixel-free action transcript.

    Every opaque code, exact public target coordinate, motor vector and tick
    delta is retained.  This is the adversarial channel that a no-sensor
    control must consume; omitting targets can hide a tracker side channel.
    """

    result: list[tuple[int, tuple[int, int], tuple[int, int], int]] = []
    for transition in _transitions(episode):
        code, target, vector = _action_parts(transition)
        before = _field(transition, "before")
        after = _field(transition, "after")
        tick_delta = _strict_code(_field(after, "tick"), "after tick") - _strict_code(
            _field(before, "tick"), "before tick"
        )
        result.append((code, target, vector, tick_delta))
    return tuple(result)


def _normalized_action_target_skeleton(
    episode: object,
    *,
    include_actions: bool,
) -> tuple[tuple[object, ...], ...]:
    transcript = action_target_transcript(episode)
    if not transcript:
        return ()
    origin = transcript[0][1]
    before = _field(_transitions(episode)[0], "before")
    pixels = _pixels(before)
    height, width = pixels.shape[:2]
    # Coarse absolute origin retains layout correlations while relative target
    # deltas make tracker-output leakage stable across camera translation.
    result: list[tuple[object, ...]] = [
        (
            "origin",
            int(np.floor(8.0 * origin[0] / max(width, 1))),
            int(np.floor(8.0 * origin[1] / max(height, 1))),
            len(transcript),
        )
    ]
    for code, target, vector, tick_delta in transcript:
        relative = (target[0] - origin[0], target[1] - origin[1])
        result.append(
            (code, vector, relative, tick_delta)
            if include_actions
            else (relative, tick_delta)
        )
    return tuple(result)


def _skeleton(episode: object, mode: BinderMode) -> tuple[tuple[object, ...], ...]:
    transitions = _transitions(episode)
    if mode == "target_only":
        return _normalized_action_target_skeleton(episode, include_actions=False)
    if mode in ("no_sensor", "action_target_only"):
        return _normalized_action_target_skeleton(episode, include_actions=True)
    result: list[tuple[object, ...]] = []
    for transition in transitions:
        code, _target, vector = _action_parts(transition)
        if mode == "action_outcome_only":
            raw_outcome = _optional_field(transition, "outcome_code", None)
            outcome = (
                None
                if raw_outcome is None
                else _strict_code(raw_outcome, "outcome code")
            )
            result.append((code, outcome))
        else:
            before = _field(transition, "before")
            after = _field(transition, "after")
            tick_delta = _strict_code(_field(after, "tick"), "after tick") - _strict_code(
                _field(before, "tick"), "before tick"
            )
            if mode == "full":
                before_pixels = _pixels(before)
                after_pixels = _pixels(after)
                if before_pixels.shape != after_pixels.shape:
                    raise ValueError("transition frames must have the same shape")
                sensor_changed: bool | None = bool(
                    np.any(np.abs(before_pixels - after_pixels) > (0.5 / 255.0))
                )
            else:
                sensor_changed = None
            result.append(
                (
                    code,
                    vector,
                    tick_delta,
                    sensor_changed,
                )
            )
    return tuple(result)


def episode_features(episode: object, mode: BinderMode = "full") -> np.ndarray:
    """Return an immutable learner-visible feature vector for an episode."""

    if mode not in (
        "full",
        "no_sensor",
        "action_only",
        "action_target_only",
        "target_only",
        "action_outcome_only",
    ):
        raise ValueError(
            "mode must be full, no_sensor, action_only, action_target_only, "
            "target_only or action_outcome_only"
        )
    transitions = _transitions(episode)
    if mode != "full":
        empty = np.empty(0, dtype=np.float64)
        empty.setflags(write=False)
        return empty
    features: list[np.ndarray] = []
    for transition in transitions:
        _code, target, _vector = _action_parts(transition)
        features.append(_effect_features(transition, target))
    result = np.concatenate(features) if features else np.empty(0, dtype=np.float64)
    result = np.array(result, dtype=np.float64, order="C", copy=True)
    result.setflags(write=False)
    return result


def _distance(left: np.ndarray, right: np.ndarray) -> float:
    if left.shape != right.shape:
        return float("inf")
    if not left.size:
        return 0.0
    return float(np.sqrt(np.mean(np.square(left - right))))


@dataclass(frozen=True, slots=True)
class BinderLedgerEntry:
    token_digest: str
    samples: int
    positives: int
    negatives: int
    skeletons: int
    contradictory_regions: int


@dataclass(frozen=True, slots=True)
class BinderManifest:
    mode: BinderMode
    feature_version: str
    records_seen: int
    token_digests: tuple[str, ...]
    ledger_digest: str

    def digest(self) -> str:
        return _stable_digest(
            (
                self.mode,
                self.feature_version,
                self.records_seen,
                self.token_digests,
                self.ledger_digest,
            )
        )


@dataclass(frozen=True, slots=True)
class _Bucket:
    positives: tuple[np.ndarray, ...]
    negatives: tuple[np.ndarray, ...]
    positive_centroid: np.ndarray | None
    negative_centroid: np.ndarray | None
    contradictory: tuple[np.ndarray, ...]


class EpisodeConceptBinder:
    """Deterministic prototype learner for ordered operational episodes."""

    def __init__(
        self,
        *,
        mode: BinderMode = "full",
        decision_margin: float = 0.05,
        novelty_radius: float = 0.24,
        contradiction_tolerance: float = 1e-12,
    ) -> None:
        if mode not in (
            "full",
            "no_sensor",
            "action_only",
            "action_target_only",
            "target_only",
            "action_outcome_only",
        ):
            raise ValueError(
                "mode must be full, no_sensor, action_only, action_target_only, "
                "target_only or action_outcome_only"
            )
        if not 0.0 <= decision_margin < 1.0:
            raise ValueError("decision_margin must lie in [0, 1)")
        if novelty_radius <= 0.0:
            raise ValueError("novelty_radius must be positive")
        if contradiction_tolerance < 0.0:
            raise ValueError("contradiction_tolerance must be non-negative")
        self.mode = mode
        self.decision_margin = float(decision_margin)
        self.novelty_radius = float(novelty_radius)
        self.contradiction_tolerance = float(contradiction_tolerance)
        self._models: Mapping[Hashable, Mapping[tuple[tuple[object, ...], ...], _Bucket]] = (
            MappingProxyType({})
        )
        self._tokens: tuple[Hashable, ...] = ()
        self.ledger: tuple[BinderLedgerEntry, ...] = ()
        self.manifest = BinderManifest(mode, _FEATURE_VERSION, 0, (), _stable_digest(()))

    def fit(self, records: Iterable[object]) -> "EpisodeConceptBinder":
        grouped: dict[
            Hashable,
            dict[tuple[tuple[object, ...], ...], dict[bool, list[np.ndarray]]],
        ] = defaultdict(lambda: defaultdict(lambda: {True: [], False: []}))
        count = 0
        for record in records:
            token = _token(_field(record, "token"))
            episode = _field(record, "episode")
            label = _training_label(record)
            skeleton = _skeleton(episode, self.mode)
            features = episode_features(episode, self.mode)
            grouped[token][skeleton][label].append(features)
            count += 1

        models: dict[Hashable, Mapping[tuple[tuple[object, ...], ...], _Bucket]] = {}
        ledger: list[BinderLedgerEntry] = []
        tokens = tuple(sorted(grouped, key=_stable_key))
        for token in tokens:
            buckets: dict[tuple[tuple[object, ...], ...], _Bucket] = {}
            positives_total = 0
            negatives_total = 0
            conflict_total = 0
            for skeleton in sorted(grouped[token], key=repr):
                positives = tuple(grouped[token][skeleton][True])
                negatives = tuple(grouped[token][skeleton][False])
                positives_total += len(positives)
                negatives_total += len(negatives)
                contradictory: list[np.ndarray] = []
                for positive in positives:
                    if any(
                        _distance(positive, negative) <= self.contradiction_tolerance
                        for negative in negatives
                    ):
                        contradictory.append(positive)
                conflict_total += len(contradictory)

                def centroid(samples: tuple[np.ndarray, ...]) -> np.ndarray | None:
                    if not samples:
                        return None
                    value = np.mean(np.stack(samples, axis=0), axis=0)
                    value = np.array(value, dtype=np.float64, copy=True)
                    value.setflags(write=False)
                    return value

                buckets[skeleton] = _Bucket(
                    positives,
                    negatives,
                    centroid(positives),
                    centroid(negatives),
                    tuple(contradictory),
                )
            models[token] = MappingProxyType(buckets)
            ledger.append(
                BinderLedgerEntry(
                    token_digest=_stable_digest(_stable_key(token)),
                    samples=positives_total + negatives_total,
                    positives=positives_total,
                    negatives=negatives_total,
                    skeletons=len(buckets),
                    contradictory_regions=conflict_total,
                )
            )

        self._models = MappingProxyType(models)
        self._tokens = tokens
        self.ledger = tuple(ledger)
        ledger_payload = tuple(
            (
                entry.token_digest,
                entry.samples,
                entry.positives,
                entry.negatives,
                entry.skeletons,
                entry.contradictory_regions,
            )
            for entry in self.ledger
        )
        self.manifest = BinderManifest(
            mode=self.mode,
            feature_version=_FEATURE_VERSION,
            records_seen=count,
            token_digests=tuple(entry.token_digest for entry in self.ledger),
            ledger_digest=_stable_digest(ledger_payload),
        )
        return self

    @property
    def tokens(self) -> tuple[Hashable, ...]:
        return self._tokens

    def predict_membership(self, episode: object, token: Hashable) -> bool | None:
        token = _token(token)
        token_model = self._models.get(token)
        if token_model is None:
            return UNKNOWN
        skeleton = _skeleton(episode, self.mode)
        bucket = token_model.get(skeleton)
        if bucket is None:
            return UNKNOWN
        features = episode_features(episode, self.mode)
        if any(
            _distance(features, conflict) <= self.contradiction_tolerance
            for conflict in bucket.contradictory
        ):
            return UNKNOWN

        positive = bucket.positive_centroid
        negative = bucket.negative_centroid
        if positive is not None and negative is not None:
            separation = _distance(positive, negative)
            if separation <= self.contradiction_tolerance:
                return UNKNOWN
            positive_distance = _distance(features, positive)
            negative_distance = _distance(features, negative)
            nearest = min(positive_distance, negative_distance)
            if nearest > max(self.novelty_radius, 2.5 * separation):
                return UNKNOWN
            signed_margin = (negative_distance - positive_distance) / separation
            if abs(signed_margin) < self.decision_margin:
                return UNKNOWN
            return signed_margin > 0.0

        centroid = positive if positive is not None else negative
        if centroid is None:
            return UNKNOWN
        samples = bucket.positives if positive is not None else bucket.negatives
        observed_radius = max((_distance(sample, centroid) for sample in samples), default=0.0)
        allowed = max(self.novelty_radius, 2.5 * observed_radius)
        if _distance(features, centroid) > allowed:
            return UNKNOWN
        return positive is not None

    def supports_token(self, episode: object, token: Hashable) -> bool | None:
        return self.predict_membership(episode, token)

    def predict_token(
        self,
        episode: object,
        candidates: Sequence[Hashable] | None = None,
    ) -> Hashable | None:
        selected = self._tokens if candidates is None else tuple(_token(item) for item in candidates)
        supported = [
            token for token in sorted(set(selected), key=_stable_key) if self.supports_token(episode, token)
        ]
        return supported[0] if len(supported) == 1 else UNKNOWN
