"""Deterministic binding of opaque symbols to sensorimotor consequences.

The binder is intentionally small and inspectable.  It receives no oracle
state, category name, object identifier or codebook decoder.  Its only inputs
are raw observations, opaque actions, observed outcome codes and optional task
feedback.  A meaning is represented by a prototype distribution over those
operational consequences, not by a privileged label.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Hashable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from hashlib import sha256
from math import ceil, exp, inf, sqrt
from types import MappingProxyType
import numpy as np


_MISSING = object()
_FORBIDDEN_OBSERVATION_FIELDS = frozenset(
    {
        "category",
        "class_name",
        "codebook",
        "decode",
        "label",
        "latent",
        "object_id",
        "oracle",
        "predicate",
        "seed",
        "semantic",
        "token",
    }
)


def _stable_key(value: Hashable) -> tuple[str, str]:
    return (type(value).__qualname__, repr(value))


def _freeze(vector: np.ndarray) -> np.ndarray:
    result = np.array(vector, dtype=np.float64, order="C", copy=True)
    result.setflags(write=False)
    return result


def _field(value: object, names: Sequence[str], default: object = _MISSING) -> object:
    if isinstance(value, Mapping):
        for name in names:
            if name in value:
                return value[name]
    else:
        for name in names:
            if hasattr(value, name):
                return getattr(value, name)
    if default is _MISSING:
        raise AttributeError(f"missing required field: {'/'.join(names)}")
    return default


def _hashable_token(value: object) -> Hashable:
    try:
        hash(value)
    except TypeError as exc:
        raise TypeError("opaque tokens must be hashable") from exc
    return value  # type: ignore[return-value]


def _canonical_bytes(value: object) -> bytes:
    if value is None:
        return b"none"
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, (str, int, float, bool)):
        return f"{type(value).__name__}:{value!r}".encode("utf-8")
    if isinstance(value, Mapping):
        parts = []
        for key in sorted(value, key=lambda item: (type(item).__name__, repr(item))):
            parts.append(_canonical_bytes(key) + b"=" + _canonical_bytes(value[key]))
        return b"{" + b";".join(parts) + b"}"
    if isinstance(value, (tuple, list)):
        return b"[" + b";".join(_canonical_bytes(item) for item in value) + b"]"
    if hasattr(value, "code"):
        return _canonical_bytes(getattr(value, "code"))
    return f"{type(value).__qualname__}:{value!r}".encode("utf-8")


def _categorical(value: object, bins: int) -> np.ndarray:
    result = np.zeros(bins + 1, dtype=np.float64)
    if value is None:
        result[-1] = 1.0
        return result
    digest = sha256(_canonical_bytes(value)).digest()
    index = int.from_bytes(digest[:8], "big") % bins
    result[index] = 1.0
    return result


def _observation_array(observation: object) -> np.ndarray:
    """Extract only learner-visible numeric sensor values.

    The official contract exposes ``Observation.pixels``.  Numeric arrays and
    mappings are accepted to keep the binder protocol-oriented for tests and
    future sensors.  Obvious oracle keys are rejected rather than ignored.
    """

    if hasattr(observation, "pixels"):
        observation = getattr(observation, "pixels")
    elif isinstance(observation, Mapping):
        forbidden = {
            str(key).lower() for key in observation if str(key).lower() in _FORBIDDEN_OBSERVATION_FIELDS
        }
        if forbidden:
            names = ", ".join(sorted(forbidden))
            raise ValueError(f"privileged observation fields are forbidden: {names}")
        if "pixels" in observation:
            observation = observation["pixels"]
        else:
            numeric: list[np.ndarray] = []
            for key in sorted(observation, key=lambda item: (type(item).__name__, repr(item))):
                candidate = np.asarray(observation[key])
                if np.issubdtype(candidate.dtype, np.number) or candidate.dtype == np.bool_:
                    numeric.append(candidate.reshape(-1))
            if not numeric:
                raise TypeError("observation mapping contains no numeric sensors")
            observation = np.concatenate(numeric)
    array = np.asarray(observation)
    if array.size == 0 or not (
        np.issubdtype(array.dtype, np.number) or array.dtype == np.bool_
    ):
        raise TypeError("observations must contain non-empty numeric sensor arrays")
    original_dtype = array.dtype
    result = np.asarray(array, dtype=np.float64)
    if not np.all(np.isfinite(result)):
        raise ValueError("observations must be finite")
    if np.issubdtype(original_dtype, np.unsignedinteger):
        maximum = float(np.iinfo(original_dtype).max)
        if maximum > 1.0:
            result = result / maximum
    return result


def _resample(values: np.ndarray, size: int) -> np.ndarray:
    flat = values.reshape(-1)
    if flat.size == 1:
        return np.full(size, float(flat[0]), dtype=np.float64)
    points = np.linspace(0.0, flat.size - 1.0, size)
    return np.interp(points, np.arange(flat.size), flat)


def _observation_features(observation: object, bins: int) -> np.ndarray:
    array = _observation_array(observation)
    flat = array.reshape(-1)
    quantile_points = np.linspace(0.0, 1.0, 17)
    quantiles = np.quantile(flat, quantile_points)
    mean = float(np.mean(flat))
    std = float(np.std(flat))
    centred = flat - mean
    norm = float(np.linalg.norm(centred))
    frequency = np.abs(np.fft.rfft(centred))
    if frequency.size:
        frequency = frequency / max(norm, 1e-12)
    frequency = _resample(frequency if frequency.size else np.zeros(1), 16)
    denominator = std if std > 1e-12 else 1.0
    standardized = np.clip(centred / denominator, -4.0, 4.0)
    histogram, _ = np.histogram(standardized, bins=16, range=(-4.0, 4.0))
    histogram = histogram.astype(np.float64) / flat.size
    shape = np.zeros(4, dtype=np.float64)
    for index, extent in enumerate(array.shape[:4]):
        shape[index] = np.log1p(extent) / 10.0
    basic = np.array(
        [
            np.log1p(flat.size) / 20.0,
            mean,
            std,
            float(np.min(flat)),
            float(np.max(flat)),
            float(np.mean(np.abs(flat))),
            float(np.linalg.norm(flat) / sqrt(flat.size)),
            float(np.mean(flat != 0.0)),
            float(np.mean(flat > mean)),
            float(np.mean(flat < mean)),
        ],
        dtype=np.float64,
    )
    ordered = _resample(flat, bins)

    # Image-only foreground geometry.  It captures object extent and colour
    # while being far less position-specific than the flattened frame.
    image = np.zeros(19, dtype=np.float64)
    if array.ndim == 3 and array.shape[2] in (1, 3, 4):
        channels = min(array.shape[2], 3)
        border = np.concatenate(
            (array[0, :, :channels], array[-1, :, :channels], array[:, 0, :channels], array[:, -1, :channels]),
            axis=0,
        )
        background = np.median(border, axis=0)
        difference = np.linalg.norm(array[:, :, :channels] - background, axis=2)
        mask = difference > max(0.02, float(np.quantile(difference, 0.6)))
        image[0] = float(np.mean(mask))
        if np.any(mask):
            ys, xs = np.nonzero(mask)
            height, width = array.shape[:2]
            image[1:5] = (
                float(np.mean(xs) / max(width - 1, 1)),
                float(np.mean(ys) / max(height - 1, 1)),
                float((np.max(xs) - np.min(xs) + 1) / width),
                float((np.max(ys) - np.min(ys) + 1) / height),
            )
            foreground = array[:, :, :channels][mask]
            for channel in range(channels):
                offset = 5 + channel * 4
                values = foreground[:, channel]
                image[offset : offset + 4] = (
                    float(np.mean(values)),
                    float(np.std(values)),
                    float(np.quantile(values, 0.25)),
                    float(np.quantile(values, 0.75)),
                )
        image[17] = float(np.mean(difference))
        image[18] = float(np.max(difference))
    return np.concatenate((shape, basic, quantiles, histogram, frequency, ordered, image))


def _action_key(action: object) -> Hashable:
    if hasattr(action, "code"):
        return _hashable_token(getattr(action, "code"))
    if isinstance(action, Mapping) and "code" in action:
        return _hashable_token(action["code"])
    return _hashable_token(action)


def _action_vector(action: object) -> tuple[float, ...]:
    value = _field(action, ("vector",), ())
    try:
        vector = np.asarray(value, dtype=np.float64).reshape(-1)
    except (TypeError, ValueError):
        return ()
    if not np.all(np.isfinite(vector)):
        raise ValueError("action vectors must be finite")
    return tuple(float(item) for item in vector)


def _action_features(action: object, bins: int) -> np.ndarray:
    code = _field(action, ("code",), action)
    categorical = _categorical(code, bins)
    target = _field(action, ("target",), (0, 0))
    vector = _field(action, ("vector",), (0, 0))
    numeric = np.zeros(8, dtype=np.float64)
    for offset, value in ((0, target), (2, vector)):
        try:
            pair = np.asarray(value, dtype=np.float64).reshape(-1)
        except (TypeError, ValueError):
            pair = np.zeros(0)
        count = min(2, pair.size)
        if count:
            numeric[offset : offset + count] = np.tanh(pair[:count] / 16.0)
    numeric[4:6] = np.sin(np.pi * numeric[2:4])
    numeric[6:8] = np.cos(np.pi * numeric[2:4])
    return np.concatenate((categorical, numeric))


@dataclass(frozen=True, slots=True, eq=False)
class ConsequenceSignature:
    """Token-free, immutable operational signature of one transition."""

    vector: np.ndarray
    digest: str
    measurements: tuple[tuple[str, float], ...]

    def __post_init__(self) -> None:
        vector = _freeze(self.vector)
        if vector.ndim != 1 or not np.all(np.isfinite(vector)):
            raise ValueError("signature vectors must be finite and one-dimensional")
        object.__setattr__(self, "vector", vector)
        object.__setattr__(self, "measurements", tuple(self.measurements))

    @property
    def operational_consequences(self) -> Mapping[str, float]:
        return MappingProxyType(dict(self.measurements))


@dataclass(frozen=True, slots=True, eq=False)
class OperationalPrototype:
    """Empirical consequence distribution associated with one opaque token."""

    token: Hashable
    center: np.ndarray
    scale: np.ndarray
    feature_weights: np.ndarray
    count: int
    threshold: float
    calibration_distances: tuple[float, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "center", _freeze(self.center))
        object.__setattr__(self, "scale", _freeze(self.scale))
        object.__setattr__(self, "feature_weights", _freeze(self.feature_weights))
        if self.count < 1:
            raise ValueError("prototype count must be positive")
        if self.threshold <= 0.0:
            raise ValueError("prototype threshold must be positive")

    def distance(self, signature: ConsequenceSignature | np.ndarray) -> float:
        vector = signature.vector if isinstance(signature, ConsequenceSignature) else signature
        values = np.asarray(vector, dtype=np.float64)
        if values.shape != self.center.shape:
            raise ValueError("signature dimensionality does not match the prototype")
        standardized = (values - self.center) / self.scale
        numerator = float(np.sum(self.feature_weights * standardized * standardized))
        denominator = max(float(np.sum(self.feature_weights)), 1e-12)
        return sqrt(max(0.0, numerator / denominator))

    def conformal_p(self, distance: float) -> float:
        if not self.calibration_distances:
            return float(exp(-0.5 * (distance / self.threshold) ** 2))
        count = sum(value >= distance for value in self.calibration_distances)
        return (count + 1.0) / (len(self.calibration_distances) + 1.0)


@dataclass(frozen=True, slots=True)
class TokenScore:
    token: Hashable
    distance: float
    threshold: float
    conformal_p: float
    compatible: bool


@dataclass(frozen=True, slots=True)
class BindingDecision:
    """Prediction with explicit abstention diagnostics."""

    token: Hashable | None
    best_token: Hashable | None
    abstained: bool
    reason: str | None
    distance: float
    threshold: float
    margin: float
    confidence: float
    distances: Mapping[Hashable, float]

    def __post_init__(self) -> None:
        object.__setattr__(self, "distances", MappingProxyType(dict(self.distances)))

    @property
    def accepted(self) -> bool:
        return not self.abstained and self.token is not None


@dataclass(frozen=True, slots=True)
class _Evidence:
    token: Hashable | None
    before: object
    action: object
    after: object
    outcome: object
    feedback: object


def _evidence(value: object, *, require_token: bool) -> _Evidence:
    if isinstance(value, tuple):
        if len(value) == 2:
            token, transition = value
            nested = _evidence(transition, require_token=False)
            return _Evidence(
                _hashable_token(token),
                nested.before,
                nested.action,
                nested.after,
                nested.outcome,
                nested.feedback,
            )
        if len(value) in (4, 5, 6):
            token, before, action, after, *rest = value
            outcome = rest[0] if rest else None
            feedback = rest[1] if len(rest) > 1 else None
            return _Evidence(
                _hashable_token(token), before, action, after, outcome, feedback
            )
    transition = _field(value, ("transition",), value)
    token = _field(value, ("token", "symbol", "instruction_token"), None)
    if require_token and token is None:
        raise TypeError("training evidence requires an opaque token")
    before = _field(transition, ("before", "observation"))
    action = _field(transition, ("action",))
    after = _field(transition, ("after", "next_observation"))
    outcome = _field(transition, ("outcome_code", "outcome"), None)
    feedback = _field(value, ("task_feedback", "feedback", "success", "reward"), None)
    if feedback is None and transition is not value:
        feedback = _field(
            transition, ("task_feedback", "feedback", "success", "reward"), None
        )
    return _Evidence(
        _hashable_token(token) if token is not None else None,
        before,
        action,
        after,
        outcome,
        feedback,
    )


def _is_positive(feedback: object) -> bool | None:
    if feedback is None:
        return None
    try:
        return float(feedback) > 0.0
    except (TypeError, ValueError):
        return bool(feedback)


def _discrete_signature(record: _Evidence) -> tuple[Hashable, tuple[float, ...], Hashable, bool]:
    before = _observation_array(record.before)
    after = _observation_array(record.after)
    changed = before.shape != after.shape or not np.array_equal(before, after)
    outcome = _hashable_token(record.outcome)
    return (_action_key(record.action), _action_vector(record.action), outcome, changed)


class SensorimotorBinder:
    """Prototype grounder for opaque, hashable symbols.

    The learner is deterministic and permutation-equivariant in token names.
    ``alpha`` controls the finite-sample conformal radius and ``min_margin``
    prevents two empirically indistinguishable tokens from being guessed.
    """

    def __init__(
        self,
        *,
        alpha: float = 0.1,
        min_margin: float = 0.08,
        feature_bins: int = 48,
        categorical_bins: int = 31,
        minimum_radius: float = 0.35,
    ) -> None:
        if not 0.0 < alpha < 1.0:
            raise ValueError("alpha must lie in (0, 1)")
        if not 0.0 <= min_margin < 1.0:
            raise ValueError("min_margin must lie in [0, 1)")
        if feature_bins < 8 or categorical_bins < 7:
            raise ValueError("feature_bins and categorical_bins are too small")
        if minimum_radius <= 0.0:
            raise ValueError("minimum_radius must be positive")
        self.alpha = float(alpha)
        self.min_margin = float(min_margin)
        self.feature_bins = int(feature_bins)
        self.categorical_bins = int(categorical_bins)
        self.minimum_radius = float(minimum_radius)
        self._prototypes: dict[Hashable, OperationalPrototype] = {}
        self._action_support: dict[Hashable, dict[Hashable, float]] = {}
        self._exact_support: dict[
            Hashable,
            Counter[tuple[Hashable, tuple[float, ...], Hashable, bool]],
        ] = {}
        self._coarse_support: dict[
            Hashable,
            Counter[tuple[Hashable, Hashable, bool]],
        ] = {}
        self._fitted = False

    def _vector(self, evidence: _Evidence) -> tuple[np.ndarray, np.ndarray, tuple[tuple[str, float], ...]]:
        before_array = _observation_array(evidence.before)
        after_array = _observation_array(evidence.after)
        before = _observation_features(before_array, self.feature_bins)
        after = _observation_features(after_array, self.feature_bins)
        if before_array.shape == after_array.shape:
            delta_array = after_array - before_array
        else:
            length = max(before_array.size, after_array.size)
            before_flat = _resample(before_array, length)
            after_flat = _resample(after_array, length)
            delta_array = after_flat - before_flat
        delta = _observation_features(delta_array, self.feature_bins)
        magnitude = _observation_features(np.abs(delta_array), self.feature_bins)
        action = _action_features(evidence.action, self.categorical_bins)
        outcome = _categorical(evidence.outcome, self.categorical_bins)
        # Task feedback controls whether an example may anchor a prototype; it
        # is deliberately not part of that prototype.  Evaluation transitions
        # do not expose feedback, and encoding it would create a train/test
        # watermark rather than a sensorimotor meaning.
        vector = np.concatenate((before, after, delta, magnitude, action, outcome))
        weights = np.concatenate(
            (
                np.full(before.size, 0.20),
                np.full(after.size, 0.20),
                np.full(delta.size, 1.00),
                np.full(magnitude.size, 0.80),
                np.full(action.size, 1.50),
                np.full(outcome.size, 1.50),
            )
        )
        if before_array.shape == after_array.shape:
            changed = np.abs(after_array - before_array) > 1e-12
            changed_fraction = float(np.mean(changed))
            l1 = float(np.mean(np.abs(after_array - before_array)))
            l2 = float(np.linalg.norm(after_array - before_array) / sqrt(before_array.size))
        else:
            changed_fraction = 1.0
            l1 = float(np.mean(np.abs(delta_array)))
            l2 = float(np.linalg.norm(delta_array) / sqrt(delta_array.size))
        measurements = (
            ("changed_fraction", changed_fraction),
            ("sensor_l1_change", l1),
            ("sensor_l2_change", l2),
            ("feedback_observed", float(evidence.feedback is not None)),
            ("positive_feedback", float(_is_positive(evidence.feedback) is True)),
        )
        return vector, weights, measurements

    def signature(
        self,
        before: object,
        action: object | None = None,
        after: object | None = None,
        *,
        outcome: object = None,
        task_feedback: object = None,
    ) -> ConsequenceSignature:
        """Encode a transition without consulting or embedding its token."""

        if action is None and after is None:
            record = _evidence(before, require_token=False)
        else:
            if action is None or after is None:
                raise TypeError("both action and after are required for a raw transition")
            record = _Evidence(None, before, action, after, outcome, task_feedback)
        vector, _weights, measurements = self._vector(record)
        digest = sha256(vector.astype("<f8", copy=False).tobytes()).hexdigest()
        return ConsequenceSignature(vector, digest, measurements)

    @staticmethod
    def _distance(
        vector: np.ndarray,
        center: np.ndarray,
        scale: np.ndarray,
        weights: np.ndarray,
    ) -> float:
        standardized = (vector - center) / scale
        return sqrt(
            max(
                0.0,
                float(np.sum(weights * standardized * standardized))
                / max(float(np.sum(weights)), 1e-12),
            )
        )

    def _quantile(self, distances: Sequence[float]) -> float:
        ordered = sorted(float(value) for value in distances)
        if not ordered:
            return self.minimum_radius
        rank = min(len(ordered) - 1, max(0, ceil((len(ordered) + 1) * (1.0 - self.alpha)) - 1))
        return max(self.minimum_radius, ordered[rank])

    def fit(
        self,
        experiences: Iterable[object],
        calibration: Iterable[object] | None = None,
    ) -> "SensorimotorBinder":
        """Fit token prototypes from positive or unlabelled trajectories.

        Explicitly failed attempts are retained for action support diagnostics
        but are not averaged into the meaning prototype.  If a token has no
        positive/unlabelled sample at all, fitting fails rather than inventing
        a meaning from negative evidence.
        """

        records = tuple(_evidence(item, require_token=True) for item in experiences)
        if not records:
            raise ValueError("at least one training experience is required")
        grouped: dict[Hashable, list[np.ndarray]] = defaultdict(list)
        weights: np.ndarray | None = None
        action_support: dict[Hashable, dict[Hashable, float]] = defaultdict(
            lambda: defaultdict(float)
        )
        exact_support: dict[
            Hashable,
            Counter[tuple[Hashable, tuple[float, ...], Hashable, bool]],
        ] = defaultdict(Counter)
        coarse_support: dict[
            Hashable,
            Counter[tuple[Hashable, Hashable, bool]],
        ] = defaultdict(Counter)
        for record in records:
            assert record.token is not None
            positive = _is_positive(record.feedback)
            if positive is not False:
                vector, current_weights, _measurements = self._vector(record)
                grouped[record.token].append(vector)
                if weights is None:
                    weights = current_weights
                elif weights.shape != current_weights.shape:
                    raise ValueError("training observations produced inconsistent signatures")
            if positive is not False:
                action_support[record.token][_action_key(record.action)] += 1.0
                discrete = _discrete_signature(record)
                exact_support[record.token][discrete] += 1
                coarse_support[record.token][(discrete[0], discrete[2], discrete[3])] += 1
        tokens_in_records = {record.token for record in records}
        empty = tokens_in_records - set(grouped)
        if empty:
            names = ", ".join(repr(token) for token in sorted(empty, key=_stable_key))
            raise ValueError(f"tokens have only failed evidence: {names}")
        assert weights is not None
        all_vectors = np.vstack([vector for values in grouped.values() for vector in values])
        if all_vectors.shape[0] > 1:
            scale = np.std(all_vectors, axis=0, ddof=1)
        else:
            scale = np.zeros(all_vectors.shape[1], dtype=np.float64)
        positive_scales = scale[scale > 1e-9]
        fallback = max(
            0.05,
            float(np.median(positive_scales)) * 0.25 if positive_scales.size else 0.05,
        )
        scale = np.where(scale > 1e-9, scale, fallback)

        calibration_grouped: dict[Hashable, list[np.ndarray]] = defaultdict(list)
        if calibration is not None:
            for item in calibration:
                record = _evidence(item, require_token=True)
                assert record.token is not None
                if record.token not in grouped or _is_positive(record.feedback) is False:
                    continue
                vector, current_weights, _measurements = self._vector(record)
                if current_weights.shape != weights.shape:
                    raise ValueError("calibration signature shape differs from training")
                calibration_grouped[record.token].append(vector)

        centers = {
            token: np.mean(np.vstack(values), axis=0) for token, values in grouped.items()
        }
        own_distances: dict[Hashable, list[float]] = defaultdict(list)
        for token, values in grouped.items():
            if calibration_grouped[token]:
                own_distances[token] = [
                    self._distance(value, centers[token], scale, weights)
                    for value in calibration_grouped[token]
                ]
            elif len(values) > 1:
                total = np.sum(np.vstack(values), axis=0)
                for value in values:
                    leave_one_out = (total - value) / (len(values) - 1)
                    own_distances[token].append(
                        self._distance(value, leave_one_out, scale, weights)
                    )
            else:
                own_distances[token] = []
        pooled = [distance for values in own_distances.values() for distance in values]
        prototypes: dict[Hashable, OperationalPrototype] = {}
        for token in sorted(grouped, key=_stable_key):
            distances = own_distances[token] or pooled
            threshold = self._quantile(distances)
            prototypes[token] = OperationalPrototype(
                token,
                centers[token],
                scale,
                weights,
                len(grouped[token]),
                threshold,
                tuple(sorted(own_distances[token])),
            )
        self._prototypes = prototypes
        self._action_support = {
            token: dict(scores) for token, scores in action_support.items()
        }
        self._exact_support = {token: Counter(scores) for token, scores in exact_support.items()}
        self._coarse_support = {
            token: Counter(scores) for token, scores in coarse_support.items()
        }
        self._fitted = True
        return self

    @property
    def fitted(self) -> bool:
        return self._fitted

    @property
    def manifest(self) -> Mapping[str, object]:
        """Stable public learner declaration for experiment certificates."""

        return MappingProxyType(
            {
                "name": "deterministic-sensorimotor-binder",
                "version": 1,
                "alpha": self.alpha,
                "min_margin": self.min_margin,
                "feature_bins": self.feature_bins,
                "categorical_bins": self.categorical_bins,
                "minimum_radius": self.minimum_radius,
                "evidence": "raw-before/action/raw-after/opaque-outcome",
                "oracle_access": False,
            }
        )

    @property
    def tokens(self) -> tuple[Hashable, ...]:
        return tuple(sorted(self._prototypes, key=_stable_key))

    @property
    def prototypes(self) -> Mapping[Hashable, OperationalPrototype]:
        return MappingProxyType(dict(self._prototypes))

    def prototype(self, token: Hashable) -> OperationalPrototype:
        if not self._fitted:
            raise RuntimeError("binder is not fitted")
        return self._prototypes[token]

    meaning = prototype

    def _candidate_tokens(
        self, candidates: Iterable[Hashable] | None
    ) -> tuple[Hashable, ...]:
        if not self._fitted:
            raise RuntimeError("binder is not fitted")
        if candidates is None:
            return self.tokens
        requested = {_hashable_token(token) for token in candidates}
        return tuple(sorted(requested & self._prototypes.keys(), key=_stable_key))

    def predict(
        self,
        before: object,
        action: object | None = None,
        after: object | None = None,
        *,
        outcome: object = None,
        task_feedback: object = None,
        candidates: Iterable[Hashable] | None = None,
    ) -> BindingDecision:
        """Return the nearest supported token or an explicit abstention."""

        if action is None and after is None:
            record = _evidence(before, require_token=False)
        else:
            if action is None or after is None:
                raise TypeError("both action and after are required for a raw transition")
            record = _Evidence(None, before, action, after, outcome, task_feedback)
        signature = self.signature(record)
        tokens = self._candidate_tokens(candidates)
        if not tokens:
            return BindingDecision(None, None, True, "no-known-candidates", inf, inf, 0.0, 0.0, {})
        distances = {
            token: self._prototypes[token].distance(signature) for token in tokens
        }
        ordered = sorted(distances, key=lambda token: (distances[token], _stable_key(token)))
        best = ordered[0]
        prototype = self._prototypes[best]
        distance = distances[best]
        second = distances[ordered[1]] if len(ordered) > 1 else inf
        margin = 1.0 if second is inf else max(0.0, (second - distance) / max(second, 1e-12))
        conformal = prototype.conformal_p(distance)
        radial = exp(-0.5 * (distance / prototype.threshold) ** 2)
        confidence = float(min(1.0, sqrt(max(0.0, conformal * radial)) * margin))

        # A finite microworld permits a stronger nonparametric result than a
        # visual centroid: repeated equality of opaque action, observed
        # outcome and actual sensor change is direct operational evidence.
        # This layer is still codebook-blind and token-permutation equivariant.
        discrete = _discrete_signature(record)
        coarse = (discrete[0], discrete[2], discrete[3])
        evidence_scores = {
            token: 2.0 * self._exact_support.get(token, Counter())[discrete]
            + self._coarse_support.get(token, Counter())[coarse]
            for token in tokens
        }
        best_evidence = max(evidence_scores.values(), default=0.0)
        winners = [token for token in tokens if evidence_scores[token] == best_evidence]
        if best_evidence > 0.0 and len(winners) == 1:
            winner = winners[0]
            winner_prototype = self._prototypes[winner]
            runner_up = max(
                (score for token, score in evidence_scores.items() if token != winner),
                default=0.0,
            )
            evidence_margin = (best_evidence - runner_up) / best_evidence
            return BindingDecision(
                winner,
                winner,
                False,
                None,
                distances[winner],
                winner_prototype.threshold,
                evidence_margin,
                float(evidence_margin),
                distances,
            )
        if best_evidence > 0.0:
            return BindingDecision(
                None,
                winners[0],
                True,
                "indistinguishable-operational-signatures",
                distances[winners[0]],
                self._prototypes[winners[0]].threshold,
                0.0,
                0.0,
                distances,
            )
        if distance > prototype.threshold:
            return BindingDecision(
                None,
                best,
                True,
                "outside-calibrated-radius",
                distance,
                prototype.threshold,
                margin,
                confidence,
                distances,
            )
        if len(ordered) > 1 and margin < self.min_margin:
            return BindingDecision(
                None,
                best,
                True,
                "indistinguishable-prototypes",
                distance,
                prototype.threshold,
                margin,
                confidence,
                distances,
            )
        return BindingDecision(
            best,
            best,
            False,
            None,
            distance,
            prototype.threshold,
            margin,
            confidence,
            distances,
        )

    def predict_token(
        self,
        before: object,
        action: object | None = None,
        after: object | None = None,
        *,
        outcome: object = None,
        task_feedback: object = None,
        candidates: Iterable[Hashable] | None = None,
    ) -> Hashable | None:
        return self.predict(
            before,
            action,
            after,
            outcome=outcome,
            task_feedback=task_feedback,
            candidates=candidates,
        ).token

    def score_token(
        self,
        token: Hashable,
        before: object,
        action: object | None = None,
        after: object | None = None,
        *,
        outcome: object = None,
        task_feedback: object = None,
    ) -> TokenScore:
        token = _hashable_token(token)
        prototype = self.prototype(token)
        signature = self.signature(
            before,
            action,
            after,
            outcome=outcome,
            task_feedback=task_feedback,
        )
        distance = prototype.distance(signature)
        return TokenScore(
            token,
            distance,
            prototype.threshold,
            prototype.conformal_p(distance),
            distance <= prototype.threshold,
        )

    def supports_token(self, evidence: object, token: Hashable) -> bool | None:
        """Return intervention support, failed-effect refutation, or unknown."""

        record = _evidence(evidence, require_token=False)
        token = _hashable_token(token)
        if token not in self._prototypes:
            return None
        action_was_learned = _action_key(record.action) in self._action_support.get(token, {})
        if action_was_learned and _is_positive(record.feedback) is False:
            return False
        decision = self.predict(record)
        if decision.accepted:
            return decision.token == token
        if action_was_learned:
            score = self.score_token(token, record)
            if score.distance > score.threshold:
                return False
        return None

    def choose_action(
        self,
        observation: object,
        instruction_token: Hashable,
        candidate_actions: Iterable[object],
    ) -> object | None:
        """Choose an empirically supported action code, otherwise abstain.

        This is intentionally not a policy learner: the observation is unused
        because prototypes cannot predict counterfactual after-states.  The
        method only exposes directly observed token/action support and refuses
        ties.
        """

        _observation_array(observation)  # validates the learner-visible input
        token = _hashable_token(instruction_token)
        support = self._action_support.get(token)
        if not support:
            return None
        candidates = tuple(candidate_actions)
        ranked = [(support.get(_action_key(action), 0.0), index, action) for index, action in enumerate(candidates)]
        ranked.sort(key=lambda item: (-item[0], item[1]))
        if not ranked or ranked[0][0] <= 0.0:
            return None
        if len(ranked) > 1 and ranked[1][0] == ranked[0][0]:
            return None
        return ranked[0][2]

    def evaluate_definition(self, first: object, second: object) -> object:
        """Evaluate a definition through either supported public adapter.

        ``evaluate_definition(definition, resolver)`` is the benchmark-facing
        protocol.  ``evaluate_definition(evidence, definition)`` is a compact
        convenience for a single diagnostic transition.  The former returns
        the full proof object; the latter returns ``bool | None``.
        """

        from .composition import evaluate

        if callable(second):
            return evaluate(first, second)  # type: ignore[arg-type]
        result = evaluate(second, lambda token: self.supports_token(first, token))
        return result.as_python()


Binder = SensorimotorBinder
