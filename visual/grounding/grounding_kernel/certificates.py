"""Conservative, scope-bound evidence certificates for operational grounding.

The certificate is deliberately non-compensatory: every registered axis and
its coverage gate must pass.  A high score on one axis can therefore never
hide a failure on another one.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields, is_dataclass
from enum import Enum
from hashlib import sha256
from math import sqrt
from pathlib import Path
from statistics import NormalDist
from types import MappingProxyType
from typing import Any, Iterable, Mapping, Sequence
import base64
import json

import numpy as np


REQUIRED_GROUND_ZERO_AXES = (
    "token_remapping_equivariance",
    "nuisance_transfer",
    "intervention_necessity",
    "unseen_composition",
    "honest_abstention",
)


def _canonical(value: Any) -> Any:
    """Convert manifest data to a deterministic JSON-compatible value."""

    if value is None or isinstance(value, (str, int, bool)):
        return value
    if isinstance(value, float):
        if not np.isfinite(value):
            raise ValueError("manifest floats must be finite")
        return value
    if isinstance(value, np.generic):
        return _canonical(value.item())
    if isinstance(value, np.ndarray):
        return {
            "dtype": str(value.dtype),
            "shape": list(value.shape),
            "data": _canonical(value.tolist()),
        }
    if isinstance(value, bytes):
        return {"base64": base64.b64encode(value).decode("ascii")}
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Enum):
        return _canonical(value.value)
    if is_dataclass(value):
        # Avoid dataclasses.asdict() here: custom mappings and arrays are more
        # reliably handled by this function's own recursion.
        return {field.name: _canonical(getattr(value, field.name)) for field in fields(value)}
    if isinstance(value, Mapping):
        if any(not isinstance(key, str) for key in value):
            raise TypeError("manifest mapping keys must be strings")
        return {key: _canonical(value[key]) for key in sorted(value)}
    if isinstance(value, (set, frozenset)):
        converted = [_canonical(item) for item in value]
        return sorted(converted, key=lambda item: json.dumps(item, sort_keys=True, separators=(",", ":")))
    if isinstance(value, Sequence):
        return [_canonical(item) for item in value]
    raise TypeError(f"unsupported manifest value: {type(value).__name__}")


def canonical_json(value: Any) -> str:
    """Return the canonical UTF-8 JSON representation used by all hashes."""

    return json.dumps(
        _canonical(value),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )


def manifest_hash(value: Any) -> str:
    """Hash explicit manifest data without inspecting arbitrary object state."""

    return sha256(canonical_json(value).encode("utf-8")).hexdigest()


def wilson_lower_bound(successes: int, trials: int, confidence: float = 0.95) -> float:
    """Two-sided Wilson interval's conservative lower endpoint.

    The two-sided critical value is intentional: it is slightly stricter than
    a one-sided interval at the same nominal confidence.
    """

    if trials < 0 or successes < 0 or successes > trials:
        raise ValueError("require 0 <= successes <= trials")
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must lie strictly between zero and one")
    if trials == 0:
        return 0.0
    z = NormalDist().inv_cdf(0.5 + confidence / 2.0)
    n = float(trials)
    proportion = successes / n
    denominator = 1.0 + z * z / n
    centre = proportion + z * z / (2.0 * n)
    radius = z * sqrt((proportion * (1.0 - proportion) + z * z / (4.0 * n)) / n)
    return max(0.0, (centre - radius) / denominator)


def bootstrap_mean_lower_bound(
    samples: Iterable[float],
    confidence: float = 0.95,
    *,
    resamples: int = 10_000,
    seed: int = 0,
) -> float:
    """Deterministic percentile-bootstrap lower bound for bounded scores."""

    values = np.asarray(tuple(samples), dtype=np.float64)
    if values.ndim != 1 or values.size == 0:
        raise ValueError("samples must be a non-empty one-dimensional sequence")
    if not np.all(np.isfinite(values)) or np.any(values < 0.0) or np.any(values > 1.0):
        raise ValueError("samples must be finite and within [0, 1]")
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must lie strictly between zero and one")
    if resamples < 1_000:
        raise ValueError("resamples must be at least 1000")
    rng = np.random.default_rng(seed)
    # Chunking prevents a large evidence ledger from producing an accidental
    # quadratic-looking allocation while preserving exact determinism.
    means = np.empty(resamples, dtype=np.float64)
    chunk = min(2_048, resamples)
    for start in range(0, resamples, chunk):
        stop = min(start + chunk, resamples)
        indices = rng.integers(0, values.size, size=(stop - start, values.size))
        means[start:stop] = values[indices].mean(axis=1)
    try:
        lower = np.quantile(means, 1.0 - confidence, method="lower")
    except TypeError:  # NumPy < 1.22
        lower = np.quantile(means, 1.0 - confidence, interpolation="lower")
    return float(max(0.0, min(float(values.mean()), float(lower))))


@dataclass(frozen=True, slots=True)
class MetricBound:
    """Observed bounded metric and its conservative lower bound."""

    estimate: float
    lower_bound: float
    confidence: float
    sample_size: int
    method: str
    successes: int | None = None

    @classmethod
    def binary(
        cls,
        outcomes: Iterable[bool | int],
        *,
        confidence: float = 0.95,
    ) -> "MetricBound":
        values = tuple(outcomes)
        if any(value not in (False, True, 0, 1) for value in values):
            raise ValueError("binary outcomes must contain only booleans or 0/1")
        successes = sum(bool(value) for value in values)
        sample_size = len(values)
        estimate = successes / sample_size if sample_size else 0.0
        return cls(
            estimate=estimate,
            lower_bound=wilson_lower_bound(successes, sample_size, confidence),
            confidence=confidence,
            sample_size=sample_size,
            method="wilson",
            successes=successes,
        )

    @classmethod
    def scores(
        cls,
        scores: Iterable[float],
        *,
        confidence: float = 0.95,
        resamples: int = 10_000,
        seed: int = 0,
    ) -> "MetricBound":
        values = tuple(float(score) for score in scores)
        if not values:
            return cls(0.0, 0.0, confidence, 0, "bootstrap", None)
        return cls(
            estimate=float(np.mean(values)),
            lower_bound=bootstrap_mean_lower_bound(
                values,
                confidence,
                resamples=resamples,
                seed=seed,
            ),
            confidence=confidence,
            sample_size=len(values),
            method="bootstrap",
            successes=None,
        )

    def __post_init__(self) -> None:
        if not 0.0 <= self.lower_bound <= self.estimate <= 1.0:
            raise ValueError("metric bounds must satisfy 0 <= lower <= estimate <= 1")
        if not 0.0 < self.confidence < 1.0:
            raise ValueError("confidence must lie strictly between zero and one")
        if self.sample_size < 0:
            raise ValueError("sample_size cannot be negative")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class CertificateAxis:
    """A necessary performance claim plus a necessary coverage claim."""

    name: str
    performance: MetricBound
    threshold: float
    baseline: float
    coverage: MetricBound
    coverage_threshold: float
    description: str = ""

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("axis name cannot be empty")
        if not 0.0 <= self.baseline < self.threshold <= 1.0:
            raise ValueError("require 0 <= baseline < threshold <= 1")
        if not 0.0 < self.coverage_threshold <= 1.0:
            raise ValueError("coverage_threshold must lie in (0, 1]")

    @property
    def performance_passed(self) -> bool:
        return self.performance.lower_bound >= self.threshold

    @property
    def coverage_passed(self) -> bool:
        return self.coverage.lower_bound >= self.coverage_threshold

    @property
    def passed(self) -> bool:
        return self.performance_passed and self.coverage_passed

    @property
    def normalized_margin(self) -> float:
        performance_margin = (self.performance.lower_bound - self.baseline) / (
            self.threshold - self.baseline
        )
        coverage_margin = self.coverage.lower_bound / self.coverage_threshold
        return min(performance_margin, coverage_margin)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "threshold": self.threshold,
            "baseline": self.baseline,
            "coverage_threshold": self.coverage_threshold,
            "performance": self.performance.to_dict(),
            "coverage": self.coverage.to_dict(),
            "performance_passed": self.performance_passed,
            "coverage_passed": self.coverage_passed,
            "passed": self.passed,
            "normalized_margin": self.normalized_margin,
        }


def binary_axis(
    name: str,
    outcomes: Iterable[bool | int],
    *,
    threshold: float,
    baseline: float,
    opportunities: Iterable[bool | int] | None = None,
    coverage_threshold: float = 0.8,
    confidence: float = 0.95,
    description: str = "",
) -> CertificateAxis:
    """Build an axis from trial outcomes and explicit opportunity coverage."""

    values = tuple(outcomes)
    if opportunities is None:
        opportunity_values = (True,) * len(values)
    else:
        opportunity_values = tuple(opportunities)
    return CertificateAxis(
        name=name,
        performance=MetricBound.binary(values, confidence=confidence),
        threshold=threshold,
        baseline=baseline,
        coverage=MetricBound.binary(opportunity_values, confidence=confidence),
        coverage_threshold=coverage_threshold,
        description=description,
    )


@dataclass(frozen=True, slots=True)
class CertificateScope:
    """Explicit validity scope and cryptographic experiment manifests."""

    benchmark_id: str
    environment_family: str
    sensor_contract: str
    action_contract: str
    target_language: str
    execution_boundary: str
    manifest_hashes: tuple[tuple[str, str], ...]
    scope_hash: str

    @classmethod
    def from_manifests(
        cls,
        *,
        benchmark_id: str,
        environment_family: str,
        sensor_contract: str,
        action_contract: str,
        target_language: str,
        execution_boundary: str,
        manifests: Mapping[str, Any],
    ) -> "CertificateScope":
        labels = (
            benchmark_id,
            environment_family,
            sensor_contract,
            action_contract,
            target_language,
            execution_boundary,
        )
        if any(not label for label in labels):
            raise ValueError("scope labels cannot be empty")
        if not manifests:
            raise ValueError("at least one manifest is required")
        if any(not isinstance(name, str) for name in manifests):
            raise TypeError("manifest names must be strings")
        hashes = tuple(sorted((name, manifest_hash(value)) for name, value in manifests.items()))
        if any(not name for name, _digest in hashes):
            raise ValueError("manifest names cannot be empty")
        scope_payload = {
            "benchmark_id": benchmark_id,
            "environment_family": environment_family,
            "sensor_contract": sensor_contract,
            "action_contract": action_contract,
            "target_language": target_language,
            "execution_boundary": execution_boundary,
            "manifest_hashes": hashes,
        }
        return cls(
            benchmark_id=benchmark_id,
            environment_family=environment_family,
            sensor_contract=sensor_contract,
            action_contract=action_contract,
            target_language=target_language,
            execution_boundary=execution_boundary,
            manifest_hashes=hashes,
            scope_hash=manifest_hash(scope_payload),
        )

    @property
    def manifests(self) -> Mapping[str, str]:
        return MappingProxyType(dict(self.manifest_hashes))

    def to_dict(self) -> dict[str, Any]:
        return {
            "benchmark_id": self.benchmark_id,
            "environment_family": self.environment_family,
            "sensor_contract": self.sensor_contract,
            "action_contract": self.action_contract,
            "target_language": self.target_language,
            "execution_boundary": self.execution_boundary,
            "manifest_hashes": dict(self.manifest_hashes),
            "scope_hash": self.scope_hash,
        }


@dataclass(frozen=True, slots=True)
class GroundingCertificate:
    """A non-compensatory claim bound to a precise evaluation scope."""

    scope: CertificateScope
    axes: tuple[CertificateAxis, ...]
    required_axes: tuple[str, ...] = REQUIRED_GROUND_ZERO_AXES

    def __post_init__(self) -> None:
        names = tuple(axis.name for axis in self.axes)
        if len(set(names)) != len(names):
            raise ValueError("axis names must be unique")
        if len(set(self.required_axes)) != len(self.required_axes):
            raise ValueError("required axis names must be unique")

    @property
    def missing_axes(self) -> tuple[str, ...]:
        present = {axis.name for axis in self.axes}
        return tuple(name for name in self.required_axes if name not in present)

    @property
    def failed_axes(self) -> tuple[str, ...]:
        return tuple(axis.name for axis in self.axes if not axis.passed)

    @property
    def passed(self) -> bool:
        return bool(self.axes) and not self.missing_axes and not self.failed_axes

    @property
    def score(self) -> float:
        """Worst normalized axis margin; descriptive, never compensatory."""

        if not self.axes or self.missing_axes:
            return 0.0
        return min(axis.normalized_margin for axis in self.axes)

    @property
    def certificate_hash(self) -> str:
        return manifest_hash(
            {
                "scope": self.scope.to_dict(),
                "axes": [axis.to_dict() for axis in self.axes],
                "required_axes": self.required_axes,
            }
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": "operational-grounding-certificate",
            "scope": self.scope.to_dict(),
            "axes": [axis.to_dict() for axis in self.axes],
            "required_axes": list(self.required_axes),
            "missing_axes": list(self.missing_axes),
            "failed_axes": list(self.failed_axes),
            "passed": self.passed,
            "score": self.score,
            "certificate_hash": self.certificate_hash,
        }
