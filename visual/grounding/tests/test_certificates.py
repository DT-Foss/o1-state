from __future__ import annotations

import json

import numpy as np
import pytest

from grounding_kernel.certificates import (
    REQUIRED_GROUND_ZERO_AXES,
    CertificateScope,
    GroundingCertificate,
    MetricBound,
    binary_axis,
    bootstrap_mean_lower_bound,
    manifest_hash,
    wilson_lower_bound,
)


def _scope(seed: int = 7) -> CertificateScope:
    return CertificateScope.from_manifests(
        benchmark_id="ground-zero-v0",
        environment_family="finite-causal-grid-v0",
        sensor_contract="rgb64+tick+terminal",
        action_contract="opaque-discrete-actions-v0",
        target_language="opaque-tokens+typed-conjunction-v0",
        execution_boundary="serialized-subprocess-v0",
        manifests={
            "learner": {"name": "counting-reference", "version": 1},
            "train_split": {"seed": seed, "episodes": list(range(16))},
            "test_split": {"seed": seed + 1, "episodes": list(range(16, 32))},
        },
    )


def test_manifest_hash_is_order_independent_but_content_sensitive() -> None:
    left = {"b": [2, 3], "a": np.array([1], dtype=np.int16)}
    right = {"a": np.array([1], dtype=np.int16), "b": [2, 3]}

    assert manifest_hash(left) == manifest_hash(right)
    assert manifest_hash(left) != manifest_hash({"a": [1], "b": [2, 3]})
    with pytest.raises(ValueError, match="finite"):
        manifest_hash({"bad": float("nan")})
    with pytest.raises(TypeError, match="keys must be strings"):
        manifest_hash({1: "a"})
    assert manifest_hash({"1": "a"}) != manifest_hash([[1, "a"]])


def test_wilson_bound_is_conservative_and_handles_empty_evidence() -> None:
    assert wilson_lower_bound(0, 0) == 0.0
    assert 0.0 < wilson_lower_bound(90, 100) < 0.9
    assert wilson_lower_bound(100, 100) < 1.0
    with pytest.raises(ValueError):
        wilson_lower_bound(2, 1)


def test_bootstrap_bound_is_deterministic_and_below_mean() -> None:
    samples = [0.0, 0.3, 0.8, 0.9, 1.0, 1.0]
    first = bootstrap_mean_lower_bound(samples, resamples=2_000, seed=99)
    second = bootstrap_mean_lower_bound(samples, resamples=2_000, seed=99)

    assert first == second
    assert 0.0 <= first <= np.mean(samples)
    assert MetricBound.scores(samples, resamples=2_000, seed=99).method == "bootstrap"


def test_certificate_is_noncompensatory_and_enforces_coverage() -> None:
    axes = []
    for name in REQUIRED_GROUND_ZERO_AXES:
        opportunities = [True] * 400
        if name == "unseen_composition":
            opportunities[-120:] = [False] * 120
        axes.append(
            binary_axis(
                name,
                [True] * 400,
                threshold=0.90,
                baseline=0.50,
                opportunities=opportunities,
                coverage_threshold=0.80,
            )
        )

    certificate = GroundingCertificate(_scope(), tuple(axes))

    assert not certificate.passed
    assert certificate.failed_axes == ("unseen_composition",)
    assert certificate.axes[3].performance_passed
    assert not certificate.axes[3].coverage_passed
    assert certificate.score < 1.0


def test_certificate_requires_every_registered_axis_and_serializes() -> None:
    passing_axes = tuple(
        binary_axis(
            name,
            [True] * 500,
            threshold=0.90,
            baseline=0.50,
            coverage_threshold=0.90,
        )
        for name in REQUIRED_GROUND_ZERO_AXES
    )
    complete = GroundingCertificate(_scope(), passing_axes)
    incomplete = GroundingCertificate(_scope(), passing_axes[:-1])

    assert complete.passed
    assert complete.score >= 1.0
    assert incomplete.missing_axes == ("honest_abstention",)
    assert not incomplete.passed
    document = complete.to_dict()
    assert json.loads(json.dumps(document))["passed"] is True
    assert len(document["certificate_hash"]) == 64
    assert complete.certificate_hash != GroundingCertificate(_scope(8), passing_axes).certificate_hash
