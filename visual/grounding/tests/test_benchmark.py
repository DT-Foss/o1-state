from __future__ import annotations

from collections import Counter
from dataclasses import replace
import json

import pytest

from grounding_kernel.benchmark import (
    BENCHMARK_ID,
    GroundingEvidence,
    build_ground_zero_dataset,
    run_benchmark,
)
from grounding_kernel.certificates import REQUIRED_GROUND_ZERO_AXES
from grounding_kernel.certificates import MetricBound


def _public_transition_digest(record: GroundingEvidence) -> tuple[object, ...]:
    return (
        record.before.digest(),
        record.action.code,
        record.action.vector,
        record.after.digest(),
        record.outcome_code,
    )


def test_ground_zero_reference_is_deterministic_and_passes_every_gate() -> None:
    first = run_benchmark(seed=17, episodes=24)
    second = run_benchmark(seed=17, episodes=24)

    assert first.passed
    assert tuple(axis.name for axis in first.certificate.axes) == REQUIRED_GROUND_ZERO_AXES
    assert all(axis.passed for axis in first.certificate.axes)
    assert first.ledger.ledger_hash == second.ledger.ledger_hash
    assert first.certificate.certificate_hash == second.certificate.certificate_hash
    assert first.ledger.controls[0].name == "static_pixels_no_action"
    assert first.ledger.controls[0].rejected_as_grounder
    assert "subprocess" in first.certificate.scope.execution_boundary
    assert json.loads(json.dumps(first.to_dict()))["benchmark_id"] == BENCHMARK_ID

    leaked_control = replace(
        first.ledger.controls[0],
        performance=MetricBound.binary([True] * 24),
    )
    control_leak = replace(
        first,
        ledger=replace(first.ledger, controls=(leaked_control,)),
    )
    assert control_leak.certificate.passed
    assert not control_leak.passed


def test_shuffle_ablation_preserves_marginals_but_breaks_pairing() -> None:
    dataset = build_ground_zero_dataset(seed=23, episodes=12)

    assert Counter(record.token for record in dataset.train) == Counter(
        record.token for record in dataset.train_intervention_shuffled
    )
    assert Counter(_public_transition_digest(record) for record in dataset.train) == Counter(
        _public_transition_digest(record)
        for record in dataset.train_intervention_shuffled
    )
    assert all(
        original.token == shuffled.token
        and _public_transition_digest(original) != _public_transition_digest(shuffled)
        for original, shuffled in zip(
            dataset.train, dataset.train_intervention_shuffled, strict=True
        )
    )
    remapping = dict(dataset.token_remapping)
    assert set(remapping) == set(remapping.values())
    assert all(source != target for source, target in remapping.items())
    ordered = tuple(sorted(remapping))
    assert tuple(remapping[token] for token in ordered) != ordered[1:] + ordered[:1]


class _AlwaysAbstain:
    manifest = {"name": "always-abstain-negative-control"}

    def __init__(self, _seed: int) -> None:
        pass

    def fit(self, experiences: object) -> "_AlwaysAbstain":
        return self

    def predict_token(self, evidence: object, *, candidates: object = None) -> None:
        return None


def test_abstention_cannot_game_identifiable_axes_or_coverage() -> None:
    result = run_benchmark(seed=31, episodes=24, learner_factory=_AlwaysAbstain)

    assert not result.passed
    assert "token_remapping_equivariance" in result.certificate.failed_axes
    assert "nuisance_transfer" in result.certificate.failed_axes
    assert "intervention_necessity" in result.certificate.failed_axes
    assert "unseen_composition" in result.certificate.failed_axes
    honest = next(
        axis for axis in result.certificate.axes if axis.name == "honest_abstention"
    )
    assert honest.passed
    assert honest.coverage.estimate == 1.0
    nuisance = next(axis for axis in result.certificate.axes if axis.name == "nuisance_transfer")
    assert nuisance.coverage.estimate == 0.0


def test_learner_payloads_contain_no_evaluator_capability() -> None:
    import grounding_kernel.benchmark as benchmark_module

    learners: list[object] = []

    class BoundaryAuditor:
        manifest = {"name": "boundary-auditor"}

        def __init__(self, seed: int) -> None:
            self.inner = benchmark_module._ReferenceTransitionLearner(seed)
            learners.append(self)

        def fit(self, experiences: object) -> "BoundaryAuditor":
            records = tuple(experiences)
            forbidden = {
                "oracle",
                "seed",
                "codebook",
                "predicate",
                "object_id",
                "latent",
                "snapshot",
            }
            assert records
            assert all(isinstance(record, GroundingEvidence) for record in records)
            assert all(not (set(dir(record)) & forbidden) for record in records)
            self.inner.fit(records)
            return self

        def predict_token(self, evidence: object, **kwargs: object) -> object:
            forbidden = {"oracle", "codebook", "predicate", "object_id", "latent"}
            assert not (set(dir(evidence)) & forbidden)
            return self.inner.predict_token(evidence, **kwargs)

        def supports_token(self, evidence: object, token: int) -> bool | None:
            return self.inner.supports_token(evidence, token)

    result = run_benchmark(seed=41, episodes=8, learner_factory=BoundaryAuditor)

    assert len(learners) == 3
    # Eight trials do not provide enough Wilson evidence for certification;
    # the boundary audit itself still ran over every learner call.
    assert not result.passed


def test_sealed_train_and_test_manifests_are_disjoint() -> None:
    dataset = build_ground_zero_dataset(seed=53, episodes=12)
    train_digests = {record.before.digest() for record in dataset.train}
    test_digests = {case.transition.before.digest() for case in dataset.nuisance_cases}

    assert train_digests.isdisjoint(test_digests)
    assert dataset.split_manifest["train_namespace"] == "train-world"
    assert "sealed-test-world" in dataset.split_manifest["test_namespaces"]
    assert all(case.invariant_verified for case in dataset.abstention_cases)


def test_too_few_episodes_are_rejected() -> None:
    with pytest.raises(ValueError, match="at least 8"):
        build_ground_zero_dataset(seed=0, episodes=7)


def test_sensorimotor_binder_passes_the_registered_v0_scope() -> None:
    from grounding_kernel.binder import SensorimotorBinder
    from grounding_kernel.microworld import WorldConfig

    result = run_benchmark(
        seed=3,
        episodes=24,
        world_config=WorldConfig(object_count=6, max_steps=8),
        learner_factory=lambda _seed: SensorimotorBinder(),
    )

    assert tuple(axis.name for axis in result.certificate.axes) == REQUIRED_GROUND_ZERO_AXES
    abstention = next(
        axis for axis in result.certificate.axes if axis.name == "honest_abstention"
    )
    assert abstention.performance.estimate == 1.0
    assert result.passed
