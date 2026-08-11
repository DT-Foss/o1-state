from __future__ import annotations

from dataclasses import FrozenInstanceError, replace
import json

import pytest

from grounding_kernel.v1_grade3_benchmark import (
    GRADE3_BENCHMARK_PROTOCOL,
    GRADE3_NEGATIVE_CONTROLS,
    GRADE3_POSITIVE_AXES,
    HONEST_GRADE3_CLAIM,
    Grade3BenchmarkReport,
    grade3_benchmark_manifest,
    run_grade3_benchmark,
)


@pytest.fixture(scope="module")
def benchmark_report() -> Grade3BenchmarkReport:
    return run_grade3_benchmark(1, support_worlds=2)


def test_generic_manifest_is_codebook_independent_and_exact() -> None:
    manifest = grade3_benchmark_manifest(2)
    assert manifest.support_record_budget == 43
    assert manifest.acquisition_cost_budget == 8.0
    assert manifest.query_budget == 16
    assert manifest.motor_action_cost_budget == 64.0
    assert manifest.motor_reset_budget == 20
    with pytest.raises(ValueError, match="at least 2"):
        grade3_benchmark_manifest(1)
    with pytest.raises(TypeError, match="integer"):
        grade3_benchmark_manifest(True)  # type: ignore[arg-type]


def test_report_closes_every_noncompensatory_axis_and_control(
    benchmark_report: Grade3BenchmarkReport,
) -> None:
    report = benchmark_report
    assert report.protocol_version == GRADE3_BENCHMARK_PROTOCOL
    assert report.passed
    assert report.noncompensatory
    assert tuple(check.name for check in report.axes) == GRADE3_POSITIVE_AXES
    assert tuple(check.name for check in report.controls) == GRADE3_NEGATIVE_CONTROLS
    assert all(check.passed for check in (*report.axes, *report.controls))
    assert report.support_records_consumed == 43
    assert report.acquisition_cost_consumed == pytest.approx(8.0)
    assert report.sealed_queries_consumed == 14
    assert report.motor_action_cost_consumed == pytest.approx(45.0)
    assert report.motor_resets_consumed == 17
    assert report.claim_scope == HONEST_GRADE3_CLAIM
    assert not report.universal_certificate
    assert not report.adversarial_certificate
    assert len(report.report_hash) == 64
    for field in (
        report.artifact_commitment,
        report.sdk_commitment,
        report.checkpoint_commitment,
        report.dataset_commitment,
        report.manifest_commitment,
        report.schema_commitment,
        report.ledger_commitment,
        report.benchmark_definition_commitment,
    ):
        assert len(field) == 64
        bytes.fromhex(field)
    # The exported report is canonical-data friendly, with no process object,
    # evaluator oracle, callback, or raw private truth attached.
    encoded = json.dumps(report.to_dict(), sort_keys=True, separators=(",", ":"))
    assert report.report_hash in encoded
    assert "oracle" not in encoded


def test_same_seed_replays_identical_hashes(
    benchmark_report: Grade3BenchmarkReport,
) -> None:
    replay = run_grade3_benchmark(1, support_worlds=2)
    assert replay == benchmark_report
    assert replay.report_hash == benchmark_report.report_hash
    assert replay.ledger_commitment == benchmark_report.ledger_commitment
    assert replay.checkpoint_commitment == benchmark_report.checkpoint_commitment
    assert replay.dataset_commitment == benchmark_report.dataset_commitment


def test_report_is_immutable_and_one_failure_cannot_be_compensated(
    benchmark_report: Grade3BenchmarkReport,
) -> None:
    with pytest.raises(FrozenInstanceError):
        benchmark_report.seed = 2  # type: ignore[misc]
    failed_axis = replace(benchmark_report.axes[0], passed=False)
    failed = replace(
        benchmark_report,
        axes=(failed_axis, *benchmark_report.axes[1:]),
    )
    assert not failed.passed
    assert failed.report_hash != benchmark_report.report_hash
    assert all(check.passed for check in failed.axes[1:])
    assert all(check.passed for check in failed.controls)


def test_report_rejects_claim_or_axis_widening(
    benchmark_report: Grade3BenchmarkReport,
) -> None:
    with pytest.raises(ValueError, match="positive axes"):
        replace(
            benchmark_report,
            axes=tuple(reversed(benchmark_report.axes)),
        )
    with pytest.raises(ValueError, match="claim_scope"):
        replace(benchmark_report, claim_scope="universal certificate")


def test_renderer_token_permutation_and_symmetric_probe_world_still_pass() -> None:
    report = run_grade3_benchmark(
        2,
        renderer_variant=2,
        permutation_variant=3,
        support_worlds=2,
    )
    assert report.passed
    assert all(check.passed for check in (*report.axes, *report.controls))
