from __future__ import annotations

from dataclasses import replace
import json

import pytest

import grounding_kernel.v1_benchmark as benchmark_module
from grounding_kernel.v1_benchmark import (
    CLAIM_GRADE,
    DEFAULT_BLOCKS,
    REQUIRED_V1_CONTROLS,
    V1_REQUIRED_AXES,
    V1Thresholds,
    main,
    run_v1_benchmark,
    run_v1_benchmark_from_master_seed,
)


def test_source_commitment_covers_every_package_module_and_project_manifest() -> None:
    manifest = benchmark_module._reference_source_manifest()
    hashes = manifest["source_hashes"]

    assert "grounding_kernel/contracts.py" in hashes
    assert "grounding_kernel/v1_benchmark.py" in hashes
    assert "pyproject.toml" in hashes
    assert len(manifest["model_commitment"]) == 64


def test_integrated_v1_reference_passes_axes_but_makes_no_certificate_claim() -> None:
    result = run_v1_benchmark(seed=3, blocks=DEFAULT_BLOCKS)

    assert result.passed
    assert not result.preregistered_default
    assert not result.certificate_eligible
    assert result.to_dict()["certified"] is False
    assert result.to_dict()["claim_grade"] == CLAIM_GRADE == 2
    assert result.certificate.required_axes == V1_REQUIRED_AXES
    assert {axis.name for axis in result.certificate.axes} == set(V1_REQUIRED_AXES)
    assert all(axis.passed for axis in result.certificate.axes)
    assert all(
        axis.performance.sample_size == DEFAULT_BLOCKS
        for axis in result.certificate.axes
    )
    assert all(control.rejected_as_grounder for control in result.ledger.controls)
    assert result.controls_complete
    assert set(REQUIRED_V1_CONTROLS) == {
        control.name for control in result.ledger.controls
    }
    assert len(result.ledger.control_trials) == (
        DEFAULT_BLOCKS * len(REQUIRED_V1_CONTROLS)
    )
    assert all(
        control.leakage.method == "clopper-pearson-one-sided"
        for control in result.ledger.controls
    )
    assert all(
        values["clusters"] == DEFAULT_BLOCKS
        for values in result.ledger.axis_counts.values()
    )
    assert "reference-self-test" in result.certificate.scope.execution_boundary
    assert all(
        control.leakage.upper_bound <= control.maximum_leakage
        or control.coverage.upper_bound < control.minimum_coverage
        for control in result.ledger.controls
    )
    with pytest.raises(ValueError, match="exactly match"):
        replace(result.ledger, controls=result.ledger.controls[1:])
    with pytest.raises(ValueError, match="one raw control trial"):
        replace(result.ledger, control_trials=result.ledger.control_trials[1:])


def test_small_diagnostic_run_is_deterministic_but_not_preregistered() -> None:
    thresholds = V1Thresholds(
        performance=0.60,
        answer_coverage=0.60,
        shortcut_leakage_ceiling=0.49,
        chance_tolerance=0.49,
    )
    master = bytes(range(32))
    first = run_v1_benchmark_from_master_seed(
        master, blocks=8, thresholds=thresholds
    )
    second = run_v1_benchmark_from_master_seed(
        master, blocks=8, thresholds=thresholds
    )

    assert first.passed and second.passed
    assert not first.preregistered_default and not second.preregistered_default
    assert not first.certificate_eligible
    assert first.certificate.certificate_hash == second.certificate.certificate_hash
    assert first.ledger.ledger_hash == second.ledger.ledger_hash
    assert first.to_dict() == second.to_dict()
    assert first.ledger.seed != 11
    with pytest.raises(TypeError, match="unexpected keyword"):
        run_v1_benchmark(  # type: ignore[call-arg]
            _evaluation_master_commitment="0" * 64
        )


def test_v1_cli_emits_compact_scoped_result(capsys: object) -> None:
    exit_code = main(
        [
            "--seed",
            "11",
            "--blocks",
            "8",
            "--threshold",
            "0.60",
            "--chance-tolerance",
            "0.49",
            "--shortcut-ceiling",
            "0.49",
            "--compact",
        ]
    )
    assert exit_code == 0
    document = json.loads(capsys.readouterr().out)  # type: ignore[attr-defined]
    assert document["passed"] is True
    assert document["certified"] is False
    assert document["certificate_eligible"] is False
    assert document["preregistered_default"] is False
    assert set(document["axes"]) == set(V1_REQUIRED_AXES)
    assert all(item["rejected_as_grounder"] for item in document["controls"].values())
