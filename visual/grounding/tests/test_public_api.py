from __future__ import annotations

import json

import grounding_kernel as kernel
from grounding_kernel.benchmark import main


def test_curated_public_api_exposes_the_complete_v0_path() -> None:
    assert kernel.__version__ == "0.1.0"
    assert kernel.Binder is kernel.SensorimotorBinder
    assert callable(kernel.least_fixed_point)
    assert callable(kernel.run_isolated_learner)
    assert kernel.EvaluatorHarness(1).agent.action_codes


def test_cli_runs_the_real_binder_and_emits_a_compact_certificate(capsys: object) -> None:
    assert main(["--seed", "3", "--episodes", "24", "--learner", "binder", "--compact"]) == 0
    output = capsys.readouterr().out  # type: ignore[attr-defined]
    document = json.loads(output)

    assert document["passed"] is True
    assert document["learner"] == "binder"
    assert set(document["axes"]) == {
        "token_remapping_equivariance",
        "nuisance_transfer",
        "intervention_necessity",
        "unseen_composition",
        "honest_abstention",
    }
    assert all(axis["passed"] for axis in document["axes"].values())
    assert document["controls"]["static_pixels_no_action"]["rejected_as_grounder"] is True
