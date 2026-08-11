from __future__ import annotations

from grounding_kernel.process_quotient import process_pair_quotient
from grounding_kernel.processworld import ProcessHarness


def test_negative_control_is_proven_ungroundable_in_declared_intervention_family() -> None:
    harness = ProcessHarness(909)
    report = process_pair_quotient(*harness.oracle.negative_control_pair())

    assert not report.groundable
    assert report.groundability.witness is not None
    assert len(report.groundability.quotient) == 1
    assert report.redacted_manifest()["witness_present"] is True
    assert "left_value" not in report.redacted_manifest()
    assert "right_value" not in report.redacted_manifest()


def test_raw_context_and_perturbed_process_are_identifiable_without_outcome_codes() -> None:
    harness = ProcessHarness(910)
    context = process_pair_quotient(*harness.oracle.context_pair())
    process = process_pair_quotient(*harness.oracle.process_pair())

    assert context.groundable
    assert process.groundable
    assert len(context.groundability.quotient) == 2
    assert len(process.groundability.quotient) == 2
    assert context.profile_hash != process.profile_hash


def test_profile_manifest_is_deterministic_and_renderer_sensitive_only_via_public_records() -> None:
    first = process_pair_quotient(*ProcessHarness(77, renderer_variant=3).oracle.context_pair())
    replay = process_pair_quotient(*ProcessHarness(77, renderer_variant=3).oracle.context_pair())
    changed = process_pair_quotient(*ProcessHarness(77, renderer_variant=4).oracle.context_pair())

    assert first.profile_hash == replay.profile_hash
    assert first.profile_hash != changed.profile_hash
    assert first.redacted_manifest()["declared_interventions"] == [
        "observe-v1",
        "declared-policy-v1",
    ]
