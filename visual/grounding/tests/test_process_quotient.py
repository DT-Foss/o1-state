from __future__ import annotations

import pytest

from grounding_kernel.process_quotient import process_pair_quotient
from grounding_kernel.processworld import OstensiveRecord, ProcessHarness


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
    assert first.redacted_manifest()["problem_commitment"] == first.problem_commitment
    assert first.redacted_manifest()["policy_commitment"] == first.policy_commitment
    assert first.redacted_manifest()["target_commitment"] == first.target_commitment


def test_quotient_pair_validation_fails_closed() -> None:
    left, right = ProcessHarness(81).oracle.affordance_pair()
    wrong_token = OstensiveRecord(right.token + 1, right.episode, right.task_feedback)
    same_label = OstensiveRecord(right.token, right.episode, left.task_feedback)

    with pytest.raises(ValueError, match="one opaque token"):
        process_pair_quotient(left, wrong_token)
    with pytest.raises(ValueError, match="opposite task feedback"):
        process_pair_quotient(left, same_label)


def test_profile_binds_blinded_target_and_executable_policy_commitments() -> None:
    pair = ProcessHarness(82).oracle.process_pair()
    first = process_pair_quotient(*pair, commitment_nonce="sealed-a")
    replay = process_pair_quotient(*pair, commitment_nonce="sealed-a")
    changed_target_blinding = process_pair_quotient(*pair, commitment_nonce="sealed-b")

    assert first.profile_hash == replay.profile_hash
    assert first.problem_commitment == changed_target_blinding.problem_commitment
    assert first.policy_commitment == changed_target_blinding.policy_commitment
    assert first.target_commitment != changed_target_blinding.target_commitment
    assert first.profile_hash != changed_target_blinding.profile_hash
