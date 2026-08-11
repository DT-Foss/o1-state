from __future__ import annotations

from math import factorial

import pytest

from grounding_kernel.active import AcquisitionCommitments
from grounding_kernel.active_process import (
    OpaqueProbePlan,
    _intervention_derangement,
    _opaque_key,
    run_active_process_block,
)
from grounding_kernel.certificates import manifest_hash
from grounding_kernel.v1_contracts import SessionPhase
from grounding_kernel.v1_controls import ChanceShortcutControl
from grounding_kernel.v1_session import SessionEventKind


def test_active_process_block_uses_real_costed_probes_and_strict_paired_saving() -> None:
    report = run_active_process_block(seed=17, block=0)
    detail = report.detail

    assert report.passed and report.answered
    assert report.hypothesis_count == 4
    assert report.support_worlds == 8
    assert report.candidate_count >= 3
    assert report.active.correct
    assert report.active.acquisition.decision.status == "RESOLVED"
    assert report.active.acquisition.probes_used == 2
    assert report.active.acquisition.cost_used == pytest.approx(4.0)
    assert report.active.session.intervention_cost_used == pytest.approx(
        report.active.acquisition.cost_used
    )
    assert report.active.session.phase is SessionPhase.COMPLETE
    assert sum(
        event.kind is SessionEventKind.EXPERIMENT for event in report.active.session.events
    ) == report.active.acquisition.cost_used
    assert report.paired_cost_saving > 0.0
    assert report.paired_probe_saving > 0.0
    assert detail["strict_saving"] is True

    shuffled = report.intervention_shuffle
    assert shuffled.prediction is None
    assert shuffled.run.acquisition.decision.status == "MODEL_MISSPECIFIED"
    assert shuffled.run.acquisition.ledger.entries[0].predictive_probability == 0.0
    assert shuffled.run.acquisition.cost_used == pytest.approx(2.0)
    assert shuffled.run.session.intervention_cost_used == pytest.approx(2.0)
    assert sum(
        event.kind is SessionEventKind.EXPERIMENT for event in shuffled.run.session.events
    ) == 2
    assert len(shuffled.permutation_commitment) == 64
    assert len(shuffled.evidence_hash) == 64
    assert detail["intervention_shuffle_prediction"] is None


def test_random_orders_are_exact_and_passive_orders_are_counterbalanced() -> None:
    report = run_active_process_block(seed=18, block=1)

    assert len(report.random_orders) == factorial(report.candidate_count)
    assert len(report.passive_orders) == 2
    assert all(run.correct for run in report.random_orders)
    assert all(run.correct for run in report.passive_orders)
    assert report.random_cost > report.active.censored_cost
    assert report.passive_cost > report.active.censored_cost
    experiment_hashes = {
        run.acquisition.ledger.experiment_hash
        for run in (report.active, *report.random_orders, *report.passive_orders)
    }
    assert experiment_hashes == {report.active.acquisition.ledger.experiment_hash}
    assert (
        report.intervention_shuffle.run.acquisition.ledger.experiment_hash
        == report.active.acquisition.ledger.experiment_hash
    )


def test_shuffle_is_a_fixpoint_free_equal_cost_execution_and_inversion_is_leakage() -> None:
    plans = tuple(
        OpaqueProbePlan(
            f"probe-{index}",
            ((701, (0, 0)), (709, (0, 0))),
            (0.1 * index, 0.25),
        )
        for index in range(4)
    )
    derangement = _intervention_derangement(7, 3, plans)

    assert set(derangement) == {plan.key for plan in plans}
    assert set(derangement.values()) == set(derangement)
    assert all(requested != executed for requested, executed in derangement.items())
    assert len({plan.action_count for plan in plans}) == 1

    inverted = ChanceShortcutControl(
        "intervention_consequence_shuffle",
        (False,) * 24,
    )
    assert inverted.orientation_invariant_successes == 24
    assert inverted.leakage.estimate == 1.0
    assert not inverted.rejected_as_grounder


def test_negative_control_is_ambiguous_and_target_is_only_a_commitment() -> None:
    report = run_active_process_block(
        seed=19,
        block=2,
        evaluator_secret=bytes(range(32)),
    )

    assert report.ambiguous_status == "AMBIGUOUS"
    assert report.ambiguous_hypothesis_id is None
    assert len(report.true_signature_commitment) == 64
    assert report.true_signature_commitment == report.commitments.target
    assert "true_hypothesis" not in report.detail
    assert len(report.negative_quotient_profile) == 64
    assert report.commitments.audit_bound
    assert all(
        run.acquisition.ledger.commitments == report.commitments
        for run in (report.active, *report.random_orders, *report.passive_orders)
    )
    design = AcquisitionCommitments(
        report.commitments.problem,
        report.commitments.policy,
    )
    assert report.active.session.model_commitment == design.digest
    assert report.active.session.model_commitment != report.commitments.digest


def test_public_seed_nonce_can_no_longer_enumerate_the_target_signature() -> None:
    seed, block = 29, 2
    report = run_active_process_block(
        seed,
        block,
        evaluator_secret=b"evaluator-private-target-key!!" + b"xx",
    )
    version_space = report.active.acquisition.version_space
    formerly_public_nonce = _opaque_key(seed, "sealed-target-nonce", block)
    recovered = tuple(
        hypothesis.hypothesis_id
        for hypothesis in version_space.hypotheses
        if manifest_hash(
            {
                "nonce": formerly_public_nonce,
                "opaque_token": version_space.token,
                "positive_operational_signature": hypothesis.hypothesis_id,
            }
        )
        == report.commitments.target
    )

    assert recovered == ()
    assert report.active.acquisition.decision.status == "RESOLVED"
    with pytest.raises(ValueError, match="at least 32 private bytes"):
        run_active_process_block(seed, block, evaluator_secret=b"public")


def test_active_process_blocks_are_deterministic_and_structurally_varied() -> None:
    evaluator_secret = b"fixed-evaluator-secret-for-replay"
    first = run_active_process_block(seed=20, block=0, evaluator_secret=evaluator_secret)
    replay = run_active_process_block(seed=20, block=0, evaluator_secret=evaluator_secret)
    variants = {run_active_process_block(seed=20, block=index).candidate_count for index in range(4)}

    assert first.detail == replay.detail
    assert first.active.acquisition.ledger.ledger_hash == replay.active.acquisition.ledger.ledger_hash
    assert variants == {3, 4}
