from __future__ import annotations

import ast
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from grounding_kernel import cross_environment as cross_environment_module
from grounding_kernel.cross_environment import (
    AlignmentReport,
    AlignmentStatus,
    CallbackProbeOracle,
    OperationalLedger,
    OperationalProbeOracle,
    ProbeManifest,
    align_environments,
    capture_operational_ledger,
    gate_transfer,
)
from grounding_kernel.contracts import Action, Observation, Trajectory, Transition
from grounding_kernel.independent_world import IndependentHarness


AlignedPair = tuple[
    IndependentHarness,
    IndependentHarness,
    OperationalLedger,
    OperationalLedger,
    AlignmentReport,
]


def _callback_adapter(
    semantic_actions: tuple[int, int],
    semantic_outcomes: tuple[int, int, int],
    semantic_tokens: tuple[int, int],
    *,
    colours: tuple[tuple[int, int, int], ...],
    target: tuple[int, int],
    token_specific_palette: bool = False,
) -> CallbackProbeOracle:
    manifest = ProbeManifest(
        semantic_actions,
        semantic_outcomes,
        semantic_tokens,
        max_steps=2,
    )

    def run(token: int, program: tuple[int, ...]) -> Trajectory:
        token_role = semantic_tokens.index(token)
        state = 0

        def observation(tick: int) -> Observation:
            pixels = np.empty((6, 6, 3), dtype=np.uint8)
            colour = colours[state]
            if token_specific_palette:
                colour = tuple((channel + 37 * token_role) % 256 for channel in colour)
            pixels[...] = colour
            return Observation(pixels, tick, tick == len(program))

        trajectory = Trajectory(observation(0))
        for tick, code in enumerate(program, start=1):
            action_role = semantic_actions.index(code)
            before = trajectory.current
            if action_role == 0:
                state = 1 if token_role == 0 else 0
                outcome_role = 0
            elif state:
                state = 2
                outcome_role = 1
            else:
                outcome_role = 2
            after = observation(tick)
            trajectory = trajectory.append(
                Transition(
                    before,
                    Action(code, target),
                    after,
                    semantic_outcomes[outcome_role],
                )
            )
        return trajectory

    return CallbackProbeOracle(manifest, run)


@pytest.fixture(scope="module")
def aligned_pair() -> AlignedPair:
    source_harness = IndependentHarness(
        7,
        codebook_variant=1,
        renderer_variant=2,
        world_variant=3,
    )
    target_harness = IndependentHarness(
        99,
        codebook_variant=4,
        renderer_variant=5,
        world_variant=6,
    )
    source = capture_operational_ledger(source_harness.oracle)
    target = capture_operational_ledger(target_harness.oracle)
    report = align_environments(source, target)
    return source_harness, target_harness, source, target, report


def test_callback_adapter_aligns_unrelated_public_runners_without_engine_imports() -> None:
    source_tokens = (181_000_201, 181_000_202)
    target_tokens = (982_000_202, 982_000_201)
    source_adapter = _callback_adapter(
        (181_000_001, 181_000_002),
        (181_000_101, 181_000_102, 181_000_103),
        source_tokens,
        colours=((7, 11, 19), (23, 29, 31), (37, 41, 43)),
        target=(1, 4),
    )
    target_adapter = _callback_adapter(
        (982_000_002, 982_000_001),
        (982_000_103, 982_000_101, 982_000_102),
        target_tokens,
        colours=((211, 199, 193), (181, 179, 173), (167, 163, 157)),
        target=(5, 2),
    )

    assert isinstance(source_adapter, OperationalProbeOracle)
    assert not hasattr(source_adapter, "decode_action")
    assert not hasattr(source_adapter, "object_ids")
    module_path = Path(cross_environment_module.__file__ or "")
    imports = {
        node.module
        for node in ast.walk(ast.parse(module_path.read_text(encoding="utf-8")))
        if isinstance(node, ast.ImportFrom) and node.module is not None
    }
    assert imports.isdisjoint({"independent_world", "processworld", "microworld"})
    source = capture_operational_ledger(source_adapter, program_length=2)
    target = capture_operational_ledger(target_adapter, program_length=2)
    report = align_environments(source, target)

    assert source.raw_hash != target.raw_hash
    assert source.structural_hash == target.structural_hash
    assert report.compatible
    assert report.mapping[source_tokens[0]] == target_tokens[0]
    assert report.mapping[source_tokens[1]] == target_tokens[1]


def test_alignment_rejects_token_specific_initial_worlds() -> None:
    actions = (281_000_001, 281_000_002)
    outcomes = (281_000_101, 281_000_102, 281_000_103)
    tokens = (281_000_201, 281_000_202)
    shared_world = _callback_adapter(
        actions,
        outcomes,
        tokens,
        colours=((7, 11, 19), (23, 29, 31), (37, 41, 43)),
        target=(1, 4),
    )
    token_specific_world = _callback_adapter(
        actions,
        outcomes,
        tokens,
        colours=((7, 11, 19), (23, 29, 31), (37, 41, 43)),
        target=(1, 4),
        token_specific_palette=True,
    )

    shared = capture_operational_ledger(shared_world, program_length=2)
    fractured = capture_operational_ledger(token_specific_world, program_length=2)
    report = align_environments(shared, fractured)

    assert shared.structural_hash != fractured.structural_hash
    assert not report.compatible
    assert all(item.status is AlignmentStatus.INCOMPATIBLE for item in report.tokens)


def test_raw_ledgers_change_but_structural_commitments_survive_every_permutation(
    aligned_pair: AlignedPair,
) -> None:
    source_harness, target_harness, source, target, report = aligned_pair

    assert set(source.action_codes).isdisjoint(target.action_codes)
    assert set(source.outcome_codes).isdisjoint(target.outcome_codes)
    assert set(source.token_codes).isdisjoint(target.token_codes)
    assert not np.array_equal(
        source_harness.agent.observe().pixels,
        target_harness.agent.observe().pixels,
    )
    assert source.raw_hash != target.raw_hash
    assert source.structural_hash == target.structural_hash
    assert len(source.records) == len(source.token_codes) * len(source.action_codes) ** 4
    assert report.compatible
    assert report.witness_count == 2

    raw_step = source.records[0].steps[0]
    assert len(raw_step.before_pixels_hash) == 64
    assert len(raw_step.after_observation_hash) == 64
    assert not hasattr(raw_step, "before_pixels")
    assert source.to_dict()["raw_hash"] == source.raw_hash


def test_alignment_uses_operational_roles_not_integer_identity_or_pixels(
    aligned_pair: AlignedPair,
) -> None:
    source_harness, target_harness, _source, _target, report = aligned_pair
    source_twins = set(source_harness.oracle.nonidentifiable_tokens())
    target_twins = set(target_harness.oracle.nonidentifiable_tokens())

    for alignment in report.tokens:
        if alignment.source_token in source_twins:
            assert alignment.status is AlignmentStatus.UNKNOWN
            assert set(alignment.candidate_target_tokens) == target_twins
            assert alignment.target_token is None
        else:
            assert alignment.status is AlignmentStatus.IDENTIFIED
            assert alignment.target_token is not None
            assert source_harness.oracle.decode_token(
                alignment.source_token
            ) == target_harness.oracle.decode_token(alignment.target_token)


def test_transfer_gate_requires_correct_mapping_and_honest_unknown(
    aligned_pair: AlignedPair,
) -> None:
    source_harness, _target_harness, _source, _target, report = aligned_pair
    claims = dict(report.mapping)
    accepted = gate_transfer(report, claims)

    assert accepted.passed
    assert len(accepted.gate_hash) == 64
    assert all(decision.accepted for decision in accepted.decisions)

    twin = source_harness.oracle.nonidentifiable_tokens()[0]
    guessed = dict(claims)
    guessed[twin] = next(
        item.candidate_target_tokens[0] for item in report.tokens if item.source_token == twin
    )
    overclaim = gate_transfer(report, guessed)
    assert not overclaim.passed
    assert "overstates" in next(
        decision.reason for decision in overclaim.decisions if decision.source_token == twin
    )

    identified = next(item for item in report.tokens if item.status is AlignmentStatus.IDENTIFIED)
    unnecessary_abstention = dict(claims)
    unnecessary_abstention[identified.source_token] = None
    assert not gate_transfer(report, unnecessary_abstention).passed


def test_ledger_and_alignment_hashes_replay_deterministically(
    aligned_pair: AlignedPair,
) -> None:
    source_harness, _target_harness, source, target, report = aligned_pair
    replay = capture_operational_ledger(source_harness.oracle)
    repeated_report = align_environments(source, target)

    assert replay == source
    assert replay.raw_hash == source.raw_hash
    assert replay.structural_hash == source.structural_hash
    assert repeated_report == report
    assert repeated_report.report_hash == report.report_hash


def test_tampered_consequence_breaks_the_cross_environment_gate(
    aligned_pair: AlignedPair,
) -> None:
    _source_harness, _target_harness, source, target, _report = aligned_pair
    record = target.records[0]
    step = record.steps[0]
    replacement_outcome = next(code for code in target.outcome_codes if code != step.outcome_code)
    changed_record = replace(
        record,
        steps=(replace(step, outcome_code=replacement_outcome),) + record.steps[1:],
    )
    records = (changed_record,) + tuple(item for item in target.records if item != record)
    tampered = OperationalLedger(
        target.action_codes,
        target.outcome_codes,
        target.token_codes,
        target.program_length,
        records,
    )
    incompatible = align_environments(source, tampered)

    assert tampered.raw_hash != target.raw_hash
    assert tampered.structural_hash != target.structural_hash
    assert not incompatible.compatible
    assert all(item.status is AlignmentStatus.INCOMPATIBLE for item in incompatible.tokens)
    rejected = gate_transfer(
        incompatible,
        {token: None for token in source.token_codes},
    )
    assert not rejected.passed


def test_ledgers_reject_missing_coverage_and_gates_reject_partial_claims(
    aligned_pair: AlignedPair,
) -> None:
    _source_harness, _target_harness, source, _target, report = aligned_pair
    with pytest.raises(ValueError, match="every token/program"):
        OperationalLedger(
            source.action_codes,
            source.outcome_codes,
            source.token_codes,
            source.program_length,
            source.records[:-1],
        )
    with pytest.raises(ValueError, match="every source token"):
        gate_transfer(report, {source.token_codes[0]: None})
