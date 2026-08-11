from __future__ import annotations

from dataclasses import dataclass

from grounding_kernel.certificates import manifest_hash
from grounding_kernel.closed_loop_programs import (
    ClosedLoopProgramExecutor,
    ClosedLoopProgramRecognizer,
    PerceptualTargetRole,
    referent_registry,
)
from grounding_kernel.episode_binder import EpisodeConceptBinder
from grounding_kernel.language import Demonstration, GroundedLanguageLearner
from grounding_kernel.perceptual_policy import VisualTargetSelector
from grounding_kernel.processworld import ProcessConceptKind, ProcessHarness
from grounding_kernel.programs import ActionScheme, ProgramSchema
from grounding_kernel.v1_adapters import (
    episode_to_query_trace,
    ostensive_record_to_support,
    support_episode_to_binder_record,
)


TARGET_TYPE = 610_000_101
SCHEME_TYPE = 610_000_103
TOKEN_SELF = 710_000_107
TOKEN_EXTERNAL = 710_000_109
TOKEN_MOVE = 710_000_111
TOKEN_RUN = 710_000_113


@dataclass(frozen=True)
class _Fixture:
    seed: int
    schema: ProgramSchema
    roles: tuple[PerceptualTargetRole, PerceptualTargetRole]
    schemes: tuple[ActionScheme, ActionScheme]
    referents: object
    learner: GroundedLanguageLearner
    held_out: tuple[int, int]
    training_targets: tuple[tuple[int, int], ...]


def _fixture(seed: int = 119) -> _Fixture:
    binder_records = []
    process_traces = []
    evidence_digests = []
    representative_process = None
    representative_movement = None
    training_targets = []
    process_token = None
    turn_id = 0
    for variant in (0, 1, 2, 3):
        harness = ProcessHarness(seed, renderer_variant=variant, world_variant=variant)
        examples = harness.oracle.examples(include_negative_control=True)
        for record in examples:
            support = ostensive_record_to_support(
                record,
                turn_id=turn_id,
                remaining_cost=100.0,
            )
            binder_records.append(support_episode_to_binder_record(support))
            turn_id += 1
        positive, negative = harness.oracle.process_pair()
        current = (
            episode_to_query_trace(positive.episode),
            episode_to_query_trace(negative.episode),
        )
        process_traces.extend(current)
        training_targets.extend(trace.transitions[0].action.target for trace in current)
        evidence_digests.extend(
            manifest_hash(
                {
                    "initial": trace.initial.digest(),
                    "after": [step.after.digest() for step in trace.transitions],
                    "actions": [step.action.code for step in trace.transitions],
                }
            )
            for trace in current
        )
        process_token = positive.token
        representative_process = positive
        representative_movement = harness.oracle.movement_pair()[0]

    assert representative_process is not None
    assert representative_movement is not None
    assert process_token is not None
    binder = EpisodeConceptBinder(mode="full").fit(binder_records)
    selector = VisualTargetSelector.from_traces(process_traces)
    running = ActionScheme.from_episode(
        episode_to_query_trace(representative_process.episode)
    )
    moving = ActionScheme.from_episode(
        episode_to_query_trace(representative_movement.episode)
    )
    positive_role = PerceptualTargetRole.from_support(
        selector=selector,
        binder=binder,
        token=process_token,
        diagnostic_scheme=running,
        required_membership=True,
        evidence_digests=evidence_digests,
    )
    negative_role = PerceptualTargetRole.from_support(
        selector=selector,
        binder=binder,
        token=process_token,
        diagnostic_scheme=running,
        required_membership=False,
        evidence_digests=evidence_digests,
    )
    schema = ProgramSchema(TARGET_TYPE, SCHEME_TYPE)
    referents = referent_registry(
        schema,
        {TOKEN_SELF: positive_role, TOKEN_EXTERNAL: negative_role},
        {TOKEN_MOVE: moving, TOKEN_RUN: running},
    )
    held_out = (TOKEN_SELF, TOKEN_RUN)
    training = tuple(
        Demonstration(tokens, referent, evidence=("public-support", index))
        for index, (tokens, referent) in enumerate(referents.items())
        if tokens != held_out
    )
    learner = GroundedLanguageLearner().fit(training)
    return _Fixture(
        seed,
        schema,
        (positive_role, negative_role),
        (moving, running),
        referents,
        learner,
        held_out,
        tuple(training_targets),
    )


def _holdout(case: _Fixture) -> ProcessHarness:
    return ProcessHarness(
        case.seed,
        renderer_variant=91_337,
        world_variant=27_119,
    )


def test_withheld_words_drive_closed_loop_perception_and_new_world_behavior() -> None:
    case = _fixture()
    interpretation = case.learner.interpret_instruction(case.held_out)
    assert interpretation.resolved and interpretation.referent is not None
    harness = _holdout(case)

    execution = ClosedLoopProgramExecutor(case.schema).execute(
        harness.agent,
        interpretation.referent,
    )

    assert execution.resolved and execution.trace is not None
    assert len(execution.evidence) >= 2
    assert execution.actions_executed > len(execution.trace.transitions)
    assert all(not hasattr(step, "outcome_code") for step in execution.trace.transitions)
    assert harness.oracle.matches_concept(
        harness.agent.episode(), ProcessConceptKind.RUNNING
    )
    assert not hasattr(case.roles[0], "targets")
    assert execution.trace.transitions[0].action.target in case.roles[0].selector.candidates(
        execution.trace.initial
    )


def test_each_composed_slot_changes_executed_world_consequence() -> None:
    case = _fixture()
    executor = ClosedLoopProgramExecutor(case.schema)
    self_run = executor.execute(_holdout(case).agent, case.referents[(TOKEN_SELF, TOKEN_RUN)])

    external_harness = _holdout(case)
    external_run = executor.execute(
        external_harness.agent,
        case.referents[(TOKEN_EXTERNAL, TOKEN_RUN)],
    )
    moving_harness = _holdout(case)
    self_move = executor.execute(
        moving_harness.agent,
        case.referents[(TOKEN_SELF, TOKEN_MOVE)],
    )

    assert self_run.resolved and self_run.trace is not None
    assert external_run.resolved and external_run.trace is not None
    assert not external_harness.oracle.matches_concept(
        external_harness.agent.episode(), ProcessConceptKind.RUNNING
    )
    assert self_move.resolved and self_move.trace is not None
    assert moving_harness.oracle.matches_concept(
        moving_harness.agent.episode(), ProcessConceptKind.MOVING
    )
    assert not moving_harness.oracle.matches_concept(
        moving_harness.agent.episode(), ProcessConceptKind.RUNNING
    )


def test_feedback_free_world_trace_is_recognized_then_described() -> None:
    case = _fixture()
    harness = _holdout(case)
    target = case.referents[case.held_out]
    execution = ClosedLoopProgramExecutor(case.schema).execute(harness.agent, target)
    assert execution.trace is not None and not execution.trace.has_feedback
    recognizer = ClosedLoopProgramRecognizer(case.schema, case.roles, case.schemes)

    recognized = recognizer.recognize(execution.trace)
    description = case.learner.describe(recognized.referent)  # type: ignore[arg-type]

    assert recognized.resolved and recognized.referent == target
    assert description.resolved
    assert description.utterance == case.held_out


def test_unknown_visual_scene_fails_closed_without_coordinate_fallback() -> None:
    case = _fixture()
    referent = case.referents[case.held_out]

    class EmptyWorld:
        def __init__(self) -> None:
            import numpy as np

            from grounding_kernel.contracts import Observation

            self.observation = Observation(
                np.zeros((72, 72, 3), dtype=np.uint8), 0, False
            )

        def reset(self):
            return self.observation

        def step(self, action):  # pragma: no cover - unresolved before action
            raise AssertionError(action)

    execution = ClosedLoopProgramExecutor(case.schema).execute(EmptyWorld(), referent)

    assert not execution.resolved
    assert execution.trace is None
    assert execution.actions_executed == 0
