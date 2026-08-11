from __future__ import annotations

import pytest

from grounding_kernel.language import Demonstration, GroundedLanguageLearner
from grounding_kernel.processworld import ProcessConceptKind, ProcessHarness
from grounding_kernel.programs import (
    ActionScheme,
    GroundedProgramExecutor,
    ProgramSchema,
    TargetTrack,
    build_program_referent,
)


TARGET_TYPE = 410_003
SCHEME_TYPE = 410_009
TOKEN_SELF = 510_001
TOKEN_EXTERNAL = 510_007
TOKEN_MOVE = 510_011
TOKEN_RUN = 510_013


def _fixture(seed: int = 81):
    harness = ProcessHarness(seed)
    self_running, external_running = harness.oracle.process_pair()
    self_moving, _external_moving = harness.oracle.movement_pair()
    self_target_demo, external_target_demo = harness.oracle.target_role_pair()
    schema = ProgramSchema(TARGET_TYPE, SCHEME_TYPE)
    moving = ActionScheme.from_episode(self_moving.episode)
    running = ActionScheme.from_episode(self_running.episode)

    def target_track_for(demo):
        observed = TargetTrack.from_episode(demo).targets
        padded = observed + (observed[-1],) * (len(running.steps) - len(observed))
        return TargetTrack(padded)

    self_track = target_track_for(self_target_demo)
    external_track = target_track_for(external_target_demo)
    referents = {
        (TOKEN_SELF, TOKEN_MOVE): build_program_referent(schema, self_track, moving),
        (TOKEN_EXTERNAL, TOKEN_MOVE): build_program_referent(
            schema, external_track, moving
        ),
        (TOKEN_EXTERNAL, TOKEN_RUN): build_program_referent(
            schema, external_track, running
        ),
        (TOKEN_SELF, TOKEN_RUN): build_program_referent(schema, self_track, running),
    }
    return schema, referents


def test_withheld_instruction_compiles_and_causes_the_grounded_process() -> None:
    schema, referents = _fixture()
    held_out = (TOKEN_SELF, TOKEN_RUN)
    training = tuple(
        Demonstration(tokens, referent)
        for tokens, referent in referents.items()
        if tokens != held_out
    )
    learner = GroundedLanguageLearner().fit(training)
    interpretation = learner.interpret_instruction(held_out)
    assert interpretation.resolved and interpretation.referent == referents[held_out]

    harness = ProcessHarness(81)
    trace = GroundedProgramExecutor(schema).execute(harness.agent, interpretation.referent)

    assert len(trace.transitions) == 4
    assert all(not hasattr(step, "outcome_code") for step in trace.transitions)
    assert harness.oracle.matches_concept(
        harness.agent.episode(), ProcessConceptKind.RUNNING
    )


def test_world_to_description_roundtrip_recovers_same_executable_program() -> None:
    schema, referents = _fixture()
    held_out = (TOKEN_SELF, TOKEN_RUN)
    learner = GroundedLanguageLearner().fit(
        Demonstration(tokens, referent)
        for tokens, referent in referents.items()
        if tokens != held_out
    )
    description = learner.describe_fact(referents[held_out])

    assert description.resolved
    assert description.utterance == held_out
    assert learner.round_trip(referents[held_out])
    parsed = learner.interpret(description.utterance or ())
    assert GroundedProgramExecutor(schema).compile(parsed.referent).actions == (
        GroundedProgramExecutor(schema).compile(referents[held_out]).actions
    )


def test_both_grounded_words_are_behaviorally_necessary() -> None:
    schema, referents = _fixture()
    executor = GroundedProgramExecutor(schema)
    self_run = executor.compile(referents[(TOKEN_SELF, TOKEN_RUN)])
    external_run = executor.compile(referents[(TOKEN_EXTERNAL, TOKEN_RUN)])
    self_move = executor.compile(referents[(TOKEN_SELF, TOKEN_MOVE)])

    assert self_run.actions != external_run.actions
    assert self_run.actions != self_move.actions


def test_incomplete_or_mistyped_referents_fail_closed() -> None:
    schema, referents = _fixture()
    executor = GroundedProgramExecutor(schema)
    complete = referents[(TOKEN_SELF, TOKEN_RUN)]
    with pytest.raises(ValueError, match="required slots"):
        executor.compile(type(complete)((complete.meanings[0],)))

    short_track = TargetTrack(((1, 1),))
    with pytest.raises(ValueError, match="shorter"):
        executor.compile(
            build_program_referent(
                schema,
                short_track,
                complete.meaning_for(SCHEME_TYPE).value,  # type: ignore[union-attr,arg-type]
            )
        )
