from __future__ import annotations

from dataclasses import fields

import pytest

from grounding_kernel.naming_game import (
    AgentNamingGame,
    NamingGameOracle,
    PrivateVocabulary,
    RoundPlan,
    RoundStateError,
    SocialEvaluatorOracle,
    context,
    create_naming_game,
    deterministic_nonce,
    make_private_vocabularies,
    primitives_for,
    substitute_tokens,
)
from grounding_kernel.social import (
    ConventionLearner,
    DecisionStatus,
    NamingPrompt,
    OpaqueSample,
    OperationalReferent,
    RelationalReferent,
    RoleArgument,
    SocialAgentEnvironment,
    audit_social_boundary,
)


def _object(value: int) -> OperationalReferent:
    return OperationalReferent((OpaqueSample(11_001, value),))


def _frame(left: int, right: int, *, relation: int = 31_001) -> RelationalReferent:
    return RelationalReferent(
        relation,
        (
            RoleArgument(21_001, _object(left)),
            RoleArgument(21_003, _object(right)),
        ),
    )


def _frames() -> tuple[RelationalReferent, ...]:
    return (
        _frame(41_001, 41_011),
        _frame(41_001, 41_013),
        _frame(41_003, 41_011),
        _frame(41_003, 41_013),
    )


def test_private_vocabularies_are_agent_salted_bijections_and_compose_holdouts() -> None:
    frames = _frames()
    training = frames[:3]
    held_out = frames[3]
    vocabularies = make_private_vocabularies((101, 103, 107), frames, seed=991)

    assert len(primitives_for(frames)) == 7
    assert len({vocabulary.fingerprint for vocabulary in vocabularies}) == 3
    assert {
        frozenset(vocabulary.token_to_primitive) for vocabulary in vocabularies
    } == {frozenset(vocabularies[0].token_to_primitive)}
    assert len({vocabulary.encode(frames[0]).surface for vocabulary in vocabularies}) == 3

    for vocabulary in vocabularies:
        training_surfaces = {vocabulary.encode(frame).surface for frame in training}
        held_out_surface = vocabulary.encode(held_out).surface
        assert held_out_surface not in training_surfaces
        assert set(held_out_surface) <= set(vocabulary.token_to_primitive) | {1, 2}


def test_prompt_action_feedback_staging_and_evaluator_capability_separation() -> None:
    frames = _frames()
    vocabularies = make_private_vocabularies((101, 103), frames, seed=17)
    round_context = context(((501, frames[0]), (503, frames[1])))
    plan = RoundPlan(
        7,
        101,
        103,
        round_context,
        503,
        deterministic_nonce(17, 7),
    )
    agent, oracle = create_naming_game(vocabularies, (plan,))

    assert isinstance(agent, AgentNamingGame)
    assert isinstance(agent, SocialAgentEnvironment)
    assert isinstance(oracle, NamingGameOracle)
    assert isinstance(oracle, SocialEvaluatorOracle)
    assert audit_social_boundary(agent) == ()
    assert agent.transcript.exchanges == ()
    with pytest.raises(RoundStateError, match="next_prompt"):
        agent.submit(503)

    prompt = agent.next_prompt()
    assert {field.name for field in fields(NamingPrompt)} == {
        "round_id",
        "speaker_id",
        "listener_id",
        "context",
        "utterance",
        "target_commitment",
    }
    assert not hasattr(prompt, "target_option_id")
    assert not hasattr(prompt, "nonce")
    assert oracle.active_target_option_id == 503
    with pytest.raises(RoundStateError, match="listener action"):
        agent.next_prompt()

    feedback = agent.submit(503)
    assert feedback.success is True
    assert feedback.target_option_id == 503
    assert feedback.joint_attention.attests(frames[1])
    assert agent.complete and oracle.complete
    assert len(agent.transcript.exchanges) == 1
    assert agent.transcript is oracle.transcript
    assert oracle.verify()
    assert oracle.private_vocabulary(101) == vocabularies[0]
    with pytest.raises(StopIteration, match="complete"):
        agent.next_prompt()


def test_abstention_is_an_action_and_feedback_is_still_strictly_post_action() -> None:
    frames = _frames()
    vocabularies = make_private_vocabularies((101, 103), frames, seed=19)
    round_context = context(((501, frames[0]), (503, frames[1])))
    agent, _ = create_naming_game(
        vocabularies,
        (
            RoundPlan(
                1,
                101,
                103,
                round_context,
                501,
                deterministic_nonce(19, 1),
            ),
        ),
    )

    prompt = agent.next_prompt()
    assert agent.pending_prompt == prompt
    with pytest.raises(ValueError, match="active context"):
        agent.submit(999_999)
    assert agent.pending_prompt == prompt
    feedback = agent.submit(None)

    assert feedback.success is False
    assert agent.transcript.exchanges[0].action.option_id is None
    assert agent.pending_prompt is None


def test_multi_speaker_models_do_not_leak_one_agents_private_permutation_to_another() -> None:
    frames = _frames()
    vocabularies = make_private_vocabularies((101, 103, 107), frames, seed=23)
    plans: list[RoundPlan] = []
    round_id = 0
    for speaker, listener in ((101, 107), (103, 107)):
        for _ in range(2):
            for target_index in range(3):
                target = frames[target_index]
                distractor = frames[(target_index + 1) % 3]
                options = (
                    ((501, target), (503, distractor))
                    if round_id % 2 == 0
                    else ((501, distractor), (503, target))
                )
                target_option = 501 if options[0][1] == target else 503
                plans.append(
                    RoundPlan(
                        round_id,
                        speaker,
                        listener,
                        context(options),
                        target_option,
                        deterministic_nonce(23, round_id),
                    )
                )
                round_id += 1
    held_context = context(((501, frames[0]), (503, frames[3])))
    plans.extend(
        (
            RoundPlan(
                round_id,
                101,
                107,
                held_context,
                503,
                deterministic_nonce(23, round_id),
            ),
            RoundPlan(
                round_id + 1,
                103,
                107,
                held_context,
                503,
                deterministic_nonce(23, round_id + 1),
            ),
        )
    )
    agent, _ = create_naming_game(vocabularies, plans)
    for _ in range(len(plans) - 2):
        agent.next_prompt()
        agent.submit(None)

    learner = ConventionLearner.from_transcript(107, agent.transcript)
    first_prompt = agent.next_prompt()
    first_decision = learner.choose(first_prompt)
    agent.submit(first_decision.option_id)
    second_prompt = agent.next_prompt()
    second_decision = learner.choose(second_prompt)

    assert first_prompt.utterance != second_prompt.utterance
    assert first_decision.option_id == 503
    assert second_decision.option_id == 503
    assert first_decision.status is DecisionStatus.RESOLVED
    assert second_decision.status is DecisionStatus.RESOLVED
    assert set(learner.observed_speakers) == {101, 103}


def test_deceptive_and_noisy_round_controls_are_explicit_evaluator_choices() -> None:
    frames = _frames()
    vocabulary = PrivateVocabulary.permuted(101, frames, seed=29)
    listener_vocabulary = PrivateVocabulary.permuted(103, frames, seed=29)
    round_context = context(((501, frames[0]), (503, frames[1])))
    original = vocabulary.encode(frames[0])
    noisy = substitute_tokens(
        original,
        ((original.arguments[1].referent_tokens[0], original.arguments[0].referent_tokens[0]),),
    )
    plans = (
        RoundPlan(
            1,
            101,
            103,
            round_context,
            501,
            deterministic_nonce(29, 1),
            spoken_option_id=503,
        ),
        RoundPlan(
            2,
            101,
            103,
            round_context,
            501,
            deterministic_nonce(29, 2),
            utterance_override=noisy,
        ),
    )

    assert plans[0].deceptive
    assert not plans[1].deceptive
    agent, oracle = create_naming_game((vocabulary, listener_vocabulary), plans)
    first = agent.next_prompt()
    assert first.utterance == vocabulary.encode(frames[1])
    agent.submit(None)
    second = agent.next_prompt()
    assert second.utterance == noisy
    assert oracle.plan(1).spoken_option_id == 503


def test_game_construction_fails_before_play_for_missing_or_incomplete_codebooks() -> None:
    frames = _frames()
    only_speaker = PrivateVocabulary.permuted(101, frames, seed=31)
    round_context = context(((501, frames[0]), (503, frames[1])))
    plan = RoundPlan(
        1,
        101,
        103,
        round_context,
        501,
        deterministic_nonce(31, 1),
    )

    with pytest.raises(ValueError, match="without private vocabularies"):
        create_naming_game((only_speaker,), (plan,))

    incomplete = PrivateVocabulary.permuted(101, (frames[0],), seed=31)
    listener = PrivateVocabulary.permuted(103, frames, seed=31)
    unseen_plan = RoundPlan(
        2,
        101,
        103,
        context(((501, frames[3]), (503, frames[0]))),
        501,
        deterministic_nonce(31, 2),
    )
    with pytest.raises(KeyError, match="does not cover"):
        create_naming_game((incomplete, listener), (unseen_plan,))
