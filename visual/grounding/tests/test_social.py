from __future__ import annotations

from dataclasses import FrozenInstanceError, replace

import pytest

from grounding_kernel.naming_game import (
    PrivateVocabulary,
    RoundPlan,
    context,
    create_naming_game,
    deterministic_nonce,
    make_private_vocabularies,
)
from grounding_kernel.social import (
    CodePermutation,
    ContextCandidate,
    ConventionLearner,
    ConventionPopulation,
    DecisionStatus,
    Exchange,
    HashChainLedger,
    HashChainRecord,
    JointAttentionTrace,
    NamingPrompt,
    OpaqueSample,
    OperationalReferent,
    PositionShortcut,
    ReferentialContext,
    ReferentialFeedback,
    RelationalReferent,
    RoleArgument,
    SocialRenaming,
    StructuredUtterance,
    SurfaceLookupShortcut,
    Transcript,
    check_equivariance,
    evaluate_counterfactual,
    joint_attention_for,
    rename_transcript,
    score_predictor,
    swap_candidate_referents,
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


def _compositional_run() -> tuple[
    tuple[PrivateVocabulary, ...],
    Transcript,
    int,
]:
    frames = _frames()
    vocabularies = make_private_vocabularies((101, 103, 107), frames, seed=1019)
    plans: list[RoundPlan] = []
    round_id = 0
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
                    101,
                    103,
                    context(options),
                    target_option,
                    deterministic_nonce(1019, round_id),
                )
            )
            round_id += 1
    held_out_target = 503
    plans.append(
        RoundPlan(
            round_id,
            101,
            107,
            context(((501, frames[0]), (held_out_target, frames[3]))),
            held_out_target,
            deterministic_nonce(1019, round_id),
        )
    )
    agent, oracle = create_naming_game(vocabularies, plans)
    for plan in plans:
        agent.next_prompt()
        agent.submit(plan.target_option_id)
    assert oracle.verify()
    return vocabularies, agent.transcript, held_out_target


def test_transcripts_commit_targets_and_are_deeply_immutable_hash_chains() -> None:
    _, transcript, _ = _compositional_run()
    exchange = transcript.exchanges[0]

    assert transcript.verify()
    assert len(transcript.ledger.records) == 3 * len(transcript.exchanges)
    assert transcript.head == transcript.ledger.records[-1].digest
    assert tuple(record.event_type for record in transcript.ledger.records[:3]) == (
        "prompt",
        "action",
        "feedback",
    )
    with pytest.raises(FrozenInstanceError):
        exchange.feedback.success = False  # type: ignore[misc]
    with pytest.raises(AttributeError):
        exchange.prompt.context.candidates.append(  # type: ignore[attr-defined]
            exchange.prompt.context.candidates[0]
        )

    forged_prompt = replace(exchange.prompt, target_commitment="0" * 64)
    with pytest.raises(ValueError, match="does not open"):
        Exchange(forged_prompt, exchange.action, exchange.feedback)
    with pytest.raises(ValueError, match="exactly these"):
        Transcript(transcript.exchanges, HashChainLedger())

    final = transcript.ledger.records[-1]
    forged_record = HashChainRecord(
        final.index,
        final.event_type,
        final.payload_digest,
        final.previous_digest,
        "0" * 64,
    )
    with pytest.raises(ValueError, match="invalid hash-chain"):
        HashChainLedger(transcript.ledger.records[:-1] + (forged_record,))


def test_joint_attention_must_publicly_attest_every_role_and_both_participants() -> None:
    _, transcript, _ = _compositional_run()
    exchange = transcript.exchanges[0]
    target = exchange.prompt.context.candidate(exchange.feedback.target_option_id)
    assert target is not None
    wrong_participants = joint_attention_for(
        target.referent,
        (exchange.prompt.speaker_id, 999_991),
        event_code=70_001,
        action_code=70_003,
    )
    forged_feedback = ReferentialFeedback(
        exchange.feedback.round_id,
        exchange.feedback.target_option_id,
        exchange.feedback.success,
        exchange.feedback.nonce,
        wrong_participants,
    )
    with pytest.raises(ValueError, match="speaker and listener"):
        Exchange(exchange.prompt, exchange.action, forged_feedback)

    events_without_relation = tuple(
        replace(event, outcome_code=None)
        for event in exchange.feedback.joint_attention.events
    )
    trace_without_relation = JointAttentionTrace(
        exchange.feedback.joint_attention.participants,
        events_without_relation,
    )
    assert not trace_without_relation.attests(target.referent)


def test_conservative_learner_composes_holdout_and_newcomer_replays_public_history() -> None:
    _, transcript, target = _compositional_run()
    training = transcript.prefix(len(transcript.exchanges) - 1)
    held_out_prompt = transcript.exchanges[-1].prompt

    learner = ConventionLearner.from_transcript(107, training)
    decision = learner.choose(held_out_prompt)
    assert decision.status is DecisionStatus.RESOLVED
    assert decision.option_id == target
    assert decision.confidence == pytest.approx(1.0)
    assert decision.evidence_count >= 2

    population = ConventionPopulation()
    incumbent = population.admit(103, training)
    newcomer_without_history = population.admit(109)
    assert newcomer_without_history.choose(held_out_prompt).status is DecisionStatus.UNKNOWN
    population.retire(109)
    newcomer = population.admit(109, training)
    assert newcomer.choose(held_out_prompt) == incumbent.choose(held_out_prompt)
    assert population.agent_ids == (103, 109)
    retired = population.retire(103)
    assert retired is incumbent
    assert population.agent_ids == (109,)


def _unary_frame(value: int) -> RelationalReferent:
    return RelationalReferent(
        31_101,
        (RoleArgument(21_101, OperationalReferent((OpaqueSample(11_101, value),))),),
    )


def _speaker_noise_game(truthful: int, deceptive: int) -> tuple[
    ConventionLearner,
    NamingPrompt,
]:
    first = _unary_frame(41_101)
    second = _unary_frame(41_103)
    frames = (first, second)
    vocabularies = make_private_vocabularies((101, 103), frames, seed=2027)
    round_context = context(((501, first), (503, second)))
    plans: list[RoundPlan] = []
    round_id = 0
    for _ in range(truthful):
        plans.append(
            RoundPlan(
                round_id,
                101,
                103,
                round_context,
                501,
                deterministic_nonce(2027, round_id),
            )
        )
        round_id += 1
    for _ in range(deceptive):
        plans.append(
            RoundPlan(
                round_id,
                101,
                103,
                round_context,
                503,
                deterministic_nonce(2027, round_id),
                spoken_option_id=501,
            )
        )
        round_id += 1
    plans.append(
        RoundPlan(
            round_id,
            101,
            103,
            round_context,
            501,
            deterministic_nonce(2027, round_id),
        )
    )
    agent, _ = create_naming_game(vocabularies, plans)
    for _ in range(truthful + deceptive):
        agent.next_prompt()
        agent.submit(None)
    learner = ConventionLearner.from_transcript(103, agent.transcript)
    return learner, agent.next_prompt()


def test_bounded_noise_resolves_but_balanced_deception_forces_abstention() -> None:
    robust, robust_prompt = _speaker_noise_game(3, 1)
    robust_decision = robust.choose(robust_prompt)
    assessment = robust.speaker_assessment(101)

    assert robust_decision.status is DecisionStatus.RESOLVED
    assert robust_decision.option_id == 501
    assert robust_decision.confidence == pytest.approx(0.75)
    assert assessment.conflicting_bindings == 1
    assert assessment.consistency < 1.0

    ambiguous, ambiguous_prompt = _speaker_noise_game(2, 2)
    ambiguous_decision = ambiguous.choose(ambiguous_prompt)
    assert ambiguous_decision.status is DecisionStatus.AMBIGUOUS
    assert ambiguous_decision.abstained
    assert "multiple primitive" in ambiguous_decision.reason


def test_unknown_and_duplicate_referents_abstain_without_hallucinating() -> None:
    vocabularies, transcript, _ = _compositional_run()
    training = transcript.prefix(len(transcript.exchanges) - 1)
    held_out = _frames()[3]
    duplicate_context = ReferentialContext(
        (
            ContextCandidate(501, held_out),
            ContextCandidate(503, held_out),
        )
    )
    plan = RoundPlan(
        99,
        101,
        107,
        duplicate_context,
        501,
        deterministic_nonce(3037, 99),
    )
    agent, _ = create_naming_game(vocabularies, (plan,))
    prompt = agent.next_prompt()

    untrained = ConventionLearner(107)
    unknown = untrained.choose(prompt)
    assert unknown.status is DecisionStatus.UNKNOWN
    assert unknown.abstained
    with pytest.raises(TypeError, match="completed Exchange"):
        untrained.observe(prompt)  # type: ignore[arg-type]

    trained = ConventionLearner.from_transcript(107, training)
    ambiguous = trained.choose(prompt)
    assert ambiguous.status is DecisionStatus.AMBIGUOUS
    assert ambiguous.candidates == (501, 503)
    assert ambiguous.abstained


def test_full_code_and_referent_permutations_are_equivariant() -> None:
    vocabularies, transcript, _ = _compositional_run()
    prefix_length = len(transcript.exchanges) - 1
    original_training = transcript.prefix(prefix_length)
    original_test = transcript.exchanges[-1]
    tokens = tuple(binding.token for binding in vocabularies[0].bindings)
    renaming = SocialRenaming(
        agents=CodePermutation.cycle((101, 103, 107)),
        options=CodePermutation.cycle((501, 503)),
        tokens=CodePermutation.cycle(tokens),
        relations=CodePermutation.cycle((31_001, 31_997)),
        roles=CodePermutation.cycle((21_001, 21_003)),
        channels=CodePermutation.cycle((11_001, 11_997)),
        values=CodePermutation.cycle((41_001, 41_003, 41_011, 41_013)),
        events=CodePermutation.cycle((70_001, 70_997)),
        actions=CodePermutation.cycle((70_003, 70_999)),
    )
    renamed_transcript = rename_transcript(transcript, renaming)
    renamed_training = renamed_transcript.prefix(prefix_length)
    renamed_test = renamed_transcript.exchanges[-1]

    original_learner = ConventionLearner.from_transcript(107, original_training)
    renamed_learner = ConventionLearner.from_transcript(
        renaming.agents.apply(107),
        renamed_training,
    )
    original = original_learner.choose(original_test.prompt)
    renamed = renamed_learner.choose(renamed_test.prompt)
    result = check_equivariance(original, renamed, renaming)

    assert result.passed
    assert renamed.option_id == renaming.options.apply(original.option_id or 0)
    assert renamed_transcript.verify()
    assert renamed_transcript.head != transcript.head
    inverse = SocialRenaming(
        agents=renaming.agents.inverse(),
        options=renaming.options.inverse(),
        tokens=renaming.tokens.inverse(),
        relations=renaming.relations.inverse(),
        roles=renaming.roles.inverse(),
        channels=renaming.channels.inverse(),
        values=renaming.values.inverse(),
        events=renaming.events.inverse(),
        actions=renaming.actions.inverse(),
    )
    assert rename_transcript(renamed_transcript, inverse) == transcript


def test_counterfactual_swap_rejects_position_and_surface_shortcuts() -> None:
    _, transcript, target = _compositional_run()
    training = transcript.prefix(len(transcript.exchanges) - 1)
    prompt = transcript.exchanges[-1].prompt
    distractor = next(option for option in prompt.context.option_ids if option != target)
    pair = swap_candidate_referents(prompt, target, distractor)
    grounded = ConventionLearner.from_transcript(107, training)

    grounded_result = evaluate_counterfactual(grounded.choose, pair)
    assert grounded_result.passed
    assert grounded_result.factual.option_id == target
    assert grounded_result.contrast.option_id == distractor
    assert pair.factual_prompt.utterance == pair.contrast_prompt.utterance
    assert pair.factual_prompt.context.option_ids == pair.contrast_prompt.context.option_ids

    position = PositionShortcut().fit(transcript)
    surface = SurfaceLookupShortcut().fit(transcript)
    assert not evaluate_counterfactual(position.choose, pair).passed
    assert not evaluate_counterfactual(surface.choose, pair).passed
    grounded_score = score_predictor(
        grounded.choose,
        (
            (pair.factual_prompt, pair.factual_target),
            (pair.contrast_prompt, pair.contrast_target),
        ),
    )
    assert grounded_score.coverage == 1.0
    assert grounded_score.accuracy == 1.0


def test_only_strict_integer_codes_are_admitted_no_text_semantics() -> None:
    with pytest.raises(TypeError, match="strict integer"):
        OpaqueSample("red", 1)  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="strict integer"):
        OpaqueSample(True, 1)
    with pytest.raises(TypeError, match="strict integer"):
        StructuredUtterance("relation", ())  # type: ignore[arg-type]
