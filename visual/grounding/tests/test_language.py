from __future__ import annotations

import inspect

import pytest

import grounding_kernel.language as language_module
from grounding_kernel.composition import And, Atom
from grounding_kernel.language import (
    UNKNOWN,
    Demonstration,
    GroundedLanguageLearner,
    GroundedReferent,
    OperationalMeaning,
    Resolution,
)


TYPE_1 = 71_003
TYPE_2 = 71_009
VALUE_11 = 81_101
VALUE_12 = 81_102
VALUE_21 = 82_201
VALUE_22 = 82_202
TOKEN_11 = 91_011
TOKEN_12 = 91_012
TOKEN_21 = 92_021
TOKEN_22 = 92_022


def _referent(value_1: int, value_2: int) -> GroundedReferent:
    # Reverse input order deliberately: referents are typed sets, not an
    # evaluator-provided word-order oracle.
    return GroundedReferent(
        (
            OperationalMeaning(TYPE_2, value_2),
            OperationalMeaning(TYPE_1, value_1),
        )
    )


def _factorial_training() -> tuple[Demonstration, ...]:
    # The TOKEN_12/TOKEN_22 combination is held out in both directions.
    return (
        Demonstration((TOKEN_11, TOKEN_21), _referent(VALUE_11, VALUE_21), (1, 101)),
        Demonstration((TOKEN_11, TOKEN_22), _referent(VALUE_11, VALUE_22), (2, 102)),
        Demonstration((TOKEN_12, TOKEN_21), _referent(VALUE_12, VALUE_21), (3, 103)),
    )


def test_factorial_holdout_is_composed_bidirectionally_without_utterance_lookup() -> None:
    training = _factorial_training()
    held_out_tokens = (TOKEN_12, TOKEN_22)
    held_out_referent = _referent(VALUE_12, VALUE_22)
    assert held_out_tokens not in {demo.tokens for demo in training}
    assert held_out_referent not in {demo.referent for demo in training}

    learner = GroundedLanguageLearner().fit(training)
    instruction = learner.ground_action(held_out_tokens)
    description = learner.describe(held_out_referent)

    assert instruction.status is Resolution.RESOLVED
    assert instruction.referent == held_out_referent
    assert instruction.proof.rule == "compose-induced-order"
    assert description.status is Resolution.RESOLVED
    assert description.utterance == held_out_tokens
    assert description.proof.rule == "realize-induced-order"
    assert learner.interpret(description.utterance).referent == held_out_referent
    assert learner.round_trip(held_out_referent)
    assert learner.order_templates[0].position_types == (TYPE_1, TYPE_2)


def test_fresh_token_permutation_is_equivariant_and_ledger_is_deterministic() -> None:
    training = _factorial_training()
    token_permutation = {
        TOKEN_11: 17,
        TOKEN_12: 1_000_003,
        TOKEN_21: 43,
        TOKEN_22: 5,
    }
    remapped_training = tuple(
        Demonstration(
            tuple(token_permutation[token] for token in demo.tokens),
            demo.referent,
            demo.evidence,
        )
        for demo in training
    )

    original = GroundedLanguageLearner().fit(training)
    remapped = GroundedLanguageLearner().fit(remapped_training)
    target = _referent(VALUE_12, VALUE_22)

    original_description = original.describe(target)
    remapped_description = remapped.describe(target)
    assert remapped_description.utterance == tuple(
        token_permutation[token] for token in original_description.utterance or ()
    )
    assert remapped.interpret(remapped_description.utterance or ()).referent == target
    assert {
        binding.meaning for binding in original.bindings
    } == {binding.meaning for binding in remapped.bindings}

    reordered = GroundedLanguageLearner().fit(reversed(training))
    assert reordered.ledger.to_dict() == original.ledger.to_dict()
    assert "opaque operational slots" in original.ledger.entries[0].conclusion
    assert "part-of-speech" in original.ledger.entries[0].conclusion
    assert original.interpret((TOKEN_12, TOKEN_22)).proof.to_dict() == original.interpret(
        (TOKEN_12, TOKEN_22)
    ).proof.to_dict()


def test_synonyms_require_direct_shared_grounding_and_roundtrip_semantically() -> None:
    synonym = 91_111
    training = _factorial_training() + (
        Demonstration((synonym, TOKEN_21), _referent(VALUE_11, VALUE_21), (4, 104)),
    )
    learner = GroundedLanguageLearner().fit(training)

    first = learner.meaning(TOKEN_11)
    second = learner.meaning(synonym)
    assert first.resolved and second.resolved
    assert first.meaning == second.meaning == OperationalMeaning(TYPE_1, VALUE_11)
    assert learner.synonyms(TOKEN_11) == (synonym,)
    assert learner.synonyms(synonym) == (TOKEN_11,)

    description = learner.describe(_referent(VALUE_11, VALUE_21))
    assert set(description.candidates) == {
        (TOKEN_11, TOKEN_21),
        (synonym, TOKEN_21),
    }
    assert {
        learner.interpret(candidate).referent for candidate in description.candidates
    } == {_referent(VALUE_11, VALUE_21)}

    # A dictionary-only alias is interpretable as a grounded definition, but
    # cannot become a lexical synonym without a paired demonstration.
    dictionary_only = 99_999
    learner.add_definitions({dictionary_only: Atom(TOKEN_11)})
    assert learner.resolve_definition(dictionary_only).resolved
    assert learner.meaning(dictionary_only).status is Resolution.UNKNOWN
    assert dictionary_only not in learner.synonyms(TOKEN_11)


def test_no_sensor_and_fresh_tokens_are_unknown_without_hallucination() -> None:
    no_sensor = GroundedLanguageLearner().fit(
        (
            Demonstration((TOKEN_11, TOKEN_21), None, (1, 101)),
            Demonstration((TOKEN_12, TOKEN_22), None, (2, 102)),
        )
    )

    assert no_sensor.meaning(TOKEN_11).status is UNKNOWN
    assert no_sensor.interpret((TOKEN_11, TOKEN_21)).status is Resolution.UNKNOWN
    assert no_sensor.describe(_referent(VALUE_11, VALUE_21)).status is Resolution.UNKNOWN
    assert no_sensor.order_templates == ()
    assert any(entry.rule == "ignored-no-sensor" for entry in no_sensor.ledger.entries)

    grounded = GroundedLanguageLearner().fit(_factorial_training())
    fresh = 123_456_789
    decision = grounded.interpret((fresh, TOKEN_21))
    assert grounded.meaning(fresh).status is Resolution.UNKNOWN
    assert decision.status is Resolution.UNKNOWN
    assert decision.referent is None
    assert decision.candidates == ()


def test_disruptively_shuffled_pairs_fail_the_consistency_gate() -> None:
    # This is not a coherent global recoding: under both possible role orders,
    # at least one repeated token is paired with two operational values.
    shuffled = (
        Demonstration((TOKEN_11, TOKEN_21), _referent(VALUE_11, VALUE_21)),
        Demonstration((TOKEN_11, TOKEN_22), _referent(VALUE_12, VALUE_22)),
        Demonstration((TOKEN_12, TOKEN_21), _referent(VALUE_11, VALUE_22)),
        Demonstration((TOKEN_12, TOKEN_22), _referent(VALUE_12, VALUE_21)),
    )
    learner = GroundedLanguageLearner().fit(shuffled)

    assert learner.order_templates == ()
    assert learner.bindings == ()
    assert learner.interpret((TOKEN_12, TOKEN_22)).status is Resolution.UNKNOWN
    assert any(
        entry.rule == "rejected-inconsistent-pairs" for entry in learner.ledger.entries
    )


def test_competing_grounded_parses_are_explicitly_ambiguous() -> None:
    shared_token = 777
    first = GroundedReferent((OperationalMeaning(10, 100),))
    second = GroundedReferent((OperationalMeaning(20, 200),))
    learner = GroundedLanguageLearner().fit(
        (
            Demonstration((shared_token,), first),
            Demonstration((shared_token,), second),
        )
    )

    lexical = learner.meaning(shared_token)
    parsed = learner.interpret((shared_token,))
    assert lexical.status is Resolution.AMBIGUOUS
    assert set(lexical.candidates) == set(first.meanings + second.meanings)
    assert parsed.status is Resolution.AMBIGUOUS
    assert parsed.referent is None
    assert set(parsed.candidates) == {first, second}
    assert parsed.proof.rule == "multiple-grounded-parses"


def test_symbolic_theft_uses_least_fixed_point_and_cycles_remain_unknown() -> None:
    composed = 500_001
    cycle_a = 500_002
    cycle_b = 500_003
    learner = GroundedLanguageLearner().fit(_factorial_training()).add_definitions(
        {
            composed: And(Atom(TOKEN_11), Atom(TOKEN_21)),
            cycle_a: Atom(cycle_b),
            cycle_b: Atom(cycle_a),
        }
    )

    grounded = learner.symbolic_theft(composed)
    ungrounded = learner.resolve_definition(cycle_a)
    closure = learner.definition_closure()

    assert grounded.status is Resolution.RESOLVED
    assert grounded.expression == And(Atom(TOKEN_11), Atom(TOKEN_21))
    assert grounded.confidence == pytest.approx(1.0)
    assert grounded.proof.rule == "least-fixed-point-symbolic-theft"
    assert ungrounded.status is Resolution.UNKNOWN
    assert ungrounded.expression is None
    assert closure.confidence(cycle_a) == 0.0
    assert closure.confidence(cycle_b) == 0.0
    assert closure.unresolved_cycles == ((cycle_a, cycle_b),)
    assert learner.composition_anchors[TOKEN_11] == pytest.approx(1.0)


def test_language_module_has_no_oracle_or_semantic_enum_dependency() -> None:
    source = inspect.getsource(language_module)
    assert "from .contracts import" not in source
    assert "from .microworld import" not in source
    for forbidden in ("ActionKind", "OutcomeKind", "PredicateKind", "OpaqueCodebook"):
        assert forbidden not in source


def test_typed_referents_reject_duplicate_slots_and_bad_arity_abstains() -> None:
    with pytest.raises(ValueError, match="one value per type_id"):
        GroundedReferent(
            (
                OperationalMeaning(TYPE_1, VALUE_11),
                OperationalMeaning(TYPE_1, VALUE_12),
            )
        )

    learner = GroundedLanguageLearner().fit(
        (Demonstration((TOKEN_11,), _referent(VALUE_11, VALUE_21)),)
    )
    assert learner.order_templates == ()
    assert learner.interpret((TOKEN_11,)).status is Resolution.UNKNOWN
    assert any(entry.rule == "rejected-arity" for entry in learner.ledger.entries)
