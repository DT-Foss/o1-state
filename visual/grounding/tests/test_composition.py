from __future__ import annotations

import pytest

from grounding_kernel.composition import (
    And,
    Atom,
    Not,
    Or,
    Relation,
    Sort,
    TruthValue,
    entity,
    evaluate,
    least_fixed_point,
)


def test_three_valued_evaluation_and_derivation() -> None:
    expression = And(Atom(101), Or(Atom(202), Not(Atom(303))))
    values = {101: True, 202: None, 303: False}
    result = evaluate(expression, values.get)

    assert result.value is TruthValue.TRUE
    proof = result.derivation.to_dict()
    assert proof["rule"] == "and-introduction"
    assert proof["premises"][1]["rule"] == "or-introduction"

    unknown = evaluate(And(Atom(101), Atom(202)), values.get)
    assert unknown.value is TruthValue.UNKNOWN
    assert unknown.as_python() is None


def test_typed_relation_calls_only_the_explicit_resolver() -> None:
    relation = Relation(909, entity("object-a"), entity("object-b"))
    calls: list[tuple[object, object, object]] = []

    def resolve(symbol: object, left: object, right: object) -> tuple[bool, str]:
        calls.append((symbol, left, right))
        return True, "controlled intervention 17"

    result = evaluate(relation, lambda _symbol: None, resolve)
    assert result.as_python() is True
    assert calls == [(909, "object-a", "object-b")]
    assert result.derivation.evidence == "controlled intervention 17"

    with pytest.raises(TypeError, match="left operand"):
        Relation(909, Atom("boolean"), entity("object-b"))
    with pytest.raises(TypeError, match="boolean"):
        And(Atom("x", Sort.ENTITY))


def test_least_fixed_point_propagates_only_from_sensorimotor_anchors() -> None:
    definitions = {
        "derived": And(Atom(11), Atom("effect")),
        "chain": Atom("derived"),
        "cycle-a": Atom("cycle-b"),
        "cycle-b": Atom("cycle-a"),
    }
    closure = least_fixed_point(definitions, {11: 0.9, "effect": 0.8})

    assert closure.confidence("derived") == pytest.approx(0.8)
    assert closure.confidence("chain") == pytest.approx(0.8)
    assert closure.confidence("cycle-a") == 0.0
    assert closure.confidence("cycle-b") == 0.0
    assert closure.unresolved_cycles == (("cycle-a", "cycle-b"),)
    assert closure.proof("chain") is not None
    assert closure.proof("chain").rule == "definition"


def test_anchor_enters_cycle_and_or_uses_a_grounded_alternative() -> None:
    definitions = {
        "a": Atom("b"),
        "b": Atom("a"),
        "via-either": Or(Atom("missing"), Atom("a")),
    }
    closure = least_fixed_point(definitions, {"a": 0.65})

    assert closure.confidence("a") == pytest.approx(0.65)
    assert closure.confidence("b") == pytest.approx(0.65)
    assert closure.confidence("via-either") == pytest.approx(0.65)
    assert closure.unresolved_cycles == ()


def test_relation_grounding_requires_operator_and_both_typed_terms() -> None:
    definition = Relation("left-of", entity("x"), entity("y"))
    definitions = {"composed": definition}

    incomplete = least_fixed_point(definitions, {"left-of": 1.0, "x": 1.0})
    assert incomplete.confidence("composed") == 0.0

    complete = least_fixed_point(
        definitions,
        {"left-of": 0.9, "x": 0.8, "y": 0.7},
    )
    assert complete.confidence("composed") == pytest.approx(0.7)
    assert complete.proof("composed") is not None

