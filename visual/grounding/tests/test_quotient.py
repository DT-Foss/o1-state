from __future__ import annotations

import pytest

from grounding_kernel.quotient import (
    FiniteInterventionalModel,
    IncompleteAnchorBasisError,
    NonGroundablePredicateError,
    NonTransitiveToleranceError,
    OperationalSemanticsError,
)


def _model() -> FiniteInterventionalModel:
    histories = ("red-left", "blue-left", "red-right", "red-fixed")
    interventions = ("push", "inspect")
    rows = {
        ("red-left", "push"): {"moved": 1.0},
        ("blue-left", "push"): {"moved": 1.0},
        ("red-right", "push"): {"moved": 1.0},
        ("red-fixed", "push"): {"fixed": 1.0},
        ("red-left", "inspect"): {"left": 1.0},
        ("blue-left", "inspect"): {"left": 1.0},
        ("red-right", "inspect"): {"right": 1.0},
        ("red-fixed", "inspect"): {"left": 1.0},
    }
    return FiniteInterventionalModel(histories, interventions, rows)


def test_operational_quotient_ignores_noncausal_surface_identity() -> None:
    model = _model()

    assert model.operational_partition() == (
        ("red-left", "blue-left"),
        ("red-right",),
        ("red-fixed",),
    )
    assert model.operational_distance("red-left", "blue-left") == 0.0
    assert model.operational_distance("red-left", "red-fixed") == 1.0


def test_predicate_must_factor_through_sensorimotor_quotient() -> None:
    model = _model()
    movable = {
        "red-left": True,
        "blue-left": True,
        "red-right": True,
        "red-fixed": False,
    }
    color = {
        "red-left": "red",
        "blue-left": "blue",
        "red-right": "red",
        "red-fixed": "red",
    }

    assert model.predicate_groundability(movable).groundable
    report = model.predicate_groundability(color)
    assert not report.groundable
    assert report.witness is not None
    assert {report.witness.left, report.witness.right} == {"red-left", "blue-left"}


def test_anchor_separation_is_necessary_and_sufficient_for_boolean_closure() -> None:
    model = _model()
    anchors = {
        "moves": {
            "red-left": True,
            "blue-left": True,
            "red-right": True,
            "red-fixed": False,
        },
        "left": {
            "red-left": True,
            "blue-left": True,
            "red-right": False,
            "red-fixed": True,
        },
    }
    certificate = model.anchor_closure_certificate(anchors)

    assert certificate.complete
    assert certificate.quotient_size == 3
    assert certificate.signature_class_count == 3
    assert certificate.expressible_subsets == 8
    assert certificate.total_groundable_subsets == 8

    target = {
        "red-left": False,
        "blue-left": False,
        "red-right": True,
        "red-fixed": True,
    }
    program = model.synthesize_boolean_program(target, anchors)
    assert program.target_positive_classes == (1, 2)
    assert {
        history: program.evaluate(history, anchors) for history in model.histories
    } == target


def test_equal_anchor_signatures_are_constructive_impossibility_witness() -> None:
    model = _model()
    only_movable = {
        "moves": {
            "red-left": True,
            "blue-left": True,
            "red-right": True,
            "red-fixed": False,
        }
    }
    certificate = model.anchor_closure_certificate(only_movable)

    assert not certificate.complete
    assert certificate.indistinguishable_classes == (0, 1)
    assert certificate.expressible_subsets == 4
    assert certificate.total_groundable_subsets == 8
    with pytest.raises(IncompleteAnchorBasisError):
        model.synthesize_boolean_program(
            {
                "red-left": True,
                "blue-left": True,
                "red-right": False,
                "red-fixed": False,
            },
            only_movable,
        )


def test_non_groundable_target_is_rejected_even_with_separating_anchors() -> None:
    model = _model()
    anchors = {
        "moves": {
            "red-left": True,
            "blue-left": True,
            "red-right": True,
            "red-fixed": False,
        },
        "left": {
            "red-left": True,
            "blue-left": True,
            "red-right": False,
            "red-fixed": True,
        },
    }
    visible_color = {
        "red-left": True,
        "blue-left": False,
        "red-right": True,
        "red-fixed": True,
    }

    with pytest.raises(NonGroundablePredicateError):
        model.synthesize_boolean_program(visible_color, anchors)


def test_approximate_equivalence_fails_closed_when_threshold_is_not_transitive() -> None:
    rows = {
        ("a", "probe"): {0: 1.0, 1: 0.0},
        ("b", "probe"): {0: 0.8, 1: 0.2},
        ("c", "probe"): {0: 0.6, 1: 0.4},
    }
    model = FiniteInterventionalModel(("a", "b", "c"), ("probe",), rows)

    with pytest.raises(NonTransitiveToleranceError):
        model.operational_partition(tolerance=0.21)


def test_kernel_validation_rejects_partial_or_unnormalized_models() -> None:
    with pytest.raises(OperationalSemanticsError, match="full"):
        FiniteInterventionalModel(("h1", "h2"), ("i",), {("h1", "i"): {"y": 1.0}})
    with pytest.raises(OperationalSemanticsError, match="sum"):
        FiniteInterventionalModel(("h",), ("i",), {("h", "i"): {"y": 0.8}})


@pytest.mark.parametrize("bad", [True, "1.0", 1.1, -0.1])
def test_kernel_rejects_nonprobability_values(bad: object) -> None:
    error = (TypeError, OperationalSemanticsError)
    with pytest.raises(error):
        FiniteInterventionalModel(("h",), ("i",), {("h", "i"): {"y": bad}})

    with pytest.raises(OperationalSemanticsError, match="positive"):
        FiniteInterventionalModel(("h",), ("i",), {("h", "i"): {"y": 0.0}})
    with pytest.raises(OperationalSemanticsError, match="1e-6"):
        FiniteInterventionalModel(
            ("h",),
            ("i",),
            {("h", "i"): {"y": 1.0}},
            normalization_tolerance=10.0,
        )


def test_accepted_near_normalized_rows_are_exact_probabilities_and_tv_is_bounded() -> None:
    model = FiniteInterventionalModel(
        ("a", "b"),
        ("i",),
        {
            ("a", "i"): {0: 0.5000000001, 1: 0.5},
            ("b", "i"): {0: 0.0, 1: 1.0},
        },
    )
    assert sum(model.distributions[("a", "i")].values()) == pytest.approx(1.0)
    assert 0.0 <= model.total_variation("a", "b", "i") <= 1.0


def test_ungroundable_anchor_has_its_own_witness_not_a_fake_separation_witness() -> None:
    model = _model()
    anchors = {
        "surface-color": {
            "red-left": True,
            "blue-left": False,
            "red-right": True,
            "red-fixed": True,
        },
        "left": {
            "red-left": True,
            "blue-left": True,
            "red-right": False,
            "red-fixed": True,
        },
    }
    certificate = model.anchor_closure_certificate(anchors)

    assert not certificate.complete
    assert not certificate.all_anchors_groundable
    assert certificate.ungroundable_anchors[0].anchor == "surface-color"
    assert "not grounded" in certificate.theorem
    assert certificate.certified_expressible_subsets == 0
