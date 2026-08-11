from __future__ import annotations

from grounding_kernel.v1_controls import (
    ChanceShortcutControl,
    MatchedTwinControl,
    matched_twin_control_bound,
)


def test_inverted_predictions_are_still_counted_as_information_leakage() -> None:
    direct = MatchedTwinControl(True, False)
    inverted = MatchedTwinControl(False, True)
    constant = MatchedTwinControl(True, True)
    abstaining = MatchedTwinControl(None, None)
    selective_positive = MatchedTwinControl(True, None)
    selective_negative = MatchedTwinControl(None, False)

    assert direct.informative_leak
    assert inverted.informative_leak
    assert inverted.direct_accuracy == 0.0
    assert not constant.informative_leak
    assert not abstaining.informative_leak
    assert selective_positive.informative_leak
    assert selective_negative.informative_leak


def test_control_rejection_uses_upper_not_lower_confidence_bound() -> None:
    clean = matched_twin_control_bound(
        "clean",
        [MatchedTwinControl(None, None)] * 24,
        maximum_leakage=0.20,
    )
    leaking = matched_twin_control_bound(
        "leaking",
        [MatchedTwinControl(False, True)] * 24,
        maximum_leakage=0.20,
    )

    assert clean.leakage.estimate == 0.0
    assert clean.leakage.upper_bound < 0.20
    assert clean.rejected_as_grounder
    assert leaking.leakage.estimate == 1.0
    assert not leaking.rejected_as_grounder


def test_too_few_zero_events_do_not_prove_a_strict_leakage_ceiling() -> None:
    underpowered = matched_twin_control_bound(
        "underpowered",
        [MatchedTwinControl(None, None)] * 8,
        maximum_leakage=0.20,
    )

    assert underpowered.leakage.estimate == 0.0
    assert underpowered.leakage.upper_bound > 0.20
    assert not underpowered.rejected_as_grounder


def test_chance_control_treats_systematic_inversion_as_predictive() -> None:
    balanced = ChanceShortcutControl(
        "balanced",
        (True, False) * 12,
        tolerance=0.25,
    )
    direct = ChanceShortcutControl("direct", (True,) * 24, tolerance=0.25)
    inverted = ChanceShortcutControl("inverted", (False,) * 24, tolerance=0.25)

    assert balanced.rejected_as_grounder
    assert direct.leakage.estimate == 1.0
    assert inverted.leakage.estimate == 1.0
    assert not direct.rejected_as_grounder
    assert not inverted.rejected_as_grounder

    abstaining = ChanceShortcutControl(
        "abstaining", (None,) * 24, tolerance=0.25
    )
    assert abstaining.rejected_as_grounder
    assert abstaining.coverage.estimate == 0.0
