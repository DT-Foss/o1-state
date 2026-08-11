from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np
import pytest

from grounding_kernel.binder import SensorimotorBinder
from grounding_kernel.composition import And, Atom
from grounding_kernel.contracts import Action, Observation, Transition


@dataclass(frozen=True)
class Evidence:
    token: object
    before: Observation
    action: Action
    after: Observation
    outcome_code: int
    task_feedback: bool = True

    @property
    def transition(self) -> Transition:
        return Transition(self.before, self.action, self.after, self.outcome_code)


def _observation(square_x: int, square_y: int, colour: tuple[int, int, int], tick: int) -> Observation:
    pixels = np.zeros((12, 12, 3), dtype=np.uint8)
    pixels[1:-1, 1:-1] = (12, 12, 12)
    pixels[square_y : square_y + 2, square_x : square_x + 2] = colour
    return Observation(pixels, tick)


def _move_evidence(token: object, index: int) -> Evidence:
    x = 2 + index % 4
    y = 2 + (index * 2) % 5
    before = _observation(x, y, (220, 40, 30), 0)
    after = _observation(x + 1, y, (220, 40, 30), 1)
    return Evidence(token, before, Action(710_001, (x, y), (1, 0)), after, 810_001)


def _toggle_evidence(token: object, index: int) -> Evidence:
    x = 2 + index % 4
    y = 2 + (index * 3) % 5
    before = _observation(x, y, (30, 80, 210), 0)
    after = _observation(x, y, (30, 230, 80), 1)
    return Evidence(token, before, Action(710_009, (x, y), (0, 0)), after, 810_009)


def _training(move_token: object = 1_234_567, toggle_token: object = "opaque-z") -> list[Evidence]:
    return [
        *(_move_evidence(move_token, index) for index in range(8)),
        *(_toggle_evidence(toggle_token, index) for index in range(8)),
    ]


def test_binder_learns_integer_and_string_tokens_from_transitions() -> None:
    binder = SensorimotorBinder(minimum_radius=0.55).fit(_training())

    move = _move_evidence(0, 2).transition
    toggle = _toggle_evidence(0, 5).transition
    assert binder.predict_token(move, candidates=(1_234_567, "opaque-z")) == 1_234_567
    assert binder.predict_token(toggle, candidates=(1_234_567, "opaque-z")) == "opaque-z"
    assert binder.supports_token(move, 1_234_567) is True
    assert binder.supports_token(move, "opaque-z") is False


def test_signature_is_token_free_deterministic_and_read_only() -> None:
    first = _move_evidence(111, 1)
    second = replace(first, token=999)
    binder = SensorimotorBinder()
    one = binder.signature(first)
    two = binder.signature(second)

    assert one.digest == two.digest
    assert np.array_equal(one.vector, two.vector)
    assert set(one.operational_consequences) >= {
        "changed_fraction",
        "sensor_l1_change",
        "sensor_l2_change",
    }
    assert not one.vector.flags.writeable
    with pytest.raises(ValueError):
        one.vector[0] = 0.0


def test_token_permutation_remaps_predictions_without_changing_meanings() -> None:
    original = SensorimotorBinder(minimum_radius=0.55).fit(_training(101, 202))
    remapped = SensorimotorBinder(minimum_radius=0.55).fit(_training(9_001, 9_002))
    query = _toggle_evidence(0, 3).transition

    assert original.predict_token(query) == 202
    assert remapped.predict_token(query) == 9_002
    assert np.allclose(original.prototype(202).center, remapped.prototype(9_002).center)


def test_abstains_on_novel_or_empirically_indistinguishable_effects() -> None:
    binder = SensorimotorBinder(minimum_radius=0.25).fit(_training())
    before = _observation(3, 3, (220, 40, 30), 0)
    after = _observation(7, 7, (255, 255, 255), 1)
    novel = Transition(before, Action(999_999, (0, 0), (-1, -1)), after, 888_888)
    decision = binder.predict(novel)
    assert decision.abstained
    assert decision.reason == "outside-calibrated-radius"

    same = [_move_evidence(11, index) for index in range(6)]
    same.extend(replace(item, token=22) for item in tuple(same))
    ambiguous = SensorimotorBinder(minimum_radius=0.55).fit(same)
    ambiguity = ambiguous.predict(_move_evidence(0, 2).transition)
    assert ambiguity.abstained
    assert ambiguity.reason == "indistinguishable-operational-signatures"
    assert ambiguous.supports_token(_move_evidence(0, 2).transition, 11) is None


def test_failed_learned_intervention_is_false_and_definition_is_executable() -> None:
    binder = SensorimotorBinder(minimum_radius=0.55).fit(_training(101, 202))
    successful_move = _move_evidence(0, 2)
    failed_move = replace(successful_move, task_feedback=False)

    assert binder.supports_token(failed_move, 101) is False
    assert binder.evaluate_definition(successful_move, And(Atom(101))) is True
    assert binder.evaluate_definition(failed_move, And(Atom(101))) is False

    result = binder.evaluate_definition(
        And(Atom(101), Atom(202)),
        lambda token: token == 101,
    )
    assert result.as_python() is False


def test_action_support_is_conservative_and_privileged_input_is_rejected() -> None:
    binder = SensorimotorBinder().fit(_training(101, 202))
    observation = _move_evidence(0, 0).before
    supported = Action(710_001, (4, 4), (1, 0))
    unsupported = Action(777_777, (4, 4), (1, 0))

    assert binder.choose_action(observation, 101, (unsupported, supported)) is supported
    assert binder.choose_action(observation, 999, (supported,)) is None
    with pytest.raises(ValueError, match="privileged"):
        binder.signature(
            {"pixels": observation.pixels, "oracle": {"predicate": True}},
            supported,
            observation,
        )
