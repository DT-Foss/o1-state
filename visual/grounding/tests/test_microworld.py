from __future__ import annotations

from dataclasses import FrozenInstanceError

import numpy as np
import pytest

from grounding_kernel.contracts import Action, ActionKind, Observation, OutcomeKind, PredicateKind
from grounding_kernel import microworld as microworld_module
from grounding_kernel.microworld import (
    DEFAULT_INVARIANT_MAX_TRANSITIONS,
    INVARIANT_PROBE_VECTORS,
    MAX_INVARIANT_DEPTH,
    MOTOR_VECTORS,
    EvaluatorHarness,
    WorldConfig,
)
from grounding_kernel.protocol import AgentEnvironment, EvaluatorOracle, audit_agent_boundary


def _positive_object(harness: EvaluatorHarness, predicate: PredicateKind) -> int:
    for state in harness.oracle.snapshot().objects:
        if state.predicates[predicate]:
            return state.object_id
    raise AssertionError(f"world has no positive example for {predicate}")


def _action(harness: EvaluatorHarness, kind: ActionKind, object_id: int, vector=(0, 0)) -> Action:
    code = next(code for code in harness.agent.action_codes if harness.oracle.decode_action(code) is kind)
    return Action(code, harness.oracle.object_center(object_id), vector)


def test_same_seed_reproduces_pixels_codebooks_and_latents() -> None:
    first = EvaluatorHarness(8128)
    second = EvaluatorHarness(8128)

    assert first.agent.manifest == second.agent.manifest
    assert first.agent.observe() == second.agent.observe()
    assert first.oracle.snapshot() == second.oracle.snapshot()

    object_id = first.oracle.snapshot().objects[0].object_id
    action_a = _action(first, ActionKind.LIFT, object_id)
    action_b = _action(second, ActionKind.LIFT, object_id)
    assert action_a == action_b
    assert first.agent.step(action_a) == second.agent.step(action_b)


def test_fresh_seed_changes_opaque_permutations_and_nuisance_rendering() -> None:
    first = EvaluatorHarness(100)
    second = EvaluatorHarness(101)

    assert set(first.agent.action_codes).isdisjoint(second.agent.action_codes)
    assert set(first.agent.symbol_codes).isdisjoint(second.agent.symbol_codes)
    assert not np.array_equal(first.agent.observe().pixels, second.agent.observe().pixels)
    # Every token has the same decimal width, removing a trivial length cue.
    all_codes = first.agent.action_codes + first.agent.symbol_codes
    assert {len(str(code)) for code in all_codes} == {9}


def test_observation_is_owned_readonly_raw_rgb() -> None:
    harness = EvaluatorHarness(7)
    observation = harness.agent.observe()

    assert observation.pixels.dtype == np.uint8
    assert observation.pixels.shape == harness.agent.manifest.observation_shape
    assert observation.pixels.flags.c_contiguous
    assert not observation.pixels.flags.writeable
    with pytest.raises(ValueError):
        observation.pixels[0, 0, 0] = 0

    source = np.zeros((3, 4, 3), dtype=np.uint8)
    record = Observation(source, tick=0)
    source[:] = 255
    assert not record.pixels.any()


def test_agent_capability_has_no_public_oracle_seed_ids_or_decoders() -> None:
    harness = EvaluatorHarness(19)

    assert isinstance(harness.agent, AgentEnvironment)
    assert isinstance(harness.oracle, EvaluatorOracle)
    assert audit_agent_boundary(harness.agent) == ()
    assert not hasattr(harness.agent, "seed")
    assert not hasattr(harness.agent, "oracle")
    assert not hasattr(harness.agent, "codebook")
    assert not hasattr(harness.agent, "objects")
    assert not hasattr(harness.agent, "decode_symbol")
    assert set(vars(harness.agent)) == set() if hasattr(harness.agent, "__dict__") else True
    assert set(harness.agent.manifest.__dataclass_fields__) == {
        "observation_shape",
        "action_codes",
        "symbol_codes",
        "motor_vectors",
        "max_steps",
    }


@pytest.mark.parametrize(
    ("kind", "predicate", "vector", "expected"),
    [
        (ActionKind.LIFT, PredicateKind.LIFTABLE, (0, 0), OutcomeKind.LIFTED),
        (ActionKind.MAGNET, PredicateKind.MAGNETIC, (0, 0), OutcomeKind.ATTRACTED),
        (ActionKind.INSERT, PredicateKind.FITS_SLOT_A, (0, -1), OutcomeKind.INSERTED),
        (ActionKind.TOGGLE, PredicateKind.SWITCHABLE, (0, 0), OutcomeKind.ACTIVATED),
    ],
)
def test_hidden_properties_control_primitive_interventions(
    kind: ActionKind,
    predicate: PredicateKind,
    vector: tuple[int, int],
    expected: OutcomeKind,
) -> None:
    harness = EvaluatorHarness(2026)
    object_id = _positive_object(harness, predicate)
    transition = harness.agent.step(_action(harness, kind, object_id, vector))

    assert harness.oracle.decode_outcome(transition.outcome_code) is expected
    assert transition.pixels_changed


def test_push_affordance_moves_a_positive_object_and_rejects_a_negative() -> None:
    harness = EvaluatorHarness(303)
    movable = _positive_object(harness, PredicateKind.MOVABLE)
    moved = False
    for vector in MOTOR_VECTORS[1:]:
        harness.agent.reset()
        transition = harness.agent.step(_action(harness, ActionKind.PUSH, movable, vector))
        if harness.oracle.decode_outcome(transition.outcome_code) is OutcomeKind.MOVED:
            moved = True
            assert transition.pixels_changed
            break
    assert moved, "at least one cardinal push must be collision-free"

    harness.agent.reset()
    immovable = next(
        state.object_id
        for state in harness.oracle.snapshot().objects
        if not state.predicates[PredicateKind.MOVABLE]
    )
    transition = harness.agent.step(_action(harness, ActionKind.PUSH, immovable, (1, 0)))
    assert harness.oracle.decode_outcome(transition.outcome_code) is OutcomeKind.NO_EFFECT
    assert not transition.pixels_changed


def test_slot_profiles_are_distinguished_only_by_controlled_insertion() -> None:
    harness = EvaluatorHarness(450)
    fits_a = _positive_object(harness, PredicateKind.FITS_SLOT_A)
    wrong = harness.agent.step(_action(harness, ActionKind.INSERT, fits_a, (0, 1)))
    assert harness.oracle.decode_outcome(wrong.outcome_code) is OutcomeKind.MISMATCH
    assert not wrong.pixels_changed

    harness.agent.reset()
    right = harness.agent.step(_action(harness, ActionKind.INSERT, fits_a, (0, -1)))
    assert harness.oracle.decode_outcome(right.outcome_code) is OutcomeKind.INSERTED
    assert right.pixels_changed


def test_codebook_covers_all_meanings_without_retaining_seed() -> None:
    harness = EvaluatorHarness(981)

    assert {harness.oracle.decode_action(code) for code in harness.agent.action_codes} == set(
        ActionKind
    )
    assert {harness.oracle.decode_symbol(code) for code in harness.agent.symbol_codes} == set(
        PredicateKind
    )
    assert not hasattr(harness.oracle, "seed")
    assert PredicateKind.NEGATIVE_CONTROL in {
        harness.oracle.decode_symbol(code) for code in harness.agent.symbol_codes
    }


def test_negative_control_is_counterfactually_intervention_invariant() -> None:
    harness = EvaluatorHarness(1729)
    assert harness.oracle.negative_control_invariant(depth=2)

    first = harness.oracle.snapshot().objects[0].object_id
    sequence = (
        _action(harness, ActionKind.LIFT, first),
        _action(harness, ActionKind.TOGGLE, first),
        _action(harness, ActionKind.INSERT, first, (0, -1)),
    )
    assert harness.oracle.negative_control_invariant(sequence)


def test_intervention_signature_is_intrinsic_across_many_seeds_and_live_states() -> None:
    for seed in range(32):
        harness = EvaluatorHarness(seed, WorldConfig(max_steps=2))
        expected = {}
        for state in harness.oracle.snapshot().objects:
            signature = harness.oracle.intervention_signature(state.object_id)
            predicates = state.predicates
            expected[state.object_id] = signature
            assert signature[ActionKind.PUSH] is (
                OutcomeKind.MOVED if predicates[PredicateKind.MOVABLE] else OutcomeKind.NO_EFFECT
            )
            assert signature[ActionKind.LIFT] is (
                OutcomeKind.LIFTED
                if predicates[PredicateKind.LIFTABLE]
                else OutcomeKind.NO_EFFECT
            )
            assert signature[ActionKind.MAGNET] is (
                OutcomeKind.ATTRACTED
                if predicates[PredicateKind.MAGNETIC]
                else OutcomeKind.NO_EFFECT
            )
            assert signature[ActionKind.INSERT] is OutcomeKind.INSERTED
            assert signature[ActionKind.TOGGLE] is (
                OutcomeKind.ACTIVATED
                if predicates[PredicateKind.SWITCHABLE]
                else OutcomeKind.NO_EFFECT
            )
            assert OutcomeKind.BLOCKED not in signature.values()
            assert OutcomeKind.MISMATCH not in signature.values()

        # Canonical probes remain meaningful after the live episode is terminal
        # and after its geometry/dynamic state has changed.
        first = harness.oracle.snapshot().objects[0].object_id
        harness.agent.step(_action(harness, ActionKind.INSERT, first, (0, -1)))
        harness.agent.step(_action(harness, ActionKind.TOGGLE, first))
        assert harness.agent.observe().terminal
        for object_id, signature in expected.items():
            assert harness.oracle.intervention_signature(object_id) == signature
        assert harness.oracle.negative_control_invariant(depth=0)
        with pytest.raises(ValueError, match="remaining episode budget"):
            harness.oracle.negative_control_invariant(depth=1)


def test_negative_control_search_is_exactly_bounded_and_covers_eight_objects() -> None:
    harness = EvaluatorHarness(8128, WorldConfig(object_count=8))
    branch = 8 * len(ActionKind) * len(INVARIANT_PROBE_VECTORS)
    required = branch + branch**2

    assert required == 40_200
    assert required <= DEFAULT_INVARIANT_MAX_TRANSITIONS
    with pytest.raises(ValueError, match="exceeding max_transitions"):
        harness.oracle.negative_control_invariant(depth=2, max_transitions=required - 1)
    assert harness.oracle.negative_control_invariant(depth=2, max_transitions=required)
    with pytest.raises(ValueError, match="depth must lie"):
        harness.oracle.negative_control_invariant(depth=MAX_INVARIANT_DEPTH + 1)


def test_negative_control_checker_detects_a_second_step_only_leak(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original_apply = microworld_module._Engine._apply

    def apply_with_delayed_leak(
        engine: microworld_module._Engine,
        kind: ActionKind,
        index: int,
        vector: tuple[int, int],
    ) -> OutcomeKind:
        outcome = original_apply(engine, kind, index, vector)
        if engine._tick >= 1:
            engine._objects[index].active = engine._objects[index].negative_control
        return outcome

    monkeypatch.setattr(microworld_module._Engine, "_apply", apply_with_delayed_leak)
    harness = EvaluatorHarness(99)

    assert harness.oracle.negative_control_invariant(depth=1)
    assert not harness.oracle.negative_control_invariant(depth=2)


@pytest.mark.parametrize("seed", range(16))
def test_static_pixels_do_not_render_the_negative_control(seed: int) -> None:
    # Depth zero is the deliberately narrow static-pixels-only diagnostic: it
    # compares the initial RGB observation before exercising any intervention.
    assert EvaluatorHarness(seed).oracle.negative_control_invariant(depth=0)


def test_every_causal_predicate_is_balanced_but_negative_control_has_no_effect() -> None:
    harness = EvaluatorHarness(55)
    snapshot = harness.oracle.snapshot()
    for predicate in PredicateKind:
        values = [state.predicates[predicate] for state in snapshot.objects]
        assert any(values) and not all(values)

    negative_token = harness.oracle.encode_symbol(PredicateKind.NEGATIVE_CONTROL)
    for state in snapshot.objects:
        assert harness.oracle.predicate(state.object_id, negative_token) == state.predicates[
            PredicateKind.NEGATIVE_CONTROL
        ]


def test_trajectory_is_immutable_contiguous_and_resettable() -> None:
    harness = EvaluatorHarness(44)
    initial = harness.agent.observe()
    object_id = harness.oracle.snapshot().objects[0].object_id
    transition = harness.agent.step(_action(harness, ActionKind.LIFT, object_id))
    trajectory = harness.agent.trajectory()

    assert trajectory.initial == initial
    assert trajectory.transitions == (transition,)
    assert trajectory.current == transition.after
    with pytest.raises(FrozenInstanceError):
        trajectory.transitions = ()  # type: ignore[misc]

    reset = harness.agent.reset()
    assert reset == initial
    assert harness.agent.trajectory().transitions == ()


def test_step_accepts_convenient_opaque_code_form_without_semantic_names() -> None:
    harness = EvaluatorHarness(88)
    object_id = harness.oracle.snapshot().objects[0].object_id
    code = next(
        code for code in harness.agent.action_codes if harness.oracle.decode_action(code) is ActionKind.LIFT
    )
    transition = harness.agent.step(code, harness.oracle.object_center(object_id))
    assert transition.action.code == code
    assert transition.before.tick == 0
    assert transition.after.tick == 1


def test_invalid_controls_do_not_advance_world() -> None:
    harness = EvaluatorHarness(66)
    before = harness.agent.observe()

    with pytest.raises(ValueError, match="unknown opaque action code"):
        harness.agent.step(Action(123_456_789, (1, 1)))
    with pytest.raises(ValueError, match="outside"):
        harness.agent.step(Action(harness.agent.action_codes[0], (-1, 0)))
    with pytest.raises(ValueError, match="vector"):
        harness.agent.step(Action(harness.agent.action_codes[0], (1, 1), (1, 1)))
    assert harness.agent.observe() == before


def test_terminal_budget_is_enforced() -> None:
    harness = EvaluatorHarness(5, WorldConfig(max_steps=2))
    code = harness.agent.action_codes[0]
    harness.agent.step(code, (0, 0))
    final = harness.agent.step(code, (0, 0))

    assert final.after.terminal
    assert harness.agent.observe().terminal
    with pytest.raises(RuntimeError, match="terminal"):
        harness.agent.step(code, (0, 0))
