from __future__ import annotations

from dataclasses import FrozenInstanceError

import numpy as np
import pytest

from grounding_kernel.contracts import Action
from grounding_kernel.processworld import (
    AffordanceCounterfactualSet,
    OstensiveRecord,
    ProcessActionKind,
    ProcessConceptKind,
    ProcessHarness,
    ProcessOutcomeKind,
    PublicEpisode,
    audit_process_agent,
    operational_signature,
)


def _crop(frame: np.ndarray, center: tuple[int, int], radius: int = 11) -> np.ndarray:
    x, y = center
    return frame[y - radius : y + radius, x - radius : x + radius]


def test_deterministic_world_and_fresh_opaque_codebooks() -> None:
    first = ProcessHarness(1234, renderer_variant=8)
    replay = ProcessHarness(1234, renderer_variant=8)
    fresh = ProcessHarness(1235, renderer_variant=8)

    assert first.agent.manifest == replay.agent.manifest
    assert first.agent.observe() == replay.agent.observe()
    assert first.oracle.snapshot() == replay.oracle.snapshot()
    assert first.oracle.examples(include_negative_control=True) == replay.oracle.examples(
        include_negative_control=True
    )

    assert set(first.agent.action_codes).isdisjoint(fresh.agent.action_codes)
    assert set(first.agent.concept_codes).isdisjoint(fresh.agent.concept_codes)
    assert not np.array_equal(first.agent.observe().pixels, fresh.agent.observe().pixels)
    assert {len(str(code)) for code in first.agent.action_codes + first.agent.concept_codes} == {
        9
    }


def test_agent_surface_has_pixels_and_opaque_integers_but_no_privileged_names() -> None:
    harness = ProcessHarness(88)
    agent = harness.agent

    assert audit_process_agent(agent) == ()
    for forbidden in (
        "seed",
        "oracle",
        "codebook",
        "objects",
        "object_ids",
        "decode_action",
        "decode_outcome",
        "decode_concept",
        "part_of_speech",
        "pos",
    ):
        assert not hasattr(agent, forbidden)
    assert all(type(code) is int for code in agent.action_codes + agent.concept_codes)
    assert agent.observe().pixels.dtype == np.uint8
    assert not agent.observe().pixels.flags.writeable
    assert set(agent.manifest.__dataclass_fields__) == {
        "observation_shape",
        "action_codes",
        "concept_codes",
        "motor_vectors",
        "max_steps",
    }
    assert set(OstensiveRecord.__dataclass_fields__) == {"token", "episode", "task_feedback"}


def test_renderer_nuisance_changes_pixels_but_not_operational_meaning() -> None:
    first = ProcessHarness(77, renderer_variant=0)
    second = ProcessHarness(77, renderer_variant=999)

    assert first.agent.manifest == second.agent.manifest
    assert first.oracle.snapshot() == second.oracle.snapshot()
    assert not np.array_equal(first.agent.observe().pixels, second.agent.observe().pixels)

    methods = ("affordance_pair", "movement_pair", "process_pair", "context_pair")
    for method in methods:
        first_records = getattr(first.oracle, method)()
        second_records = getattr(second.oracle, method)()
        for left, right in zip(first_records, second_records, strict=True):
            assert left.token == right.token
            assert left.task_feedback == right.task_feedback
            assert operational_signature(left.episode) == operational_signature(right.episode)
            assert left.episode.non_sensor_transcript() == right.episode.non_sensor_transcript()


def test_new_instances_preserve_concept_contract_under_fresh_lexicons() -> None:
    first = ProcessHarness(400)
    second = ProcessHarness(401)

    for harness in (first, second):
        decoded = {harness.oracle.decode_concept(code) for code in harness.agent.concept_codes}
        assert decoded == set(ProcessConceptKind)
        positive_shelter, negative_shelter = harness.oracle.affordance_pair()
        positive_run, negative_run = harness.oracle.process_pair()
        assert harness.oracle.matches_concept(
            positive_shelter.episode, ProcessConceptKind.SHELTER
        )
        assert not harness.oracle.matches_concept(
            negative_shelter.episode, ProcessConceptKind.SHELTER
        )
        assert harness.oracle.matches_concept(positive_run.episode, ProcessConceptKind.RUNNING)
        assert not harness.oracle.matches_concept(
            negative_run.episode, ProcessConceptKind.RUNNING
        )

    assert first.oracle.encode_concept(ProcessConceptKind.SHELTER) != second.oracle.encode_concept(
        ProcessConceptKind.SHELTER
    )


def test_world_variant_changes_instances_without_changing_public_codebook() -> None:
    first = ProcessHarness(400, world_variant=0)
    changed = ProcessHarness(400, world_variant=17)

    assert first.agent.manifest == changed.agent.manifest
    assert first.agent.action_codes == changed.agent.action_codes
    assert first.agent.concept_codes == changed.agent.concept_codes
    assert first.oracle.snapshot() != changed.oracle.snapshot()
    for harness in (first, changed):
        positive, negative = harness.oracle.affordance_pair()
        assert harness.oracle.matches_concept(positive.episode, ProcessConceptKind.SHELTER)
        assert not harness.oracle.matches_concept(
            negative.episode, ProcessConceptKind.SHELTER
        )


def test_visually_matched_structures_diverge_only_through_affordance_intervention() -> None:
    harness = ProcessHarness(900)
    snapshot = harness.oracle.snapshot()
    frame = harness.agent.observe().pixels
    protective_crop = _crop(
        frame, harness.oracle.structure_center(snapshot.protective_structure)
    )
    nonprotective_crop = _crop(
        frame, harness.oracle.structure_center(snapshot.nonprotective_structure)
    )

    # Neither shape, colour nor texture tells the learner which structure
    # protects. Only enter -> hazard consequences separate the twins.
    assert np.array_equal(protective_crop, nonprotective_crop)
    positive, negative = harness.oracle.affordance_pair()
    assert positive.token == negative.token
    assert positive.task_feedback and not negative.task_feedback
    assert tuple(step.action for step in positive.episode.transitions) == tuple(
        step.action for step in negative.episode.transitions
    )
    assert positive.episode.observations[0] == negative.episode.observations[0]
    assert [harness.oracle.decode_action(step.action.code) for step in positive.episode.transitions] == [
        ProcessActionKind.ENTER,
        ProcessActionKind.HAZARD,
    ]
    assert harness.oracle.decode_outcome(
        positive.episode.transitions[-1].outcome_code
    ) is ProcessOutcomeKind.PROTECTED
    assert harness.oracle.decode_outcome(
        negative.episode.transitions[-1].outcome_code
    ) is ProcessOutcomeKind.DAMAGED
    # The causal result is in raw geometry as well as the legacy opaque
    # outcome channel: v1 can drop outcome_code without losing the contrast.
    assert not np.array_equal(
        positive.episode.transitions[-1].after.pixels,
        negative.episode.transitions[-1].after.pixels,
    )
    assert np.count_nonzero(
        positive.episode.transitions[-1].before.pixels
        != positive.episode.transitions[-1].after.pixels
    ) == np.count_nonzero(
        negative.episode.transitions[-1].before.pixels
        != negative.episode.transitions[-1].after.pixels
    )


def test_move_is_temporally_extended_displacement_not_a_single_frame_class() -> None:
    harness = ProcessHarness(72)
    left, right = harness.oracle.movement_pair()

    assert left.task_feedback and right.task_feedback
    assert len(left.episode.transitions) == 3
    assert left.episode.non_sensor_transcript() == right.episode.non_sensor_transcript()
    assert operational_signature(left.episode) == operational_signature(right.episode)
    assert harness.oracle.matches_concept(left.episode, ProcessConceptKind.MOVING)
    assert harness.oracle.matches_concept(right.episode, ProcessConceptKind.MOVING)

    one_frame = PublicEpisode((left.episode.transitions[0],))
    assert not harness.oracle.matches_concept(one_frame, ProcessConceptKind.MOVING)


def test_passive_identical_processes_diverge_only_after_perturbation() -> None:
    harness = ProcessHarness(314)
    passive_self, passive_external = harness.oracle.passive_process_pair()

    assert passive_self.non_sensor_transcript() == passive_external.non_sensor_transcript()
    assert operational_signature(passive_self) == operational_signature(passive_external)

    self_sustaining, externally_driven = harness.oracle.process_pair()
    # Every public action field and outcome code is the same. The divergence
    # is in ordered pixels after do(perturb), not in a target-tracker side
    # channel or shortcut label.
    assert tuple(step.action for step in self_sustaining.episode.transitions) == tuple(
        step.action for step in externally_driven.episode.transitions
    )
    assert len({step.action.target for step in self_sustaining.episode.transitions}) == 1
    assert self_sustaining.episode.observations[0] == externally_driven.episode.observations[0]
    assert self_sustaining.episode.non_sensor_transcript().steps == (
        externally_driven.episode.non_sensor_transcript().steps
    )
    assert operational_signature(self_sustaining.episode) != operational_signature(
        externally_driven.episode
    )
    assert self_sustaining.task_feedback and not externally_driven.task_feedback
    assert harness.oracle.matches_concept(
        self_sustaining.episode, ProcessConceptKind.RUNNING
    )
    assert not harness.oracle.matches_concept(
        externally_driven.episode, ProcessConceptKind.RUNNING
    )


def test_target_roles_are_separate_from_matched_causal_process_twins() -> None:
    harness = ProcessHarness(315)

    first_role, second_role = harness.oracle.target_role_pair()
    first_role_actions = tuple(step.action for step in first_role.transitions)
    second_role_actions = tuple(step.action for step in second_role.transitions)
    assert first_role_actions != second_role_actions
    assert {step.action.target for step in first_role.transitions}.isdisjoint(
        {step.action.target for step in second_role.transitions}
    )

    positive, negative = harness.oracle.process_pair()
    assert tuple(step.action for step in positive.episode.transitions) == tuple(
        step.action for step in negative.episode.transitions
    )


def test_temporal_order_shuffle_breaks_process_identity() -> None:
    harness = ProcessHarness(2718)
    episode = harness.oracle.process_pair()[0].episode
    shuffled = episode.reordered((0, 2, 1, 3))

    assert operational_signature(shuffled) != operational_signature(episode)
    assert episode.non_sensor_transcript() != shuffled.non_sensor_transcript()
    assert harness.oracle.matches_concept(episode, ProcessConceptKind.RUNNING)
    assert not harness.oracle.matches_concept(shuffled, ProcessConceptKind.RUNNING)


def test_context_pair_blocks_every_nonsensory_transcript_shortcut() -> None:
    harness = ProcessHarness(1618)
    inside, outside = harness.oracle.context_pair()

    assert inside.token == outside.token
    assert inside.task_feedback and not outside.task_feedback
    assert inside.episode.non_sensor_transcript() == outside.episode.non_sensor_transcript()
    assert operational_signature(inside.episode) == operational_signature(outside.episode)
    inside_step = inside.episode.transitions[0]
    outside_step = outside.episode.transitions[0]
    assert inside_step.action == outside_step.action
    assert inside_step.outcome_code == outside_step.outcome_code
    assert inside.episode.scalar_feedback == outside.episode.scalar_feedback
    assert np.count_nonzero(inside_step.before.pixels != inside_step.after.pixels) == np.count_nonzero(
        outside_step.before.pixels != outside_step.after.pixels
    )
    assert not np.array_equal(inside_step.before.pixels, outside_step.before.pixels)
    assert not np.array_equal(inside_step.after.pixels, outside_step.after.pixels)


def test_negative_control_has_opposite_truth_on_identical_public_episodes() -> None:
    harness = ProcessHarness(99)
    left, right = harness.oracle.negative_control_pair()

    assert left.token == right.token
    assert left.task_feedback != right.task_feedback
    assert left.episode == right.episode
    assert left.episode.non_sensor_transcript() == right.episode.non_sensor_transcript()
    assert operational_signature(left.episode) == operational_signature(right.episode)
    assert harness.oracle.negative_control_invariant()
    with pytest.raises(ValueError, match="unidentifiable"):
        harness.oracle.matches_concept(left.episode, ProcessConceptKind.NEGATIVE_CONTROL)


def test_public_episodes_are_immutable_and_permutation_checked() -> None:
    episode = ProcessHarness(12).oracle.process_pair()[0].episode

    with pytest.raises(FrozenInstanceError):
        episode.transitions = ()  # type: ignore[misc]
    with pytest.raises(FrozenInstanceError):
        episode.scalar_feedback = ()  # type: ignore[misc]
    with pytest.raises(ValueError, match="permutation"):
        episode.reordered((0, 0, 1, 2))
    with pytest.raises(ValueError, match="align"):
        PublicEpisode(episode.transitions, (0.0,))
    for observation in episode.observations:
        assert not observation.pixels.flags.writeable


def test_interactive_agent_steps_only_with_opaque_codes() -> None:
    harness = ProcessHarness(808)
    snapshot = harness.oracle.snapshot()
    advance = next(
        code
        for code in harness.agent.action_codes
        if harness.oracle.decode_action(code) is ProcessActionKind.ADVANCE
    )
    target = harness.oracle.mover_center(snapshot.self_sustaining_mover)
    transition = harness.agent.step(advance, target)

    assert transition.before.tick == 0
    assert transition.after.tick == 1
    assert transition.pixels_changed
    assert harness.agent.episode().transitions == (transition,)
    with pytest.raises(ValueError, match="unknown opaque"):
        harness.agent.step(Action(123_456_789, target))


def test_counterfactual_bundle_has_matched_public_worlds_and_no_truth_field() -> None:
    bundle = ProcessHarness(717, renderer_variant=8, world_variant=4).oracle.affordance_counterfactuals()

    assert isinstance(bundle, AffordanceCounterfactualSet)
    assert set(AffordanceCounterfactualSet.__dataclass_fields__) == {
        "target",
        "diagnostic_actions",
        "worlds",
    }
    assert len(bundle.worlds) == 4
    assert len(bundle.diagnostic_actions) == 2
    assert all(action.target == bundle.target for action in bundle.diagnostic_actions)
    initial = tuple(world.observe() for world in bundle.worlds)
    assert all(observation == initial[0] for observation in initial[1:])

    final_digests: list[str] = []
    first_digests: list[str] = []
    for world in bundle.worlds:
        world.reset()
        first = world.step(bundle.diagnostic_actions[0])
        second = world.step(bundle.diagnostic_actions[1])
        first_digests.append(first.after.digest())
        final_digests.append(second.after.digest())
    assert len(set(first_digests)) == 1
    # Default pattern order varies the non-target bit inside each adjacent
    # pair, proving that only the fixed target's affordance changes evidence.
    assert final_digests[0] == final_digests[1]
    assert final_digests[2] == final_digests[3]
    assert final_digests[0] != final_digests[2]
    assert len({id(world) for world in bundle.worlds}) == len(bundle.worlds)
