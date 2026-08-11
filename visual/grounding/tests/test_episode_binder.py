from __future__ import annotations

import ast
from dataclasses import dataclass
import inspect

import numpy as np
import pytest

import grounding_kernel.episode_binder as binder_module
from grounding_kernel.episode_binder import EpisodeConceptBinder, episode_features
from grounding_kernel.processworld import ProcessHarness
from grounding_kernel.v1_adapters import (
    episode_to_query_trace,
    ostensive_record_to_support,
    support_episode_to_binder_record,
)


@dataclass(frozen=True)
class DuckRecord:
    token: object
    episode: object
    task_feedback: bool


@dataclass(frozen=True)
class OutcomeTrapTransition:
    base: object

    @property
    def before(self) -> object:
        return getattr(self.base, "before")

    @property
    def action(self) -> object:
        return getattr(self.base, "action")

    @property
    def after(self) -> object:
        return getattr(self.base, "after")

    @property
    def outcome_code(self) -> int:
        raise AssertionError("the full v1 binder touched the forbidden outcome channel")


@dataclass(frozen=True)
class OutcomeTrapEpisode:
    transitions: tuple[OutcomeTrapTransition, ...]


def test_running_process_is_learned_from_ordered_pixels_not_outcome_names() -> None:
    train = ProcessHarness(42, renderer_variant=0)
    holdout = ProcessHarness(42, renderer_variant=91)
    training_pair = train.oracle.process_pair()
    positive, external_twin = holdout.oracle.process_pair()
    token = training_pair[0].token
    binder = EpisodeConceptBinder().fit(training_pair)

    assert positive.token == token == external_twin.token
    assert binder.supports_token(positive.episode, token) is True
    assert binder.predict_membership(external_twin.episode, token) is False
    assert binder.predict_token(positive.episode) == token
    assert binder.predict_token(external_twin.episode) is None

    # The holdout episodes are genuinely new raw renderings, not memorized
    # episode byte strings.
    assert training_pair[0].episode.observations[0].digest() != positive.episode.observations[
        0
    ].digest()


def test_temporal_shuffle_is_not_accepted_as_the_learned_process() -> None:
    harness = ProcessHarness(77)
    positive, negative = harness.oracle.process_pair()
    binder = EpisodeConceptBinder().fit((positive, negative))
    shuffled = positive.episode.reordered((0, 2, 1, 3))

    assert binder.supports_token(positive.episode, positive.token) is True
    assert binder.supports_token(shuffled, positive.token) is None
    assert not np.array_equal(
        episode_features(positive.episode), episode_features(shuffled)
    )


def test_renderer_variant_transfer_for_every_identifiable_public_record() -> None:
    training_world = ProcessHarness(5150, renderer_variant=1)
    holdout_world = ProcessHarness(5150, renderer_variant=888)
    binder = EpisodeConceptBinder().fit(training_world.oracle.examples())

    for record in holdout_world.oracle.examples():
        assert binder.supports_token(record.episode, record.token) is record.task_feedback


def test_full_binder_fits_and_queries_outcome_free_v1_public_traces_directly() -> None:
    training: list[object] = []
    turn_id = 0
    for variant in (0, 1, 2):
        for record in ProcessHarness(
            8181,
            renderer_variant=variant,
            world_variant=variant,
        ).oracle.examples():
            support = ostensive_record_to_support(
                record,
                turn_id=turn_id,
                remaining_cost=20.0,
            )
            training.append(support_episode_to_binder_record(support))
            turn_id += 1
    binder = EpisodeConceptBinder().fit(training)
    holdout = ProcessHarness(8181, renderer_variant=700, world_variant=701)

    for record in holdout.oracle.examples():
        query = episode_to_query_trace(record.episode)
        assert not query.has_feedback
        assert all(not hasattr(step, "outcome_code") for step in query.transitions)
        assert binder.supports_token(query, record.token) is record.task_feedback


def test_full_path_never_even_probes_an_outcome_property() -> None:
    records = ProcessHarness(909).oracle.process_pair()
    trapped = tuple(
        DuckRecord(
            record.token,
            OutcomeTrapEpisode(
                tuple(OutcomeTrapTransition(step) for step in record.episode.transitions)
            ),
            record.task_feedback,
        )
        for record in records
    )
    binder = EpisodeConceptBinder(mode="full").fit(trapped)

    assert binder.supports_token(trapped[0].episode, trapped[0].token) is True
    assert binder.supports_token(trapped[1].episode, trapped[1].token) is False


def test_new_world_instance_and_renderer_transfer_with_fixed_opaque_codebook() -> None:
    training = tuple(
        record
        for variant in (0, 1, 2)
        for record in ProcessHarness(
            5151,
            world_variant=variant,
            renderer_variant=variant,
        ).oracle.examples()
    )
    holdout = ProcessHarness(5151, world_variant=999, renderer_variant=888)
    binder = EpisodeConceptBinder().fit(training)

    for record in holdout.oracle.examples():
        assert binder.supports_token(record.episode, record.token) is record.task_feedback


@pytest.mark.parametrize("mode", ["no_sensor", "action_outcome_only"])
def test_matched_context_forces_sensor_use_and_ablation_abstention(mode: str) -> None:
    training_world = ProcessHarness(1618, renderer_variant=0)
    holdout_world = ProcessHarness(1618, renderer_variant=500)
    train_pair = training_world.oracle.context_pair()
    test_pair = holdout_world.oracle.context_pair()

    full = EpisodeConceptBinder(mode="full").fit(train_pair)
    assert [full.supports_token(record.episode, record.token) for record in test_pair] == [
        True,
        False,
    ]

    ablated = EpisodeConceptBinder(mode=mode).fit(train_pair)  # type: ignore[arg-type]
    assert [ablated.supports_token(record.episode, record.token) for record in test_pair] == [
        None,
        None,
    ]
    assert ablated.ledger[0].contradictory_regions >= 1


def test_action_outcome_marginals_cannot_solve_process_twin() -> None:
    harness = ProcessHarness(271)
    records = harness.oracle.process_pair()

    for mode in ("no_sensor", "action_outcome_only"):
        binder = EpisodeConceptBinder(mode=mode).fit(records)  # type: ignore[arg-type]
        assert binder.supports_token(records[0].episode, records[0].token) is None
        assert binder.supports_token(records[1].episode, records[1].token) is None


def test_negative_control_contradiction_stays_unknown_in_every_mode() -> None:
    harness = ProcessHarness(999)
    left, right = harness.oracle.negative_control_pair()
    assert left.episode == right.episode
    assert left.task_feedback != right.task_feedback

    for mode in ("full", "no_sensor", "action_outcome_only"):
        binder = EpisodeConceptBinder(mode=mode).fit((left, right))  # type: ignore[arg-type]
        assert binder.supports_token(left.episode, left.token) is None
        assert binder.predict_token(left.episode) is None
        assert binder.ledger[0].contradictory_regions == 1


def test_fresh_token_and_unseen_temporal_schema_are_unknown() -> None:
    harness = ProcessHarness(303)
    records = harness.oracle.process_pair()
    binder = EpisodeConceptBinder().fit(records)

    assert binder.supports_token(records[0].episode, 987_654_321_987) is None
    assert binder.predict_membership(records[0].episode.reordered((3, 2, 1, 0)), records[0].token) is None
    assert binder.predict_token(records[0].episode, candidates=(987_654_321_987,)) is None


def test_token_remapping_is_exactly_equivariant() -> None:
    harness = ProcessHarness(606)
    records = harness.oracle.examples()
    remapping = {token: token + 2_000_000_000 for token in {record.token for record in records}}
    remapped = tuple(
        DuckRecord(remapping[record.token], record.episode, record.task_feedback)
        for record in records
    )
    original_binder = EpisodeConceptBinder().fit(records)
    remapped_binder = EpisodeConceptBinder().fit(remapped)

    for record in records:
        original = original_binder.supports_token(record.episode, record.token)
        renamed = remapped_binder.supports_token(record.episode, remapping[record.token])
        assert renamed is original
        original_prediction = original_binder.predict_token(record.episode)
        remapped_prediction = remapped_binder.predict_token(record.episode)
        assert remapped_prediction == (
            None if original_prediction is None else remapping[original_prediction]
        )


def test_fit_accepts_duck_mappings_and_is_order_deterministic() -> None:
    records = ProcessHarness(404).oracle.examples()
    mappings = tuple(
        {
            "token": record.token,
            "episode": record.episode,
            "task_feedback": record.task_feedback,
        }
        for record in records
    )
    forward = EpisodeConceptBinder().fit(mappings)
    reverse = EpisodeConceptBinder().fit(reversed(mappings))

    assert forward.tokens == reverse.tokens
    assert forward.ledger == reverse.ledger
    assert forward.manifest == reverse.manifest
    assert forward.manifest.digest() == reverse.manifest.digest()
    assert forward.manifest.records_seen == len(records)
    for record in records:
        assert forward.supports_token(record.episode, record.token) is reverse.supports_token(
            record.episode, record.token
        )


def test_feature_vectors_are_readonly_compact_prototypes_not_episode_lookup() -> None:
    episode = ProcessHarness(17).oracle.process_pair()[0].episode
    features = episode_features(episode)

    assert features.ndim == 1
    assert features.size < sum(observation.pixels.size for observation in episode.observations)
    assert not features.flags.writeable
    with pytest.raises(ValueError):
        features[0] = 1.0
    assert episode_features(episode, "no_sensor").size == 0
    assert episode_features(episode, "action_outcome_only").size == 0


def test_module_has_no_evaluator_or_process_enum_dependency() -> None:
    tree = ast.parse(inspect.getsource(binder_module))
    imported_modules = {
        node.module
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.module is not None
    }
    imported_names = {
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, (ast.Import, ast.ImportFrom))
        for alias in node.names
    }

    assert not any(name.endswith(("processworld", "microworld", "protocol")) for name in imported_modules)
    assert not any("ConceptKind" in name or "OutcomeKind" in name for name in imported_names)


def test_invalid_inputs_fail_without_mutating_previous_model() -> None:
    records = ProcessHarness(80).oracle.process_pair()
    binder = EpisodeConceptBinder().fit(records)
    manifest = binder.manifest

    with pytest.raises(ValueError, match="mode"):
        EpisodeConceptBinder(mode="pixels_only")  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="Boolean"):
        binder.fit((DuckRecord(records[0].token, records[0].episode, 1),))  # type: ignore[arg-type]
    assert binder.manifest == manifest
