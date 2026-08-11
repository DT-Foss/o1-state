from __future__ import annotations

import ast
from dataclasses import replace
import inspect

import grounding_kernel.perceptual_policy as policy_module
from grounding_kernel.contracts import Action
from grounding_kernel.episode_binder import (
    EpisodeConceptBinder,
    action_target_transcript,
    episode_features,
)
from grounding_kernel.perceptual_policy import (
    ObservationConditionedPolicy,
    VisualTargetSelector,
    audit_binary_control,
)
from grounding_kernel.processworld import (
    OstensiveRecord,
    ProcessConceptKind,
    ProcessHarness,
    PublicEpisode,
)
from grounding_kernel.programs import ActionScheme
from grounding_kernel.v1_adapters import (
    episode_to_query_trace,
    ostensive_record_to_support,
    support_episode_to_binder_record,
)


def _training_case(seed: int = 5151):
    harness = ProcessHarness(seed, world_variant=0, renderer_variant=0)
    records = harness.oracle.process_pair()
    supports = tuple(
        ostensive_record_to_support(
            record,
            turn_id=index,
            remaining_cost=8.0,
        )
        for index, record in enumerate(records)
    )
    binder = EpisodeConceptBinder().fit(
        support_episode_to_binder_record(support) for support in supports
    )
    selector = VisualTargetSelector.from_traces((supports[0].trace,))
    scheme = ActionScheme.from_episode(supports[0].trace)
    token = supports[0].turn.utterance.tokens[0]  # type: ignore[union-attr]
    return harness, binder, selector, scheme, token


def test_full_policy_rebinds_target_after_layout_correlation_reversal() -> None:
    training, binder, selector, scheme, token = _training_case()
    holdout = ProcessHarness(5151, world_variant=5, renderer_variant=905)
    training_target = training.oracle.process_pair()[0].episode.transitions[0].action.target
    holdout_target = holdout.oracle.process_pair()[0].episode.transitions[0].action.target

    assert selector.demonstrated_rank == 0
    assert training_target[1] < training.agent.observe().shape[0] // 4 * 3
    assert holdout_target[1] > holdout.agent.observe().shape[0] // 4 * 3
    assert holdout_target != training_target

    result = ObservationConditionedPolicy(
        selector,
        binder,
        token,
        scheme,
    ).execute(holdout.agent)

    assert result.resolved
    assert tuple(item.prediction for item in result.evidence) == (False, True)
    assert result.trace is not None and not result.trace.has_feedback
    assert all(not hasattr(step, "outcome_code") for step in result.trace.transitions)
    assert holdout.oracle.matches_concept(
        holdout.agent.episode(),
        ProcessConceptKind.RUNNING,
    )
    assert set(VisualTargetSelector.__dataclass_fields__) == {
        "component_descriptor",
        "match_radius",
        "demonstrated_rank",
    }


def test_target_action_and_no_sensor_controls_cannot_fake_visual_rebinding() -> None:
    _training, binder, selector, scheme, token = _training_case()

    target_world = ProcessHarness(5151, world_variant=5, renderer_variant=905)
    target_only = ObservationConditionedPolicy(
        selector,
        binder,
        token,
        scheme,
        mode="target_only",
    ).execute(target_world.agent)
    assert target_only.resolved and target_only.trace is not None
    assert not target_world.oracle.matches_concept(
        target_world.agent.episode(),
        ProcessConceptKind.RUNNING,
    )

    for mode in ("action_only", "no_sensor"):
        world = ProcessHarness(5151, world_variant=5, renderer_variant=905)
        result = ObservationConditionedPolicy(
            selector,
            binder,
            token,
            scheme,
            mode=mode,  # type: ignore[arg-type]
        ).execute(world.agent)
        assert not result.resolved
        assert result.trace is None
        assert result.evidence == ()
        assert world.agent.episode().transitions == ()


def test_control_audit_counts_perfect_inversion_as_information_not_abstention() -> None:
    inverted = audit_binary_control((False, True), (True, False))
    abstained = audit_binary_control((None, None), (True, False))

    assert inverted.coverage == 1.0
    assert inverted.accuracy == 0.0
    assert inverted.inverted_accuracy == 1.0
    assert inverted.informative
    assert abstained.coverage == 0.0
    assert not abstained.informative


def test_complete_action_target_controls_are_noninformative_after_counterfactual_fix() -> None:
    training = ProcessHarness(5151, world_variant=0, renderer_variant=0)
    supports = tuple(
        support_episode_to_binder_record(
            ostensive_record_to_support(record, turn_id=index, remaining_cost=4.0)
        )
        for index, record in enumerate(training.oracle.process_pair())
    )
    holdout = ProcessHarness(5151, world_variant=5, renderer_variant=905)
    queries = tuple(
        episode_to_query_trace(record.episode) for record in holdout.oracle.process_pair()
    )
    expected = (True, False)

    for mode in ("target_only", "action_target_only", "action_only", "no_sensor"):
        binder = EpisodeConceptBinder(mode=mode).fit(supports)  # type: ignore[arg-type]
        predictions = tuple(
            binder.supports_token(query, supports[0].token) for query in queries
        )
        assert predictions == (None, None)
        assert not audit_binary_control(predictions, expected).informative


def test_full_target_trajectory_baseline_reproduces_old_leak_but_fixed_pair_blocks_it() -> None:
    def retarget(
        record: object,
        offsets: tuple[int, ...],
        label: bool,
    ) -> OstensiveRecord:
        episode = getattr(record, "episode")
        origin = episode.transitions[0].action.target
        transitions = tuple(
            replace(
                step,
                action=Action(
                    step.action.code,
                    (origin[0] + offset, origin[1]),
                    step.action.vector,
                ),
            )
            for step, offset in zip(episode.transitions, offsets, strict=True)
        )
        return OstensiveRecord(
            getattr(record, "token"),
            PublicEpisode(transitions, episode.scalar_feedback),
            label,
        )

    def tracker_rule(record: OstensiveRecord) -> bool:
        transcript = action_target_transcript(record.episode)
        return transcript[-1][1][0] > transcript[-2][1][0]

    # This is the pre-fix tracker transcript: positive retargets after the
    # final displacement while the stalled negative remains at the old x.
    legacy_separated = 0
    fixed_separated = 0
    first_legacy: tuple[OstensiveRecord, OstensiveRecord] | None = None
    first_fixed: tuple[OstensiveRecord, OstensiveRecord] | None = None
    for block in range(32):
        pair = ProcessHarness(
            5151 + block,
            world_variant=block,
            renderer_variant=10_000 + block,
        ).oracle.process_pair()
        assert action_target_transcript(pair[0].episode) == action_target_transcript(
            pair[1].episode
        )
        legacy = (
            retarget(pair[0], (0, 5, 5, 10), True),
            retarget(pair[1], (0, 5, 5, 5), False),
        )
        legacy_separated += tuple(map(tracker_rule, legacy)) == (True, False)
        fixed_separated += tuple(map(tracker_rule, pair)) == (True, False)
        first_legacy = first_legacy or legacy
        first_fixed = first_fixed or pair

    assert legacy_separated == 32
    assert fixed_separated == 0
    assert first_legacy is not None and first_fixed is not None
    legacy_positive, legacy_negative = first_legacy
    assert episode_features(legacy_positive.episode, "no_sensor").size == 0
    legacy_binder = EpisodeConceptBinder(mode="no_sensor").fit(
        (legacy_positive, legacy_negative)
    )
    assert (
        legacy_binder.supports_token(legacy_positive.episode, legacy_positive.token)
        is True
    )
    assert (
        legacy_binder.supports_token(legacy_negative.episode, legacy_negative.token)
        is False
    )

    fixed_binder = EpisodeConceptBinder(mode="no_sensor").fit(first_fixed)
    assert (
        fixed_binder.supports_token(first_fixed[0].episode, first_fixed[0].token)
        is None
    )
    assert (
        fixed_binder.supports_token(first_fixed[1].episode, first_fixed[1].token)
        is None
    )


def test_selector_transfers_across_renderers_without_seed_or_evaluator_dependency() -> None:
    _training, _binder, selector, _scheme, _token = _training_case()
    for renderer in (1, 91, 4_001):
        observation = ProcessHarness(
            5151,
            world_variant=5,
            renderer_variant=renderer,
        ).agent.observe()
        assert len(selector.candidates(observation)) == 2

    tree = ast.parse(inspect.getsource(policy_module))
    imports = {
        node.module
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.module is not None
    }
    assert not any(name.endswith(("processworld", "microworld", "protocol")) for name in imports)
