from __future__ import annotations

from dataclasses import fields
from pathlib import Path

import numpy as np
import pytest

from grounding_kernel.certificates import manifest_hash
from grounding_kernel.unified_grounder import PersistentOperationalGrounder
from grounding_kernel.v1_contracts import PublicTrace, SessionPhase
from grounding_kernel.v1_grade3_cases import (
    GRADE3_CASE_VERSION,
    Grade3PublicCase,
    build_grade3_case,
)
from grounding_kernel.v1_grade3_contracts import (
    MotorDirective,
    MotorPhase,
    MotorQuery,
    ProbeEvidence,
    ProbeResult,
    TraceBeliefQuery,
    TraceDescriptionQuery,
)
from grounding_kernel.v1_grade3_isolation import commit_grade3_candidate
from grounding_kernel.v1_grade3_runner import Grade3EvaluationRunner


def _surface_tokens(bundle: object) -> frozenset[int]:
    facts = bundle.evaluator.facts  # type: ignore[attr-defined]
    return frozenset(
        {
            facts.affordance_predicate_token,
            facts.process_predicate_token,
            facts.positive_role_token,
            facts.negative_role_token,
            facts.move_scheme_token,
            facts.run_scheme_token,
            facts.grounded_definition_base,
            facts.grounded_definition_middle,
            facts.grounded_definition_chain,
            *facts.unanchored_cycle,
        }
    )


def _run_acquisition(
    grounder: PersistentOperationalGrounder,
    bundle: object,
) -> tuple[ProbeEvidence, ...]:
    public = bundle.public  # type: ignore[attr-defined]
    bank = bundle.evaluator.probes  # type: ignore[attr-defined]
    remaining = public.session_manifest.acquisition_cost_budget
    used: list[int] = []
    evidence: list[ProbeEvidence] = []
    for step_index in range(2):
        offer = public.offer(
            step_index=step_index,
            remaining_cost=remaining,
            exclude=used,
        )
        decision = grounder.choose_probe(offer)
        assert decision.probe_id is not None
        option = next(value for value in offer.options if value.probe_id == decision.probe_id)
        trace = bank.callbacks[decision.probe_id]()
        remaining -= option.cost
        grounder.observe_probe(
            ProbeResult(
                public.scope_id,
                public.problem_id,
                decision.probe_id,
                trace,
                option.cost,
                remaining,
            )
        )
        evidence.append(ProbeEvidence(decision.probe_id, trace))
        used.append(decision.probe_id)
    return tuple(evidence)


def test_case_is_deterministic_and_commits_every_public_record() -> None:
    first = build_grade3_case(41, support_worlds=2)
    replay = build_grade3_case(41, support_worlds=2)
    public = first.public

    assert public == replay.public
    assert public.case_manifest.version == GRADE3_CASE_VERSION
    assert public.case_manifest.ostensive_record_count == 27
    assert public.case_manifest.causal_record_count == 16
    assert len(public.support_records) == public.session_manifest.support_record_budget
    assert public.case_manifest.public_dataset_commitment == (
        replay.public.case_manifest.public_dataset_commitment
    )
    assert first.evaluator.probes.execute(public.probe_options[0].probe_id) == (
        replay.evaluator.probes.execute(public.probe_options[0].probe_id)
    )


def test_public_half_contains_no_oracle_seed_truth_or_outcome_channel() -> None:
    bundle = build_grade3_case(91, support_worlds=2)
    public = bundle.public
    public_fields = {field.name for field in fields(Grade3PublicCase)}

    assert public_fields.isdisjoint(
        {
            "seed",
            "oracle",
            "evaluator",
            "facts",
            "true_hypothesis_id",
            "hypothesis_patterns",
            "renderer_variant",
            "permutation_variant",
        }
    )
    assert _surface_tokens(bundle).isdisjoint(public.action_space.action_codes)
    assert all(record.scope_id == public.scope_id for record in public.support_records)
    for record in public.ostensive_support:
        assert record.turn.phase is SessionPhase.SUPPORT
        assert not record.trace.has_feedback
        assert record.turn.observation == record.trace.initial
        for transition in record.trace.transitions:
            assert transition.scalar_feedback is None
            assert not hasattr(transition, "outcome_code")
    for record in public.causal_support:
        assert not record.trace.has_feedback
        assert record.problem_id == public.problem_id
        for transition in record.trace.transitions:
            assert transition.scalar_feedback is None
            assert not hasattr(transition, "outcome_code")


def test_ostensive_support_identifies_concepts_roles_schemes_and_factorial() -> None:
    bundle = build_grade3_case(132, support_worlds=2)
    public = bundle.public
    facts = bundle.evaluator.facts

    unary: dict[int, list[object]] = {}
    observed_utterances: list[tuple[int, ...]] = []
    definitions: dict[int, tuple[int, ...]] = {}
    for record in public.ostensive_support:
        assert record.turn.utterance is not None
        tokens = record.turn.utterance.tokens
        observed_utterances.append(tokens)
        if len(tokens) == 1:
            unary.setdefault(tokens[0], []).append(record)
        if not record.trace.transitions and record.turn.scalar_feedback == 0.0:
            definitions[tokens[0]] = tokens[1:]

    for token in (
        facts.affordance_predicate_token,
        facts.process_predicate_token,
    ):
        corrections = {record.turn.scalar_feedback for record in unary[token]}
        assert corrections == {-1.0, 1.0}
        assert all(record.turn.ostensive_pixel_cue is None for record in unary[token])
    for token in (facts.positive_role_token, facts.negative_role_token):
        assert {record.turn.scalar_feedback for record in unary[token]} == {1.0}
        assert all(record.turn.ostensive_pixel_cue is not None for record in unary[token])
    for token in (facts.move_scheme_token, facts.run_scheme_token):
        assert {record.turn.scalar_feedback for record in unary[token]} == {1.0}
        assert all(record.turn.ostensive_pixel_cue is None for record in unary[token])

    demonstrated = {utterance.tokens for utterance in facts.demonstrated_factorials}
    assert all(observed_utterances.count(tokens) == 2 for tokens in demonstrated)
    assert facts.heldout_factorial.tokens not in observed_utterances
    assert definitions[facts.grounded_definition_base] == (
        facts.negative_role_token,
        facts.move_scheme_token,
    )
    assert definitions[facts.grounded_definition_middle] == (facts.grounded_definition_base,)
    assert definitions[facts.grounded_definition_chain] == (facts.grounded_definition_middle,)
    left, right = facts.unanchored_cycle
    assert definitions == {
        facts.grounded_definition_base: (
            facts.negative_role_token,
            facts.move_scheme_token,
        ),
        facts.grounded_definition_middle: (facts.grounded_definition_base,),
        facts.grounded_definition_chain: (facts.grounded_definition_middle,),
        left: (right,),
        right: (left,),
    }


def test_causal_table_is_complete_and_callbacks_are_real_feedback_free_probes() -> None:
    bundle = build_grade3_case(81723, support_worlds=3)
    public = bundle.public
    facts = bundle.evaluator.facts
    sources = {record.source_id for record in public.causal_support}
    observed_cells = {
        (record.source_id, record.hypothesis_id, record.probe_id)
        for record in public.causal_support
    }
    expected_cells = {
        (source, hypothesis, option.probe_id)
        for source in sources
        for hypothesis in public.hypothesis_candidates
        for option in public.probe_options
    }

    assert len(sources) == 3
    assert observed_cells == expected_cells
    assert all(len(record.trace.transitions) == 4 for record in public.causal_support)
    callbacks = bundle.evaluator.probes.callbacks
    assert set(callbacks) == {option.probe_id for option in public.probe_options}
    evidence = tuple(
        ProbeEvidence(probe_id, callback()) for probe_id, callback in callbacks.items()
    )
    assert all(not item.trace.has_feedback for item in evidence)
    assert all(len(item.trace.transitions) == 4 for item in evidence)
    expected_pattern = next(
        pattern
        for hypothesis, pattern in facts.hypothesis_patterns
        if hypothesis == facts.true_hypothesis_id
    )
    # This seed deliberately has opposite causal bits, so the two actually
    # executed consequences cannot collapse to one response signature.
    assert expected_pattern == (False, True)
    assert tuple(
        tuple(step.pixels_changed for step in item.trace.transitions) for item in evidence
    ) == ((True, False, False, False), (True, False, True, True))
    with pytest.raises(KeyError, match="unknown opaque"):
        bundle.evaluator.probes.execute(1)


def test_surface_permutation_and_renderer_are_independent_nuisance_variants() -> None:
    canonical = build_grade3_case(771, support_worlds=2)
    permuted = build_grade3_case(771, permutation_variant=1, support_worlds=2)
    rerendered = build_grade3_case(771, renderer_variant=1, support_worlds=2)

    assert canonical.public.action_space == permuted.public.action_space
    assert _surface_tokens(canonical) == _surface_tokens(permuted)
    canonical_assignment = (
        canonical.evaluator.facts.affordance_predicate_token,
        canonical.evaluator.facts.process_predicate_token,
        canonical.evaluator.facts.positive_role_token,
        canonical.evaluator.facts.negative_role_token,
        canonical.evaluator.facts.move_scheme_token,
        canonical.evaluator.facts.run_scheme_token,
    )
    permuted_assignment = (
        permuted.evaluator.facts.affordance_predicate_token,
        permuted.evaluator.facts.process_predicate_token,
        permuted.evaluator.facts.positive_role_token,
        permuted.evaluator.facts.negative_role_token,
        permuted.evaluator.facts.move_scheme_token,
        permuted.evaluator.facts.run_scheme_token,
    )
    assert canonical_assignment != permuted_assignment
    assert set(canonical.public.hypothesis_candidates) == set(permuted.public.hypothesis_candidates)
    assert {value.probe_id for value in canonical.public.probe_options} == {
        value.probe_id for value in permuted.public.probe_options
    }

    assert canonical.evaluator.facts == rerendered.evaluator.facts
    assert canonical.public.action_space == rerendered.public.action_space
    assert canonical.public.case_manifest.public_dataset_commitment != (
        rerendered.public.case_manifest.public_dataset_commitment
    )
    assert not np.array_equal(
        canonical.public.ostensive_support[0].trace.initial.pixels,
        rerendered.public.ostensive_support[0].trace.initial.pixels,
    )


@pytest.mark.parametrize(
    ("renderer_variant", "permutation_variant"),
    ((0, 0), (1, 0), (0, 1), (7, 5)),
)
def test_one_persistent_grounder_closes_all_generated_case_seams(
    renderer_variant: int,
    permutation_variant: int,
) -> None:
    bundle = build_grade3_case(
        81723,
        renderer_variant=renderer_variant,
        permutation_variant=permutation_variant,
    )
    public = bundle.public
    facts = bundle.evaluator.facts
    grounder = PersistentOperationalGrounder()
    grounder.begin(public.session_manifest)
    for record in public.support_records:
        grounder.observe_support(record)
    evidence = _run_acquisition(grounder, bundle)
    grounder.freeze()
    frozen = grounder.checkpoint_commitment()

    belief = grounder.trace_belief(
        TraceBeliefQuery(
            1,
            public.scope_id,
            public.problem_id,
            public.hypothesis_candidates,
            evidence,
        )
    )
    assert belief.candidate_probabilities == ((facts.true_hypothesis_id, 1.0),)
    assert belief.unknown_probability == 0.0

    description = grounder.describe(
        TraceDescriptionQuery(
            2,
            public.scope_id,
            (
                ProbeEvidence(
                    public.probe_options[0].probe_id,
                    bundle.evaluator.probes.heldout_description_trace(),
                ),
            ),
        )
    )
    assert description.utterance == public.heldout_instruction
    assert description.unknown_probability == 0.0

    world = bundle.evaluator.probes.fresh_motor_world()
    initial = world.reset()
    for query_id, utterance in enumerate(
        (public.heldout_instruction, public.definition_queries[2]), start=3
    ):
        decision = grounder.motor(
            MotorQuery(
                query_id,
                public.scope_id,
                0,
                utterance,
                MotorPhase.PROBE,
                (),
                PublicTrace(initial),
                public.action_space,
                24.0,
                4,
            )
        )
        assert decision.directive is MotorDirective.ACT
        assert decision.action is not None

    cycle = grounder.motor(
        MotorQuery(
            5,
            public.scope_id,
            0,
            public.definition_queries[3],
            MotorPhase.PROBE,
            (),
            PublicTrace(initial),
            public.action_space,
            24.0,
            4,
        )
    )
    assert cycle.directive is MotorDirective.ABSTAIN
    assert cycle.unknown_probability == 1.0
    assert grounder.checkpoint_commitment() == frozen


def test_case_builder_rejects_nonindependent_support_cardinality() -> None:
    with pytest.raises(ValueError, match="at least 2"):
        build_grade3_case(1, support_worlds=1)
    with pytest.raises(TypeError, match="integer"):
        build_grade3_case(True)  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="ProcessConfig"):
        build_grade3_case(1, config=object())  # type: ignore[arg-type]


def test_isolated_runner_executes_heldout_and_definition_chain_on_one_checkpoint() -> None:
    project = Path(__file__).resolve().parents[1]
    commitment = commit_grade3_candidate(
        project / "grounding_reference_candidate",
        "grounding_reference_candidate.candidate:build",
        sdk_root=project / "grounding_kernel",
    )
    bundle = build_grade3_case(81723)
    public = bundle.public
    assert public.session_manifest.motor_action_cost_budget == 64.0
    assert public.session_manifest.motor_reset_budget == 20
    codebook_commitment = manifest_hash(
        {
            "action_codes": list(public.action_space.action_codes),
            "motor_vectors": [list(value) for value in public.action_space.motor_vectors],
        }
    )

    with Grade3EvaluationRunner(
        commitment,
        public.session_manifest,
        lambda: codebook_commitment,
    ) as runner:
        for record in public.support_records:
            runner.support(record)
        runner.begin_acquisition()
        used: list[int] = []
        for step_index in range(2):
            executed = runner.acquire_probe(
                public.offer(
                    step_index=step_index,
                    remaining_cost=runner.remaining_acquisition_cost,
                    exclude=used,
                ),
                bundle.evaluator.probes.execute,
            )
            assert executed.result is not None
            used.append(executed.result.probe_id)
        frozen = runner.freeze()

        instructions = (
            public.heldout_instruction,
            public.definition_queries[0],
            public.definition_queries[1],
            public.definition_queries[2],
        )
        results = []
        for query_id, instruction in enumerate(instructions, start=10):
            trial = bundle.evaluator.new_motor_trial()
            result = runner.run_motor_episode(
                query_id,
                public.scope_id,
                instruction,
                public.action_space,
                trial.agent.reset,
                trial.agent.step,
            )
            assert result.completed and not result.abstained
            assert trial.score_result(result)
            assert not trial.score(
                result.execution_trace,
                instruction,
                completed_probes=(),
            )
            assert not trial.score(
                result.execution_trace,
                public.definition_queries[3],
                completed_probes=result.completed_probes,
            )
            results.append(result)

        cycle_world = bundle.evaluator.probes.fresh_motor_world()
        cycle = runner.run_motor_episode(
            20,
            public.scope_id,
            public.definition_queries[3],
            public.action_space,
            cycle_world.reset,
            cycle_world.step,
        )
        assert cycle.abstained and not cycle.completed
        assert runner.candidate.assert_frozen() == frozen.checkpoint_commitment
        assert sum(result.action_cost_consumed for result in results) == 45.0
        assert sum(result.resets_consumed for result in results) == 12
        assert runner.remaining_motor_action_cost == 19.0
        assert runner.remaining_motor_resets == 7
