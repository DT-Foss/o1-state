from __future__ import annotations

from dataclasses import dataclass

import pytest

from grounding_kernel.contracts import Action
from grounding_kernel.unified_grounder import PersistentOperationalGrounder
from grounding_kernel.v1_adapters import fresh_opaque_token
from grounding_kernel.v1_contracts import PublicTrace, PublicTransition, Utterance
from grounding_kernel.v1_grade3_cases import Grade3CaseBundle, build_grade3_case
from grounding_kernel.v1_grade3_contracts import (
    MotorDirective,
    MotorPhase,
    MotorQuery,
    ProbeEvidence,
    ProbeResult,
    TraceBeliefQuery,
    TraceDescriptionQuery,
)


@dataclass(frozen=True, slots=True)
class _MotorRun:
    completed: bool
    trace: PublicTrace | None
    probes: tuple[PublicTrace, ...]


def _fit(
    seed: int = 1, *, permutation_variant: int = 0
) -> tuple[
    Grade3CaseBundle,
    PersistentOperationalGrounder,
    tuple[ProbeEvidence, ...],
]:
    bundle = build_grade3_case(seed, permutation_variant=permutation_variant)
    grounder = PersistentOperationalGrounder()
    grounder.begin(bundle.public.session_manifest)
    for record in bundle.public.support_records:
        grounder.observe_support(record)
    used: list[int] = []
    evidence: list[ProbeEvidence] = []
    for index in range(2):
        offer = bundle.public.offer(
            step_index=index,
            remaining_cost=8.0 - 4.0 * index,
            exclude=used,
        )
        decision = grounder.choose_probe(offer)
        assert decision.probe_id is not None
        trace = bundle.evaluator.probes.execute(decision.probe_id)
        remaining = 4.0 - 4.0 * index
        grounder.observe_probe(
            ProbeResult(
                bundle.public.scope_id,
                bundle.public.problem_id,
                decision.probe_id,
                trace,
                4.0,
                remaining,
            )
        )
        used.append(decision.probe_id)
        evidence.append(ProbeEvidence(decision.probe_id, trace))
    grounder.freeze()
    return bundle, grounder, tuple(evidence)


def _run_motor(
    bundle: Grade3CaseBundle,
    grounder: PersistentOperationalGrounder,
    utterance: Utterance,
) -> _MotorRun:
    world = bundle.evaluator.probes.fresh_motor_world()
    current = PublicTrace(world.reset())
    completed: list[PublicTrace] = []
    phase = MotorPhase.PROBE
    for step_index in range(64):
        decision = grounder.motor(
            MotorQuery(
                10,
                bundle.public.scope_id,
                step_index,
                utterance,
                phase,
                tuple(completed),
                current,
                bundle.public.action_space,
                64.0,
                16,
            )
        )
        if decision.directive is MotorDirective.ACT:
            assert isinstance(decision.action, Action)
            raw = world.step(decision.action)
            current = current.append(PublicTransition(raw.before, raw.action, raw.after, None))
            continue
        if decision.directive in {
            MotorDirective.RESET_PROBE,
            MotorDirective.RESET_EXECUTE,
        }:
            completed.append(current)
            current = PublicTrace(world.reset())
            phase = (
                MotorPhase.EXECUTE
                if decision.directive is MotorDirective.RESET_EXECUTE
                else MotorPhase.PROBE
            )
            continue
        if decision.directive is MotorDirective.COMPLETE:
            return _MotorRun(True, current, tuple(completed))
        assert decision.directive is MotorDirective.ABSTAIN
        return _MotorRun(False, None, tuple(completed))
    raise AssertionError("motor controller failed to terminate")


@pytest.mark.parametrize("permutation_variant", [0, 3])
def test_one_persistent_multiconcept_learner_composes_describes_and_defines(
    permutation_variant: int,
) -> None:
    bundle, grounder, _evidence = _fit(permutation_variant=permutation_variant)
    checkpoint = grounder.checkpoint_commitment()

    heldout = _run_motor(bundle, grounder, bundle.public.heldout_instruction)
    assert heldout.completed and heldout.trace is not None
    assert len(heldout.trace.transitions) == 4
    description = grounder.describe(
        TraceDescriptionQuery(
            20,
            bundle.public.scope_id,
            (ProbeEvidence(1, heldout.trace),),
        )
    )
    assert description.utterance == bundle.public.heldout_instruction

    for definition in bundle.public.definition_queries[:3]:
        derived = _run_motor(bundle, grounder, definition)
        assert derived.completed and derived.trace is not None
        assert len(derived.trace.transitions) == 3
    for cycle in bundle.public.definition_queries[3:]:
        assert not _run_motor(bundle, grounder, cycle).completed
    assert grounder.checkpoint_commitment() == checkpoint


def test_active_evidence_is_provenance_bound_and_shuffle_fails_closed() -> None:
    bundle, grounder, evidence = _fit(seed=1)
    query = TraceBeliefQuery(
        30,
        bundle.public.scope_id,
        bundle.public.problem_id,
        bundle.public.hypothesis_candidates,
        evidence,
    )
    belief = grounder.trace_belief(query)
    assert belief.candidate_probabilities == ((bundle.evaluator.facts.true_hypothesis_id, 1.0),)

    swapped = (
        ProbeEvidence(evidence[0].probe_id, evidence[1].trace),
        ProbeEvidence(evidence[1].probe_id, evidence[0].trace),
    )
    corrupted = grounder.trace_belief(
        TraceBeliefQuery(
            31,
            bundle.public.scope_id,
            bundle.public.problem_id,
            bundle.public.hypothesis_candidates,
            swapped,
        )
    )
    assert corrupted.candidate_probabilities == ()
    assert corrupted.unknown_probability == 1.0


def test_fresh_symbol_abstains_without_consuming_a_semantic_default() -> None:
    bundle, grounder, _evidence = _fit()
    facts = bundle.evaluator.facts
    occupied = (
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
    )
    unknown = fresh_opaque_token(occupied, nonce=91)

    result = _run_motor(bundle, grounder, Utterance((unknown,)))
    assert not result.completed
    assert result.trace is None
