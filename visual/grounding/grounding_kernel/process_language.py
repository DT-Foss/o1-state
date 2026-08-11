"""Outcome-free ProcessWorld reference for perception-bound language learning.

This is still a transparent reference stack, but unlike the legacy macro
benchmark it constructs operational referents from public traces, executes
held-out utterances through active visual target selection in a new world
instance, and describes a raw feedback-free trace only after re-recognizing
its operational program.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from hashlib import sha256
from types import MappingProxyType

from .certificates import manifest_hash
from .closed_loop_programs import (
    ClosedLoopExecution,
    ClosedLoopProgramExecutor,
    ClosedLoopProgramRecognizer,
    PerceptualTargetRole,
    referent_registry,
)
from .composition import And, Atom
from .episode_binder import EpisodeConceptBinder
from .language import (
    Demonstration,
    GroundedLanguageLearner,
    GroundedReferent,
    Resolution,
)
from .perceptual_policy import ObservationConditionedPolicy, VisualTargetSelector
from .processworld import ProcessConceptKind, ProcessHarness
from .programs import ActionScheme, ProgramSchema
from .v1_adapters import (
    episode_to_query_trace,
    fresh_opaque_token,
    ostensive_record_to_support,
    support_episode_to_binder_record,
)


PROCESS_LANGUAGE_VERSION = "perception-bound-process-language-v3-independent-trace-controls"


def _seed(seed: int, namespace: str, block: int) -> int:
    payload = f"{PROCESS_LANGUAGE_VERSION}|{seed}|{namespace}|{block}".encode()
    return int.from_bytes(sha256(payload).digest()[:8], "big")


def _opaque(seed: int, namespace: str, count: int) -> tuple[int, ...]:
    result: list[int] = []
    counter = 0
    while len(result) < count:
        payload = f"{PROCESS_LANGUAGE_VERSION}|{seed}|{namespace}|{counter}".encode()
        value = 100_000_000 + int.from_bytes(sha256(payload).digest()[:8], "big") % 899_999_999
        if value not in result:
            result.append(value)
        counter += 1
    return tuple(result)


@dataclass(frozen=True, slots=True)
class FullUtteranceLookup:
    """Explicit shortcut baseline that cannot factor unseen combinations."""

    table: Mapping[tuple[int, ...], GroundedReferent]

    @classmethod
    def fit(cls, demonstrations: tuple[Demonstration, ...]) -> "FullUtteranceLookup":
        table = {
            tuple(int(token) for token in demo.tokens): demo.referent
            for demo in demonstrations
            if demo.referent is not None
        }
        return cls(MappingProxyType(table))

    def interpret(self, utterance: tuple[int, ...]) -> GroundedReferent | None:
        return self.table.get(utterance)


@dataclass(frozen=True, slots=True)
class ProcessLanguageCase:
    seed: int
    block: int
    world_seed: int
    schema: ProgramSchema
    learner: GroundedLanguageLearner
    remapped_learner: GroundedLanguageLearner
    roles: tuple[PerceptualTargetRole, PerceptualTargetRole]
    schemes: tuple[ActionScheme, ActionScheme]
    referents: Mapping[tuple[int, int], GroundedReferent]
    training: tuple[Demonstration, ...]
    held_out: tuple[int, int]
    remapped_held_out: tuple[int, int]
    remapping: Mapping[int, int]
    derived_token: int
    cycle_tokens: tuple[int, int]
    lookup: FullUtteranceLookup
    process_token: int


@dataclass(frozen=True, slots=True)
class ProcessLanguageReport:
    results: Mapping[str, tuple[bool, bool, Mapping[str, object]]]
    lookup_leaked: bool
    target_only_prediction: bool | None
    definition_leaf_deletion_leaked: bool
    definition_leaf_swap_leaked: bool
    language_fresh_unknown: bool


def _trace_digest(trace: object) -> str:
    return manifest_hash(
        {
            "initial": trace.initial.digest(),
            "actions": [
                {
                    "code": step.action.code,
                    "vector": list(step.action.vector),
                    "target": list(step.action.target),
                }
                for step in trace.transitions
            ],
            "after": [step.after.digest() for step in trace.transitions],
        }
    )


def build_process_language_case(seed: int, block: int) -> ProcessLanguageCase:
    world_seed = _seed(seed, "world", block) % (1 << 31)
    binder_records = []
    target_traces = []
    evidence_digests = []
    representative_process = None
    representative_movement = None
    process_token = None
    public_code_sets: set[int] = set()
    turn_id = 0
    for variant in (0, 1, 2, 3):
        harness = ProcessHarness(
            world_seed,
            renderer_variant=variant,
            world_variant=variant,
        )
        public_code_sets.update(harness.agent.action_codes)
        public_code_sets.update(harness.agent.concept_codes)
        for record in harness.oracle.examples(include_negative_control=True):
            support = ostensive_record_to_support(
                record,
                turn_id=turn_id,
                remaining_cost=1_000.0,
            )
            binder_records.append(support_episode_to_binder_record(support))
            turn_id += 1
        positive, negative = harness.oracle.process_pair()
        traces = (
            episode_to_query_trace(positive.episode),
            episode_to_query_trace(negative.episode),
        )
        target_traces.extend(traces)
        evidence_digests.extend(_trace_digest(trace) for trace in traces)
        representative_process = positive
        representative_movement = harness.oracle.movement_pair()[0]
        process_token = positive.token

    assert representative_process is not None
    assert representative_movement is not None
    assert process_token is not None
    binder = EpisodeConceptBinder(mode="full").fit(binder_records)
    selector = VisualTargetSelector.from_traces(target_traces)
    running = ActionScheme.from_episode(
        episode_to_query_trace(representative_process.episode)
    )
    moving = ActionScheme.from_episode(
        episode_to_query_trace(representative_movement.episode)
    )
    positive_role = PerceptualTargetRole.from_support(
        selector=selector,
        binder=binder,
        token=process_token,
        diagnostic_scheme=running,
        required_membership=True,
        evidence_digests=evidence_digests,
    )
    negative_role = PerceptualTargetRole.from_support(
        selector=selector,
        binder=binder,
        token=process_token,
        diagnostic_scheme=running,
        required_membership=False,
        evidence_digests=evidence_digests,
    )

    target_type, scheme_type = _opaque(seed, f"types-{block}", 2)
    target_positive, target_negative, scheme_move, scheme_run = _opaque(
        seed, f"surface-{block}", 4
    )
    remapped_values = _opaque(seed, f"remap-{block}", 4)
    surface = (target_positive, target_negative, scheme_move, scheme_run)
    if public_code_sets.intersection(surface) or public_code_sets.intersection(
        remapped_values
    ):
        raise RuntimeError("surface tokens collided with a public environment code")
    remapping = MappingProxyType(dict(zip(surface, remapped_values, strict=True)))
    schema = ProgramSchema(target_type, scheme_type)
    referents = referent_registry(
        schema,
        {target_positive: positive_role, target_negative: negative_role},
        {scheme_move: moving, scheme_run: running},
    )
    held_out = (target_positive, scheme_run)
    training = tuple(
        Demonstration(tokens, referent, evidence=("public-grounding", index))
        for index, (tokens, referent) in enumerate(referents.items())
        if tokens != held_out
    )
    remapped_training = tuple(
        Demonstration(
            tuple(remapping[int(token)] for token in demo.tokens),
            demo.referent,
            demo.evidence,
        )
        for demo in training
    )
    learner = GroundedLanguageLearner().fit(training)
    remapped = GroundedLanguageLearner().fit(remapped_training)
    derived, cycle_a, cycle_b = _opaque(seed, f"definitions-{block}", 3)
    if set((derived, cycle_a, cycle_b)).intersection(public_code_sets | set(surface)):
        raise RuntimeError("definition tokens collided with an exposed code")
    learner.add_definitions(
        {
            derived: And(Atom(target_positive), Atom(scheme_run)),
            cycle_a: Atom(cycle_b),
            cycle_b: Atom(cycle_a),
        }
    )
    return ProcessLanguageCase(
        seed,
        block,
        world_seed,
        schema,
        learner,
        remapped,
        (positive_role, negative_role),
        (moving, running),
        referents,
        training,
        held_out,
        tuple(remapping[token] for token in held_out),
        remapping,
        derived,
        (cycle_a, cycle_b),
        FullUtteranceLookup.fit(training),
        process_token,
    )


def _harness(case: ProcessLanguageCase, namespace: str) -> ProcessHarness:
    variant = 20_000 + _seed(case.seed, namespace, case.block) % 500_000
    return ProcessHarness(
        case.world_seed,
        renderer_variant=variant,
        world_variant=variant + 700_000,
    )


def _execute(
    case: ProcessLanguageCase,
    referent: GroundedReferent,
    namespace: str,
) -> tuple[ClosedLoopExecution, bool, str]:
    harness = _harness(case, namespace)
    execution = ClosedLoopProgramExecutor(case.schema).execute(harness.agent, referent)
    is_running = bool(
        execution.trace is not None
        and harness.oracle.matches_concept(
            harness.agent.episode(), ProcessConceptKind.RUNNING
        )
    )
    digest = (
        execution.proof_digest
        if execution.trace is not None
        else manifest_hash({"status": execution.status.value, "namespace": namespace})
    )
    return execution, is_running, digest


def run_process_language_block(seed: int, block: int) -> ProcessLanguageReport:
    case = build_process_language_case(seed, block)
    target = case.referents[case.held_out]
    parsed = case.learner.interpret_instruction(case.held_out)
    execution = None
    behavior = False
    behavior_digest = "unresolved"
    if parsed.resolved and parsed.referent is not None:
        execution, behavior, behavior_digest = _execute(
            case, parsed.referent, "held-out-base"
        )

    # The description path receives an independently executed evaluator-chosen
    # operational program.  It is deliberately not the trace produced by the
    # learner's held-out parse above: otherwise a consistently wrong parser and
    # recognizer could pass by cycle consistency alone.
    sealed_source = _harness(case, "sealed-description-source")
    sealed_execution = ClosedLoopProgramExecutor(case.schema).execute(
        sealed_source.agent,
        target,
    )
    sealed_source_correct = bool(
        sealed_execution.trace is not None
        and sealed_source.oracle.matches_concept(
            sealed_source.agent.episode(), ProcessConceptKind.RUNNING
        )
    )
    recognition = None
    description = None
    described_parse = None
    described_execution = None
    described_behavior = False
    if sealed_execution.trace is not None:
        recognition = ClosedLoopProgramRecognizer(
            case.schema,
            case.roles,
            case.schemes,
        ).recognize(sealed_execution.trace.feedback_stripped())
        if recognition.resolved and recognition.referent is not None:
            description = case.learner.describe(recognition.referent)
            if description.resolved and description.utterance is not None:
                described_parse = case.learner.interpret_instruction(
                    description.utterance
                )
                if described_parse.resolved and described_parse.referent is not None:
                    described_execution, described_behavior, _ = _execute(
                        case,
                        described_parse.referent,
                        "sealed-description-reexecution",
                    )
    independent_trace_round_trip = bool(
        sealed_source_correct
        and recognition is not None
        and recognition.resolved
        and recognition.referent == target
        and description is not None
        and description.resolved
        and description.utterance is not None
        and described_parse is not None
        and described_parse.resolved
        and described_parse.referent == target
        and described_execution is not None
        and described_execution.resolved
        and described_behavior
    )

    external_execution, external_running, _external_digest = _execute(
        case,
        next(
            referent
            for tokens, referent in case.referents.items()
            if tokens[0] != case.held_out[0] and tokens[1] == case.held_out[1]
        ),
        "slot-swap-target",
    )
    moving_harness = _harness(case, "slot-swap-scheme")
    moving_execution = ClosedLoopProgramExecutor(case.schema).execute(
        moving_harness.agent,
        next(
            referent
            for tokens, referent in case.referents.items()
            if tokens[0] == case.held_out[0] and tokens[1] != case.held_out[1]
        ),
    )
    moving_behavior = bool(
        moving_execution.trace is not None
        and moving_harness.oracle.matches_concept(
            moving_harness.agent.episode(), ProcessConceptKind.MOVING
        )
    )
    moving_not_running = bool(
        moving_execution.trace is not None
        and not moving_harness.oracle.matches_concept(
            moving_harness.agent.episode(), ProcessConceptKind.RUNNING
        )
    )
    factorial = (
        case.held_out not in {demo.tokens for demo in case.training}
        and parsed.resolved
        and parsed.referent == target
        and behavior
        and external_execution.resolved
        and not external_running
        and moving_execution.resolved
        and moving_behavior
        and moving_not_running
    )

    remapped_parse = case.remapped_learner.interpret_instruction(
        case.remapped_held_out
    )
    remapped_execution = None
    remapped_behavior = False
    remapped_digest = "unresolved"
    if remapped_parse.resolved and remapped_parse.referent is not None:
        remapped_execution, remapped_behavior, remapped_digest = _execute(
            case,
            remapped_parse.referent,
            "held-out-remapped",
        )
    remapped_description = case.remapped_learner.describe(target)
    permutation = (
        remapped_parse.referent == target
        and remapped_description.utterance == case.remapped_held_out
        and remapped_execution is not None
        and remapped_execution.resolved
        and remapped_behavior
    )

    materialized = case.learner.materialize_definition(case.derived_token)
    derived_execution = None
    derived_behavior = False
    if materialized.resolved and materialized.referent is not None:
        derived_execution, derived_behavior, _derived_digest = _execute(
            case,
            materialized.referent,
            "definition-derived",
        )
    cycle = case.learner.materialize_definition(case.cycle_tokens[0])
    symbolic = (
        materialized.resolved
        and materialized.referent == target
        and derived_execution is not None
        and derived_execution.resolved
        and derived_behavior
        and cycle.status is Resolution.UNKNOWN
    )

    # Mandatory definition controls.  Deleting every direct demonstration of
    # one leaf must make the composite unknown.  Swapping a grounded leaf may
    # still produce a well-typed referent, but it must change the resulting
    # world behaviour rather than secretly executing the intended program.
    deleted_leaf_learner = GroundedLanguageLearner().fit(
        demo for demo in case.training if case.held_out[1] not in demo.tokens
    )
    deleted_leaf_learner.add_definitions(
        {
            case.derived_token: And(
                Atom(case.held_out[0]), Atom(case.held_out[1])
            )
        }
    )
    deleted_leaf = deleted_leaf_learner.materialize_definition(case.derived_token)
    definition_leaf_deletion_leaked = bool(deleted_leaf.resolved)

    swapped_target_token = next(
        tokens[0] for tokens in case.referents if tokens[0] != case.held_out[0]
    )
    swapped_leaf_learner = GroundedLanguageLearner().fit(case.training)
    swapped_leaf_learner.add_definitions(
        {
            case.derived_token: And(
                Atom(swapped_target_token), Atom(case.held_out[1])
            )
        }
    )
    swapped_leaf = swapped_leaf_learner.materialize_definition(case.derived_token)
    swapped_leaf_behavior = False
    if swapped_leaf.resolved and swapped_leaf.referent is not None:
        _swapped_execution, swapped_leaf_behavior, _ = _execute(
            case,
            swapped_leaf.referent,
            "definition-swapped-leaf",
        )
    definition_leaf_swap_leaked = bool(
        swapped_leaf.resolved
        and (swapped_leaf.referent == target or swapped_leaf_behavior)
    )

    existing = tuple(int(value) for value in case.remapping) + tuple(
        int(value) for value in case.remapping.values()
    )
    fresh = fresh_opaque_token(existing, nonce=block)
    language_fresh_unknown = (
        case.learner.meaning(fresh).status is Resolution.UNKNOWN
        and case.learner.interpret((fresh, case.held_out[1])).status
        is Resolution.UNKNOWN
    )
    lookup_leaked = case.lookup.interpret(case.held_out) is not None

    target_only_harness = _harness(case, "target-only-control")
    target_only = ObservationConditionedPolicy(
        case.roles[0].selector,
        case.roles[0].binder,
        case.roles[0].token,
        case.schemes[1],
        mode="target_only",
    ).execute(target_only_harness.agent)
    target_only_prediction = (
        None
        if target_only.trace is None
        else target_only_harness.oracle.matches_concept(
            target_only_harness.agent.episode(), ProcessConceptKind.RUNNING
        )
    )

    common = {
        "language_ledger": case.learner.ledger.digest,
        "remapped_ledger": case.remapped_learner.ledger.digest,
        "held_out_hash": manifest_hash({"tokens": list(case.held_out)}),
        "behavior_digest": behavior_digest,
        "world_variant_split": True,
        "outcome_channel": "absent",
    }
    results: dict[str, tuple[bool, bool, Mapping[str, object]]] = {
        "description_to_action": (
            bool(parsed.resolved and execution is not None and execution.resolved and behavior),
            parsed.resolved,
            {
                **common,
                "closed_loop": True,
                "probe_actions": 0 if execution is None else execution.actions_executed,
            },
        ),
        "trace_to_description": (
            independent_trace_round_trip,
            bool(recognition is not None and recognition.resolved),
            {
                **common,
                "description": None if description is None else description.utterance,
                "raw_trace_input": True,
                "independent_source_trace": True,
                "recognized_expected_referent": bool(
                    recognition is not None and recognition.referent == target
                ),
                "fresh_world_reexecution": bool(described_behavior),
            },
        ),
        "factorial_composition": (
            factorial,
            parsed.resolved,
            {
                **common,
                "direct_composite_examples": 0,
                "executed_target_swap": external_execution.resolved,
                "executed_scheme_swap": moving_execution.resolved,
            },
        ),
        "lexicon_permutation_equivariance": (
            permutation,
            remapped_parse.resolved,
            {
                **common,
                "remapped_behavior_digest": remapped_digest,
                "total_surface_permutation": True,
            },
        ),
        "proof_grounded_symbolic_theft": (
            symbolic,
            True,
            {
                **common,
                "definition_materialized_referent": materialized.resolved,
                "derived_behavior": derived_behavior,
                "cycle_status": cycle.status.value,
                "definition_proof": manifest_hash(materialized.proof.to_dict()),
                "deleted_leaf_unknown": not definition_leaf_deletion_leaked,
                "swapped_leaf_changed_behavior": not definition_leaf_swap_leaked,
            },
        ),
    }
    return ProcessLanguageReport(
        MappingProxyType(results),
        lookup_leaked,
        target_only_prediction,
        definition_leaf_deletion_leaked,
        definition_leaf_swap_leaked,
        language_fresh_unknown,
    )


__all__ = [
    "FullUtteranceLookup",
    "PROCESS_LANGUAGE_VERSION",
    "ProcessLanguageCase",
    "ProcessLanguageReport",
    "build_process_language_case",
    "run_process_language_block",
]
