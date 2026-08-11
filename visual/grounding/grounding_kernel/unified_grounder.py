"""One persistent learner for causal perception, language, and action.

The evaluator supplies only :mod:`grounding_kernel.v1_contracts` values.  This
candidate induces three operational kinds from public evidence instead of
receiving parts of speech or semantic enums:

* a token with both positive and negative sensorimotor support is an
  interventional predicate;
* a positively supported, ostensively cued token denotes the predicate value
  exhibited by the cued target;
* a positively supported uncued token denotes the observed motor scheme.

Two-token demonstrations then induce surface order compositionally.  Empty,
zero-feedback demonstrations are dictionary statements: ``(head, leaf)`` is
an atomic definition and ``(head, left, right)`` is a conjunction.  They add
no grounding evidence; every executable leaf must already be grounded.

The learned state is frozen once.  Query controllers are deliberately
transient: they may remember the preceding observation while executing a
fixed policy, but cannot alter the learned checkpoint commitment.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .certificates import manifest_hash
from .closed_loop_programs import (
    ClosedLoopProgramRecognizer,
    PerceptualTargetRole,
    build_closed_loop_referent,
)
from .composition import And, Atom, Expression
from .contracts import Action, Observation
from .episode_binder import EpisodeConceptBinder
from .language import Demonstration, GroundedLanguageLearner, GroundedReferent
from .perceptual_policy import VisualTargetSelector
from .programs import ActionScheme, ProgramSchema
from .v1_adapters import BinderSupportRecord
from .v1_contracts import (
    ActionDecision,
    BeliefDecision,
    DescriptionDecision,
    ExperimentDecision,
    PublicTrace,
    PublicTransition,
    PublicTurn,
    SessionManifest,
    SessionPhase,
    Utterance,
)
from .v1_grade3_contracts import (
    CausalSupportRecord,
    Grade3SessionManifest,
    MotorDecision,
    MotorDirective,
    MotorPhase,
    MotorQuery,
    OstensiveSupportRecord,
    ProbeDecision,
    ProbeEvidence,
    ProbeOffer,
    ProbeResult,
    TraceBeliefQuery,
    TraceDescriptionQuery,
)


UNIFIED_GROUNDER_VERSION = "persistent-operational-grounder/1"


def _trace_digest(trace: PublicTrace) -> str:
    return manifest_hash(
        {
            "initial": trace.initial.digest(),
            "steps": [
                {
                    "before": transition.before.digest(),
                    "action": {
                        "code": transition.action.code,
                        "target": list(transition.action.target),
                        "vector": list(transition.action.vector),
                    },
                    "after": transition.after.digest(),
                }
                for transition in trace.transitions
            ],
        }
    )


def _observable_signature(trace: PublicTrace) -> str:
    """Renderer-tolerant signature built only from public interventions/RGB."""

    steps: list[dict[str, object]] = []
    for transition in trace.transitions:
        before = np.asarray(transition.before.pixels, dtype=np.int16)
        after = np.asarray(transition.after.pixels, dtype=np.int16)
        changed = np.any(before != after, axis=2)
        count = int(np.count_nonzero(changed))
        if count:
            ys, xs = np.nonzero(changed)
            geometry = (
                count,
                int(np.max(xs) - np.min(xs) + 1),
                int(np.max(ys) - np.min(ys) + 1),
                int(round(float(np.mean(xs)) - transition.action.target[0])),
                int(round(float(np.mean(ys)) - transition.action.target[1])),
            )
        else:
            geometry = (0, 0, 0, 0, 0)
        steps.append(
            {
                "code": transition.action.code,
                "vector": list(transition.action.vector),
                "tick": transition.after.tick - transition.before.tick,
                "geometry": list(geometry),
            }
        )
    return manifest_hash({"steps": steps})


def _scheme(trace: PublicTrace) -> ActionScheme:
    if not trace.transitions:
        raise ValueError("motor demonstrations cannot be empty")
    return ActionScheme(
        tuple(
            (transition.action.code, tuple(transition.action.vector))
            for transition in trace.transitions
        )
    )


def _cue_contains_target(turn: PublicTurn, trace: PublicTrace) -> bool:
    cue = turn.ostensive_pixel_cue
    if cue is None or not trace.transitions:
        return False
    x0, y0, x1, y1 = cue
    x, y = trace.transitions[0].action.target
    return x0 <= x < x1 and y0 <= y < y1


@dataclass(frozen=True, slots=True)
class _Support:
    turn: PublicTurn
    trace: PublicTrace
    digest: str


@dataclass(slots=True)
class _ProbeController:
    role: PerceptualTargetRole
    candidates: tuple[tuple[int, int], ...]
    candidate_index: int = 0
    scheme_index: int = 0
    current_target: tuple[int, int] | None = None
    initial: Observation | None = None
    transitions: list[PublicTransition] = field(default_factory=list)
    predictions: list[bool | None] = field(default_factory=list)
    tracked_targets: list[tuple[int, int]] = field(default_factory=list)
    last_action: Action | None = None

    def __post_init__(self) -> None:
        self.tracked_targets = list(self.candidates)

    @property
    def complete(self) -> bool:
        return self.candidate_index >= len(self.candidates)

    def next_action(self, observation: Observation) -> Action | None:
        if self.complete:
            return None
        if self.current_target is None:
            self.current_target = self.tracked_targets[self.candidate_index]
            self.initial = observation
            self.transitions = []
            self.scheme_index = 0
        code, vector = self.role.diagnostic_scheme.steps[self.scheme_index]
        action = Action(code, self.current_target, vector)
        self.last_action = action
        return action

    def observe(self, transition: PublicTransition) -> None:
        if self.last_action is None or transition.action != self.last_action:
            raise ValueError("probe transition does not match the selected action")
        self.transitions.append(transition.feedback_stripped())
        assert self.current_target is not None
        try:
            self.current_target = self.role.selector.track(transition.after, self.current_target)
        except LookupError:
            pass
        self.tracked_targets[self.candidate_index] = self.current_target
        self.scheme_index += 1
        self.last_action = None
        if self.scheme_index < len(self.role.diagnostic_scheme.steps):
            return
        assert self.initial is not None
        trace = PublicTrace(self.initial, tuple(self.transitions))
        prediction = self.role.binder.supports_token(trace, self.role.token)
        if prediction not in (True, False, None):
            raise TypeError("predicate decisions must be Boolean or UNKNOWN")
        self.predictions.append(prediction)
        self.candidate_index += 1
        self.current_target = None
        self.initial = None
        self.transitions = []
        self.scheme_index = 0

    def selected_rank(self) -> int | None:
        matches = [
            index
            for index, prediction in enumerate(self.predictions)
            if prediction is self.role.required_membership
        ]
        return matches[0] if len(matches) == 1 else None


class PersistentOperationalGrounder:
    """Reference candidate whose complete learned state survives every axis."""

    def __init__(self) -> None:
        self._manifest: SessionManifest | Grade3SessionManifest | None = None
        self._support: list[_Support] = []
        self._frozen = False
        self._checkpoint: str | None = None
        self._binder: EpisodeConceptBinder | None = None
        self._language: GroundedLanguageLearner | None = None
        self._schema: ProgramSchema | None = None
        self._roles: dict[int, PerceptualTargetRole] = {}
        self._schemes: dict[int, ActionScheme] = {}
        self._definitions: dict[int, Expression] = {}
        self._concept_tokens: tuple[int, ...] = ()
        self._acquisition: _ProbeController | None = None
        self._acquisition_role_token: int | None = None
        self._acquisition_evidence: list[str] = []
        self._acquisition_result: tuple[int, tuple[bool | None, ...]] | None = None
        self._scope_ids: set[int] = set()
        self._causal_support: list[CausalSupportRecord] = []
        self._probe_results: list[ProbeResult] = []
        self._last_probe_offer: ProbeOffer | None = None

    def begin(self, manifest: SessionManifest | Grade3SessionManifest) -> None:
        if self._manifest is not None:
            raise RuntimeError("the persistent grounder has already begun")
        if not isinstance(manifest, (SessionManifest, Grade3SessionManifest)):
            raise TypeError("manifest must be a supported session manifest")
        self._manifest = manifest

    def observe_support(
        self,
        turn: PublicTurn | OstensiveSupportRecord | CausalSupportRecord,
        trace: PublicTrace | None = None,
    ) -> None:
        if self._manifest is None or self._frozen:
            raise RuntimeError("support is allowed only before freeze")
        if isinstance(turn, CausalSupportRecord):
            if trace is not None:
                raise TypeError("causal support is a single record")
            self._scope_ids.add(turn.scope_id)
            self._causal_support.append(turn)
            return
        if isinstance(turn, OstensiveSupportRecord):
            if trace is not None:
                raise TypeError("ostensive support is a single record")
            self._scope_ids.add(turn.scope_id)
            trace = turn.trace
            turn = turn.turn
        if not isinstance(turn, PublicTurn) or not isinstance(trace, PublicTrace):
            raise TypeError("support requires PublicTurn and PublicTrace")
        if turn.phase is not SessionPhase.SUPPORT:
            raise ValueError("support turn has the wrong phase")
        if turn.observation != trace.initial:
            raise ValueError("support turn and trace must share their initial frame")
        if trace.has_feedback:
            raise ValueError("support supervision belongs on the turn, not the trace")
        if turn.utterance is None:
            raise ValueError("grounding support requires an opaque utterance")
        self._support.append(_Support(turn, trace, _trace_digest(trace)))

    @staticmethod
    def _unary(records: list[_Support]) -> dict[int, list[_Support]]:
        grouped: dict[int, list[_Support]] = defaultdict(list)
        for record in records:
            utterance = record.turn.utterance
            assert utterance is not None
            if len(utterance.tokens) == 1:
                grouped[utterance.tokens[0]].append(record)
        return grouped

    def _induce(self) -> None:
        unary = self._unary(self._support)
        concept_tokens = tuple(
            sorted(
                token
                for token, records in unary.items()
                if {
                    bool(record.turn.scalar_feedback and record.turn.scalar_feedback > 0.0)
                    for record in records
                    if record.turn.scalar_feedback is not None
                    and record.turn.scalar_feedback != 0.0
                }
                == {False, True}
                and all(record.turn.ostensive_pixel_cue is None for record in records)
                and all(record.trace.transitions for record in records)
            )
        )
        if not concept_tokens:
            raise ValueError("no contrastively grounded interventional predicate")
        binder_records: list[BinderSupportRecord] = []
        for token in concept_tokens:
            for record in unary[token]:
                feedback = record.turn.scalar_feedback
                if feedback is None or feedback == 0.0:
                    continue
                binder_records.append(
                    BinderSupportRecord(token, record.trace.feedback_stripped(), feedback)
                )
        binder = EpisodeConceptBinder(mode="full").fit(binder_records)

        role_support: dict[int, list[_Support]] = {
            token: [
                record
                for record in records
                if record.turn.ostensive_pixel_cue is not None
                and record.turn.scalar_feedback is not None
                and record.turn.scalar_feedback > 0.0
                and record.trace.transitions
            ]
            for token, records in unary.items()
            if token not in concept_tokens
        }
        role_support = {token: values for token, values in role_support.items() if values}
        if len(role_support) < 2:
            raise ValueError("at least two ostensively contrasted target roles are required")
        for records in role_support.values():
            if not all(_cue_contains_target(record.turn, record.trace) for record in records):
                raise ValueError("an ostensive role cue does not contain its public target")
        selector = VisualTargetSelector.from_traces(
            record.trace for records in role_support.values() for record in records
        )

        role_specs: dict[int, tuple[int, bool, ActionScheme, tuple[str, ...]]] = {}
        for surface_token, records in role_support.items():
            assignments: set[tuple[int, bool, ActionScheme]] = set()
            for record in records:
                # Ostensive role records are deliberately paired with a
                # separately corrected predicate record over the *same raw
                # intervention trace*.  Use that public evidence identity to
                # choose the predicate axis; asking every binary binder and
                # selecting an arbitrary False would conflate unrelated
                # predicates in a multi-concept learner.
                paired = {
                    (token, bool(candidate.turn.scalar_feedback > 0.0))
                    for token in concept_tokens
                    for candidate in unary[token]
                    if candidate.digest == record.digest
                    and candidate.turn.scalar_feedback is not None
                    and candidate.turn.scalar_feedback != 0.0
                }
                if len(paired) != 1:
                    raise ValueError(
                        "role support must pair with exactly one corrected "
                        "operational predicate trace"
                    )
                token, decision = paired.pop()
                assignments.add((token, bool(decision), _scheme(record.trace)))
            if len(assignments) != 1:
                raise ValueError("a role token has inconsistent operational support")
            concept, required, diagnostic = assignments.pop()
            role_specs[surface_token] = (
                concept,
                required,
                diagnostic,
                tuple(sorted(record.digest for record in records)),
            )

        schemes: dict[int, ActionScheme] = {}
        for token, records in unary.items():
            if token in concept_tokens or token in role_support:
                continue
            eligible = [
                record
                for record in records
                if record.turn.ostensive_pixel_cue is None
                and record.turn.scalar_feedback is not None
                and record.turn.scalar_feedback > 0.0
                and record.trace.transitions
            ]
            if not eligible:
                continue
            values = {_scheme(record.trace) for record in eligible}
            if len(values) != 1:
                raise ValueError("a motor token denotes inconsistent action schemes")
            schemes[token] = values.pop()
        if len(schemes) < 2:
            raise ValueError("at least two grounded motor schemes are required")

        schema = ProgramSchema(
            (UNIFIED_GROUNDER_VERSION, "induced-slot", 0),
            (UNIFIED_GROUNDER_VERSION, "induced-slot", 1),
        )
        roles = {
            surface: PerceptualTargetRole.from_support(
                selector=selector,
                binder=binder,
                token=concept,
                diagnostic_scheme=diagnostic,
                required_membership=required,
                evidence_digests=evidence,
            )
            for surface, (concept, required, diagnostic, evidence) in role_specs.items()
        }

        demonstrations: list[Demonstration] = []
        for record in self._support:
            utterance = record.turn.utterance
            assert utterance is not None
            if (
                len(utterance.tokens) != 2
                or record.turn.scalar_feedback is None
                or record.turn.scalar_feedback <= 0.0
                or not record.trace.transitions
            ):
                continue
            role_tokens = [token for token in utterance.tokens if token in roles]
            scheme_tokens = [token for token in utterance.tokens if token in schemes]
            if len(role_tokens) != 1 or len(scheme_tokens) != 1:
                continue
            referent = build_closed_loop_referent(
                schema, roles[role_tokens[0]], schemes[scheme_tokens[0]]
            )
            demonstrations.append(
                Demonstration(
                    utterance.tokens,
                    referent,
                    evidence=("public-support", record.digest),
                )
            )
        if len(demonstrations) < 3:
            raise ValueError("factorial language induction requires three grounded pairs")
        language = GroundedLanguageLearner().fit(demonstrations)
        if not language.order_templates:
            raise ValueError("surface order is not identifiable from support")

        definitions: dict[int, Expression] = {}
        for record in self._support:
            utterance = record.turn.utterance
            assert utterance is not None
            if (
                record.trace.transitions
                or record.turn.scalar_feedback != 0.0
                or len(utterance.tokens) not in (2, 3)
            ):
                continue
            head, *body = utterance.tokens
            expression: Expression = (
                Atom(body[0]) if len(body) == 1 else And(Atom(body[0]), Atom(body[1]))
            )
            if head in definitions and definitions[head] != expression:
                raise ValueError("a definition head has conflicting bodies")
            definitions[head] = expression
        if definitions:
            language.add_definitions(definitions)

        self._concept_tokens = concept_tokens
        self._binder = binder
        self._roles = roles
        self._schemes = schemes
        self._schema = schema
        self._language = language
        self._definitions = definitions

    def _role_from_utterance(
        self, utterance: Utterance | None
    ) -> tuple[int, PerceptualTargetRole] | None:
        if utterance is None:
            return None
        matches = [
            (token, self._roles[token]) for token in utterance.tokens if token in self._roles
        ]
        return matches[0] if len(matches) == 1 else None

    def choose_experiment(self, turn: PublicTurn) -> ExperimentDecision:
        if self._frozen:
            raise RuntimeError("acquisition requires an unfrozen learner")
        if self._language is None:
            self._induce()
        if turn.phase is not SessionPhase.ACQUISITION or turn.scalar_feedback is not None:
            raise ValueError("acquisition turns must be feedback-free")
        if self._acquisition is None:
            match = self._role_from_utterance(turn.utterance)
            if match is None:
                return ExperimentDecision(None, 1.0)
            surface_token, role = match
            candidates = role.selector.candidates(turn.observation)
            if not candidates:
                return ExperimentDecision(None, 1.0)
            self._acquisition_role_token = surface_token
            self._acquisition = _ProbeController(role, candidates)
        action = self._acquisition.next_action(turn.observation)
        if action is not None:
            return ExperimentDecision(action, 0.0)
        selected = self._acquisition.selected_rank()
        if selected is None:
            return ExperimentDecision(None, 1.0)
        self._acquisition_result = (
            selected,
            tuple(self._acquisition.predictions),
        )
        return ExperimentDecision(None, 0.0)

    def observe_experiment(
        self,
        turn: PublicTurn,
        transition: PublicTransition,
    ) -> None:
        if self._acquisition is None:
            raise RuntimeError("no acquisition decision is awaiting evidence")
        if transition.scalar_feedback is not None or turn.scalar_feedback is not None:
            raise ValueError("acquisition evidence must be feedback-free")
        self._acquisition.observe(transition)
        self._acquisition_evidence.append(
            _trace_digest(PublicTrace(transition.before, (transition,)))
        )

    def choose_probe(self, offer: ProbeOffer) -> ProbeDecision:
        """Choose the highest-partition public probe per unit cost."""

        if self._frozen:
            raise RuntimeError("active acquisition is closed after freeze")
        if not isinstance(offer, ProbeOffer):
            raise TypeError("offer must be ProbeOffer")
        rows = [
            record
            for record in self._causal_support
            if record.scope_id == offer.scope_id and record.problem_id == offer.problem_id
        ]
        if not rows:
            return ProbeDecision(None, 1.0)
        hypotheses = tuple(sorted({record.hypothesis_id for record in rows}))
        scored: list[tuple[float, int]] = []
        for option in offer.options:
            if option.cost > offer.remaining_cost + 1e-12:
                continue
            partitions = {
                hypothesis: tuple(
                    sorted(
                        {
                            _observable_signature(record.trace)
                            for record in rows
                            if record.hypothesis_id == hypothesis
                            and record.probe_id == option.probe_id
                        }
                    )
                )
                for hypothesis in hypotheses
            }
            if any(not values for values in partitions.values()):
                continue
            score = len(set(partitions.values())) / option.cost
            scored.append((score, option.probe_id))
        if not scored:
            return ProbeDecision(None, 1.0)
        _score, probe_id = max(scored, key=lambda item: (item[0], -item[1]))
        self._last_probe_offer = offer
        return ProbeDecision(probe_id, 0.0)

    def observe_probe(self, result: ProbeResult) -> None:
        if self._frozen:
            raise RuntimeError("active acquisition is closed after freeze")
        if not isinstance(result, ProbeResult):
            raise TypeError("result must be ProbeResult")
        offer = self._last_probe_offer
        if (
            offer is None
            or result.scope_id != offer.scope_id
            or result.problem_id != offer.problem_id
            or result.probe_id not in {option.probe_id for option in offer.options}
        ):
            raise ValueError("probe result does not match the authoritative offer")
        expected_cost = next(
            option.cost for option in offer.options if option.probe_id == result.probe_id
        )
        if abs(expected_cost - result.cost) > 1e-12:
            raise ValueError("probe result cost differs from the offered cost")
        self._probe_results.append(result)
        self._last_probe_offer = None

    def freeze(self) -> None:
        if self._manifest is None or self._frozen:
            raise RuntimeError("freeze requires one begun, unfrozen session")
        if self._language is None:
            self._induce()
        if self._acquisition is not None and not self._acquisition.complete:
            raise RuntimeError("cannot freeze an unfinished active probe")
        if self._last_probe_offer is not None:
            raise RuntimeError("cannot freeze with an unobserved probe decision")
        assert self._binder is not None
        assert self._language is not None
        material = {
            "version": UNIFIED_GROUNDER_VERSION,
            "manifest": {
                "protocol": self._manifest.protocol_version,
                "sensor": self._manifest.sensor_schema,
                "action": self._manifest.action_schema,
            },
            "support": [record.digest for record in self._support],
            "binder": self._binder.manifest.digest(),
            "concept_tokens": list(self._concept_tokens),
            "roles": [
                {
                    "surface": surface,
                    "predicate": role.token,
                    "required": role.required_membership,
                    "support": role.support_commitment,
                }
                for surface, role in sorted(self._roles.items())
            ],
            "schemes": [
                {"surface": token, "steps": list(scheme.steps)}
                for token, scheme in sorted(self._schemes.items())
            ],
            "language": self._language.ledger.digest,
            "definitions": [
                (token, repr(expression)) for token, expression in sorted(self._definitions.items())
            ],
            "acquisition": self._acquisition_result,
            "acquisition_evidence": list(self._acquisition_evidence),
            "scopes": sorted(self._scope_ids),
            "causal_support": [
                {
                    "scope": record.scope_id,
                    "problem": record.problem_id,
                    "hypothesis": record.hypothesis_id,
                    "probe": record.probe_id,
                    "source": record.source_id,
                    "trace": _trace_digest(record.trace),
                }
                for record in self._causal_support
            ],
            "probe_results": [
                {
                    "scope": result.scope_id,
                    "problem": result.problem_id,
                    "probe": result.probe_id,
                    "trace": _trace_digest(result.trace),
                    "cost": result.cost,
                }
                for result in self._probe_results
            ],
        }
        self._checkpoint = manifest_hash(material)
        self._frozen = True

    def checkpoint_commitment(self) -> str:
        if not self._frozen or self._checkpoint is None:
            raise RuntimeError("checkpoint is unavailable before freeze")
        return self._checkpoint

    def _resolve(self, utterance: Utterance) -> GroundedReferent | None:
        assert self._language is not None
        if len(utterance.tokens) == 1 and utterance.tokens[0] in self._definitions:
            materialized = self._language.materialize_definition(utterance.tokens[0])
            return materialized.referent if materialized.resolved else None
        interpreted = self._language.interpret_instruction(utterance.tokens)
        return interpreted.referent if interpreted.resolved else None

    def begin_goal(self, utterance: Utterance, observation: Observation) -> None:
        """Legacy stateful surface; the Grade-3 candidate intentionally abstains.

        Use :meth:`motor`, whose complete transcript is evaluator-owned and
        carried on every call.  Keeping this method side-effect free preserves
        the full post-freeze state invariant for legacy runners as well.
        """

        if not self._frozen:
            raise RuntimeError("goals are sealed queries")
        if not isinstance(utterance, Utterance) or not isinstance(observation, Observation):
            raise TypeError("legacy goals require Utterance and Observation")

    def act(self, observation: Observation) -> ActionDecision:
        if not isinstance(observation, Observation):
            raise TypeError("observation must be Observation")
        return ActionDecision(None, 1.0)

    def describe(
        self,
        trace: PublicTrace | Sequence[ProbeEvidence] | TraceDescriptionQuery,
    ) -> DescriptionDecision:
        if isinstance(trace, TraceDescriptionQuery):
            trace = trace.evidence
        if not isinstance(trace, PublicTrace):
            evidence = tuple(trace)
            if len(evidence) != 1 or not isinstance(evidence[0], ProbeEvidence):
                return DescriptionDecision(None, 1.0)
            trace = evidence[0].trace
        if not self._frozen or trace.has_feedback:
            return DescriptionDecision(None, 1.0)
        assert self._schema is not None
        assert self._language is not None
        recognition = ClosedLoopProgramRecognizer(
            self._schema,
            tuple(self._roles.values()),
            tuple(self._schemes.values()),
        ).recognize(trace.feedback_stripped())
        if not recognition.resolved or recognition.referent is None:
            return DescriptionDecision(None, 1.0)
        description = self._language.describe(recognition.referent)
        if not description.resolved or description.utterance is None:
            return DescriptionDecision(None, 1.0)
        return DescriptionDecision(
            Utterance(tuple(int(token) for token in description.utterance)), 0.0
        )

    def report_belief(self, candidates: Any) -> BeliefDecision:
        tuple(candidates)
        return BeliefDecision((), 1.0)

    def _motor_referent(
        self, query: MotorQuery
    ) -> tuple[PerceptualTargetRole, ActionScheme] | None:
        assert self._schema is not None
        referent = self._resolve(query.utterance)
        if referent is None:
            return None
        role_meaning = referent.meaning_for(self._schema.target_type_id)
        scheme_meaning = referent.meaning_for(self._schema.scheme_type_id)
        if (
            role_meaning is None
            or scheme_meaning is None
            or not isinstance(role_meaning.value, PerceptualTargetRole)
            or not isinstance(scheme_meaning.value, ActionScheme)
        ):
            return None
        return role_meaning.value, scheme_meaning.value

    @staticmethod
    def _tracked_target(
        role: PerceptualTargetRole,
        trace: PublicTrace,
        initial_target: tuple[int, int],
        expected: ActionScheme,
    ) -> tuple[int, int] | None:
        if len(trace.transitions) > len(expected.steps):
            return None
        target = initial_target
        for index, transition in enumerate(trace.transitions):
            code, vector = expected.steps[index]
            if (
                transition.action.code != code
                or tuple(transition.action.vector) != vector
                or transition.action.target != target
            ):
                return None
            try:
                target = role.selector.track(transition.after, target)
            except LookupError:
                pass
        return target

    def motor(self, query: MotorQuery) -> MotorDecision:
        """Compute a motor directive solely from the frozen state and query."""

        if not self._frozen or not isinstance(query, MotorQuery):
            return MotorDecision(MotorDirective.ABSTAIN, None, 1.0)
        if self._scope_ids and query.scope_id not in self._scope_ids:
            return MotorDecision(MotorDirective.ABSTAIN, None, 1.0)
        resolved = self._motor_referent(query)
        if resolved is None:
            return MotorDecision(MotorDirective.ABSTAIN, None, 1.0)
        role, goal_scheme = resolved
        candidates = role.selector.candidates(query.current_trace.initial)
        if not candidates:
            return MotorDecision(MotorDirective.ABSTAIN, None, 1.0)

        if query.phase is MotorPhase.PROBE:
            rank = len(query.completed_probes)
            if rank >= len(candidates):
                return MotorDecision(MotorDirective.ABSTAIN, None, 1.0)
            target = self._tracked_target(
                role,
                query.current_trace,
                candidates[rank],
                role.diagnostic_scheme,
            )
            if target is None:
                return MotorDecision(MotorDirective.ABSTAIN, None, 1.0)
            index = len(query.current_trace.transitions)
            if index < len(role.diagnostic_scheme.steps):
                code, vector = role.diagnostic_scheme.steps[index]
                action = Action(code, target, vector)
                if query.remaining_action_cost <= 0.0 or not query.action_space.permits(
                    action, query.current_trace.current.shape
                ):
                    return MotorDecision(MotorDirective.ABSTAIN, None, 1.0)
                return MotorDecision(
                    MotorDirective.ACT,
                    action,
                    0.0,
                )
            completed = len(query.completed_probes) + 1
            directive = (
                MotorDirective.RESET_EXECUTE
                if completed == len(candidates)
                else MotorDirective.RESET_PROBE
            )
            if query.remaining_resets <= 0:
                return MotorDecision(MotorDirective.ABSTAIN, None, 1.0)
            return MotorDecision(directive, None, 0.0)

        if len(query.completed_probes) != len(candidates):
            return MotorDecision(MotorDirective.ABSTAIN, None, 1.0)
        predictions = tuple(
            role.binder.supports_token(trace, role.token) for trace in query.completed_probes
        )
        selected = tuple(
            index
            for index, prediction in enumerate(predictions)
            if prediction is role.required_membership
        )
        if len(selected) != 1:
            return MotorDecision(MotorDirective.ABSTAIN, None, 1.0)
        target = self._tracked_target(
            role,
            query.current_trace,
            candidates[selected[0]],
            goal_scheme,
        )
        if target is None:
            return MotorDecision(MotorDirective.ABSTAIN, None, 1.0)
        index = len(query.current_trace.transitions)
        if index == len(goal_scheme.steps):
            return MotorDecision(MotorDirective.COMPLETE, None, 0.0)
        code, vector = goal_scheme.steps[index]
        action = Action(code, target, vector)
        if query.remaining_action_cost <= 0.0 or not query.action_space.permits(
            action, query.current_trace.current.shape
        ):
            return MotorDecision(MotorDirective.ABSTAIN, None, 1.0)
        return MotorDecision(MotorDirective.ACT, action, 0.0)

    def trace_belief(self, query: TraceBeliefQuery) -> BeliefDecision:
        """Infer an opaque hypothesis or motor word from explicit evidence."""

        if not self._frozen or not isinstance(query, TraceBeliefQuery):
            return BeliefDecision((), 1.0)
        causal_rows = [
            record
            for record in self._causal_support
            if record.scope_id == query.scope_id
            and record.problem_id == query.problem_id
            and record.hypothesis_id in query.candidates
        ]
        if causal_rows:
            # When this persistent learner actively gathered evidence before
            # freeze, a later belief report must be bound to that exact
            # probe-id/consequence ledger.  Otherwise a caller could permute
            # intervention labels after the fact and obtain a confidently
            # wrong but superficially well-formed hypothesis.
            acquired = [
                result
                for result in self._probe_results
                if result.scope_id == query.scope_id and result.problem_id == query.problem_id
            ]
            if acquired:
                acquired_rows = {
                    (result.probe_id, _observable_signature(result.trace)) for result in acquired
                }
                query_rows = {
                    (item.probe_id, _observable_signature(item.trace)) for item in query.evidence
                }
                if query_rows != acquired_rows:
                    return BeliefDecision((), 1.0)
            compatible: list[int] = []
            for hypothesis in query.candidates:
                rows = [record for record in causal_rows if record.hypothesis_id == hypothesis]
                if rows and all(
                    _observable_signature(item.trace)
                    in {
                        _observable_signature(record.trace)
                        for record in rows
                        if record.probe_id == item.probe_id
                    }
                    for item in query.evidence
                ):
                    compatible.append(hypothesis)
            if len(compatible) == 1:
                return BeliefDecision(((compatible[0], 1.0),), 0.0)
            return BeliefDecision((), 1.0)

        assert self._schema is not None
        assert self._language is not None
        descriptions: list[tuple[int, ...]] = []
        recognizer = ClosedLoopProgramRecognizer(
            self._schema, tuple(self._roles.values()), tuple(self._schemes.values())
        )
        for item in query.evidence:
            recognition = recognizer.recognize(item.trace.feedback_stripped())
            if not recognition.resolved or recognition.referent is None:
                return BeliefDecision((), 1.0)
            description = self._language.describe(recognition.referent)
            if not description.resolved or description.utterance is None:
                return BeliefDecision((), 1.0)
            descriptions.append(tuple(int(token) for token in description.utterance))
        scheme_tokens = {
            token
            for description in descriptions
            for token in description
            if token in self._schemes and token in query.candidates
        }
        if len(scheme_tokens) != 1:
            return BeliefDecision((), 1.0)
        token = next(iter(scheme_tokens))
        return BeliefDecision(((token, 1.0),), 0.0)

    # Transitional alias for early laboratory callers; Grade-3 RPC exposes
    # only ``trace_belief``.
    belief = trace_belief


def build() -> PersistentOperationalGrounder:
    """Artifact entrypoint used by :class:`IsolatedGrounder`."""

    return PersistentOperationalGrounder()


__all__ = [
    "PersistentOperationalGrounder",
    "UNIFIED_GROUNDER_VERSION",
    "build",
]
