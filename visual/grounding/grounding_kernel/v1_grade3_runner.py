"""Persistent evaluator orchestration for the sealed Grade-3 protocol.

One isolated candidate is begun once, trained through support and optional
causal probes, frozen once, and then queried without hidden candidate state.
The evaluator owns every reset, primitive world step, raw public trace, and
resource reservation.
"""

from __future__ import annotations

from collections.abc import Callable
from contextlib import suppress
from dataclasses import dataclass
from math import isclose, isfinite
from threading import RLock
from types import TracebackType

from .certificates import manifest_hash
from .contracts import Action, Observation, Transition
from .v1_contracts import (
    BeliefDecision,
    DescriptionDecision,
    PublicTrace,
    PublicTransition,
    Utterance,
)
from .v1_grade3_contracts import (
    MAX_GRADE3_PROBES,
    MAX_GRADE3_TRANSITIONS,
    Grade3SessionManifest,
    Grade3SupportRecord,
    MotorActionSpace,
    MotorDecision,
    MotorDirective,
    MotorPhase,
    MotorQuery,
    ProbeDecision,
    ProbeOffer,
    ProbeResult,
    TraceBeliefQuery,
    TraceDescriptionQuery,
)
from .v1_grade3_isolation import (
    FrozenGrade3Candidate,
    Grade3ArtifactCommitment,
    IsolatedGrade3Grounder,
)
from .v1_grade3_session import (
    Grade3AuditLedger,
    Grade3EvaluationSession,
    Grade3Resource,
    Grade3SessionPhase,
    Grade3SessionStateError,
)
from .v1_grade3_wire import encode_grade3_message


CodebookFactory = Callable[[], str]
ProbeExecutor = Callable[[int], PublicTrace]
MotorReset = Callable[[], Observation]
MotorStep = Callable[[Action], PublicTransition | Transition]
MotorActionCost = Callable[[Action], float]


def _unit_action_cost(_action: Action) -> float:
    return 1.0


def _wire_hash(value: object) -> str:
    return manifest_hash({"wire": encode_grade3_message(value)})


def _failure_hash(error: BaseException) -> str:
    return manifest_hash(
        {
            "type": f"{type(error).__module__}.{type(error).__qualname__}",
            "message": str(error),
        }
    )


def _observation_hash(observation: Observation) -> str:
    return manifest_hash(
        {
            "digest": observation.digest(),
            "shape": observation.shape,
            "tick": observation.tick,
            "terminal": observation.terminal,
        }
    )


def _public_transition(value: PublicTransition | Transition) -> PublicTransition:
    if isinstance(value, PublicTransition):
        if value.scalar_feedback is not None:
            raise Grade3SessionStateError("motor transitions must be feedback-free")
        return value
    if not isinstance(value, Transition):
        raise TypeError("motor step must return PublicTransition or Transition")
    # The evaluator-private outcome code is intentionally not projected into
    # the Grade-3 evidence channel.
    return PublicTransition(value.before, value.action, value.after, None)


def _positive_cost(value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError("motor action cost must be a finite positive number")
    result = float(value)
    if not isfinite(result) or result <= 0.0:
        raise ValueError("motor action cost must be a finite positive number")
    return result


@dataclass(frozen=True, slots=True)
class ExecutedGrade3Probe:
    decision: ProbeDecision
    result: ProbeResult | None
    ledger_event_hash: str

    @property
    def abstained(self) -> bool:
        return self.decision.probe_id is None


@dataclass(frozen=True, slots=True)
class MotorTranscriptTurn:
    query: MotorQuery
    decision: MotorDecision
    transition: PublicTransition | None = None
    reset_observation_digest: str | None = None


@dataclass(frozen=True, slots=True)
class MotorEpisodeResult:
    query_id: int
    scope_id: int
    utterance: Utterance
    completed: bool
    abstained: bool
    unknown_probability: float
    completed_probes: tuple[PublicTrace, ...]
    execution_trace: PublicTrace | None
    transcript: tuple[MotorTranscriptTurn, ...]
    action_cost_consumed: float
    resets_consumed: int
    ledger_event_hash: str


class Grade3EvaluationRunner:
    """One ledger and one persistent :class:`IsolatedGrade3Grounder`."""

    def __init__(
        self,
        commitment: Grade3ArtifactCommitment,
        manifest: Grade3SessionManifest,
        codebook_factory: CodebookFactory,
        *,
        motor_action_cost: MotorActionCost = _unit_action_cost,
        timeout: float = 20.0,
    ) -> None:
        if not isinstance(commitment, Grade3ArtifactCommitment):
            raise TypeError("commitment must be Grade3ArtifactCommitment")
        if not isinstance(manifest, Grade3SessionManifest):
            raise TypeError("manifest must be Grade3SessionManifest")
        if not callable(codebook_factory):
            raise TypeError("codebook_factory must be callable")
        if not callable(motor_action_cost):
            raise TypeError("motor_action_cost must be callable")
        if isinstance(timeout, bool) or not isinstance(timeout, (int, float)):
            raise TypeError("timeout must be a finite positive number")
        if not isfinite(float(timeout)) or float(timeout) <= 0.0:
            raise ValueError("timeout must be a finite positive number")
        self.commitment = commitment
        self.session = Grade3EvaluationSession(
            manifest, commitment.digest, commitment.sdk_commitment
        )
        self.candidate = IsolatedGrade3Grounder(commitment, timeout=float(timeout))
        self._codebook_factory: CodebookFactory | None = codebook_factory
        self._motor_action_cost = motor_action_cost
        self._started = False
        self._frozen = False
        self._closed = False
        self._terminal_failure: str | None = None
        self._callback_active = False
        self._lock = RLock()

    def __enter__(self) -> "Grade3EvaluationRunner":
        return self.start()

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback_value: TracebackType | None,
    ) -> None:
        self.close()

    @property
    def audit(self) -> Grade3AuditLedger:
        return self.session.ledger

    @property
    def terminal_failure(self) -> str | None:
        return self._terminal_failure

    @property
    def remaining_acquisition_cost(self) -> float:
        return self.session.remaining_acquisition_cost

    @property
    def remaining_queries(self) -> int:
        return self.session.remaining_queries

    @property
    def remaining_motor_action_cost(self) -> float:
        return self.session.remaining_motor_action_cost

    @property
    def remaining_motor_resets(self) -> int:
        return self.session.remaining_motor_resets

    def _require_open(self) -> None:
        if self._callback_active:
            raise Grade3SessionStateError("runner callbacks cannot re-enter the runner")
        if not self._started or self._closed:
            raise Grade3SessionStateError("runner must be started and open")
        if self._terminal_failure is not None:
            raise Grade3SessionStateError("runner is terminally failed")

    def _invoke(self, callback: Callable[..., object], *args: object) -> object:
        self._callback_active = True
        try:
            return callback(*args)
        finally:
            self._callback_active = False

    def _poison(self, error: BaseException, operation: str) -> None:
        failure = _failure_hash(error)
        self._terminal_failure = failure
        self.session.record_operation(operation, "failed", result_hash=failure)
        with suppress(BaseException):
            self.candidate.close()
        self._closed = True

    def start(self) -> "Grade3EvaluationRunner":
        with self._lock:
            if self._closed or self._started:
                raise Grade3SessionStateError("runner can be started exactly once")
            try:
                # Artifact and recursive SDK commitments already exist.  The
                # codebook is deliberately materialized only after the child
                # has staged, rehashed, imported, and begun the candidate.
                self.candidate.start()
                self.candidate.begin(self.session.manifest)
                factory = self._codebook_factory
                if factory is None:  # pragma: no cover - lifecycle invariant
                    raise RuntimeError("codebook factory was already consumed")
                commitment = self._invoke(factory)
                if not isinstance(commitment, str):
                    raise TypeError("codebook_factory must return a SHA-256 digest")
                self.session.commit_codebook(commitment)
                self._codebook_factory = None
                self._started = True
                self.session.record_operation("start", "completed", result_hash=commitment)
                return self
            except BaseException as exc:
                self._codebook_factory = None
                self._terminal_failure = _failure_hash(exc)
                with suppress(BaseException):
                    self.candidate.close()
                self._closed = True
                # commit_codebook may not yet permit an event; record only if
                # its commitment succeeded before a later start failure.
                if self.session.commitments["codebook"] is not None:
                    self.session.record_operation(
                        "start", "failed", result_hash=self._terminal_failure
                    )
                raise

    def support(self, record: Grade3SupportRecord) -> None:
        with self._lock:
            self._require_open()
            if self.session.phase is not Grade3SessionPhase.SUPPORT:
                raise Grade3SessionStateError("support requires support phase")
            # Contract validation precedes metering; the candidate RPC does not.
            request_hash = _wire_hash(record)
            reservation = self.session.reserve(
                Grade3Resource.SUPPORT_RECORD, 1, "support", request_hash
            )
            try:
                self.candidate.observe_support(record)
            except BaseException as exc:
                self.session.consume(
                    reservation,
                    status="failed_consumed",
                    result_hash=_failure_hash(exc),
                )
                self._poison(exc, "support_terminal")
                raise
            self.session.consume(reservation, status="completed", result_hash=request_hash)

    def begin_acquisition(self) -> None:
        with self._lock:
            self._require_open()
            self.session.begin_acquisition()

    def acquire_probe(self, offer: ProbeOffer, execute: ProbeExecutor) -> ExecutedGrade3Probe:
        """Choose and execute one causal probe, charging before world access."""

        with self._lock:
            self._require_open()
            if self.session.phase is not Grade3SessionPhase.ACQUISITION:
                raise Grade3SessionStateError("causal probes require acquisition phase")
            if not isinstance(offer, ProbeOffer):
                raise TypeError("offer must be ProbeOffer")
            if not callable(execute):
                raise TypeError("execute must be callable")
            if not isclose(
                offer.remaining_cost,
                self.remaining_acquisition_cost,
                rel_tol=0.0,
                abs_tol=1e-9,
            ):
                raise Grade3SessionStateError("offer.remaining_cost is not evaluator-authoritative")
            request_hash = _wire_hash(offer)
            try:
                decision = self.candidate.choose_probe(offer)
            except BaseException as exc:
                self.session.record_operation(
                    "choose_probe",
                    "failed",
                    request_hash=request_hash,
                    result_hash=_failure_hash(exc),
                )
                self._poison(exc, "choose_probe_terminal")
                raise
            decision_hash = _wire_hash(decision)
            self.session.record_operation(
                "choose_probe",
                "completed" if decision.probe_id is not None else "abstained",
                request_hash=request_hash,
                result_hash=decision_hash,
            )
            if decision.probe_id is None:
                return ExecutedGrade3Probe(decision, None, self.audit.head_hash)
            option = next(
                (item for item in offer.options if item.probe_id == decision.probe_id),
                None,
            )
            if option is None:
                error = Grade3SessionStateError("candidate selected an unoffered probe")
                self._poison(error, "probe_selection_terminal")
                raise error
            try:
                reservation = self.session.reserve(
                    Grade3Resource.ACQUISITION_COST,
                    option.cost,
                    "causal_probe",
                    decision_hash,
                )
            except BaseException as exc:
                self._poison(exc, "probe_reservation_terminal")
                raise
            try:
                trace = self._invoke(execute, decision.probe_id)
                if not isinstance(trace, PublicTrace):
                    raise TypeError("probe executor must return PublicTrace")
                if not trace.transitions:
                    raise Grade3SessionStateError("causal probe traces must be nonempty")
                if trace.has_feedback:
                    raise Grade3SessionStateError("causal probe traces must be feedback-free")
                result = ProbeResult(
                    offer.scope_id,
                    offer.problem_id,
                    decision.probe_id,
                    trace,
                    option.cost,
                    self.remaining_acquisition_cost,
                )
                self.candidate.observe_probe(result)
            except BaseException as exc:
                self.session.consume(
                    reservation,
                    status="failed_consumed",
                    result_hash=_failure_hash(exc),
                )
                self._poison(exc, "causal_probe_terminal")
                raise
            event = self.session.consume(
                reservation,
                status="completed",
                result_hash=_wire_hash(result),
            )
            return ExecutedGrade3Probe(decision, result, event.event_hash)

    def freeze(self) -> FrozenGrade3Candidate:
        with self._lock:
            self._require_open()
            if self._frozen:
                raise Grade3SessionStateError("runner can freeze exactly once")
            try:
                frozen = self.candidate.freeze()
                self.session.freeze(frozen.checkpoint_commitment)
            except BaseException as exc:
                self._poison(exc, "freeze_terminal")
                raise
            self._frozen = True
            return frozen

    def _require_query_phase(self) -> None:
        self._require_open()
        if not self._frozen or self.session.phase is not Grade3SessionPhase.FROZEN_QUERY:
            raise Grade3SessionStateError("sealed query requires the frozen checkpoint")

    def _perform_reset(
        self,
        reset: MotorReset,
        *,
        query_id: int,
        phase: MotorPhase,
        step_index: int,
        expected_shape: tuple[int, int, int] | None,
    ) -> tuple[Observation, str]:
        request_hash = manifest_hash(
            {
                "query_id": query_id,
                "phase": phase.value,
                "step_index": step_index,
            }
        )
        reservation = self.session.reserve(
            Grade3Resource.MOTOR_RESET, 1, "motor_reset", request_hash
        )
        try:
            observation = self._invoke(reset)
            if not isinstance(observation, Observation):
                raise TypeError("motor reset must return Observation")
            if expected_shape is not None and observation.shape != expected_shape:
                raise Grade3SessionStateError("motor resets must preserve the declared RGB shape")
            if observation.terminal:
                raise Grade3SessionStateError("motor reset returned a terminal frame")
        except BaseException as exc:
            self.session.consume(
                reservation,
                status="failed_consumed",
                result_hash=_failure_hash(exc),
            )
            raise
        digest = _observation_hash(observation)
        self.session.consume(reservation, status="completed", result_hash=digest)
        return observation, digest

    def _perform_action(
        self,
        action: Action,
        trace: PublicTrace,
        action_space: MotorActionSpace,
        step: MotorStep,
    ) -> tuple[PublicTrace, PublicTransition, float]:
        if len(trace.transitions) >= action_space.max_trace_steps:
            raise Grade3SessionStateError("motor trace step limit is exhausted")
        if not action_space.permits(action, trace.current.shape):
            raise Grade3SessionStateError("candidate action lies outside action_space")
        cost = _positive_cost(self._invoke(self._motor_action_cost, action))
        request_hash = manifest_hash(
            {
                "action": action,
                "before": trace.current.digest(),
                "cost": cost,
            }
        )
        reservation = self.session.reserve(
            Grade3Resource.MOTOR_ACTION_COST,
            cost,
            "motor_action",
            request_hash,
        )
        try:
            raw = self._invoke(step, action)
            transition = _public_transition(raw)  # type: ignore[arg-type]
            if transition.action != action:
                raise Grade3SessionStateError(
                    "motor step returned a transition for a different action"
                )
            if transition.before != trace.current:
                raise Grade3SessionStateError(
                    "motor transition.before is not the evaluator current frame"
                )
            if not action_space.permits(transition.action, transition.before.shape):
                raise Grade3SessionStateError(
                    "motor transition violates the committed action space"
                )
            updated = trace.append(transition)
        except BaseException as exc:
            self.session.consume(
                reservation,
                status="failed_consumed",
                result_hash=_failure_hash(exc),
            )
            raise
        self.session.consume(
            reservation,
            status="completed",
            result_hash=manifest_hash({"transition": transition}),
        )
        return updated, transition, cost

    def run_motor_episode(
        self,
        query_id: int,
        scope_id: int,
        utterance: Utterance,
        action_space: MotorActionSpace,
        reset: MotorReset,
        step: MotorStep,
        *,
        max_decisions: int | None = None,
    ) -> MotorEpisodeResult:
        """Run one complete probe→clean-execution interaction as one query."""

        with self._lock:
            self._require_query_phase()
            if isinstance(query_id, bool) or not isinstance(query_id, int):
                raise TypeError("query_id must be an integer")
            if isinstance(scope_id, bool) or not isinstance(scope_id, int):
                raise TypeError("scope_id must be an integer")
            if query_id < 0 or scope_id < 0:
                raise ValueError("query_id and scope_id must be nonnegative")
            if not isinstance(utterance, Utterance):
                raise TypeError("utterance must be Utterance")
            if not isinstance(action_space, MotorActionSpace):
                raise TypeError("action_space must be MotorActionSpace")
            if not callable(reset) or not callable(step):
                raise TypeError("reset and step must be callable")
            if max_decisions is None:
                decision_limit = (
                    MAX_GRADE3_TRANSITIONS
                    + self.session.manifest.motor_reset_budget
                    + MAX_GRADE3_PROBES
                    + 2
                )
            else:
                if isinstance(max_decisions, bool) or not isinstance(max_decisions, int):
                    raise TypeError("max_decisions must be an integer")
                if max_decisions < 1:
                    raise ValueError("max_decisions must be positive")
                decision_limit = max_decisions
            episode_request_hash = manifest_hash(
                {
                    "query_id": query_id,
                    "scope_id": scope_id,
                    "utterance": utterance,
                    "action_space": action_space,
                    "max_decisions": decision_limit,
                }
            )
            query_reservation = self.session.reserve_query(
                query_id, "motor_episode", episode_request_hash
            )
            completed_probes: list[PublicTrace] = []
            transcript: list[MotorTranscriptTurn] = []
            action_cost_used = 0.0
            resets_used = 0
            current: PublicTrace | None = None
            phase = MotorPhase.PROBE
            try:
                initial, reset_digest = self._perform_reset(
                    reset,
                    query_id=query_id,
                    phase=phase,
                    step_index=0,
                    expected_shape=None,
                )
                resets_used += 1
                current = PublicTrace(initial)
                shape = initial.shape
                for step_index in range(decision_limit):
                    query = MotorQuery(
                        query_id,
                        scope_id,
                        step_index,
                        utterance,
                        phase,
                        tuple(completed_probes),
                        current,
                        action_space,
                        self.remaining_motor_action_cost,
                        self.remaining_motor_resets,
                    )
                    decision = self.candidate.motor(query)
                    query_hash = _wire_hash(query)
                    decision_hash = _wire_hash(decision)
                    self.session.record_operation(
                        "motor_decision",
                        "abstained"
                        if decision.directive is MotorDirective.ABSTAIN
                        else "completed",
                        request_hash=query_hash,
                        result_hash=decision_hash,
                    )
                    if current.current.terminal and decision.directive not in {
                        MotorDirective.COMPLETE,
                        MotorDirective.ABSTAIN,
                    }:
                        raise Grade3SessionStateError(
                            "a terminal frame permits only COMPLETE or ABSTAIN"
                        )
                    if decision.directive is MotorDirective.ACT:
                        updated, transition, action_cost = self._perform_action(
                            decision.action,  # type: ignore[arg-type]
                            current,
                            action_space,
                            step,
                        )
                        action_cost_used += action_cost
                        current = updated
                        transcript.append(MotorTranscriptTurn(query, decision, transition, None))
                        continue
                    if decision.directive in {
                        MotorDirective.RESET_PROBE,
                        MotorDirective.RESET_EXECUTE,
                    }:
                        if phase is not MotorPhase.PROBE:
                            raise Grade3SessionStateError(
                                "reset directives are legal only during probing"
                            )
                        if not current.transitions:
                            raise Grade3SessionStateError(
                                "a reset cannot archive an empty causal probe"
                            )
                        if len(completed_probes) >= MAX_GRADE3_PROBES:
                            raise Grade3SessionStateError(
                                "completed causal probe limit is exhausted"
                            )
                        completed_probes.append(current)
                        next_phase = (
                            MotorPhase.EXECUTE
                            if decision.directive is MotorDirective.RESET_EXECUTE
                            else MotorPhase.PROBE
                        )
                        observation, digest = self._perform_reset(
                            reset,
                            query_id=query_id,
                            phase=next_phase,
                            step_index=step_index,
                            expected_shape=shape,
                        )
                        resets_used += 1
                        current = PublicTrace(observation)
                        phase = next_phase
                        transcript.append(MotorTranscriptTurn(query, decision, None, digest))
                        continue
                    if decision.directive is MotorDirective.COMPLETE:
                        if phase is not MotorPhase.EXECUTE:
                            raise Grade3SessionStateError(
                                "COMPLETE is legal only during clean execution"
                            )
                        transcript.append(MotorTranscriptTurn(query, decision))
                        provisional = MotorEpisodeResult(
                            query_id,
                            scope_id,
                            utterance,
                            True,
                            False,
                            0.0,
                            tuple(completed_probes),
                            current,
                            tuple(transcript),
                            action_cost_used,
                            resets_used,
                            "",
                        )
                        result_hash = manifest_hash(provisional)
                        event = self.session.consume(
                            query_reservation,
                            status="completed",
                            result_hash=result_hash,
                        )
                        return MotorEpisodeResult(
                            query_id,
                            scope_id,
                            utterance,
                            True,
                            False,
                            0.0,
                            tuple(completed_probes),
                            current,
                            tuple(transcript),
                            action_cost_used,
                            resets_used,
                            event.event_hash,
                        )
                    if decision.directive is MotorDirective.ABSTAIN:
                        transcript.append(MotorTranscriptTurn(query, decision))
                        provisional = MotorEpisodeResult(
                            query_id,
                            scope_id,
                            utterance,
                            False,
                            True,
                            decision.unknown_probability,
                            tuple(completed_probes),
                            current if phase is MotorPhase.EXECUTE else None,
                            tuple(transcript),
                            action_cost_used,
                            resets_used,
                            "",
                        )
                        result_hash = manifest_hash(provisional)
                        event = self.session.consume(
                            query_reservation,
                            status="abstained",
                            result_hash=result_hash,
                        )
                        return MotorEpisodeResult(
                            query_id,
                            scope_id,
                            utterance,
                            False,
                            True,
                            decision.unknown_probability,
                            tuple(completed_probes),
                            current if phase is MotorPhase.EXECUTE else None,
                            tuple(transcript),
                            action_cost_used,
                            resets_used,
                            event.event_hash,
                        )
                    raise Grade3SessionStateError("unsupported motor directive")
                raise Grade3SessionStateError("motor decision limit exhausted")
            except BaseException as exc:
                # Action/reset helpers already consumed their own reservations.
                # The one logical sealed query is independently consumed here.
                self.session.consume(
                    query_reservation,
                    status="failed_consumed",
                    result_hash=_failure_hash(exc),
                )
                self._poison(exc, "motor_episode_terminal")
                raise

    def _sealed_query(
        self,
        query_id: int,
        operation: str,
        request: TraceBeliefQuery | TraceDescriptionQuery,
        call: Callable[[], BeliefDecision | DescriptionDecision],
    ) -> BeliefDecision | DescriptionDecision:
        self._require_query_phase()
        request_hash = _wire_hash(request)
        reservation = self.session.reserve_query(query_id, operation, request_hash)
        try:
            response = call()
        except BaseException as exc:
            self.session.consume(
                reservation,
                status="failed_consumed",
                result_hash=_failure_hash(exc),
            )
            self._poison(exc, f"{operation}_terminal")
            raise
        self.session.consume(reservation, status="completed", result_hash=_wire_hash(response))
        return response

    def trace_belief(self, query: TraceBeliefQuery) -> BeliefDecision:
        with self._lock:
            if not isinstance(query, TraceBeliefQuery):
                raise TypeError("query must be TraceBeliefQuery")
            response = self._sealed_query(
                query.query_id,
                "trace_belief",
                query,
                lambda: self.candidate.trace_belief(query),
            )
            if not isinstance(response, BeliefDecision):  # pragma: no cover
                raise RuntimeError("trace_belief returned the wrong type")
            return response

    def describe(self, query: TraceDescriptionQuery) -> DescriptionDecision:
        with self._lock:
            if not isinstance(query, TraceDescriptionQuery):
                raise TypeError("query must be TraceDescriptionQuery")
            response = self._sealed_query(
                query.query_id,
                "describe",
                query,
                lambda: self.candidate.describe(query),
            )
            if not isinstance(response, DescriptionDecision):  # pragma: no cover
                raise RuntimeError("describe returned the wrong type")
            return response

    def complete(self) -> Grade3AuditLedger:
        with self._lock:
            self._require_query_phase()
            try:
                self.candidate.assert_frozen()
                return self.session.complete()
            except BaseException as exc:
                self._poison(exc, "complete_terminal")
                raise

    def close(self) -> None:
        with self._lock:
            if self._callback_active:
                raise Grade3SessionStateError("runner callbacks cannot re-enter the runner")
            if self._closed:
                return
            self.candidate.close()
            self._closed = True


# A concise alias parallels ``SealedEvaluationRunner`` in the frozen v1 API.
SealedGrade3EvaluationRunner = Grade3EvaluationRunner


__all__ = [
    "CodebookFactory",
    "ExecutedGrade3Probe",
    "Grade3EvaluationRunner",
    "MotorActionCost",
    "MotorEpisodeResult",
    "MotorReset",
    "MotorStep",
    "MotorTranscriptTurn",
    "ProbeExecutor",
    "SealedGrade3EvaluationRunner",
]
