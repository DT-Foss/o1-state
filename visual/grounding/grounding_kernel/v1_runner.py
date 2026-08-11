"""Serialized evaluator orchestration for one isolated GroundZero-v1 candidate.

The runner reserves evaluator-owned resources before any charged world effect
or sealed candidate RPC.  Reservations are append-only and hash chained, so a
failed executor or candidate call remains visibly consumed instead of being
silently refunded.  The original :class:`SessionAuditLedger` remains the
stable session API; :attr:`SealedEvaluationRunner.audit` adds runner failures
and in-flight reservations without changing that ledger's schema.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from math import isclose, isfinite
from threading import RLock
from types import TracebackType
from typing import Any

from .certificates import manifest_hash
from .contracts import Action, Transition
from .v1_contracts import (
    MAX_BELIEF_CANDIDATES,
    ActionDecision,
    BeliefDecision,
    DescriptionDecision,
    ExperimentDecision,
    PublicTrace,
    PublicTransition,
    PublicTurn,
    SessionManifest,
    SessionPhase,
)
from .v1_isolation import (
    CandidateArtifactCommitment,
    FrozenCandidate,
    IsolatedGrounder,
)
from .v1_session import (
    EvaluationSession,
    SessionAuditLedger,
    SessionBudgetError,
    SessionStateError,
)
from .v1_wire import encode_message


ActionExecutor = Callable[[Action], PublicTransition | Transition]
ActionCost = Callable[[Action], float]
CodebookFactory = Callable[[], str]

_AUDIT_PROTOCOL = "grounding-runner-audit/1"
_AUDIT_GENESIS = "0" * 64


def _unit_cost(_action: Action) -> float:
    return 1.0


def _public_transition(
    value: PublicTransition | Transition,
    *,
    scalar_feedback: float | None = None,
) -> PublicTransition:
    if scalar_feedback is not None:
        raise SessionStateError("scalar feedback is forbidden during acquisition")
    if isinstance(value, PublicTransition):
        if value.scalar_feedback is not None:
            raise SessionStateError("scalar feedback is forbidden during acquisition")
        return value
    if not isinstance(value, Transition):
        raise TypeError("action executor must return PublicTransition or Transition")
    # The legacy Transition outcome code is evaluator-private and deliberately
    # omitted rather than remapped into a learner-visible feedback channel.
    return PublicTransition(value.before, value.action, value.after, None)


def _wire_hash(value: object) -> str:
    return manifest_hash({"wire": encode_message(value)})


def _failure_hash(error: BaseException) -> str:
    return manifest_hash(
        {
            "type": f"{type(error).__module__}.{type(error).__qualname__}",
            "message": str(error),
        }
    )


@dataclass(frozen=True, slots=True)
class RunnerAuditEvent:
    """One immutable link in the runner's reservation/operation hash chain."""

    sequence: int
    operation: str
    status: str
    request_hash: str | None
    decision_hash: str | None
    result_hash: str | None
    state_before_hash: str
    state_after_hash: str
    reserved_cost: float
    consumed_cost: float
    reserved_queries: int
    consumed_queries: int
    reservation_hash: str | None
    previous_hash: str
    event_hash: str

    def _material(self) -> dict[str, object]:
        return {
            "protocol": _AUDIT_PROTOCOL,
            "sequence": self.sequence,
            "operation": self.operation,
            "status": self.status,
            "request_hash": self.request_hash,
            "decision_hash": self.decision_hash,
            "result_hash": self.result_hash,
            "state_before_hash": self.state_before_hash,
            "state_after_hash": self.state_after_hash,
            "reserved_cost": self.reserved_cost,
            "consumed_cost": self.consumed_cost,
            "reserved_queries": self.reserved_queries,
            "consumed_queries": self.consumed_queries,
            "reservation_hash": self.reservation_hash,
            "previous_hash": self.previous_hash,
        }

    def to_dict(self) -> dict[str, object]:
        return {**self._material(), "event_hash": self.event_hash}


@dataclass(frozen=True, slots=True)
class RunnerAuditLedger:
    """Snapshot of runner reservations plus the compatible session ledger."""

    session: SessionAuditLedger
    failed_cost_consumed: float
    failed_queries_consumed: int
    reserved_cost: float
    reserved_queries: int
    remaining_cost: float
    remaining_queries: int
    events: tuple[RunnerAuditEvent, ...]

    @property
    def head_hash(self) -> str:
        return self.events[-1].event_hash if self.events else _AUDIT_GENESIS

    @property
    def intervention_cost_consumed(self) -> float:
        return self.session.intervention_cost_used + self.failed_cost_consumed

    @property
    def queries_consumed(self) -> int:
        return self.session.queries_used + self.failed_queries_consumed

    @property
    def chain_valid(self) -> bool:
        previous = _AUDIT_GENESIS
        for sequence, event in enumerate(self.events):
            if event.sequence != sequence or event.previous_hash != previous:
                return False
            if manifest_hash(event._material()) != event.event_hash:
                return False
            previous = event.event_hash
        return True

    @property
    def ledger_hash(self) -> str:
        return manifest_hash(
            {
                "protocol": _AUDIT_PROTOCOL,
                "session_ledger_hash": self.session.ledger_hash,
                "failed_cost_consumed": self.failed_cost_consumed,
                "failed_queries_consumed": self.failed_queries_consumed,
                "reserved_cost": self.reserved_cost,
                "reserved_queries": self.reserved_queries,
                "remaining_cost": self.remaining_cost,
                "remaining_queries": self.remaining_queries,
                "head_hash": self.head_hash,
            }
        )


@dataclass(frozen=True, slots=True)
class _Reservation:
    operation: str
    request_hash: str
    decision_hash: str | None
    cost: float
    queries: int
    event_hash: str


@dataclass(frozen=True, slots=True)
class ExecutedExperiment:
    decision: ExperimentDecision
    transition: PublicTransition | None
    cost: float
    audit_commitment: str | None = None


class SealedEvaluationRunner:
    """One authoritative evaluator ledger plus one persistent candidate process."""

    def __init__(
        self,
        commitment: CandidateArtifactCommitment,
        manifest: SessionManifest,
        codebook_factory: CodebookFactory,
        *,
        action_cost: ActionCost = _unit_cost,
        timeout: float = 20.0,
    ) -> None:
        if not isinstance(commitment, CandidateArtifactCommitment):
            raise TypeError("commitment must be CandidateArtifactCommitment")
        if not isinstance(manifest, SessionManifest):
            raise TypeError("manifest must be SessionManifest")
        if not callable(codebook_factory):
            raise TypeError(
                "codebook_factory must be a zero-argument callable created "
                "without materializing the codebook"
            )
        if not callable(action_cost):
            raise TypeError("action_cost must be callable")
        self.commitment = commitment
        self.session = EvaluationSession(manifest, commitment.digest)
        self.candidate = IsolatedGrounder(commitment, timeout=timeout)
        self._codebook_factory: CodebookFactory | None = codebook_factory
        self._action_cost = action_cost
        self._started = False
        self._closed = False
        self._lock = RLock()
        self._reserved_cost = 0.0
        self._reserved_queries = 0
        self._failed_cost_consumed = 0.0
        self._failed_queries_consumed = 0
        self._audit_events: list[RunnerAuditEvent] = []
        self._callback_active = False

    def __enter__(self) -> "SealedEvaluationRunner":
        return self.start()

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback_value: TracebackType | None,
    ) -> None:
        self.close()

    @property
    def remaining_cost(self) -> float:
        with self._lock:
            return max(
                0.0,
                self.session.remaining_cost
                - self._reserved_cost
                - self._failed_cost_consumed,
            )

    @property
    def remaining_queries(self) -> int:
        with self._lock:
            return max(
                0,
                self.session.remaining_queries
                - self._reserved_queries
                - self._failed_queries_consumed,
            )

    @property
    def audit(self) -> RunnerAuditLedger:
        with self._lock:
            return RunnerAuditLedger(
                session=self.session.ledger,
                failed_cost_consumed=self._failed_cost_consumed,
                failed_queries_consumed=self._failed_queries_consumed,
                reserved_cost=self._reserved_cost,
                reserved_queries=self._reserved_queries,
                remaining_cost=self.remaining_cost,
                remaining_queries=self.remaining_queries,
                events=tuple(self._audit_events),
            )

    def _state_hash(self) -> str:
        frozen = self.candidate.frozen
        return manifest_hash(
            {
                "session_ledger_hash": self.session.ledger.ledger_hash,
                "candidate_requests": self.candidate.request_count,
                "candidate_checkpoint": (
                    None if frozen is None else frozen.checkpoint_commitment
                ),
                "reserved_cost": self._reserved_cost,
                "reserved_queries": self._reserved_queries,
                "failed_cost_consumed": self._failed_cost_consumed,
                "failed_queries_consumed": self._failed_queries_consumed,
                "started": self._started,
                "closed": self._closed,
                "callback_active": self._callback_active,
            }
        )

    def _append_audit(
        self,
        operation: str,
        status: str,
        *,
        request_hash: str | None,
        decision_hash: str | None,
        result_hash: str | None,
        state_before_hash: str,
        state_after_hash: str,
        reserved_cost: float = 0.0,
        consumed_cost: float = 0.0,
        reserved_queries: int = 0,
        consumed_queries: int = 0,
        reservation_hash: str | None = None,
    ) -> RunnerAuditEvent:
        previous = (
            self._audit_events[-1].event_hash
            if self._audit_events
            else _AUDIT_GENESIS
        )
        values: dict[str, Any] = {
            "protocol": _AUDIT_PROTOCOL,
            "sequence": len(self._audit_events),
            "operation": operation,
            "status": status,
            "request_hash": request_hash,
            "decision_hash": decision_hash,
            "result_hash": result_hash,
            "state_before_hash": state_before_hash,
            "state_after_hash": state_after_hash,
            "reserved_cost": float(reserved_cost),
            "consumed_cost": float(consumed_cost),
            "reserved_queries": reserved_queries,
            "consumed_queries": consumed_queries,
            "reservation_hash": reservation_hash,
            "previous_hash": previous,
        }
        event = RunnerAuditEvent(
            sequence=values["sequence"],
            operation=operation,
            status=status,
            request_hash=request_hash,
            decision_hash=decision_hash,
            result_hash=result_hash,
            state_before_hash=state_before_hash,
            state_after_hash=state_after_hash,
            reserved_cost=float(reserved_cost),
            consumed_cost=float(consumed_cost),
            reserved_queries=reserved_queries,
            consumed_queries=consumed_queries,
            reservation_hash=reservation_hash,
            previous_hash=previous,
            event_hash=manifest_hash(values),
        )
        self._audit_events.append(event)
        return event

    def _record_operation(
        self,
        operation: str,
        status: str,
        *,
        request_hash: str | None,
        decision_hash: str | None,
        result_hash: str | None,
        state_before_hash: str,
    ) -> RunnerAuditEvent:
        return self._append_audit(
            operation,
            status,
            request_hash=request_hash,
            decision_hash=decision_hash,
            result_hash=result_hash,
            state_before_hash=state_before_hash,
            state_after_hash=self._state_hash(),
        )

    def _reserve(
        self,
        operation: str,
        request_hash: str,
        *,
        decision_hash: str | None = None,
        cost: float = 0.0,
        queries: int = 0,
    ) -> _Reservation:
        if cost > self.remaining_cost + 1e-12:
            raise SessionBudgetError("intervention cost budget exhausted")
        if queries > self.remaining_queries:
            raise SessionBudgetError("sealed query budget exhausted")
        before = self._state_hash()
        self._reserved_cost += cost
        self._reserved_queries += queries
        event = self._append_audit(
            operation,
            "reserved",
            request_hash=request_hash,
            decision_hash=decision_hash,
            result_hash=None,
            state_before_hash=before,
            state_after_hash=self._state_hash(),
            reserved_cost=cost,
            reserved_queries=queries,
        )
        return _Reservation(
            operation,
            request_hash,
            decision_hash,
            cost,
            queries,
            event.event_hash,
        )

    def _finish_reservation(
        self,
        reservation: _Reservation,
        *,
        success: bool,
        session_charged: bool,
        decision_hash: str | None,
        result_hash: str,
    ) -> RunnerAuditEvent:
        before = self._state_hash()
        self._reserved_cost -= reservation.cost
        self._reserved_queries -= reservation.queries
        if not success and not session_charged:
            self._failed_cost_consumed += reservation.cost
            self._failed_queries_consumed += reservation.queries
        status = "completed" if success else "failed_consumed"
        return self._append_audit(
            reservation.operation,
            status,
            request_hash=reservation.request_hash,
            decision_hash=decision_hash or reservation.decision_hash,
            result_hash=result_hash,
            state_before_hash=before,
            state_after_hash=self._state_hash(),
            reserved_cost=reservation.cost,
            consumed_cost=reservation.cost,
            reserved_queries=reservation.queries,
            consumed_queries=reservation.queries,
            reservation_hash=reservation.event_hash,
        )

    def start(self) -> "SealedEvaluationRunner":
        with self._lock:
            if self._callback_active:
                raise SessionStateError("runner callbacks cannot re-enter the runner")
            if self._closed:
                raise RuntimeError("runner is closed")
            if self._started:
                raise RuntimeError("runner is already started")
            before = self._state_hash()
            try:
                if (
                    self.session.phase is not SessionPhase.SUPPORT
                    or self.session.ledger.events
                    or self.session.commitments["codebook"] is not None
                ):
                    raise SessionStateError(
                        "runner session must remain pristine until artifact verification"
                    )
                # start() verifies live files, snapshots them, spawns the child;
                # begin() proves the child rehashed/imported that snapshot.
                self.candidate.start()
                self.candidate.begin(self.session.manifest)
                factory = self._codebook_factory
                if factory is None:  # pragma: no cover - lifecycle invariant
                    raise RuntimeError("codebook factory was already consumed")
                self._callback_active = True
                try:
                    codebook_commitment = factory()
                finally:
                    self._callback_active = False
                self._codebook_factory = None
                self.session.commit_codebook(codebook_commitment)
                self._started = True
            except BaseException as exc:
                self._codebook_factory = None
                self.candidate.close()
                self._closed = True
                self._record_operation(
                    "start",
                    "failed",
                    request_hash=None,
                    decision_hash=None,
                    result_hash=_failure_hash(exc),
                    state_before_hash=before,
                )
                raise
            self._record_operation(
                "start",
                "completed",
                request_hash=None,
                decision_hash=None,
                result_hash=self.session.commitments["codebook"],
                state_before_hash=before,
            )
            return self

    def _require_started(self) -> None:
        if self._callback_active:
            raise SessionStateError("runner callbacks cannot re-enter the runner")
        if not self._started or self._closed:
            raise RuntimeError("runner must be started and open")

    def support(self, turn: PublicTurn, trace: PublicTrace) -> None:
        with self._lock:
            self._require_started()
            if not isinstance(turn, PublicTurn) or not isinstance(trace, PublicTrace):
                raise TypeError("support requires a PublicTurn and PublicTrace")
            if not isclose(
                turn.remaining_cost,
                self.remaining_cost,
                rel_tol=0.0,
                abs_tol=1e-9,
            ):
                raise SessionStateError(
                    "turn.remaining_cost is not evaluator-authoritative"
                )
            before = self._state_hash()
            request_hash = manifest_hash(
                {"turn": encode_message(turn), "trace": encode_message(trace)}
            )
            try:
                self.session.record_support(turn, trace)
                self.candidate.observe_support(turn, trace)
            except BaseException as exc:
                self._record_operation(
                    "support",
                    "failed_consumed",
                    request_hash=request_hash,
                    decision_hash=None,
                    result_hash=_failure_hash(exc),
                    state_before_hash=before,
                )
                raise
            self._record_operation(
                "support",
                "completed",
                request_hash=request_hash,
                decision_hash=None,
                result_hash=None,
                state_before_hash=before,
            )

    def begin_acquisition(self) -> None:
        with self._lock:
            self._require_started()
            before = self._state_hash()
            self.session.begin_acquisition()
            self._record_operation(
                "begin_acquisition",
                "completed",
                request_hash=None,
                decision_hash=None,
                result_hash=None,
                state_before_hash=before,
            )

    def experiment(
        self,
        turn: PublicTurn,
        execute: ActionExecutor,
        *,
        scalar_feedback: float | None = None,
    ) -> ExecutedExperiment:
        """Choose, reserve, execute and charge one acquisition intervention."""

        with self._lock:
            self._require_started()
            if self.session.phase is not SessionPhase.ACQUISITION:
                raise SessionStateError("experiment requires acquisition phase")
            if not isinstance(turn, PublicTurn) or turn.phase is not SessionPhase.ACQUISITION:
                raise SessionStateError("experiment turn must declare acquisition phase")
            if turn.scalar_feedback is not None or scalar_feedback is not None:
                raise SessionStateError(
                    "scalar feedback is forbidden during acquisition"
                )
            if not isclose(
                turn.remaining_cost,
                self.remaining_cost,
                rel_tol=0.0,
                abs_tol=1e-9,
            ):
                raise SessionStateError(
                    "turn.remaining_cost is not evaluator-authoritative"
                )
            if not callable(execute):
                raise TypeError("execute must be callable")
            if self.remaining_cost <= 0.0:
                raise SessionBudgetError("intervention cost budget exhausted")
            request_hash = _wire_hash(turn)
            decision_state = self._state_hash()
            try:
                decision = self.candidate.choose_experiment(turn)
            except BaseException as exc:
                self._record_operation(
                    "experiment_decision",
                    "failed",
                    request_hash=request_hash,
                    decision_hash=None,
                    result_hash=_failure_hash(exc),
                    state_before_hash=decision_state,
                )
                raise
            decision_hash = _wire_hash(decision)
            if decision.action is None:
                event = self._record_operation(
                    "experiment",
                    "abstained",
                    request_hash=request_hash,
                    decision_hash=decision_hash,
                    result_hash=None,
                    state_before_hash=decision_state,
                )
                return ExecutedExperiment(decision, None, 0.0, event.event_hash)
            try:
                self._callback_active = True
                try:
                    cost = self._action_cost(decision.action)
                finally:
                    self._callback_active = False
                if isinstance(cost, bool) or not isinstance(cost, (int, float)):
                    raise TypeError("action cost schedule must return a number")
                numeric_cost = float(cost)
                if not isfinite(numeric_cost) or numeric_cost <= 0.0:
                    raise ValueError(
                        "action cost schedule must return a finite positive cost"
                    )
                reservation = self._reserve(
                    "experiment",
                    request_hash,
                    decision_hash=decision_hash,
                    cost=numeric_cost,
                )
            except BaseException as exc:
                self._record_operation(
                    "experiment_preflight",
                    "failed",
                    request_hash=request_hash,
                    decision_hash=decision_hash,
                    result_hash=_failure_hash(exc),
                    state_before_hash=decision_state,
                )
                raise
            session_charged = False
            transition: PublicTransition | None = None
            try:
                self._callback_active = True
                try:
                    executed = execute(decision.action)
                finally:
                    self._callback_active = False
                transition = _public_transition(
                    executed, scalar_feedback=scalar_feedback
                )
                if transition.action != decision.action:
                    raise RuntimeError(
                        "executor returned a transition for a different action"
                    )
                if transition.before != turn.observation:
                    raise RuntimeError(
                        "executor transition.before must equal turn.observation"
                    )
                # The session charge precedes evidence delivery.  Candidate
                # failure therefore cannot erase an executed intervention.
                self.session.record_experiment(transition, numeric_cost)
                session_charged = True
                self.candidate.observe_experiment(turn, transition)
            except BaseException as exc:
                result_hash = manifest_hash(
                    {
                        "transition": (
                            None if transition is None else encode_message(transition)
                        ),
                        "failure": _failure_hash(exc),
                    }
                )
                self._finish_reservation(
                    reservation,
                    success=False,
                    session_charged=session_charged,
                    decision_hash=decision_hash,
                    result_hash=result_hash,
                )
                raise
            event = self._finish_reservation(
                reservation,
                success=True,
                session_charged=True,
                decision_hash=decision_hash,
                result_hash=_wire_hash(transition),
            )
            return ExecutedExperiment(
                decision,
                transition,
                numeric_cost,
                event.event_hash,
            )

    def freeze(self) -> FrozenCandidate:
        with self._lock:
            self._require_started()
            before = self._state_hash()
            try:
                frozen = self.candidate.freeze()
                self.session.freeze(frozen.checkpoint_commitment)
            except BaseException as exc:
                self._record_operation(
                    "freeze",
                    "failed",
                    request_hash=None,
                    decision_hash=None,
                    result_hash=_failure_hash(exc),
                    state_before_hash=before,
                )
                raise
            self._record_operation(
                "freeze",
                "completed",
                request_hash=None,
                decision_hash=None,
                result_hash=frozen.checkpoint_commitment,
                state_before_hash=before,
            )
            return frozen

    def _require_query_phase(self) -> None:
        if self.session.phase is not SessionPhase.FROZEN_QUERY:
            raise SessionStateError("sealed query requires frozen_query phase")

    def _run_query(
        self,
        operation: str,
        request: object,
        request_hash: str,
        call: Callable[[], ActionDecision | BeliefDecision | DescriptionDecision],
    ) -> ActionDecision | BeliefDecision | DescriptionDecision:
        self._require_query_phase()
        reservation = self._reserve(
            operation,
            request_hash,
            queries=1,
        )
        session_charged = False
        response: ActionDecision | BeliefDecision | DescriptionDecision | None = None
        try:
            response = call()
            self.candidate.assert_frozen()
            self.session.record_query(request, response)
            session_charged = True
        except BaseException as exc:
            self._finish_reservation(
                reservation,
                success=False,
                session_charged=session_charged,
                decision_hash=None if response is None else _wire_hash(response),
                result_hash=_failure_hash(exc),
            )
            raise
        self._finish_reservation(
            reservation,
            success=True,
            session_charged=True,
            decision_hash=_wire_hash(response),
            result_hash=_wire_hash(response),
        )
        return response

    def describe(self, trace: PublicTrace) -> DescriptionDecision:
        with self._lock:
            self._require_started()
            if not isinstance(trace, PublicTrace):
                raise TypeError("trace must be PublicTrace")
            request = trace.feedback_stripped()
            response = self._run_query(
                "describe",
                request,
                _wire_hash(request),
                lambda: self.candidate.describe(request),
            )
            if not isinstance(response, DescriptionDecision):  # pragma: no cover
                raise RuntimeError("describe returned the wrong decision type")
            return response

    def act(self, turn: PublicTurn) -> ActionDecision:
        """Begin/continue one goal and ledger the action decision as a query."""

        with self._lock:
            self._require_started()
            if not isinstance(turn, PublicTurn):
                raise TypeError("turn must be PublicTurn")
            if turn.phase is not SessionPhase.FROZEN_QUERY:
                raise SessionStateError(
                    "action query turn must declare frozen_query phase"
                )
            if not isclose(
                turn.remaining_cost,
                self.remaining_cost,
                rel_tol=0.0,
                abs_tol=1e-9,
            ):
                raise SessionStateError(
                    "turn.remaining_cost is not evaluator-authoritative"
                )

            def call() -> ActionDecision:
                if turn.utterance is not None:
                    self.candidate.begin_goal(turn.utterance, turn.observation)
                return self.candidate.act(turn.observation)

            response = self._run_query(
                "act",
                turn,
                _wire_hash(turn),
                call,
            )
            if not isinstance(response, ActionDecision):  # pragma: no cover
                raise RuntimeError("act returned the wrong decision type")
            return response

    def report_belief(
        self,
        turn: PublicTurn,
        candidates: Sequence[int],
    ) -> BeliefDecision:
        with self._lock:
            self._require_started()
            if not isinstance(turn, PublicTurn):
                raise TypeError("turn must be PublicTurn")
            if turn.phase is not SessionPhase.FROZEN_QUERY:
                raise SessionStateError(
                    "belief turn must declare frozen_query phase"
                )
            if not isclose(
                turn.remaining_cost,
                self.remaining_cost,
                rel_tol=0.0,
                abs_tol=1e-9,
            ):
                raise SessionStateError(
                    "turn.remaining_cost is not evaluator-authoritative"
                )
            values = tuple(candidates)
            if any(isinstance(item, bool) or not isinstance(item, int) for item in values):
                raise TypeError("belief candidates must be integers")
            if len(values) > MAX_BELIEF_CANDIDATES:
                raise ValueError(
                    f"belief candidates cannot exceed {MAX_BELIEF_CANDIDATES}"
                )
            if len(set(values)) != len(values):
                raise ValueError("belief candidates must be unique")
            request_hash = manifest_hash(
                {"turn": encode_message(turn), "candidates": values}
            )
            response = self._run_query(
                "report_belief",
                turn,
                request_hash,
                lambda: self.candidate.report_belief(values),
            )
            if not isinstance(response, BeliefDecision):  # pragma: no cover
                raise RuntimeError("report_belief returned the wrong decision type")
            return response

    def complete(self) -> SessionAuditLedger:
        with self._lock:
            self._require_started()
            if self._reserved_cost or self._reserved_queries:  # pragma: no cover
                raise SessionStateError("cannot complete with outstanding reservations")
            before = self._state_hash()
            self.candidate.assert_frozen()
            ledger = self.session.complete()
            self._record_operation(
                "complete",
                "completed",
                request_hash=None,
                decision_hash=None,
                result_hash=ledger.ledger_hash,
                state_before_hash=before,
            )
            return ledger

    def close(self) -> None:
        with self._lock:
            if self._callback_active:
                raise SessionStateError("runner callbacks cannot re-enter the runner")
            if self._closed:
                return
            before = self._state_hash()
            self.candidate.close()
            self._closed = True
            self._record_operation(
                "close",
                "completed",
                request_hash=None,
                decision_hash=None,
                result_hash=None,
                state_before_hash=before,
            )


__all__ = [
    "ActionCost",
    "ActionExecutor",
    "CodebookFactory",
    "ExecutedExperiment",
    "RunnerAuditEvent",
    "RunnerAuditLedger",
    "SealedEvaluationRunner",
]
