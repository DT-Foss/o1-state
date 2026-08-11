"""Authoritative phase and budget ledger for sealed GroundZero-v1 blocks.

This evaluator-side controller does not know world semantics.  Its job is to
make the experiment order auditable: commit the candidate first, commit the
secret codebook second, charge every support episode and intervention, freeze
one checkpoint, and only then admit bounded sealed queries.  It accepts only
values encodable by :mod:`grounding_kernel.v1_wire`.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from math import isclose, isfinite
from threading import RLock
from types import MappingProxyType
from typing import Any

from .certificates import manifest_hash
from .v1_contracts import (
    ActionDecision,
    BeliefDecision,
    DescriptionDecision,
    PublicTrace,
    PublicTransition,
    PublicTurn,
    SessionManifest,
    SessionPhase,
)
from .v1_wire import encode_message, public_schema_manifest


class SessionStateError(RuntimeError):
    """An operation is not permitted in the current evaluation phase."""


class SessionBudgetError(RuntimeError):
    """A support, intervention, or query would exceed a committed budget."""


class SessionEventKind(str, Enum):
    CODEBOOK_COMMIT = "codebook_commit"
    SUPPORT = "support"
    PHASE_CHANGE = "phase_change"
    EXPERIMENT = "experiment"
    FREEZE = "freeze"
    QUERY = "query"
    REVISION_EXPERIMENT = "revision_experiment"
    COMPLETE = "complete"


def _commitment(value: object, field: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{field} must be a lowercase SHA-256 hex string")
    if len(value) != 64 or value.lower() != value:
        raise ValueError(f"{field} must be a lowercase SHA-256 hex string")
    try:
        bytes.fromhex(value)
    except ValueError as exc:
        raise ValueError(f"{field} must be a lowercase SHA-256 hex string") from exc
    return value


@dataclass(frozen=True, slots=True)
class SessionEvent:
    sequence: int
    kind: SessionEventKind
    phase: SessionPhase
    payload_hash: str
    cost: float
    remaining_cost: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "sequence": self.sequence,
            "kind": self.kind.value,
            "phase": self.phase.value,
            "payload_hash": self.payload_hash,
            "cost": self.cost,
            "remaining_cost": self.remaining_cost,
        }


@dataclass(frozen=True, slots=True)
class SessionAuditLedger:
    manifest_hash: str
    wire_schema_hash: str
    model_commitment: str
    codebook_commitment: str | None
    checkpoint_commitment: str | None
    phase: SessionPhase
    support_episodes_used: int
    intervention_cost_used: float
    queries_used: int
    events: tuple[SessionEvent, ...]

    @property
    def ledger_hash(self) -> str:
        return manifest_hash(
            {
                "manifest_hash": self.manifest_hash,
                "wire_schema_hash": self.wire_schema_hash,
                "model_commitment": self.model_commitment,
                "codebook_commitment": self.codebook_commitment,
                "checkpoint_commitment": self.checkpoint_commitment,
                "phase": self.phase.value,
                "support_episodes_used": self.support_episodes_used,
                "intervention_cost_used": self.intervention_cost_used,
                "queries_used": self.queries_used,
                "events": [event.to_dict() for event in self.events],
            }
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "manifest_hash": self.manifest_hash,
            "wire_schema_hash": self.wire_schema_hash,
            "model_commitment": self.model_commitment,
            "codebook_commitment": self.codebook_commitment,
            "checkpoint_commitment": self.checkpoint_commitment,
            "phase": self.phase.value,
            "support_episodes_used": self.support_episodes_used,
            "intervention_cost_used": self.intervention_cost_used,
            "queries_used": self.queries_used,
            "events": [event.to_dict() for event in self.events],
            "ledger_hash": self.ledger_hash,
        }


class EvaluationSession:
    """Thread-safe fail-closed controller for one world×codebook block."""

    def __init__(self, manifest: SessionManifest, model_commitment: str) -> None:
        if not isinstance(manifest, SessionManifest):
            raise TypeError("manifest must be a SessionManifest")
        self._manifest = manifest
        self._model_commitment = _commitment(model_commitment, "model_commitment")
        self._codebook_commitment: str | None = None
        self._checkpoint_commitment: str | None = None
        self._phase = SessionPhase.SUPPORT
        self._support_used = 0
        self._cost_used = 0.0
        self._queries_used = 0
        self._turn_ids: set[int] = set()
        self._events: list[SessionEvent] = []
        self._lock = RLock()
        self._manifest_hash = manifest_hash(
            {
                "protocol_version": manifest.protocol_version,
                "sensor_schema": manifest.sensor_schema,
                "action_schema": manifest.action_schema,
                "support_episode_budget": manifest.support_episode_budget,
                "intervention_cost_budget": manifest.intervention_cost_budget,
                "query_budget": manifest.query_budget,
            }
        )
        self._wire_schema_hash = manifest_hash(public_schema_manifest())

    @property
    def manifest(self) -> SessionManifest:
        return self._manifest

    @property
    def phase(self) -> SessionPhase:
        return self._phase

    @property
    def remaining_cost(self) -> float:
        return max(0.0, self._manifest.intervention_cost_budget - self._cost_used)

    @property
    def remaining_support_episodes(self) -> int:
        return self._manifest.support_episode_budget - self._support_used

    @property
    def remaining_queries(self) -> int:
        return self._manifest.query_budget - self._queries_used

    def _require_phase(self, *phases: SessionPhase) -> None:
        if self._phase not in phases:
            expected = ", ".join(phase.value for phase in phases)
            raise SessionStateError(
                f"operation requires phase {expected}; current phase is {self._phase.value}"
            )

    def _append(
        self,
        kind: SessionEventKind,
        payload: object,
        *,
        cost: float = 0.0,
    ) -> None:
        self._events.append(
            SessionEvent(
                sequence=len(self._events),
                kind=kind,
                phase=self._phase,
                payload_hash=manifest_hash(payload),
                cost=cost,
                remaining_cost=self.remaining_cost,
            )
        )

    def commit_codebook(self, codebook_commitment: str) -> None:
        """Commit the secret lexicon after the candidate model commitment."""

        with self._lock:
            self._require_phase(SessionPhase.SUPPORT)
            if self._support_used:
                raise SessionStateError("codebook must be committed before support begins")
            if self._codebook_commitment is not None:
                raise SessionStateError("codebook is already committed")
            value = _commitment(codebook_commitment, "codebook_commitment")
            if value == self._model_commitment:
                raise ValueError("model and codebook commitments must be distinct")
            self._codebook_commitment = value
            self._append(SessionEventKind.CODEBOOK_COMMIT, {"commitment": value})

    def record_support(self, turn: PublicTurn, trace: PublicTrace) -> None:
        """Charge and ledger one ostensive support episode."""

        with self._lock:
            self._require_phase(SessionPhase.SUPPORT)
            if self._codebook_commitment is None:
                raise SessionStateError("commit the post-model codebook before support")
            if self._support_used >= self._manifest.support_episode_budget:
                raise SessionBudgetError("support episode budget exhausted")
            if not isinstance(turn, PublicTurn) or not isinstance(trace, PublicTrace):
                raise TypeError("support requires a PublicTurn and PublicTrace")
            if turn.phase is not SessionPhase.SUPPORT:
                raise SessionStateError("support turns must declare the support phase")
            if turn.turn_id in self._turn_ids:
                raise SessionStateError("turn_id values must be unique")
            if not isclose(turn.remaining_cost, self.remaining_cost, abs_tol=1e-9):
                raise SessionStateError("turn.remaining_cost is not evaluator-authoritative")
            if turn.observation != trace.initial:
                raise SessionStateError("support turn observation must equal trace.initial")
            self._turn_ids.add(turn.turn_id)
            self._support_used += 1
            self._append(
                SessionEventKind.SUPPORT,
                {
                    "turn": encode_message(turn),
                    "trace": encode_message(trace),
                },
            )

    def begin_acquisition(self) -> None:
        with self._lock:
            self._require_phase(SessionPhase.SUPPORT)
            if self._codebook_commitment is None:
                raise SessionStateError("codebook commitment is missing")
            self._phase = SessionPhase.ACQUISITION
            self._append(SessionEventKind.PHASE_CHANGE, {"to": self._phase.value})

    def _record_costed_transition(
        self,
        transition: PublicTransition,
        cost: float,
        kind: SessionEventKind,
    ) -> None:
        if not isinstance(transition, PublicTransition):
            raise TypeError("experiment result must be a PublicTransition")
        if isinstance(cost, bool) or not isinstance(cost, (int, float)):
            raise TypeError("experiment cost must be a finite positive number")
        numeric_cost = float(cost)
        if not isfinite(numeric_cost) or numeric_cost <= 0.0:
            raise ValueError("experiment cost must be finite and positive")
        if numeric_cost > self.remaining_cost + 1e-12:
            raise SessionBudgetError("intervention cost budget exhausted")
        self._cost_used += numeric_cost
        self._append(
            kind,
            {"transition": encode_message(transition)},
            cost=numeric_cost,
        )

    def record_experiment(self, transition: PublicTransition, cost: float = 1.0) -> None:
        with self._lock:
            self._require_phase(SessionPhase.ACQUISITION)
            if (
                isinstance(transition, PublicTransition)
                and transition.scalar_feedback is not None
            ):
                raise SessionStateError(
                    "acquisition transitions must be feedback-stripped"
                )
            self._record_costed_transition(
                transition, cost, SessionEventKind.EXPERIMENT
            )

    def freeze(self, checkpoint_commitment: str) -> None:
        """Freeze exactly one checkpoint before any sealed query."""

        with self._lock:
            self._require_phase(SessionPhase.ACQUISITION)
            if self._checkpoint_commitment is not None:
                raise SessionStateError("a checkpoint is already frozen")
            value = _commitment(checkpoint_commitment, "checkpoint_commitment")
            self._checkpoint_commitment = value
            self._phase = SessionPhase.FROZEN_QUERY
            self._append(SessionEventKind.FREEZE, {"checkpoint": value})

    def record_query(self, request: object, response: object) -> None:
        """Ledger one sealed request/response pair using only public wire values."""

        with self._lock:
            self._require_phase(SessionPhase.FROZEN_QUERY)
            if self._queries_used >= self._manifest.query_budget:
                raise SessionBudgetError("sealed query budget exhausted")
            if isinstance(request, PublicTrace) and request.has_feedback:
                raise SessionStateError("sealed query traces must be feedback-stripped")
            if (
                isinstance(request, PublicTransition)
                and request.scalar_feedback is not None
            ):
                raise SessionStateError("sealed query transitions cannot contain feedback")
            if isinstance(request, PublicTurn) and request.phase is not SessionPhase.FROZEN_QUERY:
                raise SessionStateError("sealed query turns must declare frozen_query phase")
            if not isinstance(
                response,
                (ActionDecision, BeliefDecision, DescriptionDecision),
            ):
                raise SessionStateError("sealed query responses must be decision records")
            request_bytes = encode_message(request)
            response_bytes = encode_message(response)
            self._queries_used += 1
            self._append(
                SessionEventKind.QUERY,
                {"request": request_bytes, "response": response_bytes},
            )

    def begin_revision(self) -> None:
        """Open the explicitly separate post-freeze revision phase."""

        with self._lock:
            self._require_phase(SessionPhase.FROZEN_QUERY)
            self._phase = SessionPhase.REVISION
            self._append(SessionEventKind.PHASE_CHANGE, {"to": self._phase.value})

    def record_revision_experiment(
        self, transition: PublicTransition, cost: float = 1.0
    ) -> None:
        with self._lock:
            self._require_phase(SessionPhase.REVISION)
            self._record_costed_transition(
                transition,
                cost,
                SessionEventKind.REVISION_EXPERIMENT,
            )

    def complete(self) -> SessionAuditLedger:
        with self._lock:
            self._require_phase(SessionPhase.FROZEN_QUERY, SessionPhase.REVISION)
            self._phase = SessionPhase.COMPLETE
            self._append(SessionEventKind.COMPLETE, {"complete": True})
            return self.ledger

    @property
    def ledger(self) -> SessionAuditLedger:
        with self._lock:
            return SessionAuditLedger(
                manifest_hash=self._manifest_hash,
                wire_schema_hash=self._wire_schema_hash,
                model_commitment=self._model_commitment,
                codebook_commitment=self._codebook_commitment,
                checkpoint_commitment=self._checkpoint_commitment,
                phase=self._phase,
                support_episodes_used=self._support_used,
                intervention_cost_used=self._cost_used,
                queries_used=self._queries_used,
                events=tuple(self._events),
            )

    @property
    def commitments(self) -> MappingProxyType[str, str | None]:
        return MappingProxyType(
            {
                "model": self._model_commitment,
                "codebook": self._codebook_commitment,
                "checkpoint": self._checkpoint_commitment,
            }
        )


__all__ = [
    "EvaluationSession",
    "SessionAuditLedger",
    "SessionBudgetError",
    "SessionEvent",
    "SessionEventKind",
    "SessionStateError",
]
