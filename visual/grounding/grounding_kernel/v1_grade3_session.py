"""Evaluator-owned phases, budgets, and hash-chain for Grade-3 sessions.

The controller is deliberately ignorant of world semantics.  It commits the
candidate and SDK before a late codebook commitment, meters five independent
resources, and consumes every reservation even when the operation fails.
Reservations are written to the ledger before the corresponding candidate or
world callback can run.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import Enum
from math import isclose, isfinite
from threading import RLock
from types import MappingProxyType
from .certificates import manifest_hash
from .v1_grade3_contracts import Grade3SessionManifest
from .v1_grade3_wire import grade3_schema_manifest


GRADE3_LEDGER_PROTOCOL = "grounding-grade3-ledger/1"
_GENESIS_HASH = "0" * 64


class Grade3SessionStateError(RuntimeError):
    """An operation is forbidden in the current Grade-3 phase."""


class Grade3SessionBudgetError(RuntimeError):
    """A reservation would exceed one independently committed budget."""


class Grade3SessionPhase(str, Enum):
    SUPPORT = "support"
    ACQUISITION = "acquisition"
    FROZEN_QUERY = "frozen_query"
    COMPLETE = "complete"


class Grade3Resource(str, Enum):
    SUPPORT_RECORD = "support_record"
    ACQUISITION_COST = "acquisition_cost"
    SEALED_QUERY = "sealed_query"
    MOTOR_ACTION_COST = "motor_action_cost"
    MOTOR_RESET = "motor_reset"


class Grade3EventKind(str, Enum):
    CODEBOOK_COMMIT = "codebook_commit"
    PHASE_CHANGE = "phase_change"
    RESERVE = "reserve"
    CONSUME = "consume"
    OPERATION = "operation"
    FREEZE = "freeze"
    COMPLETE = "complete"


def _digest(value: object, field: str) -> str:
    if not isinstance(value, str) or len(value) != 64 or value.lower() != value:
        raise ValueError(f"{field} must be a lowercase SHA-256 digest")
    try:
        bytes.fromhex(value)
    except ValueError as exc:
        raise ValueError(f"{field} must be a lowercase SHA-256 digest") from exc
    return value


def _amount(value: object, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{field} must be a finite positive number")
    result = float(value)
    if not isfinite(result) or result <= 0.0:
        raise ValueError(f"{field} must be a finite positive number")
    return result


def _optional_digest(value: str | None, field: str) -> str | None:
    return None if value is None else _digest(value, field)


@dataclass(frozen=True, slots=True)
class Grade3Reservation:
    """Opaque proof that one resource was reserved in the ledger."""

    reservation_id: int
    resource: Grade3Resource
    amount: float
    operation: str
    request_hash: str
    reservation_event_hash: str


@dataclass(frozen=True, slots=True)
class Grade3SessionEvent:
    sequence: int
    kind: Grade3EventKind
    phase: Grade3SessionPhase
    operation: str
    status: str
    resource: Grade3Resource | None
    amount: float
    reservation_id: int | None
    request_hash: str | None
    result_hash: str | None
    previous_hash: str
    event_hash: str

    def _material(self) -> dict[str, object]:
        return {
            "protocol": GRADE3_LEDGER_PROTOCOL,
            "sequence": self.sequence,
            "kind": self.kind.value,
            "phase": self.phase.value,
            "operation": self.operation,
            "status": self.status,
            "resource": None if self.resource is None else self.resource.value,
            "amount": self.amount,
            "reservation_id": self.reservation_id,
            "request_hash": self.request_hash,
            "result_hash": self.result_hash,
            "previous_hash": self.previous_hash,
        }

    def to_dict(self) -> dict[str, object]:
        return {**self._material(), "event_hash": self.event_hash}


@dataclass(frozen=True, slots=True)
class Grade3AuditLedger:
    manifest_hash: str
    wire_schema_hash: str
    artifact_commitment: str
    sdk_commitment: str
    codebook_commitment: str | None
    checkpoint_commitment: str | None
    phase: Grade3SessionPhase
    support_records_used: int
    acquisition_cost_used: float
    sealed_queries_used: int
    motor_action_cost_used: float
    motor_resets_used: int
    support_records_reserved: int
    acquisition_cost_reserved: float
    sealed_queries_reserved: int
    motor_action_cost_reserved: float
    motor_resets_reserved: int
    sealed_query_ids: tuple[int, ...]
    events: tuple[Grade3SessionEvent, ...]

    @property
    def head_hash(self) -> str:
        return self.events[-1].event_hash if self.events else _GENESIS_HASH

    @property
    def chain_valid(self) -> bool:
        previous = _GENESIS_HASH
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
                "protocol": GRADE3_LEDGER_PROTOCOL,
                "manifest_hash": self.manifest_hash,
                "wire_schema_hash": self.wire_schema_hash,
                "artifact_commitment": self.artifact_commitment,
                "sdk_commitment": self.sdk_commitment,
                "codebook_commitment": self.codebook_commitment,
                "checkpoint_commitment": self.checkpoint_commitment,
                "phase": self.phase.value,
                "support_records_used": self.support_records_used,
                "acquisition_cost_used": self.acquisition_cost_used,
                "sealed_queries_used": self.sealed_queries_used,
                "motor_action_cost_used": self.motor_action_cost_used,
                "motor_resets_used": self.motor_resets_used,
                "support_records_reserved": self.support_records_reserved,
                "acquisition_cost_reserved": self.acquisition_cost_reserved,
                "sealed_queries_reserved": self.sealed_queries_reserved,
                "motor_action_cost_reserved": self.motor_action_cost_reserved,
                "motor_resets_reserved": self.motor_resets_reserved,
                "sealed_query_ids": self.sealed_query_ids,
                "head_hash": self.head_hash,
                "events": [event.to_dict() for event in self.events],
            }
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "manifest_hash": self.manifest_hash,
            "wire_schema_hash": self.wire_schema_hash,
            "artifact_commitment": self.artifact_commitment,
            "sdk_commitment": self.sdk_commitment,
            "codebook_commitment": self.codebook_commitment,
            "checkpoint_commitment": self.checkpoint_commitment,
            "phase": self.phase.value,
            "support_records_used": self.support_records_used,
            "acquisition_cost_used": self.acquisition_cost_used,
            "sealed_queries_used": self.sealed_queries_used,
            "motor_action_cost_used": self.motor_action_cost_used,
            "motor_resets_used": self.motor_resets_used,
            "support_records_reserved": self.support_records_reserved,
            "acquisition_cost_reserved": self.acquisition_cost_reserved,
            "sealed_queries_reserved": self.sealed_queries_reserved,
            "motor_action_cost_reserved": self.motor_action_cost_reserved,
            "motor_resets_reserved": self.motor_resets_reserved,
            "sealed_query_ids": list(self.sealed_query_ids),
            "events": [event.to_dict() for event in self.events],
            "ledger_hash": self.ledger_hash,
        }


_COUNTED_RESOURCES = frozenset(
    {
        Grade3Resource.SUPPORT_RECORD,
        Grade3Resource.SEALED_QUERY,
        Grade3Resource.MOTOR_RESET,
    }
)


class Grade3EvaluationSession:
    """Thread-safe, fail-closed metering for one persistent Grade-3 process."""

    def __init__(
        self,
        manifest: Grade3SessionManifest,
        artifact_commitment: str,
        sdk_commitment: str,
    ) -> None:
        if not isinstance(manifest, Grade3SessionManifest):
            raise TypeError("manifest must be Grade3SessionManifest")
        self._manifest = manifest
        self._artifact = _digest(artifact_commitment, "artifact_commitment")
        self._sdk = _digest(sdk_commitment, "sdk_commitment")
        if self._artifact == self._sdk:
            raise ValueError("artifact and SDK commitments must be distinct")
        self._codebook: str | None = None
        self._checkpoint: str | None = None
        self._phase = Grade3SessionPhase.SUPPORT
        self._used = {resource: 0.0 for resource in Grade3Resource}
        self._reserved = {resource: 0.0 for resource in Grade3Resource}
        self._active: dict[int, Grade3Reservation] = {}
        self._next_reservation = 0
        self._query_ids: set[int] = set()
        self._events: list[Grade3SessionEvent] = []
        self._lock = RLock()
        self._manifest_hash = manifest_hash(asdict(manifest))
        self._wire_schema_hash = manifest_hash(grade3_schema_manifest())

    @property
    def manifest(self) -> Grade3SessionManifest:
        return self._manifest

    @property
    def phase(self) -> Grade3SessionPhase:
        return self._phase

    def _budget(self, resource: Grade3Resource) -> float:
        return {
            Grade3Resource.SUPPORT_RECORD: float(self._manifest.support_record_budget),
            Grade3Resource.ACQUISITION_COST: self._manifest.acquisition_cost_budget,
            Grade3Resource.SEALED_QUERY: float(self._manifest.query_budget),
            Grade3Resource.MOTOR_ACTION_COST: self._manifest.motor_action_cost_budget,
            Grade3Resource.MOTOR_RESET: float(self._manifest.motor_reset_budget),
        }[resource]

    def remaining(self, resource: Grade3Resource) -> float:
        with self._lock:
            resource = Grade3Resource(resource)
            return max(
                0.0,
                self._budget(resource) - self._used[resource] - self._reserved[resource],
            )

    @property
    def remaining_support_records(self) -> int:
        return int(self.remaining(Grade3Resource.SUPPORT_RECORD))

    @property
    def remaining_acquisition_cost(self) -> float:
        return self.remaining(Grade3Resource.ACQUISITION_COST)

    @property
    def remaining_queries(self) -> int:
        return int(self.remaining(Grade3Resource.SEALED_QUERY))

    @property
    def remaining_motor_action_cost(self) -> float:
        return self.remaining(Grade3Resource.MOTOR_ACTION_COST)

    @property
    def remaining_motor_resets(self) -> int:
        return int(self.remaining(Grade3Resource.MOTOR_RESET))

    def _append(
        self,
        kind: Grade3EventKind,
        operation: str,
        status: str,
        *,
        resource: Grade3Resource | None = None,
        amount: float = 0.0,
        reservation_id: int | None = None,
        request_hash: str | None = None,
        result_hash: str | None = None,
    ) -> Grade3SessionEvent:
        previous = self._events[-1].event_hash if self._events else _GENESIS_HASH
        material: dict[str, object] = {
            "protocol": GRADE3_LEDGER_PROTOCOL,
            "sequence": len(self._events),
            "kind": kind.value,
            "phase": self._phase.value,
            "operation": operation,
            "status": status,
            "resource": None if resource is None else resource.value,
            "amount": float(amount),
            "reservation_id": reservation_id,
            "request_hash": request_hash,
            "result_hash": result_hash,
            "previous_hash": previous,
        }
        event = Grade3SessionEvent(
            sequence=len(self._events),
            kind=kind,
            phase=self._phase,
            operation=operation,
            status=status,
            resource=resource,
            amount=float(amount),
            reservation_id=reservation_id,
            request_hash=request_hash,
            result_hash=result_hash,
            previous_hash=previous,
            event_hash=manifest_hash(material),
        )
        self._events.append(event)
        return event

    def _require_phase(self, *phases: Grade3SessionPhase) -> None:
        if self._phase not in phases:
            expected = ", ".join(phase.value for phase in phases)
            raise Grade3SessionStateError(
                f"operation requires phase {expected}; current phase is {self._phase.value}"
            )

    def _require_no_reservations(self) -> None:
        if self._active:
            raise Grade3SessionStateError(
                "phase changes require all active reservations to be consumed"
            )

    def commit_codebook(self, commitment: str) -> None:
        with self._lock:
            self._require_phase(Grade3SessionPhase.SUPPORT)
            if self._codebook is not None:
                raise Grade3SessionStateError("codebook is already committed")
            if self._events:
                raise Grade3SessionStateError("codebook must be committed before support begins")
            value = _digest(commitment, "codebook_commitment")
            if value in {self._artifact, self._sdk}:
                raise ValueError("codebook commitment must be independent")
            self._codebook = value
            self._append(
                Grade3EventKind.CODEBOOK_COMMIT,
                "commit_codebook",
                "completed",
                result_hash=value,
            )

    def begin_acquisition(self) -> None:
        with self._lock:
            self._require_phase(Grade3SessionPhase.SUPPORT)
            if self._codebook is None:
                raise Grade3SessionStateError("codebook commitment is missing")
            self._require_no_reservations()
            self._phase = Grade3SessionPhase.ACQUISITION
            self._append(
                Grade3EventKind.PHASE_CHANGE,
                "begin_acquisition",
                "completed",
            )

    def freeze(self, checkpoint_commitment: str) -> None:
        with self._lock:
            self._require_phase(Grade3SessionPhase.SUPPORT, Grade3SessionPhase.ACQUISITION)
            if self._codebook is None:
                raise Grade3SessionStateError("codebook commitment is missing")
            if self._checkpoint is not None:
                raise Grade3SessionStateError("checkpoint is already frozen")
            self._require_no_reservations()
            value = _digest(checkpoint_commitment, "checkpoint_commitment")
            self._checkpoint = value
            self._phase = Grade3SessionPhase.FROZEN_QUERY
            self._append(
                Grade3EventKind.FREEZE,
                "freeze",
                "completed",
                result_hash=value,
            )

    def _allowed_phase(self, resource: Grade3Resource) -> Grade3SessionPhase:
        return {
            Grade3Resource.SUPPORT_RECORD: Grade3SessionPhase.SUPPORT,
            Grade3Resource.ACQUISITION_COST: Grade3SessionPhase.ACQUISITION,
            Grade3Resource.SEALED_QUERY: Grade3SessionPhase.FROZEN_QUERY,
            Grade3Resource.MOTOR_ACTION_COST: Grade3SessionPhase.FROZEN_QUERY,
            Grade3Resource.MOTOR_RESET: Grade3SessionPhase.FROZEN_QUERY,
        }[resource]

    def reserve(
        self,
        resource: Grade3Resource,
        amount: int | float,
        operation: str,
        request_hash: str,
    ) -> Grade3Reservation:
        """Append a reservation before the caller performs any side effect."""

        with self._lock:
            resource = Grade3Resource(resource)
            self._require_phase(self._allowed_phase(resource))
            if self._codebook is None:
                raise Grade3SessionStateError("codebook commitment is missing")
            numeric = _amount(amount, "amount")
            if resource in _COUNTED_RESOURCES and not numeric.is_integer():
                raise ValueError(f"{resource.value} reservations must be integral")
            if not isinstance(operation, str) or not operation:
                raise TypeError("operation must be a nonempty string")
            request = _digest(request_hash, "request_hash")
            if numeric > self.remaining(resource) + 1e-12:
                raise Grade3SessionBudgetError(f"{resource.value} budget exhausted")
            reservation_id = self._next_reservation
            self._next_reservation += 1
            self._reserved[resource] += numeric
            event = self._append(
                Grade3EventKind.RESERVE,
                operation,
                "reserved",
                resource=resource,
                amount=numeric,
                reservation_id=reservation_id,
                request_hash=request,
            )
            reservation = Grade3Reservation(
                reservation_id,
                resource,
                numeric,
                operation,
                request,
                event.event_hash,
            )
            self._active[reservation_id] = reservation
            return reservation

    def reserve_query(self, query_id: int, operation: str, request_hash: str) -> Grade3Reservation:
        with self._lock:
            if isinstance(query_id, bool) or not isinstance(query_id, int):
                raise TypeError("query_id must be an integer")
            if query_id < 0:
                raise ValueError("query_id must be nonnegative")
            if query_id in self._query_ids:
                raise Grade3SessionStateError("query_id values cannot be reused")
            reservation = self.reserve(Grade3Resource.SEALED_QUERY, 1, operation, request_hash)
            # Registration occurs only after a successful reservation, but
            # remains permanent if the subsequent query fails.
            self._query_ids.add(query_id)
            return reservation

    def consume(
        self,
        reservation: Grade3Reservation,
        *,
        status: str,
        result_hash: str | None,
    ) -> Grade3SessionEvent:
        """Consume a reservation permanently, including failed operations."""

        with self._lock:
            if not isinstance(reservation, Grade3Reservation):
                raise TypeError("reservation must be Grade3Reservation")
            active = self._active.get(reservation.reservation_id)
            if active != reservation:
                raise Grade3SessionStateError("reservation is unknown or already consumed")
            if status not in {"completed", "abstained", "failed_consumed"}:
                raise ValueError("reservation status is not allowlisted")
            result = _optional_digest(result_hash, "result_hash")
            resource = reservation.resource
            self._reserved[resource] -= reservation.amount
            if isclose(self._reserved[resource], 0.0, abs_tol=1e-12):
                self._reserved[resource] = 0.0
            self._used[resource] += reservation.amount
            del self._active[reservation.reservation_id]
            return self._append(
                Grade3EventKind.CONSUME,
                reservation.operation,
                status,
                resource=resource,
                amount=reservation.amount,
                reservation_id=reservation.reservation_id,
                request_hash=reservation.request_hash,
                result_hash=result,
            )

    def record_operation(
        self,
        operation: str,
        status: str,
        *,
        request_hash: str | None = None,
        result_hash: str | None = None,
    ) -> Grade3SessionEvent:
        """Ledger an unmetered protocol step or decision transcript."""

        with self._lock:
            if not isinstance(operation, str) or not operation:
                raise TypeError("operation must be a nonempty string")
            if status not in {
                "completed",
                "abstained",
                "failed",
                "failed_consumed",
            }:
                raise ValueError("operation status is not allowlisted")
            return self._append(
                Grade3EventKind.OPERATION,
                operation,
                status,
                request_hash=_optional_digest(request_hash, "request_hash"),
                result_hash=_optional_digest(result_hash, "result_hash"),
            )

    def complete(self) -> Grade3AuditLedger:
        with self._lock:
            self._require_phase(Grade3SessionPhase.FROZEN_QUERY)
            self._require_no_reservations()
            self._phase = Grade3SessionPhase.COMPLETE
            self._append(Grade3EventKind.COMPLETE, "complete", "completed")
            return self.ledger

    @property
    def ledger(self) -> Grade3AuditLedger:
        with self._lock:

            def count(resource: Grade3Resource, source: dict[Grade3Resource, float]) -> int:
                return int(source[resource])

            return Grade3AuditLedger(
                manifest_hash=self._manifest_hash,
                wire_schema_hash=self._wire_schema_hash,
                artifact_commitment=self._artifact,
                sdk_commitment=self._sdk,
                codebook_commitment=self._codebook,
                checkpoint_commitment=self._checkpoint,
                phase=self._phase,
                support_records_used=count(Grade3Resource.SUPPORT_RECORD, self._used),
                acquisition_cost_used=self._used[Grade3Resource.ACQUISITION_COST],
                sealed_queries_used=count(Grade3Resource.SEALED_QUERY, self._used),
                motor_action_cost_used=self._used[Grade3Resource.MOTOR_ACTION_COST],
                motor_resets_used=count(Grade3Resource.MOTOR_RESET, self._used),
                support_records_reserved=count(Grade3Resource.SUPPORT_RECORD, self._reserved),
                acquisition_cost_reserved=self._reserved[Grade3Resource.ACQUISITION_COST],
                sealed_queries_reserved=count(Grade3Resource.SEALED_QUERY, self._reserved),
                motor_action_cost_reserved=self._reserved[Grade3Resource.MOTOR_ACTION_COST],
                motor_resets_reserved=count(Grade3Resource.MOTOR_RESET, self._reserved),
                sealed_query_ids=tuple(sorted(self._query_ids)),
                events=tuple(self._events),
            )

    @property
    def commitments(self) -> MappingProxyType[str, str | None]:
        return MappingProxyType(
            {
                "artifact": self._artifact,
                "sdk": self._sdk,
                "codebook": self._codebook,
                "checkpoint": self._checkpoint,
            }
        )


# Short aliases match the established v1 session vocabulary while keeping the
# public Grade-3 names explicit for callers importing both protocols.
SessionStateError = Grade3SessionStateError
SessionBudgetError = Grade3SessionBudgetError


__all__ = [
    "GRADE3_LEDGER_PROTOCOL",
    "Grade3AuditLedger",
    "Grade3EvaluationSession",
    "Grade3EventKind",
    "Grade3Reservation",
    "Grade3Resource",
    "Grade3SessionBudgetError",
    "Grade3SessionEvent",
    "Grade3SessionPhase",
    "Grade3SessionStateError",
    "SessionBudgetError",
    "SessionStateError",
]
