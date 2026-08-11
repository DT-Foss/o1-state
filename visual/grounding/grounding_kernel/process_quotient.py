"""Evaluator-only ProcessWorld adapter for finite interventional semantics."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256

from .certificates import manifest_hash
from .processworld import OstensiveRecord
from .quotient import FiniteInterventionalModel, PredicateGroundability
from .v1_adapters import episode_to_public_trace
from .v1_wire import encode_message


OBSERVE_INTERVENTION = "observe-v1"
DECLARED_POLICY_INTERVENTION = "declared-policy-v1"


def _digest(value: object) -> str:
    if hasattr(value, "pixels") and hasattr(value, "digest"):
        return str(value.digest())
    return sha256(encode_message(value).strip()).hexdigest()


@dataclass(frozen=True, slots=True)
class ProcessQuotientReport:
    """Private groundability result plus a redacted certificate digest."""

    model: FiniteInterventionalModel
    groundability: PredicateGroundability
    profile_hash: str
    declared_interventions: tuple[str, ...]

    @property
    def groundable(self) -> bool:
        return self.groundability.groundable

    def redacted_manifest(self) -> dict[str, object]:
        """Return certificate material without latent labels or witness values."""

        return {
            "profile_hash": self.profile_hash,
            "declared_interventions": list(self.declared_interventions),
            "quotient_class_sizes": [
                len(block) for block in self.groundability.quotient
            ],
            "groundable": self.groundable,
            "witness_present": self.groundability.witness is not None,
        }


def process_pair_quotient(
    left: OstensiveRecord,
    right: OstensiveRecord,
) -> ProcessQuotientReport:
    """Build the finite quotient for one matched public ProcessWorld pair.

    The declared intervention family contains passive observation and the
    pair's matched diagnostic action sequence.  Consequences are digests of
    outcome-code-free, feedback-stripped public records.  This is a scoped
    finite result; it is not a proof over actions absent from the declaration.
    """

    if not isinstance(left, OstensiveRecord) or not isinstance(right, OstensiveRecord):
        raise TypeError("process_pair_quotient expects two OstensiveRecord values")
    histories = ("matched-history-0", "matched-history-1")
    interventions = (OBSERVE_INTERVENTION, DECLARED_POLICY_INTERVENTION)
    records = (left, right)
    distributions: dict[tuple[str, str], dict[str, float]] = {}
    redacted_rows: list[dict[str, str]] = []
    for history, record in zip(histories, records, strict=True):
        trace = episode_to_public_trace(record.episode, strip_feedback=True)
        observation_outcome = "observation:" + _digest(trace.initial)
        trace_outcome = "trace:" + _digest(trace)
        distributions[(history, OBSERVE_INTERVENTION)] = {observation_outcome: 1.0}
        distributions[(history, DECLARED_POLICY_INTERVENTION)] = {trace_outcome: 1.0}
        redacted_rows.extend(
            (
                {
                    "history": history,
                    "intervention": OBSERVE_INTERVENTION,
                    "consequence_hash": sha256(observation_outcome.encode()).hexdigest(),
                },
                {
                    "history": history,
                    "intervention": DECLARED_POLICY_INTERVENTION,
                    "consequence_hash": sha256(trace_outcome.encode()).hexdigest(),
                },
            )
        )
    model = FiniteInterventionalModel(histories, interventions, distributions)
    target = {
        histories[0]: bool(left.task_feedback),
        histories[1]: bool(right.task_feedback),
    }
    report = model.predicate_groundability(target)
    return ProcessQuotientReport(
        model=model,
        groundability=report,
        profile_hash=manifest_hash(redacted_rows),
        declared_interventions=interventions,
    )


__all__ = [
    "DECLARED_POLICY_INTERVENTION",
    "OBSERVE_INTERVENTION",
    "ProcessQuotientReport",
    "process_pair_quotient",
]
