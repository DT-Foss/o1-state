"""Evaluator-only ProcessWorld adapter for finite interventional semantics."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256

from .active import sensory_trace_consequence
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
    """Private quotient-compatibility result plus a redacted audit digest."""

    model: FiniteInterventionalModel
    groundability: PredicateGroundability
    profile_hash: str
    declared_interventions: tuple[str, ...]
    problem_commitment: str
    policy_commitment: str
    target_commitment: str

    @property
    def quotient_compatible(self) -> bool:
        """Necessary finite identifiability condition, not a learning claim."""

        return self.groundability.quotient_compatible

    @property
    def groundable(self) -> bool:
        """Backward-compatible alias for :attr:`quotient_compatible`."""

        return self.quotient_compatible

    def redacted_manifest(self) -> dict[str, object]:
        """Return certificate material without latent labels or witness values."""

        return {
            "profile_hash": self.profile_hash,
            "problem_commitment": self.problem_commitment,
            "policy_commitment": self.policy_commitment,
            "target_commitment": self.target_commitment,
            "declared_interventions": list(self.declared_interventions),
            "quotient_class_sizes": [
                len(block) for block in self.groundability.quotient
            ],
            "quotient_compatible": self.quotient_compatible,
            "groundable_compatibility_alias": self.groundable,
            "witness_present": self.groundability.witness is not None,
        }


def process_pair_quotient(
    left: OstensiveRecord,
    right: OstensiveRecord,
    *,
    commitment_nonce: str = "",
) -> ProcessQuotientReport:
    """Build the finite quotient for one matched public ProcessWorld pair.

    The declared intervention family contains passive observation and the
    pair's matched diagnostic action sequence.  Consequences are digests of
    outcome-code-free, feedback-stripped public records.  This is a scoped
    finite result; it is not a proof over actions absent from the declaration.
    """

    if not isinstance(left, OstensiveRecord) or not isinstance(right, OstensiveRecord):
        raise TypeError("process_pair_quotient expects two OstensiveRecord values")
    if not isinstance(commitment_nonce, str):
        raise TypeError("commitment_nonce must be a string")
    if left.token != right.token:
        raise ValueError("a matched quotient pair must share one opaque token")
    if left.task_feedback == right.task_feedback:
        raise ValueError("a matched quotient pair must contain opposite task feedback")
    if not left.episode.transitions or not right.episode.transitions:
        raise ValueError("a matched quotient pair cannot contain empty episodes")
    if len(left.episode.transitions) != len(right.episode.transitions):
        raise ValueError("matched quotient episodes must have equal lengths")
    action_skeletons = tuple(
        tuple((step.action.code, tuple(step.action.vector)) for step in record.episode.transitions)
        for record in (left, right)
    )
    if action_skeletons[0] != action_skeletons[1]:
        raise ValueError("matched quotient episodes must use the same opaque action schema")
    observation_shapes = {
        step.before.shape
        for record in (left, right)
        for step in record.episode.transitions
    } | {
        step.after.shape
        for record in (left, right)
        for step in record.episode.transitions
    }
    if len(observation_shapes) != 1:
        raise ValueError("matched quotient episodes must share one sensor schema")
    histories = ("matched-history-0", "matched-history-1")
    interventions = (OBSERVE_INTERVENTION, DECLARED_POLICY_INTERVENTION)
    records = (left, right)
    distributions: dict[tuple[str, str], dict[str, float]] = {}
    redacted_rows: list[dict[str, str]] = []
    for history, record in zip(histories, records, strict=True):
        trace = episode_to_public_trace(record.episode, strip_feedback=True)
        observation_outcome = "observation:" + _digest(trace.initial)
        trace_outcome = "sensory-trace:" + sensory_trace_consequence(trace).digest
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
    problem_commitment = manifest_hash(
        {
            "rows": redacted_rows,
            "interventions": list(interventions),
        }
    )
    policy_commitment = manifest_hash(
        {
            "action_skeleton": [
                {"code": code, "vector": list(vector)}
                for code, vector in action_skeletons[0]
            ],
            "target_instantiations": [
                [list(step.action.target) for step in record.episode.transitions]
                for record in records
            ],
            "cost_per_action": 1.0,
        }
    )
    target_commitment = manifest_hash(
        {
            "nonce": commitment_nonce,
            "opaque_token": left.token,
            "target": [
                {"history": history, "value": target[history]} for history in histories
            ],
        }
    )
    profile_hash = manifest_hash(
        {
            "problem_commitment": problem_commitment,
            "policy_commitment": policy_commitment,
            "target_commitment": target_commitment,
        }
    )
    return ProcessQuotientReport(
        model=model,
        groundability=report,
        profile_hash=profile_hash,
        declared_interventions=interventions,
        problem_commitment=problem_commitment,
        policy_commitment=policy_commitment,
        target_commitment=target_commitment,
    )


__all__ = [
    "DECLARED_POLICY_INTERVENTION",
    "OBSERVE_INTERVENTION",
    "ProcessQuotientReport",
    "process_pair_quotient",
]
