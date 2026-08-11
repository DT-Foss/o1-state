"""Perception-bound, closed-loop execution of grounded language programs.

Unlike the legacy ``TargetTrack`` reference, a target meaning here stores no
image coordinate.  It stores an auditable visual prototype, an outcome-free
episode classifier and an opaque diagnostic action schema.  At execution time
the policy finds all matching candidates in the current frame, intervenes on
each from a reset state, selects the unique candidate with the requested
operational membership, and only then runs the utterance's action schema.
"""

from __future__ import annotations

from collections.abc import Hashable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from hashlib import sha256
from types import MappingProxyType

from .contracts import Action
from .language import GroundedReferent, OperationalMeaning, Resolution
from .perceptual_policy import (
    CandidateEvidence,
    EpisodeMembershipModel,
    PublicResetStepEnvironment,
    VisualTargetSelector,
)
from .programs import ActionScheme, ProgramSchema
from .v1_contracts import PublicTrace, PublicTransition


def _digest_text(value: object) -> str:
    return sha256(repr(value).encode("utf-8")).hexdigest()


def _binder_digest(binder: object) -> str:
    manifest = getattr(binder, "manifest", None)
    digest = getattr(manifest, "digest", None)
    if not callable(digest):
        raise TypeError("binder must expose manifest.digest()")
    value = digest()
    if not isinstance(value, str) or len(value) != 64:
        raise ValueError("binder manifest digest must be a SHA-256 hex string")
    return value


@dataclass(frozen=True, slots=True)
class PerceptualTargetRole:
    """Hashable operational target meaning backed by learner-visible evidence."""

    selector: VisualTargetSelector
    token: Hashable
    diagnostic_scheme: ActionScheme
    required_membership: bool
    binder_digest: str
    support_commitment: str
    binder: EpisodeMembershipModel = field(compare=False, hash=False, repr=False)

    def __post_init__(self) -> None:
        if not isinstance(self.selector, VisualTargetSelector):
            raise TypeError("selector must be VisualTargetSelector")
        try:
            hash(self.token)
        except TypeError as exc:
            raise TypeError("target-role token must be hashable") from exc
        if not isinstance(self.diagnostic_scheme, ActionScheme):
            raise TypeError("diagnostic_scheme must be ActionScheme")
        if not isinstance(self.required_membership, bool):
            raise TypeError("required_membership must be Boolean")
        if not isinstance(self.binder, EpisodeMembershipModel):
            raise TypeError("binder must expose supports_token")
        actual = _binder_digest(self.binder)
        if self.binder_digest != actual:
            raise ValueError("binder_digest does not bind the supplied binder")
        for name in ("binder_digest", "support_commitment"):
            value = getattr(self, name)
            if not isinstance(value, str) or len(value) != 64:
                raise ValueError(f"{name} must be a SHA-256 hex digest")
            try:
                bytes.fromhex(value)
            except ValueError as exc:
                raise ValueError(f"{name} must be a SHA-256 hex digest") from exc

    @classmethod
    def from_support(
        cls,
        *,
        selector: VisualTargetSelector,
        binder: EpisodeMembershipModel,
        token: Hashable,
        diagnostic_scheme: ActionScheme,
        required_membership: bool,
        evidence_digests: Iterable[str],
    ) -> "PerceptualTargetRole":
        evidence = tuple(sorted(evidence_digests))
        if not evidence:
            raise ValueError("target roles require public support evidence")
        if any(not isinstance(value, str) or len(value) != 64 for value in evidence):
            raise ValueError("evidence_digests must be SHA-256 hex strings")
        commitment = _digest_text(
            (
                selector,
                token,
                diagnostic_scheme,
                required_membership,
                evidence,
                _binder_digest(binder),
            )
        )
        return cls(
            selector,
            token,
            diagnostic_scheme,
            required_membership,
            _binder_digest(binder),
            commitment,
            binder,
        )


@dataclass(frozen=True, slots=True)
class ClosedLoopExecution:
    status: Resolution
    trace: PublicTrace | None
    evidence: tuple[CandidateEvidence, ...]
    actions_executed: int
    proof_digest: str

    @property
    def resolved(self) -> bool:
        return self.status is Resolution.RESOLVED


def _run_scheme(
    environment: PublicResetStepEnvironment,
    selector: VisualTargetSelector,
    rank: int,
    scheme: ActionScheme,
) -> PublicTrace:
    initial = environment.reset()
    candidates = selector.candidates(initial)
    if not 0 <= rank < len(candidates):
        raise LookupError("selected visual candidate is absent after reset")
    target = candidates[rank]
    previous = initial
    transitions: list[PublicTransition] = []
    for index, (code, vector) in enumerate(scheme.steps):
        if index:
            target = selector.track(previous, target)
        action = Action(code, target, vector)
        raw = environment.step(action)
        before = getattr(raw, "before")
        after = getattr(raw, "after")
        if before != previous:
            raise RuntimeError("environment returned a discontinuous transition")
        transitions.append(PublicTransition(before, action, after, None))
        previous = after
    return PublicTrace(initial, tuple(transitions))


class ClosedLoopProgramExecutor:
    """Resolve a perceptual role and execute the independently chosen scheme."""

    def __init__(self, schema: ProgramSchema) -> None:
        if not isinstance(schema, ProgramSchema):
            raise TypeError("schema must be ProgramSchema")
        self.schema = schema

    def execute(
        self,
        environment: PublicResetStepEnvironment,
        referent: GroundedReferent,
    ) -> ClosedLoopExecution:
        if not isinstance(environment, PublicResetStepEnvironment):
            raise TypeError("environment must expose reset() and step(Action)")
        if not isinstance(referent, GroundedReferent):
            raise TypeError("referent must be GroundedReferent")
        role_meaning = referent.meaning_for(self.schema.target_type_id)
        scheme_meaning = referent.meaning_for(self.schema.scheme_type_id)
        if role_meaning is None or scheme_meaning is None:
            raise ValueError("referent is missing a required program slot")
        if not isinstance(role_meaning.value, PerceptualTargetRole):
            raise TypeError("target slot must contain PerceptualTargetRole")
        if not isinstance(scheme_meaning.value, ActionScheme):
            raise TypeError("scheme slot must contain ActionScheme")
        role = role_meaning.value

        initial = environment.reset()
        candidates = role.selector.candidates(initial)
        evidence: list[CandidateEvidence] = []
        for rank, target in enumerate(candidates):
            trace = _run_scheme(
                environment,
                role.selector,
                rank,
                role.diagnostic_scheme,
            ).feedback_stripped()
            prediction = role.binder.supports_token(trace, role.token)
            if prediction not in (True, False, None):
                raise TypeError("binder prediction must be bool or None")
            evidence.append(CandidateEvidence(target, prediction, trace))
        selected = [
            rank
            for rank, item in enumerate(evidence)
            if item.prediction is role.required_membership
        ]
        probe_actions = len(evidence) * len(role.diagnostic_scheme.steps)
        if len(selected) != 1:
            proof = _digest_text(
                (
                    "unresolved-target-role",
                    role.support_commitment,
                    tuple((item.initial_target, item.prediction) for item in evidence),
                )
            )
            status = Resolution.AMBIGUOUS if len(selected) > 1 else Resolution.UNKNOWN
            return ClosedLoopExecution(status, None, tuple(evidence), probe_actions, proof)
        final = _run_scheme(
            environment,
            role.selector,
            selected[0],
            scheme_meaning.value,
        ).feedback_stripped()
        proof = _digest_text(
            (
                "closed-loop-grounded-program",
                role.support_commitment,
                role.binder_digest,
                selected[0],
                tuple((item.initial_target, item.prediction) for item in evidence),
                scheme_meaning.value,
                tuple(step.after.digest() for step in final.transitions),
            )
        )
        return ClosedLoopExecution(
            Resolution.RESOLVED,
            final,
            tuple(evidence),
            probe_actions + len(scheme_meaning.value.steps),
            proof,
        )


@dataclass(frozen=True, slots=True)
class ProgramRecognition:
    status: Resolution
    referent: GroundedReferent | None
    target_candidates: tuple[PerceptualTargetRole, ...]
    scheme_candidates: tuple[ActionScheme, ...]
    proof_digest: str

    @property
    def resolved(self) -> bool:
        return self.status is Resolution.RESOLVED


class ClosedLoopProgramRecognizer:
    """Recognize a grounded program from a feedback-free raw public trace."""

    def __init__(
        self,
        schema: ProgramSchema,
        roles: Sequence[PerceptualTargetRole],
        schemes: Sequence[ActionScheme],
    ) -> None:
        if not isinstance(schema, ProgramSchema):
            raise TypeError("schema must be ProgramSchema")
        role_values = tuple(roles)
        scheme_values = tuple(schemes)
        if not role_values or not all(
            isinstance(value, PerceptualTargetRole) for value in role_values
        ):
            raise ValueError("recognizer requires perceptual target roles")
        if not scheme_values or not all(
            isinstance(value, ActionScheme) for value in scheme_values
        ):
            raise ValueError("recognizer requires action schemes")
        if len(set(role_values)) != len(role_values) or len(set(scheme_values)) != len(
            scheme_values
        ):
            raise ValueError("recognizer roles and schemes must be unique")
        self.schema = schema
        self.roles = role_values
        self.schemes = scheme_values

    def recognize(self, trace: PublicTrace) -> ProgramRecognition:
        if not isinstance(trace, PublicTrace):
            raise TypeError("recognize expects PublicTrace")
        query = trace.feedback_stripped()
        role_candidates = tuple(
            role
            for role in self.roles
            if role.binder.supports_token(query, role.token)
            is role.required_membership
        )
        observed_scheme = ActionScheme(
            tuple(
                (transition.action.code, tuple(transition.action.vector))
                for transition in query.transitions
            )
        )
        scheme_candidates = tuple(
            scheme for scheme in self.schemes if scheme == observed_scheme
        )
        proof = _digest_text(
            (
                "trace-to-operational-program",
                tuple(role.support_commitment for role in role_candidates),
                scheme_candidates,
                tuple(step.after.digest() for step in query.transitions),
            )
        )
        if len(role_candidates) != 1 or len(scheme_candidates) != 1:
            status = (
                Resolution.AMBIGUOUS
                if len(role_candidates) > 1 or len(scheme_candidates) > 1
                else Resolution.UNKNOWN
            )
            return ProgramRecognition(
                status,
                None,
                role_candidates,
                scheme_candidates,
                proof,
            )
        referent = build_closed_loop_referent(
            self.schema,
            role_candidates[0],
            scheme_candidates[0],
        )
        return ProgramRecognition(
            Resolution.RESOLVED,
            referent,
            role_candidates,
            scheme_candidates,
            proof,
        )


def build_closed_loop_referent(
    schema: ProgramSchema,
    role: PerceptualTargetRole,
    scheme: ActionScheme,
) -> GroundedReferent:
    if not isinstance(schema, ProgramSchema):
        raise TypeError("schema must be ProgramSchema")
    if not isinstance(role, PerceptualTargetRole):
        raise TypeError("role must be PerceptualTargetRole")
    if not isinstance(scheme, ActionScheme):
        raise TypeError("scheme must be ActionScheme")
    return GroundedReferent(
        (
            OperationalMeaning(schema.target_type_id, role),
            OperationalMeaning(schema.scheme_type_id, scheme),
        )
    )


def referent_registry(
    schema: ProgramSchema,
    token_roles: Mapping[Hashable, PerceptualTargetRole],
    token_schemes: Mapping[Hashable, ActionScheme],
) -> Mapping[tuple[Hashable, Hashable], GroundedReferent]:
    """Factor all public role×scheme combinations without utterance lookup."""

    values = {
        (role_token, scheme_token): build_closed_loop_referent(schema, role, scheme)
        for role_token, role in token_roles.items()
        for scheme_token, scheme in token_schemes.items()
    }
    return MappingProxyType(values)


__all__ = [
    "ClosedLoopExecution",
    "ClosedLoopProgramExecutor",
    "ClosedLoopProgramRecognizer",
    "PerceptualTargetRole",
    "ProgramRecognition",
    "build_closed_loop_referent",
    "referent_registry",
]
