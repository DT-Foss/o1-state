"""Capability-separated protocols for agents and evaluators.

The important boundary is not an underscore convention: benchmark code hands
the candidate only an :class:`AgentEnvironment`.  The evaluator retains the
separate :class:`EvaluatorOracle` capability and never serialises it into a
learner trajectory.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Protocol, runtime_checkable

from .contracts import (
    Action,
    ActionKind,
    AgentManifest,
    Observation,
    OracleSnapshot,
    OutcomeKind,
    Pixel,
    PredicateKind,
    Trajectory,
    Transition,
)


@runtime_checkable
class AgentEnvironment(Protocol):
    """The complete and intentionally narrow learner-facing API."""

    @property
    def manifest(self) -> AgentManifest: ...

    @property
    def action_codes(self) -> tuple[int, ...]: ...

    @property
    def symbol_codes(self) -> tuple[int, ...]: ...

    def reset(self) -> Observation: ...

    def observe(self) -> Observation: ...

    def step(self, action: Action) -> Transition: ...

    def trajectory(self) -> Trajectory: ...


@runtime_checkable
class EvaluatorOracle(Protocol):
    """Privileged evaluator capability; never pass this object to a learner."""

    @property
    def manifest(self) -> AgentManifest: ...

    def snapshot(self) -> OracleSnapshot: ...

    def decode_action(self, code: int) -> ActionKind: ...

    def decode_outcome(self, code: int) -> OutcomeKind: ...

    def decode_symbol(self, code: int) -> PredicateKind: ...

    def encode_symbol(self, predicate: PredicateKind) -> int: ...

    def predicate(self, object_id: int, predicate: int | PredicateKind) -> bool: ...

    def object_at(self, target: Pixel) -> int | None: ...

    def object_center(self, object_id: int) -> Pixel: ...

    def intervention_signature(self, object_id: int) -> Mapping[ActionKind, OutcomeKind]: ...

    def negative_control_invariant(
        self,
        actions: Iterable[Action] = (),
        *,
        depth: int = 2,
        max_transitions: int = 100_000,
    ) -> bool: ...


_FORBIDDEN_AGENT_PUBLIC_NAMES = frozenset(
    {
        "seed",
        "rng",
        "random_state",
        "oracle",
        "codebook",
        "decode_action",
        "decode_outcome",
        "decode_symbol",
        "encode_symbol",
        "latent",
        "latents",
        "objects",
        "object_ids",
        "snapshot",
        "predicate",
    }
)


def audit_agent_boundary(environment: AgentEnvironment) -> tuple[str, ...]:
    """Return forbidden public capabilities found on an agent environment.

    This is a regression guard for accidental API leaks, not a substitute for
    the benchmark's process boundary.  Python reflection is not a security
    sandbox; the sealed evaluator must still execute untrusted candidates in a
    separate process and pass only serialised agent records.
    """

    public = {name for name in dir(environment) if not name.startswith("_")}
    return tuple(sorted(public & _FORBIDDEN_AGENT_PUBLIC_NAMES))
