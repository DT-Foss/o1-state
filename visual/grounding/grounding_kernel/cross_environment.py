"""Evaluator-side operational alignment across independent environments.

Alignment is computed from complete finite intervention/consequence ledgers.
Opaque action, outcome, and token integers may all be renamed, and raw RGB
values may be rendered differently.  The matcher uses only the relational
structure of programs, outcome co-occurrence, and equality/change relations
among observations.  Exact identifiers, coordinates, and pixel values remain
in the raw audit hash but are excluded from the structural decision.

The result is a certificate about this declared finite probe suite.  It is not
a claim of universal semantics or transfer to an untested physical system.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from enum import Enum
from hashlib import sha256
from itertools import permutations, product
import json
from math import factorial
from types import MappingProxyType
from typing import Callable, Protocol, runtime_checkable

import numpy as np

from .contracts import Trajectory


__all__ = [
    "AlignmentReport",
    "AlignmentStatus",
    "CallbackProbeOracle",
    "OperationalLedger",
    "OperationalManifest",
    "OperationalProbeOracle",
    "ProbeManifest",
    "RawProbeRecord",
    "RawStepEntry",
    "TokenAlignment",
    "TransferDecision",
    "TransferGateResult",
    "align_environments",
    "capture_operational_ledger",
    "gate_transfer",
]


LEDGER_VERSION = "independent-operational-ledger/1"
MAX_LEDGER_RECORDS = 100_000
MAX_CANONICAL_ACTIONS = 5
MAX_CANONICAL_TOKENS = 5
MAX_CANONICAL_WORK = 5_000_000


def _strict_int(value: object, field_name: str, *, minimum: int | None = None) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise TypeError(f"{field_name} must be an integer")
    result = int(value)
    if minimum is not None and result < minimum:
        raise ValueError(f"{field_name} must be at least {minimum}")
    return result


def _codes(values: Sequence[object], field_name: str) -> tuple[int, ...]:
    result = tuple(_strict_int(value, field_name) for value in values)
    if not result or len(set(result)) != len(result):
        raise ValueError(f"{field_name} must be non-empty and unique")
    return tuple(sorted(result))


def _validate_digest(value: str, field_name: str) -> str:
    if not isinstance(value, str) or len(value) != 64:
        raise ValueError(f"{field_name} must be a lowercase SHA-256 hex digest")
    try:
        decoded = bytes.fromhex(value)
    except ValueError as error:
        raise ValueError(f"{field_name} must be a lowercase SHA-256 hex digest") from error
    if len(decoded) != 32 or value != value.lower():
        raise ValueError(f"{field_name} must be a lowercase SHA-256 hex digest")
    return value


def _json_hash(payload: object) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    return sha256(encoded).hexdigest()


def _pixel_hash(pixels: np.ndarray) -> str:
    frame = np.asarray(pixels)
    header = json.dumps(
        {"dtype": frame.dtype.str, "shape": list(frame.shape)},
        sort_keys=True,
        separators=(",", ":"),
    ).encode("ascii")
    return sha256(header + b"\x00" + frame.tobytes(order="C")).hexdigest()


@runtime_checkable
class OperationalManifest(Protocol):
    """Opaque vocabulary and budget required from any source environment."""

    @property
    def action_codes(self) -> tuple[int, ...]: ...

    @property
    def outcome_codes(self) -> tuple[int, ...]: ...

    @property
    def token_codes(self) -> tuple[int, ...]: ...

    @property
    def max_steps(self) -> int: ...


@dataclass(frozen=True, slots=True)
class ProbeManifest:
    """Generic adapter manifest containing no renderer or semantic metadata."""

    action_codes: tuple[int, ...]
    outcome_codes: tuple[int, ...]
    token_codes: tuple[int, ...]
    max_steps: int

    def __post_init__(self) -> None:
        actions = _codes(self.action_codes, "action_codes")
        outcomes = _codes(self.outcome_codes, "outcome_codes")
        tokens = _codes(self.token_codes, "token_codes")
        if len(set(actions + outcomes + tokens)) != len(actions + outcomes + tokens):
            raise ValueError("opaque adapter alphabets must be globally disjoint")
        object.__setattr__(self, "action_codes", actions)
        object.__setattr__(self, "outcome_codes", outcomes)
        object.__setattr__(self, "token_codes", tokens)
        object.__setattr__(self, "max_steps", _strict_int(self.max_steps, "max_steps", minimum=1))


ProbeRunner = Callable[[int, tuple[int, ...]], Trajectory]


@dataclass(frozen=True, slots=True)
class CallbackProbeOracle:
    """Protocol-only adapter for evaluator-owned public intervention runners.

    The callback receives only an opaque token and opaque action program.  An
    evaluator can therefore adapt another simulator without exposing its
    semantic enums, latent identifiers, renderer, or decoder to this module.
    """

    manifest: ProbeManifest
    _runner: ProbeRunner = field(repr=False, compare=False)

    def __post_init__(self) -> None:
        if not isinstance(self.manifest, ProbeManifest):
            raise TypeError("manifest must be a ProbeManifest")
        if not callable(self._runner):
            raise TypeError("runner must be callable")

    def run_probe(self, token: int, program: Sequence[int]) -> Trajectory:
        token_code = _strict_int(token, "token")
        codes = tuple(_strict_int(code, "program") for code in program)
        if token_code not in self.manifest.token_codes:
            raise KeyError(f"unknown token code: {token_code}")
        if not codes or len(codes) > self.manifest.max_steps:
            raise ValueError("program must be non-empty and within the step budget")
        if any(code not in self.manifest.action_codes for code in codes):
            raise KeyError("program contains an unknown action code")
        trace = self._runner(token_code, codes)
        if not isinstance(trace, Trajectory):
            raise TypeError("runner must return a Trajectory")
        if tuple(step.action.code for step in trace.transitions) != codes:
            raise ValueError("runner trajectory does not match the requested action program")
        return trace


@runtime_checkable
class OperationalProbeOracle(Protocol):
    """Minimal evaluator capability consumed by :func:`capture_operational_ledger`."""

    @property
    def manifest(self) -> OperationalManifest: ...

    def run_probe(self, token: int, program: Sequence[int]) -> Trajectory: ...


@dataclass(frozen=True, slots=True)
class RawStepEntry:
    """One exact opaque interaction plus hashes of its raw RGB endpoints."""

    action_code: int
    target: tuple[int, int]
    vector: tuple[int, int]
    outcome_code: int
    before_observation_hash: str
    after_observation_hash: str
    before_pixels_hash: str
    after_pixels_hash: str
    pixels_changed: bool

    def __post_init__(self) -> None:
        object.__setattr__(self, "action_code", _strict_int(self.action_code, "action_code"))
        object.__setattr__(self, "outcome_code", _strict_int(self.outcome_code, "outcome_code"))
        target = tuple(_strict_int(value, "target") for value in self.target)
        vector = tuple(_strict_int(value, "vector") for value in self.vector)
        if len(target) != 2 or len(vector) != 2:
            raise ValueError("target and vector must each contain two integers")
        object.__setattr__(self, "target", target)
        object.__setattr__(self, "vector", vector)
        for name in (
            "before_observation_hash",
            "after_observation_hash",
            "before_pixels_hash",
            "after_pixels_hash",
        ):
            object.__setattr__(self, name, _validate_digest(getattr(self, name), name))
        changed = bool(self.pixels_changed)
        if changed != (self.before_pixels_hash != self.after_pixels_hash):
            raise ValueError("pixels_changed must agree with the raw pixel hashes")
        object.__setattr__(self, "pixels_changed", changed)

    def to_dict(self) -> dict[str, object]:
        return {
            "action_code": self.action_code,
            "target": list(self.target),
            "vector": list(self.vector),
            "outcome_code": self.outcome_code,
            "before_observation_hash": self.before_observation_hash,
            "after_observation_hash": self.after_observation_hash,
            "before_pixels_hash": self.before_pixels_hash,
            "after_pixels_hash": self.after_pixels_hash,
            "pixels_changed": self.pixels_changed,
        }


@dataclass(frozen=True, slots=True)
class RawProbeRecord:
    """Complete raw record for one token-conditioned opaque action program."""

    token: int
    program: tuple[int, ...]
    steps: tuple[RawStepEntry, ...]

    def __post_init__(self) -> None:
        token = _strict_int(self.token, "token")
        program = tuple(_strict_int(code, "program") for code in self.program)
        steps = tuple(self.steps)
        if not program or len(program) != len(steps):
            raise ValueError("program and steps must be non-empty and aligned")
        if not all(isinstance(step, RawStepEntry) for step in steps):
            raise TypeError("steps must contain only RawStepEntry values")
        if tuple(step.action_code for step in steps) != program:
            raise ValueError("program must equal the recorded action-code sequence")
        for left, right in zip(steps, steps[1:], strict=False):
            if left.after_observation_hash != right.before_observation_hash:
                raise ValueError("observation hashes are discontinuous within the probe")
            if left.after_pixels_hash != right.before_pixels_hash:
                raise ValueError("pixel hashes are discontinuous within the probe")
        object.__setattr__(self, "token", token)
        object.__setattr__(self, "program", program)
        object.__setattr__(self, "steps", steps)

    @property
    def frame_hashes(self) -> tuple[str, ...]:
        return (self.steps[0].before_pixels_hash,) + tuple(
            step.after_pixels_hash for step in self.steps
        )

    @property
    def outcome_trace(self) -> tuple[int, ...]:
        return tuple(step.outcome_code for step in self.steps)

    @property
    def change_trace(self) -> tuple[bool, ...]:
        return tuple(step.pixels_changed for step in self.steps)

    def to_dict(self) -> dict[str, object]:
        return {
            "token": self.token,
            "program": list(self.program),
            "steps": [step.to_dict() for step in self.steps],
        }


@dataclass(frozen=True, slots=True)
class OperationalLedger:
    """Complete, immutable probe ledger with raw and structural commitments."""

    action_codes: tuple[int, ...]
    outcome_codes: tuple[int, ...]
    token_codes: tuple[int, ...]
    program_length: int
    records: tuple[RawProbeRecord, ...]
    version: str = LEDGER_VERSION
    _raw_hash: str = field(init=False, repr=False, compare=False)
    _structural_hash: str = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        if self.version != LEDGER_VERSION:
            raise ValueError(f"version must equal {LEDGER_VERSION!r}")
        actions = _codes(self.action_codes, "action_codes")
        outcomes = _codes(self.outcome_codes, "outcome_codes")
        tokens = _codes(self.token_codes, "token_codes")
        if len(actions) > MAX_CANONICAL_ACTIONS or len(tokens) > MAX_CANONICAL_TOKENS:
            raise ValueError(
                "opaque alphabets exceed the bounded canonical-alignment implementation"
            )
        if len(set(actions + outcomes + tokens)) != len(actions + outcomes + tokens):
            raise ValueError("opaque code alphabets must be globally disjoint")
        length = _strict_int(self.program_length, "program_length", minimum=1)
        expected_count = len(tokens) * (len(actions) ** length)
        if expected_count > MAX_LEDGER_RECORDS:
            raise ValueError(f"ledger would exceed {MAX_LEDGER_RECORDS} records")
        canonical_work = factorial(len(actions)) * factorial(len(tokens)) * expected_count
        if canonical_work > MAX_CANONICAL_WORK:
            raise ValueError("ledger exceeds the bounded canonical-alignment work budget")
        records = tuple(self.records)
        if not all(isinstance(record, RawProbeRecord) for record in records):
            raise TypeError("records must contain only RawProbeRecord values")
        records = tuple(sorted(records, key=lambda record: (record.token, record.program)))
        if len(records) != expected_count:
            raise ValueError("ledger must contain every token/program pair exactly once")
        expected_programs = set(product(actions, repeat=length))
        observed_keys: set[tuple[int, tuple[int, ...]]] = set()
        for record in records:
            if record.token not in tokens:
                raise ValueError("record contains an unknown token")
            if record.program not in expected_programs:
                raise ValueError("record contains an invalid or incomplete program")
            if any(step.outcome_code not in outcomes for step in record.steps):
                raise ValueError("record contains an unknown outcome code")
            key = (record.token, record.program)
            if key in observed_keys:
                raise ValueError("ledger contains a duplicate token/program pair")
            observed_keys.add(key)
        object.__setattr__(self, "action_codes", actions)
        object.__setattr__(self, "outcome_codes", outcomes)
        object.__setattr__(self, "token_codes", tokens)
        object.__setattr__(self, "program_length", length)
        object.__setattr__(self, "records", records)
        raw_hash = _json_hash(self.to_dict(include_hashes=False))
        structural_hash = _canonical_structural_hash(self)
        object.__setattr__(self, "_raw_hash", raw_hash)
        object.__setattr__(self, "_structural_hash", structural_hash)

    @property
    def raw_hash(self) -> str:
        """Commitment to exact code IDs, coordinates, outcomes, and RGB hashes."""

        return self._raw_hash

    @property
    def structural_hash(self) -> str:
        """Commitment invariant to code and renderer-value permutations."""

        return self._structural_hash

    def to_dict(self, *, include_hashes: bool = True) -> dict[str, object]:
        payload: dict[str, object] = {
            "version": self.version,
            "action_codes": list(self.action_codes),
            "outcome_codes": list(self.outcome_codes),
            "token_codes": list(self.token_codes),
            "program_length": self.program_length,
            "records": [record.to_dict() for record in self.records],
        }
        if include_hashes:
            payload["raw_hash"] = self.raw_hash
            payload["structural_hash"] = self.structural_hash
        return payload


def capture_operational_ledger(
    oracle: OperationalProbeOracle,
    *,
    program_length: int = 4,
) -> OperationalLedger:
    """Execute the complete fixed-length program suite against fresh clones."""

    if not isinstance(oracle, OperationalProbeOracle):
        raise TypeError("oracle does not provide the operational probe capability")
    length = _strict_int(program_length, "program_length", minimum=1)
    supplied_manifest = oracle.manifest
    manifest = ProbeManifest(
        tuple(supplied_manifest.action_codes),
        tuple(supplied_manifest.outcome_codes),
        tuple(supplied_manifest.token_codes),
        supplied_manifest.max_steps,
    )
    if length > manifest.max_steps:
        raise ValueError("program_length exceeds the environment step budget")
    expected = len(manifest.token_codes) * (len(manifest.action_codes) ** length)
    if expected > MAX_LEDGER_RECORDS:
        raise ValueError(f"probe suite would exceed {MAX_LEDGER_RECORDS} records")
    canonical_work = (
        factorial(len(manifest.action_codes)) * factorial(len(manifest.token_codes)) * expected
    )
    if canonical_work > MAX_CANONICAL_WORK:
        raise ValueError("probe suite exceeds the bounded canonical-alignment work budget")
    records: list[RawProbeRecord] = []
    for token in manifest.token_codes:
        for program in product(manifest.action_codes, repeat=length):
            trace = oracle.run_probe(token, program)
            if len(trace.transitions) != length:
                raise ValueError("oracle returned a trace with the wrong program length")
            steps = tuple(
                RawStepEntry(
                    action_code=transition.action.code,
                    target=transition.action.target,
                    vector=transition.action.vector,
                    outcome_code=transition.outcome_code,
                    before_observation_hash=transition.before.digest(),
                    after_observation_hash=transition.after.digest(),
                    before_pixels_hash=_pixel_hash(transition.before.pixels),
                    after_pixels_hash=_pixel_hash(transition.after.pixels),
                    pixels_changed=transition.pixels_changed,
                )
                for transition in trace.transitions
            )
            records.append(RawProbeRecord(token, tuple(program), steps))
    return OperationalLedger(
        manifest.action_codes,
        manifest.outcome_codes,
        manifest.token_codes,
        length,
        tuple(records),
    )


def _canonical_structural_hash(ledger: OperationalLedger) -> str:
    """Canonicalize the finite transition evidence under all opaque renamings."""

    best: tuple[object, ...] | None = None
    for action_order in permutations(ledger.action_codes):
        action_labels = {code: index for index, code in enumerate(action_order)}
        for token_order in permutations(ledger.token_codes):
            outcome_labels: dict[int, int] = {}
            state_labels: dict[str, int] = {}
            encoded_tokens: list[tuple[object, ...]] = []
            for token in token_order:
                encoded_records: list[tuple[object, ...]] = []
                token_records = [record for record in ledger.records if record.token == token]
                token_records.sort(
                    key=lambda record: tuple(action_labels[code] for code in record.program)
                )
                for record in token_records:
                    program = tuple(action_labels[code] for code in record.program)
                    outcomes: list[int] = []
                    for outcome in record.outcome_trace:
                        if outcome not in outcome_labels:
                            outcome_labels[outcome] = len(outcome_labels)
                        outcomes.append(outcome_labels[outcome])
                    states: list[int] = []
                    for frame_hash in record.frame_hashes:
                        if frame_hash not in state_labels:
                            state_labels[frame_hash] = len(state_labels)
                        states.append(state_labels[frame_hash])
                    encoded_records.append(
                        (program, tuple(outcomes), tuple(states), record.change_trace)
                    )
                encoded_tokens.append(tuple(encoded_records))
            candidate: tuple[object, ...] = (
                len(ledger.action_codes),
                len(ledger.outcome_codes),
                len(ledger.token_codes),
                ledger.program_length,
                tuple(encoded_tokens),
            )
            if best is None or candidate < best:
                best = candidate
    if best is None:
        raise AssertionError("validated ledgers have at least one canonical labelling")
    return _json_hash(best)


class AlignmentStatus(str, Enum):
    """Evaluator conclusion for one source token."""

    IDENTIFIED = "identified"
    UNKNOWN = "unknown"
    INCOMPATIBLE = "incompatible"


@dataclass(frozen=True, slots=True)
class TokenAlignment:
    source_token: int
    candidate_target_tokens: tuple[int, ...]
    status: AlignmentStatus
    reason: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "source_token", _strict_int(self.source_token, "source_token"))
        candidates = tuple(
            sorted(
                _strict_int(value, "candidate_target_tokens")
                for value in self.candidate_target_tokens
            )
        )
        if len(set(candidates)) != len(candidates):
            raise ValueError("candidate_target_tokens must be unique")
        status = AlignmentStatus(self.status)
        if status is AlignmentStatus.IDENTIFIED and len(candidates) != 1:
            raise ValueError("identified alignments require exactly one candidate")
        if status is AlignmentStatus.UNKNOWN and len(candidates) < 2:
            raise ValueError("unknown alignments require multiple operational candidates")
        if status is AlignmentStatus.INCOMPATIBLE and candidates:
            raise ValueError("incompatible alignments cannot have candidates")
        if not isinstance(self.reason, str) or not self.reason:
            raise ValueError("reason must be non-empty")
        object.__setattr__(self, "candidate_target_tokens", candidates)
        object.__setattr__(self, "status", status)

    @property
    def target_token(self) -> int | None:
        return (
            self.candidate_target_tokens[0] if self.status is AlignmentStatus.IDENTIFIED else None
        )


@dataclass(frozen=True, slots=True)
class AlignmentReport:
    """All operationally admissible token correspondences between two ledgers."""

    source_raw_hash: str
    target_raw_hash: str
    source_structural_hash: str
    target_structural_hash: str
    witness_count: int
    witness_hash: str
    tokens: tuple[TokenAlignment, ...]

    def __post_init__(self) -> None:
        for name in (
            "source_raw_hash",
            "target_raw_hash",
            "source_structural_hash",
            "target_structural_hash",
            "witness_hash",
        ):
            object.__setattr__(self, name, _validate_digest(getattr(self, name), name))
        object.__setattr__(
            self,
            "witness_count",
            _strict_int(self.witness_count, "witness_count", minimum=0),
        )
        tokens = tuple(self.tokens)
        if not tokens or not all(isinstance(item, TokenAlignment) for item in tokens):
            raise TypeError("tokens must contain TokenAlignment values")
        tokens = tuple(sorted(tokens, key=lambda item: item.source_token))
        if len({item.source_token for item in tokens}) != len(tokens):
            raise ValueError("source tokens must be unique")
        if bool(self.witness_count) != all(
            item.status is not AlignmentStatus.INCOMPATIBLE for item in tokens
        ):
            raise ValueError("token statuses must agree with witness_count")
        object.__setattr__(self, "tokens", tokens)

    @property
    def compatible(self) -> bool:
        return self.witness_count > 0

    @property
    def mapping(self) -> Mapping[int, int | None]:
        return MappingProxyType({item.source_token: item.target_token for item in self.tokens})

    @property
    def report_hash(self) -> str:
        return _json_hash(
            {
                "source_raw_hash": self.source_raw_hash,
                "target_raw_hash": self.target_raw_hash,
                "source_structural_hash": self.source_structural_hash,
                "target_structural_hash": self.target_structural_hash,
                "witness_count": self.witness_count,
                "witness_hash": self.witness_hash,
                "tokens": [
                    {
                        "source": item.source_token,
                        "candidates": list(item.candidate_target_tokens),
                        "status": item.status.value,
                        "reason": item.reason,
                    }
                    for item in self.tokens
                ],
            }
        )


_Witness = tuple[
    tuple[tuple[int, int], ...],
    tuple[tuple[int, int], ...],
    tuple[tuple[int, int], ...],
]


def _compatible_witness(
    source: OperationalLedger,
    source_records: Mapping[tuple[int, tuple[int, ...]], RawProbeRecord],
    target_records: Mapping[tuple[int, tuple[int, ...]], RawProbeRecord],
    action_map: Mapping[int, int],
    token_map: Mapping[int, int],
) -> tuple[bool, tuple[tuple[int, int], ...]]:
    outcome_map: dict[int, int] = {}
    reverse_outcomes: dict[int, int] = {}
    state_map: dict[str, str] = {}
    reverse_states: dict[str, str] = {}
    for source_token in source.token_codes:
        target_token = token_map[source_token]
        for source_program in product(source.action_codes, repeat=source.program_length):
            mapped_program = tuple(action_map[code] for code in source_program)
            source_record = source_records[(source_token, source_program)]
            target_record = target_records[(target_token, mapped_program)]
            if source_record.change_trace != target_record.change_trace:
                return False, ()
            for source_state, target_state in zip(
                source_record.frame_hashes,
                target_record.frame_hashes,
                strict=True,
            ):
                if source_state in state_map and state_map[source_state] != target_state:
                    return False, ()
                if target_state in reverse_states and reverse_states[target_state] != source_state:
                    return False, ()
                state_map[source_state] = target_state
                reverse_states[target_state] = source_state
            for source_outcome, target_outcome in zip(
                source_record.outcome_trace,
                target_record.outcome_trace,
                strict=True,
            ):
                if source_outcome in outcome_map and outcome_map[source_outcome] != target_outcome:
                    return False, ()
                if (
                    target_outcome in reverse_outcomes
                    and reverse_outcomes[target_outcome] != source_outcome
                ):
                    return False, ()
                outcome_map[source_outcome] = target_outcome
                reverse_outcomes[target_outcome] = source_outcome
    return True, tuple(sorted(outcome_map.items()))


def _alignment_witnesses(
    source: OperationalLedger,
    target: OperationalLedger,
) -> tuple[_Witness, ...]:
    if (
        len(source.action_codes) != len(target.action_codes)
        or len(source.outcome_codes) != len(target.outcome_codes)
        or len(source.token_codes) != len(target.token_codes)
        or source.program_length != target.program_length
    ):
        return ()
    source_records = {(record.token, record.program): record for record in source.records}
    target_records = {(record.token, record.program): record for record in target.records}
    witnesses: set[_Witness] = set()
    for target_action_order in permutations(target.action_codes):
        action_map = dict(zip(source.action_codes, target_action_order, strict=True))
        action_pairs = tuple(sorted(action_map.items()))
        for target_token_order in permutations(target.token_codes):
            token_map = dict(zip(source.token_codes, target_token_order, strict=True))
            compatible, outcome_pairs = _compatible_witness(
                source,
                source_records,
                target_records,
                action_map,
                token_map,
            )
            if compatible:
                witnesses.add((action_pairs, tuple(sorted(token_map.items())), outcome_pairs))
    return tuple(sorted(witnesses))


def align_environments(
    source: OperationalLedger,
    target: OperationalLedger,
) -> AlignmentReport:
    """Align tokens only when every admissible structural isomorphism agrees."""

    if not isinstance(source, OperationalLedger) or not isinstance(target, OperationalLedger):
        raise TypeError("source and target must be OperationalLedger values")
    witnesses = _alignment_witnesses(source, target)
    witness_payload = [
        {
            "actions": action_pairs,
            "tokens": token_pairs,
            "outcomes": outcome_pairs,
        }
        for action_pairs, token_pairs, outcome_pairs in witnesses
    ]
    witness_hash = _json_hash(witness_payload)
    alignments: list[TokenAlignment] = []
    for source_token in source.token_codes:
        candidates = tuple(
            sorted(
                {
                    dict(token_pairs)[source_token]
                    for _action_pairs, token_pairs, _outcome_pairs in witnesses
                }
            )
        )
        if not witnesses:
            status = AlignmentStatus.INCOMPATIBLE
            reason = "no intervention-consequence isomorphism"
        elif len(candidates) == 1:
            status = AlignmentStatus.IDENTIFIED
            reason = "all complete-suite isomorphisms agree"
        else:
            status = AlignmentStatus.UNKNOWN
            reason = "multiple tokens remain equivalent under every available probe"
        alignments.append(TokenAlignment(source_token, candidates, status, reason))
    return AlignmentReport(
        source.raw_hash,
        target.raw_hash,
        source.structural_hash,
        target.structural_hash,
        len(witnesses),
        witness_hash,
        tuple(alignments),
    )


@dataclass(frozen=True, slots=True)
class TransferDecision:
    source_token: int
    claimed_target_token: int | None
    status: AlignmentStatus
    accepted: bool
    reason: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "source_token", _strict_int(self.source_token, "source_token"))
        if self.claimed_target_token is not None:
            object.__setattr__(
                self,
                "claimed_target_token",
                _strict_int(self.claimed_target_token, "claimed_target_token"),
            )
        object.__setattr__(self, "status", AlignmentStatus(self.status))
        object.__setattr__(self, "accepted", bool(self.accepted))
        if not isinstance(self.reason, str) or not self.reason:
            raise ValueError("reason must be non-empty")


@dataclass(frozen=True, slots=True)
class TransferGateResult:
    """Fail-closed result for a complete set of token-transfer claims."""

    passed: bool
    alignment_report_hash: str
    source_raw_hash: str
    target_raw_hash: str
    decisions: tuple[TransferDecision, ...]

    def __post_init__(self) -> None:
        for name in ("alignment_report_hash", "source_raw_hash", "target_raw_hash"):
            object.__setattr__(self, name, _validate_digest(getattr(self, name), name))
        decisions = tuple(self.decisions)
        if not all(isinstance(item, TransferDecision) for item in decisions):
            raise TypeError("decisions must contain TransferDecision values")
        if not decisions or len({item.source_token for item in decisions}) != len(decisions):
            raise ValueError("decisions must be non-empty with unique source tokens")
        if bool(self.passed) != all(item.accepted for item in decisions):
            raise ValueError("passed must equal the conjunction of decision acceptance")
        object.__setattr__(self, "passed", bool(self.passed))
        object.__setattr__(self, "decisions", decisions)

    @property
    def gate_hash(self) -> str:
        return _json_hash(
            {
                "alignment_report_hash": self.alignment_report_hash,
                "source_raw_hash": self.source_raw_hash,
                "target_raw_hash": self.target_raw_hash,
                "passed": self.passed,
                "decisions": [
                    {
                        "source": item.source_token,
                        "claim": item.claimed_target_token,
                        "status": item.status.value,
                        "accepted": item.accepted,
                        "reason": item.reason,
                    }
                    for item in self.decisions
                ],
            }
        )


def gate_transfer(
    report: AlignmentReport,
    claims: Mapping[int, int | None],
) -> TransferGateResult:
    """Require correct identified mappings and explicit abstention on twins."""

    if not isinstance(report, AlignmentReport):
        raise TypeError("report must be an AlignmentReport")
    if not isinstance(claims, Mapping):
        raise TypeError("claims must be a mapping")
    normalized: dict[int, int | None] = {}
    for source_token, claimed_target in claims.items():
        source_code = _strict_int(source_token, "claim source token")
        target_code = (
            None if claimed_target is None else _strict_int(claimed_target, "claimed target token")
        )
        normalized[source_code] = target_code
    expected = {item.source_token for item in report.tokens}
    if set(normalized) != expected:
        raise ValueError("claims must cover every source token exactly once")
    target_vocabulary = {
        candidate for item in report.tokens for candidate in item.candidate_target_tokens
    }
    for target_code in normalized.values():
        if target_code is not None and target_code not in target_vocabulary:
            raise ValueError("claim contains a target token outside the aligned vocabulary")
    decisions: list[TransferDecision] = []
    for item in report.tokens:
        claim = normalized[item.source_token]
        if item.status is AlignmentStatus.IDENTIFIED:
            accepted = claim == item.target_token
            reason = (
                "identified operational mapping verified"
                if accepted
                else "claim disagrees with the unique operational mapping"
            )
        elif item.status is AlignmentStatus.UNKNOWN:
            accepted = claim is None
            reason = (
                "abstention accepted for an operationally non-identifiable token"
                if accepted
                else "concrete claim overstates evidence for equivalent twins"
            )
        else:
            accepted = False
            reason = "environments have no complete-suite operational alignment"
        decisions.append(TransferDecision(item.source_token, claim, item.status, accepted, reason))
    return TransferGateResult(
        all(item.accepted for item in decisions),
        report.report_hash,
        report.source_raw_hash,
        report.target_raw_hash,
        tuple(decisions),
    )
