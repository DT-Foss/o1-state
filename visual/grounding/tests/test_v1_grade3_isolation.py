from __future__ import annotations

from pathlib import Path
import textwrap

import numpy as np
import pytest

from grounding_kernel.contracts import Action, Observation
from grounding_kernel.v1_contracts import (
    ACTION_SCHEMA_OPAQUE_MOTOR,
    SENSOR_SCHEMA_RGB_U8,
    PublicTrace,
    PublicTransition,
    Utterance,
)
from grounding_kernel.v1_grade3_contracts import (
    GRADE3_PROTOCOL_VERSION,
    Grade3SessionManifest,
    MotorActionSpace,
    MotorPhase,
    MotorQuery,
)
from grounding_kernel.v1_grade3_isolation import (
    CandidateExecutionError,
    CandidateProtocolError,
    IsolatedGrade3Grounder,
    _decode_json,
    commit_grade3_candidate,
    verify_grade3_artifact,
)


_SOURCE = """
from hashlib import sha256

from grounding_kernel.v1_contracts import BeliefDecision, DescriptionDecision
from grounding_kernel.v1_grade3_contracts import (
    MotorDecision,
    MotorDirective,
    ProbeDecision,
)


class Candidate:
    def __init__(self):
        self.manifest = None
        self.support = []
        self.probes = []
        self.frozen = False

    def begin(self, manifest):
        self.manifest = manifest

    def observe_support(self, record):
        self.support.append(repr(record))

    def choose_probe(self, offer):
        return ProbeDecision(None, 1.0)

    def observe_probe(self, result):
        self.probes.append(repr(result))

    def freeze(self):
        self.frozen = True

    def checkpoint_commitment(self):
        return sha256(repr((self.support, self.probes, self.frozen)).encode()).hexdigest()

    def motor(self, query):
        MUTATION
        return MotorDecision(MotorDirective.ABSTAIN, None, 1.0)

    def trace_belief(self, query):
        return BeliefDecision((), 1.0)

    def describe(self, query):
        return DescriptionDecision(None, 1.0)


def build():
    return Candidate()
"""


def _candidate(tmp_path: Path, *, mutate_query: bool = False) -> Path:
    root = tmp_path / ("mutating_candidate" if mutate_query else "toy_candidate")
    root.mkdir()
    mutation = "self.support.append('post-freeze mutation')" if mutate_query else "pass"
    (root / "__init__.py").write_text(
        textwrap.dedent(_SOURCE).replace("MUTATION", mutation), encoding="utf-8"
    )
    (root / "helper.py").write_text("VALUE = 1\n", encoding="utf-8")
    return root


def _manifest() -> Grade3SessionManifest:
    return Grade3SessionManifest(
        GRADE3_PROTOCOL_VERSION,
        SENSOR_SCHEMA_RGB_U8,
        ACTION_SCHEMA_OPAQUE_MOTOR,
        support_record_budget=8,
        acquisition_cost_budget=3.0,
        query_budget=8,
        motor_action_cost_budget=12.0,
        motor_reset_budget=4,
    )


def _trace() -> PublicTrace:
    before = Observation(np.zeros((8, 9, 3), dtype=np.uint8), 0)
    after = Observation(np.ones((8, 9, 3), dtype=np.uint8), 1)
    action = Action(91, (3, 4), (1, 0))
    return PublicTrace(before, (PublicTransition(before, action, after, None),))


def _motor_query() -> MotorQuery:
    trace = _trace()
    return MotorQuery(
        1,
        2,
        0,
        Utterance((901, 902)),
        MotorPhase.PROBE,
        (),
        PublicTrace(trace.initial),
        MotorActionSpace((91,), ((1, 0),), 8),
        12.0,
        4,
    )


def test_recursive_commitment_binds_every_candidate_helper(tmp_path: Path) -> None:
    root = _candidate(tmp_path)
    commitment = commit_grade3_candidate(root, "toy_candidate:build")

    assert [path for path, _digest in commitment.candidate_files] == [
        "__init__.py",
        "helper.py",
    ]
    assert verify_grade3_artifact(commitment) == commitment.digest

    (root / "helper.py").write_text("VALUE = 2\n", encoding="utf-8")
    with pytest.raises(CandidateProtocolError, match="changed"):
        verify_grade3_artifact(commitment)


def test_commitment_rejects_symlinks_and_special_artifacts(tmp_path: Path) -> None:
    root = _candidate(tmp_path)
    (root / "escape.py").symlink_to(root / "helper.py")

    with pytest.raises(ValueError, match="symlinks"):
        commit_grade3_candidate(root, "toy_candidate:build")


def test_one_spawned_candidate_crosses_begin_freeze_and_repeated_query(
    tmp_path: Path,
) -> None:
    root = _candidate(tmp_path)
    commitment = commit_grade3_candidate(root, "toy_candidate:build")

    with IsolatedGrade3Grounder(commitment) as candidate:
        candidate.begin(_manifest())
        frozen = candidate.freeze()
        first = candidate.motor(_motor_query())
        second = candidate.motor(_motor_query())

        assert first == second
        assert first.unknown_probability == 1.0
        assert candidate.assert_frozen() == frozen.checkpoint_commitment
        assert frozen.artifact_commitment == commitment.digest
        assert frozen.sdk_commitment == commitment.sdk_commitment


def test_sealed_query_before_freeze_fails_closed(tmp_path: Path) -> None:
    root = _candidate(tmp_path)
    commitment = commit_grade3_candidate(root, "toy_candidate:build")

    with IsolatedGrade3Grounder(commitment) as candidate:
        candidate.begin(_manifest())
        with pytest.raises(CandidateExecutionError, match="requires freeze"):
            candidate.motor(_motor_query())


def test_post_freeze_mutation_is_detected_around_every_query(tmp_path: Path) -> None:
    root = _candidate(tmp_path, mutate_query=True)
    commitment = commit_grade3_candidate(root, "mutating_candidate:build")

    with IsolatedGrade3Grounder(commitment) as candidate:
        candidate.begin(_manifest())
        candidate.freeze()
        with pytest.raises(CandidateExecutionError, match="mutated during sealed query"):
            candidate.motor(_motor_query())


@pytest.mark.parametrize(
    "payload, message",
    [
        (b'{"id":1,"id":2}', "duplicate"),
        (b'{"value":NaN}', "non-finite"),
        (b'{"value":1e999}', "finite"),
    ],
)
def test_rpc_parser_rejects_ambiguous_or_nonfinite_json(
    payload: bytes, message: str
) -> None:
    with pytest.raises(CandidateProtocolError, match=message):
        _decode_json(payload, 1_024)
