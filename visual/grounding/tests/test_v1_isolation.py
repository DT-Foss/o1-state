from __future__ import annotations

from importlib import invalidate_caches
from pathlib import Path
import textwrap

import numpy as np
import pytest

from grounding_kernel.contracts import Action, Observation, Transition
from grounding_kernel.v1_contracts import (
    ACTION_SCHEMA_OPAQUE_MOTOR,
    MAX_BELIEF_CANDIDATES,
    PROTOCOL_VERSION,
    SENSOR_SCHEMA_RGB_U8,
    PublicTrace,
    PublicTransition,
    PublicTurn,
    SessionManifest,
    SessionPhase,
)
from grounding_kernel.v1_isolation import (
    CandidateArtifactCommitment,
    CandidateExecutionError,
    CandidateProtocolError,
    IsolatedGrounder,
    commit_candidate_artifact,
)
from grounding_kernel.v1_runner import SealedEvaluationRunner
from grounding_kernel.v1_session import SessionBudgetError, SessionStateError


_CANDIDATE_SOURCE = """
from hashlib import sha256

from grounding_kernel.v1_contracts import (
    ActionDecision,
    BeliefDecision,
    DescriptionDecision,
    ExperimentDecision,
    Utterance,
)
from grounding_kernel.contracts import Action


class Candidate:
    def __init__(self):
        self.support = []
        self.frozen = False
        self.goal = None

    def begin(self, manifest):
        self.manifest = manifest

    def observe_support(self, turn, trace):
        self.support.append((turn.utterance, trace.initial.digest()))

    def choose_experiment(self, turn):
        return ExperimentDecision(Action(101, (1, 1)), 0.0)

    def observe_experiment(self, turn, transition):
        self.last_experiment = (turn.turn_id, transition.after.digest())

    def freeze(self):
        self.frozen = True

    def checkpoint_commitment(self):
        material = repr((tuple(self.support), self.frozen)).encode()
        return sha256(material).hexdigest()

    def describe(self, trace):
        utterance = self.support[0][0] if self.support else None
        return DescriptionDecision(utterance, 0.0 if utterance else 1.0)

    def begin_goal(self, utterance, observation):
        self.goal = utterance

    def act(self, observation):
        return ActionDecision(None, 1.0)

    def report_belief(self, candidates):
        return BeliefDecision((), 1.0)


def build():
    return Candidate()
"""


def _candidate_module(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    mutate_query: bool = False,
) -> str:
    source = _CANDIDATE_SOURCE
    if mutate_query:
        source = source.replace(
            "    def describe(self, trace):\n",
            "    def describe(self, trace):\n        self.support.append((None, 'mutation'))\n",
        )
    name = "isolated_candidate_mutating" if mutate_query else "isolated_candidate"
    (tmp_path / f"{name}.py").write_text(textwrap.dedent(source), encoding="utf-8")
    monkeypatch.syspath_prepend(str(tmp_path))
    invalidate_caches()
    return f"{name}:build"


def _manifest(
    *,
    intervention_cost_budget: float = 2.0,
    query_budget: int = 4,
) -> SessionManifest:
    return SessionManifest(
        PROTOCOL_VERSION,
        SENSOR_SCHEMA_RGB_U8,
        ACTION_SCHEMA_OPAQUE_MOTOR,
        support_episode_budget=2,
        intervention_cost_budget=intervention_cost_budget,
        query_budget=query_budget,
    )


def _observation() -> Observation:
    return Observation(np.zeros((4, 4, 3), dtype=np.uint8), 0, False)


def _support() -> tuple[PublicTurn, PublicTrace]:
    from grounding_kernel.v1_contracts import Utterance

    observation = _observation()
    return (
        PublicTurn(
            0,
            SessionPhase.SUPPORT,
            observation,
            Utterance((781_223_091,)),
            None,
            1.0,
            2.0,
        ),
        PublicTrace(observation),
    )


def test_one_persistent_candidate_crosses_support_freeze_and_query(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    entrypoint = _candidate_module(tmp_path, monkeypatch)
    commitment = commit_candidate_artifact(entrypoint)
    turn, trace = _support()

    with IsolatedGrounder(commitment) as candidate:
        candidate.begin(_manifest())
        candidate.observe_support(turn, trace)
        frozen = candidate.freeze()
        decision = candidate.describe(trace)

        assert decision.utterance == turn.utterance
        assert candidate.assert_frozen() == frozen.checkpoint_commitment
        assert frozen.artifact_commitment == commitment.digest
        assert candidate.request_count >= 5


def test_query_before_freeze_fails_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    entrypoint = _candidate_module(tmp_path, monkeypatch)
    commitment = commit_candidate_artifact(entrypoint)

    with IsolatedGrounder(commitment) as candidate:
        candidate.begin(_manifest())
        with pytest.raises(CandidateExecutionError, match="requires freeze"):
            candidate.describe(PublicTrace(_observation()))


def test_post_freeze_model_mutation_is_detected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    entrypoint = _candidate_module(tmp_path, monkeypatch, mutate_query=True)
    commitment = commit_candidate_artifact(entrypoint)
    turn, trace = _support()

    with IsolatedGrounder(commitment) as candidate:
        candidate.begin(_manifest())
        candidate.observe_support(turn, trace)
        candidate.freeze()
        with pytest.raises(CandidateExecutionError, match="mutated during sealed query"):
            candidate.describe(trace)


def test_artifact_commitment_binds_declared_weights(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    entrypoint = _candidate_module(tmp_path, monkeypatch)
    weights = tmp_path / "weights.bin"
    weights.write_bytes(b"first")
    first = commit_candidate_artifact(entrypoint, (weights,))
    weights.write_bytes(b"second")
    second = commit_candidate_artifact(entrypoint, (weights,))

    assert first.digest != second.digest
    assert first.entrypoint == second.entrypoint


def test_duplicate_artifact_paths_are_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    entrypoint = _candidate_module(tmp_path, monkeypatch)
    artifact = tmp_path / "weights.bin"
    artifact.write_bytes(b"weights")

    with pytest.raises(ValueError, match="unique"):
        commit_candidate_artifact(entrypoint, (artifact, artifact))


def test_runner_binds_real_action_cost_freeze_and_query_to_one_process(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from hashlib import sha256

    entrypoint = _candidate_module(tmp_path, monkeypatch)
    commitment = commit_candidate_artifact(entrypoint)
    codebook_commitment = sha256(b"post-model-codebook").hexdigest()
    support_turn, support_trace = _support()

    with SealedEvaluationRunner(
        commitment,
        _manifest(),
        lambda: codebook_commitment,
        action_cost=lambda _action: 0.75,
    ) as runner:
        runner.support(support_turn, support_trace)
        runner.begin_acquisition()
        acquisition_turn = PublicTurn(
            1,
            SessionPhase.ACQUISITION,
            _observation(),
            None,
            None,
            None,
            runner.remaining_cost,
        )

        def execute(action: Action) -> Transition:
            before = _observation()
            pixels = before.pixels.copy()
            pixels[1, 1] = 255
            after = Observation(pixels, 1, False)
            return Transition(before, action, after, 9_999)

        experiment = runner.experiment(acquisition_turn, execute)
        frozen = runner.freeze()
        description = runner.describe(support_trace)
        ledger = runner.complete()
        audit = runner.audit

    assert experiment.cost == 0.75
    assert experiment.transition is not None
    assert not hasattr(experiment.transition, "outcome_code")
    assert description.utterance == support_turn.utterance
    assert ledger.intervention_cost_used == 0.75
    assert ledger.queries_used == 1
    assert ledger.checkpoint_commitment == frozen.checkpoint_commitment
    assert audit.chain_valid
    assert audit.intervention_cost_consumed == pytest.approx(0.75)
    reserved = next(
        event
        for event in audit.events
        if event.operation == "experiment" and event.status == "reserved"
    )
    completed = next(
        event
        for event in audit.events
        if event.operation == "experiment" and event.status == "completed"
    )
    assert reserved.request_hash is not None
    assert reserved.decision_hash is not None
    assert reserved.state_before_hash != reserved.state_after_hash
    assert completed.reservation_hash == reserved.event_hash
    assert experiment.audit_commitment == completed.event_hash


def test_runner_rejects_precomputed_codebook_and_calls_factory_after_child_begin(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from hashlib import sha256

    commitment = commit_candidate_artifact(_candidate_module(tmp_path, monkeypatch))
    digest = sha256(b"late-codebook").hexdigest()
    with pytest.raises(TypeError, match="codebook_factory"):
        SealedEvaluationRunner(commitment, _manifest(), digest)  # type: ignore[arg-type]

    calls: list[int] = []
    holder: dict[str, SealedEvaluationRunner] = {}

    def factory() -> str:
        runner = holder["runner"]
        assert runner.candidate.request_count == 1
        assert runner.candidate._process is not None
        assert runner.candidate._process.is_alive()
        calls.append(runner.candidate.request_count)
        return digest

    runner = SealedEvaluationRunner(commitment, _manifest(), factory)
    holder["runner"] = runner
    with runner:
        assert runner.session.commitments["codebook"] == digest

    assert calls == [1]


def test_runner_rejects_session_codebook_mutation_before_artifact_verification(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from hashlib import sha256

    commitment = commit_candidate_artifact(_candidate_module(tmp_path, monkeypatch))
    precomputed = sha256(b"premature-codebook").hexdigest()
    factory_called = False

    def factory() -> str:
        nonlocal factory_called
        factory_called = True
        return precomputed

    runner = SealedEvaluationRunner(commitment, _manifest(), factory)
    runner.session.commit_codebook(precomputed)
    with pytest.raises(SessionStateError, match="pristine"):
        runner.start()
    assert not factory_called
    assert runner.candidate._process is None


def test_start_rehashes_source_and_every_declared_artifact_before_factory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from hashlib import sha256

    entrypoint = _candidate_module(tmp_path, monkeypatch)
    config = tmp_path / "candidate.json"
    config.write_text('{"version": 1}', encoding="utf-8")
    commitment = commit_candidate_artifact(entrypoint, (config,))
    factory_called = False

    def factory() -> str:
        nonlocal factory_called
        factory_called = True
        return sha256(b"must-not-exist-yet").hexdigest()

    config.write_text('{"version": 2}', encoding="utf-8")
    runner = SealedEvaluationRunner(commitment, _manifest(), factory)
    with pytest.raises(CandidateProtocolError, match="changed after commitment"):
        runner.start()

    assert not factory_called
    assert runner.candidate._process is None


def test_started_child_executes_snapshot_not_late_mutated_live_source(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    entrypoint = _candidate_module(tmp_path, monkeypatch)
    commitment = commit_candidate_artifact(entrypoint)
    candidate = IsolatedGrounder(commitment).start()
    try:
        Path(commitment.source_path).write_text(
            "raise RuntimeError('live source must never execute')\n",
            encoding="utf-8",
        )
        candidate.begin(_manifest())
        frozen = candidate.freeze()
        assert len(frozen.checkpoint_commitment) == 64
    finally:
        candidate.close()


def test_child_imports_manifested_helper_and_config_from_staged_snapshot(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    marker = tmp_path / "allow-child-import"
    helper = tmp_path / "staged_candidate_helper.py"
    config = tmp_path / "staged-config.txt"
    module = tmp_path / "staged_candidate.py"
    helper.write_text("VALUE = 'committed'\n", encoding="utf-8")
    config.write_text("committed", encoding="utf-8")
    module.write_text(
        textwrap.dedent(
            f"""
            from hashlib import sha256
            from pathlib import Path
            import time

            from grounding_kernel.v1_contracts import (
                ActionDecision,
                BeliefDecision,
                DescriptionDecision,
                ExperimentDecision,
            )

            while not Path({str(marker)!r}).exists():
                time.sleep(0.01)
            from staged_candidate_helper import VALUE
            CONFIG = Path("staged-config.txt").read_text(encoding="utf-8")

            class Candidate:
                def begin(self, manifest):
                    if VALUE != "committed" or CONFIG != "committed":
                        raise RuntimeError("loaded mutable live artifact")
                def observe_support(self, turn, trace): pass
                def choose_experiment(self, turn): return ExperimentDecision(None, 1.0)
                def observe_experiment(self, turn, transition): pass
                def freeze(self): pass
                def checkpoint_commitment(self):
                    return sha256((VALUE + CONFIG).encode()).hexdigest()
                def describe(self, trace): return DescriptionDecision(None, 1.0)
                def begin_goal(self, utterance, observation): pass
                def act(self, observation): return ActionDecision(None, 1.0)
                def report_belief(self, candidates): return BeliefDecision((), 1.0)

            def build(): return Candidate()
            """
        ),
        encoding="utf-8",
    )
    monkeypatch.syspath_prepend(str(tmp_path))
    invalidate_caches()
    commitment = commit_candidate_artifact(
        "staged_candidate:build",
        (helper, config),
    )
    candidate = IsolatedGrounder(commitment).start()
    try:
        helper.write_text("VALUE = 'mutated'\n", encoding="utf-8")
        config.write_text("mutated", encoding="utf-8")
        marker.touch()
        candidate.begin(_manifest())
        assert len(candidate.freeze().checkpoint_commitment) == 64
    finally:
        marker.touch(exist_ok=True)
        candidate.close()


def test_commitment_constructor_and_start_fail_closed_on_forged_bytes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    entrypoint = _candidate_module(tmp_path, monkeypatch)
    commitment = commit_candidate_artifact(entrypoint)
    with pytest.raises(CandidateProtocolError, match="does not match"):
        CandidateArtifactCommitment(
            entrypoint=commitment.entrypoint,
            source_path=commitment.source_path,
            artifact_paths=commitment.artifact_paths,
            module_root=commitment.module_root,
            file_digests=commitment.file_digests,
            digest="0" * 64,
        )

    Path(commitment.source_path).write_text("# forged after commit\n", encoding="utf-8")
    candidate = IsolatedGrounder(commitment)
    with pytest.raises(CandidateProtocolError, match="changed after commitment"):
        candidate.start()
    candidate.close()


def _acquisition_turn(runner: SealedEvaluationRunner, *, feedback: float | None = None) -> PublicTurn:
    return PublicTurn(
        11,
        SessionPhase.ACQUISITION,
        _observation(),
        None,
        None,
        feedback,
        runner.remaining_cost,
    )


def _matching_transition(action: Action, *, feedback: float | None = None) -> PublicTransition:
    before = _observation()
    pixels = before.pixels.copy()
    pixels[0, 0] = 44
    return PublicTransition(
        before,
        action,
        Observation(pixels, 1, False),
        feedback,
    )


def test_acquisition_feedback_is_rejected_before_runner_or_child_delivery(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from hashlib import sha256
    import grounding_kernel.v1_isolation as isolation

    commitment = commit_candidate_artifact(_candidate_module(tmp_path, monkeypatch))
    runner = SealedEvaluationRunner(
        commitment,
        _manifest(),
        lambda: sha256(b"feedback-free-codebook").hexdigest(),
    ).start()
    try:
        runner.begin_acquisition()
        feedback_turn = _acquisition_turn(runner, feedback=0.5)
        requests_before = runner.candidate.request_count
        with pytest.raises(SessionStateError, match="feedback"):
            runner.experiment(feedback_turn, lambda action: _matching_transition(action))
        with pytest.raises(SessionStateError, match="feedback"):
            runner.experiment(
                _acquisition_turn(runner),
                lambda action: _matching_transition(action),
                scalar_feedback=0.5,
            )
        assert runner.candidate.request_count == requests_before

        with pytest.raises(SessionStateError, match="feedback"):
            runner.experiment(
                _acquisition_turn(runner),
                lambda action: _matching_transition(action, feedback=0.5),
            )
        assert runner.audit.failed_cost_consumed == pytest.approx(1.0)

        clean_turn = _acquisition_turn(runner)
        leaked = _matching_transition(Action(101, (1, 1)), feedback=0.5)
        parent_requests = runner.candidate.request_count
        with pytest.raises(CandidateProtocolError, match="feedback"):
            runner.candidate.observe_experiment(clean_turn, leaked)
        assert runner.candidate.request_count == parent_requests
        with pytest.raises(CandidateExecutionError, match="feedback"):
            runner.candidate._call(
                "observe_experiment",
                [isolation._wire(clean_turn), isolation._wire(leaked)],
            )
    finally:
        runner.close()


def test_cost_is_reserved_before_execute_and_failed_attempt_is_consumed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from hashlib import sha256

    commitment = commit_candidate_artifact(_candidate_module(tmp_path, monkeypatch))
    executed = 0
    runner = SealedEvaluationRunner(
        commitment,
        _manifest(intervention_cost_budget=0.5),
        lambda: sha256(b"small-budget-codebook").hexdigest(),
        action_cost=lambda _action: 0.75,
    ).start()
    try:
        runner.begin_acquisition()

        def execute(action: Action) -> PublicTransition:
            nonlocal executed
            executed += 1
            return _matching_transition(action)

        with pytest.raises(SessionBudgetError, match="cost"):
            runner.experiment(_acquisition_turn(runner), execute)
        assert executed == 0
        assert runner.session.ledger.intervention_cost_used == 0.0
    finally:
        runner.close()

    commitment = commit_candidate_artifact(_candidate_module(tmp_path, monkeypatch))
    runner = SealedEvaluationRunner(
        commitment,
        _manifest(),
        lambda: sha256(b"failed-execute-codebook").hexdigest(),
        action_cost=lambda _action: 0.75,
    ).start()
    try:
        runner.begin_acquisition()

        def fail(_action: Action) -> PublicTransition:
            raise RuntimeError("world failure after reservation")

        with pytest.raises(RuntimeError, match="world failure"):
            runner.experiment(_acquisition_turn(runner), fail)
        assert runner.remaining_cost == pytest.approx(1.25)
        assert runner.session.ledger.intervention_cost_used == 0.0
        assert runner.audit.failed_cost_consumed == pytest.approx(0.75)
        assert runner.audit.chain_valid
        assert runner.audit.events[-1].status == "failed_consumed"
    finally:
        runner.close()


def test_executor_callback_cannot_reenter_runner_during_reservation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from hashlib import sha256

    commitment = commit_candidate_artifact(_candidate_module(tmp_path, monkeypatch))
    runner = SealedEvaluationRunner(
        commitment,
        _manifest(),
        lambda: sha256(b"non-reentrant-codebook").hexdigest(),
        action_cost=lambda _action: 0.25,
    ).start()
    try:
        runner.begin_acquisition()

        def reenter(_action: Action) -> PublicTransition:
            runner.freeze()
            raise AssertionError("unreachable")

        with pytest.raises(SessionStateError, match="re-enter"):
            runner.experiment(_acquisition_turn(runner), reenter)
        assert runner.session.phase is SessionPhase.ACQUISITION
        assert runner.audit.failed_cost_consumed == pytest.approx(0.25)
        assert runner.audit.reserved_cost == 0.0
    finally:
        runner.close()


def test_transition_before_is_exactly_bound_and_mismatch_consumes_reservation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from hashlib import sha256

    commitment = commit_candidate_artifact(_candidate_module(tmp_path, monkeypatch))
    runner = SealedEvaluationRunner(
        commitment,
        _manifest(),
        lambda: sha256(b"before-binding-codebook").hexdigest(),
        action_cost=lambda _action: 0.5,
    ).start()
    try:
        runner.begin_acquisition()
        turn = _acquisition_turn(runner)

        def mismatched(action: Action) -> PublicTransition:
            before = Observation(np.ones((4, 4, 3), dtype=np.uint8), 0, False)
            after = Observation(np.ones((4, 4, 3), dtype=np.uint8), 1, False)
            return PublicTransition(before, action, after)

        with pytest.raises(RuntimeError, match=r"transition\.before"):
            runner.experiment(turn, mismatched)
        assert runner.remaining_cost == pytest.approx(1.5)
        assert runner.audit.failed_cost_consumed == pytest.approx(0.5)
        assert runner.candidate.request_count == 2  # begin + choose; no evidence RPC
    finally:
        runner.close()


def test_query_budget_is_reserved_before_any_candidate_rpc(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from concurrent.futures import ThreadPoolExecutor
    from hashlib import sha256

    commitment = commit_candidate_artifact(_candidate_module(tmp_path, monkeypatch))
    runner = SealedEvaluationRunner(
        commitment,
        _manifest(query_budget=1),
        lambda: sha256(b"one-query-codebook").hexdigest(),
    ).start()
    try:
        runner.begin_acquisition()
        runner.freeze()
        trace = PublicTrace(_observation())
        forged_belief_turn = PublicTurn(
            99,
            SessionPhase.FROZEN_QUERY,
            _observation(),
            remaining_cost=999.0,
        )
        requests_before_forgery = runner.candidate.request_count
        with pytest.raises(SessionStateError, match="authoritative"):
            runner.report_belief(forged_belief_turn, (1, 2))
        assert runner.candidate.request_count == requests_before_forgery
        assert runner.remaining_queries == 1
        with ThreadPoolExecutor(max_workers=2) as pool:
            futures = [pool.submit(runner.describe, trace) for _ in range(2)]
            outcomes: list[object] = []
            for future in futures:
                try:
                    outcomes.append(future.result())
                except BaseException as exc:  # asserted below
                    outcomes.append(exc)
        assert sum(not isinstance(item, BaseException) for item in outcomes) == 1
        failures = [item for item in outcomes if isinstance(item, BaseException)]
        assert len(failures) == 1
        assert isinstance(failures[0], SessionBudgetError)
        # begin + freeze + describe + checkpoint; the rejected query made no RPC.
        assert runner.candidate.request_count == 4
        assert runner.audit.queries_consumed == 1
        assert runner.audit.chain_valid
    finally:
        runner.close()


def test_direct_belief_rpc_rejects_duplicate_and_oversized_lists_parent_and_child(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    commitment = commit_candidate_artifact(_candidate_module(tmp_path, monkeypatch))
    candidate = IsolatedGrounder(commitment).start()
    try:
        candidate.begin(_manifest())
        candidate.freeze()
        requests_before = candidate.request_count
        with pytest.raises(CandidateProtocolError, match="unique"):
            candidate.report_belief((3, 3))
        with pytest.raises(CandidateProtocolError, match="limit"):
            candidate.report_belief(tuple(range(MAX_BELIEF_CANDIDATES + 1)))
        assert candidate.request_count == requests_before

        with pytest.raises(CandidateExecutionError, match="unique"):
            candidate._call("report_belief", [[3, 3]])
        assert candidate.request_count == requests_before + 1
    finally:
        candidate.close()
