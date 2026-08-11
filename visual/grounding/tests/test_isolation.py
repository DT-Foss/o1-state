from __future__ import annotations

import gc
import json

import pytest

from grounding_kernel.contracts import Action, Observation, Trajectory, Transition
from grounding_kernel.isolation import (
    LearnerClient,
    LearnerExecutionError,
    LearnerProcess,
    LearnerProtocolError,
    LearnerTimeoutError,
    RemoteEnvironmentError,
    run_isolated_learner,
)
from grounding_kernel.microworld import EvaluatorHarness


def _entrypoint(name: str) -> str:
    return f"{__name__}:{name}"


def learner_round_trip(environment: LearnerClient) -> dict[str, object]:
    initial = environment.observe()
    action = Action(environment.action_codes[0], (0, 0))
    transition = environment.step(action)
    trajectory = environment.trajectory()
    return {
        "client_type": type(environment).__name__,
        "initial_is_observation": isinstance(initial, Observation),
        "transition_is_record": isinstance(transition, Transition),
        "trajectory_is_record": isinstance(trajectory, Trajectory),
        "initial_digest": initial.digest(),
        "current_digest": trajectory.current.digest(),
        "tick": trajectory.current.tick,
        "readonly": not initial.pixels.flags.writeable,
    }


def learner_reflection_attack(environment: LearnerClient) -> dict[str, object]:
    methods = (environment.observe, environment.reset, environment.step, environment.trajectory)
    owners = [type(method.__self__).__name__ for method in methods]
    closures = [method.__func__.__closure__ is None for method in methods]
    forbidden_instances = sorted(
        {
            type(value).__name__
            for value in gc.get_objects()
            if type(value).__name__ in {"_Engine", "_Oracle", "Microworld", "EvaluatorHarness"}
        }
    )
    retained_bound_owners: list[str] = []
    for slot in LearnerClient.__slots__:
        mangled = f"_LearnerClient{slot}" if slot.startswith("__") else slot
        value = object.__getattribute__(environment, mangled)
        owner = getattr(value, "__self__", None)
        if owner is not None:
            retained_bound_owners.append(type(owner).__name__)
    return {
        "owners": owners,
        "closures_are_empty": closures,
        "forbidden_instances": forbidden_instances,
        "retained_bound_owners": retained_bound_owners,
        "public_names": [name for name in dir(environment) if not name.startswith("_")],
    }


def learner_invalid_action(environment: LearnerClient) -> dict[str, object]:
    try:
        environment.step(Action(123_456_789, (0, 0)))
    except RemoteEnvironmentError as error:
        return {"remote_type": error.remote_type, "message": error.remote_message}
    raise AssertionError("invalid action unexpectedly succeeded")


def learner_unknown_command(environment: LearnerClient) -> None:
    connection = object.__getattribute__(environment, "_LearnerClient__connection")
    connection.send_bytes(
        json.dumps(
            {"version": 1, "id": 99, "command": "snapshot", "payload": {}},
            separators=(",", ":"),
        ).encode("utf-8")
    )


def learner_oversized_frame(environment: LearnerClient) -> None:
    connection = object.__getattribute__(environment, "_LearnerClient__connection")
    connection.send_bytes(b"x" * 2_048)


def learner_raises(environment: LearnerClient) -> None:
    environment.observe()
    raise LookupError("deliberate learner failure")


def learner_hangs(environment: LearnerClient) -> None:
    environment.observe()
    while True:
        pass


def test_spawn_rpc_round_trips_only_owned_serialized_records() -> None:
    harness = EvaluatorHarness(404)
    result = run_isolated_learner(harness.agent, _entrypoint("learner_round_trip"), timeout=5)

    assert result.exitcode == 0
    assert result.request_count == 5  # manifest, observe, step, trajectory, complete
    assert result.value == {
        "client_type": "LearnerClient",
        "initial_is_observation": True,
        "transition_is_record": True,
        "trajectory_is_record": True,
        "initial_digest": harness.agent.trajectory().initial.digest(),
        "current_digest": harness.agent.trajectory().current.digest(),
        "tick": 1,
        "readonly": True,
    }
    assert len(harness.agent.trajectory().transitions) == 1


def test_bound_method_self_attack_reaches_only_the_child_rpc_client() -> None:
    harness = EvaluatorHarness(405)
    result = run_isolated_learner(
        harness.agent,
        _entrypoint("learner_reflection_attack"),
        timeout=5,
    )

    assert result.value == {
        "owners": ["LearnerClient"] * 4,
        "closures_are_empty": [True] * 4,
        "forbidden_instances": [],
        "retained_bound_owners": [],
        "public_names": [
            "action_codes",
            "manifest",
            "observe",
            "reset",
            "step",
            "symbol_codes",
            "trajectory",
        ],
    }


def test_environment_errors_are_records_not_parent_objects() -> None:
    result = run_isolated_learner(
        EvaluatorHarness(406).agent,
        _entrypoint("learner_invalid_action"),
        timeout=5,
    )

    assert result.value == {
        "remote_type": "ValueError",
        "message": "unknown opaque action code",
    }


def test_non_allowlisted_command_poisoning_terminates_the_child() -> None:
    process = LearnerProcess(
        EvaluatorHarness(407).agent,
        _entrypoint("learner_unknown_command"),
        timeout=5,
    )

    with pytest.raises(LearnerProtocolError, match="not allowlisted"):
        process.run()
    assert not process.is_alive


def test_oversized_child_frame_is_rejected_without_pickle() -> None:
    process = LearnerProcess(
        EvaluatorHarness(408).agent,
        _entrypoint("learner_oversized_frame"),
        timeout=5,
        max_message_bytes=1_024,
    )

    with pytest.raises(LearnerProtocolError, match="byte limit"):
        process.run()
    assert not process.is_alive


def test_child_exception_is_reported_and_process_is_reaped() -> None:
    process = LearnerProcess(
        EvaluatorHarness(409).agent,
        _entrypoint("learner_raises"),
        timeout=5,
    )

    with pytest.raises(LearnerExecutionError, match="LookupError") as captured:
        process.run()
    assert "deliberate learner failure" in captured.value.remote_traceback
    assert not process.is_alive


def test_entrypoint_import_failure_is_reported_before_client_construction() -> None:
    process = LearnerProcess(
        EvaluatorHarness(412).agent,
        _entrypoint("does_not_exist"),
        timeout=5,
    )

    with pytest.raises(LearnerExecutionError, match="AttributeError") as captured:
        process.run()
    assert captured.value.remote_type == "AttributeError"
    assert not process.is_alive


def test_timeout_terminates_and_reaps_the_spawned_learner() -> None:
    process = LearnerProcess(
        EvaluatorHarness(410).agent,
        _entrypoint("learner_hangs"),
        timeout=1,
    )

    with pytest.raises(LearnerTimeoutError, match="exceeded timeout"):
        process.run()
    assert not process.is_alive


@pytest.mark.parametrize(
    ("entrypoint", "message"),
    [("missing-colon", "module:callable"), ("bad module:callable", "identifier")],
)
def test_entrypoint_is_an_import_reference_not_a_pickled_callable(
    entrypoint: str,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        LearnerProcess(EvaluatorHarness(411).agent, entrypoint)
