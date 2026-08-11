from __future__ import annotations

import ast
from hashlib import sha256
from itertools import product
from pathlib import Path

import numpy as np
import pytest

from grounding_kernel.contracts import Action
from grounding_kernel import independent_world as independent_world_module
from grounding_kernel.independent_world import (
    IndependentConfig,
    IndependentHarness,
    audit_independent_agent,
    pixel_change_pattern,
    trace_is_continuous,
)
from grounding_kernel.contracts import Trajectory


def _crop(frame: np.ndarray, center: tuple[int, int], radius: int = 10) -> np.ndarray:
    x, y = center
    return frame[y - radius : y + radius + 1, x - radius : x + radius + 1]


def _state_partition(trace: Trajectory) -> tuple[int, ...]:
    frames = (trace.initial,) + tuple(step.after for step in trace.transitions)
    labels: dict[str, int] = {}
    result: list[int] = []
    for observation in frames:
        digest = sha256(observation.pixels.tobytes()).hexdigest()
        if digest not in labels:
            labels[digest] = len(labels)
        result.append(labels[digest])
    return tuple(result)


def test_implementation_imports_only_generic_kernel_contracts() -> None:
    module_path = Path(independent_world_module.__file__ or "")
    imported_modules = {
        node.module
        for node in ast.walk(ast.parse(module_path.read_text(encoding="utf-8")))
        if isinstance(node, ast.ImportFrom) and node.module is not None
    }

    assert "processworld" not in imported_modules
    assert "microworld" not in imported_modules
    assert all("Meaning" not in name for name in independent_world_module.__all__)


def test_same_seed_replays_pixels_codebooks_latents_and_transitions() -> None:
    first = IndependentHarness(8128, codebook_variant=3, renderer_variant=4, world_variant=5)
    second = IndependentHarness(8128, codebook_variant=3, renderer_variant=4, world_variant=5)

    assert first.agent.manifest == second.agent.manifest
    assert first.agent.observe() == second.agent.observe()
    assert first.oracle.snapshot() == second.oracle.snapshot()

    token = first.agent.token_codes[0]
    meaning = first.oracle.decode_token(token)
    second_token = second.oracle.encode_token(meaning)
    code = first.oracle.encode_action("excite")
    second_code = second.oracle.encode_action("excite")
    first_step = first.agent.step(Action(code, first.oracle.device_center(token)))
    second_step = second.agent.step(Action(second_code, second.oracle.device_center(second_token)))
    assert first_step == second_step
    assert trace_is_continuous(first.agent.trajectory())


def test_agent_surface_is_raw_opaque_and_separate_from_oracle() -> None:
    harness = IndependentHarness(19)
    observation = harness.agent.observe()

    assert audit_independent_agent(harness.agent) == ()
    assert observation.pixels.dtype == np.uint8
    assert observation.pixels.shape == harness.agent.manifest.observation_shape
    assert observation.pixels.flags.c_contiguous
    assert not observation.pixels.flags.writeable
    with pytest.raises(ValueError):
        observation.pixels[0, 0] = 0

    assert set(harness.agent.manifest.__dataclass_fields__) == {
        "observation_shape",
        "action_codes",
        "outcome_codes",
        "token_codes",
        "motor_vectors",
        "max_steps",
    }
    public_values = (
        harness.agent.action_codes + harness.agent.outcome_codes + harness.agent.token_codes
    )
    assert all(type(value) is int for value in public_values)
    assert {len(str(value)) for value in public_values} == {9}
    with pytest.raises(TypeError):
        vars(harness.agent)


def test_codebook_and_renderer_can_be_permuted_independently() -> None:
    base = IndependentHarness(91, codebook_variant=0, renderer_variant=0)
    renamed = IndependentHarness(91, codebook_variant=99, renderer_variant=0)
    rerendered = IndependentHarness(91, codebook_variant=0, renderer_variant=99)

    assert np.array_equal(base.agent.observe().pixels, renamed.agent.observe().pixels)
    assert set(base.agent.action_codes).isdisjoint(renamed.agent.action_codes)
    assert set(base.agent.outcome_codes).isdisjoint(renamed.agent.outcome_codes)
    assert set(base.agent.token_codes).isdisjoint(renamed.agent.token_codes)

    assert base.agent.manifest == rerendered.agent.manifest
    assert base.oracle.snapshot() == rerendered.oracle.snapshot()
    assert not np.array_equal(base.agent.observe().pixels, rerendered.agent.observe().pixels)


def test_matched_causal_twins_require_an_ordered_intervention() -> None:
    harness = IndependentHarness(2026)
    retainer, relay = harness.oracle.matched_causal_tokens()
    retainer_center = harness.oracle.device_center(retainer)
    relay_center = harness.oracle.device_center(relay)
    initial = harness.agent.observe().pixels

    assert np.array_equal(_crop(initial, retainer_center), _crop(initial, relay_center))
    retainer_trace, relay_trace = harness.oracle.causal_twin_traces()
    assert retainer_trace.initial == relay_trace.initial
    assert tuple(step.action for step in retainer_trace.transitions) == tuple(
        step.action for step in relay_trace.transitions
    )
    assert retainer_trace.transitions[0].after == relay_trace.transitions[0].after
    assert retainer_trace.transitions[1].after != relay_trace.transitions[1].after
    assert tuple(
        harness.oracle.decode_outcome(step.outcome_code) for step in retainer_trace.transitions[:2]
    ) == tuple(
        harness.oracle.decode_outcome(step.outcome_code) for step in relay_trace.transitions[:2]
    )
    assert harness.oracle.decode_outcome(retainer_trace.transitions[-1].outcome_code) == "active"
    assert harness.oracle.decode_outcome(relay_trace.transitions[-1].outcome_code) == "inactive"
    assert pixel_change_pattern(retainer_trace) != pixel_change_pattern(relay_trace)


def test_latent_delay_twins_are_identical_under_every_four_step_program() -> None:
    harness = IndependentHarness(314)
    left, right = harness.oracle.nonidentifiable_tokens()

    for program in product(harness.agent.action_codes, repeat=4):
        left_trace = harness.oracle.run_probe(left, program)
        right_trace = harness.oracle.run_probe(right, program)
        assert tuple(step.outcome_code for step in left_trace.transitions) == tuple(
            step.outcome_code for step in right_trace.transitions
        )
        assert pixel_change_pattern(left_trace) == pixel_change_pattern(right_trace)
        assert _state_partition(left_trace) == _state_partition(right_trace)


def test_delayed_response_is_genuinely_multi_step() -> None:
    harness = IndependentHarness(72)
    token = harness.oracle.nonidentifiable_tokens()[0]
    excite = harness.oracle.encode_action("excite")
    advance = harness.oracle.encode_action("advance")
    query = harness.oracle.encode_action("query")

    early = harness.oracle.run_probe(token, (excite, advance, query))
    late = harness.oracle.run_probe(token, (excite, advance, advance, query))
    assert harness.oracle.decode_outcome(early.transitions[-1].outcome_code) == "inactive"
    assert harness.oracle.decode_outcome(late.transitions[-1].outcome_code) == "active"
    assert len(late.transitions) == 4


def test_invalid_actions_fail_closed_and_terminal_budget_is_authoritative() -> None:
    harness = IndependentHarness(55, IndependentConfig(max_steps=4))
    token = harness.agent.token_codes[0]
    center = harness.oracle.device_center(token)
    code = harness.agent.action_codes[0]

    with pytest.raises(KeyError, match="unknown action"):
        harness.agent.step(Action(999_999_999, center))
    with pytest.raises(ValueError, match="vector"):
        harness.agent.step(Action(code, center, (8, 8)))

    miss = harness.agent.step(Action(code, (3, 3)))
    assert harness.oracle.decode_outcome(miss.outcome_code) == "missed"
    for _ in range(3):
        harness.agent.step(Action(code, center))
    assert harness.agent.observe().terminal
    with pytest.raises(RuntimeError, match="terminal"):
        harness.agent.step(Action(code, center))
    assert not harness.agent.reset().terminal


@pytest.mark.parametrize(
    "kwargs",
    [
        {"frame_size": 40},
        {"max_steps": 3},
        {"device_radius": 2},
    ],
)
def test_configuration_rejects_invalid_finite_geometry(kwargs: dict[str, int]) -> None:
    with pytest.raises(ValueError):
        IndependentConfig(**kwargs)
