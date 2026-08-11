"""Tests for fertig.diagnostics (ported from _codex_lab/primitive_schema_snapshot's
test_primitives.py -- only the diagnostics-specific tests, since primitives/
relations were NOT merged in this pass, see LAB_NOTES.md's "sicherer Transfer"
warning: the lab and live primitives.py/relations.py diverged independently
and are not a safe drop-in replacement without a design decision on which
API the live consumers (stream.py, video.py, sources.py) should target."""

from __future__ import annotations

import numpy as np
import pytest

from fertig import diagnostics


def test_twonn_estimates_random_continuous_clouds():
    rng = np.random.RandomState(7)
    one_d = rng.random((500, 1))
    two_d = rng.random((700, 2))
    assert 0.6 < diagnostics.two_nn_intrinsic_dimension(one_d) < 1.6
    assert 1.2 < diagnostics.two_nn_intrinsic_dimension(two_d) < 3.2


def test_twonn_rejects_degenerate_relation_like_geometry():
    one_hot = np.eye(8)
    with pytest.raises(ValueError, match="degenerate|near-regular"):
        diagnostics.two_nn_intrinsic_dimension(one_hot)
