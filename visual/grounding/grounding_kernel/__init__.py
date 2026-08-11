"""FERTIG Grounding Kernel v0.

This package is an isolated, deterministic reference experiment. It does not
claim universal or phenomenal meaning. It tests operational grounding inside
a sealed finite environment family with opaque token identities.
"""

from __future__ import annotations

from .binder import (
    Binder,
    BindingDecision,
    ConsequenceSignature,
    OperationalPrototype,
    SensorimotorBinder,
)
from .certificates import CertificateScope, GroundingCertificate
from .composition import (
    And,
    Atom,
    Not,
    Or,
    Relation,
    TruthValue,
    evaluate,
    least_fixed_point,
)
from .contracts import Action, Observation, Trajectory, Transition
from .isolation import LearnerProcess, LearnerRunResult, run_isolated_learner
from .microworld import EvaluatorHarness, Microworld, WorldConfig

__version__ = "0.1.0"

__all__ = [
    "Action",
    "And",
    "Atom",
    "Binder",
    "BindingDecision",
    "CertificateScope",
    "ConsequenceSignature",
    "EvaluatorHarness",
    "GroundingCertificate",
    "LearnerProcess",
    "LearnerRunResult",
    "Microworld",
    "Not",
    "Observation",
    "OperationalPrototype",
    "Or",
    "Relation",
    "SensorimotorBinder",
    "Trajectory",
    "Transition",
    "TruthValue",
    "WorldConfig",
    "evaluate",
    "least_fixed_point",
    "run_isolated_learner",
]
