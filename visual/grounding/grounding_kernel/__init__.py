"""FERTIG Grounding Kernel v0/v1 research package.

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
from .closed_loop_programs import (
    ClosedLoopProgramExecutor,
    ClosedLoopProgramRecognizer,
    PerceptualTargetRole,
)
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
from .episode_binder import EpisodeConceptBinder
from .isolation import LearnerProcess, LearnerRunResult, run_isolated_learner
from .microworld import EvaluatorHarness, Microworld, WorldConfig
from .language import (
    Demonstration,
    GroundedLanguageLearner,
    GroundedReferent,
    OperationalMeaning,
    Resolution,
)
from .perceptual_policy import ObservationConditionedPolicy, VisualTargetSelector
from .processworld import ProcessHarness
from .v1_contracts import PublicTrace, PublicTransition, SessionManifest, Utterance
from .v1_isolation import IsolatedGrounder, commit_candidate_artifact
from .v1_runner import SealedEvaluationRunner

__version__ = "0.2.0"

__all__ = [
    "Action",
    "And",
    "Atom",
    "Binder",
    "BindingDecision",
    "CertificateScope",
    "ClosedLoopProgramExecutor",
    "ClosedLoopProgramRecognizer",
    "ConsequenceSignature",
    "EvaluatorHarness",
    "EpisodeConceptBinder",
    "GroundingCertificate",
    "GroundedLanguageLearner",
    "GroundedReferent",
    "Demonstration",
    "IsolatedGrounder",
    "LearnerProcess",
    "LearnerRunResult",
    "Microworld",
    "Not",
    "Observation",
    "ObservationConditionedPolicy",
    "OperationalMeaning",
    "OperationalPrototype",
    "Or",
    "PerceptualTargetRole",
    "ProcessHarness",
    "PublicTrace",
    "PublicTransition",
    "Relation",
    "SensorimotorBinder",
    "SealedEvaluationRunner",
    "SessionManifest",
    "Trajectory",
    "Transition",
    "TruthValue",
    "Resolution",
    "Utterance",
    "VisualTargetSelector",
    "WorldConfig",
    "evaluate",
    "commit_candidate_artifact",
    "least_fixed_point",
    "run_isolated_learner",
]
