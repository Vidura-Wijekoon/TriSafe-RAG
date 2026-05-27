"""Trust layer: grounding, calibration & abstention, fairness."""

from src.trust.grounding import GroundingChecker, GroundingResult
from src.trust.calibration import (
    ConfidenceEstimator,
    ConfidenceSignals,
    expected_calibration_error,
)
from src.trust.abstention import AbstentionPolicy, AbstentionDecision
from src.trust.fairness import FairnessAuditor, FairnessReport

__all__ = [
    "GroundingChecker",
    "GroundingResult",
    "ConfidenceEstimator",
    "ConfidenceSignals",
    "expected_calibration_error",
    "AbstentionPolicy",
    "AbstentionDecision",
    "FairnessAuditor",
    "FairnessReport",
]
