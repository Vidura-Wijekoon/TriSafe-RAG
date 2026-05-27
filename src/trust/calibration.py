"""Confidence estimation and Expected Calibration Error (ECE)."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

import numpy as np

from src.trust.grounding import GroundingResult


@dataclass
class ConfidenceSignals:
    """Raw signals fed into the confidence estimator."""

    retrieval_score: float       # top-1 retriever score (or fused score)
    retrieval_margin: float      # gap between top-1 and top-2 hits
    grounding_support: float     # GroundingResult.overall_support
    citation_coverage: float     # GroundingResult.citation_coverage
    language_match: float        # 1.0 if at least one hit matches query language, else 0.0


@dataclass
class ConfidenceEstimator:
    """Combines retrieval, grounding, and language signals into a [0, 1] score.

    The weights are configurable; defaults emphasise grounding (which is the
    strongest predictor of factuality in our ablations) followed by retrieval
    margin (separation between top hits) and language match.
    """

    w_retrieval: float = 0.20
    w_margin: float = 0.20
    w_grounding: float = 0.40
    w_citation: float = 0.10
    w_language: float = 0.10

    def estimate(self, signals: ConfidenceSignals) -> float:
        # Squash retrieval score into [0, 1] using a logistic.
        s_ret = 1 / (1 + math.exp(-signals.retrieval_score))
        s_margin = max(0.0, min(1.0, signals.retrieval_margin))
        score = (
            self.w_retrieval * s_ret
            + self.w_margin * s_margin
            + self.w_grounding * signals.grounding_support
            + self.w_citation * signals.citation_coverage
            + self.w_language * signals.language_match
        )
        return float(max(0.0, min(1.0, score)))

    def from_components(
        self,
        retrieval_top1: float,
        retrieval_top2: float,
        grounding: GroundingResult,
        language_match: bool,
    ) -> tuple[float, ConfidenceSignals]:
        signals = ConfidenceSignals(
            retrieval_score=retrieval_top1,
            retrieval_margin=max(0.0, retrieval_top1 - retrieval_top2),
            grounding_support=grounding.overall_support,
            citation_coverage=grounding.citation_coverage,
            language_match=1.0 if language_match else 0.0,
        )
        return self.estimate(signals), signals


def expected_calibration_error(
    confidences: Sequence[float],
    correct: Sequence[int],
    n_bins: int = 10,
) -> float:
    """Standard ECE: weighted gap between average confidence and accuracy per bin."""
    if len(confidences) != len(correct):
        raise ValueError("confidences and correct must have equal length")
    if not confidences:
        return 0.0
    confs = np.asarray(confidences, dtype=np.float64)
    corr = np.asarray(correct, dtype=np.float64)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n = len(confs)
    for i in range(n_bins):
        mask = (confs > bins[i]) & (confs <= bins[i + 1])
        if not mask.any():
            continue
        acc = corr[mask].mean()
        conf = confs[mask].mean()
        ece += (mask.sum() / n) * abs(acc - conf)
    return float(ece)
