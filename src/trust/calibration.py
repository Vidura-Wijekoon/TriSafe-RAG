"""Confidence estimation, post-hoc calibration, and calibration metrics.

Two things live here:

* :class:`ConfidenceEstimator` -- the fixed weighted sum of five signals used
  by the abstention policy.  The retrieval signals can be taken from the raw
  RRF scores (``retrieval_signal="rrf"``, the legacy behaviour) or from the
  dense cosine similarity of the top hits (``"cosine"``).  RRF scores with
  ``k=60`` live in ``(0, 1/61]``, so a logistic squash of them is almost
  constant; the cosine variant is the normalised signal the paper recommends.
* :class:`PostHocCalibrator` -- Platt (logistic), isotonic, and temperature
  scaling fitted on a development split, used as external baselines and as an
  optional calibration step on top of the weighted sum.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

import numpy as np

from src.trust.grounding import GroundingResult


@dataclass
class ConfidenceSignals:
    """Raw signals fed into the confidence estimator (already normalised)."""

    retrieval_score: float       # top-1 retriever signal in [0, 1]
    retrieval_margin: float      # top-1 minus top-2, in [0, 1]
    grounding_support: float     # GroundingResult.overall_support
    citation_coverage: float     # GroundingResult.citation_coverage
    language_match: float        # 1.0 if at least one hit matches query language, else 0.0

    def as_vector(self) -> List[float]:
        return [
            self.retrieval_score,
            self.retrieval_margin,
            self.grounding_support,
            self.citation_coverage,
            self.language_match,
        ]


@dataclass
class ConfidenceEstimator:
    """Combines retrieval, grounding, and language signals into a [0, 1] score.

    ``retrieval_signal`` selects how the two retrieval signals are normalised:

    * ``"rrf"``    -- legacy: ``sigma(r1)`` and ``max(0, r1 - r2)`` on fused RRF
      scores.  Nearly constant for RRF with k=60 (documented limitation).
    * ``"cosine"`` -- dense cosine of the top-1 hit, linearly rescaled from
      ``[ret_lo, ret_hi]`` to ``[0, 1]``; margin = ``(cos1 - cos2) * margin_scale``
      clipped to ``[0, 1]``.
    """

    w_retrieval: float = 0.20
    w_margin: float = 0.20
    w_grounding: float = 0.40
    w_citation: float = 0.10
    w_language: float = 0.10
    retrieval_signal: str = "rrf"
    ret_lo: float = 0.0
    ret_hi: float = 1.0
    margin_scale: float = 1.0

    def normalise_retrieval(self, top1: float, top2: float) -> tuple[float, float]:
        if self.retrieval_signal == "cosine":
            span = max(self.ret_hi - self.ret_lo, 1e-6)
            s_ret = (top1 - self.ret_lo) / span
            s_margin = (top1 - top2) * self.margin_scale
        else:
            s_ret = 1 / (1 + math.exp(-top1))
            s_margin = top1 - top2
        return float(min(1.0, max(0.0, s_ret))), float(min(1.0, max(0.0, s_margin)))

    def estimate(self, signals: ConfidenceSignals) -> float:
        score = (
            self.w_retrieval * signals.retrieval_score
            + self.w_margin * signals.retrieval_margin
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
        s_ret, s_margin = self.normalise_retrieval(retrieval_top1, retrieval_top2)
        signals = ConfidenceSignals(
            retrieval_score=s_ret,
            retrieval_margin=s_margin,
            grounding_support=grounding.overall_support,
            citation_coverage=grounding.citation_coverage,
            language_match=1.0 if language_match else 0.0,
        )
        return self.estimate(signals), signals

    @property
    def weights(self) -> Dict[str, float]:
        return {
            "w_retrieval": self.w_retrieval,
            "w_margin": self.w_margin,
            "w_grounding": self.w_grounding,
            "w_citation": self.w_citation,
            "w_language": self.w_language,
        }


# --------------------------------------------------------------------------- #
# Calibration metrics
# --------------------------------------------------------------------------- #

def _bin_mask(confs: np.ndarray, lo: float, hi: float, first: bool) -> np.ndarray:
    return (confs >= lo) & (confs <= hi) if first else (confs > lo) & (confs <= hi)


def expected_calibration_error(
    confidences: Sequence[float],
    correct: Sequence[int],
    n_bins: int = 10,
) -> float:
    """Standard ECE: weighted gap between average confidence and accuracy per
    equal-width bin.  Bin ``b`` covers ``(b/B, (b+1)/B]``; a confidence of
    exactly 0 falls into the first bin."""
    if len(confidences) != len(correct):
        raise ValueError("confidences and correct must have equal length")
    if not len(confidences):
        return 0.0
    confs = np.asarray(confidences, dtype=np.float64)
    corr = np.asarray(correct, dtype=np.float64)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n = len(confs)
    for i in range(n_bins):
        mask = _bin_mask(confs, bins[i], bins[i + 1], i == 0)
        if not mask.any():
            continue
        ece += (mask.sum() / n) * abs(corr[mask].mean() - confs[mask].mean())
    return float(ece)


def reliability_bins(
    confidences: Sequence[float], correct: Sequence[int], n_bins: int = 10
) -> List[dict]:
    """Per-bin (count, mean confidence, accuracy) for reliability diagrams."""
    confs = np.asarray(confidences, dtype=np.float64)
    corr = np.asarray(correct, dtype=np.float64)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    out = []
    for i in range(n_bins):
        mask = _bin_mask(confs, bins[i], bins[i + 1], i == 0)
        out.append(
            {
                "lo": float(bins[i]),
                "hi": float(bins[i + 1]),
                "n": int(mask.sum()),
                "conf": float(confs[mask].mean()) if mask.any() else None,
                "acc": float(corr[mask].mean()) if mask.any() else None,
            }
        )
    return out


def brier_score(confidences: Sequence[float], correct: Sequence[int]) -> float:
    if not len(confidences):
        return 0.0
    c = np.asarray(confidences, dtype=np.float64)
    y = np.asarray(correct, dtype=np.float64)
    return float(np.mean((c - y) ** 2))


def auroc(scores: Sequence[float], labels: Sequence[int]) -> Optional[float]:
    """Rank-based AUROC (Mann-Whitney U) with tie handling.  ``None`` when a
    class is absent."""
    s = np.asarray(scores, dtype=np.float64)
    y = np.asarray(labels, dtype=np.int64)
    pos, neg = s[y == 1], s[y == 0]
    if len(pos) == 0 or len(neg) == 0:
        return None
    order = np.argsort(s, kind="mergesort")
    ranks = np.empty(len(s), dtype=np.float64)
    sorted_s = s[order]
    i = 0
    while i < len(s):
        j = i
        while j + 1 < len(s) and sorted_s[j + 1] == sorted_s[i]:
            j += 1
        ranks[order[i : j + 1]] = (i + j) / 2.0 + 1.0
        i = j + 1
    sum_pos = ranks[y == 1].sum()
    return float((sum_pos - len(pos) * (len(pos) + 1) / 2.0) / (len(pos) * len(neg)))


# --------------------------------------------------------------------------- #
# Post-hoc calibration (fit on the development split only)
# --------------------------------------------------------------------------- #

def _sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-z))


@dataclass
class PostHocCalibrator:
    """Platt / isotonic / temperature calibrators.

    * ``platt``       -- logistic regression on the 5-dim signal vector
      (L2-regularised, Newton iterations; no sklearn dependency).
    * ``isotonic``    -- monotone (PAV) map from a scalar score to P(correct).
    * ``temperature`` -- single scalar T on logit(scalar score).
    """

    method: str = "platt"
    l2: float = 1.0
    _coef: Optional[np.ndarray] = field(default=None, repr=False)
    _iso_x: Optional[np.ndarray] = field(default=None, repr=False)
    _iso_y: Optional[np.ndarray] = field(default=None, repr=False)
    _temperature: float = 1.0

    def fit(
        self,
        X: Sequence[Sequence[float]],
        y: Sequence[int],
        sums: Optional[Sequence[float]] = None,
    ) -> "PostHocCalibrator":
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        if self.method == "platt":
            self._coef = self._fit_logistic(np.hstack([X, np.ones((len(X), 1))]), y)
        elif self.method == "isotonic":
            s = np.asarray(sums if sums is not None else X[:, 0], dtype=np.float64)
            self._iso_x, self._iso_y = self._fit_isotonic(s, y)
        elif self.method == "temperature":
            s = np.asarray(sums if sums is not None else X[:, 0], dtype=np.float64)
            self._temperature = self._fit_temperature(s, y)
        else:
            raise ValueError(f"unknown method {self.method}")
        return self

    def predict(
        self, X: Sequence[Sequence[float]], sums: Optional[Sequence[float]] = None
    ) -> np.ndarray:
        X = np.asarray(X, dtype=np.float64)
        if self.method == "platt":
            Xb = np.hstack([X, np.ones((len(X), 1))])
            return _sigmoid(Xb @ self._coef)
        s = np.asarray(sums if sums is not None else X[:, 0], dtype=np.float64)
        if self.method == "isotonic":
            return np.interp(s, self._iso_x, self._iso_y)
        sc = np.clip(s, 1e-6, 1 - 1e-6)
        z = np.log(sc / (1 - sc))
        return _sigmoid(z / self._temperature)

    def describe(self) -> dict:
        if self.method == "platt":
            return {"method": "platt", "coef": [float(c) for c in self._coef]}
        if self.method == "isotonic":
            return {"method": "isotonic", "n_knots": int(len(self._iso_x))}
        return {"method": "temperature", "T": self._temperature}

    # ---- helpers -----------------------------------------------------------
    def _fit_logistic(self, Xb: np.ndarray, y: np.ndarray, iters: int = 100) -> np.ndarray:
        w = np.zeros(Xb.shape[1])
        reg = np.eye(Xb.shape[1]) * self.l2
        reg[-1, -1] = 0.0  # do not regularise the bias
        for _ in range(iters):
            p = _sigmoid(Xb @ w)
            grad = Xb.T @ (p - y) + reg @ w
            W = p * (1 - p)
            H = (Xb * W[:, None]).T @ Xb + reg + 1e-9 * np.eye(Xb.shape[1])
            step = np.linalg.solve(H, grad)
            w -= step
            if np.max(np.abs(step)) < 1e-8:
                break
        return w

    @staticmethod
    def _fit_isotonic(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        order = np.argsort(x, kind="mergesort")
        xs, ys = x[order], y[order]
        blocks = [[float(ys[i]), 1.0, float(xs[i]), float(xs[i])] for i in range(len(xs))]
        i = 0
        while i < len(blocks) - 1:
            if blocks[i][0] > blocks[i + 1][0]:
                m = (blocks[i][0] * blocks[i][1] + blocks[i + 1][0] * blocks[i + 1][1]) / (
                    blocks[i][1] + blocks[i + 1][1]
                )
                blocks[i] = [m, blocks[i][1] + blocks[i + 1][1], blocks[i][2], blocks[i + 1][3]]
                del blocks[i + 1]
                i = max(i - 1, 0)
            else:
                i += 1
        bx, by = [], []
        for m, _, lo, hi in blocks:
            bx.extend([lo, hi])
            by.extend([m, m])
        bx = np.asarray(bx) + np.arange(len(bx)) * 1e-9  # strictly increasing for interp
        return bx, np.asarray(by)

    @staticmethod
    def _fit_temperature(s: np.ndarray, y: np.ndarray) -> float:
        s = np.clip(s, 1e-6, 1 - 1e-6)
        z = np.log(s / (1 - s))
        best_t, best_nll = 1.0, float("inf")
        for t in np.exp(np.linspace(np.log(0.05), np.log(20.0), 200)):
            p = np.clip(_sigmoid(z / t), 1e-9, 1 - 1e-9)
            nll = -np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))
            if nll < best_nll:
                best_t, best_nll = float(t), float(nll)
        return best_t
