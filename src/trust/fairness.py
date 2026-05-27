"""Cross-lingual fairness auditing.

Given per-query outcomes labelled by language, the auditor computes
group-wise metric averages, disparity gaps, and the Equalized Odds gap
across the three language groups (Sinhala, Tamil, English) plus the
``mixed`` code-switched bucket.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Sequence

import numpy as np


@dataclass
class FairnessReport:
    per_group_accuracy: Dict[str, float]
    per_group_abstention: Dict[str, float]
    per_group_grounding: Dict[str, float]
    max_gap_accuracy: float
    max_gap_grounding: float
    equalized_odds_gap: float
    cross_lingual_consistency: float
    n_per_group: Dict[str, int] = field(default_factory=dict)


def _safe_mean(xs: Sequence[float]) -> float:
    return float(np.mean(xs)) if len(xs) else 0.0


@dataclass
class FairnessAuditor:
    """Aggregates per-query records into a fairness report."""

    languages: Iterable[str] = ("si", "ta", "en", "mixed")

    def audit(self, records: List[dict]) -> FairnessReport:
        """Each record needs: language, correct (0/1), abstained (0/1),
        grounding_support (float)."""
        by_lang: Dict[str, List[dict]] = {l: [] for l in self.languages}
        for r in records:
            lang = r.get("language", "unknown")
            if lang not in by_lang:
                by_lang.setdefault(lang, [])
            by_lang.setdefault(lang, []).append(r)

        per_acc: Dict[str, float] = {}
        per_abs: Dict[str, float] = {}
        per_gr: Dict[str, float] = {}
        n_per: Dict[str, int] = {}
        tpr: Dict[str, float] = {}
        fpr: Dict[str, float] = {}

        for lang, rs in by_lang.items():
            if not rs:
                continue
            n_per[lang] = len(rs)
            per_acc[lang] = _safe_mean([r.get("correct", 0) for r in rs])
            per_abs[lang] = _safe_mean([r.get("abstained", 0) for r in rs])
            per_gr[lang] = _safe_mean([r.get("grounding_support", 0.0) for r in rs])

            # Equalized-odds proxies using answer/abstention as the decision.
            positives = [r for r in rs if r.get("gold_answerable", 1) == 1]
            negatives = [r for r in rs if r.get("gold_answerable", 1) == 0]
            tpr[lang] = _safe_mean([1 - r.get("abstained", 0) for r in positives]) if positives else 0.0
            fpr[lang] = _safe_mean([1 - r.get("abstained", 0) for r in negatives]) if negatives else 0.0

        def _gap(d: Dict[str, float]) -> float:
            return float(max(d.values()) - min(d.values())) if d else 0.0

        eo = max(_gap(tpr), _gap(fpr))
        # Consistency = 1 - max accuracy gap (higher is more equal).
        consistency = max(0.0, 1.0 - _gap(per_acc))

        return FairnessReport(
            per_group_accuracy=per_acc,
            per_group_abstention=per_abs,
            per_group_grounding=per_gr,
            max_gap_accuracy=_gap(per_acc),
            max_gap_grounding=_gap(per_gr),
            equalized_odds_gap=eo,
            cross_lingual_consistency=consistency,
            n_per_group=n_per,
        )
