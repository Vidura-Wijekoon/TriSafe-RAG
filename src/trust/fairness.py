"""Cross-lingual fairness auditing.

Given per-query outcomes labelled by language, the auditor computes
group-wise metrics, disparity gaps, and an Equalized Odds proxy across the
language groups (Sinhala, Tamil, English) plus the ``mixed`` code-switched
bucket.

Per group ``g`` the auditor reports: number of queries, accuracy on the
gold-answerable queries (abstentions count as errors), *selective* accuracy
(accuracy among the answerable queries that were answered), answer rate,
abstention rate, TPR (answer rate on gold-answerable queries), FPR (answer
rate on gold-unanswerable queries), and mean grounding support.  The gaps are
max minus min over groups.  Groups below ``min_group_size`` are reported but
excluded from the gap computations, because a gap over two queries is noise.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence

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
    per_group_answer_rate: Dict[str, float] = field(default_factory=dict)
    per_group_selective_accuracy: Dict[str, Optional[float]] = field(default_factory=dict)
    per_group_tpr: Dict[str, Optional[float]] = field(default_factory=dict)
    per_group_fpr: Dict[str, Optional[float]] = field(default_factory=dict)
    n_answerable_per_group: Dict[str, int] = field(default_factory=dict)
    n_unanswerable_per_group: Dict[str, int] = field(default_factory=dict)
    max_gap_answer_rate: float = 0.0
    max_gap_tpr: float = 0.0
    max_gap_fpr: float = 0.0
    groups_in_gaps: List[str] = field(default_factory=list)


def _safe_mean(xs: Sequence[float]) -> Optional[float]:
    return float(np.mean(xs)) if len(xs) else None


@dataclass
class FairnessAuditor:
    """Aggregates per-query records into a fairness report."""

    languages: Iterable[str] = ("si", "ta", "en", "mixed")
    min_group_size: int = 5

    def audit(self, records: List[dict]) -> FairnessReport:
        """Each record needs: language, correct (0/1), abstained (0/1),
        grounding_support (float), gold_answerable (0/1)."""
        by_lang: Dict[str, List[dict]] = {l: [] for l in self.languages}
        for r in records:
            by_lang.setdefault(r.get("language", "unknown"), []).append(r)

        per_acc: Dict[str, float] = {}
        per_abs: Dict[str, float] = {}
        per_gr: Dict[str, float] = {}
        per_ans: Dict[str, float] = {}
        per_sel: Dict[str, Optional[float]] = {}
        n_per: Dict[str, int] = {}
        n_ans: Dict[str, int] = {}
        n_unans: Dict[str, int] = {}
        tpr: Dict[str, Optional[float]] = {}
        fpr: Dict[str, Optional[float]] = {}

        for lang, rs in by_lang.items():
            if not rs:
                continue
            n_per[lang] = len(rs)
            positives = [r for r in rs if int(r.get("gold_answerable", 1)) == 1]
            negatives = [r for r in rs if int(r.get("gold_answerable", 1)) == 0]
            n_ans[lang], n_unans[lang] = len(positives), len(negatives)
            per_acc[lang] = _safe_mean([int(r.get("correct", 0)) for r in positives]) or 0.0
            answered_pos = [r for r in positives if not int(r.get("abstained", 0))]
            per_sel[lang] = _safe_mean([int(r.get("correct", 0)) for r in answered_pos])
            per_abs[lang] = _safe_mean([int(r.get("abstained", 0)) for r in rs]) or 0.0
            per_ans[lang] = 1.0 - per_abs[lang]
            per_gr[lang] = _safe_mean([float(r.get("grounding_support", 0.0)) for r in rs]) or 0.0
            tpr[lang] = _safe_mean([1 - int(r.get("abstained", 0)) for r in positives])
            fpr[lang] = _safe_mean([1 - int(r.get("abstained", 0)) for r in negatives])

        eligible = [l for l, n in n_per.items() if n >= self.min_group_size] or list(n_per)

        def _gap(d: Dict[str, Optional[float]], groups: List[str]) -> float:
            vals = [d[g] for g in groups if g in d and d[g] is not None]
            return float(max(vals) - min(vals)) if len(vals) >= 2 else 0.0

        gap_acc = _gap(per_acc, eligible)
        gap_tpr = _gap(tpr, eligible)
        gap_fpr = _gap(fpr, eligible)
        return FairnessReport(
            per_group_accuracy=per_acc,
            per_group_abstention=per_abs,
            per_group_grounding=per_gr,
            max_gap_accuracy=gap_acc,
            max_gap_grounding=_gap(per_gr, eligible),
            equalized_odds_gap=max(gap_tpr, gap_fpr),
            cross_lingual_consistency=max(0.0, 1.0 - gap_acc),
            n_per_group=n_per,
            per_group_answer_rate=per_ans,
            per_group_selective_accuracy=per_sel,
            per_group_tpr=tpr,
            per_group_fpr=fpr,
            n_answerable_per_group=n_ans,
            n_unanswerable_per_group=n_unans,
            max_gap_answer_rate=_gap(per_ans, eligible),
            max_gap_tpr=gap_tpr,
            max_gap_fpr=gap_fpr,
            groups_in_gaps=eligible,
        )
