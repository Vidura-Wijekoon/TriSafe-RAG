"""Retrieval and abstention metrics."""

from __future__ import annotations

from typing import Sequence


def precision_at_k(predicted: Sequence[str], relevant: Sequence[str], k: int = 5) -> float:
    if k <= 0 or not predicted:
        return 0.0
    top = predicted[:k]
    return sum(1 for p in top if p in set(relevant)) / k


def mean_reciprocal_rank(predicted: Sequence[str], relevant: Sequence[str]) -> float:
    rel = set(relevant)
    for i, p in enumerate(predicted, start=1):
        if p in rel:
            return 1.0 / i
    return 0.0


def abstention_precision_recall(
    abstained: Sequence[int],
    gold_unanswerable: Sequence[int],
) -> tuple[float, float, float]:
    """Returns (precision, recall, f1) for the abstention decision treating
    "unanswerable" as the positive class."""
    if len(abstained) != len(gold_unanswerable):
        raise ValueError("length mismatch")
    tp = sum(1 for a, g in zip(abstained, gold_unanswerable) if a == 1 and g == 1)
    fp = sum(1 for a, g in zip(abstained, gold_unanswerable) if a == 1 and g == 0)
    fn = sum(1 for a, g in zip(abstained, gold_unanswerable) if a == 0 and g == 1)
    p = tp / (tp + fp) if (tp + fp) else 0.0
    r = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * p * r / (p + r) if (p + r) else 0.0
    return p, r, f1
