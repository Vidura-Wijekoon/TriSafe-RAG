"""Uncertainty-aware abstention policy.

Decision space: answer / defer / refuse.
- answer  → confidence above ``answer_threshold`` and grounding above ``min_support``
- refuse  → confidence below ``refuse_threshold`` OR zero retrieval recall
- defer   → middle band; the user is shown the top evidence and asked to confirm
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

Decision = Literal["answer", "defer", "refuse"]


@dataclass
class AbstentionDecision:
    decision: Decision
    confidence: float
    reason: str
    debug: dict = field(default_factory=dict)


@dataclass
class AbstentionPolicy:
    answer_threshold: float = 0.55
    refuse_threshold: float = 0.30
    min_support: float = 0.20
    min_citation_coverage: float = 0.40

    def decide(
        self,
        confidence: float,
        grounding_support: float,
        citation_coverage: float,
        has_evidence: bool,
    ) -> AbstentionDecision:
        debug = {
            "confidence": confidence,
            "grounding_support": grounding_support,
            "citation_coverage": citation_coverage,
            "has_evidence": has_evidence,
        }
        if not has_evidence:
            return AbstentionDecision("refuse", confidence, "no_evidence_retrieved", debug)
        if confidence < self.refuse_threshold:
            return AbstentionDecision("refuse", confidence, "low_confidence", debug)
        if grounding_support < self.min_support:
            return AbstentionDecision("refuse", confidence, "weak_grounding", debug)
        if confidence < self.answer_threshold or citation_coverage < self.min_citation_coverage:
            return AbstentionDecision("defer", confidence, "borderline_evidence", debug)
        return AbstentionDecision("answer", confidence, "ok", debug)
