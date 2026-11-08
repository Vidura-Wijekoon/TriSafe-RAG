"""Uncertainty-aware abstention policy.

Decision space: answer / defer / refuse.
- refuse  -> no retrieved evidence, the generator itself abstained, confidence
             below ``refuse_threshold``, or grounding below ``min_support``
- defer   -> middle band; the generated answer is withheld, the user is shown
             the top evidence and asked to confirm or escalate to a human officer
- answer  -> confidence >= ``answer_threshold`` and citation coverage >=
             ``min_citation_coverage``

Thresholds are global by default.  ``per_language`` optionally overrides any
threshold for a language group, which turns the fairness *audit* into
*enforcement* (evaluated as a separate configuration in the paper).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Literal, Optional

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
    respect_model_abstention: bool = True
    # e.g. {"ta": {"answer_threshold": 0.50}}
    per_language: Dict[str, Dict[str, float]] = field(default_factory=dict)

    def thresholds_for(self, language: Optional[str]) -> Dict[str, float]:
        base = {
            "answer_threshold": self.answer_threshold,
            "refuse_threshold": self.refuse_threshold,
            "min_support": self.min_support,
            "min_citation_coverage": self.min_citation_coverage,
        }
        if language and language in self.per_language:
            for k, v in self.per_language[language].items():
                if k in base:
                    base[k] = float(v)
        return base

    def decide(
        self,
        confidence: float,
        grounding_support: float,
        citation_coverage: float,
        has_evidence: bool,
        language: Optional[str] = None,
        model_abstained: bool = False,
    ) -> AbstentionDecision:
        th = self.thresholds_for(language)
        debug = {
            "confidence": confidence,
            "grounding_support": grounding_support,
            "citation_coverage": citation_coverage,
            "has_evidence": has_evidence,
            "model_abstained": model_abstained,
            "thresholds": th,
        }
        if not has_evidence:
            return AbstentionDecision("refuse", confidence, "no_evidence_retrieved", debug)
        if model_abstained and self.respect_model_abstention:
            return AbstentionDecision("refuse", confidence, "model_abstained", debug)
        if confidence < th["refuse_threshold"]:
            return AbstentionDecision("refuse", confidence, "low_confidence", debug)
        if grounding_support < th["min_support"]:
            return AbstentionDecision("refuse", confidence, "weak_grounding", debug)
        if confidence < th["answer_threshold"] or citation_coverage < th["min_citation_coverage"]:
            return AbstentionDecision("defer", confidence, "borderline_evidence", debug)
        return AbstentionDecision("answer", confidence, "ok", debug)
