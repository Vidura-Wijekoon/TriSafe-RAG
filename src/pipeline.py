"""End-to-end TriSafe-RAG pipeline wiring.

This module is the single entry point that orchestrates the seven stages
described in the paper:

    Query -> Normalize/Detect -> Retrieve -> Generate (grounded) ->
    Ground-check -> Confidence -> Abstention policy -> Audit log

The pipeline is split into two halves so that evaluation can cache the
expensive half and replay the cheap half under many configurations:

* :meth:`TriRAGPipeline.draft`    -- retrieval + generation (LLM call)
* :meth:`TriRAGPipeline.finalize` -- grounding check, confidence, policy, audit

:meth:`TriRAGPipeline.query` simply chains the two.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import List, Optional

from src.audit.logger import AuditLogger, AuditRecord
from src.generation.generator import GroundedAnswer, GroundedGenerator
from src.ingestion.normalize import detect_language, normalize_text
from src.retrieval.hybrid import HybridRetriever
from src.retrieval.vector_store import RetrievalHit
from src.trust.abstention import AbstentionDecision, AbstentionPolicy
from src.trust.calibration import ConfidenceEstimator, ConfidenceSignals
from src.trust.grounding import GroundingChecker, GroundingResult, LexicalOverlapChecker

logger = logging.getLogger(__name__)

REFUSAL_TEXT = "I don't have enough verified evidence to answer this confidently."
DEFER_TEXT = (
    "I found some evidence but cannot verify the answer well enough to state it. "
    "Please review the cited passages below or ask a public-service officer to confirm."
)


@dataclass
class Draft:
    """Retrieval hits plus the raw generated answer, before the trust layer."""

    query: str
    normalized_query: str
    language: str
    hits: List[RetrievalHit]
    answer: GroundedAnswer
    metadata: dict = field(default_factory=dict)


@dataclass
class TriRAGResponse:
    query: str
    language: str
    hits: List[RetrievalHit]
    answer: GroundedAnswer
    grounding: GroundingResult
    confidence: float
    decision: AbstentionDecision
    audit: AuditRecord
    signals: Optional[ConfidenceSignals] = None
    metadata: dict = field(default_factory=dict)


@dataclass
class TriRAGPipeline:
    retriever: HybridRetriever
    generator: GroundedGenerator
    grounding_checker: GroundingChecker = field(default_factory=LexicalOverlapChecker)
    confidence: ConfidenceEstimator = field(default_factory=ConfidenceEstimator)
    abstention: AbstentionPolicy = field(default_factory=AbstentionPolicy)
    audit_logger: Optional[AuditLogger] = None
    top_k: int = 5

    # ------------------------------------------------------------------ #
    def draft(self, raw_query: str, language: Optional[str] = None) -> Draft:
        q = normalize_text(raw_query)
        lang = language or detect_language(q)
        hits = self.retriever.retrieve(q, top_k=self.top_k, language=None)
        answer = self.generator.generate(q, hits)
        return Draft(query=raw_query, normalized_query=q, language=lang, hits=hits, answer=answer)

    def finalize(self, draft: Draft, grounding: Optional[GroundingResult] = None) -> TriRAGResponse:
        hits, answer, lang = draft.hits, draft.answer, draft.language
        if grounding is None:
            grounding = self.grounding_checker.check(answer.text, hits)

        top1 = hits[0].dense_score if (hits and self.confidence.retrieval_signal == "cosine") else (hits[0].score if hits else 0.0)
        top2 = (
            hits[1].dense_score if (len(hits) > 1 and self.confidence.retrieval_signal == "cosine")
            else (hits[1].score if len(hits) > 1 else 0.0)
        )
        lang_match = any((h.chunk.language or "") in (lang, "mixed") for h in hits) if hits else False
        conf, signals = self.confidence.from_components(top1, top2, grounding, lang_match)
        decision = self.abstention.decide(
            confidence=conf,
            grounding_support=grounding.overall_support,
            citation_coverage=grounding.citation_coverage,
            has_evidence=bool(hits),
            language=lang,
            model_abstained=answer.is_abstention,
        )

        final_text = answer.text
        if decision.decision != "answer":
            answer.is_abstention = True
            answer.abstention_reason = decision.reason
            final_text = REFUSAL_TEXT if decision.decision == "refuse" else DEFER_TEXT

        record = AuditRecord(
            timestamp=AuditLogger.now(),
            query=draft.query,
            language=lang,
            retrieved_chunk_ids=[h.chunk.chunk_id for h in hits],
            retrieval_scores=[h.score for h in hits],
            grounding_support=grounding.overall_support,
            citation_coverage=grounding.citation_coverage,
            confidence=conf,
            decision=decision.decision,
            reason=decision.reason,
            answer_hash=AuditLogger.hash_answer(final_text),
            model=answer.raw_generation.model,
            extra={
                "dense_scores": [h.dense_score for h in hits],
                "signals": signals.as_vector(),
                "n_unsupported": grounding.n_unsupported,
                "draft_answer_hash": AuditLogger.hash_answer(answer.text),
                "source_paths": [h.chunk.metadata.get("source_path", h.chunk.doc_id) for h in hits],
            },
        )
        if self.audit_logger:
            self.audit_logger.log(record)

        # keep the draft text available for evaluation; surface the final text to callers
        answer.metadata.setdefault("draft_text", answer.text)
        answer.text = final_text
        return TriRAGResponse(
            query=draft.query,
            language=lang,
            hits=hits,
            answer=answer,
            grounding=grounding,
            confidence=conf,
            decision=decision,
            audit=record,
            signals=signals,
        )

    def query(self, raw_query: str, language: Optional[str] = None) -> TriRAGResponse:
        return self.finalize(self.draft(raw_query, language=language))
