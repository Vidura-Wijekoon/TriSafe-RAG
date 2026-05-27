"""End-to-end TriSafe-RAG pipeline wiring.

This module is the single entry point that orchestrates the seven stages
described in the paper:

    Query → Normalize/Detect → Retrieve → Ground-check → Confidence →
    Generate (grounded) → Abstention policy → Audit log

Each stage can be swapped out via the constructor (dependency injection),
which keeps the ablation studies in ``src/eval/runner.py`` clean.
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
from src.trust.calibration import ConfidenceEstimator
from src.trust.grounding import GroundingChecker, GroundingResult, LexicalOverlapChecker

logger = logging.getLogger(__name__)


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

    def query(self, raw_query: str, language: Optional[str] = None) -> TriRAGResponse:
        # 1. Normalize + detect language.
        q = normalize_text(raw_query)
        lang = language or detect_language(q)

        # 2. Retrieve.
        hits = self.retriever.retrieve(q, top_k=self.top_k, language=None)

        # 3. Generate a grounded draft.
        answer = self.generator.generate(q, hits)

        # 4. Ground-check the draft.
        grounding = self.grounding_checker.check(answer.text, hits)

        # 5. Confidence + abstention decision.
        top1 = hits[0].score if hits else 0.0
        top2 = hits[1].score if len(hits) > 1 else 0.0
        lang_match = any(
            (h.chunk.language or "") in (lang, "mixed") for h in hits
        ) if hits else False
        conf, _ = self.confidence.from_components(top1, top2, grounding, lang_match)
        decision = self.abstention.decide(
            confidence=conf,
            grounding_support=grounding.overall_support,
            citation_coverage=grounding.citation_coverage,
            has_evidence=bool(hits),
        )

        # 6. Override the answer when policy says refuse/defer.
        if decision.decision != "answer":
            answer.is_abstention = True
            answer.abstention_reason = decision.reason
            if decision.decision == "refuse":
                answer.text = "I don't have enough verified evidence to answer this confidently."

        # 7. Audit.
        record = AuditRecord(
            timestamp=AuditLogger.now(),
            query=raw_query,
            language=lang,
            retrieved_chunk_ids=[h.chunk.chunk_id for h in hits],
            retrieval_scores=[h.score for h in hits],
            grounding_support=grounding.overall_support,
            citation_coverage=grounding.citation_coverage,
            confidence=conf,
            decision=decision.decision,
            reason=decision.reason,
            answer_hash=AuditLogger.hash_answer(answer.text),
            model=answer.raw_generation.model,
        )
        if self.audit_logger:
            self.audit_logger.log(record)

        return TriRAGResponse(
            query=raw_query,
            language=lang,
            hits=hits,
            answer=answer,
            grounding=grounding,
            confidence=conf,
            decision=decision,
            audit=record,
        )
