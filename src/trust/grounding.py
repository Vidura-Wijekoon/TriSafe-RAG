"""Evidence grounding checks: verify each answer sentence is supported by the retrieved passages.

We expose two backends:
  * ``LexicalOverlapChecker`` — fast, zero-dependency, computes per-sentence
    token-overlap support; useful as a default and for CI.
  * ``NLIGroundingChecker`` — uses an off-the-shelf cross-lingual NLI model
    (XNLI / mDeBERTa) and treats ``entailment`` as support. Selected when
    the ``transformers`` extra is installed.

Both return a normalized ``GroundingResult`` so the abstention policy is
backend-agnostic.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import List, Optional, Protocol

from src.retrieval.vector_store import RetrievalHit

logger = logging.getLogger(__name__)

_SENT = re.compile(r"(?<=[\.\!\?।॥௺])\s+")
_TOK = re.compile(r"\w+", re.UNICODE)


@dataclass
class GroundingResult:
    """Per-sentence support breakdown for a generated answer."""

    overall_support: float          # mean per-sentence support, [0, 1]
    sentence_supports: List[float]  # support score for each sentence
    citation_coverage: float        # fraction of sentences that cite a passage
    n_unsupported: int              # sentences below the support threshold
    details: dict = field(default_factory=dict)


class GroundingChecker(Protocol):
    def check(self, answer: str, hits: List[RetrievalHit]) -> GroundingResult: ...


@dataclass
class LexicalOverlapChecker:
    """Token-overlap grounding.

    For each answer sentence we compute the maximum Jaccard overlap against
    any retrieved passage. The overall support is the mean across sentences.
    """

    support_threshold: float = 0.25

    def check(self, answer: str, hits: List[RetrievalHit]) -> GroundingResult:
        sents = [s.strip() for s in _SENT.split(answer or "") if s.strip()]
        if not sents:
            return GroundingResult(0.0, [], 0.0, 0)
        passage_tokens = [set(_TOK.findall(h.chunk.text.lower())) for h in hits]
        supports: List[float] = []
        citations = 0
        for s in sents:
            s_toks = set(_TOK.findall(s.lower()))
            if not s_toks:
                supports.append(0.0)
                continue
            best = 0.0
            for p_toks in passage_tokens:
                if not p_toks:
                    continue
                inter = len(s_toks & p_toks)
                union = len(s_toks | p_toks)
                best = max(best, inter / union if union else 0.0)
            supports.append(best)
            if re.search(r"\[\d+\]", s):
                citations += 1
        overall = sum(supports) / len(supports)
        n_unsupported = sum(1 for x in supports if x < self.support_threshold)
        return GroundingResult(
            overall_support=overall,
            sentence_supports=supports,
            citation_coverage=citations / len(sents),
            n_unsupported=n_unsupported,
            details={"checker": "lexical"},
        )


class NLIGroundingChecker:
    """Cross-lingual NLI grounding using a Hugging Face zero-shot model."""

    def __init__(self, model_name: str = "MoritzLaurer/mDeBERTa-v3-base-mnli-xnli",
                 support_threshold: float = 0.5) -> None:
        try:
            from transformers import pipeline  # type: ignore
        except ImportError as exc:  # pragma: no cover
            raise ImportError("transformers is required for NLIGroundingChecker") from exc
        self._pipe = pipeline("text-classification", model=model_name, top_k=None)
        self.support_threshold = support_threshold

    def check(self, answer: str, hits: List[RetrievalHit]) -> GroundingResult:
        sents = [s.strip() for s in _SENT.split(answer or "") if s.strip()]
        supports: List[float] = []
        citations = 0
        for s in sents:
            best = 0.0
            for h in hits:
                pair = f"{h.chunk.text} </s></s> {s}"
                preds = self._pipe(pair)[0]
                ent = next((p["score"] for p in preds if p["label"].lower().startswith("ent")), 0.0)
                best = max(best, ent)
            supports.append(best)
            if re.search(r"\[\d+\]", s):
                citations += 1
        if not sents:
            return GroundingResult(0.0, [], 0.0, 0)
        return GroundingResult(
            overall_support=sum(supports) / len(supports),
            sentence_supports=supports,
            citation_coverage=citations / len(sents),
            n_unsupported=sum(1 for x in supports if x < self.support_threshold),
            details={"checker": "nli"},
        )
