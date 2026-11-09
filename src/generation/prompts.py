"""Prompt templates used by the grounded answer generator."""

from __future__ import annotations

import re
from typing import List

from src.retrieval.vector_store import RetrievalHit

SYSTEM_PROMPT = (
    "You are a multilingual public-service assistant. "
    "Answer ONLY using the numbered evidence passages. "
    "If the passages do not contain a clear answer, reply exactly with: "
    "INSUFFICIENT_EVIDENCE. "
    "Always include inline citations of the form [#] that refer to the passage numbers. "
    "Respond in the same language as the user query."
)


ABSTAIN_MARK = "INSUFFICIENT_EVIDENCE"


def is_model_abstention(text: str) -> bool:
    """True when the generator emitted the abstention string anywhere in its reply.

    The prompt asks for the literal ``INSUFFICIENT_EVIDENCE``; small instruction-tuned
    models often explain first ("The evidence does not state ... INSUFFICIENT_EVIDENCE")
    or prefix a citation ("[2] INSUFFICIENT_EVIDENCE").  An earlier version only
    recognised replies that *started* with the string and counted such refusals as
    wrong answers; cached drafts are re-detected with this function on load.
    """
    return ABSTAIN_MARK in (text or "").upper()


_MARKER_RE = re.compile(r"\[#?\d+\]|#\d*|\bAnswer:|[\s\W\d_]+", re.UNICODE)


def is_empty_reply(text: str) -> bool:
    """True when the reply carries no content: only citation markers such as
    ``[1] #1``, digits, punctuation and whitespace.  Small generators emit such
    replies instead of an answer or an abstention; they are counted as wrong
    answers by alias matching and are the largest class of unsafe deliveries
    in Sinhala and Tamil in our adjudication (scripts/adjudicate.py)."""
    if is_model_abstention(text):
        return False
    return _MARKER_RE.sub("", text or "") == ""


def format_evidence(hits: List[RetrievalHit]) -> str:
    blocks = []
    for h in hits:
        tag = f"[{h.rank + 1}]"
        src = h.chunk.metadata.get("source_path", h.chunk.doc_id)
        blocks.append(f"{tag} (source: {src})\n{h.chunk.text}")
    return "\n\n".join(blocks)


def build_grounded_prompt(query: str, hits: List[RetrievalHit]) -> str:
    """Build the user-side prompt for the LLM."""
    evidence = format_evidence(hits) if hits else "(no evidence retrieved)"
    return (
        f"Evidence:\n{evidence}\n\n"
        f"Question: {query}\n\n"
        f"Answer using the evidence above. Cite passages by number."
    )
