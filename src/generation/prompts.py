"""Prompt templates used by the grounded answer generator."""

from __future__ import annotations

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
