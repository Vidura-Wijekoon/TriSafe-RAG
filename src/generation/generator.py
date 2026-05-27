"""Glue layer: takes retrieval hits, builds the prompt, calls the LLM."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import List

from src.generation.llm import LLM, GenerationResult
from src.generation.prompts import SYSTEM_PROMPT, build_grounded_prompt
from src.retrieval.vector_store import RetrievalHit


@dataclass
class GroundedAnswer:
    text: str
    cited_passages: List[int]
    raw_generation: GenerationResult
    is_abstention: bool = False
    abstention_reason: str = ""
    metadata: dict = field(default_factory=dict)


_CITATION_RE = re.compile(r"\[(\d+)\]")


@dataclass
class GroundedGenerator:
    llm: LLM
    max_new_tokens: int = 512

    def generate(self, query: str, hits: List[RetrievalHit]) -> GroundedAnswer:
        prompt = build_grounded_prompt(query, hits)
        result = self.llm.generate(SYSTEM_PROMPT, prompt, self.max_new_tokens)
        text = (result.text or "").strip()
        cited = [int(m) for m in _CITATION_RE.findall(text)]
        abstain = text.upper().startswith("INSUFFICIENT_EVIDENCE") or not hits
        return GroundedAnswer(
            text=text,
            cited_passages=cited,
            raw_generation=result,
            is_abstention=abstain,
            abstention_reason="no_evidence" if abstain and not hits else "model_abstained" if abstain else "",
        )
