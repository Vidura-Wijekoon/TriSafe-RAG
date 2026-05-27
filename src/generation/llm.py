"""LLM wrappers: a deterministic echo backend for tests and an OpenAI backend."""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from typing import List, Optional, Protocol

logger = logging.getLogger(__name__)


@dataclass
class GenerationResult:
    text: str
    tokens_in: int = 0
    tokens_out: int = 0
    logprobs: Optional[List[float]] = None
    model: str = ""
    finish_reason: str = "stop"
    raw: dict = field(default_factory=dict)


class LLM(Protocol):
    name: str

    def generate(self, system: str, user: str, max_new_tokens: int = 512) -> GenerationResult:
        ...


class EchoLLM:
    """Returns a deterministic extractive answer from the evidence block.

    Used in tests and offline runs. It picks the longest sentence overlapping
    the query keywords from the provided evidence.
    """

    name: str = "echo"

    def generate(self, system: str, user: str, max_new_tokens: int = 512) -> GenerationResult:
        evidence_section = ""
        if "Evidence:\n" in user:
            evidence_section = user.split("Evidence:\n", 1)[1].split("\n\nQuestion:")[0]
        question = user.split("Question:", 1)[-1].split("\n", 1)[0].strip()

        sentences = [s.strip() for s in evidence_section.replace("\n", " ").split(".") if s.strip()]
        if not sentences:
            return GenerationResult(text="INSUFFICIENT_EVIDENCE", model=self.name, finish_reason="abstain")

        q_tokens = {t.lower() for t in question.split() if len(t) > 2}
        scored = [
            (len(q_tokens & {t.lower() for t in s.split()}), -i, i, s)
            for i, s in enumerate(sentences)
        ]
        scored.sort(reverse=True)
        best = scored[0]
        if best[0] == 0:
            return GenerationResult(text="INSUFFICIENT_EVIDENCE", model=self.name, finish_reason="abstain")
        ans = best[3]
        return GenerationResult(
            text=f"{ans}. [1]",
            tokens_in=len(user.split()),
            tokens_out=len(ans.split()) + 1,
            model=self.name,
        )


class OpenAILLM:
    """Thin wrapper around the official OpenAI chat completions API."""

    def __init__(self, model: str = "gpt-4o-mini", api_key: Optional[str] = None) -> None:
        try:
            from openai import OpenAI  # type: ignore
        except ImportError as exc:  # pragma: no cover
            raise ImportError("openai>=1.0 is required for OpenAILLM") from exc

        self._client = OpenAI(api_key=api_key or os.getenv("OPENAI_API_KEY"))
        self.model = model
        self.name = f"openai:{model}"

    def generate(self, system: str, user: str, max_new_tokens: int = 512) -> GenerationResult:
        resp = self._client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            max_tokens=max_new_tokens,
            temperature=0.2,
        )
        choice = resp.choices[0]
        usage = getattr(resp, "usage", None)
        return GenerationResult(
            text=choice.message.content or "",
            tokens_in=getattr(usage, "prompt_tokens", 0) if usage else 0,
            tokens_out=getattr(usage, "completion_tokens", 0) if usage else 0,
            model=self.name,
            finish_reason=choice.finish_reason or "stop",
        )


def get_llm(name: str = "echo", **kwargs) -> LLM:
    if name == "echo":
        return EchoLLM()
    if name in {"openai", "gpt"}:
        return OpenAILLM(**kwargs)
    raise ValueError(f"Unknown LLM backend: {name}")
