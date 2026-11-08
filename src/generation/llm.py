"""LLM wrappers.

* :class:`EchoLLM`     -- deterministic extractive backend for tests / offline CI.
* :class:`HFLocalLLM`  -- any open-weight chat model via ``transformers``
                          (default Qwen2.5-1.5B-Instruct), runs on CPU or GPU.
* :class:`OpenAILLM`   -- hosted chat-completions backend.

All backends implement :meth:`generate`.  Two auxiliary prompts used by the
external baselines, verbalized confidence and query translation, are exposed
as module-level helpers so every backend shares the exact same wording.
"""

from __future__ import annotations

import logging
import os
import re
import time
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


# --------------------------------------------------------------------------- #
# Shared auxiliary prompts (external baselines)
# --------------------------------------------------------------------------- #

CONFIDENCE_SYSTEM = (
    "You are a careful evaluator. You will see a question, numbered evidence "
    "passages, and a draft answer. Estimate how likely it is that the draft "
    "answer is correct AND fully supported by the evidence. Reply with a single "
    "integer from 0 to 100 and nothing else."
)

TRANSLATE_SYSTEM = (
    "You are a professional translator. Translate the user's question into "
    "English. Reply with the English translation only, no explanations."
)

_INT_RE = re.compile(r"\d{1,3}")


def build_confidence_prompt(evidence_block: str, question: str, draft_answer: str) -> str:
    return (
        f"Evidence:\n{evidence_block}\n\nQuestion: {question}\n\n"
        f"Draft answer: {draft_answer}\n\nConfidence (0-100):"
    )


def parse_verbalized_confidence(text: str) -> Optional[float]:
    m = _INT_RE.search(text or "")
    if not m:
        return None
    v = int(m.group(0))
    return float(min(100, max(0, v))) / 100.0


# --------------------------------------------------------------------------- #
# Backends
# --------------------------------------------------------------------------- #

class EchoLLM:
    """Returns a deterministic extractive answer from the evidence block.

    Used in tests and offline runs. It picks the sentence with the largest
    keyword overlap with the question from the provided evidence and appends a
    citation to passage 1.  It cannot fabricate text that is not in the
    evidence, which is why it is only a smoke-test backend.
    """

    name: str = "echo"

    def generate(self, system: str, user: str, max_new_tokens: int = 512) -> GenerationResult:
        if system == CONFIDENCE_SYSTEM:
            return GenerationResult(text="50", model=self.name)
        if system == TRANSLATE_SYSTEM:
            return GenerationResult(text=user.strip(), model=self.name)

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


class HFLocalLLM:
    """Open-weight chat model through ``transformers``.

    ``seed`` controls sampling so that repeated runs at ``temperature > 0``
    are reproducible; each call advances an internal counter so that
    different queries do not share the same random stream.
    """

    def __init__(
        self,
        model: str = "Qwen/Qwen2.5-1.5B-Instruct",
        temperature: float = 0.2,
        top_p: float = 0.95,
        seed: int = 0,
        device: Optional[str] = None,
        dtype: str = "auto",
        max_input_tokens: int = 3072,
    ) -> None:
        try:
            import torch  # type: ignore
            from transformers import AutoModelForCausalLM, AutoTokenizer  # type: ignore
        except ImportError as exc:  # pragma: no cover
            raise ImportError("torch and transformers are required for HFLocalLLM") from exc

        self._torch = torch
        self.model_name = model
        self.name = f"hf:{model}"
        self.temperature = temperature
        self.top_p = top_p
        self.seed = seed
        self._calls = 0
        self.max_input_tokens = max_input_tokens
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        torch_dtype = {
            "auto": (torch.bfloat16 if self.device == "cuda" else torch.float32),
            "float32": torch.float32,
            "bfloat16": torch.bfloat16,
            "float16": torch.float16,
        }[dtype]
        self._tok = AutoTokenizer.from_pretrained(model)
        self._model = AutoModelForCausalLM.from_pretrained(model, torch_dtype=torch_dtype)
        self._model.to(self.device)
        self._model.eval()

    def generate(self, system: str, user: str, max_new_tokens: int = 512) -> GenerationResult:
        torch = self._torch
        messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
        text = self._tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self._tok(text, return_tensors="pt", truncation=True, max_length=self.max_input_tokens)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        self._calls += 1
        gen_kwargs = dict(max_new_tokens=max_new_tokens, pad_token_id=self._tok.eos_token_id)
        if self.temperature > 0:
            torch.manual_seed(self.seed * 1_000_003 + self._calls)
            gen_kwargs.update(do_sample=True, temperature=self.temperature, top_p=self.top_p)
        else:
            gen_kwargs.update(do_sample=False)
        with torch.no_grad():
            out = self._model.generate(**inputs, **gen_kwargs)
        new_tokens = out[0][inputs["input_ids"].shape[1]:]
        completion = self._tok.decode(new_tokens, skip_special_tokens=True).strip()
        return GenerationResult(
            text=completion,
            tokens_in=int(inputs["input_ids"].shape[1]),
            tokens_out=int(len(new_tokens)),
            model=self.name,
            finish_reason="stop",
        )


class OpenAILLM:
    """Thin wrapper around the official OpenAI chat completions API."""

    def __init__(
        self,
        model: str = "gpt-4o-mini",
        api_key: Optional[str] = None,
        temperature: float = 0.2,
        seed: Optional[int] = None,
    ) -> None:
        try:
            from openai import OpenAI  # type: ignore
        except ImportError as exc:  # pragma: no cover
            raise ImportError("openai>=1.0 is required for OpenAILLM") from exc

        self._client = OpenAI(api_key=api_key or os.getenv("OPENAI_API_KEY"))
        self.model = model
        self.name = f"openai:{model}"
        self.temperature = temperature
        self.seed = seed

    def generate(self, system: str, user: str, max_new_tokens: int = 512) -> GenerationResult:
        kwargs = {}
        if self.seed is not None:
            kwargs["seed"] = self.seed
        resp = self._client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            max_tokens=max_new_tokens,
            temperature=self.temperature,
            **kwargs,
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


class OllamaLLM:
    """Local open-weight models served by Ollama (llama.cpp) over its HTTP API.

    Uses ``POST /api/chat`` with ``stream=False``.  ``seed`` and ``temperature``
    are passed as sampling options so that runs are reproducible per seed;
    both are plain attributes and may be changed between calls (the benchmark
    does this to draw self-consistency samples at a higher temperature).
    """

    def __init__(
        self,
        model: str = "qwen2.5:7b-instruct",
        temperature: float = 0.2,
        top_p: float = 0.95,
        seed: Optional[int] = None,
        host: Optional[str] = None,
        num_ctx: int = 4096,
        timeout: float = 600.0,
        **_ignored,
    ) -> None:
        import json as _json
        import urllib.request

        self._json = _json
        self._request = urllib.request
        self.host = (host or os.getenv("OLLAMA_HOST") or "http://localhost:11434").rstrip("/")
        self.model = model
        self.name = f"ollama:{model}"
        self.temperature = temperature
        self.top_p = top_p
        self.seed = seed
        self.num_ctx = num_ctx
        self.timeout = timeout

    def generate(self, system: str, user: str, max_new_tokens: int = 512) -> GenerationResult:
        options = {"temperature": self.temperature, "top_p": self.top_p, "num_predict": max_new_tokens,
                   "num_ctx": self.num_ctx}
        if self.seed is not None:
            options["seed"] = int(self.seed)
        payload = self._json.dumps({
            "model": self.model, "stream": False, "options": options,
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
        }).encode("utf-8")
        req = self._request.Request(f"{self.host}/api/chat", data=payload,
                                    headers={"Content-Type": "application/json"}, method="POST")
        last_exc: Optional[Exception] = None
        for attempt in range(3):
            try:
                with self._request.urlopen(req, timeout=self.timeout) as resp:
                    data = self._json.loads(resp.read().decode("utf-8"))
                break
            except Exception as exc:  # pragma: no cover - network
                last_exc = exc
                time.sleep(2.0 * (attempt + 1))
        else:
            raise RuntimeError(f"Ollama request failed: {last_exc}")
        return GenerationResult(
            text=(data.get("message") or {}).get("content", "") or "",
            tokens_in=int(data.get("prompt_eval_count", 0) or 0),
            tokens_out=int(data.get("eval_count", 0) or 0),
            model=self.name,
            finish_reason=str(data.get("done_reason", "stop")),
        )


def get_llm(name: str = "echo", **kwargs) -> LLM:
    if name == "echo":
        return EchoLLM()
    if name in {"hf", "local", "transformers"}:
        return HFLocalLLM(**kwargs)
    if name in {"openai", "gpt"}:
        return OpenAILLM(**kwargs)
    if name == "ollama":
        return OllamaLLM(**kwargs)
    raise ValueError(f"Unknown LLM backend: {name}")
