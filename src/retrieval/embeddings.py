"""Embedding interface with a sentence-transformers backend and a hash fallback.

The fallback is deterministic and dependency-free, so the rest of the pipeline
can be exercised in CI or on machines without GPU access. In production runs,
``SentenceTransformerEmbedding`` (multilingual-e5 by default) is preferred.
"""

from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass
from typing import List, Protocol, Sequence

import numpy as np

logger = logging.getLogger(__name__)


class EmbeddingModel(Protocol):
    """Minimal interface every embedder must satisfy."""

    dim: int

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        ...


@dataclass
class HashEmbedding:
    """A deterministic hash-based bag-of-tokens embedder.

    Used as a fallback when no GPU / network is available. Not competitive
    with neural models but produces stable, comparable vectors for tests.
    """

    dim: int = 256
    seed: int = 1729

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for i, t in enumerate(texts):
            for tok in (t or "").lower().split():
                h = int(hashlib.md5(f"{self.seed}:{tok}".encode()).hexdigest(), 16)
                out[i, h % self.dim] += 1.0
        # L2 normalize so cosine == dot product
        norms = np.linalg.norm(out, axis=1, keepdims=True) + 1e-9
        return out / norms


class SentenceTransformerEmbedding:
    """Wraps a SentenceTransformer model. Multilingual by default."""

    def __init__(self, model_name: str = "intfloat/multilingual-e5-base") -> None:
        try:
            from sentence_transformers import SentenceTransformer  # type: ignore
        except ImportError as exc:  # pragma: no cover
            raise ImportError(
                "sentence-transformers is required for SentenceTransformerEmbedding"
            ) from exc
        self._model = SentenceTransformer(model_name)
        self.dim = self._model.get_sentence_embedding_dimension()
        self.model_name = model_name

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        # e5 models expect a "passage:" / "query:" prefix; callers can apply it.
        vecs = self._model.encode(
            list(texts), convert_to_numpy=True, normalize_embeddings=True
        )
        return vecs.astype(np.float32)


def get_embedder(name: str = "hash", **kwargs) -> EmbeddingModel:
    """Factory used by the pipeline; ``hash`` is the offline-safe default."""
    if name == "hash":
        return HashEmbedding(**kwargs)
    if name in {"st", "sentence-transformers"}:
        return SentenceTransformerEmbedding(**kwargs)
    raise ValueError(f"Unknown embedder: {name}")
