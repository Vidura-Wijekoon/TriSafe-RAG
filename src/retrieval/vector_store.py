"""Tiny in-memory vector store (cosine similarity).

For larger corpora swap in FAISS or Chroma using the same ``search`` signature.
"""

from __future__ import annotations

import pickle
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Sequence

import numpy as np

from src.ingestion.chunker import Chunk


@dataclass
class RetrievalHit:
    """One retrieval result with its similarity score and source chunk."""

    chunk: Chunk
    score: float
    rank: int


@dataclass
class InMemoryVectorStore:
    """A NumPy-backed cosine-similarity store.

    Vectors are expected to be L2-normalized, so similarity reduces to a dot
    product. The store keeps chunks in insertion order; ``search`` returns the
    top-k highest-scoring chunks, optionally filtered by language.
    """

    dim: int
    chunks: List[Chunk] = field(default_factory=list)
    matrix: Optional[np.ndarray] = None

    def add(self, chunks: Sequence[Chunk], vectors: np.ndarray) -> None:
        if vectors.shape[1] != self.dim:
            raise ValueError(f"vector dim {vectors.shape[1]} != store dim {self.dim}")
        if len(chunks) != vectors.shape[0]:
            raise ValueError("chunks/vectors length mismatch")
        if self.matrix is None:
            self.matrix = vectors.astype(np.float32)
        else:
            self.matrix = np.vstack([self.matrix, vectors.astype(np.float32)])
        self.chunks.extend(chunks)

    def search(
        self,
        query_vec: np.ndarray,
        top_k: int = 5,
        language: Optional[str] = None,
    ) -> List[RetrievalHit]:
        if self.matrix is None or len(self.chunks) == 0:
            return []
        scores = self.matrix @ query_vec.reshape(-1)
        order = np.argsort(-scores)
        hits: List[RetrievalHit] = []
        for rank, idx in enumerate(order):
            chunk = self.chunks[int(idx)]
            if language and chunk.language and chunk.language != language:
                continue
            hits.append(RetrievalHit(chunk=chunk, score=float(scores[idx]), rank=len(hits)))
            if len(hits) >= top_k:
                break
        return hits

    # ----- persistence -------------------------------------------------------

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as f:
            pickle.dump(
                {"dim": self.dim, "chunks": self.chunks, "matrix": self.matrix}, f
            )

    @classmethod
    def load(cls, path: str | Path) -> "InMemoryVectorStore":
        with Path(path).open("rb") as f:
            payload = pickle.load(f)
        store = cls(dim=payload["dim"])
        store.chunks = payload["chunks"]
        store.matrix = payload["matrix"]
        return store
