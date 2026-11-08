"""Hybrid (dense + sparse) retrieval with Reciprocal Rank Fusion."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import numpy as np

from src.retrieval.bm25 import BM25Index
from src.retrieval.embeddings import EmbeddingModel
from src.retrieval.vector_store import InMemoryVectorStore, RetrievalHit


@dataclass
class HybridRetriever:
    """Combines a dense vector store and a BM25 index via RRF fusion.

    ``alpha`` weights dense vs sparse contributions (1.0 = dense only,
    0.0 = sparse only). The default 0.5 balances the two — empirically a
    good starting point for multilingual corpora.
    """

    embedder: EmbeddingModel
    vector_store: InMemoryVectorStore
    bm25: BM25Index
    alpha: float = 0.5
    rrf_k: int = 60

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        candidate_k: int = 25,
        language: Optional[str] = None,
    ) -> List[RetrievalHit]:
        # E5-style models expect a "query: " prefix; harmless for hash embedder.
        q_vec = self.embedder.encode([f"query: {query}"])[0]
        dense_hits = self.vector_store.search(q_vec, top_k=candidate_k, language=language)
        sparse_hits = self.bm25.search(query, top_k=candidate_k)

        # Reciprocal rank fusion across the two ranked lists.
        fused: dict[str, float] = {}
        sources: dict[str, RetrievalHit] = {}
        dense_by_id: dict[str, float] = {}
        sparse_by_id: dict[str, float] = {}
        for h in dense_hits:
            fused[h.chunk.chunk_id] = fused.get(h.chunk.chunk_id, 0.0) + self.alpha / (
                self.rrf_k + h.rank + 1
            )
            sources[h.chunk.chunk_id] = h
            dense_by_id[h.chunk.chunk_id] = float(h.score)
        for h in sparse_hits:
            fused[h.chunk.chunk_id] = fused.get(h.chunk.chunk_id, 0.0) + (1 - self.alpha) / (
                self.rrf_k + h.rank + 1
            )
            sources.setdefault(h.chunk.chunk_id, h)
            sparse_by_id[h.chunk.chunk_id] = float(h.score)

        ordered = sorted(fused.items(), key=lambda kv: -kv[1])[:top_k]
        out: List[RetrievalHit] = []
        for rank, (cid, score) in enumerate(ordered):
            base = sources[cid]
            dense = dense_by_id.get(cid)
            if dense is None:
                # Chunk surfaced only through BM25: compute its cosine directly so
                # the dense signal is defined for every returned hit.
                idx = self._chunk_index(cid)
                dense = (
                    float(self.vector_store.matrix[idx] @ q_vec) if idx is not None else 0.0
                )
            out.append(
                RetrievalHit(
                    chunk=base.chunk,
                    score=float(score),
                    rank=rank,
                    dense_score=dense,
                    sparse_score=sparse_by_id.get(cid, 0.0),
                )
            )
        return out

    def _chunk_index(self, chunk_id: str) -> Optional[int]:
        if not hasattr(self, "_id_to_idx") or len(self._id_to_idx) != len(self.vector_store.chunks):
            self._id_to_idx = {c.chunk_id: i for i, c in enumerate(self.vector_store.chunks)}
        return self._id_to_idx.get(chunk_id)
