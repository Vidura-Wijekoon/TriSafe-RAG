"""Small-footprint BM25 implementation for hybrid retrieval.

Multilingual tokenization is intentionally simple (whitespace + lowercase)
to keep the dependency footprint zero. Replace with a language-aware tokenizer
when Sinhala / Tamil morphology becomes a bottleneck.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from typing import List, Sequence

from src.ingestion.chunker import Chunk
from src.retrieval.vector_store import RetrievalHit

_WORD_RE = re.compile(r"\w+", re.UNICODE)


def _tokenize(text: str) -> List[str]:
    return [t.lower() for t in _WORD_RE.findall(text or "")]


@dataclass
class BM25Index:
    k1: float = 1.5
    b: float = 0.75
    chunks: List[Chunk] = field(default_factory=list)
    _doc_freqs: List[dict] = field(default_factory=list)
    _doc_lens: List[int] = field(default_factory=list)
    _avgdl: float = 0.0
    _idf: dict = field(default_factory=dict)

    def fit(self, chunks: Sequence[Chunk]) -> None:
        self.chunks = list(chunks)
        self._doc_freqs = []
        self._doc_lens = []
        df: dict = {}
        for chunk in self.chunks:
            tokens = _tokenize(chunk.text)
            tf: dict = {}
            for tok in tokens:
                tf[tok] = tf.get(tok, 0) + 1
            self._doc_freqs.append(tf)
            self._doc_lens.append(len(tokens))
            for tok in tf:
                df[tok] = df.get(tok, 0) + 1

        n = max(len(self.chunks), 1)
        self._avgdl = sum(self._doc_lens) / n if self._doc_lens else 0.0
        # BM25+ style idf to avoid negatives
        self._idf = {
            tok: math.log(1 + (n - dfi + 0.5) / (dfi + 0.5)) for tok, dfi in df.items()
        }

    def search(self, query: str, top_k: int = 5) -> List[RetrievalHit]:
        if not self.chunks:
            return []
        q_tokens = _tokenize(query)
        scores: List[float] = []
        for i, tf in enumerate(self._doc_freqs):
            dl = self._doc_lens[i] or 1
            score = 0.0
            for tok in q_tokens:
                if tok not in tf:
                    continue
                idf = self._idf.get(tok, 0.0)
                f = tf[tok]
                denom = f + self.k1 * (1 - self.b + self.b * dl / (self._avgdl or 1))
                score += idf * (f * (self.k1 + 1)) / (denom or 1.0)
            scores.append(score)

        order = sorted(range(len(scores)), key=lambda i: -scores[i])[:top_k]
        return [
            RetrievalHit(chunk=self.chunks[i], score=scores[i], rank=r)
            for r, i in enumerate(order)
            if scores[i] > 0
        ]
