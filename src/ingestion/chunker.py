"""Sliding-window chunker that respects sentence boundaries when possible."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import List, Optional

# Sentence boundary regex covering Latin punctuation plus Sinhala/Tamil full stops.
_SENT_SPLIT = re.compile(r"(?<=[\.\!\?।॥௺])\s+")


@dataclass
class Chunk:
    """A retrieval unit produced from a source Document."""

    chunk_id: str
    doc_id: str
    text: str
    language: Optional[str] = None
    start_char: int = 0
    end_char: int = 0
    metadata: dict = field(default_factory=dict)


def _split_sentences(text: str) -> List[str]:
    parts = _SENT_SPLIT.split(text)
    return [p.strip() for p in parts if p.strip()]


def chunk_text(
    text: str,
    doc_id: str,
    chunk_size: int = 320,
    overlap: int = 64,
    language: Optional[str] = None,
) -> List[Chunk]:
    """Split ``text`` into overlapping chunks of ~``chunk_size`` words.

    The chunker greedily packs sentences into a chunk until it exceeds the
    budget, then emits a chunk and rolls forward by ``chunk_size - overlap``
    words. Sentence boundaries are preserved when present so that retrieved
    passages remain readable for citation rendering.
    """
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if overlap < 0 or overlap >= chunk_size:
        raise ValueError("overlap must satisfy 0 <= overlap < chunk_size")

    sentences = _split_sentences(text) or [text]
    chunks: List[Chunk] = []
    buffer: List[str] = []
    buffer_words = 0
    cursor = 0
    chunk_idx = 0

    def _emit():
        nonlocal buffer, buffer_words, chunk_idx, cursor
        if not buffer:
            return
        chunk_text_ = " ".join(buffer)
        start = cursor
        end = start + len(chunk_text_)
        chunks.append(
            Chunk(
                chunk_id=f"{doc_id}::chunk_{chunk_idx:04d}",
                doc_id=doc_id,
                text=chunk_text_,
                language=language,
                start_char=start,
                end_char=end,
            )
        )
        chunk_idx += 1
        # advance cursor and slide window by (chunk_size - overlap) words
        if overlap > 0:
            tail = " ".join(chunk_text_.split()[-overlap:])
            buffer = [tail]
            buffer_words = len(tail.split())
            cursor = end - len(tail)
        else:
            buffer = []
            buffer_words = 0
            cursor = end

    for sent in sentences:
        sent_words = len(sent.split())
        if buffer_words + sent_words > chunk_size and buffer:
            _emit()
        buffer.append(sent)
        buffer_words += sent_words

    _emit()
    return chunks
