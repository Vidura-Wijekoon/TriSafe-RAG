"""Append-only JSONL audit log for every TriRAG query.

Each record captures the query, language, retrieved passage ids, grounding
support, confidence signals, abstention decision, and a hash of the model
output — enough to reconstruct and replay any decision later.
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import List, Optional


@dataclass
class AuditRecord:
    timestamp: float
    query: str
    language: Optional[str]
    retrieved_chunk_ids: List[str]
    retrieval_scores: List[float]
    grounding_support: float
    citation_coverage: float
    confidence: float
    decision: str
    reason: str
    answer_hash: str
    model: str
    extra: dict = field(default_factory=dict)


class AuditLogger:
    """Append-only JSONL writer."""

    def __init__(self, path: str | Path = "logs/audit.jsonl") -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def log(self, record: AuditRecord) -> None:
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(asdict(record), ensure_ascii=False) + "\n")

    @staticmethod
    def hash_answer(text: str) -> str:
        return hashlib.sha256((text or "").encode("utf-8")).hexdigest()[:16]

    @staticmethod
    def now() -> float:
        return time.time()
