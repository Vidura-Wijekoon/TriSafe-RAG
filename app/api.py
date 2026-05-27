"""FastAPI service exposing the TriSafe-RAG pipeline."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from src.build import build_pipeline, load_config

CONFIG_PATH = os.getenv("TRISAFE_CONFIG", "configs/experiment_v1.yaml")
DATA_DIR = os.getenv("TRISAFE_DATA", "data/raw")

app = FastAPI(title="TriSafe-RAG", version="0.2.0")

_pipeline = None


def _get_pipeline():
    global _pipeline
    if _pipeline is None:
        cfg = load_config(CONFIG_PATH)
        src = DATA_DIR if Path(DATA_DIR).exists() else None
        _pipeline = build_pipeline(cfg, source_dir=src)
    return _pipeline


class QueryIn(BaseModel):
    query: str
    language: Optional[str] = None


@app.get("/health")
def health():
    return {"status": "ok", "version": "0.2.0"}


@app.post("/query")
def query(payload: QueryIn):
    pipe = _get_pipeline()
    if pipe is None:
        raise HTTPException(503, "Pipeline not ready")
    resp = pipe.query(payload.query, language=payload.language)
    return {
        "query": resp.query,
        "language": resp.language,
        "answer": resp.answer.text,
        "decision": resp.decision.decision,
        "reason": resp.decision.reason,
        "confidence": resp.confidence,
        "grounding_support": resp.grounding.overall_support,
        "citation_coverage": resp.grounding.citation_coverage,
        "citations": [
            {
                "rank": h.rank + 1,
                "chunk_id": h.chunk.chunk_id,
                "score": h.score,
                "text": h.chunk.text[:300],
            }
            for h in resp.hits
        ],
    }
