"""Build a TriRAGPipeline from a config dict and an iterable of Documents."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Optional

import yaml

from src.audit.logger import AuditLogger
from src.generation.generator import GroundedGenerator
from src.generation.llm import get_llm
from src.ingestion.chunker import chunk_text
from src.ingestion.loaders import Document, load_directory
from src.ingestion.normalize import detect_language, normalize_text
from src.pipeline import TriRAGPipeline
from src.retrieval.bm25 import BM25Index
from src.retrieval.embeddings import get_embedder
from src.retrieval.hybrid import HybridRetriever
from src.retrieval.vector_store import InMemoryVectorStore
from src.trust.abstention import AbstentionPolicy
from src.trust.calibration import ConfidenceEstimator
from src.trust.grounding import LexicalOverlapChecker, NLIGroundingChecker


def load_config(path: str | Path) -> dict:
    with Path(path).open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def build_index(documents: Iterable[Document], embedder_cfg: dict, chunker_cfg: dict):
    embedder = get_embedder(**embedder_cfg)
    store = InMemoryVectorStore(dim=embedder.dim)
    bm25 = BM25Index()
    chunks = []
    for doc in documents:
        norm = normalize_text(doc.text)
        if not norm:
            continue
        lang = doc.language_hint or detect_language(norm)
        chunks.extend(
            chunk_text(norm, doc_id=doc.doc_id, language=lang, **chunker_cfg)
        )
    if not chunks:
        return embedder, store, bm25
    vectors = embedder.encode([f"passage: {c.text}" for c in chunks])
    store.add(chunks, vectors)
    bm25.fit(chunks)
    return embedder, store, bm25


def build_pipeline(config: dict, source_dir: Optional[str] = None) -> TriRAGPipeline:
    documents = list(load_directory(source_dir)) if source_dir else []
    embedder, store, bm25 = build_index(
        documents,
        embedder_cfg=config.get("embedder", {"name": "hash"}),
        chunker_cfg=config.get("chunker", {"chunk_size": 320, "overlap": 64}),
    )
    retriever = HybridRetriever(
        embedder=embedder, vector_store=store, bm25=bm25,
        alpha=config.get("retrieval", {}).get("alpha", 0.5),
    )
    llm = get_llm(**config.get("llm", {"name": "echo"}))
    generator = GroundedGenerator(llm=llm,
                                  max_new_tokens=config.get("generation", {}).get("max_new_tokens", 512))

    grounding_cfg = config.get("grounding", {"backend": "lexical"})
    if grounding_cfg.get("backend") == "nli":
        checker = NLIGroundingChecker(
            support_threshold=grounding_cfg.get("support_threshold", 0.5)
        )
    else:
        checker = LexicalOverlapChecker(
            support_threshold=grounding_cfg.get("support_threshold", 0.25)
        )

    pipeline = TriRAGPipeline(
        retriever=retriever,
        generator=generator,
        grounding_checker=checker,
        confidence=ConfidenceEstimator(**config.get("confidence", {})),
        abstention=AbstentionPolicy(**config.get("abstention", {})),
        audit_logger=AuditLogger(config.get("audit", {}).get("path", "logs/audit.jsonl")),
        top_k=config.get("retrieval", {}).get("top_k", 5),
    )
    return pipeline
