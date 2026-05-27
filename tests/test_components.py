"""Unit tests for the core building blocks."""

import numpy as np

from src.eval.metrics import abstention_precision_recall, mean_reciprocal_rank, precision_at_k
from src.ingestion.chunker import chunk_text
from src.ingestion.normalize import detect_language, normalize_text
from src.retrieval.bm25 import BM25Index
from src.retrieval.embeddings import HashEmbedding
from src.retrieval.vector_store import InMemoryVectorStore
from src.trust.abstention import AbstentionPolicy
from src.trust.calibration import (
    ConfidenceEstimator,
    ConfidenceSignals,
    expected_calibration_error,
)
from src.trust.fairness import FairnessAuditor
from src.trust.grounding import LexicalOverlapChecker


def test_normalize_strips_bidi_and_whitespace():
    assert normalize_text("  hello​world  ") == "helloworld"


def test_detect_language_english():
    assert detect_language("This is a normal English sentence.") == "en"


def test_chunker_emits_chunks_with_overlap():
    chunks = chunk_text("one two three four five. six seven eight nine ten.",
                        doc_id="t", chunk_size=4, overlap=1)
    assert len(chunks) >= 2
    assert all(c.chunk_id.startswith("t::chunk_") for c in chunks)


def test_hash_embedder_returns_unit_vectors():
    emb = HashEmbedding(dim=32)
    v = emb.encode(["hello world", "foo bar"])
    assert v.shape == (2, 32)
    norms = np.linalg.norm(v, axis=1)
    assert np.allclose(norms, 1.0, atol=1e-4)


def test_vector_store_topk():
    emb = HashEmbedding(dim=32)
    chunks = list(chunk_text("apple banana cherry. dog elephant fox.", "x", chunk_size=3, overlap=0))
    store = InMemoryVectorStore(dim=32)
    store.add(chunks, emb.encode([c.text for c in chunks]))
    hits = store.search(emb.encode(["banana cherry"])[0], top_k=2)
    assert hits and hits[0].chunk.text


def test_bm25_returns_a_hit_when_term_present():
    chunks = chunk_text("renewal of passport requires form K-35.", "p", chunk_size=20, overlap=0)
    idx = BM25Index()
    idx.fit(chunks)
    hits = idx.search("passport renewal")
    assert hits


def test_grounding_lexical_overlap():
    from src.ingestion.chunker import Chunk
    from src.retrieval.vector_store import RetrievalHit
    chunk = Chunk("c0", "d0", "The passport renewal fee is LKR 3500 for one-day service.")
    hit = RetrievalHit(chunk=chunk, score=0.9, rank=0)
    g = LexicalOverlapChecker(support_threshold=0.1).check(
        "The passport renewal fee is LKR 3500. [1]", [hit]
    )
    assert g.overall_support > 0.25
    assert g.citation_coverage > 0.0


def test_confidence_estimator_bounds():
    est = ConfidenceEstimator()
    s = ConfidenceSignals(0.8, 0.2, 0.7, 0.5, 1.0)
    assert 0.0 <= est.estimate(s) <= 1.0


def test_abstention_refuses_without_evidence():
    p = AbstentionPolicy()
    d = p.decide(confidence=0.9, grounding_support=0.9, citation_coverage=1.0, has_evidence=False)
    assert d.decision == "refuse"


def test_fairness_audit_reports_gaps():
    records = [
        {"language": "en", "correct": 1, "abstained": 0, "grounding_support": 0.8, "gold_answerable": 1},
        {"language": "en", "correct": 1, "abstained": 0, "grounding_support": 0.7, "gold_answerable": 1},
        {"language": "si", "correct": 0, "abstained": 1, "grounding_support": 0.3, "gold_answerable": 1},
        {"language": "ta", "correct": 1, "abstained": 0, "grounding_support": 0.6, "gold_answerable": 1},
    ]
    rep = FairnessAuditor().audit(records)
    assert rep.max_gap_accuracy > 0.0
    assert "en" in rep.per_group_accuracy


def test_metrics_basic_cases():
    assert precision_at_k(["a", "b", "c"], ["b", "z"], k=3) == 1 / 3
    assert mean_reciprocal_rank(["a", "b", "c"], ["b"]) == 0.5
    p, r, f1 = abstention_precision_recall([1, 0, 1, 0], [1, 0, 0, 1])
    assert 0.0 <= p <= 1.0 and 0.0 <= r <= 1.0 and 0.0 <= f1 <= 1.0


def test_ece_zero_when_perfectly_calibrated():
    ece = expected_calibration_error([1.0, 1.0, 1.0], [1, 1, 1], n_bins=5)
    assert ece == 0.0
