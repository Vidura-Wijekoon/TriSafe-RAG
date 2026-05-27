"""End-to-end smoke test using the echo LLM + hash embedder."""

from src.build import build_index
from src.generation.generator import GroundedGenerator
from src.generation.llm import EchoLLM
from src.ingestion.loaders import Document
from src.pipeline import TriRAGPipeline
from src.retrieval.hybrid import HybridRetriever


def _corpus():
    return [
        Document(
            doc_id="passport_renewal",
            text=(
                "To renew a Sri Lankan passport, applicants must submit form K-35 "
                "to the Department of Immigration. The renewal fee is LKR 3500 for "
                "a one-day service and LKR 1000 for a normal service."
            ),
            source_path="seed/passport.md",
            language_hint="en",
        ),
        Document(
            doc_id="nic_renewal",
            text=(
                "National Identity Card renewal requires a birth certificate and "
                "a recent photograph. Applications are processed at the divisional "
                "secretariat office."
            ),
            source_path="seed/nic.md",
            language_hint="en",
        ),
    ]


def _pipeline():
    embedder, store, bm25 = build_index(
        _corpus(),
        embedder_cfg={"name": "hash"},
        chunker_cfg={"chunk_size": 64, "overlap": 8},
    )
    return TriRAGPipeline(
        retriever=HybridRetriever(embedder=embedder, vector_store=store, bm25=bm25, alpha=0.5),
        generator=GroundedGenerator(llm=EchoLLM()),
        top_k=3,
    )


def test_answer_question():
    pipe = _pipeline()
    resp = pipe.query("How much is the passport renewal fee?")
    assert resp.hits, "expected at least one retrieval hit"
    assert resp.decision.decision in {"answer", "defer"}


def test_abstains_when_no_relevant_evidence():
    pipe = _pipeline()
    resp = pipe.query("What is the capital of Mongolia?")
    # No relevant evidence -> grounding low -> refuse or defer.
    assert resp.decision.decision in {"refuse", "defer"}


def test_audit_record_populated():
    pipe = _pipeline()
    resp = pipe.query("Where do I apply for an NIC?")
    rec = resp.audit
    assert rec.query and rec.retrieved_chunk_ids
    assert 0.0 <= rec.confidence <= 1.0
