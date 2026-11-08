"""Staged draft generation that holds only one model in memory at a time.

The straightforward :class:`~src.eval.benchmark.Backend` loads the embedder,
the LLM and the NLI checker together (about 8 GB in fp32 for the neural
backend).  On a memory-constrained laptop that thrashes.  This module runs
the same computation in stages, each with a single model loaded, and keeps
every intermediate result in ``<out_dir>/stages_seed<k>/`` so that an
interrupted run resumes where it stopped:

1. **translate**  (LLM)       non-English queries -> English (E6 baseline)
2. **retrieve**   (embedder)  hits for original and translated queries
3. **generate**   (LLM)       drafts + verbalized confidence
4. **ground**     (NLI)       grounding support per checker

The output is identical in format to :meth:`Backend.draft_one`, written to
the same :class:`DraftStore` files, so everything downstream is unchanged.
"""

from __future__ import annotations

import gc
import json
import logging
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence

from src.build import build_index
from src.eval.benchmark import BackendSpec, DraftStore, _grounding_to_json, _hits_to_json
from src.generation.generator import GroundedGenerator
from src.eval.metrics import self_consistency_score
from src.generation.llm import (
    CONFIDENCE_SYSTEM,
    TRANSLATE_SYSTEM,
    build_confidence_prompt,
    get_llm,
    parse_verbalized_confidence,
)
from src.generation.prompts import format_evidence
from src.ingestion.chunker import Chunk
from src.ingestion.loaders import load_directory
from src.retrieval.hybrid import HybridRetriever
from src.retrieval.vector_store import RetrievalHit
from src.trust.grounding import LexicalOverlapChecker, NLIGroundingChecker

log = logging.getLogger(__name__)


class _Stage:
    """A JSON dict on disk, saved after every ``save_every`` updates."""

    def __init__(self, path: Path, save_every: int = 5) -> None:
        self.path = path
        self.data: Dict[str, dict] = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
        self._dirty = 0
        self.save_every = save_every

    def put(self, key: str, value) -> None:
        self.data[key] = value
        self._dirty += 1
        if self._dirty >= self.save_every:
            self.flush()

    def flush(self) -> None:
        self.path.write_text(json.dumps(self.data, ensure_ascii=False), encoding="utf-8")
        self._dirty = 0


def _hits_from_json(hits_json: List[dict], chunk_lookup: Dict[str, Chunk]) -> List[RetrievalHit]:
    out = []
    for h in hits_json:
        c = chunk_lookup[h["chunk_id"]]
        out.append(RetrievalHit(chunk=c, score=h["score"], rank=h["rank"],
                                dense_score=h.get("dense_score", 0.0), sparse_score=h.get("sparse_score", 0.0)))
    return out


def _release() -> None:
    """Callers must drop their own references first (``x = None``); this just
    forces the collector so large CPU tensors are returned to the OS."""
    gc.collect()


def _progress(stage: str, i: int, n: int, t0: float) -> None:
    if i == n or i % 10 == 0:
        rate = (time.time() - t0) / max(i, 1)
        log.info("  %s %d/%d (%.1fs each, ~%.0f min left)", stage, i, n, rate, rate * (n - i) / 60)


def run_staged(
    spec: BackendSpec,
    source_dirs: Sequence[Path],
    seed: int,
    queries: Sequence[dict],
    store: DraftStore,
    tstore: Optional[DraftStore],
    trans_queries: Sequence[dict],
    out_dir: Path,
) -> dict:
    """Fill ``store`` (and ``tstore`` when ``trans_queries`` is non-empty)."""
    sdir = out_dir / f"stages_seed{seed}"
    sdir.mkdir(parents=True, exist_ok=True)
    todo = [q for q in queries if q["id"] not in store]
    todo_t = [q for q in trans_queries if tstore is not None and q["id"] not in tstore]
    info: dict = {}
    if not todo and not todo_t:
        return info

    # ---- 1. translate (LLM) -------------------------------------------- #
    translations = _Stage(sdir / "translations.json")
    need_t = [q for q in todo_t if q["id"] not in translations.data]
    if need_t:
        log.info("stage translate: %d queries", len(need_t))
        llm = get_llm(**{**spec.llm, "seed": seed})
        t0 = time.time()
        for i, q in enumerate(need_t, 1):
            txt = llm.generate(TRANSLATE_SYSTEM, q["query"], max_new_tokens=64).text.strip()
            translations.put(q["id"], txt)
            _progress("translate", i, len(need_t), t0)
        translations.flush()
        llm = None
        _release()

    # ---- 2. retrieve (embedder) ---------------------------------------- #
    retrieval = _Stage(sdir / "retrieval.json")
    retrieval_t = _Stage(sdir / "retrieval_translated.json")
    docs = [d for dd in source_dirs for d in load_directory(dd) if "status: superseded" not in d.text[:200]]
    need_r = [q for q in todo if q["id"] not in retrieval.data]
    need_rt = [q for q in todo_t if q["id"] not in retrieval_t.data]
    embedder, vstore, bm25 = build_index(docs, spec.embedder, spec.chunker)
    info.update({"n_docs": len(docs), "n_chunks": len(vstore.chunks), "embedder": spec.embedder,
                 "checkers": list(spec.checkers), "chunker": spec.chunker, "retrieval": spec.retrieval})
    chunk_lookup = {c.chunk_id: c for c in vstore.chunks}
    if need_r or need_rt:
        log.info("stage retrieve: %d + %d queries", len(need_r), len(need_rt))
        retriever = HybridRetriever(embedder=embedder, vector_store=vstore, bm25=bm25,
                                    alpha=spec.retrieval.get("alpha", 0.5))
        k = spec.retrieval.get("top_k", 4)
        for q in need_r:
            retrieval.put(q["id"], _hits_to_json(retriever.retrieve(q["query"], top_k=k)))
        for q in need_rt:
            rq = translations.data.get(q["id"]) or q["query"]
            retrieval_t.put(q["id"], _hits_to_json(retriever.retrieve(rq, top_k=k)))
        retrieval.flush()
        retrieval_t.flush()
        retriever = None
    embedder = None
    bm25 = None
    vstore.matrix = None
    _release()

    # ---- 3. generate (LLM) --------------------------------------------- #
    generation = _Stage(sdir / "generation.json")
    generation_t = _Stage(sdir / "generation_translated.json")
    need_g = [(q, retrieval.data[q["id"]], generation) for q in todo if q["id"] not in generation.data]
    need_g += [(q, retrieval_t.data[q["id"]], generation_t) for q in todo_t if q["id"] not in generation_t.data]
    if need_g:
        log.info("stage generate: %d drafts", len(need_g))
        llm = get_llm(**{**spec.llm, "seed": seed})
        info["llm"] = llm.name
        generator = GroundedGenerator(llm=llm, max_new_tokens=spec.max_new_tokens)
        t0 = time.time()
        for i, (q, hits_json, stage) in enumerate(need_g, 1):
            hits = _hits_from_json(hits_json, chunk_lookup)
            answer = generator.generate(q["query"], hits)
            rec = {"draft_text": answer.text, "model": answer.raw_generation.model,
                   "model_abstained": bool(answer.is_abstention), "verbalized_confidence": None}
            if spec.self_consistency and hasattr(llm, "temperature") and hasattr(llm, "seed"):
                # extra drafts at a higher temperature; agreement with the main draft is the
                # gold-independent score used by the E7 self-consistency gate
                base_t, base_s = llm.temperature, llm.seed
                samples = []
                try:
                    llm.temperature = spec.sc_temperature
                    for j in range(spec.self_consistency):
                        llm.seed = 1000 * (seed + 1) + j + 1
                        samples.append(generator.generate(q["query"], hits).text)
                finally:
                    llm.temperature, llm.seed = base_t, base_s
                rec["sc_samples"] = samples
                rec["self_consistency"] = self_consistency_score(answer.text, samples)
            if spec.verbalized_confidence:
                if hits and not answer.is_abstention:
                    prompt = build_confidence_prompt(format_evidence(hits), q["query"], answer.text)
                    txt = llm.generate(CONFIDENCE_SYSTEM, prompt, max_new_tokens=8).text
                    rec["verbalized_confidence"] = parse_verbalized_confidence(txt)
                    rec["verbalized_raw"] = txt
                else:
                    rec["verbalized_confidence"] = 0.0
            stage.put(q["id"], rec)
            _progress("generate", i, len(need_g), t0)
        generation.flush()
        generation_t.flush()
        generator = None
        llm = None
        _release()
    else:
        info["llm"] = next(iter(generation.data.values()))["model"] if generation.data else spec.llm.get("name")

    # ---- 4. ground (checkers) ------------------------------------------ #
    grounding = _Stage(sdir / "grounding.json")
    grounding_t = _Stage(sdir / "grounding_translated.json")
    need_n = [(q, retrieval.data[q["id"]], generation.data[q["id"]], grounding) for q in todo if q["id"] not in grounding.data]
    need_n += [(q, retrieval_t.data[q["id"]], generation_t.data[q["id"]], grounding_t) for q in todo_t if q["id"] not in grounding_t.data]
    if need_n:
        log.info("stage ground: %d drafts", len(need_n))
        checkers = {}
        for c in spec.checkers:
            checkers[c] = LexicalOverlapChecker(support_threshold=0.2) if c == "lexical" else NLIGroundingChecker(support_threshold=0.5)
        t0 = time.time()
        zero = {"overall_support": 0.0, "sentence_supports": [], "citation_coverage": 0.0, "n_unsupported": 0}
        for i, (q, hits_json, gen, stage) in enumerate(need_n, 1):
            hits = _hits_from_json(hits_json, chunk_lookup)
            g = {}
            for name, chk in checkers.items():
                g[name] = zero if gen["model_abstained"] else _grounding_to_json(chk.check(gen["draft_text"], hits))
            stage.put(q["id"], g)
            _progress("ground", i, len(need_n), t0)
        grounding.flush()
        grounding_t.flush()
        checkers = None
        _release()

    # ---- 5. assemble ---------------------------------------------------- #
    def assemble(q: dict, hits_json, gen, g, translated):
        return {
            "id": q["id"], "group": q.get("group"), "domain": q.get("domain"), "language": q["language"],
            "query": q["query"], "translated_query": translated,
            "gold_answers": q.get("gold_answers", [q.get("gold_answer", "")]),
            "gold_answerable": int(q.get("gold_answerable", 1)), "tags": q.get("tags", []),
            "hits": [dict(h, text=chunk_lookup[h["chunk_id"]].text) for h in hits_json],
            "draft_text": gen["draft_text"], "model": gen["model"], "model_abstained": gen["model_abstained"],
            "grounding": g, "verbalized_confidence": gen.get("verbalized_confidence"),
            "verbalized_raw": gen.get("verbalized_raw"), "seed": seed, "backend": spec.name,
            "self_consistency": gen.get("self_consistency"), "sc_samples": gen.get("sc_samples"),
        }

    for q in todo:
        store.put(assemble(q, retrieval.data[q["id"]], generation.data[q["id"]], grounding.data[q["id"]], None))
    for q in todo_t:
        tstore.put(assemble(q, retrieval_t.data[q["id"]], generation_t.data[q["id"]], grounding_t.data[q["id"]],
                            translations.data.get(q["id"])))
    return info
