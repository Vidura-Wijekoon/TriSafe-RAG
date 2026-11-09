"""Benchmark runner: cached drafts + replayable trust-layer configurations.

The expensive half of the pipeline (retrieval + LLM generation, plus the
grounding checkers and the auxiliary LLM calls used by external baselines) is
run **once per seed** and cached as JSONL *drafts*.  Every system compared in
the paper (ablations, dev-tuned variants, external baselines) is then a cheap,
deterministic *replay* of the trust layer over the same drafts.  This makes
the comparison exactly paired and makes every number in the paper
reproducible from the cached drafts alone.

Terminology
-----------
backend   -- how drafts are produced (embedder, LLM, grounding checkers).
draft     -- per-query cache record (hits, draft text, grounding, aux scores).
system    -- a trust-layer configuration replayed over drafts.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Sequence

import numpy as np

from src.build import build_index
from src.eval.metrics import alias_match, gaps, per_language, summarise
from src.generation.generator import GroundedGenerator
from src.generation.llm import (
    CONFIDENCE_SYSTEM,
    TRANSLATE_SYSTEM,
    build_confidence_prompt,
    get_llm,
    parse_verbalized_confidence,
)
from src.generation.prompts import format_evidence
from src.ingestion.loaders import load_directory
from src.retrieval.hybrid import HybridRetriever
from src.retrieval.vector_store import RetrievalHit
from src.trust.abstention import AbstentionPolicy
from src.trust.calibration import ConfidenceEstimator, ConfidenceSignals, PostHocCalibrator
from src.trust.grounding import GroundingResult, LexicalOverlapChecker, NLIGroundingChecker

logger = logging.getLogger(__name__)

SIGNAL_NAMES = ["retrieval", "margin", "grounding", "citation", "language"]


# --------------------------------------------------------------------------- #
# Backend: produce drafts
# --------------------------------------------------------------------------- #

@dataclass
class BackendSpec:
    name: str
    embedder: dict
    llm: dict
    chunker: dict = field(default_factory=lambda: {"chunk_size": 80, "overlap": 0})
    retrieval: dict = field(default_factory=lambda: {"alpha": 0.5, "top_k": 4})
    checkers: Sequence[str] = ("lexical",)
    verbalized_confidence: bool = False
    translate_queries: bool = False
    max_new_tokens: int = 128
    # extra drafts sampled at ``sc_temperature`` for the self-consistency gate (E7); 0 disables
    self_consistency: int = 0
    sc_temperature: float = 0.7


OFFLINE_BACKEND = BackendSpec(
    name="offline",
    embedder={"name": "hash", "dim": 256},
    llm={"name": "echo"},
    checkers=("lexical",),
)

# Open-weight generators served locally by Ollama (llama.cpp).  The 1.5B model is
# run at fp16, i.e. without weight quantisation; the 7B model at Q4_K_M so that
# it fits an 8 GB GPU.  Retrieval (multilingual-e5-base) and the NLI grounding
# checker run on the CPU through sentence-transformers / transformers.
NEURAL_BACKEND = BackendSpec(
    name="neural",
    embedder={"name": "st", "model_name": "intfloat/multilingual-e5-base"},
    llm={"name": "ollama", "model": "qwen2.5:1.5b-instruct-fp16", "temperature": 0.2},
    checkers=("lexical", "nli"),
    verbalized_confidence=True,
    translate_queries=True,
    self_consistency=3,
)

NEURAL7B_BACKEND = BackendSpec(
    name="neural7b",
    embedder={"name": "st", "model_name": "intfloat/multilingual-e5-base"},
    llm={"name": "ollama", "model": "qwen2.5:7b-instruct", "temperature": 0.2},
    checkers=("lexical", "nli"),
    verbalized_confidence=True,
    translate_queries=True,
    self_consistency=3,
)

# The seed-0 CPU run reported in the first draft of the paper: the same 1.5B
# model through HuggingFace transformers at fp32.  Kept so that results/neural_hf_fp32
# can be regenerated; not used for the paper's tables.
NEURAL_HF_FP32_BACKEND = BackendSpec(
    name="neural_hf_fp32",
    embedder={"name": "st", "model_name": "intfloat/multilingual-e5-base"},
    llm={"name": "hf", "model": "Qwen/Qwen2.5-1.5B-Instruct", "temperature": 0.2, "dtype": "float32"},
    checkers=("lexical", "nli"),
    verbalized_confidence=True,
    translate_queries=True,
)

BACKENDS = {b.name: b for b in (OFFLINE_BACKEND, NEURAL_BACKEND, NEURAL7B_BACKEND, NEURAL_HF_FP32_BACKEND)}


class DraftStore:
    """Append-only JSONL cache keyed by query id."""

    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.records: Dict[str, dict] = {}
        if self.path.exists():
            for line in self.path.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    r = json.loads(line)
                    self.records[r["id"]] = r

    def __contains__(self, qid: str) -> bool:
        return qid in self.records

    def put(self, rec: dict) -> None:
        self.records[rec["id"]] = rec
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    def get(self, qid: str) -> dict:
        return self.records[qid]


def _hits_to_json(hits: List[RetrievalHit]) -> List[dict]:
    return [
        {
            "chunk_id": h.chunk.chunk_id,
            "doc_id": h.chunk.doc_id,
            "language": h.chunk.language,
            "rank": h.rank,
            "score": h.score,
            "dense_score": h.dense_score,
            "sparse_score": h.sparse_score,
        }
        for h in hits
    ]


def _grounding_to_json(g: GroundingResult) -> dict:
    return {
        "overall_support": g.overall_support,
        "sentence_supports": g.sentence_supports,
        "citation_coverage": g.citation_coverage,
        "n_unsupported": g.n_unsupported,
    }


def chunk_text_lookup(source_dirs: Sequence[Path], chunker_cfg: dict) -> Dict[str, str]:
    """chunk_id -> text for the corpus, without loading any embedder (used to
    back-fill ``text`` on cached hits)."""
    from src.ingestion.chunker import chunk_text
    from src.ingestion.normalize import detect_language, normalize_text
    out: Dict[str, str] = {}
    for d in source_dirs:
        for doc in load_directory(d):
            if "status: superseded" in doc.text[:200]:
                continue
            norm = normalize_text(doc.text)
            if not norm:
                continue
            lang = doc.language_hint or detect_language(norm)
            for c in chunk_text(norm, doc_id=doc.doc_id, language=lang, **chunker_cfg):
                out[c.chunk_id] = c.text
    return out


class Backend:
    """Builds the index once and produces drafts for a list of queries."""

    def __init__(self, spec: BackendSpec, source_dirs: Sequence[Path], seed: int = 0) -> None:
        self.spec = spec
        self.seed = seed
        docs = []
        for d in source_dirs:
            docs.extend(
                doc for doc in load_directory(d)
                if "status: superseded" not in doc.text[:200]
            )
        self.n_docs = len(docs)
        self.embedder, self.store, self.bm25 = build_index(docs, spec.embedder, spec.chunker)
        self.n_chunks = len(self.store.chunks)
        self.retriever = HybridRetriever(
            embedder=self.embedder, vector_store=self.store, bm25=self.bm25,
            alpha=spec.retrieval.get("alpha", 0.5),
        )
        llm_cfg = dict(spec.llm)
        if llm_cfg.get("name") in {"hf", "local", "transformers", "openai", "gpt"}:
            llm_cfg["seed"] = seed
        self.llm = get_llm(**llm_cfg)
        self.generator = GroundedGenerator(llm=self.llm, max_new_tokens=spec.max_new_tokens)
        self.checkers = {}
        for c in spec.checkers:
            if c == "lexical":
                self.checkers[c] = LexicalOverlapChecker(support_threshold=0.2)
            elif c == "nli":
                self.checkers[c] = NLIGroundingChecker(support_threshold=0.5)
            else:
                raise ValueError(f"unknown checker {c}")
        self.top_k = spec.retrieval.get("top_k", 4)

    # -------------------------------------------------------------------- #
    def draft_one(self, q: dict, translate: bool = False) -> dict:
        question = q["query"]
        retrieval_query = question
        translated = None
        if translate and q["language"] != "en":
            translated = self.llm.generate(TRANSLATE_SYSTEM, question, max_new_tokens=64).text.strip()
            retrieval_query = translated or question
        hits = self.retriever.retrieve(retrieval_query, top_k=self.top_k)
        answer = self.generator.generate(question, hits)
        rec = {
            "id": q["id"],
            "group": q.get("group"),
            "domain": q.get("domain"),
            "language": q["language"],
            "query": question,
            "translated_query": translated,
            "gold_answers": q.get("gold_answers", [q.get("gold_answer", "")]),
            "gold_answerable": int(q.get("gold_answerable", 1)),
            "tags": q.get("tags", []),
            "hits": _hits_to_json(hits),
            "draft_text": answer.text,
            "model": answer.raw_generation.model,
            "model_abstained": bool(answer.is_abstention),
            "grounding": {
                name: (_grounding_to_json(chk.check(answer.text, hits)) if not answer.is_abstention
                       else {"overall_support": 0.0, "sentence_supports": [], "citation_coverage": 0.0, "n_unsupported": 0})
                for name, chk in self.checkers.items()
            },
            "verbalized_confidence": None,
            "seed": self.seed,
            "backend": self.spec.name,
        }
        if self.spec.verbalized_confidence and hits and not answer.is_abstention:
            prompt = build_confidence_prompt(format_evidence(hits), question, answer.text)
            txt = self.llm.generate(CONFIDENCE_SYSTEM, prompt, max_new_tokens=8).text
            rec["verbalized_confidence"] = parse_verbalized_confidence(txt)
            rec["verbalized_raw"] = txt
        elif self.spec.verbalized_confidence:
            rec["verbalized_confidence"] = 0.0
        return rec

    def fill(self, queries: Sequence[dict], store: DraftStore, translate: bool = False,
             progress: Optional[Callable[[int, int, float], None]] = None) -> None:
        todo = [q for q in queries if q["id"] not in store]
        t0 = time.time()
        for i, q in enumerate(todo, start=1):
            store.put(self.draft_one(q, translate=translate))
            if progress:
                progress(i, len(todo), time.time() - t0)


# --------------------------------------------------------------------------- #
# Systems: replay the trust layer over drafts
# --------------------------------------------------------------------------- #

@dataclass
class SystemSpec:
    name: str
    label: str
    checker: str = "lexical"
    # how the confidence is produced
    mode: str = "weighted"  # weighted | score | calibrated
    score_field: str = ""   # for mode=score: "retrieval" | "verbalized"
    confidence: dict = field(default_factory=dict)
    abstention: dict = field(default_factory=dict)
    two_way: bool = False   # True -> answer/refuse only (external baselines)
    calibrator: Optional[PostHocCalibrator] = None
    calibrator_input: str = "signals"  # signals | sum
    tuned_on: Optional[str] = None
    notes: str = ""


def retrieval_scale_from_drafts(drafts: Iterable[dict]) -> dict:
    """Data-derived normalisation for the dense cosine signal: 5th/95th
    percentiles of top-1 cosine and 95th percentile of the top-1/top-2 margin
    over the given (development) drafts."""
    top1, margins = [], []
    for d in drafts:
        hs = d["hits"]
        if not hs:
            continue
        top1.append(hs[0]["dense_score"])
        if len(hs) > 1:
            margins.append(max(0.0, hs[0]["dense_score"] - hs[1]["dense_score"]))
    if not top1:
        return {"ret_lo": 0.0, "ret_hi": 1.0, "margin_scale": 1.0}
    lo, hi = float(np.percentile(top1, 5)), float(np.percentile(top1, 95))
    m95 = float(np.percentile(margins, 95)) if margins else 1.0
    return {"ret_lo": lo, "ret_hi": max(hi, lo + 1e-3), "margin_scale": 1.0 / max(m95, 1e-6)}


def signals_for(draft: dict, estimator: ConfidenceEstimator, checker: str) -> ConfidenceSignals:
    hs = draft["hits"]
    if estimator.retrieval_signal == "cosine":
        t1 = hs[0]["dense_score"] if hs else 0.0
        t2 = hs[1]["dense_score"] if len(hs) > 1 else 0.0
    else:
        t1 = hs[0]["score"] if hs else 0.0
        t2 = hs[1]["score"] if len(hs) > 1 else 0.0
    s_ret, s_mar = estimator.normalise_retrieval(t1, t2)
    g = draft["grounding"][checker]
    lang = draft["language"]
    lang_match = any((h.get("language") or "") in (lang, "mixed") for h in hs) if hs else False
    return ConfidenceSignals(s_ret, s_mar, g["overall_support"], g["citation_coverage"], 1.0 if lang_match else 0.0)


def normalise_draft(d: dict) -> bool:
    """Re-detect the generator's own abstention on a cached draft (see
    ``is_model_abstention``) and zero its grounding signals, as the generation stage
    does for a detected abstention.  Returns True when the draft was changed."""
    from src.generation.prompts import is_model_abstention
    if d.get("model_abstained") or not is_model_abstention(d.get("draft_text") or ""):
        return False
    d["model_abstained"] = True
    d["abstention_redetected"] = True
    for g in (d.get("grounding") or {}).values():
        g["overall_support"] = 0.0
        g["citation_coverage"] = 0.0
        g["sentence_supports"] = []
        g["n_unsupported"] = 0
    if d.get("verbalized_confidence") is not None:
        d["verbalized_confidence"] = 0.0
    return True


def draft_correct(draft: dict) -> int:
    if int(draft["gold_answerable"]) == 1:
        return int(alias_match(draft["draft_text"], draft["gold_answers"]))
    return int(bool(draft["model_abstained"]))


def replay(drafts: Sequence[dict], spec: SystemSpec) -> List[dict]:
    """Apply one trust-layer configuration to cached drafts -> eval records."""
    from src.generation.prompts import is_empty_reply
    estimator = ConfidenceEstimator(**spec.confidence)
    refuse_empty = bool(spec.abstention.get("refuse_empty", False))
    policy = AbstentionPolicy(**{k: v for k, v in spec.abstention.items() if k != "refuse_empty"})
    records = []
    for d in drafts:
        sig = signals_for(d, estimator, spec.checker)
        g = d["grounding"][spec.checker]
        if spec.mode == "weighted":
            conf = estimator.estimate(sig)
        elif spec.mode == "score":
            if spec.score_field == "retrieval":
                conf = sig.retrieval_score
            elif spec.score_field == "verbalized":
                v = d.get("verbalized_confidence")
                conf = float(v) if v is not None else 0.0
            elif spec.score_field == "consistency":
                v = d.get("self_consistency")
                conf = float(v) if v is not None else 0.0
            else:
                raise ValueError(spec.score_field)
        elif spec.mode == "calibrated":
            if spec.calibrator_input == "signals":
                conf = float(spec.calibrator.predict([sig.as_vector()])[0])
            else:
                conf = float(spec.calibrator.predict([[0.0]], sums=[estimator.estimate(sig)])[0])
        else:
            raise ValueError(spec.mode)
        conf = float(min(1.0, max(0.0, conf)))

        if spec.two_way:
            # external baselines gate on a single score: answer / refuse
            if not d["hits"]:
                decision, reason = "refuse", "no_evidence_retrieved"
            elif d["model_abstained"] and policy.respect_model_abstention:
                decision, reason = "refuse", "model_abstained"
            elif refuse_empty and is_empty_reply(d["draft_text"]):
                decision, reason = "refuse", "empty_reply"
            elif conf < policy.answer_threshold:
                decision, reason = "refuse", "low_score"
            else:
                decision, reason = "answer", "ok"
        elif refuse_empty and is_empty_reply(d["draft_text"]) and d["hits"] and not d["model_abstained"]:
            decision, reason = "refuse", "empty_reply"
        else:
            dec = policy.decide(
                confidence=conf,
                grounding_support=g["overall_support"],
                citation_coverage=g["citation_coverage"],
                has_evidence=bool(d["hits"]),
                language=d["language"],
                model_abstained=bool(d["model_abstained"]),
            )
            decision, reason = dec.decision, dec.reason

        answered = decision == "answer"
        dc = draft_correct(d)
        correct = int(answered and int(d["gold_answerable"]) == 1 and dc == 1)
        records.append(
            {
                "id": d["id"],
                "group": d.get("group"),
                "domain": d.get("domain"),
                "language": d["language"],
                "tags": d.get("tags", []),
                "seed": d.get("seed", 0),
                "gold_answerable": int(d["gold_answerable"]),
                "decision": decision,
                "reason": reason,
                "abstained": int(not answered),
                "correct": correct,
                "draft_correct": dc,
                "confidence": conf,
                "grounding_support": g["overall_support"],
                "citation_coverage": g["citation_coverage"],
                "signals": sig.as_vector(),
                "system": spec.name,
            }
        )
    return records


# --------------------------------------------------------------------------- #
# Tuning on the development split
# --------------------------------------------------------------------------- #

def _objective(records: List[dict]) -> tuple:
    s = summarise(records)
    # primary: decision accuracy; tie-break: higher answer rate (be useful)
    return (round(s["decision_accuracy"], 6), round(s["answer_rate"], 6))


def tune_two_way_threshold(drafts_dev: Sequence[dict], spec: SystemSpec, grid: Optional[Sequence[float]] = None) -> float:
    grid = grid if grid is not None else [round(x, 2) for x in np.arange(0.0, 1.0001, 0.02)]
    best_t, best = None, None
    for t in grid:
        s2 = SystemSpec(**{**spec.__dict__, "abstention": {**spec.abstention, "answer_threshold": float(t)}})
        obj = _objective(replay(drafts_dev, s2))
        if best is None or obj > best:
            best, best_t = obj, float(t)
    return best_t


def tune_three_way_thresholds(drafts_dev: Sequence[dict], spec: SystemSpec) -> dict:
    """Grid search of (tau_ans, tau_ref, tau_g) on the development drafts."""
    best, best_th = None, None
    for t_ans in np.arange(0.30, 0.851, 0.05):
        for t_ref in np.arange(0.10, 0.501, 0.05):
            if t_ref >= t_ans:
                continue
            for t_g in (0.10, 0.20, 0.30, 0.40):
                th = {"answer_threshold": round(float(t_ans), 2), "refuse_threshold": round(float(t_ref), 2),
                      "min_support": t_g, "min_citation_coverage": spec.abstention.get("min_citation_coverage", 0.4)}
                s2 = SystemSpec(**{**spec.__dict__, "abstention": {**spec.abstention, **th}})
                obj = _objective(replay(drafts_dev, s2))
                if best is None or obj > best:
                    best, best_th = obj, th
    return best_th


def tune_per_language(drafts_dev: Sequence[dict], spec: SystemSpec, languages: Sequence[str] = ("en", "si", "ta", "mixed")) -> dict:
    """Per-language answer threshold chosen on the development split."""
    per_lang = {}
    for lang in languages:
        sub = [d for d in drafts_dev if d["language"] == lang]
        if len(sub) < 3:
            continue
        best, best_t = None, None
        for t in np.arange(0.30, 0.851, 0.05):
            s2 = SystemSpec(**{**spec.__dict__, "abstention": {**spec.abstention, "answer_threshold": float(round(t, 2))}})
            obj = _objective(replay(sub, s2))
            if best is None or obj > best:
                best, best_t = obj, float(round(t, 2))
        per_lang[lang] = {"answer_threshold": best_t}
    return per_lang


def group_folds(drafts: Sequence[dict], n_folds: int = 5, seed: int = 13) -> List[set]:
    """Item groups shuffled into ``n_folds`` folds (every language version of an item
    stays in one fold, like the dev/test split)."""
    groups = sorted({d["group"] for d in drafts})
    rng = np.random.default_rng(seed)
    rng.shuffle(groups)
    return [set(groups[i::n_folds]) for i in range(n_folds)]


def cross_validate(drafts: Sequence[dict], checker: str, scale: dict, default_weights: dict,
                   default_thresholds: dict, no_thresholds: dict, n_folds: int = 5, seed: int = 13,
                   quick: bool = False) -> Dict[str, object]:
    """Group k-fold cross-fitting of the tuned decision rules.

    For every fold the TriSafe thresholds and the E1 / E7 gates are tuned on the other
    folds and applied to the held-out fold, so each query receives exactly one
    out-of-fold decision.  Returns the out-of-fold records per system (over all
    queries and seeds) and the thresholds chosen per fold.  B1 has nothing to tune
    and is included as the reference.  This is a robustness check of the single
    dev/test split used for the main tables, not a replacement for it.
    """
    conf = {**default_weights, "retrieval_signal": "cosine", **scale}
    folds = group_folds(drafts, n_folds, seed)
    has_e1 = any(d.get("verbalized_confidence") is not None for d in drafts)
    has_e7 = any(d.get("self_consistency") is not None for d in drafts)
    recs: Dict[str, List[dict]] = {"B1_vanilla": [], "TriSafe_devtuned": []}
    if has_e1:
        recs["E1_verbalized"] = []
    if has_e7:
        recs["E7_self_consistency"] = []
    fold_th = []
    for k, held in enumerate(folds):
        train = [d for d in drafts if d["group"] not in held]
        test = [d for d in drafts if d["group"] in held]
        b1 = SystemSpec("B1_vanilla", "B1", checker=checker, mode="score", score_field="retrieval",
                        confidence=conf, abstention=dict(no_thresholds), two_way=True)
        recs["B1_vanilla"] += replay(test, b1)
        base = SystemSpec("TriSafe_devtuned", "TriSafe", checker=checker, mode="weighted", confidence=conf,
                          abstention=dict(default_thresholds))
        th = tune_three_way_thresholds(train, base) if not quick else dict(default_thresholds)
        recs["TriSafe_devtuned"] += replay(test, SystemSpec(**{**base.__dict__, "abstention": th}))
        entry = {"fold": k, "n_train": len(train), "n_test": len(test), "trisafe": th}
        for name, field_ in (("E1_verbalized", "verbalized"), ("E7_self_consistency", "consistency")):
            if name not in recs:
                continue
            g = SystemSpec(name, name, checker=checker, mode="score", score_field=field_, confidence=conf,
                           abstention=dict(no_thresholds), two_way=True)
            t = tune_two_way_threshold(train, g, grid=[0.0, 0.5, 1.0] if quick else None)
            g.abstention["answer_threshold"] = t
            recs[name] += replay(test, g)
            entry[name] = t
        fold_th.append(entry)
    return {"n_folds": n_folds, "seed": seed, "records": recs, "fold_thresholds": fold_th}


def fit_calibrator(drafts_dev: Sequence[dict], estimator_cfg: dict, checker: str, method: str) -> PostHocCalibrator:
    est = ConfidenceEstimator(**estimator_cfg)
    X, y, sums = [], [], []
    for d in drafts_dev:
        sig = signals_for(d, est, checker)
        X.append(sig.as_vector())
        y.append(draft_correct(d))
        sums.append(est.estimate(sig))
    cal = PostHocCalibrator(method=method)
    if method == "platt":
        cal.fit(X, y)
    else:
        cal.fit(X, y, sums=sums)
    return cal


# --------------------------------------------------------------------------- #
# Reporting helpers
# --------------------------------------------------------------------------- #

def evaluate(records: List[dict]) -> dict:
    pl = per_language(records)
    return {"overall": summarise(records), "per_language": pl, "gaps": gaps(pl)}


def by_domain(records: List[dict]) -> Dict[str, dict]:
    out = {}
    for dom in sorted({r.get("domain") for r in records if r.get("domain")}):
        out[dom] = evaluate([r for r in records if r.get("domain") == dom])
    return out
