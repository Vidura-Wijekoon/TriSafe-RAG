"""Shared helpers for the case-study drivers.

Each case study is a *domain slice* of the SL-PubServ-QA benchmark
(``data/benchmark/queries.json``): the civil-documents, public-health and
disaster-response queries over that domain's corpus.  The drivers run the
offline backend (hash embedder + extractive EchoLLM + lexical grounding) so
that the numbers reproduce in seconds on any machine, and they reuse the
system definitions of ``scripts/run_benchmark.py``.  The neural-backend
numbers for the same slices are read from ``results/neural/results.json``
(``test_by_domain``) by ``scripts/make_tables.py``.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

from src.eval.benchmark import (
    OFFLINE_BACKEND,
    Backend,
    DraftStore,
    SystemSpec,
    evaluate,
    replay,
    retrieval_scale_from_drafts,
    tune_three_way_thresholds,
)
from src.eval.splits import load_split

ROOT = Path(__file__).resolve().parents[1]
SPLIT = ROOT / "data" / "benchmark" / "split.json"

DEFAULT_WEIGHTS = {"w_retrieval": 0.20, "w_margin": 0.20, "w_grounding": 0.40, "w_citation": 0.10, "w_language": 0.10}
DEFAULT_THRESHOLDS = {"answer_threshold": 0.55, "refuse_threshold": 0.30, "min_support": 0.20, "min_citation_coverage": 0.40}
NO_THRESHOLDS = {"answer_threshold": 0.0, "refuse_threshold": 0.0, "min_support": 0.0, "min_citation_coverage": 0.0}


def run_case(case_dir, seed: int = 0) -> dict:
    case_dir = Path(case_dir)
    data_dir = case_dir / "data" / "raw"
    eval_path = case_dir / "eval" / "queries.json"
    out_dir = case_dir / "results"
    out_dir.mkdir(parents=True, exist_ok=True)

    queries: List[dict] = json.loads(eval_path.read_text(encoding="utf-8"))
    split = load_split(SPLIT) if SPLIT.exists() else {"dev": [], "test": [q["id"] for q in queries]}
    dev_ids = set(split["dev"])

    store = DraftStore(out_dir / "drafts_offline.jsonl")
    if any(q["id"] not in store for q in queries):
        be = Backend(OFFLINE_BACKEND, [data_dir], seed=seed)
        be.fill(queries, store)
    drafts = [store.get(q["id"]) for q in queries]
    dev_drafts = [d for d in drafts if d["id"] in dev_ids] or drafts
    scale = retrieval_scale_from_drafts(dev_drafts)
    conf = {**DEFAULT_WEIGHTS, "retrieval_signal": "cosine", **scale}

    systems = [
        SystemSpec("B1_vanilla", "B1 vanilla", checker="lexical", mode="score", score_field="retrieval",
                   confidence=conf, abstention=NO_THRESHOLDS, two_way=True),
        SystemSpec("TriSafe_default", "TriSafe-RAG (default thresholds)", checker="lexical", mode="weighted",
                   confidence=conf, abstention=DEFAULT_THRESHOLDS),
    ]
    base = SystemSpec("tmp", "", checker="lexical", mode="weighted", confidence=conf, abstention=DEFAULT_THRESHOLDS)
    tuned = tune_three_way_thresholds(dev_drafts, base)
    systems.append(SystemSpec("TriSafe_devtuned", "TriSafe-RAG (dev-tuned thresholds)", checker="lexical",
                              mode="weighted", confidence=conf, abstention=tuned, tuned_on="dev"))

    summaries: Dict[str, dict] = {}
    for s in systems:
        recs = replay(drafts, s)
        with (out_dir / f"{s.name}_per_query.jsonl").open("w", encoding="utf-8") as f:
            for r in recs:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        summaries[s.name] = {
            "label": s.label,
            "abstention": s.abstention,
            "all": evaluate(recs),
            "test": evaluate([r for r in recs if r["id"] not in dev_ids]),
        }
    payload = {"case_dir": str(case_dir.relative_to(ROOT)), "backend": "offline", "n_queries": len(queries),
               "retrieval_scale": scale, "configs": summaries}
    (out_dir / "summary.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    payload["summary_path"] = str(out_dir / "summary.json")
    return payload


def print_table(case_name: str, payload: dict) -> None:
    print(f"\n=== {case_name} (offline backend, {payload['n_queries']} queries, all splits) ===")
    header = ("Config", "AccAns", "AnsRate", "SelAcc", "UnansAbst", "AbstF1", "ECE", "GapAcc")
    rows = [header]
    for name, s in payload["configs"].items():
        o = s["all"]["overall"]
        g = s["all"]["gaps"]
        rows.append((
            name,
            f"{o['acc_answerable']:.3f}",
            f"{o['answer_rate']:.2f}",
            f"{(o['selective_accuracy'] if o['selective_accuracy'] is not None else float('nan')):.3f}",
            f"{(o['unanswerable_abstention'] if o['unanswerable_abstention'] is not None else float('nan')):.2f}",
            f"{o['abstention_f1']:.3f}",
            f"{o['ece_draft']:.3f}",
            f"{(g['gap_acc_answerable'] if g['gap_acc_answerable'] is not None else float('nan')):.2f}",
        ))
    widths = [max(len(r[i]) for r in rows) for i in range(len(rows[0]))]
    for r in rows:
        print("  " + "  ".join(c.ljust(w) for c, w in zip(r, widths)))
    for name, s in payload["configs"].items():
        pl = s["all"]["per_language"]
        print(f"  {name} per-language acc/ans-rate: " + ", ".join(
            f"{k}={v['acc_answerable']:.2f}/{v['answer_rate']:.2f}" for k, v in sorted(pl.items())))
