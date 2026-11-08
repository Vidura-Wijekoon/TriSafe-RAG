"""Run the SL-PubServ-QA benchmark end to end for one backend.

    python scripts/run_benchmark.py --backend offline
    python scripts/run_benchmark.py --backend neural --seeds 0 1 2

Outputs (under results/<backend>/):
    drafts_seed<k>.jsonl            cached drafts (retrieval + generation + grounding)
    drafts_translate_seed<k>.jsonl  drafts for the translate-then-retrieve baseline
    records/<system>.jsonl          per-query evaluation records, all seeds
    results.json                    every table in the paper, machine-readable
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

from src.eval.benchmark import (  # noqa: E402
    NEURAL_BACKEND,
    OFFLINE_BACKEND,
    DraftStore,
    SystemSpec,
    by_domain,
    evaluate,
    fit_calibrator,
    replay,
    retrieval_scale_from_drafts,
    tune_per_language,
    tune_three_way_thresholds,
    tune_two_way_threshold,
)
from src.eval.bootstrap import paired_bootstrap, paired_tests  # noqa: E402
from src.eval.splits import load_split  # noqa: E402
from src.trust.calibration import reliability_bins  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("benchmark")

SOURCE_DIRS = [
    ROOT / "casestudies" / "civil_documents" / "data" / "raw",
    ROOT / "casestudies" / "public_health" / "data" / "raw",
    ROOT / "casestudies" / "disaster_response" / "data" / "raw",
]
QUERIES = ROOT / "data" / "benchmark" / "queries.json"
SPLIT = ROOT / "data" / "benchmark" / "split.json"

DEFAULT_WEIGHTS = {"w_retrieval": 0.20, "w_margin": 0.20, "w_grounding": 0.40, "w_citation": 0.10, "w_language": 0.10}
DEFAULT_THRESHOLDS = {"answer_threshold": 0.55, "refuse_threshold": 0.30, "min_support": 0.20, "min_citation_coverage": 0.40}
NO_THRESHOLDS = {"answer_threshold": 0.0, "refuse_threshold": 0.0, "min_support": 0.0, "min_citation_coverage": 0.0}
# B3: abstention without grounding-aware confidence; weights renormalised to sum to 1
B3_WEIGHTS = {"w_retrieval": 0.40, "w_margin": 0.40, "w_grounding": 0.0, "w_citation": 0.0, "w_language": 0.20}


def _progress(i, n, elapsed):
    if i == n or i % 10 == 0:
        rate = elapsed / max(i, 1)
        log.info("  drafts %d/%d  (%.1fs/query, ~%.0f min left)", i, n, rate, rate * (n - i) / 60)


def build_systems(checker: str, scale: dict, dev_drafts, has_verbalized: bool, has_translate: bool):
    conf_cos = {**DEFAULT_WEIGHTS, "retrieval_signal": "cosine", **scale}
    conf_rrf = {**DEFAULT_WEIGHTS, "retrieval_signal": "rrf"}
    systems = []

    # --- ablations -------------------------------------------------------- #
    systems.append(SystemSpec("B1_vanilla", "B1 vanilla (LLM self-abstention only)", checker=checker,
                              mode="score", score_field="retrieval", confidence=conf_cos,
                              abstention=NO_THRESHOLDS, two_way=True,
                              notes="Answers everything the LLM answers; confidence = normalised top-1 cosine."))
    systems.append(SystemSpec("B2_grounding_only", "B2 grounding only (no thresholds)", checker=checker,
                              mode="weighted", confidence=conf_cos, abstention=NO_THRESHOLDS,
                              notes="Full confidence score reported, no abstention thresholds."))
    systems.append(SystemSpec("B3_abstention_only", "B3 abstention only (w_gr = 0)", checker=checker,
                              mode="weighted", confidence={**conf_cos, **B3_WEIGHTS},
                              abstention=DEFAULT_THRESHOLDS))
    systems.append(SystemSpec("TriSafe_default", "TriSafe-RAG (default thresholds)", checker=checker,
                              mode="weighted", confidence=conf_cos, abstention=DEFAULT_THRESHOLDS))
    if checker != "lexical":
        systems.append(SystemSpec("TriSafe_lexical", "TriSafe-RAG with lexical (Jaccard) grounding", checker="lexical",
                                  mode="weighted", confidence=conf_cos, abstention=DEFAULT_THRESHOLDS,
                                  notes="Same policy, Jaccard grounding instead of NLI."))
    systems.append(SystemSpec("TriSafe_rrf", "TriSafe-RAG with raw RRF retrieval signal", checker=checker,
                              mode="weighted", confidence=conf_rrf, abstention=DEFAULT_THRESHOLDS,
                              notes="Legacy retrieval signal; shows the effect of the near-constant RRF scale."))

    # --- dev-tuned variants ----------------------------------------------- #
    base = SystemSpec("tmp", "", checker=checker, mode="weighted", confidence=conf_cos, abstention=DEFAULT_THRESHOLDS)
    th = tune_three_way_thresholds(dev_drafts, base)
    systems.append(SystemSpec("TriSafe_devtuned", "TriSafe-RAG (thresholds tuned on dev)", checker=checker,
                              mode="weighted", confidence=conf_cos, abstention=th, tuned_on="dev"))
    per_lang = tune_per_language(dev_drafts, SystemSpec("tmp", "", checker=checker, mode="weighted",
                                                        confidence=conf_cos, abstention=th))
    systems.append(SystemSpec("TriSafe_perlang", "TriSafe-RAG + per-language thresholds (dev)", checker=checker,
                              mode="weighted", confidence=conf_cos,
                              abstention={**th, "per_language": per_lang}, tuned_on="dev",
                              notes="Fairness enforced by language-specific answer thresholds."))

    # --- external baselines (two-way gates, thresholds tuned on dev) -------- #
    if has_verbalized:
        e1 = SystemSpec("E1_verbalized", "E1 LLM verbalized confidence", checker=checker, mode="score",
                        score_field="verbalized", confidence=conf_cos, abstention=dict(NO_THRESHOLDS), two_way=True)
        e1.abstention["answer_threshold"] = tune_two_way_threshold(dev_drafts, e1)
        e1.tuned_on = "dev"
        systems.append(e1)
    e2 = SystemSpec("E2_retrieval_gate", "E2 retrieval-similarity gate (CRAG-style evaluator proxy)", checker=checker,
                    mode="score", score_field="retrieval", confidence=conf_cos, abstention=dict(NO_THRESHOLDS), two_way=True)
    e2.abstention["answer_threshold"] = tune_two_way_threshold(dev_drafts, e2)
    e2.tuned_on = "dev"
    systems.append(e2)
    for method, name, label, inp in (
        ("platt", "E3_platt", "E3 Platt (logistic) on the five signals", "signals"),
        ("isotonic", "E4_isotonic", "E4 isotonic on the TriSafe sum", "sum"),
        ("temperature", "E5_temperature", "E5 temperature scaling on the TriSafe sum", "sum"),
    ):
        cal = fit_calibrator(dev_drafts, conf_cos, checker, method)
        e = SystemSpec(name, label, checker=checker, mode="calibrated", calibrator=cal, calibrator_input=inp,
                       confidence=conf_cos, abstention=dict(NO_THRESHOLDS), two_way=True, tuned_on="dev",
                       notes=json.dumps(cal.describe()))
        e.abstention["answer_threshold"] = tune_two_way_threshold(dev_drafts, e)
        systems.append(e)
    # 3-way TriSafe policy on top of the Platt-calibrated confidence
    cal = fit_calibrator(dev_drafts, conf_cos, checker, "platt")
    tp = SystemSpec("TriSafe_platt", "TriSafe-RAG with Platt-calibrated confidence", checker=checker,
                    mode="calibrated", calibrator=cal, calibrator_input="signals", confidence=conf_cos,
                    abstention=DEFAULT_THRESHOLDS, tuned_on="dev")
    tp.abstention = tune_three_way_thresholds(dev_drafts, tp)
    systems.append(tp)
    return systems, th, per_lang


def weight_sweep(drafts_dev, drafts_test, checker, scale, thresholds):
    rows = []
    for w_gr in (0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6):
        rest = 1.0 - w_gr
        # redistribute the remaining weight in the default proportions (0.2,0.2,0.1,0.1 -> 1/3,1/3,1/6,1/6)
        w = {"w_retrieval": rest / 3, "w_margin": rest / 3, "w_grounding": w_gr, "w_citation": rest / 6, "w_language": rest / 6}
        spec = SystemSpec("sweep", "", checker=checker, mode="weighted",
                          confidence={**w, "retrieval_signal": "cosine", **scale}, abstention=thresholds)
        d = evaluate(replay(drafts_dev, spec))["overall"]
        t = evaluate(replay(drafts_test, spec))["overall"]
        rows.append({"w_gr": w_gr, "dev_ece": d["ece_draft"], "test_ece": t["ece_draft"],
                     "dev_auroc": d["auroc_draft"], "test_auroc": t["auroc_draft"],
                     "dev_decision_acc": d["decision_accuracy"], "test_decision_acc": t["decision_accuracy"]})
    return rows


def retrieval_stats(drafts):
    """Recall@k of the gold-bearing document is not annotated; we report the
    share of queries whose top-k passages contain a gold alias (answerable
    only), by language, as an evidence-availability proxy."""
    from src.eval.metrics import alias_match
    out = {}
    for lang in ("en", "si", "ta", "mixed"):
        sub = [d for d in drafts if d["language"] == lang and d["gold_answerable"] == 1]
        if not sub:
            continue
        hit = sum(1 for d in sub if any(alias_match(h.get("text", ""), d["gold_answers"]) for h in d["hits"]))
        out[lang] = {"n": len(sub), "gold_in_topk": hit / len(sub) if sub else None}
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", choices=["offline", "neural"], default="offline")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0])
    ap.add_argument("--out", default=None)
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--skip-translate", action="store_true")
    args = ap.parse_args()

    spec = OFFLINE_BACKEND if args.backend == "offline" else NEURAL_BACKEND
    try:  # hybrid CPUs: fewer, faster threads beat oversubscription for decode
        import os
        import torch
        torch.set_num_threads(int(os.environ.get("TRISAFE_TORCH_THREADS", "16")))
    except Exception:  # pragma: no cover
        pass
    out_dir = Path(args.out or ROOT / "results" / spec.name)
    (out_dir / "records").mkdir(parents=True, exist_ok=True)
    queries = json.loads(QUERIES.read_text(encoding="utf-8"))
    split = load_split(SPLIT)
    dev_ids, test_ids = set(split["dev"]), set(split["test"])
    seeds = args.seeds if spec.llm.get("name") != "echo" else [0]

    # ---- 1. drafts ------------------------------------------------------- #
    drafts_by_seed, trans_by_seed = {}, {}
    backend_info = {}
    for seed in seeds:
        store = DraftStore(out_dir / f"drafts_seed{seed}.jsonl")
        tstore = DraftStore(out_dir / f"drafts_translate_seed{seed}.jsonl")
        need = any(q["id"] not in store for q in queries)
        # translate-then-retrieve drafts are only needed for non-English *test*
        # queries: the E6 baseline uses the default policy and is not tuned.
        trans_queries = [q for q in queries if q["language"] != "en" and q["id"] in test_ids]
        need_t = spec.translate_queries and not args.skip_translate and any(
            q["id"] not in tstore for q in trans_queries)
        if need or need_t:
            log.info("producing drafts for backend %s (seed %d), staged to limit memory", spec.name, seed)
            from src.eval.staged import run_staged
            info = run_staged(spec, SOURCE_DIRS, seed, queries, store,
                              tstore if (spec.translate_queries and not args.skip_translate) else None,
                              trans_queries if (spec.translate_queries and not args.skip_translate) else [],
                              out_dir)
            backend_info = info or backend_info
        drafts_by_seed[seed] = [store.get(q["id"]) for q in queries]
        if spec.translate_queries and not args.skip_translate:
            trans_by_seed[seed] = [
                tstore.get(q["id"]) if q["language"] != "en" else store.get(q["id"])
                for q in queries if q["id"] in test_ids
            ]
    if not backend_info:
        backend_info = {"embedder": spec.embedder, "llm": spec.llm, "checkers": list(spec.checkers),
                        "chunker": spec.chunker, "retrieval": spec.retrieval}

    # back-fill chunk text on cached hits (needed by the evidence-availability statistic)
    from src.eval.benchmark import chunk_text_lookup
    _lookup = chunk_text_lookup(SOURCE_DIRS, spec.chunker)
    for seed in seeds:
        for coll in (drafts_by_seed.get(seed, []), trans_by_seed.get(seed, [])):
            for d in coll:
                for h in d["hits"]:
                    if not h.get("text"):
                        h["text"] = _lookup.get(h["chunk_id"], "")

    all_drafts = [d for s in seeds for d in drafts_by_seed[s]]
    dev_drafts = [d for d in all_drafts if d["id"] in dev_ids]
    test_drafts = [d for d in all_drafts if d["id"] in test_ids]
    checker = "nli" if "nli" in spec.checkers else "lexical"
    scale = retrieval_scale_from_drafts(dev_drafts)
    log.info("retrieval scale from dev: %s", scale)

    # ---- 2. systems ------------------------------------------------------ #
    systems, tuned_th, per_lang = build_systems(checker, scale, dev_drafts,
                                                has_verbalized=spec.verbalized_confidence,
                                                has_translate=bool(trans_by_seed))
    results = {"backend": spec.name, "backend_info": backend_info, "seeds": seeds,
               "n_queries": len(queries), "n_dev": len(dev_ids), "n_test": len(test_ids),
               "checker": checker, "retrieval_scale": scale, "tuned_thresholds": tuned_th,
               "per_language_thresholds": per_lang, "systems": {}, "generated_at": time.strftime("%Y-%m-%d %H:%M:%S")}

    records_by_system = {}
    for s in systems:
        recs = replay(all_drafts, s)
        records_by_system[s.name] = recs
        with (out_dir / "records" / f"{s.name}.jsonl").open("w", encoding="utf-8") as f:
            for r in recs:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        test_recs = [r for r in recs if r["id"] in test_ids]
        dev_recs = [r for r in recs if r["id"] in dev_ids]
        results["systems"][s.name] = {
            "label": s.label, "mode": s.mode, "two_way": s.two_way, "checker": s.checker,
            "confidence": {k: v for k, v in s.confidence.items()}, "abstention": s.abstention,
            "tuned_on": s.tuned_on, "notes": s.notes,
            "test": evaluate(test_recs), "dev": evaluate(dev_recs), "all": evaluate(recs),
            "test_by_domain": by_domain(test_recs), "all_by_domain": by_domain(recs),
            "reliability_test": reliability_bins([r["confidence"] for r in test_recs],
                                                 [r["draft_correct"] for r in test_recs]),
            "reliability_answered_test": reliability_bins(
                [r["confidence"] for r in test_recs if not r["abstained"]],
                [r["correct"] if r["gold_answerable"] else 0 for r in test_recs if not r["abstained"]]),
        }
        log.info("%-22s test: acc_ans=%.3f ans_rate=%.3f ece=%.3f f1=%.3f gapA=%s", s.name,
                 results["systems"][s.name]["test"]["overall"]["acc_answerable"] or 0,
                 results["systems"][s.name]["test"]["overall"]["answer_rate"],
                 results["systems"][s.name]["test"]["overall"]["ece_draft"],
                 results["systems"][s.name]["test"]["overall"]["abstention_f1"],
                 results["systems"][s.name]["test"]["gaps"]["gap_acc_answerable"])

    # translate-then-retrieve baseline: different drafts, TriSafe default policy
    if trans_by_seed:
        tdrafts = [d for s in seeds for d in trans_by_seed[s]]
        e6 = SystemSpec("E6_translate_retrieve", "E6 translate-then-retrieve + TriSafe policy", checker=checker,
                        mode="weighted", confidence={**DEFAULT_WEIGHTS, "retrieval_signal": "cosine", **scale},
                        abstention=DEFAULT_THRESHOLDS)
        recs = replay(tdrafts, e6)
        records_by_system[e6.name] = recs
        with (out_dir / "records" / f"{e6.name}.jsonl").open("w", encoding="utf-8") as f:
            for r in recs:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        test_recs = [r for r in recs if r["id"] in test_ids]
        results["systems"][e6.name] = {
            "label": e6.label, "mode": e6.mode, "two_way": False, "checker": checker,
            "confidence": e6.confidence, "abstention": e6.abstention, "tuned_on": None,
            "notes": "Non-English queries translated to English by the same LLM before retrieval.",
            "test": evaluate(test_recs), "dev": evaluate([r for r in recs if r["id"] in dev_ids]),
            "all": evaluate(recs), "test_by_domain": by_domain(test_recs), "all_by_domain": by_domain(recs),
            "reliability_test": reliability_bins([r["confidence"] for r in test_recs], [r["draft_correct"] for r in test_recs]),
        }
        results["translate_retrieval_stats"] = retrieval_stats([d for d in tdrafts if d["id"] in test_ids])

    # ---- 3. statistics --------------------------------------------------- #
    test_sets = {k: [r for r in v if r["id"] in test_ids] for k, v in records_by_system.items()}
    log.info("bootstrap (%d resamples) ...", args.n_boot)
    results["bootstrap_test"] = paired_bootstrap(test_sets, n_boot=args.n_boot, seed=0, reference="B1_vanilla")
    results["paired_tests_vs_B1"] = paired_tests(test_sets, reference="B1_vanilla")
    results["paired_tests_vs_default"] = paired_tests(
        {k: v for k, v in test_sets.items() if k != "B1_vanilla"}, reference="TriSafe_default")
    results["weight_sweep"] = weight_sweep(dev_drafts, test_drafts, checker, scale, DEFAULT_THRESHOLDS)
    results["retrieval_stats_test"] = retrieval_stats(test_drafts)
    results["model_abstention_rate"] = {
        lang: float(np.mean([d["model_abstained"] for d in test_drafts if d["language"] == lang]))
        for lang in ("en", "si", "ta", "mixed")
    }
    results["draft_accuracy_test"] = {
        lang: float(np.mean([
            int(__import__("src.eval.metrics", fromlist=["alias_match"]).alias_match(d["draft_text"], d["gold_answers"]))
            for d in test_drafts if d["language"] == lang and d["gold_answerable"] == 1]))
        for lang in ("en", "si", "ta", "mixed")
    }
    (out_dir / "results.json").write_text(json.dumps(results, indent=1, ensure_ascii=False), encoding="utf-8")
    log.info("wrote %s", out_dir / "results.json")


if __name__ == "__main__":
    main()
