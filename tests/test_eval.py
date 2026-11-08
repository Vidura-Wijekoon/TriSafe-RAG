"""Tests for the evaluation layer: metrics, bootstrap, splits, calibrators,
normalised retrieval signals, per-language thresholds and the replay path."""

import json
from pathlib import Path

import numpy as np

from src.eval.benchmark import OFFLINE_BACKEND, Backend, DraftStore, SystemSpec, replay, retrieval_scale_from_drafts
from src.eval.bootstrap import average_over_seeds, paired_bootstrap, paired_permutation_test
from src.eval.metrics import alias_match, gaps, per_language, summarise
from src.eval.splits import make_split
from src.trust.abstention import AbstentionPolicy
from src.trust.calibration import ConfidenceEstimator, PostHocCalibrator, auroc, expected_calibration_error
from src.trust.fairness import FairnessAuditor

ROOT = Path(__file__).resolve().parents[1]


# ---------------------------------------------------------------- matching --
def test_alias_match_numbers_are_whole_tokens():
    assert alias_match("The fee is LKR 10,000.00 [1]", ["10,000"])
    assert alias_match("fee: 10000 rupees", ["10,000"])
    assert not alias_match("launched in 1978", ["8"])
    assert alias_match("at least 8 visits", ["8"])


def test_alias_match_text_is_normalised_substring():
    assert alias_match("Use PARACETAMOL only.", ["Paracetamol"])
    assert alias_match("පාරසිටමෝල් පමණක් භාවිතා කරන්න", ["පාරසිටමෝල්"])
    assert not alias_match("Use ibuprofen.", ["Paracetamol"])


# ---------------------------------------------------------------- metrics ---
def _rec(lang, answerable, abstained, correct, conf, draft_ok=None, qid=None):
    return {
        "id": qid or f"{lang}-{answerable}-{abstained}-{correct}-{conf}",
        "language": lang, "gold_answerable": answerable, "abstained": abstained,
        "correct": correct, "draft_correct": correct if draft_ok is None else draft_ok,
        "confidence": conf, "grounding_support": 0.5, "decision": "refuse" if abstained else "answer",
    }


def test_summarise_definitions():
    recs = [
        _rec("en", 1, 0, 1, 0.9, qid="a"),   # answered correctly
        _rec("en", 1, 1, 0, 0.2, qid="b"),   # abstained on answerable -> error
        _rec("en", 0, 1, 0, 0.1, qid="c"),   # correct refusal
        _rec("en", 0, 0, 0, 0.8, qid="d"),   # answered an unanswerable
    ]
    s = summarise(recs)
    assert s["acc_answerable"] == 0.5
    assert s["answer_rate"] == 0.5
    assert s["selective_accuracy"] == 1.0
    assert s["decision_accuracy"] == 0.5
    assert s["unanswerable_abstention"] == 0.5
    assert abs(s["abstention_f1"] - 0.5) < 1e-9  # tp=1 fp=1 fn=1


def test_never_abstaining_system_has_zero_f1():
    recs = [_rec("en", 1, 0, 1, 0.9, qid="a"), _rec("en", 0, 0, 0, 0.9, qid="b")]
    assert summarise(recs)["abstention_f1"] == 0.0


def test_gaps_ignore_tiny_groups():
    recs = [_rec("en", 1, 0, 1, 0.9, qid=f"e{i}") for i in range(6)] + \
           [_rec("si", 1, 0, 0, 0.9, qid=f"s{i}") for i in range(6)] + \
           [_rec("mixed", 1, 1, 0, 0.1, qid="m0")]
    g = gaps(per_language(recs), min_n=5)
    assert g["gap_acc_answerable"] == 1.0
    assert g["groups"] == ["en", "si"]


def test_fairness_auditor_reports_tpr_fpr():
    recs = [_rec("en", 1, 0, 1, 0.9, qid=f"e{i}") for i in range(5)] + [_rec("en", 0, 1, 0, 0.1, qid="eu")] + \
           [_rec("ta", 1, 1, 0, 0.2, qid=f"t{i}") for i in range(5)] + [_rec("ta", 0, 0, 0, 0.7, qid="tu")]
    rep = FairnessAuditor().audit(recs)
    assert rep.per_group_tpr["en"] == 1.0 and rep.per_group_tpr["ta"] == 0.0
    assert rep.per_group_fpr["en"] == 0.0 and rep.per_group_fpr["ta"] == 1.0
    assert rep.equalized_odds_gap == 1.0
    assert abs(rep.per_group_answer_rate["ta"] - 1 / 6) < 1e-9


# ------------------------------------------------------------ calibration ---
def test_ece_and_auroc_basic():
    assert expected_calibration_error([1.0, 1.0], [1, 1]) == 0.0
    assert abs(expected_calibration_error([0.9, 0.9], [1, 0]) - 0.4) < 1e-9
    assert auroc([0.9, 0.8, 0.2, 0.1], [1, 1, 0, 0]) == 1.0
    assert auroc([0.5, 0.5], [1, 1]) is None


def test_posthoc_calibrators_fit_and_predict_in_unit_interval():
    rng = np.random.default_rng(0)
    X = rng.uniform(size=(200, 5))
    y = (X[:, 2] + 0.2 * rng.normal(size=200) > 0.5).astype(int)
    sums = X.mean(axis=1)
    for method in ("platt", "isotonic", "temperature"):
        cal = PostHocCalibrator(method=method).fit(X, y, sums=sums)
        p = cal.predict(X, sums=sums)
        assert p.shape == (200,) and p.min() >= 0.0 and p.max() <= 1.0
    platt = PostHocCalibrator(method="platt").fit(X, y)
    assert platt.describe()["coef"][2] > 0  # grounding-like feature is predictive


def test_cosine_retrieval_signal_is_rescaled_and_rrf_is_flat():
    rrf = ConfidenceEstimator(retrieval_signal="rrf")
    s1, _ = rrf.normalise_retrieval(1 / 61, 1 / 62)
    assert 0.50 <= s1 <= 0.51  # sigma of a tiny RRF score is ~0.5
    cos = ConfidenceEstimator(retrieval_signal="cosine", ret_lo=0.7, ret_hi=0.9, margin_scale=10)
    s_ret, s_mar = cos.normalise_retrieval(0.8, 0.75)
    assert abs(s_ret - 0.5) < 1e-9 and abs(s_mar - 0.5) < 1e-9


# -------------------------------------------------------------- abstention --
def test_per_language_thresholds_override_global():
    p = AbstentionPolicy(answer_threshold=0.55, per_language={"ta": {"answer_threshold": 0.40}})
    kw = dict(grounding_support=0.9, citation_coverage=1.0, has_evidence=True)
    assert p.decide(confidence=0.45, language="en", **kw).decision == "defer"
    assert p.decide(confidence=0.45, language="ta", **kw).decision == "answer"


def test_model_abstention_is_respected():
    p = AbstentionPolicy(answer_threshold=0.0, refuse_threshold=0.0, min_support=0.0, min_citation_coverage=0.0)
    d = p.decide(confidence=0.9, grounding_support=0.9, citation_coverage=1.0, has_evidence=True, model_abstained=True)
    assert d.decision == "refuse" and d.reason == "model_abstained"


# ---------------------------------------------------------- bootstrap/split --
def test_split_is_group_disjoint_and_deterministic():
    items = json.loads((ROOT / "data" / "benchmark" / "queries.json").read_text(encoding="utf-8"))
    s1, s2 = make_split(items, 0.2, seed=13), make_split(items, 0.2, seed=13)
    assert s1 == s2
    dev_groups = {q["group"] for q in items if q["id"] in set(s1["dev"])}
    test_groups = {q["group"] for q in items if q["id"] in set(s1["test"])}
    assert not (dev_groups & test_groups)
    assert len(s1["dev"]) + len(s1["test"]) == len(items)


def test_paired_bootstrap_and_permutation():
    rng = np.random.default_rng(1)
    base = [_rec("en", 1, 0, int(rng.uniform() < 0.5), 0.6, qid=f"q{i}") for i in range(40)]
    better = [dict(r, correct=int(rng.uniform() < 0.9), draft_correct=1) for r in base]
    out = paired_bootstrap({"A": base, "B": better}, n_boot=200, seed=0, reference="A")
    e = out["B"]["acc_answerable"]
    assert e["ci_lo"] <= e["point"] <= e["ci_hi"]
    assert e["delta_vs_ref"]["point"] > 0
    p = paired_permutation_test([r["correct"] for r in better], [r["correct"] for r in base], n_perm=2000)
    assert p < 0.05
    assert len(average_over_seeds(base + base)) == 40


# ------------------------------------------------------------------ replay --
def test_offline_backend_replay_roundtrip(tmp_path):
    queries = json.loads((ROOT / "casestudies" / "civil_documents" / "eval" / "queries.json").read_text(encoding="utf-8"))[:6]
    be = Backend(OFFLINE_BACKEND, [ROOT / "casestudies" / "civil_documents" / "data" / "raw"])
    assert be.n_chunks > 0
    store = DraftStore(tmp_path / "drafts.jsonl")
    be.fill(queries, store)
    drafts = [store.get(q["id"]) for q in queries]
    scale = retrieval_scale_from_drafts(drafts)
    spec = SystemSpec("t", "t", checker="lexical", mode="weighted",
                      confidence={"retrieval_signal": "cosine", **scale},
                      abstention={"answer_threshold": 0.3, "refuse_threshold": 0.1, "min_support": 0.05, "min_citation_coverage": 0.3})
    recs = replay(drafts, spec)
    assert len(recs) == 6
    assert all(r["decision"] in {"answer", "defer", "refuse"} for r in recs)
    assert all(0.0 <= r["confidence"] <= 1.0 for r in recs)
    # a second store load replays identically
    store2 = DraftStore(tmp_path / "drafts.jsonl")
    assert replay([store2.get(q["id"]) for q in queries], spec) == recs


def test_staged_runner_matches_draft_format(tmp_path):
    from src.eval.staged import run_staged
    queries = json.loads((ROOT / "data" / "benchmark" / "queries.json").read_text(encoding="utf-8"))[:5]
    store = DraftStore(tmp_path / "drafts.jsonl")
    tstore = DraftStore(tmp_path / "drafts_t.jsonl")
    dirs = [ROOT / "casestudies" / c / "data" / "raw" for c in ("civil_documents", "public_health", "disaster_response")]
    info = run_staged(OFFLINE_BACKEND, dirs, 0, queries, store, tstore, [q for q in queries if q["language"] != "en"][:2], tmp_path)
    assert info["n_docs"] == 12 and info["n_chunks"] > 0
    rec = store.get(queries[0]["id"])
    for key in ("hits", "draft_text", "model_abstained", "grounding", "gold_answers", "gold_answerable"):
        assert key in rec
    assert "lexical" in rec["grounding"] and "text" in rec["hits"][0]
    assert len(tstore.records) == 2
    # resuming does no new work and leaves the stores unchanged
    before = dict(store.records)
    run_staged(OFFLINE_BACKEND, dirs, 0, queries, store, tstore, [], tmp_path)
    assert store.records == before
