"""Evaluation metrics: answer matching, selective-prediction metrics,
abstention precision/recall, and per-language tables.

Metric definitions (these are the definitions used in the paper):

* **answered**      -- decision == "answer".  ``defer`` and ``refuse`` both count
                       as abstention.
* **correct**       -- defined only for gold-answerable queries: the system
                       *answered* and at least one gold alias appears in the
                       returned text (alias match, see :func:`alias_match`).
                       An abstention on an answerable query is an error.
* **answerable accuracy** (``acc_ans``) -- mean ``correct`` over gold-answerable
                       queries.  This is the headline "accuracy" in the paper.
* **selective accuracy** -- mean ``correct`` over answerable queries that were
                       answered (precision of the delivered answers).
* **decision accuracy** -- over all queries: answerable & correct, or
                       unanswerable & abstained.  This is the only metric that
                       credits a correct refusal, and it is always labelled.
* **unanswerable abstention rate** -- fraction of gold-unanswerable queries on
                       which the system abstained (= abstention recall).
* **abstention P/R/F1** -- abstention as the predicted label, gold-unanswerable
                       as the positive class.  A system that never abstains has
                       F1 = 0.
* **answer rate**   -- fraction of all queries answered (coverage).
* **draft correct** -- whether the *draft* answer (before the policy) matched a
                       gold alias; for unanswerable queries the draft is correct
                       only if the generator itself abstained.  Used as the
                       target for confidence-quality metrics (ECE, Brier, AUROC)
                       so that those metrics do not depend on the policy.
* **ECE (answered)** -- ECE over answered queries, target = correct (an answer
                       given to an unanswerable query counts as wrong).
* **catch rate**    -- among answerable queries whose draft is wrong, the
                       fraction the policy abstained on (wrong answers caught).
* **retain rate**   -- among answerable queries whose draft is right, the
                       fraction the policy answered (correct answers kept).
"""

from __future__ import annotations

import re
import unicodedata
from typing import Dict, Iterable, List, Optional, Sequence

import numpy as np

from src.trust.calibration import auroc, brier_score, expected_calibration_error

_WS = re.compile(r"\s+")
_NUM_ALIAS = re.compile(r"^[\d,\.]+$")


def precision_at_k(predicted: Sequence[str], relevant: Sequence[str], k: int = 5) -> float:
    if k <= 0 or not predicted:
        return 0.0
    top = predicted[:k]
    return sum(1 for p in top if p in set(relevant)) / k


def mean_reciprocal_rank(predicted: Sequence[str], relevant: Sequence[str]) -> float:
    rel = set(relevant)
    for i, p in enumerate(predicted, start=1):
        if p in rel:
            return 1.0 / i
    return 0.0


def abstention_precision_recall(
    abstained: Sequence[int],
    gold_unanswerable: Sequence[int],
) -> tuple[float, float, float]:
    """Returns (precision, recall, f1) for the abstention decision treating
    "unanswerable" as the positive class."""
    if len(abstained) != len(gold_unanswerable):
        raise ValueError("length mismatch")
    tp = sum(1 for a, g in zip(abstained, gold_unanswerable) if a == 1 and g == 1)
    fp = sum(1 for a, g in zip(abstained, gold_unanswerable) if a == 1 and g == 0)
    fn = sum(1 for a, g in zip(abstained, gold_unanswerable) if a == 0 and g == 1)
    p = tp / (tp + fp) if (tp + fp) else 0.0
    r = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * p * r / (p + r) if (p + r) else 0.0
    return p, r, f1


# --------------------------------------------------------------------------- #
# Answer matching
# --------------------------------------------------------------------------- #

def _norm(text: str) -> str:
    t = unicodedata.normalize("NFKC", text or "").lower()
    t = t.replace("‍", "")  # zero-width joiner (common in Sinhala)
    t = _WS.sub(" ", t).strip()
    return t


def _norm_number(text: str) -> str:
    """'10,000.00' -> '10000'; '7.5' stays '7.5'; trailing dots are dropped."""
    t = text.replace(",", "").replace(" ", "").rstrip(".")
    if re.fullmatch(r"\d+\.0+", t):
        t = t.split(".")[0]
    return t


def alias_match(prediction: str, gold_aliases: Iterable[str]) -> bool:
    """True if any gold alias occurs in the prediction.

    Textual aliases are matched as normalised substrings.  Purely numeric
    aliases (digits, commas, dots) are matched as whole numbers: ``"8"`` does
    not match ``"1978"`` and ``"10,000"`` matches ``"10000"``.
    """
    pred = _norm(prediction)
    pred_nums = {_norm_number(n) for n in re.findall(r"\d[\d,\.]*", pred)}
    for alias in gold_aliases:
        a = _norm(alias)
        if not a:
            continue
        if _NUM_ALIAS.match(a):
            if _norm_number(a) in pred_nums:
                return True
        elif a in pred:
            return True
    return False


# --------------------------------------------------------------------------- #
# Record-level metrics
# --------------------------------------------------------------------------- #

def _mean(xs: Sequence[float]) -> Optional[float]:
    return float(np.mean(xs)) if len(xs) else None


def risk_coverage_aurc(records: List[dict]) -> Optional[float]:
    """Area under the risk-coverage curve when queries are released in order
    of decreasing confidence (lower is better).  Risk of a released query is
    1 - draft_correct.  Policy-independent."""
    if not records:
        return None
    rs = sorted(records, key=lambda r: -float(r.get("confidence", 0.0)))
    risks = []
    wrong = 0
    for i, r in enumerate(rs, start=1):
        wrong += 1 - int(r.get("draft_correct", 0))
        risks.append(wrong / i)
    return float(np.mean(risks))


def summarise(records: List[dict], n_bins: int = 10) -> Dict[str, Optional[float]]:
    """Aggregate metrics over per-query records.  Each record needs:
    gold_answerable, abstained, correct, draft_correct, confidence."""
    n = len(records)
    if n == 0:
        return {"n": 0}
    ans = [r for r in records if int(r["gold_answerable"]) == 1]
    unans = [r for r in records if int(r["gold_answerable"]) == 0]
    answered_ans = [r for r in ans if not int(r["abstained"])]
    answered_all = [r for r in records if not int(r["abstained"])]
    abst = [int(r["abstained"]) for r in records]
    gold_un = [1 - int(r["gold_answerable"]) for r in records]
    p, rcl, f1 = abstention_precision_recall(abst, gold_un)

    decision_correct = [
        int((int(r["gold_answerable"]) == 1 and int(r["correct"]) == 1)
            or (int(r["gold_answerable"]) == 0 and int(r["abstained"]) == 1))
        for r in records
    ]
    confs = [float(r["confidence"]) for r in records]
    draft_ok = [int(r.get("draft_correct", 0)) for r in records]
    wrong_drafts = [r for r in ans if int(r.get("draft_correct", 0)) == 0]
    right_drafts = [r for r in ans if int(r.get("draft_correct", 0)) == 1]
    reasons: Dict[str, int] = {}
    for r in records:
        if int(r["abstained"]):
            reasons[r.get("reason", "?")] = reasons.get(r.get("reason", "?"), 0) + 1
    answered_conf = [float(r["confidence"]) for r in answered_all]
    answered_ok = [int(r["correct"]) if int(r["gold_answerable"]) == 1 else 0 for r in answered_all]

    return {
        "n": n,
        "n_answerable": len(ans),
        "n_unanswerable": len(unans),
        "answer_rate": 1.0 - float(np.mean(abst)),
        "acc_answerable": _mean([int(r["correct"]) for r in ans]),
        "selective_accuracy": _mean([int(r["correct"]) for r in answered_ans]),
        "decision_accuracy": float(np.mean(decision_correct)),
        "unanswerable_abstention": _mean([int(r["abstained"]) for r in unans]),
        "abstention_precision": p,
        "abstention_recall": rcl,
        "abstention_f1": f1,
        "ece_draft": expected_calibration_error(confs, draft_ok, n_bins),
        "brier_draft": brier_score(confs, draft_ok),
        "auroc_draft": auroc(confs, draft_ok),
        "aurc": risk_coverage_aurc(records),
        "ece_answered": expected_calibration_error(answered_conf, answered_ok, n_bins)
        if answered_conf else None,
        "mean_confidence": float(np.mean(confs)),
        "mean_grounding": _mean([float(r.get("grounding_support", 0.0)) for r in records]),
        "defer_rate": _mean([int(r.get("decision") == "defer") for r in records]),
        "refuse_rate": _mean([int(r.get("decision") == "refuse") for r in records]),
        "draft_acc_answerable": _mean([int(r.get("draft_correct", 0)) for r in ans]),
        "catch_rate_wrong": _mean([int(r["abstained"]) for r in wrong_drafts]),
        "retain_rate_correct": _mean([1 - int(r["abstained"]) for r in right_drafts]),
        "abstention_reasons": reasons,
    }


def per_language(records: List[dict], languages: Sequence[str] = ("en", "si", "ta", "mixed")) -> Dict[str, dict]:
    out: Dict[str, dict] = {}
    for lang in languages:
        rs = [r for r in records if r.get("language") == lang]
        if not rs:
            continue
        s = summarise(rs)
        ans = [r for r in rs if int(r["gold_answerable"]) == 1]
        unans = [r for r in rs if int(r["gold_answerable"]) == 0]
        s["tpr"] = _mean([1 - int(r["abstained"]) for r in ans])
        s["fpr"] = _mean([1 - int(r["abstained"]) for r in unans])
        out[lang] = s
    return out


def gaps(per_lang: Dict[str, dict], min_n: int = 5) -> Dict[str, Optional[float]]:
    """Max-min gaps over languages with at least ``min_n`` queries."""
    eligible = {l: v for l, v in per_lang.items() if v.get("n", 0) >= min_n}

    def _gap(key: str) -> Optional[float]:
        vals = [v[key] for v in eligible.values() if v.get(key) is not None]
        return float(max(vals) - min(vals)) if len(vals) >= 2 else None

    g_acc = _gap("acc_answerable")
    g_tpr, g_fpr = _gap("tpr"), _gap("fpr")
    eo = None if (g_tpr is None and g_fpr is None) else max(x for x in (g_tpr, g_fpr) if x is not None)
    return {
        "gap_acc_answerable": g_acc,
        "gap_selective_accuracy": _gap("selective_accuracy"),
        "gap_answer_rate": _gap("answer_rate"),
        "gap_tpr": g_tpr,
        "gap_fpr": g_fpr,
        "equalized_odds_gap": eo,
        "cross_lingual_consistency": (max(0.0, 1.0 - g_acc) if g_acc is not None else None),
        "groups": sorted(eligible),
    }
