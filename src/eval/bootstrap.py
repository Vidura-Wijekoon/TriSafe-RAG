"""Paired bootstrap confidence intervals and paired permutation tests over
per-query records.

The unit of resampling is the *query*.  When several seeds are available,
each query's outcome is first averaged over seeds, so seed noise is folded
into the per-query estimate instead of being treated as independent samples.
All systems are resampled with the *same* index vector, which gives paired
intervals for differences between systems.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence

import numpy as np

from src.eval.metrics import gaps, per_language, summarise

def average_over_seeds(records: List[dict]) -> List[dict]:
    """Collapse records sharing an ``id`` into one record whose binary fields
    are replaced by their mean over seeds.  Fields: abstained, correct,
    draft_correct, confidence, grounding_support.  ``decision`` keeps the
    majority label."""
    by_id: Dict[str, List[dict]] = {}
    for r in records:
        by_id.setdefault(str(r["id"]), []).append(r)
    out = []
    for qid, rs in by_id.items():
        base = dict(rs[0])
        for k in ("abstained", "correct", "draft_correct", "confidence", "grounding_support"):
            vals = [float(r.get(k, 0.0)) for r in rs]
            base[k] = float(np.mean(vals))
        decisions = [r.get("decision") for r in rs]
        base["decision"] = max(set(decisions), key=decisions.count)
        base["n_seeds"] = len(rs)
        out.append(base)
    return out


def _all_metrics(records: List[dict]) -> Dict[str, Optional[float]]:
    s = summarise(records)
    g = gaps(per_language(records))
    out = {k: s.get(k) for k in ("acc_answerable", "answer_rate", "selective_accuracy", "decision_accuracy",
                                 "unanswerable_abstention", "abstention_f1", "ece_draft", "ece_answered", "aurc")}
    out.update({k: g.get(k) for k in ("gap_acc_answerable", "gap_answer_rate", "equalized_odds_gap")})
    return out


def paired_bootstrap(
    systems: Dict[str, List[dict]],
    metrics: Optional[Sequence[str]] = None,
    n_boot: int = 2000,
    seed: int = 0,
    alpha: float = 0.05,
    reference: Optional[str] = None,
) -> Dict[str, dict]:
    """Percentile bootstrap CIs for every (system, metric) and, when
    ``reference`` is given, for the difference system - reference.

    ``systems`` maps a system name to its per-query records.  All systems must
    cover the same query ids (records are aligned by ``id``).  Every system is
    resampled with the same index vectors, so the difference intervals are
    paired.
    """
    names = list(systems)
    aligned: Dict[str, List[dict]] = {}
    ids = None
    for name in names:
        recs = [_rounded(r) for r in average_over_seeds(systems[name])]
        recs.sort(key=lambda r: str(r["id"]))
        these = [str(r["id"]) for r in recs]
        if ids is None:
            ids = these
        elif these != ids:
            raise ValueError(f"system {name} does not cover the same query ids")
        aligned[name] = recs
    n = len(ids)
    rng = np.random.default_rng(seed)
    idx_matrix = rng.integers(0, n, size=(n_boot, n))

    points = {name: _all_metrics(aligned[name]) for name in names}
    metric_names = list(metrics) if metrics else list(points[names[0]].keys())
    samples: Dict[str, Dict[str, List[float]]] = {name: {m: [] for m in metric_names} for name in names}
    for b in range(n_boot):
        idx = idx_matrix[b]
        for name in names:
            vals = _all_metrics([aligned[name][i] for i in idx])
            for m in metric_names:
                samples[name][m].append(vals.get(m))

    def _pct(vals: List[Optional[float]], q: float) -> Optional[float]:
        v = [x for x in vals if x is not None]
        return float(np.percentile(v, q)) if v else None

    out: Dict[str, dict] = {}
    for name in names:
        out[name] = {}
        for m in metric_names:
            entry = {"point": points[name][m],
                     "ci_lo": _pct(samples[name][m], 100 * alpha / 2),
                     "ci_hi": _pct(samples[name][m], 100 * (1 - alpha / 2))}
            if reference and name != reference:
                diffs = [a - c for a, c in zip(samples[name][m], samples[reference][m])
                         if a is not None and c is not None]
                p_pt, r_pt = points[name][m], points[reference][m]
                entry["delta_vs_ref"] = {
                    "point": (p_pt - r_pt) if (p_pt is not None and r_pt is not None) else None,
                    "ci_lo": _pct(diffs, 100 * alpha / 2),
                    "ci_hi": _pct(diffs, 100 * (1 - alpha / 2)),
                }
            out[name][m] = entry
    return out


def _rounded(r: dict) -> dict:
    """Seed-averaged binary fields back to {0,1} by majority (>= 0.5)."""
    rr = dict(r)
    for k in ("abstained", "correct", "draft_correct"):
        rr[k] = int(float(r.get(k, 0.0)) >= 0.5)
    return rr


def paired_permutation_test(
    a: Sequence[float], b: Sequence[float], n_perm: int = 10000, seed: int = 0
) -> float:
    """Two-sided sign-flip permutation test on paired differences."""
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if a.shape != b.shape:
        raise ValueError("paired samples must align")
    d = a - b
    obs = abs(d.mean())
    if obs == 0.0 or len(d) == 0:
        return 1.0
    rng = np.random.default_rng(seed)
    signs = rng.choice([-1.0, 1.0], size=(n_perm, len(d)))
    perm_means = np.abs((signs * d).mean(axis=1))
    return float((np.sum(perm_means >= obs - 1e-12) + 1) / (n_perm + 1))


def paired_tests(systems: Dict[str, List[dict]], reference: str, fields: Sequence[str] = ("correct", "abstained", "draft_correct")) -> Dict[str, Dict[str, float]]:
    """p-values of paired permutation tests on per-query binary outcomes
    (answerable queries only for ``correct``)."""
    ref = {str(r["id"]): r for r in average_over_seeds(systems[reference])}
    out: Dict[str, Dict[str, float]] = {}
    for name, recs in systems.items():
        if name == reference:
            continue
        sysr = {str(r["id"]): r for r in average_over_seeds(recs)}
        out[name] = {}
        for f in fields:
            ids = [i for i in ref if i in sysr]
            if f == "correct":
                ids = [i for i in ids if int(ref[i]["gold_answerable"]) == 1]
            a = [float(sysr[i][f]) for i in ids]
            b = [float(ref[i][f]) for i in ids]
            out[name][f] = paired_permutation_test(a, b)
    return out
