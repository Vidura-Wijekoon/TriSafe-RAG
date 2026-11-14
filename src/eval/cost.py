"""Cost analysis of delivered answers.

An *unsafe delivery* is an answer the system delivered that it should not
have: a wrong answer on a gold-answerable query, or any answer on a
gold-unanswerable query.  A *correct delivery* is a delivered answer that
matches the gold.  Every abstention policy trades the two: it removes unsafe
deliveries at the price of correct ones.

Let a deployment weigh one unsafe delivery as ``c`` lost correct answers.
The loss of a system is ``c * U - K`` (``U`` unsafe, ``K`` correct deliveries),
and a system beats a reference exactly when ``c`` exceeds the *break-even
cost*

    c* = (K_ref - K_sys) / (U_ref - U_sys),

defined when the system has fewer unsafe deliveries than the reference.  A
break-even of 4 means: TriSafe is the better choice as soon as one confident
wrong answer costs more than four withheld correct ones.  Records may be
seed-averaged (fractional ``abstained`` / ``correct``), in which case the
counts are expected counts over the seeds.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence

import numpy as np

LANGS = ("en", "si", "ta", "mixed")


def _unsafe(r: dict) -> float:
    answered = 1.0 - float(r["abstained"])
    correct = float(r["correct"]) if int(r["gold_answerable"]) == 1 else 0.0
    return max(0.0, answered - correct)


def _lost_correct(r: dict) -> float:
    """Answerable query whose draft was right but which the policy withheld."""
    if int(r["gold_answerable"]) != 1:
        return 0.0
    return max(0.0, float(r.get("draft_correct", 0.0)) - float(r["correct"]))


def cost_summary(records: Sequence[dict]) -> Dict[str, object]:
    rs = list(records)
    per_lang = {}
    for lang in LANGS:
        sub = [r for r in rs if r.get("language") == lang]
        if sub:
            per_lang[lang] = {"n": len(sub), "unsafe": float(sum(_unsafe(r) for r in sub)),
                              "correct": float(sum(float(r["correct"]) for r in sub))}
    return {
        "n": len(rs),
        "unsafe": float(sum(_unsafe(r) for r in rs)),
        "unsafe_unanswerable": float(sum(_unsafe(r) for r in rs if int(r["gold_answerable"]) == 0)),
        "unsafe_wrong_answer": float(sum(_unsafe(r) for r in rs if int(r["gold_answerable"]) == 1)),
        "correct": float(sum(float(r["correct"]) for r in rs)),
        "lost_correct": float(sum(_lost_correct(r) for r in rs)),
        "per_language": per_lang,
    }


def break_even(ref: Dict[str, object], sys_: Dict[str, object]) -> Optional[float]:
    du = float(ref["unsafe"]) - float(sys_["unsafe"])
    dk = float(ref["correct"]) - float(sys_["correct"])
    if du <= 1e-9:
        return None  # the system does not remove unsafe deliveries; no finite break-even
    return max(0.0, dk / du)


def cost_analysis(systems: Dict[str, List[dict]], reference: str = "B1_vanilla",
                  n_boot: int = 2000, seed: int = 0, alpha: float = 0.05,
                  extra_references: Sequence[str] = ("B1_empty_filter",)) -> Dict[str, dict]:
    """Unsafe / correct deliveries per system with paired cluster-bootstrap
    intervals (item groups resampled together), the break-even cost of every
    system against ``reference``, and against each of ``extra_references``
    (stored as ``break_even_vs_<name>`` / ``break_even_vs_<name>_ci``).

    ``systems`` maps a name to per-query records that all cover the same ids
    (seed-averaged records are fine).  The same group draws are used for every
    system, so the intervals of differences and break-even costs are paired.
    """
    from src.eval.bootstrap import average_over_seeds, group_resamples, _groups_of

    aligned: Dict[str, List[dict]] = {}
    ids = None
    for name, recs in systems.items():
        rs = sorted(average_over_seeds(recs), key=lambda r: str(r["id"]))
        these = [str(r["id"]) for r in rs]
        if ids is None:
            ids = these
        elif these != ids:
            raise ValueError(f"system {name} does not cover the same query ids")
        aligned[name] = rs
    n = len(ids or [])
    out: Dict[str, dict] = {}
    extras = [x for x in extra_references if x in aligned]
    for name, rs in aligned.items():
        s = cost_summary(rs)
        s["break_even_vs_ref"] = break_even(cost_summary(aligned[reference]), s) if reference in aligned and name != reference else None
        for x in extras:
            s[f"break_even_vs_{x}"] = break_even(cost_summary(aligned[x]), s) if name != x else None
        out[name] = s
    if n == 0 or n_boot <= 0:
        return out
    rng = np.random.default_rng(seed)
    draws = group_resamples(_groups_of(aligned[reference] if reference in aligned else next(iter(aligned.values()))), n_boot, rng)
    U = {name: np.array([_unsafe(r) for r in rs]) for name, rs in aligned.items()}
    K = {name: np.array([float(r["correct"]) for r in rs]) for name, rs in aligned.items()}
    Ub = {name: np.array([U[name][d].sum() for d in draws]) for name in aligned}
    Kb = {name: np.array([K[name][d].sum() for d in draws]) for name in aligned}
    lo, hi = 100 * alpha / 2, 100 * (1 - alpha / 2)

    def _be_ci(ref_name: str, name: str):
        du = Ub[ref_name] - Ub[name]
        dk = Kb[ref_name] - Kb[name]
        ok = du > 1e-9
        if ok.sum() >= 0.5 * n_boot:
            be = np.maximum(0.0, dk[ok] / du[ok])
            return [float(np.percentile(be, lo)), float(np.percentile(be, hi))], float(ok.mean()), du, dk
        return None, float(ok.mean()), du, dk

    for name in aligned:
        u, k = Ub[name], Kb[name]
        out[name]["unsafe_ci"] = [float(np.percentile(u, lo)), float(np.percentile(u, hi))]
        out[name]["correct_ci"] = [float(np.percentile(k, lo)), float(np.percentile(k, hi))]
        if reference in aligned and name != reference:
            ci_, frac, du, dk = _be_ci(reference, name)
            out[name]["delta_unsafe_vs_ref_ci"] = [float(np.percentile(-du, lo)), float(np.percentile(-du, hi))]
            out[name]["delta_correct_vs_ref_ci"] = [float(np.percentile(-dk, lo)), float(np.percentile(-dk, hi))]
            out[name]["break_even_ci"] = ci_
            out[name]["break_even_defined_frac"] = frac
        for x in extras:
            if name != x:
                ci_, frac, _, _ = _be_ci(x, name)
                out[name][f"break_even_vs_{x}_ci"] = ci_
                out[name][f"break_even_vs_{x}_defined_frac"] = frac
    return out


def crossing_cost(a: Dict[str, object], b: Dict[str, object]) -> Optional[float]:
    """Cost at which system ``a`` and ``b`` have equal loss (``a`` is better above it
    when it has fewer unsafe deliveries).  Symmetric wrapper around break_even."""
    if float(a["unsafe"]) < float(b["unsafe"]):
        return break_even(b, a)
    if float(b["unsafe"]) < float(a["unsafe"]):
        return break_even(a, b)
    return None
