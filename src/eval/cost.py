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
                  n_boot: int = 2000, seed: int = 0, alpha: float = 0.05) -> Dict[str, dict]:
    """Unsafe / correct deliveries per system with paired bootstrap intervals,
    and the break-even cost of every system against ``reference``.

    ``systems`` maps a name to per-query records that all cover the same ids
    (seed-averaged records are fine).  The same resample indices are used for
    every system, so the intervals of differences and break-even costs are paired.
    """
    from src.eval.bootstrap import average_over_seeds

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
    for name, rs in aligned.items():
        s = cost_summary(rs)
        s["break_even_vs_ref"] = break_even(cost_summary(aligned[reference]), s) if reference in aligned and name != reference else None
        out[name] = s
    if n == 0 or n_boot <= 0:
        return out
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(n_boot, n))
    U = {name: np.array([_unsafe(r) for r in rs]) for name, rs in aligned.items()}
    K = {name: np.array([float(r["correct"]) for r in rs]) for name, rs in aligned.items()}
    lo, hi = 100 * alpha / 2, 100 * (1 - alpha / 2)
    for name in aligned:
        u = U[name][idx].sum(axis=1)
        k = K[name][idx].sum(axis=1)
        out[name]["unsafe_ci"] = [float(np.percentile(u, lo)), float(np.percentile(u, hi))]
        out[name]["correct_ci"] = [float(np.percentile(k, lo)), float(np.percentile(k, hi))]
        if reference in aligned and name != reference:
            du = U[reference][idx].sum(axis=1) - u
            dk = K[reference][idx].sum(axis=1) - k
            out[name]["delta_unsafe_vs_ref_ci"] = [float(np.percentile(-du, lo)), float(np.percentile(-du, hi))]
            out[name]["delta_correct_vs_ref_ci"] = [float(np.percentile(-dk, lo)), float(np.percentile(-dk, hi))]
            ok = du > 1e-9
            if ok.sum() >= 0.5 * n_boot:
                be = np.maximum(0.0, dk[ok] / du[ok])
                out[name]["break_even_ci"] = [float(np.percentile(be, lo)), float(np.percentile(be, hi))]
                out[name]["break_even_defined_frac"] = float(ok.mean())
            else:
                out[name]["break_even_ci"] = None
                out[name]["break_even_defined_frac"] = float(ok.mean())
    return out


def crossing_cost(a: Dict[str, object], b: Dict[str, object]) -> Optional[float]:
    """Cost at which system ``a`` and ``b`` have equal loss (``a`` is better above it
    when it has fewer unsafe deliveries).  Symmetric wrapper around break_even."""
    if float(a["unsafe"]) < float(b["unsafe"]):
        return break_even(b, a)
    if float(b["unsafe"]) < float(a["unsafe"]):
        return break_even(a, b)
    return None
