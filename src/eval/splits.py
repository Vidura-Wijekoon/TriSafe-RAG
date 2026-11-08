"""Deterministic development / test split for the benchmark.

Queries in the benchmark come in *parallel groups*: one English question and
its Sinhala, Tamil and (sometimes) code-switched versions share a ``group``
id.  The split is made at the group level so that a fact seen during
threshold tuning never appears, in any language, in the test set.  The
split is stratified on answerability and fixed by ``seed``; it is written
to ``data/benchmark/split.json`` and read back by every experiment.
"""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Dict, Iterable, List, Sequence


def make_split(items: Sequence[dict], dev_fraction: float = 0.2, seed: int = 13) -> Dict[str, List[str]]:
    """``items`` are benchmark queries with ``group`` and ``gold_answerable``.
    Returns {"dev": [query ids], "test": [query ids], "dev_groups": [...]}."""
    groups: Dict[str, int] = {}
    for it in items:
        groups.setdefault(it["group"], int(it["gold_answerable"]))
    rng = random.Random(seed)
    answerable = sorted(g for g, a in groups.items() if a == 1)
    unanswerable = sorted(g for g, a in groups.items() if a == 0)
    rng.shuffle(answerable)
    rng.shuffle(unanswerable)
    n_dev_a = max(1, round(dev_fraction * len(answerable)))
    n_dev_u = max(1, round(dev_fraction * len(unanswerable))) if unanswerable else 0
    dev_groups = set(answerable[:n_dev_a] + unanswerable[:n_dev_u])
    dev = [it["id"] for it in items if it["group"] in dev_groups]
    test = [it["id"] for it in items if it["group"] not in dev_groups]
    return {"dev": dev, "test": test, "dev_groups": sorted(dev_groups), "seed": seed,
            "dev_fraction": dev_fraction}


def load_split(path: str | Path) -> Dict[str, List[str]]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def save_split(split: Dict, path: str | Path) -> None:
    Path(path).write_text(json.dumps(split, indent=2), encoding="utf-8")


def filter_records(records: Iterable[dict], ids: Iterable[str]) -> List[dict]:
    keep = set(ids)
    return [r for r in records if str(r["id"]) in keep]
