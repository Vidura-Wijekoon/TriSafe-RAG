"""Expand ``data/benchmark/items.json`` into the flat query list used by the
evaluation, write the deterministic dev/test split, and regenerate the
per-case-study ``eval/queries.json`` files as domain slices of the benchmark.

Usage:
    python scripts/build_benchmark.py
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.eval.splits import make_split, save_split  # noqa: E402

ITEMS = ROOT / "data" / "benchmark" / "items.json"
QUERIES = ROOT / "data" / "benchmark" / "queries.json"
SPLIT = ROOT / "data" / "benchmark" / "split.json"
CASE_DIRS = {
    "civil": ROOT / "casestudies" / "civil_documents",
    "health": ROOT / "casestudies" / "public_health",
    "disaster": ROOT / "casestudies" / "disaster_response",
}
LANG_SUFFIX = {"en": "en", "si": "si", "ta": "ta", "mixed": "mx"}


def expand(items: list[dict]) -> list[dict]:
    out = []
    for it in items:
        gold_all = list(it["gold"].get("all", []))
        for lang, q in it["queries"].items():
            gold = gold_all + list(it["gold"].get(lang, []))
            out.append(
                {
                    "id": f"{it['group']}-{LANG_SUFFIX[lang]}",
                    "group": it["group"],
                    "domain": it["domain"],
                    "language": lang,
                    "query": q,
                    "gold_answers": gold,
                    "gold_answer": gold[0] if gold else "",
                    "gold_answerable": int(it["answerable"]),
                    "tags": list(it.get("tags", [])),
                }
            )
    return out


def main() -> None:
    spec = json.loads(ITEMS.read_text(encoding="utf-8"))
    queries = expand(spec["items"])
    QUERIES.write_text(json.dumps(queries, indent=1, ensure_ascii=False), encoding="utf-8")

    split = make_split(queries, dev_fraction=0.2, seed=13)
    save_split(split, SPLIT)

    for domain, case_dir in CASE_DIRS.items():
        sub = [q for q in queries if q["domain"] == domain]
        (case_dir / "eval").mkdir(parents=True, exist_ok=True)
        (case_dir / "eval" / "queries.json").write_text(
            json.dumps(sub, indent=1, ensure_ascii=False), encoding="utf-8"
        )

    c_lang = Counter(q["language"] for q in queries)
    c_ans = Counter((q["language"], q["gold_answerable"]) for q in queries)
    print(f"{len(spec['items'])} items -> {len(queries)} queries")
    print("per language:", dict(c_lang))
    print("per (language, answerable):", {f"{k[0]}/{k[1]}": v for k, v in sorted(c_ans.items())})
    print(f"dev {len(split['dev'])} / test {len(split['test'])} queries; dev groups {split['dev_groups']}")
    for domain in CASE_DIRS:
        sub = [q for q in queries if q["domain"] == domain]
        print(f"  {domain}: {len(sub)} queries, {sum(1 for q in sub if not q['gold_answerable'])} unanswerable")


if __name__ == "__main__":
    main()
