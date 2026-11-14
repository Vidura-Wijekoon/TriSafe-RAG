"""Deferral as an outcome for the user.

A *defer* decision withholds the draft and shows the retrieved evidence.  The main
metrics score it as a plain abstention.  This script asks, for every deferred
answerable test query of the rules that can defer, whether the evidence the user
is shown contains a gold alias:

* in the top-1 passage and in the top-k passages shown (English text), and
* *readable*: in the shown passages for English and code-switched queries, and
  in the Sinhala or Tamil summary of a shown document for Sinhala and Tamil
  queries (the part of the corpus such a user can read).

It then re-prices each rule when a deferral with a readable correct answer costs
``d`` withheld correct answers instead of 1 (``d = 1`` is the cost analysis of
src/eval/cost.py).  Loss of a rule: ``c*U + (N_ans - K) - (1 - d)*D_vis`` with
``U`` unsafe deliveries, ``K`` correct deliveries, ``D_vis`` deferred answerable
queries with a readable gold alias.  Break-even against a reference rule:
``c* = [(K_ref - K) - (1 - d)(D_vis - D_vis_ref)] / (U_ref - U)``.

Replay only: reads results/<backend>/records/*.jsonl, the draft caches and the
corpus; writes results/<backend>/defer_analysis.json.

    python scripts/defer_analysis.py --backend neural7b
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.eval.metrics import alias_match  # noqa: E402

SOURCE_DIRS = [ROOT / "casestudies" / d / "data" / "raw" for d in ("civil_documents", "public_health", "disaster_response")]
SYSTEMS = ("TriSafe_default", "TriSafe_devtuned", "TriSafe_empty", "B3_abstention_only", "TriSafe_perlang", "TriSafe_platt", "TriSafe_nolang")
D_VALUES = (1.0, 0.5, 0.25)
LANGS = ("en", "si", "ta", "mixed")
SUMMARY_HEADINGS = {"si": "සිංහල", "ta": "தமிழ்"}


def corpus_summaries() -> dict:
    """{doc_id: {"si": text, "ta": text}} from the '## <script> (summary)' sections of the raw documents."""
    out = {}
    for d in SOURCE_DIRS:
        for path in sorted(d.glob("*.md")):
            text = path.read_text(encoding="utf-8")
            sections = re.split(r"^## ", text, flags=re.M)
            summ = {}
            for sec in sections[1:]:
                head, _, body = sec.partition("\n")
                for lang, word in SUMMARY_HEADINGS.items():
                    if word in head:
                        summ[lang] = body
            out[path.stem] = summ
    return out


def load_records(path: Path, test_ids: set) -> list:
    return [r for r in (json.loads(l) for l in path.read_text(encoding="utf-8").splitlines()) if r["id"] in test_ids]


def load_hits(backend_dir: Path, seeds) -> dict:
    """{seed: {id: hits}} from the draft caches (retrieval does not depend on the seed, but we read each cache)."""
    out = {}
    for sd in seeds:
        p = backend_dir / f"drafts_seed{sd}.jsonl"
        out[sd] = {d["id"]: d["hits"] for d in (json.loads(l) for l in p.read_text(encoding="utf-8").splitlines())}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", default="neural7b")
    args = ap.parse_args()
    bdir = ROOT / "results" / args.backend
    R = json.loads((bdir / "results.json").read_text(encoding="utf-8"))
    seeds = R.get("seeds") or [0]
    n_seeds = len(seeds)
    split = json.loads((ROOT / "data" / "benchmark" / "split.json").read_text(encoding="utf-8"))
    test_ids = set(split["test"])
    queries = {q["id"]: q for q in json.loads((ROOT / "data" / "benchmark" / "queries.json").read_text(encoding="utf-8"))}
    hits_by_seed = load_hits(bdir, seeds)
    # the offline cache stores hits without their text: back-fill from the corpus
    if any("text" not in h for hs in hits_by_seed[seeds[0]].values() for h in hs):
        from src.eval.benchmark import chunk_text_lookup
        lookup = chunk_text_lookup(SOURCE_DIRS, (R.get("backend_info") or {}).get("chunker") or {"chunk_size": 80, "overlap": 0})
        for hs in hits_by_seed.values():
            for hits in hs.values():
                for h in hits:
                    h.setdefault("text", lookup.get(h["chunk_id"], ""))
    summaries = corpus_summaries()
    k = max(len(h) for h in hits_by_seed[seeds[0]].values())
    cost = R.get("cost_test") or {}

    out = {"backend": args.backend, "seeds": seeds, "k": k, "d_values": list(D_VALUES), "systems": {}}
    for sysname in SYSTEMS:
        p = bdir / "records" / f"{sysname}.jsonl"
        if not p.exists():
            continue
        recs = load_records(p, test_ids)
        ans = {"n": 0, "draft_correct": 0, "gold_in_top1": 0, "gold_in_topk": 0, "gold_readable": 0,
               "per_language": {l: {"n": 0, "draft_correct": 0, "gold_in_top1": 0, "gold_in_topk": 0, "gold_readable": 0} for l in LANGS}}
        unans = {"n": 0}
        rows = []
        for r in recs:
            if r.get("decision") != "defer":
                continue
            q = queries[r["id"]]
            if not int(r["gold_answerable"]):
                unans["n"] += 1
                continue
            hits = hits_by_seed[r["seed"]][r["id"]][:k]
            gold = q["gold_answers"]
            top1 = bool(hits) and alias_match(hits[0]["text"], gold)
            topk = any(alias_match(h["text"], gold) for h in hits)
            lang = r["language"]
            if lang in SUMMARY_HEADINGS:
                readable = any(alias_match(summaries.get(h["doc_id"], {}).get(lang, ""), gold) for h in hits)
            else:
                readable = topk
            for bucket in (ans, ans["per_language"][lang]):
                bucket["n"] += 1
                bucket["draft_correct"] += int(r.get("draft_correct", 0))
                bucket["gold_in_top1"] += int(top1); bucket["gold_in_topk"] += int(topk); bucket["gold_readable"] += int(readable)
            rows.append({"id": r["id"], "seed": r["seed"], "language": lang, "draft_correct": int(r.get("draft_correct", 0)),
                         "gold_in_top1": int(top1), "gold_in_topk": int(topk), "gold_readable": int(readable), "reason": r.get("reason")})

        def _mean(b):
            return {kk: (v / n_seeds if isinstance(v, (int, float)) else v) for kk, v in b.items()}
        ans_mean = _mean({kk: v for kk, v in ans.items() if kk != "per_language"})
        ans_mean["per_language"] = {l: _mean(v) for l, v in ans["per_language"].items()}
        entry = {"answerable": ans_mean, "unanswerable": _mean(unans), "deferred_queries": rows}

        c = cost.get(sysname)
        if c:
            be = {}
            for d in D_VALUES:
                be[f"{d:g}"] = {}
                for ref in ("B1_vanilla", "B1_empty_filter"):
                    cr = cost.get(ref)
                    if not cr:
                        continue
                    dvis_ref = 0.0  # the references never defer
                    du = cr["unsafe"] - c["unsafe"]
                    if du <= 0:
                        be[f"{d:g}"][ref] = None
                        continue
                    num = (cr["correct"] - c["correct"]) - (1.0 - d) * (ans_mean["gold_readable"] - dvis_ref)
                    be[f"{d:g}"][ref] = max(0.0, num / du)
            entry["break_even"] = be
            entry["cost"] = {"unsafe": c["unsafe"], "correct": c["correct"], "lost_correct": c["lost_correct"]}
        out["systems"][sysname] = entry
        pl = ans_mean["per_language"]
        print(f"{sysname:22s} deferred(ans) {ans_mean['n']:.1f}/seed  correct drafts {ans_mean['draft_correct']:.1f}  "
              f"gold top1 {ans_mean['gold_in_top1']:.1f} topk {ans_mean['gold_in_topk']:.1f} readable {ans_mean['gold_readable']:.1f}  "
              + " ".join(f"{l}:{pl[l]['gold_readable']:.1f}/{pl[l]['n']:.1f}" for l in LANGS)
              + (f"  BE vs filter d=1/0.5/0.25: " + "/".join(
                  "--" if entry['break_even'][f'{d:g}'].get('B1_empty_filter') is None else f"{entry['break_even'][f'{d:g}']['B1_empty_filter']:.1f}"
                  for d in D_VALUES) if c else ""))
    (bdir / "defer_analysis.json").write_text(json.dumps(out, indent=1, ensure_ascii=False), encoding="utf-8")
    print("wrote", bdir / "defer_analysis.json")


if __name__ == "__main__":
    main()
