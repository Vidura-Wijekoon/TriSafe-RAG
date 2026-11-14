"""Probe whether the citation-only replies of the 7B generator are a pipeline artefact.

The adjudication (scripts/adjudicate.py) found that many answers B1 delivered in Sinhala
and Tamil consist of nothing but citation markers ("[1] #1").  This script takes every
such reply of one cached seed on the test split and regenerates it with the same model,
passages and sampling settings under four conditions:

    base      the benchmark prompt, 128 new tokens (as in the runs)
    tokens    the benchmark prompt, 512 new tokens      (rules out the token cap)
    instruct  an extra instruction forbidding bare citations, 128 tokens
    both      extra instruction and 512 tokens

and reports, per condition, how many replies are still empty, how many are abstentions,
how many carry content, and how many of those match a gold alias.  The evidence blocks are
rebuilt from the cached hits (passage text and rank); the source label is the document id
rather than the file path used in the run, which is the only difference to the run prompt.

    python scripts/probe_empty_replies.py --backend neural7b --seed 0 [--model qwen2.5:7b-instruct]

Writes results/<backend>/probe_empty_replies.json and prints a summary.  Requires Ollama.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.eval.metrics import alias_match  # noqa: E402
from src.generation.llm import OllamaLLM  # noqa: E402
from src.generation.prompts import SYSTEM_PROMPT, is_empty_reply, is_model_abstention  # noqa: E402

EXTRA = (" State the answer in words in the user's language before the citation; "
         "a reply that consists only of citation markers is not an answer.")


def build_prompt(query: str, hits: list[dict]) -> str:
    blocks = [f"[{h['rank'] + 1}] (source: {h['doc_id']})\n{h['text']}" for h in hits]
    evidence = "\n\n".join(blocks) if blocks else "(no evidence retrieved)"
    return f"Evidence:\n{evidence}\n\nQuestion: {query}\n\nAnswer using the evidence above. Cite passages by number."


def classify(text: str, gold: list[str]) -> str:
    if is_model_abstention(text):
        return "abstain"
    if is_empty_reply(text):
        return "empty"
    return "correct" if gold and alias_match(text, gold) else "content_wrong"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", default="neural7b")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--model", default=None)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()
    out_dir = ROOT / "results" / args.backend
    results = json.loads((out_dir / "results.json").read_text(encoding="utf-8"))
    llm_info = results["backend_info"]["llm"]
    model = args.model or (llm_info["model"] if isinstance(llm_info, dict) else llm_info.split(":", 1)[1])
    split = json.loads((ROOT / "data" / "benchmark" / "split.json").read_text(encoding="utf-8"))
    test = set(split["test"])
    drafts = [json.loads(l) for l in (out_dir / f"drafts_seed{args.seed}.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    targets = [d for d in drafts if d["id"] in test and not is_model_abstention(d["draft_text"]) and is_empty_reply(d["draft_text"])]
    if args.limit:
        targets = targets[: args.limit]
    print(f"{len(targets)} citation-only replies on the test split of seed {args.seed} ({model})")
    conds = {"base": (SYSTEM_PROMPT, 128), "tokens": (SYSTEM_PROMPT, 512),
             "instruct": (SYSTEM_PROMPT + EXTRA, 128), "both": (SYSTEM_PROMPT + EXTRA, 512)}
    llm = OllamaLLM(model=model, temperature=0.2, top_p=0.95, seed=args.seed)
    rows = []
    for d in targets:
        row = {"id": d["id"], "language": d["language"], "gold_answerable": d["gold_answerable"],
               "original": d["draft_text"], "conditions": {}}
        prompt = build_prompt(d["query"], d.get("hits") or [])
        for name, (system, cap) in conds.items():
            text = None
            for attempt in range(8):  # Ollama occasionally returns HTTP 500 under load; wait and retry
                try:
                    text = (llm.generate(system, prompt, max_new_tokens=cap).text or "").strip()
                    break
                except RuntimeError as exc:
                    print(f"  retry {attempt + 1} after error: {exc}", flush=True)
                    time.sleep(15 * (attempt + 1))
            if text is None:
                raise RuntimeError("Ollama unavailable")
            row["conditions"][name] = {"text": text, "class": classify(text, d.get("gold_answers") or [])}
        rows.append(row)
        print(d["id"], {k: v["class"] for k, v in row["conditions"].items()}, flush=True)
    summary = {}
    for name in conds:
        cls = [r["conditions"][name]["class"] for r in rows]
        summary[name] = {c: cls.count(c) for c in ("empty", "abstain", "correct", "content_wrong")}
        summary[name]["n"] = len(cls)
        summary[name]["max_new_tokens"] = conds[name][1]
        summary[name]["extra_instruction"] = name in ("instruct", "both")
    by_lang = {}
    for r in rows:
        by_lang.setdefault(r["language"], 0)
        by_lang[r["language"]] += 1
    out = {"backend": args.backend, "seed": args.seed, "model": model, "n": len(rows), "by_language": by_lang,
           "extra_instruction": EXTRA.strip(), "summary": summary, "rows": rows}
    (out_dir / "probe_empty_replies.json").write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(summary, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
