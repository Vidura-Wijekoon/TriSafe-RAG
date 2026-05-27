"""Command-line entry points: build index, run query, run evaluation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.build import build_pipeline, load_config


def cmd_query(args):
    cfg = load_config(args.config)
    pipe = build_pipeline(cfg, source_dir=args.data)
    resp = pipe.query(args.query, language=args.lang)
    out = {
        "query": resp.query,
        "language": resp.language,
        "decision": resp.decision.decision,
        "reason": resp.decision.reason,
        "confidence": resp.confidence,
        "grounding_support": resp.grounding.overall_support,
        "answer": resp.answer.text,
        "citations": [
            {"rank": h.rank + 1, "id": h.chunk.chunk_id, "score": h.score}
            for h in resp.hits
        ],
    }
    print(json.dumps(out, indent=2, ensure_ascii=False))


def cmd_eval(args):
    from src.eval.runner import EvalConfig, EvalRunner

    cfg = load_config(args.config)
    pipe = build_pipeline(cfg, source_dir=args.data)
    dataset = json.loads(Path(args.dataset).read_text(encoding="utf-8"))
    runner = EvalRunner(pipeline=pipe, config=EvalConfig(config_id=args.id))
    summary = runner.run(dataset)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


def main():
    p = argparse.ArgumentParser(prog="trirag")
    sub = p.add_subparsers(dest="command", required=True)

    pq = sub.add_parser("query", help="Run a single query through the pipeline")
    pq.add_argument("--config", default="configs/experiment_v1.yaml")
    pq.add_argument("--data", default=None)
    pq.add_argument("--lang", default=None)
    pq.add_argument("query")
    pq.set_defaults(func=cmd_query)

    pe = sub.add_parser("eval", help="Run a JSON-list evaluation set")
    pe.add_argument("--config", default="configs/experiment_v1.yaml")
    pe.add_argument("--data", default="data/raw")
    pe.add_argument("--dataset", required=True)
    pe.add_argument("--id", default="run01")
    pe.set_defaults(func=cmd_eval)

    args = p.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
