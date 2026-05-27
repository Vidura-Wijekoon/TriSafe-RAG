"""End-to-end evaluation runner used for ablations and the main results table."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List

from src.eval.metrics import abstention_precision_recall
from src.pipeline import TriRAGPipeline, TriRAGResponse
from src.trust.fairness import FairnessAuditor

logger = logging.getLogger(__name__)


@dataclass
class EvalConfig:
    output_dir: str = "logs/eval"
    config_id: str = "baseline_v1"
    keep_per_query: bool = True


@dataclass
class EvalRunner:
    pipeline: TriRAGPipeline
    config: EvalConfig = field(default_factory=EvalConfig)
    auditor: FairnessAuditor = field(default_factory=FairnessAuditor)

    def run(self, dataset: List[dict]) -> Dict:
        """Each dataset item must contain: id, query, language, gold_answer,
        gold_answerable (0/1). Returns a summary dict written to JSON."""
        out_dir = Path(self.config.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        per_query: List[dict] = []
        abstained: List[int] = []
        unanswerable: List[int] = []
        records_for_fairness: List[dict] = []

        for item in dataset:
            resp: TriRAGResponse = self.pipeline.query(
                item["query"], language=item.get("language")
            )
            gold_answerable = int(item.get("gold_answerable", 1))
            ab = 1 if resp.decision.decision != "answer" else 0
            correct = int(self._exact_or_substring_match(resp.answer.text, item.get("gold_answer", "")))
            grounding = resp.grounding.overall_support

            row = {
                "id": item.get("id"),
                "language": item.get("language", "unknown"),
                "predicted_text": resp.answer.text,
                "gold_answer": item.get("gold_answer", ""),
                "abstained": ab,
                "gold_answerable": gold_answerable,
                "correct": correct,
                "confidence": resp.confidence,
                "grounding_support": grounding,
                "decision": resp.decision.decision,
                "reason": resp.decision.reason,
            }
            per_query.append(row)
            abstained.append(ab)
            unanswerable.append(1 - gold_answerable)
            records_for_fairness.append(row)

        p, r, f1 = abstention_precision_recall(abstained, unanswerable)
        fairness = self.auditor.audit(records_for_fairness)

        n = max(len(per_query), 1)
        summary = {
            "config_id": self.config.config_id,
            "n_queries": len(per_query),
            "accuracy": sum(x["correct"] for x in per_query) / n,
            "answer_rate": 1 - sum(abstained) / n,
            "abstention_precision": p,
            "abstention_recall": r,
            "abstention_f1": f1,
            "mean_grounding_support": sum(x["grounding_support"] for x in per_query) / n,
            "mean_confidence": sum(x["confidence"] for x in per_query) / n,
            "fairness": fairness.__dict__,
        }

        (out_dir / f"{self.config.config_id}_summary.json").write_text(
            json.dumps(summary, indent=2, ensure_ascii=False)
        )
        if self.config.keep_per_query:
            with (out_dir / f"{self.config.config_id}_per_query.jsonl").open("w", encoding="utf-8") as f:
                for row in per_query:
                    f.write(json.dumps(row, ensure_ascii=False) + "\n")
        return summary

    @staticmethod
    def _exact_or_substring_match(pred: str, gold: str) -> bool:
        if not gold:
            return False
        p = pred.strip().lower()
        g = gold.strip().lower()
        return g in p or p in g
