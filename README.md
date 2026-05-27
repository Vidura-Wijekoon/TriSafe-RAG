# TriSafe-RAG

**A trustworthy multilingual retrieval-augmented generation framework that jointly enforces evidence grounding, cross-lingual fairness, and selective abstention.**

[![Python](https://img.shields.io/badge/python-3.10%2B-blue)]()
[![License: MIT](https://img.shields.io/badge/license-MIT-green)]()
[![Tests](https://img.shields.io/badge/tests-passing-brightgreen)]()
[![Status](https://img.shields.io/badge/status-research--preview-orange)]()

---

## Why TriSafe-RAG

Most production RAG systems treat factuality, fairness, and abstention as separate after-thoughts. That is fine for English-only search assistants. It breaks for public-service queries in low-resource languages, where a confident but unsupported answer in Sinhala or Tamil can cause real harm.

TriSafe-RAG is a single end-to-end pipeline that:

- grounds every generated sentence in retrieved evidence, with per-sentence support scores,
- estimates a calibrated confidence from retrieval, grounding, and language-match signals,
- decides whether to **answer**, **defer**, or **refuse** based on an explicit policy,
- audits each decision so cross-lingual disparities can be measured and reported.

The codebase is intentionally backend-agnostic: a deterministic hash embedder and an `EchoLLM` ship by default so the pipeline runs offline in CI; swap in `sentence-transformers` and OpenAI (or any other LLM) when you have GPU and API access.

---

## Architecture

```
Query
  │
  ▼  src/ingestion/      normalize + language detect
  ▼  src/retrieval/      dense (FAISS-like) + BM25 + RRF fusion
  ▼  src/generation/     grounded prompt + citation rendering
  ▼  src/trust/grounding lexical or NLI per-sentence support
  ▼  src/trust/calibration  confidence = f(retrieval, grounding, language)
  ▼  src/trust/abstention  answer | defer | refuse
  ▼  src/trust/fairness  per-language metric audit
  └─▶ src/audit/         append-only JSONL log per query
```

Every stage is injected through `TriRAGPipeline`, so ablations live in `src/eval/runner.py` rather than in the pipeline itself.

---

## Repository layout

```
TriSafe-RAG/
├── src/
│   ├── ingestion/      # loaders, NFC normalize, script-based lang detect, sentence-aware chunker
│   ├── retrieval/      # HashEmbedding | SentenceTransformer, in-memory store, BM25, hybrid RRF
│   ├── generation/     # prompt templates, EchoLLM / OpenAI wrapper, grounded generator
│   ├── trust/          # grounding (lexical & NLI), calibration + ECE, abstention policy, fairness auditor
│   ├── eval/           # IR + abstention metrics, eval runner with ablation hooks
│   ├── audit/          # AuditRecord + JSONL logger
│   ├── pipeline.py     # end-to-end orchestrator
│   ├── build.py        # YAML config → pipeline factory
│   └── cli.py          # `trirag query` and `trirag eval`
├── app/
│   ├── api.py          # FastAPI service
│   └── streamlit_app.py
├── configs/
│   ├── experiment_v1.yaml
│   └── ablations.yaml
├── data/
│   ├── raw/            # source documents (seed files included)
│   ├── interim/
│   ├── processed/
│   └── evaluation/     # seed_eval.json gold set
├── tests/              # unit + integration tests
├── paper/              # Springer LNCS LaTeX source for the accompanying paper
└── logs/               # per-query audit and eval outputs
```

---

## Quickstart

```bash
git clone https://github.com/Vidura-Wijekoon/TriSafe-RAG
cd TriSafe-RAG
pip install -e ".[loaders,app]"      # core + loaders + demo app
# optional: pip install -e ".[neural]"  # for sentence-transformers + transformers

# offline smoke test (no API keys, no models)
python -m src.cli query --data data/raw "What is the passport renewal fee?"
```

Expected output is a JSON record with `decision`, `confidence`, `grounding_support`, the answer, and ranked citations.

### Run the demo UI

```bash
streamlit run app/streamlit_app.py
# or the FastAPI service:
uvicorn app.api:app --reload
```

### Run the evaluation harness

```bash
python -m src.cli eval \
  --data data/raw \
  --dataset data/evaluation/seed_eval.json \
  --id baseline_v1
```

This writes `logs/eval/baseline_v1_summary.json` and a `logs/eval/baseline_v1_per_query.jsonl` trace.

---

## Configuration

`configs/experiment_v1.yaml` covers every knob:

```yaml
embedder:   { name: "hash", dim: 256 }     # or {name: "st", model_name: "intfloat/multilingual-e5-base"}
chunker:    { chunk_size: 320, overlap: 64 }
retrieval:  { alpha: 0.5, top_k: 5 }       # alpha = dense vs sparse weight in RRF
llm:        { name: "echo" }                # or {name: "openai", model: "gpt-4o-mini"}
grounding:  { backend: "lexical", support_threshold: 0.25 }   # or {backend: "nli"}
confidence: { w_retrieval: 0.20, w_margin: 0.20, w_grounding: 0.40,
              w_citation: 0.10, w_language: 0.10 }
abstention: { answer_threshold: 0.55, refuse_threshold: 0.30,
              min_support: 0.20, min_citation_coverage: 0.40 }
```

`configs/ablations.yaml` lists the four canonical configurations used in the paper: `B1_vanilla`, `B2_grounding_only`, `B3_abstention_only`, `TriSafe_full`.

---

## Evaluation metrics

| Layer       | Metric                                                            |
| ----------- | ----------------------------------------------------------------- |
| Retrieval   | Precision@k, Mean Reciprocal Rank                                 |
| Generation  | Per-sentence grounding support, citation coverage                 |
| Calibration | Expected Calibration Error (ECE), Brier score                     |
| Abstention  | Abstention precision / recall / F1, answer rate                   |
| Fairness    | Per-language accuracy, max gap, Equalized Odds gap, CL Consistency |

All implementations live in `src/eval/metrics.py` and `src/trust/`.

---

## Programmatic use

```python
from src.build import build_pipeline, load_config

cfg = load_config("configs/experiment_v1.yaml")
pipeline = build_pipeline(cfg, source_dir="data/raw")

resp = pipeline.query("Where do I apply to renew my NIC?")
print(resp.decision.decision)      # 'answer' | 'defer' | 'refuse'
print(resp.confidence)             # 0..1
print(resp.grounding.overall_support)
print(resp.answer.text)
for h in resp.hits:
    print(h.rank, h.chunk.chunk_id, round(h.score, 3), h.chunk.text[:80])
```

---

## Tests

```bash
PYTHONPATH=. python -m pytest tests/
```

The default suite uses the offline backends and runs in seconds.

---

## Project status

This is a research preview. The trust layer (grounding, calibration, abstention, fairness) is the focus; the retrieval and generation backends are deliberately interchangeable.

The accompanying paper, *TriSafe-RAG: Joint Grounding, Fairness, and Abstention for Multilingual Public-Service RAG*, ships under `paper/lncs/`.

---

## Citation

If you use this work, please cite:

```bibtex
@inproceedings{wijekoon2026trisafe,
  title   = {TriSafe-RAG: Joint Grounding, Fairness, and Abstention for
             Trustworthy Multilingual Public-Service RAG},
  author  = {Wijekoon, Vidura},
  year    = {2026},
  booktitle = {Preprint}
}
```

---

## License

Released under the [MIT License](LICENSE).
