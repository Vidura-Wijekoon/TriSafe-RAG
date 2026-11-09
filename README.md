# TriSafe-RAG

**Grounding, calibrated abstention, and a per-language fairness audit on one audit record, for Sinhala / Tamil / English public-service retrieval-augmented generation.**

[![Python](https://img.shields.io/badge/python-3.10%2B-blue)]()
[![License: MIT](https://img.shields.io/badge/license-MIT-green)]()
[![Tests](https://img.shields.io/badge/tests-32%20passing-brightgreen)]()
[![Status](https://img.shields.io/badge/status-research--preview-orange)]()

---

## Why TriSafe-RAG

Most production RAG systems treat factuality, fairness, and abstention as separate after-thoughts. That is fine for English-only search assistants. It breaks for public-service queries in low-resource languages, where a confident but unsupported answer in Sinhala or Tamil can cause real harm.

TriSafe-RAG is a single end-to-end pipeline that:

- grounds every generated sentence in retrieved evidence, with per-sentence support scores (lexical or cross-lingual NLI),
- combines retrieval, grounding, citation and language-match signals into one confidence score, and measures how well that score is calibrated,
- decides whether to **answer**, **defer**, or **refuse** based on an explicit policy (global thresholds by default, optional per-language thresholds),
- writes one audit record per query that a fairness auditor reads to report per-language accuracy, answer rate, TPR/FPR and an Equalized Odds gap.

The codebase is backend-agnostic. A deterministic hash embedder and an extractive `EchoLLM` ship by default so the pipeline runs offline in CI. The paper's neural backends use `multilingual-e5-base` retrieval and `mDeBERTa-v3` NLI grounding on the CPU with one of two open-weight generators served locally by [Ollama](https://ollama.com) (llama.cpp): `neural` = Qwen2.5-1.5B-Instruct (fp16) and `neural7b` = Qwen2.5-7B-Instruct (Q4_K_M), both on a single 8 GB laptop GPU, three generation seeds each. `neural_hf_fp32` is the earlier CPU-only run of the 1.5B model through HuggingFace transformers, kept under `results/neural_hf_fp32` for reference. An OpenAI backend is included but was not used for any reported number.

---

## Architecture

```
Query
  │
  ▼  src/ingestion/      NFC normalisation + script-based language detection (si / ta / en / mixed)
  ▼  src/retrieval/      dense (cosine) + BM25 + RRF fusion; dense scores carried through fusion
  ▼  src/generation/     grounded prompt, citation rendering; Echo | local HF model | OpenAI
  ▼  src/trust/grounding    per-sentence support: Jaccard (offline) or mDeBERTa NLI (cross-lingual)
  ▼  src/trust/calibration  confidence = weighted sum of 5 normalised signals; ECE/Brier/AUROC;
  │                         Platt / isotonic / temperature post-hoc calibrators (fit on dev split)
  ▼  src/trust/abstention   answer | defer | refuse; global or per-language thresholds
  ▼  src/trust/fairness     per-language accuracy, answer rate, TPR/FPR, gaps, Equalized Odds
  └─▶ src/audit/            append-only JSONL log per query (signals included)
```

`TriRAGPipeline` is split into `draft()` (retrieval + generation) and `finalize()` (trust layer), so the evaluation can cache drafts once and replay every configuration over the same drafts.

---

## Repository layout

```
TriSafe-RAG/
├── src/                 # pipeline, trust layer, evaluation (see Architecture)
│   └── eval/            # metrics (definitions in docstring), paired bootstrap, dev/test split,
│                        # benchmark runner with draft cache and external baselines
├── data/benchmark/      # SL-PubServ-QA: items.json (source of truth), queries.json, split.json
├── casestudies/         # three domain slices over real gov.lk content + offline drivers
├── scripts/
│   ├── build_benchmark.py   # items.json -> queries.json, split.json, per-case slices
│   ├── run_benchmark.py     # run a backend end to end -> results/<backend>/results.json
│   ├── run_all_seeds.sh     # seeds 0,1,2 per neural backend; progress_ui.py shows live progress
│   ├── native_review.py     # native-speaker review sheet -> provenance paragraph / corrections
│   ├── make_tables.py       # results.json -> paper macros, tables (LaTeX) and figures
│   ├── make_architecture.py # evaluation-architecture figure from repository counts
│   └── check_paper.py, make_overleaf_zip.py   # consistency check; XeLaTeX zip for Overleaf
├── results/             # committed benchmark outputs (drafts, per-system records, results.json)
├── paper/trust2027/     # ACM sigconf source; tables are \input from generated/
├── configs/             # YAML configs (experiment_v1, ablations)
├── app/                 # FastAPI service + Streamlit demo
└── tests/               # 32 unit and integration tests (offline)
```

---

## Quickstart

```bash
git clone https://github.com/Vidura-Wijekoon/TriSafe-RAG
cd TriSafe-RAG
pip install -e ".[loaders,app]"        # core + loaders + demo app
# optional: pip install -e ".[neural,eval]"   # sentence-transformers, transformers, torch, matplotlib

# offline smoke test (no API keys, no models)
python -m src.cli query --data casestudies/civil_documents/data/raw "What is the passport renewal fee?"

python -m pytest -q
```

### Reproduce the paper

```bash
python scripts/build_benchmark.py                       # 62 items -> 208 queries, dev/test split
python scripts/run_benchmark.py --backend offline       # seconds; hash embedder + EchoLLM
ollama pull qwen2.5:1.5b-instruct-fp16 && ollama pull qwen2.5:7b-instruct   # generators (3.1 GB + 4.7 GB)
bash scripts/run_all_seeds.sh neural7b neural           # seeds 0,1,2 per backend; ~1.5 h/seed (7B) and ~1 h/seed (1.5B) on an 8 GB GPU
python scripts/progress_ui.py --backend neural7b        # live progress at http://localhost:8765 while a run is going
python scripts/make_tables.py                           # paper/trust2027/generated/*.tex + figures (all backends found)
python scripts/make_architecture.py                     # paper/trust2027/figures/eval_harness.pdf
python scripts/make_overleaf_zip.py                     # paper/TriSafe-RAG_TRUST2027_overleaf.zip (compile with XeLaTeX)
PYTHONPATH=. python casestudies/civil_documents/run.py  # offline case-study drivers
```

`run_benchmark.py` caches drafts per seed under `results/<backend>/drafts_seed<k>.jsonl` (plus stage caches under `stages_seed<k>/`); re-running only replays the trust layer. Each query is drafted once at temperature 0.2 and re-sampled three times at 0.7 for the E7 self-consistency gate. Every replay also writes the gate-isolation systems, the `B1 + empty-reply filter`, the cost analysis (unsafe / correct deliveries, break-even cost vs. B1), a group 5-fold cross-fitting of the tuned rules, and the show-the-passage reference point. The generator's `INSUFFICIENT_EVIDENCE` abstention is detected anywhere in the reply; cached drafts from the earlier start-of-reply detector are re-detected on load.

### Adjudication of delivered answers

```bash
python scripts/adjudicate.py make --backend neural7b --seed 0   # data/benchmark/adjudication_B1_delivered.csv
python scripts/adjudicate.py stats                              # -> paper/trust2027/generated/adjudication.tex
```

Every answer B1 delivered on the test split is listed with its alias-match label; the `adjudicated` column (correct | partial | wrong) feeds the agreement numbers in the paper. The shipped sheet covers the 7B seed-0 run (116 answers; 30 of them are citation-only replies such as `[1] #1`).

### Native-speaker review of the Sinhala and Tamil queries

```bash
python scripts/native_review.py make    # data/benchmark/native_review_sheet.csv (146 si/ta/mixed queries)
#   fill in: verdict = ok | minor | major | wrong, corrected_query, gold_verdict (ok|add|wrong), corrected_gold, reviewer
python scripts/native_review.py stats   # verdict counts -> paper/trust2027/generated/provenance.tex (Cohen's kappa with --second)
python scripts/native_review.py apply   # write corrections into items.json, rebuild queries/split, purge cached drafts of changed queries
python scripts/native_review.py make-corpus   # data/benchmark/native_review_corpus.csv: the 22 Sinhala/Tamil corpus summaries (not yet reviewed)
bash scripts/run_all_seeds.sh neural7b neural   # regenerates only the purged drafts, then re-evaluates everything
```

---

## Benchmark: SL-PubServ-QA

`data/benchmark/items.json` holds 62 parallel items over 12 documents quoted from seven Sri Lankan government agencies (Immigration, DRP, DMT, Ministry of Health units, DMC, NBRO, Meteorology). Every item has an English question, a Sinhala and a Tamil translation, and for 22 items a code-switched (romanised Sinhala–English) version: 208 queries, 52 answerable and 10 unanswerable per single-language group. Gold answers are alias lists; numeric aliases match as whole numbers, textual aliases as normalised substrings. Twelve item groups (39 queries) form the development split used for every threshold, weight and calibrator choice; the remaining 169 queries are the test split. The Sinhala, Tamil and code-switched queries were machine-drafted and then reviewed query by query by an author who is a native speaker of both languages (`data/benchmark/native_review_sheet.csv`; all 146 accepted unchanged on 2026-11-09). A second, independent reviewer can fill a copy of the sheet; `native_review.py stats --second` then reports Cohen's kappa.

---

## Status (2026-11-09)

| Component | State |
|---|---|
| Benchmark (62 items, 208 queries, dev/test split) | final; native-speaker review complete |
| Offline backend (`results/offline`) | final |
| `neural7b` (Qwen2.5-7B Q4_K_M) | three seeds done and evaluated with every system (2,000-resample bootstrap): **headline backend of the paper** |
| `neural` (Qwen2.5-1.5B fp16) | three seeds done and evaluated (`results/neural`); fills the scale section and appendix |
| `neural_hf_fp32` (1.5B, CPU, 1 seed) | archived reference run |
| Paper (`paper/trust2027`) | complete: three seeds per generator, 7B headline, 1.5B scale comparison; `check_paper.py` reports 0 problems |
| Adjudication (`data/benchmark/adjudication_B1_delivered.csv`) | 116 delivered 7B seed-0 answers labelled |
| Corpus-summary review (`data/benchmark/native_review_corpus.csv`) | sheet generated, not yet reviewed |

The evaluation architecture (benchmark artefacts, backends, per-seed draft cache, replay of 16 systems, statistics, paper generation) is drawn in `paper/trust2027/figures/eval_harness.pdf`.

---

## Configuration

`configs/experiment_v1.yaml` covers every knob:

```yaml
embedder:   { name: "hash", dim: 256 }     # or {name: "st", model_name: "intfloat/multilingual-e5-base"}
chunker:    { chunk_size: 320, overlap: 64 }
retrieval:  { alpha: 0.5, top_k: 5 }       # alpha = dense vs sparse weight in RRF
llm:        { name: "echo" }               # echo | hf (model: Qwen/Qwen2.5-1.5B-Instruct) | openai
grounding:  { backend: "lexical", support_threshold: 0.25 }   # lexical | nli
confidence: { w_retrieval: 0.2, w_margin: 0.2, w_grounding: 0.4, w_citation: 0.1, w_language: 0.1,
              retrieval_signal: "cosine", ret_lo: 0.70, ret_hi: 0.90, margin_scale: 10 }
abstention: { answer_threshold: 0.55, refuse_threshold: 0.30, min_support: 0.20, min_citation_coverage: 0.40,
              per_language: {} }           # e.g. {ta: {answer_threshold: 0.50}}
```

`retrieval_signal: rrf` reproduces the legacy behaviour, in which the two retrieval signals are nearly constant (RRF scores with k=60 lie in (0, 1/61]); `cosine` uses the dense similarity of the top hit rescaled from `[ret_lo, ret_hi]`.

---

## Metric definitions

Defined once in `src/eval/metrics.py` and used everywhere, including the paper: **answered** = decision is `answer` (defer and refuse are abstentions); **accuracy** = on gold-answerable queries only, abstention counts as an error; **selective accuracy** = accuracy among answered answerable queries; **decision accuracy** = the only metric that credits a correct refusal; **abstention P/R/F1** with gold-unanswerable as the positive class (a system that never abstains has F1 = 0); **ECE / AUROC** of the confidence score against draft correctness over all queries, so they do not depend on the policy. Confidence intervals are paired bootstraps over queries (2,000 resamples); significance uses paired sign-flip permutation tests.

---

## Citation

```bibtex
@inproceedings{trisaferag2027,
  title     = {TriSafe-RAG: Joint Grounding, Fairness Auditing, and Abstention for Trustworthy Multilingual Public-Service Retrieval-Augmented Generation},
  author    = {Wijekoon, Vidura},
  booktitle = {ACM International Conference on Trustworthy and Responsible AI and Computing Systems (TRUST '27)},
  year      = {2027}
}
```

## License

MIT.
