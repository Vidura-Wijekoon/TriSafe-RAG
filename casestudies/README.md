# Case studies

Three case studies that each stress a different part of the TriSafe-RAG
trust layer. Each one is a **domain slice of the SL-PubServ-QA benchmark**
(`data/benchmark/`): the same parallel Sinhala / Tamil / English /
code-switched queries, the same gold aliases, the same dev/test split, and the
same metric definitions as the main benchmark, restricted to one domain and
its corpus. All corpus content is **real, publicly-published content from Sri
Lankan government sources**, with English text quoted verbatim and Sinhala /
Tamil summaries that preserve the same numeric values, drug names and form
names. Source URLs are listed in each case study's `SOURCES.md`.

| # | Folder                 | Domain                               | Stresses                        | Test queries (unanswerable) | Primary sources |
| - | ---------------------- | ------------------------------------ | ------------------------------- | --------------------------- | --------------- |
| 1 | `civil_documents/`     | Passport, NIC, driving licence, eRL  | Grounding + citation            | 75 total, 63 test (10)      | immigration.gov.lk, drp.gov.lk, dmt.gov.lk, web.erl2.gov.lk |
| 2 | `public_health/`       | Dengue, immunisation, antenatal, flu | Misinformation / false premises | 73 total, 56 test (7)       | health.gov.lk, epid.gov.lk, fhb.health.gov.lk |
| 3 | `disaster_response/`   | Flood, landslide, cyclone            | Abstention under contradiction  | 60 total, 50 test (17)      | dmc.gov.lk, nbro.gov.lk, meteo.gov.lk |

Each folder ships:

```
<case>/
├── data/raw/          # source corpus: English verbatim from gov.lk,
│                      # Sinhala/Tamil summaries with the same numbers
├── eval/queries.json  # domain slice of data/benchmark/queries.json (generated)
├── run.py             # offline-backend driver
├── results/           # summary.json + <system>_per_query.jsonl (committed)
├── README.md
└── SOURCES.md         # all source URLs and access dates
```

`eval/queries.json` is generated from `data/benchmark/items.json` by
`scripts/build_benchmark.py`; edit the items file, not the slices.

## Running

```bash
PYTHONPATH=. python casestudies/civil_documents/run.py
PYTHONPATH=. python casestudies/public_health/run.py
PYTHONPATH=. python casestudies/disaster_response/run.py
```

Each driver runs the **offline reproduction backend** (deterministic hash
embedder, extractive `EchoLLM`, lexical grounding) on the slice, replays three
systems over the cached drafts (`B1_vanilla`, `TriSafe_default`,
`TriSafe_devtuned`, the last with thresholds tuned on the slice's development
queries), and writes `results/summary.json` with overall, per-language, and
gap metrics for both the full slice and its test portion.

The **neural-backend** numbers for the same slices (multilingual-e5-base,
Qwen2.5-1.5B-Instruct, mDeBERTa-v3 NLI grounding) are not produced by these
drivers; they are the `test_by_domain` entries of
`results/neural/results.json`, written by

```bash
python scripts/run_benchmark.py --backend neural --seeds 0
```

and both appear side by side in the paper's case-study table
(`scripts/make_tables.py`).

## What the offline backend can and cannot show

The hash embedder cannot represent Sinhala or Tamil script meaningfully and
`EchoLLM` cannot fabricate text, so the offline numbers say little about
hallucination or cross-lingual quality. They do show, in seconds and without
any model download, that the trust layer behaves as specified: it lowers the
answer rate, refuses unanswerable queries at a higher rate than the baseline,
and never answers a query the generator itself abstained on. Read the
neural-backend table for the substantive comparison.
