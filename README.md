# TriSafe-RAG 🛡️
### Trustworthy Multilingual Public-Service AI for Emerging Regions

[![Conference](https://img.shields.io/badge/Target-ICTer%202026-blueviolet)](https://icter.lk)
[![Publisher](https://img.shields.io/badge/Publisher-Springer%20CCIS-orange)](https://springer.com)
[![Track](https://img.shields.io/badge/Track-Responsible%20%26%20Trustworthy%20AI-green)]()
[![License](https://img.shields.io/badge/License-MIT-blue)]()
[![Status](https://img.shields.io/badge/Status-Active%20Development-yellow)]()
[![Python](https://img.shields.io/badge/Python-3.10%2B-blue)]()

---

## 📌 Overview

**TriSafe-RAG** is an end-to-end multilingual Retrieval-Augmented Generation (RAG) framework that jointly enforces:

| Safety Pillar | Description |
|---|---|
| 🔍 **Evidence Grounding** | Answers are linked to verified source passages from trusted public-service documents |
| ⚖️ **Cross-Lingual Fairness** | Performance disparities across Sinhala, Tamil, and English are measured and audited |
| 🚫 **Selective Abstention** | The system defers or refuses responses when retrieved evidence is weak or inconsistent |

> _Trustworthiness in multilingual low-resource AI should be optimized jointly as groundedness + fairness + selective abstention — not as accuracy alone._

---

## 🎯 Research Objectives

1. Build a multilingual public-service knowledge pipeline (Sinhala, Tamil, English)
2. Develop a retrieval-augmented generation architecture with evidence-grounded responses
3. Design a cross-lingual fairness evaluation layer across language groups and code-switched inputs
4. Introduce an uncertainty-aware abstention mechanism for weak or conflicting evidence
5. Create an auditable trust layer with full traceability per query
6. Benchmark TriSafe-RAG against baseline multilingual RAG systems
7. Propose a composite trustworthiness evaluation framework for multilingual public-service RAG

---

## 🗂️ Repository Structure

```
TriSafe-RAG/
├── .github/
│   ├── ISSUE_TEMPLATE/         # Bug report, feature request templates
│   └── workflows/              # CI/CD pipeline (lint, test, experiment)
├── paper/                      # Research writing assets
│   ├── abstract_v1.md
│   ├── objectives.md
│   ├── gap_statement.md
│   └── references.bib
├── data/
│   ├── raw/                    # Original public-service documents (Sinhala, Tamil, English)
│   ├── interim/                # Cleaned, normalized documents
│   ├── processed/              # Chunked, embedded, indexed corpus
│   └── evaluation/             # Gold test sets, fairness slices, adversarial queries
├── src/
│   ├── ingestion/              # Document loaders, cleaners, chunkers
│   ├── retrieval/              # Embeddings, vector store, hybrid retrieval, reranking
│   ├── generation/             # Prompts, grounded answer generation
│   ├── trust/                  # Confidence scoring, abstention, refusal logic
│   ├── eval/                   # Metrics, fairness analysis, ablations
│   └── audit/                  # Query logs, evidence traces, decision records
├── configs/
│   └── experiment_v1.yaml      # Reproducible experiment configuration
├── logs/                       # Run outputs and audit traces
├── app/                        # Streamlit/FastAPI demo interface
├── notebooks/                  # Exploratory Jupyter notebooks
├── tests/                      # Unit and integration tests
├── docs/                       # Architecture diagrams, paper figures
├── requirements.txt
├── .env.example
├── .gitignore
└── README.md
```

---

## 🏗️ System Architecture

```
User Query (Sinhala / Tamil / English / Code-switched)
        │
        ▼
┌─────────────────────┐
│   Query Normalizer  │  ← Language detection, script normalization
└────────┬────────────┘
         │
         ▼
┌─────────────────────┐
│  Multilingual       │  ← Embedding + hybrid retrieval + reranking
│  Retriever          │
└────────┬────────────┘
         │
         ▼
┌─────────────────────┐
│  Evidence Grounding │  ← Context consistency check, support scoring
│  Checker            │
└────────┬────────────┘
         │
         ▼
┌─────────────────────┐
│  Confidence &       │  ← Calibration, uncertainty estimation
│  Abstention Layer   │  ← Refuse / Defer / Escalate if needed
└────────┬────────────┘
         │
         ▼
┌─────────────────────┐
│  Grounded Answer    │  ← Citation-linked response generation
│  Generator          │
└────────┬────────────┘
         │
         ▼
┌─────────────────────┐
│  Fairness Auditor   │  ← Cross-lingual performance comparison
└────────┬────────────┘
         │
         ▼
┌─────────────────────┐
│  Audit Logger       │  ← Query, evidence, confidence, fairness, decision
└─────────────────────┘
```

---

## 📅 5-Week Development Timeline

| Week | Focus | Milestone |
|---|---|---|
| Week 1 | Scope, data, paper skeleton | M1: Scope frozen, corpus plan ready |
| Week 2 | Baseline multilingual RAG | M2: End-to-end retrieval + generation working |
| Week 3 | Trust layer | M3: Confidence scoring + abstention active |
| Week 4 | Fairness + evaluation | M4: Cross-lingual evaluation pipeline ready |
| Week 5 | Experiments + Draft V1 | M5: 70% complete, paper Draft V1 frozen |

---

## 📊 Evaluation Metrics

| Metric | Layer |
|---|---|
| Retrieval Precision@k, MRR | Retrieval |
| Answer faithfulness, factual grounding score | Generation |
| Expected Calibration Error (ECE) | Calibration |
| Abstention Precision/Recall | Trust layer |
| Equalized Odds Gap | Fairness |
| Cross-Lingual Consistency Score | Fairness |
| Human Trust Rating | End-to-end |

---

## 🔬 Baselines

- **B1:** Standard multilingual RAG (no grounding check, no fairness audit, no abstention)
- **B2:** RAG + grounding only
- **B3:** RAG + abstention only
- **Proposed:** TriSafe-RAG (joint: grounding + fairness + abstention)

---

## 📄 Target Publication

- **Conference:** ICTer 2026 — 26th International Conference on Advances in ICT for Emerging Regions
- **Track:** Responsible, Ethical, and Trustworthy AI
- **Publisher:** Springer CCIS
- **Submission Deadline:** 31 May 2026
- **Conference Dates:** 04–05 November 2026

---

## 📚 Key References

1. Investigating Language Preference of Multilingual RAG Systems — arXiv 2502.11175
2. Does RAG Introduce Unfairness in LLMs? — arXiv 2409.19804
3. Language Bias in Multilingual IR — ACL MRL 2024
4. Language Fairness in Multilingual IR — SIGIR 2024
5. RAGTruth: A Hallucination Corpus for Trustworthy RAG — ACL 2024
6. Know Your Limits: A Survey of Abstention in LLMs — arXiv 2407.18418
7. The Art of Abstention — ACL 2021
8. XTRUST: Multilingual Trustworthiness of LLMs — arXiv 2409.15762
9. Bias and Fairness in LLMs: A Survey — Computational Linguistics, 2024
10. A Comprehensive Survey on Trustworthiness of LLMs — arXiv 2502.15871

---

## ⚙️ Setup

```bash
git clone https://github.com/YOUR_USERNAME/TriSafe-RAG.git
cd TriSafe-RAG
pip install -r requirements.txt
cp .env.example .env
# Add your API keys and config paths
```

---

## 🤝 Contributing

This is an active research project. Contributions, suggestions, and peer feedback are welcome.  
Please open an issue or submit a pull request.

---

## 📜 License

MIT License — See [LICENSE](LICENSE) for details.

---

## ✉️ Contact

For research collaboration or paper correspondence:  
📧 publicationchair.icter@ucsc.cmb.ac.lk (conference)  
🌐 [www.icter.lk](https://www.icter.lk)