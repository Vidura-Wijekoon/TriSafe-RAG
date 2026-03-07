# Research Objectives — TriSafe-RAG

> **Date:** 2026-03-07  
> **Status:** Confirmed

---

## Objective Summary

| Type | Objective |
|---|---|
| **Main Objective** | Design and evaluate a trustworthy multilingual RAG system for public-service query answering that is factual, fair across language groups, and safe under uncertainty. |
| **Objective 1** | Build a multilingual document pipeline for Sinhala, Tamil, and English public-service knowledge sources. |
| **Objective 2** | Develop a retrieval-and-generation architecture with evidence grounding and citation-aware response generation. |
| **Objective 3** | Introduce a fairness evaluation layer to test whether answer quality changes across languages, dialects, or code-switched inputs. |
| **Objective 4** | Add an uncertainty-aware abstention mechanism so the model refuses or defers low-confidence answers. |
| **Objective 5** | Create an explanation and audit module that logs retrieved evidence, confidence, fairness indicators, and refusal reasons. |
| **Objective 6** | Benchmark the full system against a baseline multilingual RAG pipeline and a non-abstaining version. |

---

## Objective-to-Component Mapping

| Objective | Primary `src/` Module |
|---|---|
| O1 — Multilingual document pipeline | `src/ingestion/` |
| O2 — Evidence-grounded retrieval & generation | `src/retrieval/`, `src/generation/` |
| O3 — Cross-lingual fairness evaluation | `src/trust/` (fairness sub-module) |
| O4 — Uncertainty-aware abstention | `src/trust/` (abstention sub-module) |
| O5 — Explanation & audit logging | `src/eval/`, `logs/` |
| O6 — Benchmarking | `src/eval/` |
