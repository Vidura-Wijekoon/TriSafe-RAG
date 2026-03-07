# Abstract — TriSafe-RAG (v1)

> **Version:** 1.0  
> **Date:** 2026-03-07  
> **Status:** Draft

---

Retrieval-augmented generation has advanced context-aware question answering, yet multilingual deployments continue to exhibit language preference in retrieval and generation, resulting in measurable disparities across low-resource language communities. Existing systems address language fairness, hallucination control, and selective abstention as separate concerns, leaving a critical gap in unified trustworthy multilingual AI for public-service applications in emerging regions. This paper presents TriSafe-RAG, an integrated trustworthy multilingual retrieval-augmented generation framework designed for citizen-facing query answering in Sinhala, Tamil, and English. The framework combines three coordinated safety mechanisms: evidence-grounded multilingual retrieval that links each response to verified source passages, cross-lingual fairness auditing that measures and reports disparities across language groups and code-switched inputs, and uncertainty-aware selective abstention that defers or refuses responses when retrieved evidence is weak or inconsistent. The system is evaluated on a curated public-service knowledge corpus using metrics covering factual grounding, cross-lingual consistency, calibration quality, and abstention effectiveness. Experimental results demonstrate that jointly optimising these three dimensions outperforms baseline multilingual RAG systems that apply each property independently. The study contributes an open reproducible framework, a multilingual evaluation benchmark, and design guidelines for responsible public-service AI deployment in low-resource multilingual settings.
