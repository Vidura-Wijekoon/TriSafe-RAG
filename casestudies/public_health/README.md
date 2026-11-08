# Case Study 2 — Public Health Advisory

**Domain.** Public-health guidance from the Sri Lankan Ministry of Health:
dengue patient instructions, the national child immunization schedule,
antenatal care, and seasonal-flu guidance.

**Real data sources used in this case study** (see `SOURCES.md` for the full
list). All clinical guidance is quoted from official ministry publications.

| Document         | Source                                                |
| ---------------- | ----------------------------------------------------- |
| `dengue.md`      | [NDCU — Instructions for patients](https://www.dengue.health.gov.lk/web/index.php/en/information/instructions-for-patients) |
| `dengue.md`      | [Epidemiology Unit — dengue fact sheet (PDF)](https://www.epid.gov.lk/storage/post/pdfs/fact_sheet_on_dengue_for_primary_care_doctors.pdf) |
| `vaccination.md` | [Epidemiology Unit — National Immunization Schedule](https://www.epid.gov.lk/storage/post/pdfs/en_6403b42a75fa4_Doc2.pdf) |
| `vaccination.md` | [Epidemiology Unit — MMR Catch-up Programme](https://www.epid.gov.lk/mmr-catch-up-immunization-programme) |
| `maternal.md`    | [Family Health Bureau — Antenatal & Postnatal Care Unit](https://fhb.health.gov.lk/index.php/en/technical-units/antenatal-postnatal-care-unit) |

**Corpus layout.** English section verbatim from the source (drug names,
vaccine names, age thresholds), with Sinhala and Tamil summaries that
preserve the same drug and vaccine names.

**Evaluation set.** 73 queries in `eval/queries.json` (the public-health
slice of `data/benchmark/queries.json`): 22 parallel items (20 answerable,
including 2 adversarial false-premise items such as "does drinking lime juice
cure dengue?", and 2 unanswerable items about conditions not covered by the
corpus), each asked in English, Sinhala script and Tamil script, plus 7
code-switched versions. 56 of the 73 queries are in the test split.

**Run (offline backend; neural numbers come from `scripts/run_benchmark.py --backend neural`):**
```bash
PYTHONPATH=. python casestudies/public_health/run.py
```

**What this case study stresses.** Cross-lingual fairness and refusal of
medical misinformation. The grounding check forces the system to either
cite a real Ministry-of-Health passage or refuse, which protects users in
Sinhala and Tamil from confident-but-fabricated answers.
