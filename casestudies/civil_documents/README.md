# Case Study 1 — Civil Documents

**Domain.** Routine civil-service queries: passport renewal, National Identity
Card replacement, driving licence, online vehicle revenue licence (eRL).

**Real data sources used in this case study** (see `SOURCES.md` for the full
list). All fees and form names are quoted directly from the official notices
listed below.

| Document            | Source                                  |
| ------------------- | --------------------------------------- |
| `passport.md`       | [immigration.gov.lk](https://www.immigration.gov.lk/pages_e.php?id=8) |
| `nic.md`            | [DRP fee notice (PDF, effective 2022-11-01)](https://drp.gov.lk/en/assets/downloads/fees-english.pdf) |
| `driving_licence.md`| [DMT — Charges for Services](https://dmt.gov.lk/index.php?option=com_content&view=article&id=85&Itemid=170&lang=en) |
| `revenue_licence.md`| [eRL portal — web.erl2.gov.lk](https://web.erl2.gov.lk/) |

**Corpus layout.** Each markdown document has an English section (verbatim
from the source), followed by a Sinhala and a Tamil summary using the same
numeric values and form names.

**Evaluation set.** 75 queries in `eval/queries.json` (the civil-documents
slice of `data/benchmark/queries.json`): 22 parallel items (19 answerable,
3 unanswerable), each asked in English, Sinhala script and Tamil script, plus
9 code-switched (romanised Sinhala--English) versions. Gold answers are lists
of aliases for short factual spans (e.g. `10,000`, `K-35A`, `Battaramulla`,
with Sinhala and Tamil spellings where the answer is a word). 63 of the 75
queries are in the test split; the rest belong to the development split used
for threshold tuning.

**Run (offline backend; neural numbers come from `scripts/run_benchmark.py --backend neural`):**
```bash
PYTHONPATH=. python casestudies/civil_documents/run.py
```

Output: `results/summary.json` plus `<system>_per_query.jsonl` traces that
show the per-query decision, confidence, grounding score and signals.

**What this case study stresses.** Grounding and citation. A confidently
wrong fee number is a hard failure. The TriSafe pipeline lowers the answer
rate (it refuses queries where the retriever found no supporting passage),
and the per-language audit makes it visible when Sinhala or Tamil queries
are systematically worse-served than English.
