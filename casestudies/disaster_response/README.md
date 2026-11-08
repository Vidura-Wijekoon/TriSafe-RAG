# Case Study 3 — Disaster Response

**Domain.** Disaster Management Centre flood operations, NBRO landslide
early warnings, and Department of Meteorology cyclone advisories.

**Real data sources used in this case study** (see `SOURCES.md` for the full
list). Hotline numbers, rainfall thresholds, and safety advice are quoted
from official ministry and DMC publications.

| Document             | Source                                              |
| -------------------- | --------------------------------------------------- |
| `flood_current.md`   | [DMC — 117 hotline & operations page](https://dmc.gov.lk/index.php?Itemid=226&id=61&lang=en&option=com_content&view=article) |
| `flood_current.md`   | [DMC — Situation Reports portal](https://www.dmc.gov.lk/index.php?option=com_dmcreports&view=reports&report_type_id=1&Itemid=273&lang=en) |
| `landslide.md`       | [NBRO — Landslide Early Warning Issued](https://www.nbro.gov.lk/index.php?option=com_content&view=article&id=436:landslide-early-warning-issued&catid=8&Itemid=190&lang=en) |
| `cyclone.md`         | [Department of Meteorology](https://www.meteo.gov.lk/) |
| `flood_archived.md`  | *Synthetic stale bulletin (adversarial test case)*  |

**Corpus layout.** English verbatim from the source. Sinhala and Tamil
summaries preserve the same numbers (117, 75 mm, 21 operators, 20–30 minutes
of eye calm).

The `flood_archived.md` file is deliberately retained as an *adversarial*
entry: it carries a fictional past-season hotline (`110`) that contradicts
the current bulletin's `117`. Its purpose is to test whether the grounding
check and abstention policy refuse rather than pick one of two contradictory
passages.

**Evaluation set.** 60 queries in `eval/queries.json` (the disaster-response
slice of `data/benchmark/queries.json`): 18 parallel items (13 answerable,
including the probe "what is the current DMC hotline this season?" whose
answer must come from the current bulletin, and 5 unanswerable: three
stale-bulletin probes about the past-season helpline and shelters, and two
out-of-scope questions), each asked in English, Sinhala script and Tamil
script, plus 6 code-switched versions. 50 of the 60 queries are in the test
split, 17 of them unanswerable.

**Run (offline backend; neural numbers come from `scripts/run_benchmark.py --backend neural`):**
```bash
PYTHONPATH=. python casestudies/disaster_response/run.py
```

**What this case study stresses.** Abstention under contradictory or stale
evidence. The vanilla baseline will happily quote the archived hotline; the
full pipeline notices both the current and archived passages and refuses or
defers rather than pick a side.
