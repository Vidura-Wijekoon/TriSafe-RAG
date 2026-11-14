# Paper sources

- `trust2027/` — **canonical** ACM sigconf source for the TRUST 2027 submission.
  - `main.tex`: anonymous, review-mode build (line numbers).
  - `main_named.tex`: camera-ready variant with author block (same body via `\input{body}`).
  - `main_supplement.tex` + `supplement.tex`: supplementary material compiled as a second PDF (evaluation architecture, benchmark examples, qualitative records, probes, full external-baseline table, calibration figures, case studies, 1.5B and offline tables, cross-validation, per-seed tables). TRUST 2027 counts appendices toward the 9-page limit, so nothing beyond the limit is in `body.tex`.
  - `generated/`: tables and macros written by `scripts/make_tables.py` from `results/*/results.json`. **Do not edit by hand**; rerun the script.
  - `figures/`: `architecture.png` (pipeline), `eval_harness.pdf` (evaluation architecture, drawn by `scripts/make_architecture.py` from repository counts), and the reliability and language trade-off plots written by `make_tables.py`.
  - `fonts/`: Noto Sans Sinhala and Tamil (OFL) used by `preamble.tex` under XeLaTeX to typeset the benchmark queries.
  - `refs.bib`, `acmart.cls`, `ACM-Reference-Format.bst`: vendored ACM template files.
- `abstract_v1.md`, `gap_statement.md`, `objectives.md` — early planning notes (historical).

## Build

```bash
cd paper/trust2027
xelatex main && bibtex main && xelatex main && xelatex main
xelatex main_supplement && xelatex main_supplement   # supplementary PDF
# or: latexmk -xelatex main      (pdflatex also compiles; Sinhala/Tamil then print as markers)
```

Or build `paper/TriSafe-RAG_TRUST2027_overleaf.zip` with `python scripts/make_overleaf_zip.py` and upload it to Overleaf with the compiler set to XeLaTeX.

Before building, make sure the generated tables match the results you intend to report:

```bash
python scripts/run_benchmark.py --backend offline
bash scripts/run_all_seeds.sh neural7b neural     # three seeds per generator (hours per seed on an 8 GB GPU)
python scripts/make_tables.py
python scripts/make_architecture.py
python scripts/check_paper.py                      # 0 problems expected once every backend has a results.json
```

## Status (2026-11-14, review v5 addressed)

- Benchmark final; all 146 Sinhala, Tamil and code-switched queries reviewed and accepted by a native-speaker author (`generated/provenance.tex`).
- Offline backend results final.
- Neural runs complete: three seeds per generator (7B = `L` macros, headline; 1.5B = `N` macros, Sec. Generator scale and the appendix). Counts in the cost and gate tables are means per seed. `scripts/check_paper.py` reports 0 problems.
- Abstention detector fixed (string anywhere in the reply); gate-isolation, empty-reply filter, cost / break-even, 5-fold group cross-fitting, show-the-passage and adjudication (`generated/adjudication.tex`) added on 2026-11-09. The 7B seed-0 results were replayed with the new systems on 2026-11-09.
- Review v5 (2026-11-14): main paper cut to the 9-page limit with the rest in `supplement.tex`; intervals and permutation tests are cluster bootstraps over the 50 test items; ECE on answered drafts added to the main table; break-even against the empty-reply filter (and all pairwise crossings incl. B3 and E6) added to the cost analysis; the empty-reply filter is checked on the development split, on the 1.5B generator and by regenerating the 30 citation-only replies under a longer token cap and a stricter instruction (`scripts/probe_empty_replies.py`, `results/neural7b/probe_empty_replies.json`); title changed from "Joint ..." to "Auditing ..."; text contradictions listed in the review fixed. Page count must be confirmed on Overleaf (no local LaTeX).
- The `%% STATUS` block at the top of `body.tex` lists the remaining steps.
