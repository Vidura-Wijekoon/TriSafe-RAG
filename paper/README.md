# Paper sources

- `trust2027/` — **canonical** ACM sigconf source for the TRUST 2027 submission.
  - `main.tex`: anonymous, review-mode build (line numbers).
  - `main_named.tex`: camera-ready variant with author block (same body via `\input{body}`).
  - `generated/`: tables and macros written by `scripts/make_tables.py` from `results/*/results.json`. **Do not edit by hand**; rerun the script.
  - `figures/`: `architecture.png` (pipeline), `eval_harness.pdf` (evaluation architecture, drawn by `scripts/make_architecture.py` from repository counts), and the reliability and language trade-off plots written by `make_tables.py`.
  - `fonts/`: Noto Sans Sinhala and Tamil (OFL) used by `preamble.tex` under XeLaTeX to typeset the benchmark queries.
  - `refs.bib`, `acmart.cls`, `ACM-Reference-Format.bst`: vendored ACM template files.
- `abstract_v1.md`, `gap_statement.md`, `objectives.md` — early planning notes (historical).

## Build

```bash
cd paper/trust2027
xelatex main && bibtex main && xelatex main && xelatex main
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

## Status (2026-11-09)

- Benchmark final; all 146 Sinhala, Tamil and code-switched queries reviewed and accepted by a native-speaker author (`generated/provenance.tex`).
- Offline backend results final.
- Neural runs complete: three seeds per generator (7B = `L` macros, headline; 1.5B = `N` macros, Sec. Generator scale and the appendix). Counts in the cost and gate tables are means per seed. `scripts/check_paper.py` reports 0 problems.
- Abstention detector fixed (string anywhere in the reply); gate-isolation, empty-reply filter, cost / break-even, 5-fold group cross-fitting, show-the-passage and adjudication (`generated/adjudication.tex`) added on 2026-11-09. The 7B seed-0 results were replayed with the new systems on 2026-11-09.
- The `%% STATUS` block at the top of `body.tex` lists the remaining steps.
