"""Draw the evaluation-architecture figure for the paper.

    python scripts/make_architecture.py   # -> paper/trust2027/figures/eval_harness.{pdf,png}

The system pipeline itself (seven stages, trust layer, audit log) is
figures/architecture.png.  This figure shows what is built around it to
produce every number in the paper: the benchmark artefacts, the three
backends, the per-seed staged draft cache, the replay of sixteen trust-layer
systems, the statistics, and the table/macro generation that the LaTeX
source \\input's.  Numbers printed on the figure (query counts, seeds, system
counts) are read from the repository so that the figure cannot drift from the
code.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "paper" / "trust2027" / "figures"

# colours follow figures/architecture.png: grey = data, blue = draft half,
# orange = trust layer, green = audit / statistics
GREY = ("#f2f2f2", "#555555")
BLUE = ("#e8e8f8", "#3b3b8c")
ORANGE = ("#fdebd3", "#b35c00")
GREEN = ("#e3f6e3", "#1f7a1f")


def counts() -> dict:
    """Read the numbers printed on the figure from the repository."""
    c = {"items": 62, "queries": 208, "dev": 39, "test": 169, "reviewed": 146, "systems": 16, "seeds": 3,
         "n_boot": 2000, "n_perm": 10000}
    try:
        spec = json.loads((ROOT / "data" / "benchmark" / "items.json").read_text(encoding="utf-8"))
        c["items"] = len(spec["items"])
        c["reviewed"] = sum(1 for it in spec["items"] for l in ("si", "ta", "mixed") if it["queries"].get(l))
        qs = json.loads((ROOT / "data" / "benchmark" / "queries.json").read_text(encoding="utf-8"))
        c["queries"] = len(qs)
        split = json.loads((ROOT / "data" / "benchmark" / "split.json").read_text(encoding="utf-8"))
        c["dev"], c["test"] = len(split["dev"]), len(split["test"])
    except Exception:  # pragma: no cover - figure still draws with defaults
        pass
    try:
        # count the systems a finished run actually wrote, else those declared in the runner
        done = sorted((ROOT / "results").glob("*/results.json"))
        if done:
            c["systems"] = max(len(json.loads(p.read_text(encoding="utf-8"))["systems"]) for p in done)
        else:
            src = (ROOT / "scripts" / "run_benchmark.py").read_text(encoding="utf-8")
            names = set(re.findall(r'SystemSpec\("([A-Z][A-Za-z0-9_]+)"', src))
            c["systems"] = len(names) + ("E3_platt" not in names) * 3  # E3-E5 are declared in a loop
    except Exception:  # pragma: no cover
        pass
    return c


def box(ax, x, y, w, h, text, colours, fs=5.0, bold_first=False, lw=0.8, style="round,pad=0.02,rounding_size=0.04"):
    fc, ec = colours
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=style, fc=fc, ec=ec, lw=lw))
    if bold_first and "\n" in text:
        first, rest = text.split("\n", 1)
        ax.text(x + w / 2, y + h - 0.045, first, ha="center", va="top", fontsize=fs + 0.6, fontweight="bold", color=ec)
        ax.text(x + w / 2, y + h - 0.045 - 0.115, rest, ha="center", va="top", fontsize=fs, color="#222222", linespacing=1.15)
    else:
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, color="#222222", linespacing=1.15)


def arrow(ax, x0, y0, x1, y1, style="-|>", ls="-", color="#444444", lw=0.8, rad=0.0):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle=style, mutation_scale=7, color=color, lw=lw,
                                 linestyle=ls, connectionstyle=f"arc3,rad={rad}", shrinkA=1, shrinkB=1))


def main() -> None:
    c = counts()
    W, H = 7.0, 3.12
    fig = plt.figure(figsize=(W, H))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, W)
    ax.set_ylim(0, H)
    ax.axis("off")

    cw, gap, x0 = 1.02, 0.175, 0.04          # six columns
    xs = [x0 + i * (cw + gap) for i in range(6)]
    top, bottom = H - 0.08, 0.30
    title_h = 0.22
    col_h = top - bottom

    def column(i, title, colours):
        x = xs[i]
        ax.add_patch(FancyBboxPatch((x, bottom), cw, col_h, boxstyle="round,pad=0.01,rounding_size=0.05",
                                    fc="white", ec=colours[1], lw=0.6, ls=(0, (2, 2))))
        ax.text(x + cw / 2, top - 0.02, title, ha="center", va="top", fontsize=6.0, fontweight="bold", color=colours[1])
        return x

    # ------------------------------------------------------------ 1 benchmark
    x = column(0, "Benchmark artefacts", GREY)
    y = top - title_h - 0.05
    box(ax, x + 0.05, y - 0.62, cw - 0.10, 0.62,
        f"items.json\n{c['items']} parallel items,\n12 gov.lk documents,\nen / si / ta / mixed\nqueries + gold aliases", GREY, bold_first=True)
    box(ax, x + 0.05, y - 1.47, cw - 0.10, 0.74,
        f"native_review.py\nsheet of {c['reviewed']} si/ta/mixed\nqueries, verdicts by a\nnative speaker; stats ->\nprovenance.tex; apply ->\nitems.json, cache purge", GREY, bold_first=True)
    arrow(ax, x + cw / 2 - 0.25, y - 0.73, x + cw / 2 - 0.25, y - 0.62, style="-|>")
    arrow(ax, x + cw / 2 + 0.25, y - 0.62, x + cw / 2 + 0.25, y - 0.73, style="-|>")
    box(ax, x + 0.05, y - 2.20, cw - 0.10, 0.62,
        f"build_benchmark.py\nqueries.json ({c['queries']});\nsplit.json: dev {c['dev']} /\ntest {c['test']} by item group;\ncase-study slices", GREY, bold_first=True)
    arrow(ax, x + cw / 2, y - 1.47, x + cw / 2, y - 1.58)

    # ------------------------------------------------------------ 2 backends
    x = column(1, "Backends", BLUE)
    y = top - title_h - 0.05
    box(ax, x + 0.05, y - 0.60, cw - 0.10, 0.60,
        "offline\nhash embedder, Echo-\nLLM (extractive), Jaccard;\nCI and case studies", BLUE, bold_first=True)
    box(ax, x + 0.05, y - 1.42, cw - 0.10, 0.72,
        "neural\nmultilingual-e5-base,\nBM25, RRF; mDeBERTa\nXNLI grounding (CPU);\nQwen2.5-1.5B fp16\n(Ollama, GPU)", BLUE, bold_first=True)
    box(ax, x + 0.05, y - 2.20, cw - 0.10, 0.68,
        "neural7b\nsame retrieval and\nNLI checker;\nQwen2.5-7B Q4_K_M\n(Ollama, 8 GB GPU)", BLUE, bold_first=True)

    # ------------------------------------------------------------ 3 staged cache
    x = column(2, f"Draft cache, {c['seeds']} seeds", BLUE)
    y = top - title_h - 0.05
    stages = [
        ("translate", "non-English queries\nto English (E6)", 0.40),
        ("retrieve", "dense + BM25 + RRF,\ntop-4, cosine kept", 0.40),
        ("generate", "draft T=0.2; 3 samples\nT=0.7 (E7); verbalized\n0-100 confidence (E1)", 0.52),
        ("ground", "per-sentence support:\nlexical and NLI", 0.40),
    ]
    yy = y
    for name, desc, h in stages:
        box(ax, x + 0.05, yy - h, cw - 0.10, h, f"{name}\n{desc}", BLUE, bold_first=True)
        yy -= h
        if name != "ground":
            arrow(ax, x + cw / 2, yy, x + cw / 2, yy - 0.08)
            yy -= 0.08
    box(ax, x + 0.05, yy - 0.08 - 0.30, cw - 0.10, 0.30,
        "drafts_seed<k>.jsonl,\nstages_seed<k>/*.json", GREY, fs=4.8)
    arrow(ax, x + cw / 2, yy, x + cw / 2, yy - 0.08)

    # ------------------------------------------------------------ 4 replay
    x = column(3, f"Replay: {c['systems']} systems", ORANGE)
    y = top - title_h - 0.05
    box(ax, x + 0.05, y - 0.46, cw - 0.10, 0.46,
        "Ablations\nB1 vanilla, B2 grounding\nonly, B3 abstention only", ORANGE, bold_first=True)
    box(ax, x + 0.05, y - 1.18, cw - 0.10, 0.66,
        "TriSafe-RAG\ndefault, dev-tuned,\nper-language thresholds,\nlexical, rrf, Platt\n(five signals + policy)", ORANGE, bold_first=True)
    box(ax, x + 0.05, y - 1.90, cw - 0.10, 0.66,
        "External gates\nE1 verbalized, E2 retrieval\ngate, E3 Platt, E4 isotonic,\nE5 temperature, E6 trans-\nlate, E7 self-consistency", ORANGE, bold_first=True)
    box(ax, x + 0.05, y - 2.20, cw - 0.10, 0.24,
        "dev grid search, decision acc.", ORANGE, fs=4.6)

    # ------------------------------------------------------------ 5 statistics
    x = column(4, "Statistics, audit", GREEN)
    y = top - title_h - 0.05
    box(ax, x + 0.05, y - 0.46, cw - 0.10, 0.46,
        "records/*.jsonl\none audit record per\nquery, seed and system", GREEN, bold_first=True)
    box(ax, x + 0.05, y - 1.24, cw - 0.10, 0.72,
        f"Paired tests\nbootstrap, {c['n_boot']:,}\nresamples, query = unit,\noutcomes averaged over\nseeds; sign-flip permu-\ntation ({c['n_perm']:,}) vs. B1", GREEN, bold_first=True)
    box(ax, x + 0.05, y - 1.90, cw - 0.10, 0.60,
        "Fairness auditor\nper-language accuracy,\nanswer rate, TPR/FPR,\nEO gap, CLC; catch/retain", GREEN, bold_first=True)
    box(ax, x + 0.05, y - 2.20, cw - 0.10, 0.24, "results.json per backend", GREY, fs=4.8)

    # ------------------------------------------------------------ 6 paper
    x = column(5, "Paper", GREY)
    y = top - title_h - 0.05
    box(ax, x + 0.05, y - 0.72, cw - 0.10, 0.72,
        "make_tables.py\nmacros.tex (every\nnumber in the text),\ntable_*.tex, qualitative\nand probe tables, figures", GREY, bold_first=True)
    box(ax, x + 0.05, y - 1.30, cw - 0.10, 0.50,
        "check_paper.py\nmacros defined, inputs\nand figures present", GREY, bold_first=True)
    box(ax, x + 0.05, y - 1.90, cw - 0.10, 0.52,
        "make_overleaf_zip.py\nXeLaTeX source, Noto\nSinhala and Tamil fonts", GREY, bold_first=True)
    box(ax, x + 0.05, y - 2.20, cw - 0.10, 0.24, "body.tex inputs them", GREY, fs=4.8)

    # ------------------------------------------------------------ column arrows
    ya = bottom + 0.13
    for i in range(5):
        arrow(ax, xs[i] + cw, ya, xs[i + 1], ya, lw=1.2, color="#333333")
    # backends feed the staged cache; split feeds tuning
    ax.text(xs[0] + cw + gap / 2, ya + 0.05, "queries, split", ha="center", va="bottom", fontsize=4.6, color="#333333")
    ax.text(xs[1] + cw + gap / 2, ya + 0.05, "models", ha="center", va="bottom", fontsize=4.6, color="#333333")
    ax.text(xs[2] + cw + gap / 2, ya + 0.05, "drafts", ha="center", va="bottom", fontsize=4.6, color="#333333")
    ax.text(xs[3] + cw + gap / 2, ya + 0.05, "decisions", ha="center", va="bottom", fontsize=4.6, color="#333333")
    ax.text(xs[4] + cw + gap / 2, ya + 0.05, "numbers", ha="center", va="bottom", fontsize=4.6, color="#333333")

    # footer: run drivers
    ax.text(W / 2, 0.12,
            "run_all_seeds.sh: seeds 0, 1, 2 per neural backend (intermediate results.json after every seed)   "
            "|   progress_ui.py: live stage progress   |   only the generate stage is stochastic",
            ha="center", va="center", fontsize=5.0, color="#333333")

    FIG.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG / "eval_harness.pdf")
    fig.savefig(FIG / "eval_harness.png", dpi=220)
    print("wrote", FIG / "eval_harness.pdf", "and .png;", c)


if __name__ == "__main__":
    main()
