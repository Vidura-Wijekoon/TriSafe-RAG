"""Consistency checks for the paper source: every macro used in body.tex is
defined in generated/macros.tex, every \\input target exists, every figure
exists, and no stray non-ASCII characters remain in generated files (pdflatex
cannot typeset Sinhala/Tamil script without fontspec).

    python scripts/check_paper.py
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PAPER = ROOT / "paper" / "trust2027"


def main() -> int:
    body = (PAPER / "body.tex").read_text(encoding="utf-8")
    macros = (PAPER / "generated" / "macros.tex").read_text(encoding="utf-8")
    defined = set(re.findall(r"\\newcommand\{\\([A-Za-z]+)\}", macros))
    used = sorted(set(re.findall(r"\\([NO][A-Z][A-Za-z]*)", body)))
    missing = [m for m in used if m not in defined]
    problems = 0
    if missing:
        problems += 1
        print("Missing macros:", ", ".join(missing))
    for inc in re.findall(r"\\input\{([^}]+)\}", body + (PAPER / "main.tex").read_text(encoding="utf-8")):
        p = PAPER / (inc if inc.endswith(".tex") else inc + ".tex")
        if not p.exists():
            problems += 1
            print("Missing \\input target:", p.relative_to(ROOT))
    for fig in re.findall(r"\\includegraphics\[[^\]]*\]\{([^}]+)\}", body):
        if not (PAPER / fig).exists():
            problems += 1
            print("Missing figure:", fig)
    for gen in (PAPER / "generated").glob("*.tex"):
        txt = gen.read_text(encoding="utf-8")
        bad = sorted({c for c in txt if ord(c) > 127})
        if bad:
            problems += 1
            print(f"Non-ASCII in {gen.name}: {''.join(bad)[:40]}")
    dashes = [m.start() for m in re.finditer("\u2014", body)]
    if dashes:
        print(f"note: {len(dashes)} em-dash characters in body.tex")
    print("macros used:", len(used), "| defined:", len(defined), "| problems:", problems)
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
