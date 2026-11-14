"""Package paper/trust2027 as an Overleaf-ready zip.

    python scripts/make_overleaf_zip.py   -> paper/TriSafe-RAG_TRUST2027_overleaf.zip

The zip contains the ACM class and bibliography style, main.tex (anonymous),
main_named.tex, preamble/body, refs.bib, the generated tables/macros and the
figures. If a backend run has not finished yet, the script writes
``generated/fallbacks.tex`` (\\providecommand stubs rendering "??") and stub
files for missing \\input targets so that the project still compiles.
Overleaf settings: compiler pdfLaTeX, main document main.tex.
"""

from __future__ import annotations

import re
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PAPER = ROOT / "paper" / "trust2027"
ZIP = ROOT / "paper" / "TriSafe-RAG_TRUST2027_overleaf.zip"


def write_fallbacks() -> list[str]:
    body = (PAPER / "body.tex").read_text(encoding="utf-8") + (PAPER / "supplement.tex").read_text(encoding="utf-8")
    macros_path = PAPER / "generated" / "macros.tex"
    macros = macros_path.read_text(encoding="utf-8") if macros_path.exists() else ""
    defined = set(re.findall(r"\\newcommand\{\\([A-Za-z]+)\}", macros))
    used = sorted(set(re.findall(r"\\([NO][A-Z][A-Za-z]*)", body)))
    missing = [m for m in used if m not in defined]
    lines = ["%% auto-generated: placeholders for macros whose run is pending"]
    lines += [f"\\providecommand{{\\{m}}}{{\\textbf{{??}}}}" for m in missing]
    (PAPER / "generated" / "fallbacks.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")
    stubs = []
    for inc in re.findall(r"\\input\{([^}]+)\}", body):
        p = PAPER / (inc if inc.endswith(".tex") else inc + ".tex")
        if not p.exists():
            p.write_text("\\emph{(table pending: this backend run has not finished)}\n", encoding="utf-8")
            stubs.append(str(p.relative_to(PAPER)))
    return missing + stubs


def main() -> None:
    pending = write_fallbacks()
    files = ["main.tex", "main_named.tex", "main_supplement.tex", "preamble.tex", "body.tex", "supplement.tex",
             "refs.bib", "acmart.cls", "ACM-Reference-Format.bst"]
    with zipfile.ZipFile(ZIP, "w", zipfile.ZIP_DEFLATED) as z:
        for f in files:
            z.write(PAPER / f, f)
        for sub in ("generated", "figures", "fonts"):
            for p in sorted((PAPER / sub).iterdir()):
                if p.suffix in {".tex", ".pdf", ".png", ".ttf"}:
                    z.write(p, f"{sub}/{p.name}")
        z.writestr(
            "README_OVERLEAF.txt",
            "Upload this zip as a new Overleaf project.\n"
            "Menu -> Compiler: XeLaTeX (required to typeset the Sinhala and Tamil queries from fonts/;\n"
            "pdfLaTeX also compiles but prints [Sinhala text]/[Tamil text] markers instead).\n"
            "Main document: main.tex (anonymous review build, 9 pages + references) or main_named.tex (camera-ready).\n"
            "Supplementary material: compile main_supplement.tex as a second document (not counted in the page limit).\n"
            "Tables and numbers are generated from results/*/results.json by scripts/make_tables.py;\n"
            "do not edit generated/*.tex by hand.\n"
            + ("\nPENDING (placeholders rendered as ?? until the neural run finishes):\n  " + "\n  ".join(pending) + "\n" if pending else "\nAll generated inputs present.\n"),
        )
    print(f"wrote {ZIP.relative_to(ROOT)} ({ZIP.stat().st_size // 1024} KB); pending items: {len(pending)}")


if __name__ == "__main__":
    sys.exit(main())
