"""Package paper/trust2027 as an Overleaf-ready zip.

    python scripts/make_overleaf_zip.py   -> paper/TriSafe-RAG_TRUST2027_overleaf.zip      (double-blind submission)
                                             paper/TriSafe-RAG_TRUST2027_camera_ready.zip  (author block, ACM front matter)

Each zip contains the ACM class and bibliography style, a main.tex and a
main_supplement.tex, preamble/body, refs.bib, the generated tables/macros and the
figures.  The submission zip holds only the anonymous drivers (nothing in it
names the author); in the camera-ready zip the named drivers
main_named.tex / main_supplement_named.tex ARE main.tex / main_supplement.tex. If a backend run has not finished yet, the script writes
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
ZIP_CR = ROOT / "paper" / "TriSafe-RAG_TRUST2027_camera_ready.zip"
COMMON = ["preamble.tex", "body.tex", "supplement.tex", "refs.bib", "acmart.cls", "ACM-Reference-Format.bst"]


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


def build(zip_path: Path, drivers: dict, readme: str, pending: list[str]) -> None:
    """drivers: {name in zip: source file in paper/trust2027}."""
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as z:
        for name, src in drivers.items():
            z.write(PAPER / src, name)
        for f in COMMON:
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
            + readme
            + "Supplementary material: compile main_supplement.tex as a second document (not counted in the page limit).\n"
            "Tables and numbers are generated from results/*/results.json by scripts/make_tables.py;\n"
            "do not edit generated/*.tex by hand.\n"
            + ("\nPENDING (placeholders rendered as ?? until the neural run finishes):\n  " + "\n  ".join(pending) + "\n" if pending else "\nAll generated inputs present.\n"),
        )
    print(f"wrote {zip_path.relative_to(ROOT)} ({zip_path.stat().st_size // 1024} KB); pending items: {len(pending)}")


def main() -> None:
    pending = write_fallbacks()
    build(ZIP,
          {"main.tex": "main.tex", "main_supplement.tex": "main_supplement.tex"},
          "Main document: main.tex (DOUBLE-BLIND review build: anonymous, line numbers; 9 pages + references).\n"
          "This zip contains no author name or affiliation; the camera-ready drivers are in the separate camera-ready zip.\n",
          pending)
    build(ZIP_CR,
          {"main.tex": "main_named.tex", "main_supplement.tex": "main_supplement_named.tex"},
          "Main document: main.tex (CAMERA-READY build: author block, ACM reference block, rights footnote, no line numbers).\n"
          "Before the final upload replace the placeholders in main.tex (\\setcopyright, \\acmDOI, \\acmISBN)\n"
          "with the values from the ACM rights-form e-mail.\n",
          pending)


if __name__ == "__main__":
    sys.exit(main())
