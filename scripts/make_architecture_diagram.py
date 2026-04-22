"""RETFound + Ordinal CapsNet architecture diagram — TikZ build driver.

Compiles the standalone TikZ source `scripts/fig_architecture.tex` (kept
under version control) via pdflatex, writes the result to
`results/figures/fig_architecture.pdf` (canonical, 300 dpi PNG alongside
for slides/non-LaTeX contexts), and also drops a copy at
`paper/submission/figs/fig_architecture.pdf` so the paper's
\\includegraphics path continues to resolve.

Requires: pdflatex (TeX Live or MacTeX) and pdftoppm (poppler-utils).

Usage:
    uv run python scripts/make_architecture_diagram.py
"""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
TIKZ_SRC = REPO / "scripts" / "fig_architecture.tex"
BUILD_DIR = REPO / "results" / "figures"
BUILD_PDF = BUILD_DIR / "fig_architecture.pdf"  # pdflatex writes here
OUT_PNG = BUILD_DIR / "fig_architecture.png"
PAPER_PDF = REPO / "paper" / "submission" / "figs" / "fig_architecture.pdf"


def _check_tool(name: str) -> None:
    if shutil.which(name) is None:
        sys.exit(f"ERROR: `{name}` not on PATH. Install TeX Live / MacTeX and poppler-utils.")


def main() -> None:
    if not TIKZ_SRC.exists():
        sys.exit(f"TikZ source missing: {TIKZ_SRC}")
    _check_tool("pdflatex")
    _check_tool("pdftoppm")

    BUILD_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Compiling {TIKZ_SRC.relative_to(REPO)} ...")
    proc = subprocess.run(
        ["pdflatex", "-interaction=nonstopmode",
         "-output-directory", str(BUILD_DIR), str(TIKZ_SRC)],
        capture_output=True, text=True,
    )
    if proc.returncode != 0:
        print(proc.stdout[-2000:])
        sys.exit("pdflatex failed — see log above")

    if not BUILD_PDF.exists():
        sys.exit(f"pdflatex ran but {BUILD_PDF} is missing")
    print(f"  -> {BUILD_PDF.relative_to(REPO)}")

    subprocess.run(
        ["pdftoppm", "-png", "-r", "300", "-singlefile",
         str(BUILD_PDF), str(OUT_PNG.with_suffix(""))],
        check=True,
    )
    print(f"  -> {OUT_PNG.relative_to(REPO)}")

    if PAPER_PDF.parent.exists():
        shutil.copy2(BUILD_PDF, PAPER_PDF)
        print(f"  -> {PAPER_PDF.relative_to(REPO)}  (for the paper's \\includegraphics)")

    # Tidy pdflatex aux artifacts to keep results/figures/ showing only final outputs.
    for ext in ("aux", "log", "out", "synctex.gz"):
        for p in BUILD_DIR.glob(f"fig_architecture.{ext}"):
            p.unlink()

    print("Done.")


if __name__ == "__main__":
    main()
