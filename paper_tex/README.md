# LaTeX paper build

Source for the LaTeX version of the paper draft. Compiles with a standard `pdflatex` + `bibtex` toolchain.

## Files

- `main.tex` — paper source. Mirrors `paper/draft_v2.md` section-by-section.
- `references.bib` — BibTeX bibliography. All arXiv IDs verified against arxiv.org (see `paper/ml_intern_v2_review_*.md`).

## Build

```bash
cd paper_tex
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex
```

Or with `latexmk`:

```bash
cd paper_tex
latexmk -pdf main.tex
```

Output: `main.pdf`.

## Sync with markdown draft

When updating the paper, edit `paper/draft_v2.md` first, then port changes to `main.tex`. The two are intentionally kept in lock-step. New citations require both an inline `\cite{key}` in `main.tex` and a corresponding entry in `references.bib`.
