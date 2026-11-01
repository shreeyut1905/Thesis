# B.Tech. Project Report — Accelerating Diffusion Models with Training-Free Caching

**ReSPect and KalSea, with Applications in Mechanical Engineering**
Shreeyut Maheshwari (230003071), IIT Indore — guided by Dr. Mohd. Arshad, November 2026.

`main.pdf` is the compiled report.

## Build

```bash
latexmk -pdf main.tex          # or: pdflatex main && bibtex main && pdflatex main && pdflatex main
```

Requires a TeX Live installation with `newtx`, `tex-gyre`, `algorithmicx`, `listings`, `cleveref`, `natbib`.

## Layout

- `main.tex` — preamble, title pages, declaration/certificate, preface, acknowledgments, lists
- `chapters/` — abstract, abbreviations, Chapters 1–7, appendix
- `references.bib` — bibliography
- `figures/` — IIT Indore logo, figures cropped from the ReSPect and KalSea papers, generated plots
- `code/` — source files included verbatim in the code appendix (`\lstinputlisting`)
- `applications/` — drivers for the mechanical-engineering applications of Chapter 6
  (`oat_cache.py`: Optimize Any Topology; `pde_cache.py`: DiffusionPDE). They need `code/cachecore.py`
  and the ReSPect / KalSea forecaster modules next to them.

## Status

Chapter 6 (applications): the DiffusionPDE results (bounded Navier-Stokes, forward and inverse, both released cases,
ReSPect and KalSea at two budgets) are in. The OAT results (full 5000-problem OpenTO test split, 10 samples each,
OAT's own FEA) are in for both methods at both budgets. Baseline rows of the uncached models are copied from the
original papers. Result figures are produced by `applications/make_figs.py`.
