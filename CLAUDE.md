# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this repository is

The LaTeX source of Shreeyut Maheshwari's (230003071) B.Tech. project report at IIT Indore (guide: Dr. Mohd. Arshad),
"Accelerating Diffusion Models with Training-Free Caching: ReSPect and KalSea, with Applications in Mechanical
Engineering". Remote: `git@github.com:shreeyut1905/Thesis.git` (branch `main`, SSH auth works on this machine).
`main.pdf` is committed on purpose so the author can read it on GitHub — rebuild and commit it with every content change.

The research code the thesis describes lives outside this repo: `../ReSpect` and `../KalSea-v0.2` (the two method
releases, each with its own CLAUDE.md context in `../CLAUDE.md`), the paper PDFs in `..` (`ReSPect__...pdf`;
`main_authors.pdf` is the KalSea paper), and the application harness in `~/mech_apps/cache`.

## Build

```bash
latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex   # build main.pdf
latexmk -c main.tex                                              # remove aux files before committing
```

- TeX packages come from apt (`texlive-latex-extra`, `texlive-science`, `texlive-fonts-extra`, `texlive-plain-generic`,
  `tex-gyre`, `latexmk`). Missing `binhex.tex` → `texlive-plain-generic`; missing `ts1-qtmr` → `tex-gyre`.
- `amsthm` must be loaded before `newtxmath` (else "Command \openbox already defined").
- The shell's `grep` is a wrapper that skips git-ignored files, so it silently finds nothing in `main.log`.
  Use `command grep` or Python to look for `Overfull`, `undefined` or `^!` in the log after a build.

## Document structure

- `main.tex`: preamble, project macros, and all front matter (two title pages with `figures/iiti_logo.jpg`,
  declaration + guide certificate, preface, acknowledgments). The layout mirrors the reference report
  `../BTP_REPORT_AKSHAY_200003007.pdf`: continuous "Page | N" footer from the cover onward. Title pages therefore use
  `\clearpage\begingroup ... \endgroup` — the `titlepage` environment resets page numbering and must not be used.
- `chapters/`: `abstract`, `abbreviations`, `ch1_introduction` … `ch7_conclusion`, `appendix` (all `\input`).
- Macros: `\ReSPect`, `\KalSea`, `\Emb`, `\Out`, `\SEA`, `\NFE`, `\Lone`, and `\TBD` (red **[TBD]** for numbers that
  are still pending). Labels are prefixed per chapter (`sec:lit-*`, `sec:bg-*`, `*:respect-*`, `*:kalsea-*`, `*:app-*`);
  later chapters refer back to background labels such as `sec:bg-sea`, `eq:bg-wiener`, `eq:bg-gate`, `sec:bg-residual`.
- `references.bib` is the single bibliography (natbib, numeric, `unsrtnat`).
- `figures/`: `respect_*` / `kalsea_*` are vector crops of the two paper PDFs; `bg_*`, `lit_chebyshev` are generated
  plots; `app_*` are produced by `applications/make_figs.py` from the run results.
- `code/` holds verbatim copies of source files that `chapters/appendix.tex` includes with
  `\lstinputlisting[firstline=..,lastline=..]`. If a copy is refreshed from its source, re-check those line ranges.

## Content rules (from the author)

- Every number in the ReSPect and KalSea chapters is copied exactly from the paper text; do not round or invent.
- In Chapter 6 the rows of the uncached original models are **copied from their papers, never re-run** (OAT: paper
  Table 4 — failure 39.22%, CE* 8.17%, median 3.75%, VFE* −0.12%; DiffusionPDE bounded NS: 3.9 ± 0.2% forward,
  2.7 ± 0.2% inverse). Only the cached samplers (ReSPect, KalSea) are run.
- Do not claim the papers were accepted at a venue. Pending numbers stay `\TBD` rather than estimates.

## Chapter 6 applications (`applications/`)

The files in `applications/` are copies of `~/mech_apps/cache/`, where they actually run next to `cachecore.py` and
the forecaster modules (`spectrum_forecaster.py` from ReSpect, `kalsea_forecaster.py` from KalSea-v0.2/FLUX2,
`util_seacache.py`), imported unchanged. Environment: `~/mech_apps/.venv` (Python 3.10, torch 2.8 cu128, diffusers 0.35,
cupy 13.6), `HF_HOME=~/mech_apps/hf_cache`, third-party code in `~/mech_apps/OAT` and `~/mech_apps/DiffusionPDE`,
outputs in `~/mech_apps/results/{oat,pde}`.

- **OAT** (topology optimisation, NeurIPS 2025): `run_oat_all.sh` shards the 5000-problem OpenTO test split over 8 GPUs
  (paper protocol: 20 DDIM steps, CFG 2.0, 10 samples/problem); `eval_oat.py` runs OAT's own
  `scripts/RunComplianceTest.py` unchanged via `runpy`, only redirecting `load_dataset("OpenTO/OpenTO")` to the local
  test parquet. pyEDGE needs cupy 13 (`cupy.sparse`) and `LD_LIBRARY_PATH` pointing at the venv's `nvidia/*/lib`
  (`run_eval_oat.sh` sets it).
- `tab:app-oat` uses OAT's "@ 1" statistics from `Compliance_{method}_d{delta}.pkl["stats"]`: failure = (1 − `Pass @ 1`)·100,
  CE* = `Avg CE @ 1`, median = `Median CE @ 1`, VFE* = `Avg VFE @ 1`; Evals = 2 × mean `nfe` of the shard JSONs (CFG).
  `applications/make_figs.py {oat,oat_dist,schedules,pde_tradeoff}` regenerates the `app_*` figures from these results.
- **DiffusionPDE** (bounded Navier–Stokes, NeurIPS 2024): `pde_cache.py --case {1,2} --problem {forward,inverse}`,
  2000 guided Heun steps; its checkpoints are pickles.
- Invariants the drivers were validated against and must keep: with caching off the loop is bitwise identical to the
  official pipeline/denoiser; with δ = 0 ReSPect and KalSea are bitwise identical to it. This relies on computed steps
  passing `h_out` straight to the output layers (not `h_in + r` in bf16) and residuals kept in float32. Decoding must
  run under `torch.no_grad()` (otherwise CPU copies keep GPU autograd graphs alive and memory explodes).
- When killing runs, `pkill -f <script>` also matches the invoking shell; kill by PID instead.
