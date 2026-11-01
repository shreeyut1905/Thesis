"""Figures of Chapter 6 of the thesis (run after generation and evaluation).

    python make_figs.py oat            # selected OAT examples            -> app_oat_examples.pdf
    python make_figs.py oat_tradeoff   # OAT metrics vs. evaluations      -> app_oat_tradeoff.pdf
    python make_figs.py schedules      # steps computed by the gate       -> app_gate_schedules.pdf
    python make_figs.py pde_tradeoff   # DiffusionPDE error vs. calls     -> app_pde_tradeoff.pdf
    python make_figs.py pde forward 1  # DiffusionPDE fields and errors   -> app_pde_forward_case1.pdf
    python make_figs.py all

All figures are drawn at the printed width of the report (TEXTW) so that their text appears at 7-8 pt.
"""
import glob
import json
import os
import pickle
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
RES = os.path.expanduser("~/mech_apps/results")
FIG = os.path.expanduser("~/BTP/thesis/figures")

TEXTW = 6.0  # \textwidth of the report in inches (A4 with 1.25 in + 1 in side margins)
plt.rcParams.update({
    "font.family": "STIXGeneral", "mathtext.fontset": "stix",
    "font.size": 8, "axes.titlesize": 8, "axes.labelsize": 8, "legend.fontsize": 7,
    "xtick.labelsize": 7, "ytick.labelsize": 7,
    "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
    "xtick.major.size": 2.5, "ytick.major.size": 2.5, "xtick.minor.size": 1.5, "ytick.minor.size": 1.5,
    "lines.linewidth": 1.2, "pdf.fonttype": 42, "savefig.dpi": 300,
})

# Colours: categorical slots 1 and 2 of the reference palette for the two methods (identity follows the method in
# every figure), neutral inks for everything else.
C_KAL, C_RSP = "#2a78d6", "#eb6834"
INK, INK2, MUTED, GRID, AXIS, PLANE = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7", "#f0efec"
LOAD = "#e34948"
NAME = {"kalsea": "KalSea", "respect": "ReSPect"}
COL = {"kalsea": C_KAL, "respect": C_RSP}
OAT_CFGS = [("respect", 0.15), ("kalsea", 0.15), ("respect", 0.3), ("kalsea", 0.3)]
PDE_CFGS = [("respect", 0.01), ("kalsea", 0.01), ("respect", 0.02), ("kalsea", 0.02)]
OAT_PAPER = {"fail": 39.22, "ce": 8.17, "median": 3.75, "vfe": -0.12, "evals": 40}  # OAT paper, Table 4 (not re-run)


def _style(ax, grid="y"):
    if grid:
        ax.grid(True, axis=grid, color=GRID, lw=0.5)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(AXIS)
    ax.tick_params(colors=INK2, labelcolor=INK2)


def _image_axes(ax):
    ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_color(AXIS); sp.set_linewidth(0.5)


def _save(fig, name):
    fig.savefig(os.path.join(FIG, name), bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    print("saved", name)


# ------------------------------------------------------------------------------------------------ OAT
def _oat_dataset():
    from datasets import load_dataset
    from oat_cache import TEST_PARQUET
    return load_dataset("parquet", data_files={"test": TEST_PARQUET})["test"]


def _oat_compliance(m, d):
    return pickle.load(open(os.path.join(RES, "oat", f"Compliance_{m}_d{d:g}.pkl"), "rb"))


def _oat_nfe(m, d):
    out = {}
    for f in glob.glob(os.path.join(RES, "oat", f"{m}_d{d:g}_shard*.json")):
        for r in json.load(open(f))["records"]:
            out[r["idx"]] = r["nfe"]
    return out


def _draw_problem(ax, s, H, W):
    """Supports (filled triangle: both displacements fixed, open: one) and loads (arrows ending at the node).

    Coordinates as in OAT's RunComplianceTest.py: the first coordinate runs along the rows, the second along the
    columns, both scaled by max(H, W)."""
    L = max(H, W)
    for x, y, f0, f1 in np.array(s["boundary conditions"]):
        ax.plot(y * L - 0.5, x * L - 0.5, marker="^", ms=5.5, mfc=INK if (f0 and f1) else "white", mec=INK, mew=0.8,
                ls="", zorder=5, clip_on=False)
    ld = np.array(s["loads"])
    n = np.linalg.norm(ld[:, 2:], axis=1).max()
    for x, y, f0, f1 in ld:
        r, c = x * L - 0.5, y * L - 0.5
        k = 0.2 * L / n
        ax.annotate("", xy=(c, r), xytext=(c - f1 * k, r - f0 * k), zorder=6, annotation_clip=False,
                    arrowprops=dict(arrowstyle="-|>,head_length=0.5,head_width=0.25", color=LOAD, lw=1.3,
                                    shrinkA=0, shrinkB=0))


def oat_examples(problems=(4153, 4494, 437, 310), sample=0):
    """Selected OpenTO test problems: SIMP reference with supports and loads, and one sample of each cached sampler."""
    data = _oat_dataset()
    gens = {c: pickle.load(open(os.path.join(RES, "oat", f"GeneratedSamples_{c[0]}_d{c[1]:g}.pkl"), "rb")) for c in OAT_CFGS}
    ces = {c: _oat_compliance(*c)["CEs"] for c in OAT_CFGS}
    nfe = {c: _oat_nfe(*c) for c in OAT_CFGS}
    shapes = [data[p]["topology"].size[::-1] for p in problems]  # (H, W)
    ncol = 1 + len(OAT_CFGS)
    lab_w, col_w, gap = 0.42, (TEXTW - 0.42) / ncol, 0.34
    heights = [col_w * 0.9 * H / W for H, W in shapes]
    fig = plt.figure(figsize=(TEXTW, sum(heights) + gap * len(problems) + 0.35))
    gs = GridSpec(len(problems), ncol, figure=fig, height_ratios=heights, hspace=gap / np.mean(heights), wspace=0.1,
                  left=lab_w / TEXTW, right=1, top=1 - 0.35 / fig.get_figheight(), bottom=0.02)
    heads = ["SIMP reference"] + [f"{NAME[m]}, $\\delta={d:g}$" for m, d in OAT_CFGS]
    for r, p in enumerate(problems):
        s = data[p]
        gt = np.array(s["topology"]).astype(float)
        H, W = gt.shape
        ax = fig.add_subplot(gs[r, 0])
        ax.imshow(gt, cmap="gray_r", vmin=0, vmax=1, interpolation="nearest")
        _draw_problem(ax, s, H, W)
        _image_axes(ax)
        ax.text(0.5, -0.05, f"$V^\\star={s['volume fraction']:.2f}$, {W}$\\times${H}", transform=ax.transAxes,
                ha="center", va="top", fontsize=7, color=INK2)
        ax.text(-0.2, 0.5, f"problem\n{p}", transform=ax.transAxes, ha="right", va="center", fontsize=7.5)
        if r == 0:
            ax.set_title(heads[0], pad=4)
        for c, cfg in enumerate(OAT_CFGS):
            ax = fig.add_subplot(gs[r, 1 + c])
            x = gens[cfg][p][sample].squeeze().float().numpy()
            ax.imshow(x, cmap="gray_r", vmin=0, vmax=1, interpolation="nearest")
            _image_axes(ax)
            ax.text(0.5, -0.05, f"CE {ces[cfg][p, sample]:.1f}%,  {nfe[cfg][p]}/20 steps", transform=ax.transAxes,
                    ha="center", va="top", fontsize=7, color=INK2)
            if r == 0:
                ax.set_title(heads[1 + c], pad=4)
    _save(fig, "app_oat_examples.pdf")


def oat_tradeoff():
    """(a) failure rate and (b) CE* against U-Net evaluations; (c) volume-fraction error of non-failed samples."""
    rng = np.random.default_rng(0)
    st = {}
    for m, d in OAT_CFGS:
        c = _oat_compliance(m, d)
        fail_p = (c["CEs"] >= 100).mean(1)
        se = np.std([fail_p[rng.integers(0, fail_p.size, fail_p.size)].mean() for _ in range(500)]) * 100
        ev = 2 * np.mean(list(_oat_nfe(m, d).values()))
        ok = c["CEs"] < 100
        st[(m, d)] = dict(fail=100 * (1 - c["stats"]["Pass @ 1"]), se=se, ce=c["stats"]["Avg CE @ 1"], ev=ev,
                          vfe=c["VFEs"][ok])
    fig, axes = plt.subplots(1, 3, figsize=(TEXTW, 2.05), gridspec_kw={"width_ratios": [1, 1, 1.3], "wspace": 0.42})
    for ax, key, lab, title in ((axes[0], "fail", "failure rate (%)", "(a) failures (CE > 100%)"),
                                (axes[1], "ce", "CE$^*$ (%)", "(b) compliance error of non-failed")):
        for m in ("respect", "kalsea"):
            xs = [st[(m, d)]["ev"] + (-0.35 if m == "respect" else 0.35) for d in (0.3, 0.15)]
            ys = [st[(m, d)][key] for d in (0.3, 0.15)]
            yerr = [st[(m, d)]["se"] for d in (0.3, 0.15)] if key == "fail" else None
            ax.errorbar(xs, ys, yerr=yerr, color=COL[m], marker="o", ms=4.5, lw=1.2, capsize=2, elinewidth=0.8,
                        mec="white", mew=0.6, zorder=3)
        ax.plot([OAT_PAPER["evals"]], [OAT_PAPER[key]], marker="D", ms=4.5, color=INK, ls="", zorder=3)
        ax.set_xlim(17, 43)
        ax.set_xticks([20, 27.9, 40])
        ax.set_xticklabels(["20\n$\\delta=0.3$", "27.9\n$\\delta=0.15$", "40\nuncached"])
        ax.set_xlabel("U-Net evaluations per sample", labelpad=2)
        ax.set_ylabel(lab)
        ax.set_title(title, loc="left")
        _style(ax)
    axes[0].set_ylim(38.5, 45)
    axes[1].set_ylim(7.5, 11)
    ax = axes[2]
    bins = np.linspace(-5, 4, 46)
    ctr = 0.5 * (bins[1:] + bins[:-1])
    for m, d in OAT_CFGS:
        h, _ = np.histogram(st[(m, d)]["vfe"], bins=bins, density=True)
        h = np.convolve(h, [0.25, 0.5, 0.25], mode="same")
        ax.plot(ctr, h, color=COL[m], ls="-" if d == 0.15 else "--", lw=1.1)
    ax.axvline(0, color=MUTED, lw=0.6, ls=":")
    ax.set_xlabel("volume-fraction error VFE (%)")
    ax.set_ylabel("density")
    ax.set_ylim(0, None)
    ax.set_title("(c) material use of non-failed", loc="left")
    _style(ax)
    m = st[("kalsea", 0.3)]["vfe"].mean()
    ax.annotate(f"KalSea, $\\delta=0.3$\nmean {m:.2f}%".replace("-", "−"), xy=(-1.6, 0.17), xytext=(-4.9, 0.36),
                fontsize=6.5, color=INK2, arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.6))
    handles = [Line2D([], [], color=C_RSP, marker="o", ms=4, label="ReSPect"),
               Line2D([], [], color=C_KAL, marker="o", ms=4, label="KalSea"),
               Line2D([], [], color=INK, marker="D", ms=4, ls="", label="OAT uncached (paper)"),
               Line2D([], [], color=INK2, ls="-", label="$\\delta=0.15$ in (c)"),
               Line2D([], [], color=INK2, ls="--", label="$\\delta=0.3$ in (c)")]
    fig.legend(handles=handles, loc="upper center", ncol=5, frameon=False, bbox_to_anchor=(0.5, -0.1), handlelength=1.8,
               columnspacing=1.4)
    _save(fig, "app_oat_tradeoff.pdf")


# ------------------------------------------------------------------------------------------------ gate schedules
def _edm_sigma(n=2000, smax=80.0, smin=0.002, rho=7.0):
    i = np.arange(n)
    return (smax ** (1 / rho) + i / (n - 1) * (smin ** (1 / rho) - smax ** (1 / rho))) ** rho


def gate_schedules(window=50):
    fig = plt.figure(figsize=(TEXTW, 3.2))
    gs = GridSpec(2, 2, figure=fig, height_ratios=[0.5, 1], width_ratios=[1, 0.018], hspace=1.25, wspace=0.03)
    # (a) OAT: fraction of problems in which each DDIM step is computed (both methods pooled)
    ax = fig.add_subplot(gs[0, 0])
    M, lab = [], []
    for d in (0.15, 0.3):
        v, n = np.zeros(20), 0
        for m in ("respect", "kalsea"):
            for f in glob.glob(os.path.join(RES, "oat", f"{m}_d{d:g}_shard*.json")):
                for r in json.load(open(f))["records"]:
                    v[r["computed"]] += 1; n += 1
        M.append(v / n); lab.append(f"$\\delta={d:g}$  ({v.sum() / n:.1f} of 20)")
    im = ax.imshow(np.array(M), cmap="Greys", vmin=0, vmax=1, aspect="auto", interpolation="nearest")
    ax.set_yticks([0, 1]); ax.set_yticklabels(lab)
    ax.set_xticks(range(20))
    ax.set_xticks(np.arange(-0.5, 20, 1), minor=True); ax.set_yticks([-0.5, 0.5, 1.5], minor=True)
    ax.grid(which="minor", color="white", lw=1.2)
    ax.tick_params(which="minor", length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.tick_params(colors=INK2, labelcolor=INK2, length=0)
    ax.set_xlabel("DDIM step", labelpad=3)
    ax.set_title("(a) OAT: fraction of problems in which a step is computed", loc="left")
    cax = fig.add_subplot(gs[0, 1])
    cb = fig.colorbar(im, cax=cax, ticks=[0, 0.5, 1])
    cb.outline.set_visible(False); cax.tick_params(labelsize=6.5, colors=INK2, length=0)
    # (b) DiffusionPDE: local fraction of computed steps
    ax = fig.add_subplot(gs[1, 0])
    ax.axvspan(1600, 2000, color=PLANE, lw=0)
    ax.text(1800, 1.02, "PDE loss on", ha="center", va="bottom", fontsize=6.5, color=INK2)
    for d, col, ls in ((0.01, MUTED, "-"), (0.02, INK, "-")):
        r = torch.load(os.path.join(RES, "pde", f"forward_case1_kalsea_d{d:g}.pt"), weights_only=False)
        ind = np.zeros(2000); ind[r["info"]["computed"]] = 1
        loc = np.convolve(ind, np.ones(window), mode="same") / np.convolve(np.ones(2000), np.ones(window), mode="same")
        ax.plot(np.arange(2000), loc, color=col, ls=ls, lw=1.2,
                label=f"$\\delta={d:g}$  ({int(ind.sum())} of 2000 steps computed)")
    ax.set_xlim(0, 2000); ax.set_ylim(0, 1.12)
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1])
    ax.set_xlabel("Heun step")
    ax.set_ylabel(f"fraction computed\n({window}-step window)")
    ax.legend(frameon=False, loc="upper left", bbox_to_anchor=(0.0, 1.0))
    _style(ax)
    sig = _edm_sigma()
    sec = ax.secondary_xaxis("top", functions=(lambda i: np.interp(i, np.arange(2000), np.log10(sig)),
                                                lambda l: np.interp(-np.asarray(l), -np.log10(sig), np.arange(2000))))
    sec.set_xticks([1, 0, -1, -2])
    sec.set_xticklabels(["$\\sigma=10$", "1", "0.1", "0.01"])
    sec.tick_params(colors=INK2, labelcolor=INK2, labelsize=6.5)
    sec.spines["top"].set_color(AXIS)
    ax.set_title("(b) DiffusionPDE (forward problem, case 1, KalSea): local fraction of computed steps", loc="left", pad=16)
    _save(fig, "app_gate_schedules.pdf")


# ------------------------------------------------------------------------------------------------ DiffusionPDE
def _pde_run(problem, case, m, d):
    return torch.load(os.path.join(RES, "pde", f"{problem}_case{case}_{m}_d{d:g}.pt"), weights_only=False)


def pde_tradeoff():
    """Relative L2 error against network calls for the forward and inverse problems."""
    fig, axes = plt.subplots(1, 2, figsize=(TEXTW, 2.25), gridspec_kw={"wspace": 0.28})
    ref = {"forward": (3.9, 0.2), "inverse": (2.7, 0.2)}
    marker = {1: "o", 2: "s"}
    for ax, prob in zip(axes, ("forward", "inverse")):
        for m in ("respect", "kalsea"):
            for case in (1, 2):
                pts = []
                for d in (0.02, 0.01):
                    r = _pde_run(prob, case, m, d)
                    pts.append((r["info"]["nfe"], 100 * (r["err_u"] if prob == "forward" else r["err_a"])))
                pts = np.array(pts)
                ax.plot(pts[:, 0], pts[:, 1], color=COL[m], lw=1.1, marker=marker[case], ms=4.5,
                        mfc=COL[m] if case == 1 else "white", mec=COL[m], mew=1.0, zorder=3)
        if prob == "forward":  # KalSea calibration runs on forward case 1 at larger thresholds
            cal = np.array([(181, 24.6), (499, 12.2), (1041, 10.1)])
            ax.plot(cal[:, 0], cal[:, 1], color=C_KAL, lw=0.8, ls=":", marker="o", ms=3.5, mfc="white", mec=C_KAL, zorder=2)
            for (x, y), d in zip(cal, (0.3, 0.1, 0.03)):
                ax.annotate(f"$\\delta={d:g}$", (x, y), textcoords="offset points", xytext=(4, 3), fontsize=6.5, color=INK2)
        mu, sd = ref[prob]
        ax.errorbar([3999], [mu], yerr=[sd], color=INK, marker="D", ms=4, capsize=2, lw=0.8, zorder=3)
        ax.set_xscale("log")
        ax.set_xlim(150, 5200)
        ax.set_xticks([200, 500, 1000, 2000, 4000]); ax.set_xticklabels(["200", "500", "1000", "2000", "4000"])
        ax.minorticks_off()
        ax.set_ylim(0, 27 if prob == "forward" else 10)
        ax.set_xlabel("network calls per solution")
        ax.set_ylabel("relative $L^2$ error (%)")
        ax.set_title(f"({'a' if prob == 'forward' else 'b'}) {prob} problem (recover ${'u' if prob == 'forward' else 'a'}$)",
                     loc="left")
        _style(ax, grid="both")
    handles = [Line2D([], [], color=C_RSP, lw=1.1, label="ReSPect"), Line2D([], [], color=C_KAL, lw=1.1, label="KalSea"),
               Line2D([], [], color=INK2, marker="o", ls="", ms=4.5, label="case 1"),
               Line2D([], [], color=INK2, marker="s", ls="", ms=4.5, mfc="white", label="case 2"),
               Line2D([], [], color=C_KAL, ls=":", marker="o", ms=3.5, mfc="white", label="KalSea calibration"),
               Line2D([], [], color=INK, marker="D", ls="", ms=4, label="uncached (paper, mean $\\pm$ std)")]
    fig.legend(handles=handles, loc="upper center", ncol=6, frameon=False, bbox_to_anchor=(0.5, -0.04),
               handlelength=1.6, columnspacing=1.1)
    _save(fig, "app_pde_tradeoff.pdf")


def pde_examples(problem="forward", case=1):
    """Recovered fields (top) and absolute errors (bottom) of the four cached runs, with ground truth and input."""
    runs = {c: _pde_run(problem, case, *c) for c in PDE_CFGS}
    r0 = runs[PDE_CFGS[0]]
    a_GT, u_GT = r0["a_GT"].numpy(), r0["u_GT"].numpy()
    tgt, src = (u_GT, a_GT) if problem == "forward" else (a_GT, u_GT)
    tname, sname = ("u", "a") if problem == "forward" else ("a", "u")
    obs = (r0["mask_a"] if problem == "forward" else r0["mask_u"]).numpy()
    vmin, vmax = float(tgt.min()), float(tgt.max())
    errs = {c: np.abs((r["u"] if problem == "forward" else r["a"]).squeeze().numpy() - tgt) for c, r in runs.items()}
    emax = float(np.percentile(np.stack(list(errs.values())), 99.5))
    w = (TEXTW - 0.35) / 5
    fig = plt.figure(figsize=(TEXTW, 2 * w + 0.75))
    gs = GridSpec(2, 6, figure=fig, width_ratios=[1] * 5 + [0.06], wspace=0.06, hspace=0.32,
                  left=0, right=0.94, top=0.88, bottom=0.02)
    ax = fig.add_subplot(gs[0, 0])
    ax.imshow(tgt, cmap="viridis", vmin=vmin, vmax=vmax)
    _image_axes(ax); ax.set_title(f"ground truth ${tname}$\n(simulation)", fontsize=7.5)
    ax = fig.add_subplot(gs[1, 0])
    ax.imshow(src, cmap="viridis")
    ys, xs = np.nonzero(obs)
    ax.scatter(xs, ys, s=1.2, c="white", edgecolors=INK, linewidths=0.2)
    _image_axes(ax); ax.set_title(f"input: ${sname}$ at 1% sensors", fontsize=7.5)
    for k, c in enumerate(PDE_CFGS):
        r = runs[c]
        f = (r["u"] if problem == "forward" else r["a"]).squeeze().numpy()
        err = 100 * (r["err_u"] if problem == "forward" else r["err_a"])
        ax = fig.add_subplot(gs[0, 1 + k])
        im_f = ax.imshow(f, cmap="viridis", vmin=vmin, vmax=vmax)
        _image_axes(ax)
        ax.set_title(f"{NAME[c[0]]}, $\\delta={c[1]:g}$\n{err:.1f}% error, {r['info']['nfe']} calls", fontsize=7.5)
        ax = fig.add_subplot(gs[1, 1 + k])
        im_e = ax.imshow(errs[c], cmap="Reds", vmin=0, vmax=emax)
        _image_axes(ax)
        ax.set_title("absolute error", fontsize=7, color=INK2)
    for row, im, lab in ((0, im_f, f"${tname}$"), (1, im_e, f"$|{tname}-{tname}_{{\\mathrm{{GT}}}}|$")):
        cax = fig.add_subplot(gs[row, 5])
        cb = fig.colorbar(im, cax=cax)
        cb.outline.set_visible(False)
        cax.tick_params(labelsize=6.5, colors=INK2, length=2)
        cb.set_label(lab, fontsize=7)
    _save(fig, f"app_pde_{problem}_case{case}.pdf")


if __name__ == "__main__":
    what = sys.argv[1]
    if what in ("oat", "all"):
        oat_examples()
    if what in ("oat_tradeoff", "all"):
        oat_tradeoff()
    if what in ("schedules", "all"):
        gate_schedules()
    if what in ("pde_tradeoff", "all"):
        pde_tradeoff()
    if what == "all":
        pde_examples("forward", 1)
        pde_examples("inverse", 2)
    elif what == "pde":
        pde_examples(problem=sys.argv[2], case=int(sys.argv[3]))
