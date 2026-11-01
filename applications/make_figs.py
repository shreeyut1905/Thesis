"""Figures of the worked examples for Chapter 6 of the thesis (run after generation and evaluation)."""
import glob, os, pickle, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
RES = os.path.expanduser("~/mech_apps/results")
FIG = os.path.expanduser("~/BTP/thesis/figures")
plt.rcParams.update({"font.size": 8, "font.family": "serif"})


def oat_examples(problems=(0, 1), configs=(("respect", 0.15), ("kalsea", 0.15), ("respect", 0.3), ("kalsea", 0.3)), out="app_oat_examples.pdf"):
    """Problem set-up, SIMP ground truth and the first of the ten samples of each cached configuration.

    Coordinates follow OAT's RunComplianceTest.py: the mesh has shape (H, W) = topology.size[::-1], the first
    coordinate of a load / support runs along the rows (length H / max(H, W)), the second along the columns.
    """
    import json
    from datasets import load_dataset
    from oat_cache import TEST_PARQUET
    data = load_dataset("parquet", data_files={"test": TEST_PARQUET})["test"]
    gens, ces, nfe = {}, {}, {}
    for m, d in configs:
        g = pickle.load(open(os.path.join(RES, "oat", f"GeneratedSamples_{m}_d{d:g}.pkl"), "rb"))
        gens[(m, d)] = {p: g[p] for p in problems}
        f = os.path.join(RES, "oat", f"Compliance_{m}_d{d:g}.pkl")
        ces[(m, d)] = pickle.load(open(f, "rb"))["CEs"] if os.path.exists(f) else None
        for f in glob.glob(os.path.join(RES, "oat", f"{m}_d{d:g}_shard*.json")):
            for r in json.load(open(f))["records"]:
                if r["idx"] in problems:
                    nfe[(m, d, r["idx"])] = r["nfe"]
    ncol = 2 + len(configs)
    widths = []
    for p in problems:
        W, H = data[p]["topology"].size
        widths.append(W / H)
    fig, axes = plt.subplots(len(problems), ncol, figsize=(7.2, 1.45 * len(problems) + 0.4),
                             gridspec_kw={"wspace": 0.1, "hspace": 0.05})
    axes = np.atleast_2d(axes)
    for r, p in enumerate(problems):
        s = data[p]
        gt = np.array(s["topology"]).astype(float)
        H, W = gt.shape
        L = max(H, W)
        ld = np.array(s["loads"]); bc = np.array(s["boundary conditions"])
        ax = axes[r, 0]
        ax.imshow(np.zeros_like(gt), cmap="gray_r", vmin=0, vmax=1)
        ax.add_patch(plt.Rectangle((-0.5, -0.5), W, H, fill=False, ec=INK2, lw=0.6))
        if bc.size:
            ax.scatter(bc[:, 1] * L, bc[:, 0] * L, s=16, c=INK, marker="^", zorder=3, clip_on=False)
        if ld.size:
            F = ld[:, 2:]
            n = np.linalg.norm(F, axis=1).max()
            k = 0.18 * L / n if n > 0 else 0
            ax.quiver(ld[:, 1] * L, ld[:, 0] * L, F[:, 1] * k, F[:, 0] * k, angles="xy", scale_units="xy", scale=1,
                      color="#e34948", width=0.012 if len(ld) < 10 else 0.005, zorder=4)
        ax.set_xlim(-0.08 * W, 1.08 * W); ax.set_ylim(1.08 * H, -0.08 * H)
        ax.set_title(f"problem {p}: set-up\n$V^\\star$ = {s['volume fraction']:.2f}\n{W}$\\times${H} grid", fontsize=7)
        axes[r, 1].imshow(gt, cmap="gray_r", vmin=0, vmax=1)
        axes[r, 1].set_title("SIMP\n(ground truth)\n", fontsize=7)
        for c, (m, d) in enumerate(configs):
            x = gens[(m, d)][p][0].squeeze().float().numpy()
            axes[r, 2 + c].imshow(x, cmap="gray_r", vmin=0, vmax=1)
            name = "ReSPect" if m == "respect" else "KalSea"
            ce = f"{ces[(m, d)][p, 0]:.1f}%" if ces[(m, d)] is not None else "pending"
            axes[r, 2 + c].set_title(f"{name}, $\\delta$={d:g}\n{nfe.get((m, d, p), '?')}/20 steps\nCE {ce}", fontsize=7)
    for ax in axes.ravel():
        ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_color("#c3c2b7"); sp.set_linewidth(0.5)
    for ax in axes[:, 0]:
        for sp in ax.spines.values():
            sp.set_visible(False)
    fig.savefig(os.path.join(FIG, out), bbox_inches="tight")
    print("saved", out)


def pde_examples(problem="forward", case=1, configs=(("respect",), ("kalsea",))):
    fs = sorted(glob.glob(os.path.join(RES, "pde", f"{problem}_case{case}_*.pt")))
    runs = [torch.load(f, weights_only=False) for f in fs]
    runs = sorted(runs, key=lambda r: (r["method"], r["delta"]))
    r0 = runs[0]
    a_GT, u_GT = r0["a_GT"].numpy(), r0["u_GT"].numpy()
    obs = (r0["mask_a"] if problem == "forward" else r0["mask_u"]).numpy()
    ncol = 2 + len(runs)
    fig, axes = plt.subplots(2, ncol, figsize=(1.75 * ncol, 3.6))
    tgt_name = "u" if problem == "forward" else "a"
    tgt = u_GT if problem == "forward" else a_GT
    vmin, vmax = tgt.min(), tgt.max()
    src = a_GT if problem == "forward" else u_GT
    axes[0, 0].imshow(src, cmap="viridis"); ys, xs = np.nonzero(obs)
    axes[0, 0].scatter(xs, ys, s=0.4, c="w"); axes[0, 0].set_title(f"observed field ({'a' if problem == 'forward' else 'u'})\n+ 1% sensors")
    axes[0, 1].imshow(tgt, cmap="viridis", vmin=vmin, vmax=vmax); axes[0, 1].set_title(f"ground truth {tgt_name}")
    axes[1, 0].axis("off"); axes[1, 1].axis("off")
    for c, r in enumerate(runs):
        f = (r["u"] if problem == "forward" else r["a"]).squeeze().numpy()
        err = r["err_u"] if problem == "forward" else r["err_a"]
        name = "ReSPect" if r["method"] == "respect" else "KalSea"
        axes[0, 2 + c].imshow(f, cmap="viridis", vmin=vmin, vmax=vmax)
        axes[0, 2 + c].set_title(f"{name}, $\\delta$={r['delta']:g}\n{r['info']['nfe']} evals, err {100 * err:.1f}%")
        axes[1, 2 + c].imshow(np.abs(f - tgt), cmap="magma", vmin=0, vmax=0.3 * (vmax - vmin))
        axes[1, 2 + c].set_title("|error|")
    for ax in axes.ravel():
        ax.set_xticks([]); ax.set_yticks([])
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, f"app_pde_{problem}_case{case}.pdf"), bbox_inches="tight")
    print("saved", f"app_pde_{problem}_case{case}.pdf")


# ---- result figures (colours: categorical slots 1/2 of the reference palette; KalSea = blue, ReSPect = orange) ----
C_KAL, C_RSP = "#2a78d6", "#eb6834"
INK, INK2, MUTED, GRID = "#0b0b0b", "#52514e", "#898781", "#e1e0d9"
METHOD = {"kalsea": ("KalSea", C_KAL), "respect": ("ReSPect", C_RSP)}
LS = {0.15: "-", 0.3: "--", 0.01: "-", 0.02: "--"}


def _style(ax):
    ax.grid(True, color=GRID, lw=0.5)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color("#c3c2b7")
    ax.tick_params(colors=INK2, labelcolor=INK2)


def _oat_configs():
    out = []
    for d in (0.15, 0.3):
        for m in ("respect", "kalsea"):
            f = os.path.join(RES, "oat", f"Compliance_{m}_d{d:g}.pkl")
            if os.path.exists(f):
                out.append((m, d, pickle.load(open(f, "rb"))))
    return out


def _fmt(x):
    s = f"{x:.2f}"
    return "0.00" if s in ("-0.00", "0.00") else s.replace("-", "\u2212")


def oat_distributions():
    """(a) empirical CDF of the compliance error over all samples; (b) volume-fraction error of non-failed samples."""
    cfgs = _oat_configs()
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(6.3, 2.5))
    for m, d, c in cfgs:
        name, col = METHOD[m]
        ce = np.sort(c["CEs"].ravel())
        y = np.arange(1, ce.size + 1) / ce.size
        a1.plot(ce, y, color=col, ls=LS[d], lw=1.2, label=f"{name}, $\\delta$={d:g}")
        ok = c["CEs"] < 100
        v = c["VFEs"][ok]
        a2.hist(v, bins=np.linspace(-6, 6, 61), density=True, histtype="step",
                color=col, ls=LS[d], lw=1.1, label=f"{name}, $\\delta$={d:g} (mean {_fmt(v.mean())}%)")
    a1.set_xscale("symlog", linthresh=1)
    a1.set_xlim(-100, 1e8)
    a1.set_xticks([-100, 0, 1, 10, 100, 1e4, 1e6, 1e8])
    a1.set_xticklabels(["$-100$", "0", "1", "10", "100", "$10^4$", "$10^6$", "$10^8$"])
    a1.minorticks_off()
    a1.axvline(100, color=INK2, lw=0.8, ls=":")
    a1.text(80, 0.93, "failure:\nCE > 100%", color=INK2, fontsize=7, va="top", ha="right")
    a1.set_xlabel("compliance error CE (%)")
    a1.set_ylabel("fraction of samples")
    a1.set_title("(a) distribution of CE (50,000 samples each)", fontsize=8)
    a1.legend(frameon=False, fontsize=6.5, loc="lower right")
    a2.axvline(0, color=INK2, lw=0.8, ls=":")
    a2.set_xlabel("volume-fraction error VFE (%)")
    a2.set_ylabel("density")
    a2.set_title("(b) VFE of non-failed samples", fontsize=8)
    a2.set_ylim(0, 0.75)
    a2.legend(frameon=False, fontsize=6, loc="upper left")
    for ax in (a1, a2):
        _style(ax)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "app_oat_distributions.pdf"), bbox_inches="tight")
    print("saved app_oat_distributions.pdf")


def gate_schedules(window=50):
    """(a) OAT: fraction of problems in which each DDIM step is computed; (b) DiffusionPDE: local fraction of computed steps."""
    import json
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(6.3, 2.3), gridspec_kw={"width_ratios": [1, 1.25]})
    shades = {0.15: "#a8a69f", 0.3: INK2, 0.01: "#a8a69f", 0.02: INK2}
    wbar = 0.4
    for k, d in enumerate((0.15, 0.3)):
        M, n = np.zeros(20), 0
        for m in ("respect", "kalsea"):
            for f in glob.glob(os.path.join(RES, "oat", f"{m}_d{d:g}_shard*.json")):
                for r in json.load(open(f))["records"]:
                    M[r["computed"]] += 1; n += 1
        x = np.arange(20) + (k - 0.5) * wbar
        a1.bar(x, M / n, width=wbar - 0.06, color=shades[d], label=f"$\\delta$={d:g} ({M.sum() / n:.1f} of 20 computed)")
    a1.set_xticks(range(0, 20, 2))
    a1.set_xlabel("DDIM step")
    a1.set_ylabel("fraction computed")
    a1.set_title("(a) OAT (5000 problems, both methods)", fontsize=8)
    a1.set_ylim(0, 1.45)
    a1.set_yticks([0, 0.25, 0.5, 0.75, 1])
    a1.legend(frameon=False, fontsize=6.5, loc="upper left")
    for d in (0.01, 0.02):
        r = torch.load(os.path.join(RES, "pde", f"forward_case1_kalsea_d{d:g}.pt"), weights_only=False)
        ind = np.zeros(2000); ind[r["info"]["computed"]] = 1
        loc = np.convolve(ind, np.ones(window), mode="same") / np.convolve(np.ones(2000), np.ones(window), mode="same")
        a2.plot(np.arange(2000), loc, color=shades[d], lw=1.1, label=f"$\\delta$={d:g} ({int(ind.sum())} of 2000 computed)")
    a2.axvline(1600, color=INK2, lw=0.8, ls=":")
    a2.text(1590, 0.06, "PDE loss on", color=INK2, fontsize=7, ha="right")
    a2.set_ylim(0, 1.45)
    a2.set_yticks([0, 0.25, 0.5, 0.75, 1])
    a2.set_xlabel("Heun step")
    a2.set_ylabel(f"fraction computed ({window}-step window)")
    a2.set_title("(b) DiffusionPDE, forward case 1, KalSea", fontsize=8)
    a2.legend(frameon=False, fontsize=6.5, loc="upper left")
    for ax in (a1, a2):
        _style(ax)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "app_gate_schedules.pdf"), bbox_inches="tight")
    print("saved app_gate_schedules.pdf")


def pde_tradeoff():
    """Relative L2 error vs. network calls for the forward and inverse problems."""
    from matplotlib.lines import Line2D
    fig, axes = plt.subplots(1, 2, figsize=(6.3, 2.6), sharey=False)
    ref = {"forward": (3.9, 0.2), "inverse": (2.7, 0.2)}
    marker = {1: "o", 2: "s"}
    for ax, prob in zip(axes, ("forward", "inverse")):
        for m in ("respect", "kalsea"):
            name, col = METHOD[m]
            for case in (1, 2):
                pts = []
                for d in (0.01, 0.02):
                    r = torch.load(os.path.join(RES, "pde", f"{prob}_case{case}_{m}_d{d:g}.pt"), weights_only=False)
                    err = r["err_u"] if prob == "forward" else r["err_a"]
                    pts.append((r["info"]["nfe"], 100 * err))
                pts = np.array(pts)
                ax.plot(pts[:, 0], pts[:, 1], color=col, lw=1.0, marker=marker[case], ms=5,
                        mfc=col if case == 1 else "white", mec=col)
        if prob == "forward":  # KalSea calibration runs on forward case 1 (larger thresholds)
            cal = np.array([(1041, 10.1), (499, 12.2), (181, 24.6)])
            ax.plot(cal[:, 0], cal[:, 1], color=C_KAL, lw=0.8, ls=":", marker="o", ms=4, mfc="white", mec=C_KAL)
            for (x, y), d in zip(cal, (0.03, 0.1, 0.3)):
                ax.annotate(f"$\\delta$={d:g}", (x, y), textcoords="offset points", xytext=(4, 3), fontsize=6, color=INK2)
        mu, sd = ref[prob]
        ax.errorbar([3999], [mu], yerr=[sd], color=INK, marker="D", ms=4, capsize=2, lw=0.8)
        ax.set_xscale("log")
        ax.set_xlim(150, 5000)
        ax.set_xticks([200, 500, 1000, 2000, 4000])
        ax.set_xticklabels(["200", "500", "1000", "2000", "4000"])
        ax.minorticks_off()
        ax.set_ylim(0, None)
        ax.set_xlabel("network calls per solution")
        ax.set_ylabel("relative $L^2$ error (%)")
        ax.set_title(f"({'a' if prob == 'forward' else 'b'}) {prob} problem", fontsize=8)
        _style(ax)
    handles = [Line2D([], [], color=C_KAL, lw=1, label="KalSea"), Line2D([], [], color=C_RSP, lw=1, label="ReSPect"),
               Line2D([], [], color=INK2, marker="o", ls="", ms=5, label="case 1"),
               Line2D([], [], color=INK2, marker="s", ls="", ms=5, mfc="white", label="case 2"),
               Line2D([], [], color=C_KAL, ls=":", marker="o", ms=4, mfc="white", label="KalSea, calibration (case 1)"),
               Line2D([], [], color=INK, marker="D", ls="", ms=4, label="uncached sampler (paper: mean $\\pm$ std, 1000 scenes)")]
    fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False, fontsize=6.5, bbox_to_anchor=(0.5, -0.06))
    fig.tight_layout(rect=(0, 0.1, 1, 1))
    fig.savefig(os.path.join(FIG, "app_pde_tradeoff.pdf"), bbox_inches="tight")
    print("saved app_pde_tradeoff.pdf")


if __name__ == "__main__":
    what = sys.argv[1]
    if what == "oat":
        oat_examples()
    elif what == "oat_dist":
        oat_distributions()
    elif what == "schedules":
        gate_schedules()
    elif what == "pde_tradeoff":
        pde_tradeoff()
    else:
        pde_examples(problem=sys.argv[2], case=int(sys.argv[3]))
