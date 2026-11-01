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


def oat_examples(problems=(0, 1), configs=(("respect", 0.15), ("kalsea", 0.15), ("respect", 0.3), ("kalsea", 0.3))):
    from datasets import load_dataset
    from oat_cache import TEST_PARQUET
    data = load_dataset("parquet", data_files={"test": TEST_PARQUET})["test"]
    gens, ces, nfe = {}, {}, {}
    for m, d in configs:
        g = pickle.load(open(os.path.join(RES, "oat", f"GeneratedSamples_{m}_d{d:g}.pkl"), "rb"))
        gens[(m, d)] = {p: g[p] for p in problems}
        c = pickle.load(open(os.path.join(RES, "oat", f"Compliance_{m}_d{d:g}.pkl"), "rb"))
        ces[(m, d)] = c["CEs"]
        import json
        for f in glob.glob(os.path.join(RES, "oat", f"{m}_d{d:g}_shard*.json")):
            for r in json.load(open(f))["records"]:
                if r["idx"] in problems:
                    nfe[(m, d, r["idx"])] = r["nfe"]
    ncol = 2 + len(configs)
    fig, axes = plt.subplots(len(problems), ncol, figsize=(1.75 * ncol, 1.9 * len(problems)))
    axes = np.atleast_2d(axes)
    for r, p in enumerate(problems):
        s = data[p]
        gt = np.array(s["topology"]).astype(float)
        H, W = gt.shape
        ax = axes[r, 0]
        ax.imshow(np.zeros_like(gt), cmap="gray_r", vmin=0, vmax=1)
        bc = np.array(s["boundary conditions"]); ld = np.array(s["loads"])
        if bc.size:
            ax.scatter(bc[:, 0] * max(H, W), bc[:, 1] * max(H, W), s=2, c="tab:blue", marker="s", label="supports")
        if ld.size:
            ax.quiver(ld[:, 0] * max(H, W), ld[:, 1] * max(H, W), ld[:, 2], -ld[:, 3], color="tab:red", scale=8, width=0.012)
        ax.set_title(f"problem {p}: BCs / loads\nVF = {s['volume fraction']:.2f}")
        axes[r, 1].imshow(gt, cmap="gray_r", vmin=0, vmax=1)
        axes[r, 1].set_title("SIMP ground truth")
        for c, (m, d) in enumerate(configs):
            x = gens[(m, d)][p][0].squeeze().numpy()
            axes[r, 2 + c].imshow(x, cmap="gray_r", vmin=0, vmax=1)
            name = "ReSPect" if m == "respect" else "KalSea"
            axes[r, 2 + c].set_title(f"{name}, {nfe.get((m, d, p), '?')}/20 NFE\nCE = {ces[(m, d)][p, 0]:.2f}%")
    for ax in axes.ravel():
        ax.set_xticks([]); ax.set_yticks([])
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "app_oat_examples.pdf"), bbox_inches="tight")
    print("saved app_oat_examples.pdf")


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


if __name__ == "__main__":
    what = sys.argv[1]
    if what == "oat":
        oat_examples()
    else:
        pde_examples(problem=sys.argv[2], case=int(sys.argv[3]))
