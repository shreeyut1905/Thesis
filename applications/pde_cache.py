"""Training-free caching (SeaCache / ReSPect / KalSea) for DiffusionPDE (NeurIPS 2024), bounded Navier-Stokes
(flow past a cylinder), forward problem: recover the later velocity-magnitude field u from 1% sparse observations of the
earlier field a plus the known cylinder region.  NOT YET EXECUTED (see the thesis, Chapter 6).

DiffusionPDE = EDM (EDMPrecond + SongUNet / DDPM++), joint 2-channel field (a, u) at 128x128, 2000 Heun steps with
DPS-style guidance from the observation loss and a PDE loss (the guidance gradient back-propagates through the network).

Residual caching on the SongUNet:
  Emb = enc['128x128_conv'] + enc['128x128_block0']   (128 -> 256 channels at full resolution)
  F   = all remaining encoder / decoder blocks (consuming the encoder skips)
  Out = dec['128x128_aux_norm'] -> SiLU -> dec['128x128_aux_conv']
so that h_in and h_out both have 256 channels and r = h_out - h_in is well defined.  The gate reads the timestep-
modulated input of the next block (enc['128x128_block1']: GroupNorm after conv0 with the noise embedding added),
SEA-filtered with beta = 2 and a = 1/sqrt(1+sigma^2), b = sigma/sqrt(1+sigma^2).
Each Heun step makes two network calls (at x_i, sigma_i and at the Euler point, sigma_{i+1}); the gate decides once per
step and the two calls are two residual streams with their own forecasters (like the CFG streams of Wan2.1).
On skipped steps the guidance gradient flows through Emb and Out only (the forecast residual is a constant).
PCGS: at a computed step b the gap is replayed with the smoothed residuals through the full guided Heun update
(this script owns the loop), and step b is then recomputed from the corrected latent with the observed residuals.
"""
import argparse, os, pickle, sys, time
import numpy as np
import torch
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
PDE_ROOT = os.path.expanduser("~/mech_apps/DiffusionPDE")
sys.path.insert(0, PDE_ROOT)
from cachecore import SEAGate, Stream  # noqa: E402
sys.path.insert(0, os.path.join(PDE_ROOT, "scripts"))
from generate_ns_bounded import random_index_and_cylinder, cylinder_index, get_ns_bounded_loss  # noqa: E402

silu = torch.nn.functional.silu


class SplitEDM:
    """EDMPrecond + SongUNet evaluated as Emb -> F -> Out."""

    def __init__(self, net):
        self.net, self.u = net, net.model
        self.enc_items = list(self.u.enc.items())
        self.dec_items = list(self.u.dec.items())
        res = self.u.img_resolution
        self.k_conv, self.k_b0, self.k_b1 = f"{res}x{res}_conv", f"{res}x{res}_block0", f"{res}x{res}_block1"
        assert self.enc_items[0][0] == self.k_conv and self.enc_items[1][0] == self.k_b0

    def precond(self, sigma):
        sd = self.net.sigma_data
        c_skip = sd ** 2 / (sigma ** 2 + sd ** 2)
        c_out = sigma * sd / (sigma ** 2 + sd ** 2).sqrt()
        c_in = 1 / (sd ** 2 + sigma ** 2).sqrt()
        return c_skip, c_out, c_in, sigma.log() / 4

    def emb(self, c_noise):
        u = self.u
        e = u.map_noise(c_noise.flatten())
        e = e.reshape(e.shape[0], 2, -1).flip(1).reshape(*e.shape)
        return silu(u.map_layer1(silu(u.map_layer0(e))))

    def embed(self, x, sigma):
        c_skip, c_out, c_in, c_noise = self.precond(sigma)
        e = self.emb(c_noise)
        h0 = self.u.enc[self.k_conv]((c_in * x).float())
        h1 = self.u.enc[self.k_b0](h0, e)
        return dict(e=e, skips=[h0, h1], h_in=h1, c_skip=c_skip, c_out=c_out)

    def gate_input(self, ctx):
        blk = self.u.enc[self.k_b1]
        x = blk.conv0(silu(blk.norm0(ctx["h_in"])))
        params = blk.affine(ctx["e"]).unsqueeze(2).unsqueeze(3)
        if blk.adaptive_scale:
            scale, shift = params.chunk(2, dim=1)
            return torch.addcmul(shift, blk.norm1(x), scale + 1)
        return blk.norm1(x + params)

    def body(self, ctx):
        x, skips, aux = ctx["h_in"], list(ctx["skips"]), None
        for name, block in self.enc_items[2:]:
            if "aux" in name:
                raise NotImplementedError("standard encoder expected (ddpmpp)")
            x = block(x, ctx["e"]) if hasattr(block, "affine") else block(x)
            skips.append(x)
        for name, block in self.dec_items:
            if "aux_norm" in name:
                return x                            # h_out: input of the output layers
            if "aux" in name:
                raise NotImplementedError
            if x.shape[1] != block.in_channels:
                x = torch.cat([x, skips.pop()], dim=1)
            x = block(x, ctx["e"])
        raise RuntimeError("no output layers found")

    def out(self, x_in, h_out, ctx):
        res = self.u.img_resolution
        F_x = self.u.dec[f"{res}x{res}_aux_conv"](silu(self.u.dec[f"{res}x{res}_aux_norm"](h_out)))
        return ctx["c_skip"] * x_in + ctx["c_out"] * F_x.float()


def run(split, cfg, method, delta, N, seed, device, a_GT, u_GT, mask_a, mask_u):
    torch.manual_seed(seed)
    net = split.net
    x_next = torch.randn([1, net.img_channels, net.img_resolution, net.img_resolution], device=device)
    smin, smax, rho = max(0.002, net.sigma_min), min(80.0, net.sigma_max), 7
    idx = torch.arange(N, dtype=torch.float64, device=device)
    sig = (smax ** (1 / rho) + idx / (N - 1) * (smin ** (1 / rho) - smax ** (1 / rho))) ** rho
    sig = torch.cat([sig, torch.zeros_like(sig[:1])])
    x_next = x_next.to(torch.float64) * sig[0]
    use_cache = method != "reference"
    gate = SEAGate(delta, N, power_exp=2.0, dims=(-1, -2)) if use_cache else None
    streams = [Stream(method, N), Stream(method, N)] if use_cache else None
    nfe, computed = 0, []

    def call(x, s, mode, k, i, r_fixed=None):
        """mode: 'compute' (run F, return residual), 'skip' (forecast), 'fixed' (given residual)."""
        ctx = split.embed(x, s.reshape(-1, 1, 1, 1).float())
        if mode == "compute":
            h_out = split.body(ctx)
            r = h_out - ctx["h_in"]
        elif mode == "skip":
            r = streams[k].forecast(i, ctx=None).to(ctx["h_in"].dtype)
        else:
            r = r_fixed
        return split.out(x.float(), ctx["h_in"] + r, ctx).to(torch.float64), r, ctx

    def heun_guided(i, x_cur_in, mode, r_fix=(None, None)):
        """One guided Heun step of DiffusionPDE; returns x_next and the residuals of the two calls."""
        x_cur = x_cur_in.detach().clone().requires_grad_(True)
        s_cur, s_nxt = sig[i], sig[i + 1]
        x_N, r0, ctx0 = call(x_cur, s_cur, mode, 0, i, r_fix[0])
        d_cur = (x_cur - x_N) / s_cur
        x_nx = x_cur + (s_nxt - s_cur) * d_cur
        r1 = None
        if i < N - 1:
            x_N, r1, _ = call(x_nx, s_nxt, mode, 1, i, r_fix[1])
            d_pr = (x_nx - x_N) / s_nxt
            x_nx = x_cur + (s_nxt - s_cur) * (0.5 * d_cur + 0.5 * d_pr)
        a_N, u_N = (x_N[:, 0] * 10).unsqueeze(0), (x_N[:, 1] * 10).unsqueeze(0)
        pde, la, lu = get_ns_bounded_loss(a_N, u_N, a_GT, u_GT, mask_a, mask_u, device=device)
        L_pde, L_a, L_u = torch.norm(pde, 2) / (128 * 128), torch.norm(la, 2), torch.norm(lu, 2)
        ga = torch.autograd.grad(L_a, x_cur, retain_graph=True)[0]
        gu = torch.autograd.grad(L_u, x_cur, retain_graph=True)[0]
        gp = torch.autograd.grad(L_pde, x_cur)[0]
        za, zu, zp = cfg["zeta_obs_a"], cfg["zeta_obs_u"], cfg["zeta_pde"]
        if i <= 0.8 * N:
            x_nx = x_nx - za * ga - zu * gu
        else:
            x_nx = x_nx - 0.1 * (za * ga + zu * gu) - zp * gp
        return x_nx.detach(), (r0.detach(), None if r1 is None else r1.detach()), ctx0

    stats = dict(n_corr=0, gap_len=[])
    for i in range(N):
        x_cur = x_next
        if use_cache:
            with torch.no_grad():
                ctx = split.embed(x_cur.float(), sig[i].reshape(-1, 1, 1, 1).float())
                s = float(sig[i]); a, b = 1 / np.sqrt(1 + s * s), s / np.sqrt(1 + s * s)
                compute = gate.decide(i, gate.filtered(split.gate_input(ctx), a, b))
        else:
            compute = True
        if not compute:
            x_next, _, _ = heun_guided(i, x_cur, "skip")
            for k, st in enumerate(streams):          # remember the latent at the first skipped step
                if st.gap and "x" not in (st.gap[0]["ctx"] or {}):
                    st.gap[0]["ctx"] = dict(x=x_cur)
            continue
        x_next, (r0, r1), _ = heun_guided(i, x_cur, "compute")
        nfe += 1 if i == N - 1 else 2
        computed.append(i)
        if use_cache:
            streams[0].observe(i, r0)
            if r1 is not None:
                streams[1].observe(i, r1)
            sm = [st.smooth_gap(i) for st in streams]
            if sm[0] is not None:                       # PCGS: replay the gap with smoothed residuals
                gap0 = sm[0]
                gap1 = sm[1] if sm[1] is not None else [dict(step=g["step"], r=g["pred"]) for g in streams[1].gap]
                xr = streams[0].gap[0]["ctx"]["x"]
                for g0, g1 in zip(gap0, gap1):
                    xr, _, _ = heun_guided(g0["step"], xr, "fixed", (g0["r"], g1["r"]))
                stats["gap_len"].append(len(gap0))
                if not torch.equal(xr, x_cur):
                    stats["n_corr"] += 1
                    x_next, _, ctx_new = heun_guided(i, xr, "fixed", (r0, r1))   # continue from x_b'
                    if i < N - 1:
                        with torch.no_grad():
                            s = float(sig[i]); a, b = 1 / np.sqrt(1 + s * s), s / np.sqrt(1 + s * s)
                            gate.refresh(gate.filtered(split.gate_input(ctx_new), a, b))
            for st in streams:
                st.close(i)
    a_f, u_f = (x_next[:, 0] * 10).detach(), (x_next[:, 1] * 10).detach()
    return a_f.cpu(), u_f.cpu(), dict(nfe=nfe, computed=computed, **stats)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--case", type=int, default=1)
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--methods", type=str, default="reference,seacache,respect,kalsea_nopcgs,kalsea")
    ap.add_argument("--deltas", type=str, default="0.05,0.1")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=str, default=os.path.expanduser("~/mech_apps/results/pde"))
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    device = torch.device("cuda")
    cfg = dict(zeta_obs_a=0.5, zeta_obs_u=0.5, zeta_pde=1.0)            # configs/ns-bounded-forward.yaml
    with open(os.path.join(PDE_ROOT, "pretrained-models/pretrained-ns-bounded.pkl"), "rb") as f:
        net = pickle.load(f)["ema"].to(device).eval()
    split = SplitEDM(net)
    d = os.path.join(PDE_ROOT, f"data/testing/ns-bounded/{args.case}")
    v = np.load(os.path.join(d, "v.npy"))
    a_GT = torch.tensor(v[0, :, :, 4], dtype=torch.float64, device=device)
    u_GT = torch.tensor(v[0, :, :, 8], dtype=torch.float64, device=device)
    center = (int(np.load(os.path.join(d, "cx.npy"))[0]), int(np.load(os.path.join(d, "cy.npy"))[0]))
    radius = int(np.load(os.path.join(d, "r.npy"))[0])
    mask_a, _ = random_index_and_cylinder(center, radius, 0.01, 128, seed=1, device=device)
    mask_u, _ = cylinder_index(center, radius, 128, device=device)
    for method in args.methods.split(","):
        for delta in ([0.0] if method == "reference" else [float(x) for x in args.deltas.split(",")]):
            torch.cuda.synchronize(); t0 = time.time()
            a_f, u_f, info = run(split, cfg, method, delta, args.steps, args.seed, device, a_GT, u_GT, mask_a, mask_u)
            torch.cuda.synchronize(); lat = time.time() - t0
            err_u = float(torch.norm(u_f.squeeze() - u_GT.cpu()) / torch.norm(u_GT.cpu()))
            err_a = float(torch.norm(a_f.squeeze() - a_GT.cpu()) / torch.norm(a_GT.cpu()))
            tag = f"case{args.case}_{method}_d{delta:g}"
            torch.save(dict(a=a_f, u=u_f, info=info, latency=lat, err_a=err_a, err_u=err_u), os.path.join(args.out, tag + ".pt"))
            print(f"{tag}: nfe={info['nfe']} lat={lat:.1f}s err_a={err_a:.4f} err_u={err_u:.4f} corr={info['n_corr']}", flush=True)


if __name__ == "__main__":
    main()
