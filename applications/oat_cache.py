"""Training-free caching (SeaCache / ReSPect / KalSea) for Optimize Any Topology (OAT, NeurIPS 2025).

OAT = neural-field autoencoder (OpenTO/NFAE) + conditional latent diffusion U-Net (OpenTO/LDM, CTOPUNet, a diffusers
UNet2DModel with a boundary-condition / load / volume-fraction problem encoder), v-prediction, DDIM sampling.

Residual caching on the U-Net:  Emb = conv_in,  F = down/mid/up blocks (with their skip connections),
Out = conv_norm_out -> SiLU -> conv_out.  Both conv_in and the last up block have 320 channels at 64x64, so the
block-stack residual r = h_out - h_in is well defined.  Gate input: the timestep/problem-modulated input of the first
ResNet block, conv1(SiLU(GroupNorm(h_in))) + time_emb_proj(SiLU(emb)), SEA-filtered with beta = 2 over (H, W) and the
VP coefficients a = sqrt(alpha_bar_t), b = sqrt(1 - alpha_bar_t) of the DDIM step.

PCGS: the DDIM update is affine in the network output; since this script owns the sampling loop, the skipped gap is
re-integrated with the scheduler's own step() and sampling continues from the corrected latent (mathematically the same
as returning the velocity offset of Eq. (9) in the KalSea chapter).
"""
import argparse, copy, json, os, sys, time
import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.expanduser("~/mech_apps/OAT"))
from cachecore import SEAGate, Stream  # noqa: E402
from util_seacache import ab_from_scheduler  # noqa: E402
from OAT.Models import NFAE, CTOPUNet  # noqa: E402
from OAT.Pipelines import DDIMPipeline  # noqa: E402
from OAT.DataUtils import OpenTO, NFAECollator, DiffusionCollator  # noqa: E402
from datasets import load_dataset  # noqa: E402

TEST_PARQUET = os.path.expanduser(
    "~/mech_apps/hf_cache/hub/datasets--OpenTO--OpenTO/snapshots/c553a645a10feb46d4906c899fd354c1f8f2db13/data/"
    "test-00000-of-00001.parquet")


# ------------------------------------------------------------------------------------------------ U-Net split
def embed(model, t, cond, B, device):
    ts = torch.as_tensor(t, device=device).reshape(1).long().expand(B)
    emb = model.time_embedding(model.time_proj(ts).to(model.dtype))
    pe = model.problem_embedding(model.problem_encoder(**{k: v for k, v in cond.items() if k != "unconditioned"}))
    return emb + pe


def gate_input(model, h_in, emb):
    rn = model.down_blocks[0].resnets[0]
    x = rn.conv1(rn.nonlinearity(rn.norm1(h_in)))
    return x + rn.time_emb_proj(rn.nonlinearity(emb))[:, :, None, None]


def body(model, h, emb):
    res = (h,)
    for blk in model.down_blocks:
        h, r = blk(hidden_states=h, temb=emb)
        res += r
    h = model.mid_block(h, emb)
    for blk in model.up_blocks:
        r = res[-len(blk.resnets):]
        res = res[: -len(blk.resnets)]
        h = blk(h, r, emb)
    return h


def out_layers(model, h):
    return model.conv_out(model.conv_act(model.conv_norm_out(h)))


# ------------------------------------------------------------------------------------------------ sampler
@torch.no_grad()
def sample(model, sched_base, cond, x0, N, method, delta=0.0):
    sched = copy.deepcopy(sched_base)
    sched.set_timesteps(N)
    device, B = x0.device, x0.shape[0]
    x = x0.clone()
    use_cache = method not in ("reference", "fewer_steps")
    gate = SEAGate(delta, N, power_exp=2.0, dims=(-1, -2)) if use_cache else None
    st = Stream(method, N) if use_cache else None
    computed, nfe = [], 0
    stats = dict(n_corr=0, gap_len=[])
    ts = list(sched.timesteps)
    for i, t in enumerate(ts):
        emb = embed(model, t, cond, B, device)
        h_in = model.conv_in(x.to(model.dtype))
        if use_cache:
            a, b = ab_from_scheduler(sched, int(t), mode="vp")
            compute = gate.decide(i, gate.filtered(gate_input(model, h_in, emb), a, b))
        else:
            compute = True
        if compute:
            h_out = body(model, h_in, emb)
            r = h_out - h_in
            nfe += 1
            computed.append(i)
            if use_cache:
                st.observe(i, r)
                sm = st.smooth_gap(i)
                if sm is not None:                        # PCGS: re-integrate the gap with the smoothed residuals
                    xr = sm[0]["ctx"]["x"]
                    for g in sm:
                        tj = ts[g["step"]]
                        hj = model.conv_in(xr.to(model.dtype))
                        vj = out_layers(model, hj + g["r"])
                        xr = sched.step(vj, tj, xr).prev_sample
                    stats["gap_len"].append(len(sm))
                    if not torch.equal(xr, x):
                        stats["n_corr"] += 1
                        x = xr                            # continue from the corrected latent x_b'
                        h_in = model.conv_in(x.to(model.dtype))
                        if i < N - 1:                     # refresh the gate reference from x_b'
                            a, b = ab_from_scheduler(sched, int(t), mode="vp")
                            gate.refresh(gate.filtered(gate_input(model, h_in, emb), a, b))
                st.close(i)
            v = out_layers(model, h_in + r)
        else:
            pred = st.forecast(i, ctx=dict(x=x))
            v = out_layers(model, h_in + pred)
        x = sched.step(v, t, x).prev_sample
    return x, dict(nfe=nfe, computed=computed, **stats)


def decode(model, nfae, latents, ae_batch):
    z = latents * model.latent_scale - model.latent_shift
    if nfae.use_vq:
        z, _, _ = nfae.quantizer(z)
    phi = nfae.decoder(z.to(torch.bfloat16))
    return nfae.renderer(phi, ae_batch["gt_coord"], ae_batch["gt_cell"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--indices", type=str, default="0,1")
    ap.add_argument("--seeds", type=str, default="0")
    ap.add_argument("--steps", type=int, default=50)
    ap.add_argument("--deltas", type=str, default="0.1,0.2,0.3")
    ap.add_argument("--fewer", type=str, default="25,15,10")
    ap.add_argument("--methods", type=str, default="seacache,respect,kalsea_nopcgs,kalsea")
    ap.add_argument("--out", type=str, default=os.path.expanduser("~/mech_apps/results/oat"))
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    device = torch.device("cuda")
    torch.backends.cuda.matmul.allow_tf32 = True

    data = load_dataset("parquet", data_files={"test": TEST_PARQUET})["test"]
    dataset = OpenTO(data, full_sampling=True)
    collator = DiffusionCollator(unconditional_prob=0.0, inference=True,
                                 inference_collator=NFAECollator(zero_centering=False, full_sampling=True, coords_only=True))
    nfae = NFAE.from_pretrained("OpenTO/NFAE").to(device=device, dtype=torch.bfloat16).eval()
    model = CTOPUNet.from_pretrained("OpenTO/LDM").to(device=device, dtype=torch.bfloat16).eval()
    sched_base = DDIMPipeline().InferenceScheduler

    runs = [("reference", args.steps, 0.0)]
    runs += [("fewer_steps", int(n), 0.0) for n in args.fewer.split(",") if n]
    for d in [float(v) for v in args.deltas.split(",") if v]:
        runs += [(m, args.steps, d) for m in args.methods.split(",") if m]

    for idx in [int(v) for v in args.indices.split(",")]:
        cond, ae_batch = collator([dataset[idx]])
        cond, ae_batch = cond.to(device), ae_batch.to(device)
        cond["unconditioned"] = False
        for seed in [int(s) for s in args.seeds.split(",")]:
            g = torch.Generator(device="cpu").manual_seed(seed)
            x0 = torch.randn((1, model.conv_in.in_channels, model.sample_size, model.sample_size), generator=g).to(device)
            for method, N, delta in runs:
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    torch.cuda.synchronize(); t0 = time.time()
                    lat, info = sample(model, sched_base, cond, x0, N, method, delta)
                    torch.cuda.synchronize(); lat_s = time.time() - t0
                    field = decode(model, nfae, lat, ae_batch)[0].float().cpu()
                tag = f"idx{idx}_seed{seed}_{method}_N{N}_d{delta:g}"
                torch.save(dict(field=field, latent=lat.float().cpu(), info=info, latency=lat_s, idx=idx, seed=seed,
                                method=method, N=N, delta=delta), os.path.join(args.out, tag + ".pt"))
                print(f"{tag}: nfe={info['nfe']} lat={lat_s:.2f}s corr={info.get('n_corr', 0)} "
                      f"computed={info['computed']}", flush=True)


if __name__ == "__main__":
    main()
