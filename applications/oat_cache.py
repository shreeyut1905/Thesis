"""Training-free caching (ReSPect / KalSea) for Optimize Any Topology (OAT, NeurIPS 2025).

OAT = neural-field autoencoder (OpenTO/NFAE) + conditional latent diffusion U-Net (OpenTO/LDM, CTOPUNet, a diffusers
UNet2DModel with a boundary-condition / load / volume-fraction problem encoder), v-prediction, DDIM sampling.
Protocol of the OAT paper (general OpenTO benchmark, Table 4): 20 deterministic DDIM steps, classifier-free guidance
2.0, 10 samples per problem, OpenTO test split (5000 problems); evaluated with OAT's own FEA script
(scripts/RunComplianceTest.py), which reports failure rate, CE*, median CE* and VFE*.

Residual caching on the U-Net:  Emb = conv_in,  F = down/mid/up blocks (with their skip connections),
Out = conv_norm_out -> SiLU -> conv_out (both ends 320 channels at 64x64).
Gate input: the timestep/problem-modulated input of the first ResNet block of the CONDITIONAL pass,
conv1(SiLU(GroupNorm(h_in))) + time_emb_proj(SiLU(emb)), SEA-filtered (beta = 2 over H, W) with the VP coefficients
a = sqrt(alpha_bar_t), b = sqrt(1 - alpha_bar_t). The gate distance is the mean over the 10 samples of a problem, so
the samples of one problem share a schedule.  With CFG every step makes two calls (conditional / unconditional); they
are two residual streams with their own forecasters under the shared gate (as for Wan2.1 in KalSea).
PCGS: the DDIM update is affine in the guided output u + g (c - u); the gap is replayed with the scheduler's own step()
and the smoothed residuals of both streams, and sampling continues from the corrected latent.
"""
import argparse, copy, json, os, pickle, sys, time
import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.expanduser("~/mech_apps/OAT"))
from cachecore import SEAGate, Stream  # noqa: E402
from util_seacache import ab_from_scheduler  # noqa: E402
from OAT.Models import NFAE, CTOPUNet  # noqa: E402
from OAT.Pipelines import DDIMPipeline, OATPipeline  # noqa: E402
from OAT.DataUtils import OpenTO, NFAECollator, DiffusionCollator  # noqa: E402
from datasets import load_dataset  # noqa: E402
from OAT.DataUtils._utils import BatchDict  # noqa: E402

TEST_PARQUET = os.path.expanduser(
    "~/mech_apps/hf_cache/hub/datasets--OpenTO--OpenTO/snapshots/c553a645a10feb46d4906c899fd354c1f8f2db13/data/"
    "test-00000-of-00001.parquet")


# ------------------------------------------------------------------------------------------------ U-Net split
def embed(model, t, cond, B, device, unconditioned):
    ts = torch.as_tensor(t, device=device).reshape(1).long().expand(B)
    emb = model.time_embedding(model.time_proj(ts).to(model.dtype))
    if not unconditioned:
        pe = model.problem_embedding(model.problem_encoder(**{k: v for k, v in cond.items() if k != "unconditioned"}))
        emb = emb + pe
    return emb


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
def sample(model, sched_base, cond, x0, N, method, delta=0.0, g=2.0):
    """method: 'reference' (no cache) | 'seacache' | 'respect' | 'kalsea_nopcgs' | 'kalsea'."""
    sched = copy.deepcopy(sched_base)
    sched.set_timesteps(N)
    device, B = x0.device, x0.shape[0]
    x = x0.clone()
    use_cfg = g != 1.0
    use_cache = method != "reference"
    gate = SEAGate(delta, N, power_exp=2.0, dims=(-1, -2)) if use_cache else None
    sc = Stream(method, N) if use_cache else None
    su = Stream(method, N) if (use_cache and use_cfg) else None
    computed, nfe, stats = [], 0, dict(n_corr=0, gap_len=[])
    ts = list(sched.timesteps)

    def add(h, r):                                            # h_in + residual in float32, cast for Out
        return (h.float() + r.float()).to(h.dtype)

    def guided(c, u):
        return c if u is None else u + g * (c - u)

    for i, t in enumerate(ts):
        emb_c = embed(model, t, cond, B, device, False)
        emb_u = embed(model, t, cond, B, device, True) if use_cfg else None
        h_in = model.conv_in(x.to(model.dtype))
        if use_cache:
            a, b = ab_from_scheduler(sched, int(t), mode="vp")
            compute = gate.decide(i, gate.filtered(gate_input(model, h_in, emb_c), a, b))
        else:
            compute = True
        if compute:
            ho_c = body(model, h_in, emb_c)
            ho_u = body(model, h_in, emb_u) if use_cfg else None
            r_c = ho_c.float() - h_in.float()                    # residuals kept in float32
            r_u = (ho_u.float() - h_in.float()) if use_cfg else None
            nfe += 1
            computed.append(i)
            if use_cache:
                sc.observe(i, r_c)
                if su is not None:
                    su.observe(i, r_u)
                sm_c = sc.smooth_gap(i)
                if sm_c is not None:                          # PCGS: re-integrate the gap with smoothed residuals
                    sm_u = su.smooth_gap(i) if su is not None else [None] * len(sm_c)
                    xr = sm_c[0]["ctx"]["x"]
                    for gc, gu in zip(sm_c, sm_u):
                        tj = ts[gc["step"]]
                        hj = model.conv_in(xr.to(model.dtype))
                        vj = guided(out_layers(model, add(hj, gc["r"])), None if gu is None else out_layers(model, add(hj, gu["r"])))
                        xr = sched.step(vj, tj, xr).prev_sample
                    stats["gap_len"].append(len(sm_c))
                    if not torch.equal(xr, x):
                        stats["n_corr"] += 1
                        x = xr                                # continue from the corrected latent x_b'
                        h_in = model.conv_in(x.to(model.dtype))
                        ho_c = add(h_in, r_c)                 # residual of b attached to the corrected input
                        ho_u = add(h_in, r_u) if use_cfg else None
                        if i < N - 1:
                            a, b = ab_from_scheduler(sched, int(t), mode="vp")
                            gate.refresh(gate.filtered(gate_input(model, h_in, emb_c), a, b))
                sc.close(i)
                if su is not None:
                    su.close(i)
            v = guided(out_layers(model, ho_c), out_layers(model, ho_u) if use_cfg else None)
        else:
            pc = sc.forecast(i, ctx=dict(x=x))
            pu = su.forecast(i, ctx=dict(x=x)) if su is not None else None
            v = guided(out_layers(model, add(h_in, pc)), out_layers(model, add(h_in, pu)) if use_cfg else None)
        x = sched.step(v, t, x).prev_sample
    return x, dict(nfe=nfe, computed=computed, **stats)


@torch.no_grad()
def decode(model, nfae, latents, ae_batch):
    z = latents * model.latent_scale - model.latent_shift
    if nfae.use_vq:
        z, _, _ = nfae.quantizer(z)
    phi = nfae.decoder(z.to(torch.bfloat16))
    return nfae.renderer(phi, ae_batch["gt_coord"], ae_batch["gt_cell"])


def load_all(device):
    data = load_dataset("parquet", data_files={"test": TEST_PARQUET})["test"]
    dataset = OpenTO(data, full_sampling=True)
    collator = DiffusionCollator(unconditional_prob=0.0, inference=True,
                                 inference_collator=NFAECollator(zero_centering=False, full_sampling=True, coords_only=True))
    nfae = NFAE.from_pretrained("OpenTO/NFAE").to(device=device, dtype=torch.bfloat16).eval()
    model = CTOPUNet.from_pretrained("OpenTO/LDM").to(device=device, dtype=torch.bfloat16).eval()
    return data, dataset, collator, nfae, model


def problem_batch(dataset, collator, idx, n_samples, device):
    cond, ae = collator([dataset[idx]])
    ae, cond = OATPipeline.expand_batch_to_multi_sample(n_samples, ae, 1, cond)
    cond, ae = cond.to(device), ae.to(device)
    cond["unconditioned"] = False
    return cond, ae


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", type=str, required=True)
    ap.add_argument("--delta", type=float, default=0.0)
    ap.add_argument("--steps", type=int, default=20)
    ap.add_argument("--guidance", type=float, default=2.0)
    ap.add_argument("--n_samples", type=int, default=10)
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--end", type=int, default=5000)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--num_shards", type=int, default=1)
    ap.add_argument("--out", type=str, required=True)
    ap.add_argument("--log_every", type=int, default=50)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    device = torch.device("cuda")
    data, dataset, collator, nfae, model = load_all(device)
    sched_base = DDIMPipeline().InferenceScheduler
    idxs = [i for i in range(args.start, min(args.end, len(data))) if i % args.num_shards == args.shard]
    gens, recs = {}, []
    t_all = time.time()
    for k, idx in enumerate(idxs):
        cond, ae = problem_batch(dataset, collator, idx, args.n_samples, device)
        gen = torch.Generator(device="cpu").manual_seed(1000 + idx)
        x0 = torch.randn((args.n_samples, 1, model.sample_size, model.sample_size), generator=gen).to(device)
        torch.cuda.synchronize(); t0 = time.time()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            lat, info = sample(model, sched_base, cond, x0, args.steps, args.method, args.delta, args.guidance)
            torch.cuda.synchronize(); dt = time.time() - t0      # sampling time (decode excluded)
            pred = []
            for j in range(args.n_samples):                     # decode one sample at a time (large domains)
                aj = BatchDict({"gt_coord": ae["gt_coord"][j:j + 1], "gt_cell": ae["gt_cell"][j:j + 1]})
                pred.append(decode(model, nfae, lat[j:j + 1], aj)[0].detach().float().cpu())
        gens[idx] = pred
        del lat, pred
        torch.cuda.empty_cache()
        recs.append(dict(idx=idx, nfe=info["nfe"], computed=info["computed"], n_corr=info["n_corr"],
                         gaps=info["gap_len"], time=dt))
        if k % args.log_every == 0:
            print(f"[shard {args.shard}] {k}/{len(idxs)} idx={idx} nfe={info['nfe']} t={dt:.3f}s "
                  f"mem={torch.cuda.memory_allocated()/2**30:.1f}GiB peak={torch.cuda.max_memory_allocated()/2**30:.1f}GiB", flush=True)
    tag = f"{args.method}_d{args.delta:g}_shard{args.shard}"
    with open(os.path.join(args.out, tag + ".pkl"), "wb") as f:
        pickle.dump(gens, f)
    with open(os.path.join(args.out, tag + ".json"), "w") as f:
        json.dump(dict(records=recs, wall=time.time() - t_all), f)
    print(f"[shard {args.shard}] done {len(idxs)} problems in {time.time()-t_all:.1f}s", flush=True)


if __name__ == "__main__":
    main()
