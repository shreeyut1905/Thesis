"""Shared caching core for the mechanical-engineering applications of the thesis.

A cache makes two decisions per sampler step i (see the thesis, Chapters 4 and 5):
  * WHEN to compute the block stack F: the SEA gate (SeaCache's spectrally aligned gate), unchanged.
  * WHAT to write on a skipped step:
      - 'seacache' : reuse of the last computed residual (SeaCache),
      - 'respect'  : ReSPect's Chebyshev + discrete-Taylor residual forecaster (SpectrumResidualForecaster,
                     imported unchanged from the ReSpect repository),
      - 'kalsea_nopcgs' : KalSea's damped-trend Kalman forecaster (KalSeaForecaster, per-element,
                     calibration-free, imported unchanged from KalSea-v0.2/FLUX2),
      - 'kalsea'   : the Kalman forecaster + predictor-corrector gap smoothing (PCGS).

The application drivers (oat_cache.py, pde_cache.py) own the sampler loop and call
  Stream.observe / Stream.forecast / Stream.smooth_gap and SEAGate.decide.
"""
from __future__ import annotations

import numpy as np
import torch

from util_seacache import apply_sea_from_ab, rel_l1
from spectrum_forecaster import SpectrumResidualForecaster
from kalsea_forecaster import KalSeaForecaster

METHODS = ("reference", "fewer_steps", "seacache", "respect", "kalsea_nopcgs", "kalsea")


class SEAGate:
    """Accumulate-and-reset gate on the SEA-filtered, timestep-modulated input of the first block.

    The reference input is refreshed at every step (computed or skipped); the first and last steps are always computed.
    """

    def __init__(self, delta: float, num_steps: int, power_exp: float = 2.0, dims=(-1, -2)):
        self.delta, self.N, self.beta, self.dims = float(delta), int(num_steps), float(power_exp), tuple(dims)
        self.acc, self.prev, self.distances = 0.0, None, []

    def filtered(self, I: torch.Tensor, a: float, b: float) -> torch.Tensor:
        return apply_sea_from_ab(I.float(), a, b, power_exp=self.beta, dims=self.dims, norm_mode="mean")

    def decide(self, i: int, I_f: torch.Tensor) -> bool:
        if i == 0 or i == self.N - 1 or self.prev is None:
            compute, d = True, 0.0
            self.acc = 0.0
        else:
            d = rel_l1(I_f, self.prev)
            self.acc += d
            compute = self.acc >= self.delta
            if compute:
                self.acc = 0.0
        self.distances.append(d)
        self.prev = I_f
        return compute

    def refresh(self, I_f: torch.Tensor):
        """PCGS step 4: replace the stored reference by the filtered input of the corrected latent."""
        self.prev = I_f


class Stream:
    """One residual stream (e.g. one network call per sampler step) with its filler and PCGS bookkeeping."""

    def __init__(self, method: str, num_steps: int):
        self.method, self.N = method, int(num_steps)
        self.fc = None
        if method == "respect":
            self.fc = SpectrumResidualForecaster(m=4, k=100, lam=0.1, w=0.5, t_max=float(num_steps), min_cheb_obs=4)
        elif method in ("kalsea", "kalsea_nopcgs"):
            self.fc = KalSeaForecaster(calib=None, num_steps=num_steps, tau=0.5, r_obs=1e-3, time_mode="step",
                                       use_dct=False, adapt_q=True)
        self.last = None            # last computed residual (SeaCache reuse, and fallback)
        self.shape = None
        self.pcgs = method == "kalsea"
        self.gap = []               # skipped steps since the last computed step: dict(step, pred, ctx)
        self.prior_P = {}
        self.upd = None
        self.hist = None            # last computed step index

    # -- shapes: KalSeaForecaster wants a 3-D tensor [B, Nt, D]; any per-element layout works -----------------
    def _flat(self, r):
        self.shape = r.shape
        return r.reshape(r.shape[0], r.shape[1], -1)

    def _unflat(self, r):
        return r.reshape(self.shape)

    # -- computed step -------------------------------------------------------------------------------------------
    @torch.no_grad()
    def observe(self, i: int, r: torch.Tensor):
        r = r.detach()
        if self.pcgs and self.gap and self.fc is not None and self.fc._ready and self.fc.m is not None:
            fc = self.fc
            fc._advance_to(i)
            Z = fc._to_state(self._flat(r))
            self.upd = dict(step=i, Pb=fc.P.copy(), e_rel=Z * fc.scale - fc.m)
        if self.fc is not None:
            x = self._flat(r) if isinstance(self.fc, KalSeaForecaster) else r
            self.fc.update(i, x)
        self.last = r

    # -- skipped step --------------------------------------------------------------------------------------------
    @torch.no_grad()
    def forecast(self, i: int, ctx=None) -> torch.Tensor:
        if self.fc is None or self.fc.n_obs < 1:
            pred = self.last
        elif isinstance(self.fc, KalSeaForecaster):
            pred = self._unflat(self.fc.predict(i)).to(self.last.dtype)
            if self.pcgs:
                self.prior_P[i] = self.fc.P.copy()
        else:
            pred = self.fc.predict(i).to(self.last.dtype)
        if self.pcgs:
            self.gap.append(dict(step=i, pred=pred, ctx=ctx))
        return pred

    # -- PCGS smoother -------------------------------------------------------------------------------------------
    def _gains(self, a, b, steps):
        fc = self.fc
        Pb = self.upd["Pb"]
        S = Pb[:, 0, 0] + fc.r_obs
        L = Pb.shape[0]
        Phi = np.tile(np.eye(2), (L, 1, 1))
        gains = {}
        for i in range(b - 1, a, -1):
            dl = fc.delta[:, min(i + 1, fc.N - 1)]
            rho = np.exp(-dl / fc.tau)
            A = np.zeros((L, 2, 2))
            A[:, 0, 0], A[:, 0, 1], A[:, 1, 1] = 1.0, dl, rho
            Phi = Phi @ A                                  # Phi(b, i) = Phi(b, i+1) A_{i+1}
            if i in steps:
                gains[i] = float(((self.prior_P[i][:, 0, :] * Phi[:, 0, :]).sum(-1) / S)[0])
        return gains

    @torch.no_grad()
    def smooth_gap(self, b: int):
        """Smoothed residuals r~_i = r^_i + g_i e_b for the gap closed by the computed step b (RTS gains)."""
        if not (self.pcgs and self.gap and self.hist is not None and self.upd is not None and self.upd["step"] == b):
            return None
        a = self.hist
        steps = [g["step"] for g in self.gap]
        assert steps == list(range(a + 1, b)), (a, b, steps)
        gains = self._gains(a, b, set(steps))
        E = self._unflat(self.upd["e_rel"] / self.fc.scale)
        return [dict(step=g["step"], r=(g["pred"].float() + gains[g["step"]] * E).to(g["pred"].dtype),
                     gain=gains[g["step"]], ctx=g["ctx"]) for g in self.gap]

    def close(self, b: int):
        self.gap, self.prior_P, self.upd = [], {}, None
        self.hist = b
