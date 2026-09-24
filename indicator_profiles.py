"""
Stage 5 -- where does the Fisher information sit, and does that depend on the
machine or on the indicator?

rho_w is built from Dhat'(tau)^2. If two indicators extracted from the SAME
accelerometer stream put their information in different parts of life, then the
minimum-window rule is a property of the feature-extraction chain, not of the
bearing -- which is the practically important statement for a sensor designer.

Reports, per indicator: the empirical information profile, the exponent of the
best-fitting power-law class, and the infeasible-band threshold tau_min it
implies at q* = 0.35.
"""
import os
import json
import numpy as np
from scipy.signal import savgol_filter
from scipy.optimize import minimize_scalar

HERE = os.path.dirname(os.path.abspath(__file__))
fz = np.load(os.path.join(HERE, "feat_raw.npz"), allow_pickle=True)
FN = [str(s) for s in fz["featnames"]]
BEARINGS = sorted(k for k in fz.files if k != "featnames")
I = {n: k for k, n in enumerate(FN)}
SMOOTH_FRAC, SIGMA_MAX, QSTAR = 0.08, 0.5, 0.35

INDICATORS = {
    "rms":      lambda M: M[:, I["rms"]],
    "kurtosis": lambda M: M[:, I["kurt"]],
    "peak":     lambda M: M[:, I["peak"]],
    "lf 0-2kHz": lambda M: M[:, I["b0_1"]] + M[:, I["b1_2"]],
    "mf 2-6kHz": lambda M: M[:, I["b2_4"]] + M[:, I["b4_6"]],
    "hf 4-10kHz": lambda M: M[:, I["b4_6"]] + M[:, I["b6_8"]] + M[:, I["b8_10"]],
    "vhf 10-12.8kHz": lambda M: M[:, I["b10_12.8"]],
}


def profile(x):
    n = len(x)
    base = np.median(x[:max(3, int(0.05 * n))])
    end = np.median(x[-3:])
    if not np.isfinite(base) or not np.isfinite(end) or abs(end - base) < 1e-12:
        return None
    D = (x - base) / (end - base)
    win = max(7, int(SMOOTH_FRAC * n))
    win = win + 1 if win % 2 == 0 else win
    win = min(win, n - 1 if (n - 1) % 2 == 1 else n - 2)
    if win < 7:
        return None
    tr = savgol_filter(D, win, 2)
    dt = savgol_filter(D, win, 2, deriv=1, delta=1.0 / n)
    resid = D - tr
    sig = float(1.4826 * np.median(np.abs(resid - np.median(resid))))
    if not np.isfinite(sig) or sig > SIGMA_MAX:
        return None
    dens = dt ** 2
    if dens.sum() <= 0:
        return None
    return dens / dens.sum(), n


def beta_from_profile(cum, n):
    """Best power-law exponent matching the observed cumulative information."""
    tau = np.arange(1, n + 1) / n
    f = lambda b: float(np.mean((cum - tau ** (2 * b - 1)) ** 2))
    r = minimize_scalar(f, bounds=(0.51, 30.0), method="bounded",
                        options={"xatol": 1e-3})
    return float(r.x)


print(f"{'indicator':<17}{'units':>7}{'info<0.5':>10}{'info<0.8':>10}"
      f"{'beta_eff':>10}{'tau_min obs':>13}{'tau_min beta=3':>16}")
print("-" * 83)
tau_min_pow3 = QSTAR ** (1.0 / (2 * 3 - 1))
out = {}
for name, fn in INDICATORS.items():
    f50, f80, betas, tmins = [], [], [], []
    for b in BEARINGS:
        p = profile(fn(fz[b]))
        if p is None:
            continue
        dens, n = p
        cum = np.cumsum(dens)
        f50.append(cum[int(0.5 * n) - 1])
        f80.append(cum[int(0.8 * n) - 1])
        betas.append(beta_from_profile(cum, n))
        idx = np.searchsorted(cum, QSTAR)
        tmins.append((idx + 1) / n if idx < n else 1.0)
    if not betas:
        continue
    out[name] = dict(n_units=len(betas),
                     info_lt_50=float(np.median(f50)),
                     info_lt_80=float(np.median(f80)),
                     beta_eff=float(np.median(betas)),
                     tau_min_obs=float(np.median(tmins)),
                     tau_min_obs_iqr=[float(np.percentile(tmins, 25)),
                                      float(np.percentile(tmins, 75))])
    print(f"{name:<17}{len(betas):>7}{np.median(f50):>10.3f}{np.median(f80):>10.3f}"
          f"{np.median(betas):>10.2f}{np.median(tmins):>13.3f}{tau_min_pow3:>16.3f}")
print("-" * 83)
print(f"\nq* = {QSTAR}.  'tau_min obs' is the earliest age at which the target is")
print(f"reachable by the widest possible window; 'tau_min beta=3' is what the")
print(f"nominal bearing class of the parent paper predicts.")
print(f"beta_eff is the power-law exponent whose cumulative information profile")
print(f"tau^(2b-1) best matches the observed one (beta=1 means uniform in time).")

with open(os.path.join(HERE, "indicator_profiles.json"), "w") as fh:
    json.dump(dict(qstar=QSTAR, tau_min_pow3=tau_min_pow3, indicators=out), fh, indent=2)
