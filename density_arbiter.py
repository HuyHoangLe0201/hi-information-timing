"""
Weighted or unweighted density: which one recovers a known answer?

Theory says the information density is D'^2 / sigma(tau)^2, so the weighted form
is the one the Fisher information is written in. But sigma(tau) has to be
estimated, and there are two ways that estimate can go wrong in a way that
would make the weighted density worse in practice than the approximation it
corrects:

  leakage   -- if the trend estimate under-fits, unmodelled trend lands in the
               residual, so the local scale rises exactly where the indicator
               moves fastest. Dividing by it then suppresses the informative
               part of the record.
  noise     -- a local scale estimated from few points is itself noisy, and
               dividing by a noisy quantity biases the ratio.

Synthetic records settle it, because there the answer is known. Two regimes:
constant noise, where the unweighted density is exactly right and the weighted
one can only add estimation error; and genuinely varying noise, where the
weighted density is right and the unweighted one is misspecified. An estimator
worth using should win the second without losing much of the first.

The leakage question is then asked of the real records directly.
"""
import os
import json
import numpy as np
from scipy.optimize import minimize_scalar
from scipy.signal import savgol_filter
from pipeline import robust_scale, oof_trend, local_scale, _win, LOCAL_BW

HERE = os.path.dirname(os.path.abspath(__file__))
QSTAR = 0.35
REPS = 40


def fit_beta(dens, n):
    d = np.asarray(dens, float)
    if not np.all(np.isfinite(d)) or d.sum() <= 0:
        return np.nan
    F = np.cumsum(d) / d.sum()
    tau = np.arange(1, n + 1) / n
    f = lambda b: float(np.mean((F - tau ** (2 * b - 1)) ** 2))
    return float(minimize_scalar(f, bounds=(0.51, 30.0), method="bounded").x)


def both_betas(x):
    n = len(x)
    w = _win(n)
    dt = savgol_filter(x, w, 2, deriv=1, delta=1.0 / n)
    res = x - oof_trend(x, w // 2)
    sl = np.clip(local_scale(res, max(5, int(LOCAL_BW * n))), 1e-12, None)
    return fit_beta(dt ** 2, n), fit_beta((dt / sl) ** 2, n), dt, sl


def truth_beta(beta, sig_fn, n):
    """The exponent implied by the true density D'^2/sigma^2, computed exactly."""
    tau = np.arange(1, n + 1) / n
    d = (beta * tau ** (beta - 1)) ** 2 / sig_fn(tau) ** 2
    return fit_beta(d, n)


rng = np.random.default_rng(917)
n = 1200
tau = np.arange(1, n + 1) / n

CASES = [
    ("constant noise", lambda t: np.full_like(t, 0.04)),
    ("noise rising 5x", lambda t: 0.02 * (1 + 4 * t)),
    ("noise falling 5x", lambda t: 0.02 * (5 - 4 * t)),
    ("noise tracks signal", lambda t: 0.02 + 0.10 * t ** 3),
]

print("recovering a known exponent (beta = 3, n = 1200)\n")
print(f"{'noise profile':<22}{'true beta':>11}{'unweighted':>12}"
      f"{'weighted':>10}{'winner':>10}")
print("-" * 65)
rows = []
for lab, sfn in CASES:
    bt = truth_beta(3.0, sfn, n)
    U, W = [], []
    for _ in range(REPS):
        x = tau ** 3.0 + rng.normal(0, 1, n) * sfn(tau)
        bu, bw, _, _ = both_betas(x)
        U.append(bu); W.append(bw)
    u, w = float(np.median(U)), float(np.median(W))
    win = "weighted" if abs(w - bt) < abs(u - bt) else "unweighted"
    rows.append(dict(case=lab, truth=bt, unweighted=u, weighted=w, winner=win))
    print(f"{lab:<22}{bt:>11.2f}{u:>12.2f}{w:>10.2f}{win:>10}")
print("-" * 65)
print("true beta is the exponent of the TRUE density D'^2/sigma^2, which is 3")
print("only when the noise is constant; varying noise genuinely changes it.\n")

# ---- is the local scale contaminated by trend? -----------------------------
print("leakage check on real records: does the local scale rise where the")
print("indicator moves fastest?\n")
z = np.load(os.path.join(HERE, "feat_raw.npz"), allow_pickle=True)
FN = [str(s) for s in z["featnames"]]
I = {k: i for i, k in enumerate(FN)}
BK = sorted(k for k in z.files if k != "featnames")
CH = {"rms": ["rms"], "kurt": ["kurt"], "0--2 kHz": ["b0_1", "b1_2"],
      "4--10 kHz": ["b4_6", "b6_8", "b8_10"]}
print(f"{'channel':<14}{'units':>6}{'corr(log sig, log |D\'|)':>26}")
print("-" * 46)
leak = []
for lab, parts in CH.items():
    C = []
    for b in BK:
        x = np.asarray(sum(z[b][:, I[p]] for p in parts), float)
        if len(x) < 100:
            continue
        try:
            _, _, dt, sl = both_betas(x)
        except Exception:
            continue
        m = np.isfinite(dt) & np.isfinite(sl) & (np.abs(dt) > 0) & (sl > 0)
        if m.sum() < 60:
            continue
        C.append(float(np.corrcoef(np.log(sl[m]),
                                   np.log(np.abs(dt[m])))[0, 1]))
    if len(C) < 4:
        continue
    leak.append(dict(channel=lab, units=len(C), corr=float(np.median(C))))
    print(f"{lab:<14}{len(C):>6}{np.median(C):>26.3f}")
print("-" * 46)
print("A strongly positive correlation would mean the local scale is partly")
print("unmodelled trend, and dividing by it would suppress the informative")
print("part of the record rather than correcting for noise. Synthetic control")
print("below gives the value this statistic takes when there IS no leakage.")

ctrl = []
for _ in range(REPS):
    x = tau ** 3.0 + rng.normal(0, 0.04, n)
    _, _, dt, sl = both_betas(x)
    m = np.isfinite(dt) & np.isfinite(sl) & (np.abs(dt) > 0) & (sl > 0)
    ctrl.append(float(np.corrcoef(np.log(sl[m]), np.log(np.abs(dt[m])))[0, 1]))
print(f"\nsynthetic control, constant noise: corr = {np.median(ctrl):+.3f}")

json.dump(dict(recovery=rows, leakage=leak,
               control_corr=float(np.median(ctrl))),
          open(os.path.join(HERE, "density_arbiter.json"), "w"), indent=2)
