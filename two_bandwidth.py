"""
One bandwidth is being asked to do two incompatible jobs.

The density D'^2 / sigma^2 needs two estimates from the same record. They want
opposite things from a smoothing window:

  D'    wants a WIDE window. Differentiating amplifies noise, so a narrow window
        gives a derivative dominated by it.
  sigma wants a NARROW window. Whatever the trend fails to capture lands in the
        residual, so a wide window inflates the scale exactly where the trend
        turns fastest -- which is where the information is.

Both currently come from the same 8% window, and the bandwidth sweep showed the
consequence: the coupling between local scale and local derivative on real
records grows steadily with that window, well past what constant-noise data of
the same length and noise produce.

This separates them, h_D for the derivative and h_S for the scale, and asks
which pair recovers a known exponent across noise regimes that break the single
-bandwidth choice in opposite directions.
"""
import os
import json
import numpy as np
from scipy.signal import savgol_filter
from scipy.optimize import minimize_scalar
from pipeline import robust_scale, oof_trend, local_scale, LOCAL_BW

HERE = os.path.dirname(os.path.abspath(__file__))
REPS = 30
N = 1200
HD = [0.04, 0.08, 0.15]
HS = [0.02, 0.04, 0.08, 0.15]


def win(n, frac):
    w = max(7, int(frac * n))
    w += (w % 2 == 0)
    return min(w, n - 1 if (n - 1) % 2 == 1 else n - 2)


def fit_beta(dens, n):
    d = np.asarray(dens, float)
    if not np.all(np.isfinite(d)) or d.sum() <= 0:
        return np.nan
    F = np.cumsum(d) / d.sum()
    tau = np.arange(1, n + 1) / n
    f = lambda b: float(np.mean((F - tau ** (2 * b - 1)) ** 2))
    return float(minimize_scalar(f, bounds=(0.51, 30.0), method="bounded").x)


def beta_two(x, h_d, h_s):
    n = len(x)
    wd, ws = win(n, h_d), win(n, h_s)
    if wd < 7 or ws < 7:
        return np.nan
    dt = savgol_filter(x, wd, 2, deriv=1, delta=1.0 / n)
    res = x - oof_trend(x, ws // 2)          # scale from the NARROW fit
    sl = np.clip(local_scale(res, max(5, int(LOCAL_BW * n))), 1e-12, None)
    return fit_beta((dt / sl) ** 2, n)


def truth(beta, sfn, n):
    tau = np.arange(1, n + 1) / n
    return fit_beta((beta * tau ** (beta - 1)) ** 2 / sfn(tau) ** 2, n)


rng = np.random.default_rng(1207)
tau = np.arange(1, N + 1) / N
CASES = [
    ("constant noise, beta=3", 3.0, lambda t: np.full_like(t, 0.04)),
    ("constant noise, beta=8", 8.0, lambda t: np.full_like(t, 0.04)),
    ("noise falls 5x, beta=3", 3.0, lambda t: 0.02 * (5 - 4 * t)),
    ("noise tracks signal, beta=3", 3.0, lambda t: 0.02 + 0.20 * t ** 3),
    ("noise tracks signal, beta=8", 8.0, lambda t: 0.02 + 0.20 * t ** 8),
]

data = {}
for lab, beta, sfn in CASES:
    tb = truth(beta, sfn, N)
    xs = [tau ** beta + rng.normal(0, 1, N) * sfn(tau) for _ in range(REPS)]
    grid = {}
    for hd in HD:
        for hs in HS:
            v = [beta_two(x, hd, hs) for x in xs]
            v = [q for q in v if np.isfinite(q)]
            grid[(hd, hs)] = float(np.median(v)) if v else np.nan
    data[lab] = dict(truth=tb, grid=grid)

print("recovered exponent, by (derivative bandwidth, scale bandwidth)\n")
for lab, d in data.items():
    print(f"{lab}   true beta = {d['truth']:.2f}")
    print(f"{'h_D \\ h_S':<12}" + "".join(f"{h:>9.2f}" for h in HS))
    for hd in HD:
        cells = "".join(f"{d['grid'][(hd, hs)]:>9.2f}" for hs in HS)
        print(f"{hd:<12.2f}{cells}")
    print()

print("relative error |recovered - true| / true, averaged over the five cases\n")
print(f"{'h_D \\ h_S':<12}" + "".join(f"{h:>9.2f}" for h in HS))
best, bestv = None, np.inf
for hd in HD:
    row = []
    for hs in HS:
        e = np.mean([abs(d["grid"][(hd, hs)] - d["truth"]) / d["truth"]
                     for d in data.values()])
        row.append(e)
        if e < bestv:
            bestv, best = e, (hd, hs)
    print(f"{hd:<12.2f}" + "".join(f"{e:>9.3f}" for e in row))
print()
same = np.mean([abs(d["grid"][(0.08, 0.08)] - d["truth"]) / d["truth"]
                for d in data.values()])
print(f"single bandwidth as used throughout (0.08, 0.08): error {same:.3f}")
print(f"best separated pair (h_D={best[0]:.2f}, h_S={best[1]:.2f}): "
      f"error {bestv:.3f}")
print(f"improvement: {100*(1-bestv/same):.0f}%")
json.dump({lab: dict(truth=d["truth"],
                     grid={f"{k[0]}|{k[1]}": v for k, v in d["grid"].items()})
           for lab, d in data.items()} | {"best": list(best),
                                          "best_err": bestv,
                                          "single_err": float(same)},
          open(os.path.join(HERE, "two_bandwidth.json"), "w"), indent=2)
