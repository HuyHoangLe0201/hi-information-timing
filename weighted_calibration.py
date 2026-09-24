"""
The de-biasing map has to be rebuilt for the density the theory actually uses.

beta_calibration.json was built by running the UNWEIGHTED estimator over a grid
of known exponents, noise scales and record lengths, and inverting the result.
The weighted estimator D'^2/sigma^2 is a different estimator with a different
bias, so that map does not apply to it: it divides by an estimated local scale,
which adds its own error, and it responds differently to a trend that steepens.

This rebuilds the forward map for the weighted estimator on the same grid, so
the two can be compared on equal terms and the real measurements re-corrected.

The reference exponent for each grid point is the exponent of the TRUE density
D'^2/sigma^2, not the trend exponent beta. Under constant noise the two coincide;
under varying noise they do not, and it is the former the estimator is trying to
recover.
"""
import os
import json
import numpy as np
from scipy.signal import savgol_filter
from scipy.optimize import minimize_scalar
from pipeline import robust_scale, oof_trend, local_scale, _win, LOCAL_BW

HERE = os.path.dirname(os.path.abspath(__file__))
BETAS = [0.7, 1.0, 1.5, 2.0, 3.0, 5.0, 8.0, 15.0, 25.0]
SIGMAS = [0.01, 0.02, 0.04, 0.08, 0.16]
LENGTHS = [90, 200, 600, 1400]
REPS = 14


def fit_beta(dens, n):
    d = np.asarray(dens, float)
    if not np.all(np.isfinite(d)) or d.sum() <= 0:
        return np.nan
    F = np.cumsum(d) / d.sum()
    tau = np.arange(1, n + 1) / n
    f = lambda b: float(np.mean((F - tau ** (2 * b - 1)) ** 2))
    return float(minimize_scalar(f, bounds=(0.51, 30.0), method="bounded").x)


def measure(x, weighted):
    n = len(x)
    w = _win(n)
    if w < 7:
        return np.nan
    dt = savgol_filter(x, w, 2, deriv=1, delta=1.0 / n)
    if not weighted:
        return fit_beta(dt ** 2, n)
    res = x - oof_trend(x, w // 2)
    sl = np.clip(local_scale(res, max(5, int(LOCAL_BW * n))), 1e-12, None)
    return fit_beta((dt / sl) ** 2, n)


rng = np.random.default_rng(6060)
grid = {}
for n in LENGTHS:
    tau = np.arange(1, n + 1) / n
    for b in BETAS:
        clean = tau ** b
        for s in SIGMAS:
            v = []
            for _ in range(REPS):
                q = measure(clean + rng.normal(0, s, n), True)
                if np.isfinite(q):
                    v.append(q)
            grid[f"{n}|{b}|{s}"] = float(np.median(v)) if v else None

json.dump(grid, open(os.path.join(HERE, "weighted_calibration.json"), "w"),
          indent=1)

OLD = json.load(open(os.path.join(HERE, "beta_calibration.json")))
print("forward map of the weighted estimator, against the unweighted one")
print("(true exponent -> measured; constant noise, so the true density")
print(" exponent equals the trend exponent)\n")
for n in (200, 1400):
    print(f"  n = {n}")
    print(f"{'true beta':>10}" +
          "".join(f"{s:>9}" for s in ("s=.02 w", "s=.02 u",
                                      "s=.08 w", "s=.08 u")))
    print("  " + "-" * 48)
    for b in BETAS:
        vals = []
        for s in (0.02, 0.08):
            vals.append(grid.get(f"{n}|{b}|{s}"))
            vals.append(OLD.get(f"{n}|{b}|{s}"))
        cells = "".join("      n/a" if v is None else f"{v:>9.2f}"
                        for v in (vals[0], vals[1], vals[2], vals[3]))
        print(f"{b:>10.1f}{cells}")
    print()

print("summary: median |measured - true| / true over the whole grid")
for tag, G in (("weighted", grid), ("unweighted", OLD)):
    e = []
    for n in LENGTHS:
        for b in BETAS:
            for s in SIGMAS:
                v = G.get(f"{n}|{b}|{s}")
                if v is not None and b < 23:      # exclude the saturated end
                    e.append(abs(v - b) / b)
    print(f"  {tag:<12}{np.median(e):.3f}")
print("\nw = weighted, u = unweighted. Under constant noise neither estimator")
print("has an advantage in principle; a difference here is estimation error")
print("from dividing by a local scale that has itself been estimated.")
