"""
What shape does the noise floor have?

Dropping the power-law summary means summarising the information curve by its
own quantiles instead. Noise biases those quantiles, because differentiating
noise produces spurious density that has to go somewhere, so the correction has
to know where.

There is reason to expect the floor to be FLAT in the weighted density. The
Savitzky-Golay derivative of white noise of local scale sigma has variance
c(w, n) * sigma^2 for a constant c depending only on the window and the
sampling, so the weighted density (d/sigma)^2 has expectation c -- free of
sigma. If that holds, the floor is one number per record rather than a curve,
and correcting for it is a subtraction of a constant.

The unweighted density D'^2 has expectation c * sigma^2 and is therefore NOT
flat wherever the noise scale varies, which would be a second reason to prefer
the weighted form.

Both are measured here on trend-free records with deliberately varying noise.
"""
import os
import json
import numpy as np
from scipy.signal import savgol_filter
from pipeline import robust_scale, oof_trend, local_scale, _win, LOCAL_BW

HERE = os.path.dirname(os.path.abspath(__file__))
REPS = 200
N = 1200


def densities(x):
    n = len(x)
    w = _win(n)
    dt = savgol_filter(x, w, 2, deriv=1, delta=1.0 / n)
    res = x - oof_trend(x, w // 2)
    sl = np.clip(local_scale(res, max(5, int(LOCAL_BW * n))), 1e-12, None)
    return dt ** 2, (dt / sl) ** 2


def flatness(d):
    """Slope of the density across the record, as a fraction of its mean."""
    n = len(d)
    tau = np.arange(1, n + 1) / n
    m = d.mean()
    if m <= 0:
        return np.nan
    s = np.polyfit(tau, d, 1)[0]
    return float(s / m)


rng = np.random.default_rng(1130)
tau = np.arange(1, N + 1) / N

PROFILES = [
    ("constant", lambda t: np.full_like(t, 0.05)),
    ("rising 5x", lambda t: 0.02 * (1 + 4 * t)),
    ("falling 5x", lambda t: 0.02 * (5 - 4 * t)),
    ("rising 20x", lambda t: 0.01 * (1 + 19 * t)),
]

print("noise floor shape, on records with NO trend at all\n")
print(f"{'noise profile':<18}{'unweighted tilt':>18}{'weighted tilt':>16}")
print("-" * 52)
rows = []
for lab, sfn in PROFILES:
    U, W = [], []
    for _ in range(REPS):
        x = rng.normal(0, 1, N) * sfn(tau)
        du, dw = densities(x)
        U.append(flatness(du)); W.append(flatness(dw))
    u, w = float(np.median(U)), float(np.median(W))
    rows.append(dict(profile=lab, unweighted_tilt=u, weighted_tilt=w))
    print(f"{lab:<18}{u:>18.3f}{w:>16.3f}")
print("-" * 52)
print("tilt = slope of the density across the record over its own mean;")
print("0 is flat, 1 means it doubles from start to end.\n")

wt = [abs(r["weighted_tilt"]) for r in rows]
ut = [abs(r["unweighted_tilt"]) for r in rows]
print(f"weighted   floor tilt: {min(wt):.3f} to {max(wt):.3f}")
print(f"unweighted floor tilt: {min(ut):.3f} to {max(ut):.3f}")

# the constant itself, and whether it depends on sigma or on n
print("\nthe flat level, against record length and noise scale\n")
print(f"{'n':>7}" + "".join(f"{s:>12}" for s in (0.01, 0.05, 0.25)))
print("-" * 43)
level = {}
for n in (300, 600, 1200, 2400):
    t = np.arange(1, n + 1) / n
    row = []
    for s in (0.01, 0.05, 0.25):
        v = []
        for _ in range(60):
            _, dw = densities(rng.normal(0, s, n))
            v.append(float(np.median(dw)))
        row.append(float(np.median(v)))
        level[f"{n}|{s}"] = row[-1]
    print(f"{n:>7}" + "".join(f"{q:>12.1f}" for q in row))
print("-" * 43)
print("If the level is constant along each row, the floor does not depend on")
print("the noise scale, and one number per record suffices to remove it.")
json.dump(dict(tilt=rows, level=level),
          open(os.path.join(HERE, "floor_shape.json"), "w"), indent=2)
