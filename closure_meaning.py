"""
What does the closure relation actually test?

    tau_min = (q*)^{1 / (2 beta - 1)}

was reported as agreeing with a directly measured tau_min to within 0.01 of
life, and read as evidence that the theory describes the data. On the real
records that agreement holds for the UNWEIGHTED density and is roughly twice as
bad for the weighted one -- yet the weighted density is the one the Fisher
information is written in, and the one that recovers a known exponent to 5%
across five noise regimes.

The proposed explanation is that the relation is an identity for any power-law
density and says nothing beyond that. D'^2 is close to power-law whenever D is,
so the unweighted density satisfies it almost by construction. D'^2/sigma^2 is
power-law only if sigma is constant or itself a power of tau, so on real records
the correct density should satisfy it WORSE -- and that degradation is a
property of the noise, not a defect of the theory.

The explanation is checkable on synthetic records, where sigma is known. If it
is right, the two closures should agree under constant noise and separate as the
noise is made to vary, with the weighted one degrading.
"""
import os
import json
import numpy as np
from scipy.optimize import minimize_scalar
from scipy.signal import savgol_filter
from pipeline import robust_scale, oof_trend, local_scale, _win, LOCAL_BW

HERE = os.path.dirname(os.path.abspath(__file__))
QSTAR = 0.35
REPS = 30
N = 1200


def profile(dens, n):
    d = np.asarray(dens, float)
    if not np.all(np.isfinite(d)) or d.sum() <= 0:
        return np.nan, np.nan
    F = np.cumsum(d) / d.sum()
    tau = np.arange(1, n + 1) / n
    f = lambda b: float(np.mean((F - tau ** (2 * b - 1)) ** 2))
    beta = float(minimize_scalar(f, bounds=(0.51, 30.0), method="bounded").x)
    k = int(np.searchsorted(F, QSTAR))
    return beta, ((k + 1) / n if k < n else 1.0)


def closure(beta, tmin):
    if not (np.isfinite(beta) and np.isfinite(tmin)) or beta <= 0.51:
        return np.nan
    return abs(QSTAR ** (1.0 / (2 * beta - 1)) - tmin)


def both(x):
    n = len(x)
    w = _win(n)
    dt = savgol_filter(x, w, 2, deriv=1, delta=1.0 / n)
    res = x - oof_trend(x, w // 2)
    sl = np.clip(local_scale(res, max(5, int(LOCAL_BW * n))), 1e-12, None)
    return profile(dt ** 2, n), profile((dt / sl) ** 2, n)


rng = np.random.default_rng(2024)
tau = np.arange(1, N + 1) / N

print("closure error, as the noise is made to vary more\n")
print(f"{'noise profile':<28}{'sigma span':>12}{'unweighted':>12}"
      f"{'weighted':>10}")
print("-" * 62)
rows = []
CASES = [
    ("constant", lambda t: np.full_like(t, 0.04)),
    ("rises 2x", lambda t: 0.03 * (1 + 1.0 * t)),
    ("rises 5x", lambda t: 0.02 * (1 + 4.0 * t)),
    ("rises 10x", lambda t: 0.012 * (1 + 9.0 * t)),
    ("tracks signal (t^3)", lambda t: 0.02 + 0.20 * t ** 3),
]
for lab, sfn in CASES:
    EU, EW = [], []
    for _ in range(REPS):
        x = tau ** 3.0 + rng.normal(0, 1, N) * sfn(tau)
        (bu, tu), (bw, tw) = both(x)
        EU.append(closure(bu, tu)); EW.append(closure(bw, tw))
    s = sfn(tau)
    span = float(s.max() / s.min())
    u, w = float(np.nanmedian(EU)), float(np.nanmedian(EW))
    rows.append(dict(case=lab, span=span, unweighted=u, weighted=w))
    print(f"{lab:<28}{span:>12.1f}{u:>12.4f}{w:>10.4f}")
print("-" * 62)

c = rows[0]
print(f"under constant noise the two agree to "
      f"{abs(c['unweighted'] - c['weighted']):.4f}")
grow = rows[-1]
print(f"as the noise comes to track the signal the weighted closure degrades")
print(f"from {c['weighted']:.4f} to {grow['weighted']:.4f}, "
      f"while the unweighted one stays at {grow['unweighted']:.4f}")
print()
REAL = dict(unweighted=0.0200, weighted=0.0440)
print(f"real bearing records: unweighted {REAL['unweighted']:.4f}, "
      f"weighted {REAL['weighted']:.4f}")
if grow["weighted"] > grow["unweighted"] and \
        abs(c["unweighted"] - c["weighted"]) < 0.02:
    print("\nThe pattern on synthetic records with KNOWN varying noise is the")
    print("pattern the real records show. So the better closure of the")
    print("unweighted density is evidence that the trend is power-law-shaped,")
    print("not that the unweighted density is the right one. The closure test")
    print("checks the shape of D, and cannot arbitrate between the densities.")
else:
    print("\nThe synthetic records do not reproduce the pattern, so the")
    print("explanation is not supported and the discrepancy is unexplained.")
json.dump(dict(synthetic=rows, real=REAL),
          open(os.path.join(HERE, "closure_meaning.json"), "w"), indent=2)
