"""
The looseness is not a property of the bound. Is it a property of the estimate of G?

The excess of the achieved error over the predicted one is not a function of the
predicted error: two conditions with the same predicted error, 0.00056 and
0.00058, gave excesses of 2.03 and 1.10. Correcting by predicted error left the
spread at 4.4x against 5.8x uncorrected, which is no correction at all.

Noise still correlated with the excess at +0.55 after its effect through G. That
points at G itself rather than at the bound. G is measured by subtracting a noise
floor, and when the floor is a large share of the raw density the subtraction
carries most of the uncertainty: the surviving signal is a small difference
between two large numbers. An overstated G understates the predicted error and
inflates the excess, with no failure of the bound involved.

So the candidate is the signal fraction -- how much of the raw density survives
the subtraction. If the excess tracks it, the diagnostic is actionable: trust
tau_min where the signal fraction is high, distrust it where it is low, and the
fraction is measurable on any real record without knowing the answer.
"""
import os
import json
import numpy as np
from nonparam import weighted_density, estimate_floor

HERE = os.path.dirname(os.path.abspath(__file__))
N_UNITS, SEED = 30, 8128
LENGTHS = (300, 800)
NOISES = (0.01, 0.04, 0.12, 0.40)
AGES = (0.30, 0.50, 0.70, 0.90)


def make_fleet(rng, n_mean, life_cv=0.20, beta_sd=0.25):
    lives = np.clip(rng.normal(n_mean, life_cv * n_mean, N_UNITS),
                    0.4 * n_mean, 2.0 * n_mean).astype(int)
    shapes, series = [], []
    for L in lives:
        b = float(np.clip(rng.normal(2.5, beta_sd), 1.2, 6.0))
        a = float(np.exp(rng.normal(0.0, beta_sd)))
        shapes.append(lambda t, a=a, b=b: a * np.asarray(t, float) ** b)
        series.append(shapes[-1](np.arange(1, L + 1) / L))
    return shapes, series


def oracle_sd(shapes, noisy, t0, grid=81):
    err, edge = [], 0
    for shape, x in zip(shapes, noisy):
        n = len(x)
        k0 = int(round(t0 * n)) - 1
        if k0 < 10:
            continue
        idx = np.arange(k0 + 1)
        obs = x[:k0 + 1]
        sse = lambda L: float(np.sum((obs - shape(np.clip(idx / L, 0, 1))) ** 2))
        lo, hi = 0.45 * n, 3.0 * n
        for _ in range(3):
            cand = np.linspace(lo, hi, grid)
            vals = [sse(L) for L in cand]
            j = int(np.argmin(vals))
            if j in (0, len(cand) - 1):
                edge += 1
            step = cand[1] - cand[0]
            lo, hi = cand[j] - step, cand[j] + step
        err.append((k0 + 1) / float(cand[j]) - (k0 + 1) / n)
    if len(err) <= 2:
        return np.nan, 1
    return float(np.std(err, ddof=1)), edge


def G_and_fraction(noisy, rng, t0):
    """G after floor subtraction, and how much of the raw density survived."""
    G, S = [], []
    for raw in noisy:
        dens, res = weighted_density(np.asarray(raw, float))
        if dens is None:
            continue
        fl = estimate_floor(dens, res, "surrogate", rng, reps=6)
        d = np.clip(dens - fl, 0.0, None)
        k0 = int(round(t0 * len(d))) - 1
        if k0 <= 1:
            continue
        raw_sum = float(dens[:k0 + 1].sum())
        G.append(float(d[:k0 + 1].sum()))
        S.append(G[-1] / raw_sum if raw_sum > 0 else np.nan)
    if not G:
        return np.nan, np.nan
    return float(np.median(G)), float(np.median(S))


rng = np.random.default_rng(SEED)
rows = []
for n_mean in LENGTHS:
    shapes, fleet = make_fleet(rng, n_mean)
    for lv in NOISES:
        noisy = [s + rng.normal(0, lv, len(s)) for s in fleet]
        for t0 in AGES:
            g, sf = G_and_fraction(noisy, rng, t0)
            o, edge = oracle_sd(shapes, noisy, t0)
            if not (np.isfinite(g) and g > 0 and np.isfinite(o) and o > 0
                    and np.isfinite(sf)):
                continue
            rows.append(dict(n=n_mean, noise=lv, age=t0, G=g, signal_frac=sf,
                             predicted=1 / np.sqrt(g), achieved=o,
                             excess=o * np.sqrt(g), edge=int(edge)))

good = [r for r in rows if not r["edge"]]
print(f"{len(good)} clean points of {len(rows)}\n")
print(f"{'n':>6}{'noise':>7}{'age':>6}{'signal %':>10}{'excess':>9}")
print("-" * 38)
for r in sorted(good, key=lambda r: r["signal_frac"]):
    print(f"{r['n']:>6}{r['noise']:>7.2f}{r['age']:>6.2f}"
          f"{100*r['signal_frac']:>9.0f}%{r['excess']:>9.2f}")
print("-" * 38)


def rank_corr(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    return float(np.corrcoef(np.argsort(np.argsort(a)),
                             np.argsort(np.argsort(b)))[0, 1])


ex = [r["excess"] for r in good]
sf = [r["signal_frac"] for r in good]
print(f"\nexcess against the signal fraction: rank correlation "
      f"{rank_corr(sf, ex):+.2f}")
print(f"excess against the predicted error: rank correlation "
      f"{rank_corr([r['predicted'] for r in good], ex):+.2f}  (for comparison)")

hi = [r for r in good if r["signal_frac"] > 0.9]
lo = [r for r in good if r["signal_frac"] <= 0.9]
if hi and lo:
    print(f"\nsignal fraction above 90% ({len(hi)} points): excess "
          f"{min(r['excess'] for r in hi):.2f}-{max(r['excess'] for r in hi):.2f}, "
          f"median {np.median([r['excess'] for r in hi]):.2f}")
    print(f"signal fraction at or below 90% ({len(lo)} points): excess "
          f"{min(r['excess'] for r in lo):.2f}-{max(r['excess'] for r in lo):.2f}, "
          f"median {np.median([r['excess'] for r in lo]):.2f}")
    print("\nIf the high-fraction group is tight and the low-fraction group is")
    print("not, the signal fraction is the diagnostic: it is measurable on a")
    print("real record, and it says when tau_min can be trusted.")
json.dump(rows, open(os.path.join(HERE, "tightness2.json"), "w"), indent=2,
          default=float)
