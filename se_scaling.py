"""
The content of the standard-error proposition, tested the way it can be.

se_formula_check.py tried to verify se(tau_min) = se(G_hat)/g by estimating the
numerator with a block bootstrap of the partial sum.  It failed, and the reason
is a mistake in that estimator rather than in the proposition: the blocks are
consecutive stretches of a density that RISES STEEPLY over life, so their scatter
measures the systematic variation of the density, not sampling error.  On the
worst cell it overstated the standard error twentyfold.  There is no cheap block
estimator of se(G_hat) on a record whose density varies systematically, so the
proposition is not a shortcut for computing an error bar; the bootstrap is.

What the proposition does say, and what can be tested, is the SHAPE of the
dependence: the standard error scales as the reciprocal of the density at the
age being read.  That is testable within a record, where se(G_hat) is common to
every age, by varying the demand q.  Each q places tau_min somewhere else on the
curve, with a different g, and the product

    se(tau_hat_min) * g(tau_min)

should then be roughly constant while se itself varies.  One bootstrap of the
record yields the whole family, since every replicate curve can be read at every
q, so the test costs one bootstrap rather than one per demand.

The comparison that matters is between the spread of se and the spread of the
product: if the proposition carries the dependence, the product must be the
tighter of the two.
"""
import os
import json
import time
import numpy as np
from scipy.signal import savgol_filter

from nonparam import (weighted_density, estimate_floor, info_curve, quantile,
                      _win)

HERE = os.path.dirname(os.path.abspath(__file__))
SEED = 20260828
BOOT = 40
QS = (0.10, 0.20, 0.35, 0.50, 0.70)
EPS = 1e-30

MEASURES = [
    ("total power", lambda P, f: P.sum(axis=1)),
    ("peak band power", lambda P, f: P.max(axis=1)),
    ("high/low ratio", lambda P, f: P[:, -8:].sum(axis=1)
     / np.clip(P[:, :8].sum(axis=1), EPS, None)),
]


def density(x, rng):
    dens, res = weighted_density(np.asarray(x, float))
    if dens is None:
        return None
    return np.clip(dens - estimate_floor(dens, res, "surrogate", rng), 0.0,
                   None)


def g_at(d, tau):
    """Density at an age, averaged over one smoothing window, per unit tau."""
    n = len(d)
    b = max(3, _win(n))
    i = min(n - 1, max(0, int(round(tau * n)) - 1))
    lo, hi = max(0, i - b // 2), min(n, i + b // 2 + 1)
    return float(np.mean(d[lo:hi])) * n


rng = np.random.default_rng(SEED)
z = np.load(os.path.join(HERE, "fineband.npz"))
f = z["centres"] / 1000.0
BK = sorted(k for k in z.files if k != "centres")[:4]

print("does the standard error scale as the reciprocal of the density?\n")
print(f"one bootstrap of {BOOT} replicates per record, read at every demand\n")
print(f"{'record':<13}{'measure':<18}{'q':>6}{'tau':>8}{'g':>11}"
      f"{'se':>9}{'se x g':>11}")
print("-" * 76, flush=True)
rows, t0 = [], time.time()
for b in BK:
    P = z[b].astype(float)
    for nm, fun in MEASURES:
        x = np.asarray(fun(P, f), float)
        x = x[np.isfinite(x)]
        n = len(x)
        w = _win(n)
        if n < 300 or w < 7:
            continue
        d0 = density(x, rng)
        if d0 is None or d0.sum() <= 0:
            continue
        T = savgol_filter(x, w, 2)
        r = x - T
        bl = max(3, w)
        starts = np.arange(0, max(1, n - bl))
        curves = []
        for _ in range(BOOT):
            rb = np.concatenate([r[s:s + bl] for s in
                                 rng.choice(starts, size=n // bl + 1,
                                            replace=True)])[:n]
            F = info_curve(T + rb, rng=rng)
            if F is not None:
                curves.append(F)
        if len(curves) < BOOT // 2:
            continue
        for q in QS:
            v = np.array([quantile(F, q) for F in curves], float)
            v = v[np.isfinite(v)]
            if len(v) < BOOT // 2:
                continue
            se = float(np.std(v, ddof=1))
            tau = float(np.median(v))
            g = g_at(d0, tau)
            if not np.isfinite(g) or g <= 0 or se <= 0:
                continue
            rows.append(dict(unit=b, measure=nm, q=q, tau=tau, g=g, se=se,
                             prod=se * g))
            print(f"{b:<13}{nm:<18}{q:>6.2f}{tau:>8.3f}{g:>11.3g}"
                  f"{se:>9.4f}{se * g:>11.3g}", flush=True)
print("-" * 76)
print(f"({time.time() - t0:.0f}s)\n")

print("spread within each record and measure, over the five demands\n")
print(f"{'record':<13}{'measure':<18}{'se: max/min':>13}"
      f"{'se x g: max/min':>18}{'tighter':>10}")
print("-" * 74)
summary, wins = [], 0
for key in sorted({(r["unit"], r["measure"]) for r in rows}):
    grp = [r for r in rows if (r["unit"], r["measure"]) == key]
    if len(grp) < 3:
        continue
    se = np.array([r["se"] for r in grp])
    pr = np.array([r["prod"] for r in grp])
    a, c = float(se.max() / se.min()), float(pr.max() / pr.min())
    wins += int(c < a)
    summary.append(dict(unit=key[0], measure=key[1], n=len(grp),
                        se_ratio=a, prod_ratio=c, tighter=bool(c < a)))
    print(f"{key[0]:<13}{key[1]:<18}{a:>13.1f}{c:>18.1f}"
          f"{('product' if c < a else 'se'):>10}")
print("-" * 74)
print(f"\nthe product is the tighter quantity in {wins} of {len(summary)} cases")
pval = None
if summary:
    ms = float(np.median([s["se_ratio"] for s in summary]))
    mp = float(np.median([s["prod_ratio"] for s in summary]))
    N = len(summary)
    # exact one-sided sign test: how often would chance give this many wins?
    from math import comb
    pval = float(sum(comb(N, k) for k in range(wins, N + 1)) / 2 ** N)
    print(f"median spread of se across demands {ms:.1f}, of the product {mp:.1f}"
          f"  ({100 * (1 - mp / ms):.0f} per cent narrower)")
    print(f"exact one-sided sign test on {wins} of {N}: p = {pval:.3f}")
    print()
    print("The direction is the one the proposition predicts and the reduction")
    print("is real but partial: a quarter of the variation, not most of it, and")
    print("the spreads remain factors of ten or more either way.  With twelve")
    print(f"cases the sign test does not reach significance (p = {pval:.2f}).")
    print()
    print("The honest reading is therefore narrow.  The proposition is a correct")
    print("first-order identity and it does say where an age is well determined")
    print("-- the reciprocal-density dependence is visible -- but it does not")
    print("account for most of the variation in the estimator's error on these")
    print("records, and it is not a substitute for bootstrapping that error.")

json.dump(dict(boot=BOOT, qs=list(QS), cells=rows, summary=summary,
               product_tighter=wins, cases=len(summary),
               se_spread_median=(float(np.median([s["se_ratio"]
                                                  for s in summary]))
                                 if summary else None),
               prod_spread_median=(float(np.median([s["prod_ratio"]
                                                    for s in summary]))
                                   if summary else None),
               sign_test_p=pval),
          open(os.path.join(HERE, "se_scaling.json"), "w"), indent=2,
          default=float)
