"""
Every indicator, re-measured without a power law.

This replaces the exponent-based numbers. For each indicator it reports the
quantile function of the information distribution, which IS the design curve:
the entry at q is the earliest age at which a target costing q of the budget is
reachable, and no other object is needed to state it.

Alongside each, the value the power-law route gave, so the size of the change
is visible per indicator rather than in aggregate. Where the two agree the
exponent was harmless; where they diverge it was carrying the error.

The noise floor is removed before the quantiles are formed. Without that,
differentiating noise contributes density that no degradation put there, and it
pulls every quantile toward the middle of the record.
"""
import os
import json
import numpy as np
from scipy.optimize import minimize_scalar
from nonparam import (weighted_density, info_curve, quantile, w_star, spread)

HERE = os.path.dirname(os.path.abspath(__file__))
QS = (0.10, 0.25, 0.35, 0.50, 0.75, 0.90)
SEED = 707


def beta_of(x):
    dens, _ = weighted_density(x)
    if dens is None or dens.sum() <= 0:
        return np.nan
    n = len(dens)
    F = np.cumsum(dens) / dens.sum()
    tau = np.arange(1, n + 1) / n
    f = lambda b: float(np.mean((F - tau ** (2 * b - 1)) ** 2))
    return float(minimize_scalar(f, bounds=(0.51, 30.0), method="bounded").x)


def analyse(label, series, rng):
    Q = {q: [] for q in QS}
    B, SP, W = [], [], []
    for x in series:
        x = np.asarray(x, float)
        F = info_curve(x, floor="surrogate", rng=rng)
        if F is None:
            continue
        for q in QS:
            Q[q].append(quantile(F, q))
        SP.append(spread(F))
        w = w_star(F, 0.8, 0.35)
        if np.isfinite(w):
            W.append(w)
        b = beta_of(x)
        if np.isfinite(b):
            B.append(b)
    if not SP:
        return None
    med = {q: float(np.median(Q[q])) for q in QS}
    beta = float(np.median(B)) if B else np.nan
    r = dict(indicator=label, units=len(SP), quantiles=med,
             spread=float(np.median(SP)),
             w_star_at_0p8=float(np.median(W)) if W else np.nan,
             beta=beta,
             beta_tau35=(0.35 ** (1.0 / (2 * beta - 1))
                         if np.isfinite(beta) and beta > 0.51 else np.nan))
    r["gap_at_0p35"] = abs(r["beta_tau35"] - med[0.35])
    return r


def show(rows, title):
    print(f"\n=== {title} ===\n")
    print(f"{'indicator':<16}{'units':>6}" +
          "".join(f"{q:>8.2f}" for q in QS) +
          f"{'spread':>8}{'w*':>7}{'via beta':>10}{'gap':>7}")
    print("-" * (30 + 8 * len(QS) + 32))
    for r in rows:
        print(f"{r['indicator']:<16}{r['units']:>6}" +
              "".join(f"{r['quantiles'][q]:>8.3f}" for q in QS) +
              f"{r['spread']:>8.3f}"
              f"{r['w_star_at_0p8']:>7.2f}"
              f"{r['beta_tau35']:>10.3f}{r['gap_at_0p35']:>7.3f}")
    print("-" * (30 + 8 * len(QS) + 32))


rng = np.random.default_rng(SEED)
out = {}

z = np.load(os.path.join(HERE, "feat_raw.npz"), allow_pickle=True)
FN = [str(s) for s in z["featnames"]]
I = {k: i for i, k in enumerate(FN)}
BK = sorted(k for k in z.files if k != "featnames")
BCH = {"rms": ["rms"], "peak": ["peak"], "kurt": ["kurt"],
       "0--2 kHz": ["b0_1", "b1_2"], "2--4 kHz": ["b2_4"],
       "4--10 kHz": ["b4_6", "b6_8", "b8_10"], "10--12.8 kHz": ["b10_12.8"]}
rows = []
for lab, parts in BCH.items():
    r = analyse(lab, [sum(z[b][:, I[p]] for p in parts) for b in BK], rng)
    if r:
        rows.append(r)
out["bearings"] = rows
show(rows, "PRONOSTIA bearings")

zt = np.load(os.path.join(HERE, "cmapss_FD001.npz"))
un = sorted({k.split("__")[0] for k in zt.files})[:60]
rows = []
for c in (4, 9, 11, 15, 20, 21):
    r = analyse(f"s{c}", [zt[f"{u}__sensors"][:, c - 1].astype(float)
                          for u in un], rng)
    if r:
        rows.append(r)
out["turbofan"] = rows
show(rows, "C-MAPSS FD001")

zb = np.load(os.path.join(HERE, "battery_raw.npz"))
cl = sorted({k.split("__")[0] for k in zb.files})
rows = []
for nm, lab in (("cap", "capacity"), ("Re", "R_e"), ("Rct", "R_ct")):
    r = analyse(lab, [zb[f"{c}__{nm}"] for c in cl], rng)
    if r:
        rows.append(r)
out["cells"] = rows
show(rows, "NASA cells")

print("\nquantile at q = earliest age at which a target costing q of the")
print("budget is reachable; spread = tau_0.90 - tau_0.10; w* = shortest")
print("window ending at 0.8 collecting 0.35 of the budget.")
print("via beta = what the power-law route gives for q = 0.35; gap = its error.\n")
allr = [r for v in out.values() for r in v]
g = [r["gap_at_0p35"] for r in allr if np.isfinite(r["gap_at_0p35"])]
print(f"the exponent route differs from the direct reading by "
      f"{np.median(g):.3f} of life at the median, up to {max(g):.3f}.")
worst = sorted(allr, key=lambda r: -r["gap_at_0p35"])[:4]
for r in worst:
    print(f"  {r['indicator']:<16} direct {r['quantiles'][0.35]:.3f}, "
          f"via beta {r['beta_tau35']:.3f}")
json.dump(out, open(os.path.join(HERE, "nonparam_indicators.json"), "w"),
          indent=2, default=float)
