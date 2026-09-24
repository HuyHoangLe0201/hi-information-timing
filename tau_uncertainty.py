"""
How well is the earliest usable age determined?

Every age this paper quotes is a point estimate read off a curve estimated from
one record, and none carries an error bar.  That is a gap in rigour on its own,
and it is a gap in the headline applied claim in particular: distributional
indicators are reported as usable 0.35 to 0.65 of a lifetime earlier than amount
indicators, with nothing said about whether that separation survives the noise
in the estimate.

The first-order answer is exact and interpretable.  With tau_min defined by
G(tau_min) = c and G estimated,

    tau_hat_min - tau_min  =  -( G_hat(tau_min) - G(tau_min) ) / g(tau_min)
                                                            + higher order,

so

    se(tau_hat_min)  =  se(G_hat(tau_min)) / g(tau_min).                    (*)

The age is sharply determined exactly where the information density is high and
loosely determined where the curve is flat -- the same geometry that makes a
flat curve give a late age in the first place.  One number governs both the
value and its reliability.

Two things are done.  The first-order formula is checked against a direct block
bootstrap of the estimation chain, which is the only way to know whether the
leading term dominates.  Then the uncertainty is carried into the applied claim:
the distribution-minus-amount gap is recomputed with the fleet resampled, and
reported as an interval rather than a point.

A first attempt ran the bootstrap through the whole pipeline for every cell and
did not finish: one curve costs about two seconds, so sixty resamples across
sixteen cells is half an hour.  The check is therefore run on a subset large
enough to state a median ratio, and the per-unit ages -- which the fleet
bootstrap needs -- are computed once and cached.  A separate bug in that attempt
is fixed here: the residual used for resampling must be the residual of the
record itself, not of the standardised series the curve is built from.
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
QSTAR = 0.35
BOOT_WITHIN = 40
BOOT_FLEET = 4000
BLOCK = 0.05
EPS = 1e-30
CACHE = os.path.join(HERE, "tau_percell.json")


def norm_rows(P):
    return P / np.clip(P.sum(axis=1, keepdims=True), EPS, None)


MEASURES = [
    ("spectral entropy", "distribution",
     lambda P, f: -(norm_rows(P) * np.log(np.clip(norm_rows(P), EPS, None))
                    ).sum(axis=1)),
    ("spectral centroid", "distribution",
     lambda P, f: (norm_rows(P) * f).sum(axis=1)),
    ("spectral spread", "distribution",
     lambda P, f: np.sqrt((norm_rows(P)
                           * (f - (norm_rows(P) * f).sum(axis=1, keepdims=True))
                           ** 2).sum(axis=1))),
    ("high/low ratio", "distribution",
     lambda P, f: P[:, -8:].sum(axis=1)
     / np.clip(P[:, :8].sum(axis=1), EPS, None)),
    ("total power", "amount", lambda P, f: P.sum(axis=1)),
    ("peak band power", "amount", lambda P, f: P.max(axis=1)),
    ("high-band power", "amount", lambda P, f: P[:, -8:].sum(axis=1)),
    ("log total power", "amount",
     lambda P, f: np.log(np.clip(P.sum(axis=1), EPS, None))),
]
RIGS = {"PRONOSTIA": "fineband.npz", "XJTU": "fineband_xjtu.npz"}


def tau_of(x, rng):
    F = info_curve(np.asarray(x, float), rng=rng)
    return quantile(F, QSTAR) if F is not None else np.nan


def delta_se(x, rng):
    """(*): se of the partial sum at tau_min, over the density there."""
    dens, res = weighted_density(np.asarray(x, float))
    if dens is None:
        return np.nan, np.nan
    d = np.clip(dens - estimate_floor(dens, res, "surrogate", rng), 0.0, None)
    tot = d.sum()
    if tot <= 0:
        return np.nan, np.nan
    n = len(d)
    k = int(np.searchsorted(np.cumsum(d), QSTAR * tot))
    if k >= n:
        return np.nan, np.nan
    b = max(3, int(BLOCK * n))
    nb = max(1, (k + b - 1) // b)
    blocks = np.array([d[i * b:min((i + 1) * b, k + 1)].sum()
                       for i in range(nb)], float)
    if len(blocks) < 2:
        return np.nan, np.nan
    se_G = float(np.sqrt(len(blocks)) * np.std(blocks, ddof=1))
    lo, hi = max(0, k - b // 2), min(n, k + b // 2 + 1)
    g = float(np.mean(d[lo:hi])) * n
    if not np.isfinite(se_G) or not np.isfinite(g) or g <= 0:
        return np.nan, np.nan
    return se_G / g, (k + 1.0) / n


def block_bootstrap(x, rng, B=BOOT_WITHIN):
    """Resample the record's OWN residuals in blocks and redo the chain."""
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    n = len(x)
    w = _win(n)
    if w < 7 or n < 200:
        return np.nan
    T = savgol_filter(x, w, 2)
    r = x - T
    b = max(3, int(BLOCK * n))
    starts = np.arange(0, max(1, n - b))
    out = []
    for _ in range(B):
        pieces = [r[s:s + b] for s in rng.choice(starts,
                                                 size=n // b + 1, replace=True)]
        rb = np.concatenate(pieces)[:n]
        v = tau_of(T + rb, rng)
        if np.isfinite(v):
            out.append(v)
    return float(np.std(out, ddof=1)) if len(out) > 3 else np.nan


rng = np.random.default_rng(SEED)

# --- 1. does the first-order formula describe the bootstrap? ----------------
print("standard error of the earliest usable age")
print("the first-order formula against a block bootstrap of the whole chain\n")
print(f"{'rig':<11}{'record':<14}{'measure':<20}{'tau':>7}"
      f"{'formula':>10}{'bootstrap':>11}{'ratio':>7}")
print("-" * 80, flush=True)
checks = []
t0 = time.time()
for rig, fn in RIGS.items():
    p = os.path.join(HERE, fn)
    if not os.path.exists(p):
        continue
    zz = np.load(p)
    f = zz["centres"] / 1000.0
    for b in sorted(k for k in zz.files if k != "centres")[:2]:
        P = zz[b].astype(float)
        for nm, kind, fun in (MEASURES[0], MEASURES[4]):
            x = np.asarray(fun(P, f), float)
            if not np.isfinite(x).all() or len(x) < 200:
                continue
            se_d, t = delta_se(x, rng)
            se_b = block_bootstrap(x, rng)
            if not (np.isfinite(se_d) and np.isfinite(se_b) and se_b > 0):
                continue
            checks.append(dict(rig=rig, unit=b, measure=nm, tau=t,
                               formula=se_d, boot=se_b, ratio=se_d / se_b))
            print(f"{rig:<11}{b:<14}{nm:<20}{t:>7.3f}{se_d:>10.4f}"
                  f"{se_b:>11.4f}{se_d / se_b:>7.2f}", flush=True)
print("-" * 80)
rr = np.array([c["ratio"] for c in checks], float)
print(f"formula over bootstrap: median {np.median(rr):.2f}, "
      f"range {rr.min():.2f} to {rr.max():.2f} over {len(rr)} cells "
      f"({time.time() - t0:.0f}s)")
print()
print("Both estimate the same standard error, so agreement to a factor of a few")
print("is what a first-order formula can be asked for.  The bootstrap is used")
print("for every figure below; the formula's role is to say WHERE the age is")
print("well determined, and it says: wherever the density at that age is high.")

# --- 2. the applied claim, with the fleet resampled -------------------------
print("\n\nthe distribution-minus-amount gap, with the fleet resampled\n",
      flush=True)
if os.path.exists(CACHE):
    per = json.load(open(CACHE))
    print(f"per-unit ages read from cache ({sum(len(v) for v in per.values())} "
          f"cells)")
else:
    per = {}
    t0 = time.time()
    for rig, fn in RIGS.items():
        p = os.path.join(HERE, fn)
        if not os.path.exists(p):
            continue
        zz = np.load(p)
        f = zz["centres"] / 1000.0
        ks = sorted(k for k in zz.files if k != "centres")
        per[rig] = []
        for i, b in enumerate(ks):
            P = zz[b].astype(float)
            row = {"unit": b}
            for nm, kind, fun in MEASURES:
                x = np.asarray(fun(P, f), float)
                row[nm] = (float(tau_of(x, rng))
                           if np.isfinite(x).all() and len(x) >= 200 else None)
            per[rig].append(row)
            print(f"  {rig} {i + 1}/{len(ks)}  ({time.time() - t0:.0f}s)",
                  flush=True)
    json.dump(per, open(CACHE, "w"), indent=2)

DIST = [m[0] for m in MEASURES if m[1] == "distribution"]
AMT = [m[0] for m in MEASURES if m[1] == "amount"]

# The statistic the paper reports is not a paired per-unit difference.  It is
# the difference of two medians over MEASURES, each measure's age being itself a
# median over units.  The bootstrap must therefore resample units and rebuild
# that whole chain, or it would put an interval on a different quantity.
def gap_from(rows, idx):
    sub = [rows[i] for i in idx]

    def med_measure(nm):
        v = [r[nm] for r in sub if r.get(nm) is not None]
        return float(np.median(v)) if v else np.nan

    d = [med_measure(nm) for nm in DIST]
    a = [med_measure(nm) for nm in AMT]
    d = [v for v in d if np.isfinite(v)]
    a = [v for v in a if np.isfinite(v)]
    if not d or not a:
        return np.nan
    return float(np.median(a) - np.median(d))


print(f"\n{'rig':<12}{'units':>7}{'gap':>9}{'95% interval':>20}"
      f"{'excludes zero':>16}")
print("-" * 64)
gaps = []
for rig, rows in per.items():
    n = len(rows)
    if n < 5:
        continue
    g = gap_from(rows, range(n))
    boot = np.array([gap_from(rows, rng.integers(0, n, size=n))
                     for _ in range(BOOT_FLEET)], float)
    boot = boot[np.isfinite(boot)]
    lo, hi = np.percentile(boot, [2.5, 97.5])
    gaps.append(dict(rig=rig, units=n, gap=float(g), lo=float(lo),
                     hi=float(hi), excludes_zero=bool(lo > 0),
                     draws=int(len(boot))))
    print(f"{rig:<12}{n:>7}{g:>9.3f}{f'[{lo:+.3f}, {hi:+.3f}]':>20}"
          f"{('yes' if lo > 0 else 'NO'):>16}")
print("-" * 64)
print("The gap is amount minus distribution, so a positive value means the")
print("distributional indicators are usable earlier.  The interval resamples")
print("bearings and rebuilds the paper's own statistic -- the difference of two")
print("medians over measures -- rather than a paired difference, which is a")
print("different quantity with a different interval.\n")
if gaps and all(g["excludes_zero"] for g in gaps):
    print("The separation survives resampling on both rigs, so the headline")
    print("result is not an artefact of which bearings happened to be tested.")
elif gaps:
    bad = [g["rig"] for g in gaps if not g["excludes_zero"]]
    print(f"The interval includes zero on {', '.join(bad)}, so on that rig the")
    print("separation is not established by these bearings and the paper must")
    print("say so.")

json.dump(dict(delta_vs_boot=checks,
               ratio_median=float(np.median(rr)),
               ratio_min=float(rr.min()), ratio_max=float(rr.max()),
               fleet=gaps, boot_draws=BOOT_FLEET),
          open(os.path.join(HERE, "tau_uncertainty.json"), "w"), indent=2,
          default=float)
