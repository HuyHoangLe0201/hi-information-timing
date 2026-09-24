"""
Does the applied result depend on the demand it is quoted at?

Every age in this paper is read at one budget fraction, q = 0.35, and that value
is never swept.  The headline claim -- distributional indicators usable a third
to two thirds of a lifetime before amount indicators -- is therefore quoted at a
single demand, and a reader is entitled to ask whether the conclusion is a
property of the indicators or of that choice.  It is a cheap question to answer
and it has not been answered.

The demand enters only through the quantile at which the normalised curve is
read, so caching the CURVE rather than one of its quantiles makes the sweep
almost free: one pass over the records, then every demand read off the stored
curves.  The previous cache stored only q = 0.35 and had to be rebuilt for this.

Reported at each demand: the two medians, their difference, and a bootstrap
interval over bearings, so the question is answered with the same statistic and
the same uncertainty as the headline number rather than with a bare point
estimate.
"""
import os
import json
import time
import numpy as np

from nonparam import info_curve

HERE = os.path.dirname(os.path.abspath(__file__))
SEED = 20260828
MIN_N = 60                       # the paper's own inclusion rule
GRIDN = 1000
BOOT = 2000
QS = (0.05, 0.10, 0.20, 0.35, 0.50, 0.65, 0.80)
EPS = 1e-30
CACHE = os.path.join(HERE, "curves_cache_v2.npz")


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
DIST = [m[0] for m in MEASURES if m[1] == "distribution"]
AMT = [m[0] for m in MEASURES if m[1] == "amount"]
RIGS = {"PRONOSTIA": "fineband.npz", "XJTU": "fineband_xjtu.npz"}
GRID = (np.arange(GRIDN) + 1.0) / GRIDN

rng = np.random.default_rng(SEED)

if os.path.exists(CACHE):
    zc = np.load(CACHE, allow_pickle=True)
    curves = {k: zc[k] for k in zc.files if k != "index"}
    index = json.loads(str(zc["index"]))
    print(f"curves from cache: {len(curves)} cells")
else:
    curves, index, t0 = {}, [], time.time()
    for rig, fn in RIGS.items():
        p = os.path.join(HERE, fn)
        if not os.path.exists(p):
            continue
        zz = np.load(p)
        f = zz["centres"] / 1000.0
        ks = sorted(k for k in zz.files if k != "centres")
        for i, b in enumerate(ks):
            P = zz[b].astype(float)
            for nm, kind, fun in MEASURES:
                x = np.asarray(fun(P, f), float)
                x = x[np.isfinite(x)]
                if len(x) < MIN_N:
                    continue
                F = info_curve(x, rng=rng)
                if F is None or not np.isfinite(F).all():
                    continue
                key = f"{rig}||{b}||{nm}"
                tau = (np.arange(len(F)) + 1.0) / len(F)
                curves[key] = np.interp(GRID, tau, np.asarray(F, float))
                index.append(dict(rig=rig, unit=b, measure=nm, kind=kind))
            print(f"  {rig} {i + 1}/{len(ks)}  ({time.time() - t0:.0f}s)",
                  flush=True)
    np.savez_compressed(CACHE, index=json.dumps(index), **curves)

by_rig = {}
for e in index:
    by_rig.setdefault(e["rig"], {}).setdefault(e["unit"], {})[e["measure"]] = \
        f"{e['rig']}||{e['unit']}||{e['measure']}"
for rig, units in by_rig.items():
    print(f"{rig}: {len(units)} bearings, "
          f"{sum(len(v) for v in units.values())} cells")
print()


def age(key, q):
    F = curves[key]
    k = int(np.searchsorted(F, q))
    return GRID[k] if k < len(F) else np.nan


def gap_at(units, names, q, idx):
    """The paper's statistic at demand q, over the bearings indexed by idx."""
    us = [names[i] for i in idx]

    def med_measure(nm):
        v = [age(units[u][nm], q) for u in us if nm in units[u]]
        v = [x for x in v if np.isfinite(x)]
        return float(np.median(v)) if v else np.nan

    d = [v for v in (med_measure(nm) for nm in DIST) if np.isfinite(v)]
    a = [v for v in (med_measure(nm) for nm in AMT) if np.isfinite(v)]
    if not d or not a:
        return np.nan, np.nan, np.nan
    return float(np.median(d)), float(np.median(a)), \
        float(np.median(a) - np.median(d))


print("the gap across demands, with bearings resampled at each\n")
print(f"{'rig':<11}{'q':>6}{'distribution':>14}{'amount':>9}{'gap':>8}"
      f"{'95% interval':>21}{'sign':>7}")
print("-" * 76)
rows = []
for rig, units in by_rig.items():
    names = sorted(units)
    n = len(names)
    for q in QS:
        d, a, g = gap_at(units, names, q, range(n))
        if not np.isfinite(g):
            continue
        boot = np.array([gap_at(units, names, q, rng.integers(0, n, size=n))[2]
                         for _ in range(BOOT)], float)
        boot = boot[np.isfinite(boot)]
        lo, hi = np.percentile(boot, [2.5, 97.5])
        rows.append(dict(rig=rig, q=q, dist=d, amount=a, gap=g,
                         lo=float(lo), hi=float(hi),
                         positive=bool(lo > 0)))
        print(f"{rig:<11}{q:>6.2f}{d:>14.3f}{a:>9.3f}{g:>8.3f}"
              f"{f'[{lo:+.3f}, {hi:+.3f}]':>21}"
              f"{('yes' if lo > 0 else 'no'):>7}")
    print()
print("-" * 76)

for rig in by_rig:
    r = [x for x in rows if x["rig"] == rig]
    if not r:
        continue
    pos = sum(x["positive"] for x in r)
    gs = [x["gap"] for x in r]
    print(f"{rig}: gap positive and its interval clear of zero at {pos} of "
          f"{len(r)} demands;")
    print(f"  the gap itself runs {min(gs):.3f} to {max(gs):.3f} across them")
print()
allr = rows
if allr and all(x["gap"] > 0 for x in allr):
    print("The sign of the effect does not depend on the demand: at every")
    print("fraction tested, on both rigs, the distributional indicators become")
    print("usable first.  Its SIZE does depend on the demand, which it must,")
    print("since a demand near zero is met almost at once by anything and a")
    print("demand near one by nothing until the end.")
else:
    bad = [(x["rig"], x["q"]) for x in allr if x["gap"] <= 0]
    print(f"The sign reverses at {bad}, so the conclusion is specific to the")
    print("demand and the paper must say at which demands it holds.")

json.dump(dict(qs=list(QS), min_n=MIN_N, boot=BOOT, rows=rows),
          open(os.path.join(HERE, "qstar_sweep.json"), "w"), indent=2,
          default=float)
