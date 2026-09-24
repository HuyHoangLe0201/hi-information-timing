"""
The estimator's free parameters, swept.

Section 2 says the framework has no free parameter, and of the framework that is
true: the demand q is the user's, and Section 8 shows the applied result holds at
every q tested.  The ESTIMATOR is another matter.  It carries two bandwidths --
the Savitzky-Golay window used for the trend and its derivative, and the window
of the rolling noise scale -- both fixed at 8 per cent of life and neither ever
swept for anything except the linearisation radius of Section 6.6.  Every number
in this paper is computed at that setting, and a reader is entitled to ask how
much of the applied result is a property of it.

Both are swept here, one at a time, and the quantity watched is the one the paper
reports: the difference between the median age of the amount indicators and that
of the distributional ones, rebuilt from scratch at each setting on both rigs.

A sweep like this can fail in two ways and they are not equally serious.  If the
gap changes size, that is expected -- a wider trend window smooths away early
structure and a narrower one admits more noise into the derivative, so the ages
must move.  If the gap changes SIGN, or approaches zero at a bandwidth as
defensible as the one chosen, the applied result is a property of the choice and
the paper has to say so.
"""
import os
import json
import time
import numpy as np

import pipeline
import nonparam
from nonparam import info_curve, quantile

HERE = os.path.dirname(os.path.abspath(__file__))
SEED = 20260828
QSTAR = 0.35
MIN_N = 60
EPS = 1e-30
TREND_BW = (0.04, 0.06, 0.08, 0.12, 0.16)
LOCAL_BW = (0.04, 0.08, 0.16)


def norm_rows(P):
    return P / np.clip(P.sum(axis=1, keepdims=True), EPS, None)


def gini(P):
    p = np.sort(norm_rows(P), axis=1)
    n = p.shape[1]
    idx = np.arange(1, n + 1)
    return (2 * (p * idx).sum(axis=1)) / np.clip(p.sum(axis=1), EPS, None) \
        / n - (n + 1) / n


def top_share(P, k=8):
    return np.sort(norm_rows(P), axis=1)[:, -k:].sum(axis=1)


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
    # the spectral Gini and the top-8-band share were added so that the sweep
    # reads the same ten measures as Table 5; the definitions are those of
    # distribution_test.py, which built the table
    ("spectral Gini", "distribution", lambda P, f: gini(P)),
    ("top-8-band share", "distribution", lambda P, f: top_share(P)),
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

DATA = {}
for rig, fn in RIGS.items():
    p = os.path.join(HERE, fn)
    if not os.path.exists(p):
        continue
    zz = np.load(p)
    f = zz["centres"] / 1000.0
    DATA[rig] = [(b, zz[b].astype(float), f)
                 for b in sorted(k for k in zz.files if k != "centres")]
print(", ".join(f"{k}: {len(v)} bearings" for k, v in DATA.items()) + "\n")


def gap_at_setting(rng):
    """The paper's statistic on both rigs, at whatever bandwidths are set."""
    out = {}
    for rig, recs in DATA.items():
        per = {}
        for nm, kind, fun in MEASURES:
            vals = []
            for b, P, f in recs:
                x = np.asarray(fun(P, f), float)
                x = x[np.isfinite(x)]
                if len(x) < MIN_N:
                    continue
                F = info_curve(x, rng=rng)
                if F is None:
                    continue
                v = quantile(F, QSTAR)
                if np.isfinite(v):
                    vals.append(float(v))
            if vals:
                per[nm] = float(np.median(vals))
        d = [per[n] for n in DIST if n in per]
        a = [per[n] for n in AMT if n in per]
        if not d or not a:
            continue
        out[rig] = dict(dist=float(np.median(d)), amount=float(np.median(a)),
                        gap=float(np.median(a) - np.median(d)))
    return out


def set_bw(trend=None, local=None):
    """Both bandwidths live as module globals; _win reads its one at call time,
    while nonparam imported LOCAL_BW by value and needs its own copy set."""
    if trend is not None:
        pipeline.SMOOTH_FRAC = trend
    if local is not None:
        pipeline.LOCAL_BW = local
        nonparam.LOCAL_BW = local


rows, t0 = [], time.time()
print("sweeping the trend bandwidth, local scale held at 0.08\n")
print(f"{'trend bw':>10}{'rig':<12}{'distribution':>14}{'amount':>9}"
      f"{'gap':>8}")
print("-" * 54)
for bw in TREND_BW:
    set_bw(trend=bw, local=0.08)
    g = gap_at_setting(np.random.default_rng(SEED))
    for rig, v in g.items():
        rows.append(dict(kind="trend", bw=bw, rig=rig, **v))
        print(f"{bw:>10.2f}{rig:<12}{v['dist']:>14.3f}{v['amount']:>9.3f}"
              f"{v['gap']:>8.3f}")
    print(f"{'':>10}({time.time() - t0:.0f}s)", flush=True)
print("-" * 54)

print("\nsweeping the local-scale bandwidth, trend held at 0.08\n")
print(f"{'local bw':>10}{'rig':<12}{'distribution':>14}{'amount':>9}"
      f"{'gap':>8}")
print("-" * 54)
for bw in LOCAL_BW:
    set_bw(trend=0.08, local=bw)
    g = gap_at_setting(np.random.default_rng(SEED))
    for rig, v in g.items():
        rows.append(dict(kind="local", bw=bw, rig=rig, **v))
        print(f"{bw:>10.2f}{rig:<12}{v['dist']:>14.3f}{v['amount']:>9.3f}"
              f"{v['gap']:>8.3f}")
    print(f"{'':>10}({time.time() - t0:.0f}s)", flush=True)
print("-" * 54)
set_bw(trend=0.08, local=0.08)

print()
for kind in ("trend", "local"):
    r = [x for x in rows if x["kind"] == kind]
    if not r:
        continue
    gs = np.array([x["gap"] for x in r])
    print(f"{kind} bandwidth: gap runs {gs.min():.3f} to {gs.max():.3f} over "
          f"{len(r)} rig-settings, all positive: {bool((gs > 0).all())}")
allg = np.array([x["gap"] for x in rows])
print()
if (allg > 0).all():
    print("The sign of the applied result does not depend on either bandwidth.")
    print("Its size does, which it must: a wider trend window smooths away early")
    print("structure and a narrower one admits noise into the derivative, so the")
    print("ages move.  What matters is that no defensible setting brings the gap")
    print("to zero or reverses it.")
    ref = [x for x in rows if x["kind"] == "trend" and abs(x["bw"] - 0.08) < 1e-9]
    for x in ref:
        same = [y for y in rows if y["rig"] == x["rig"]]
        gg = np.array([y["gap"] for y in same])
        print(f"  {x['rig']}: working setting gives {x['gap']:.3f}, the sweep "
              f"{gg.min():.3f} to {gg.max():.3f}")
else:
    bad = [(x["kind"], x["bw"], x["rig"], x["gap"]) for x in rows
           if x["gap"] <= 0]
    print(f"The gap is not positive at {bad}, so the applied result is a")
    print("property of the bandwidth and the paper must state at which settings")
    print("it holds.")

json.dump(dict(qstar=QSTAR, trend_bw=list(TREND_BW), local_bw=list(LOCAL_BW),
               rows=rows,
               gap_min=float(allg.min()), gap_max=float(allg.max()),
               all_positive=bool((allg > 0).all())),
          open(os.path.join(HERE, "bandwidth_sweep.json"), "w"), indent=2,
          default=float)
