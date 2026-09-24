"""
Does the out-of-fold scale leak across time, and is purging the cure?

The noise scale is taken from out-of-fold residuals: ten interleaved folds, and
the local-quadratic trend at sample i is fitted without i's fold.  That removes
i itself but keeps i-1 and i+1, and Section 6.3 measures a lag-one correlation
of up to 0.399.  A smoother that sees the neighbours of a serially correlated
sample partly sees the sample, so the residual can understate the noise and the
interleaved fold is not a temporally independent hold-out.

The control purges the neighbourhood: at every sample the trend is fitted with
every sample within m of i removed -- the h-block form of cross-validation for
dependent data -- and everything downstream (local scale, noise floor, curve,
age, family gap) is rebuilt unchanged.  m = 3 covers the largest integrated
autocorrelation time measured (2.33); m = 10 is generous.

The purge must widen the window by m on each side (pipeline.oof_trend_purged).
Without that, a record of 120 samples has a half-window of 4, the purge removes
7 of its 9 samples, the fit falls back to the observation itself, and the
residual is identically zero: the first version of this control did exactly that
on three XJTU-SY records.  The fallbacks are counted for both forms.

Part 2 asks which scheme is closer to the truth, on synthetic records whose
noise scale is known, at the record lengths the corpus actually has.

    python oof_control.py   ->  oof_control.json   (about a quarter of an hour)
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
RIGS = {"PRONOSTIA": "fineband.npz", "XJTU": "fineband_xjtu.npz"}


def norm_rows(P):
    return P / np.clip(P.sum(axis=1, keepdims=True), EPS, None)


def gini(P):
    p = np.sort(norm_rows(P), axis=1)
    n = p.shape[1]
    idx = np.arange(1, n + 1)
    return (2 * (p * idx).sum(axis=1)) / np.clip(p.sum(axis=1), EPS, None) \
        / n - (n + 1) / n


# the ten measures of Table 5, defined as in distribution_test.py
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
    ("spectral Gini", "distribution", lambda P, f: gini(P)),
    ("top-8-band share", "distribution",
     lambda P, f: np.sort(norm_rows(P), axis=1)[:, -8:].sum(axis=1)),
    ("high/low band ratio", "distribution",
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

INTERLEAVED = pipeline.oof_trend


def purged_trend(m):
    """Local quadratic at each i, |j - i| <= m removed, window widened by m."""
    return lambda D, half: pipeline.oof_trend_purged(D, half, m)


def unwidened_fallbacks(n, m):
    """Samples the purge could not fit if the window were NOT widened."""
    half = pipeline._win(n) // 2
    idx = np.arange(n)
    c = 0
    for i in range(n):
        lo, hi = max(0, i - half), min(n, i + half + 1)
        c += int((np.abs(idx[lo:hi] - i) > m).sum() < 6)
    return c


DATA = {}
for rig, fn in RIGS.items():
    p = os.path.join(HERE, fn)
    if os.path.exists(p):
        zz = np.load(p)
        f = zz["centres"] / 1000.0
        DATA[rig] = [(b, zz[b].astype(float), f)
                     for b in sorted(k for k in zz.files if k != "centres")]


def run(variant, rng):
    nonparam.oof_trend = variant
    out, scales = {}, []
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
        d = float(np.median([per[n] for n in DIST if n in per]))
        a = float(np.median([per[n] for n in AMT if n in per]))
        out[rig] = dict(per=per, dist=d, amount=a, gap=a - d)
    nonparam.oof_trend = INTERLEAVED
    return out


def scale_ratio(m):
    """Median ratio of the purged residual scale to the interleaved one."""
    r = []
    purge = purged_trend(m)
    for rig, recs in DATA.items():
        for nm, kind, fun in MEASURES:
            for b, P, f in recs:
                x = np.asarray(fun(P, f), float)
                x = x[np.isfinite(x)]
                n = len(x)
                if n < MIN_N:
                    continue
                s = pipeline.robust_scale(x)
                if not np.isfinite(s) or s <= 0:
                    continue
                D = (x - np.median(x)) / s
                w = pipeline._win(n)
                a = pipeline.robust_scale(D - INTERLEAVED(D, w // 2))
                c = pipeline.robust_scale(D - purge(D, w // 2))
                if a > 0 and c > 0 and np.isfinite(a) and np.isfinite(c):
                    r.append(c / a)
    return (float(np.median(r)), float(np.percentile(r, 10)),
            float(np.percentile(r, 90)), len(r))


def ar1(n, phi, sigma, r):
    e = r.normal(0, sigma * np.sqrt(1 - phi ** 2), n)
    out = np.empty(n)
    out[0] = r.normal(0, sigma)
    for k in range(1, n):
        out[k] = phi * out[k - 1] + e[k]
    return out


def synthetic(rng, reps=40, sigma=0.05):
    """Scale estimate over the true scale: 1 is right, below 1 understates."""
    rows = []
    schemes = (("interleaved", INTERLEAVED), ("purged m=1", purged_trend(1)),
               ("purged m=3", purged_trend(3)), ("purged m=10", purged_trend(10)))
    for n in (120, 240, 800, 2400):
        for phi in (0.0, 0.2, 0.4):
            acc = {k: [] for k, _ in schemes}
            for _ in range(reps):
                t = np.linspace(1e-3, 1.0, n)
                D = 3.0 * t ** 2 + ar1(n, phi, sigma, rng)
                half = pipeline._win(n) // 2
                for k, fn in schemes:
                    acc[k].append(pipeline.robust_scale(D - fn(D, half)) / sigma)
            row = dict(n=n, phi=phi, **{k: float(np.median(v))
                                        for k, v in acc.items()})
            rows.append(row)
            print("  n %5d phi %.1f  " % (n, phi)
                  + "  ".join("%s %.3f" % (k, row[k]) for k, _ in schemes),
                  flush=True)
    return rows


if __name__ == "__main__":
    t0 = time.time()
    res, fb = {}, {}
    for name, var in (("interleaved", INTERLEAVED), ("purged m=3", purged_trend(3)),
                      ("purged m=10", purged_trend(10))):
        f0 = list(pipeline.FALLBACKS)
        res[name] = run(var, np.random.default_rng(SEED))
        if name != "interleaved":
            fb[name] = dict(unfitted=pipeline.FALLBACKS[0] - f0[0],
                            samples=pipeline.FALLBACKS[1] - f0[1])
        g = {r: round(v["gap"], 3) for r, v in res[name].items()}
        print(f"{name:<14} gaps {g}  ({time.time() - t0:.0f}s)", flush=True)
    # what the purge would leave unfitted without the widening, per rig
    unw = {}
    for m in (3, 10):
        for rig, recs in DATA.items():
            tot = sum(len(P) for _, P, _ in recs if len(P) >= MIN_N)
            bad = [(b, len(P)) for b, P, _ in recs if len(P) >= MIN_N
                   and unwidened_fallbacks(len(P), m) == len(P)]
            unw[f"m={m} {rig}"] = dict(
                unfitted=sum(unwidened_fallbacks(len(P), m)
                             for _, P, _ in recs if len(P) >= MIN_N),
                samples=tot, records_wholly_unfitted=bad)
    ratios = {}
    for m in (3, 10):
        med, p10, p90, k = scale_ratio(m)
        ratios[f"purged m={m}"] = dict(median=med, p10=p10, p90=p90, cells=k)
        print(f"scale, purged m={m} over interleaved: median {med:.4f}, "
              f"10th {p10:.4f}, 90th {p90:.4f} over {k} record-measure cells")
    shifts = {}
    for name in ("purged m=3", "purged m=10"):
        dd = []
        for rig in res[name]:
            for nm, v in res[name][rig]["per"].items():
                base = res["interleaved"][rig]["per"].get(nm)
                if base is not None:
                    dd.append(abs(v - base))
        shifts[name] = dict(median=float(np.median(dd)), max=float(np.max(dd)))
        print(f"{name}: per-measure age moves by a median {np.median(dd):.4f}, "
              f"at most {np.max(dd):.4f}")
    print("known-scale records:")
    syn = synthetic(np.random.default_rng(SEED + 1))
    json.dump(dict(qstar=QSTAR, variants=res, scale_ratio=ratios,
                   age_shift=shifts, fallbacks=fb, unwidened=unw,
                   synthetic=syn),
              open(os.path.join(HERE, "oof_control.json"), "w"), indent=2)
