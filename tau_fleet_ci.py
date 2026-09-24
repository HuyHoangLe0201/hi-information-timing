"""
A confidence interval for the applied claim, on the paper's own statistic.

tau_uncertainty.py computed the per-unit ages but two things about that run must
not be carried forward.  Its inclusion rule required at least two hundred
samples, which is stricter than the rule the paper uses (sixty) and excludes
eight of the fifteen XJTU bearings, whose median life is a hundred and
sixty-one snapshots; an interval built on seven units would describe a different
sample from the one the claim is made on.  And the statistic it resampled was a
paired per-unit difference, whereas the paper reports the difference of two
medians over MEASURES, each measure's age being itself a median over units.
Those are different quantities with different intervals.

Both are corrected here.  The ages are recomputed under the paper's own
inclusion rule and cached, and the bootstrap resamples bearings and rebuilds the
paper's chain end to end.
"""
import os
import json
import time
import numpy as np

from nonparam import info_curve, quantile

HERE = os.path.dirname(os.path.abspath(__file__))
SEED = 20260828
QSTAR = 0.35
MIN_N = 60                     # the paper's rule, not a stricter one
BOOT = 4000
EPS = 1e-30
CACHE = os.path.join(HERE, "tau_percell_v3.json")


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
    ("spectral Gini", "distribution",
     lambda P, f: (2.0 * (np.sort(norm_rows(P), axis=1)
                          * np.arange(1, P.shape[1] + 1)).sum(axis=1))
     / np.clip(np.sort(norm_rows(P), axis=1).sum(axis=1), EPS, None)
     / P.shape[1] - (P.shape[1] + 1.0) / P.shape[1]),
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
RIGS = {"PRONOSTIA": "fineband.npz", "XJTU": "fineband_xjtu.npz"}

rng = np.random.default_rng(SEED)

if os.path.exists(CACHE):
    per = json.load(open(CACHE))
    print(f"per-unit ages from cache: "
          + ", ".join(f"{k} {len(v)}" for k, v in per.items()))
else:
    per, t0 = {}, time.time()
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
            row = {"unit": b, "n": int(P.shape[0])}
            for nm, kind, fun in MEASURES:
                x = np.asarray(fun(P, f), float)
                x = x[np.isfinite(x)]
                v = np.nan
                if len(x) >= MIN_N:
                    F = info_curve(x, rng=rng)
                    if F is not None:
                        v = float(quantile(F, QSTAR))
                row[nm] = None if not np.isfinite(v) else v
            per[rig].append(row)
            print(f"  {rig} {i + 1}/{len(ks)}  n={row['n']}  "
                  f"({time.time() - t0:.0f}s)", flush=True)
    json.dump(per, open(CACHE, "w"), indent=2)

print()
for rig, rows in per.items():
    miss = sum(1 for r in rows for nm in DIST + AMT if r.get(nm) is None)
    print(f"{rig}: {len(rows)} bearings, lives {min(r['n'] for r in rows)} to "
          f"{max(r['n'] for r in rows)}, {miss} missing cells of "
          f"{len(rows) * len(DIST + AMT)}")
print()


def gap_from(rows, idx):
    """The paper's statistic: median over amount measures minus median over
    distribution measures, each measure's age a median over the units drawn."""
    sub = [rows[i] for i in idx]

    def med_measure(nm):
        v = [r[nm] for r in sub if r.get(nm) is not None]
        return float(np.median(v)) if v else np.nan

    d = [v for v in (med_measure(nm) for nm in DIST) if np.isfinite(v)]
    a = [v for v in (med_measure(nm) for nm in AMT) if np.isfinite(v)]
    if not d or not a:
        return np.nan
    return float(np.median(a) - np.median(d))


print("the distribution-minus-amount gap, bearings resampled\n")
print(f"{'rig':<12}{'units':>7}{'gap':>9}{'95% interval':>21}"
      f"{'excludes zero':>16}")
print("-" * 65)
gaps = []
for rig, rows in per.items():
    n = len(rows)
    g = gap_from(rows, range(n))
    if not np.isfinite(g):
        continue
    boot = np.array([gap_from(rows, rng.integers(0, n, size=n))
                     for _ in range(BOOT)], float)
    boot = boot[np.isfinite(boot)]
    lo, hi = np.percentile(boot, [2.5, 97.5])
    gaps.append(dict(rig=rig, units=n, gap=float(g), lo=float(lo),
                     hi=float(hi), excludes_zero=bool(lo > 0),
                     draws=int(len(boot))))
    print(f"{rig:<12}{n:>7}{g:>9.3f}{f'[{lo:+.3f}, {hi:+.3f}]':>21}"
          f"{('yes' if lo > 0 else 'NO'):>16}")
print("-" * 65)
print("A positive gap means the distributional indicators are usable earlier.")
print("The interval resamples bearings and rebuilds the paper's own statistic,")
print("so it answers the question the claim invites: would another draw of")
print("bearings from the same rig have shown the same thing?\n")

if gaps and all(g["excludes_zero"] for g in gaps):
    print("The separation survives on both rigs.  The headline result is not an")
    print("artefact of which bearings happened to be tested.")
elif gaps:
    bad = [g["rig"] for g in gaps if not g["excludes_zero"]]
    print(f"The interval includes zero on {', '.join(bad)}; on that rig the")
    print("separation is not established by these bearings.")

# the point estimates the paper quotes, for comparison
print()
for rig, rows in per.items():
    def med_measure(nm):
        v = [r[nm] for r in rows if r.get(nm) is not None]
        return float(np.median(v)) if v else np.nan
    d = np.median([med_measure(nm) for nm in DIST])
    a = np.median([med_measure(nm) for nm in AMT])
    print(f"{rig}: distribution {d:.3f}, amount {a:.3f}, gap {a - d:.3f}")

json.dump(dict(min_n=MIN_N, boot=BOOT, fleet=gaps,
               per_unit={k: v for k, v in per.items()}),
          open(os.path.join(HERE, "tau_fleet_ci.json"), "w"), indent=2,
          default=float)
