"""
Detection time, with a null that is actually a null.

Two surrogates were tried and both are wrong, in opposite directions, and the
choice matters enough that it changes the answer.

A plain permutation destroys serial correlation. The alarm rule requires an
excursion to persist, and persistence is exactly what permutation removes, so
the surrogate is easier than real noise and the calibrated threshold comes out
absurdly low -- below one robust deviation on PRONOSTIA, at which every
indicator alarms almost immediately and none can be distinguished.

Shuffling in blocks keeps the correlation but keeps something else too: the
large-amplitude blocks from late life. Moved early, they are indistinguishable
from the fault the rule is looking for, so the surrogate always fires and half
the indicators cannot be calibrated at all.

The null has to contain the record's noise and none of its degradation. Both are
available: the early part of each record is the healthy part -- it is what the
baseline is taken from -- so resampling THAT in blocks to full length gives a
surrogate with realistic, correlated, healthy noise and no trend by
construction. Choosing between surrogates by which gives the nicer answer would
be choosing the null to fit the result, so the criterion is stated first: the
surrogate must contain no degradation and must preserve the correlation the
rule is sensitive to. Only the third does both.
"""
import os
import json
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
EPS = 1e-30
BASE, PERSIST = 0.20, 0.02
TARGET_FA = 0.10
SEED = 31337


def norm(P):
    return P / np.clip(P.sum(axis=1, keepdims=True), EPS, None)


MEASURES = [
    ("spectral entropy", "distribution",
     lambda P, f: -(norm(P) * np.log(np.clip(norm(P), EPS, None))).sum(axis=1)),
    ("spectral centroid", "distribution", lambda P, f: (norm(P) * f).sum(axis=1)),
    ("spectral spread", "distribution",
     lambda P, f: np.sqrt((norm(P) * (f - (norm(P) * f).sum(axis=1,
                                                            keepdims=True)) ** 2
                           ).sum(axis=1))),
    ("high/low ratio", "distribution",
     lambda P, f: P[:, -8:].sum(axis=1) / np.clip(P[:, :8].sum(axis=1), EPS, None)),
    ("total power", "amount", lambda P, f: P.sum(axis=1)),
    ("peak band power", "amount", lambda P, f: P.max(axis=1)),
    ("high-band power", "amount", lambda P, f: P[:, -8:].sum(axis=1)),
    ("log total power", "amount",
     lambda P, f: np.log(np.clip(P.sum(axis=1), EPS, None))),
]


def alarm_age(x, k):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    n = len(x)
    if n < 60:
        return np.nan
    b = x[:max(5, int(BASE * n))]
    mu = float(np.median(b))
    s = float(np.median(np.abs(b - mu)) * 1.4826)
    if not np.isfinite(s) or s <= 0:
        return np.nan
    out = np.abs(x - mu) > k * s
    need = max(3, int(PERSIST * n))
    run = 0
    for i in range(int(BASE * n), n):
        run = run + 1 if out[i] else 0
        if run >= need:
            return (i - need + 2) / n
    return np.nan


def healthy_surrogate(x, rng, block=0.05):
    """A full-length record made only of the healthy stretch, blocks preserved."""
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    n = len(x)
    h = x[:max(10, int(BASE * n))]
    b = max(3, int(block * n))
    if len(h) < b + 2:
        return None
    out = []
    while sum(len(p) for p in out) < n:
        i = int(rng.integers(0, len(h) - b))
        out.append(h[i:i + b])
    return np.concatenate(out)[:n]


def fa_rate(series, k, rng, reps=10):
    fires = tot = 0
    for x in series:
        for _ in range(reps):
            s = healthy_surrogate(x, rng)
            if s is None:
                continue
            tot += 1
            if np.isfinite(alarm_age(s, k)):
                fires += 1
    return fires / tot if tot else np.nan


def tune(series, rng, target=TARGET_FA, lo=0.2, hi=60.0, iters=14):
    if fa_rate(series, hi, rng) > target:
        return None, np.nan
    if fa_rate(series, lo, rng) <= target:
        return lo, fa_rate(series, lo, rng)     # bound not binding
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        if fa_rate(series, mid, rng) > target:
            lo = mid
        else:
            hi = mid
    return hi, fa_rate(series, hi, rng)


RIGS = {"PRONOSTIA": "fineband.npz", "XJTU": "fineband_xjtu.npz"}
rng = np.random.default_rng(SEED)
out = {}
for rig, fn in RIGS.items():
    p = os.path.join(HERE, fn)
    if not os.path.exists(p):
        continue
    z = np.load(p)
    f = z["centres"] / 1000.0
    BK = sorted(k for k in z.files if k != "centres")
    per = {}
    print(f"\n=== {rig} ===  null: the healthy stretch, resampled in blocks\n")
    print(f"{'indicator':<20}{'kind':<14}{'k':>7}{'false alarm':>13}"
          f"{'detected':>10}{'alarm age':>11}")
    print("-" * 76)
    for lab, kind, fun in MEASURES:
        series = [fun(z[b].astype(float), f) for b in BK]
        k, fa = tune(series, rng)
        if k is None:
            print(f"{lab:<20}{kind:<14}     --   cannot reach the target rate")
            continue
        ages = np.array([alarm_age(s, k) for s in series], float)
        per[lab] = dict(kind=kind, k=float(k), fa=float(fa), ages=ages)
        ok = np.isfinite(ages)
        print(f"{lab:<20}{kind:<14}{k:>7.2f}{100*fa:>12.0f}%"
              f"{ok.sum():>7}/{len(ages):<3}"
              f"{np.median(ages[ok]) if ok.any() else np.nan:>11.3f}")
    print("-" * 76)

    gains = []
    for da, va in per.items():
        if va["kind"] != "distribution":
            continue
        for am, vb in per.items():
            if vb["kind"] != "amount":
                continue
            m = np.isfinite(va["ages"]) & np.isfinite(vb["ages"])
            if m.sum() < 6:
                continue
            gains.append(dict(distribution=da, amount=am, units=int(m.sum()),
                              gain=float(np.median(vb["ages"][m]
                                                   - va["ages"][m]))))
    if gains:
        gs = [g["gain"] for g in gains]
        won = sum(1 for g in gs if g > 0)
        print(f"\n{len(gains)} matched pairs, {won} won by the distribution "
              f"indicator")
        print(f"  median gain {np.median(gs):+.3f} of life, "
              f"range {min(gs):+.3f} to {max(gs):+.3f}")
    else:
        print("\nno pair had enough jointly detected units to compare")
    out[rig] = dict(per={k: dict(kind=v["kind"], k=v["k"], fa=v["fa"],
                                 detected=int(np.isfinite(v["ages"]).sum()),
                                 units=len(v["ages"]),
                                 age=float(np.nanmedian(v["ages"])))
                         for k, v in per.items()},
                    pairs=gains)

json.dump(out, open(os.path.join(HERE, "detection_final.json"), "w"), indent=2,
          default=float)
print("\nThe surrogate was fixed on stated criteria, not chosen for its answer;")
print("the two rejected ones are described in the source.")
