"""
Detection, with the false-alarm rate and the detection rate both held fixed.

A first pass found distribution indicators raising the alarm 0.17 to 0.19 of a
lifetime earlier than amount indicators. Two confounds make that comparison
unfair as it stands.

The false-alarm rates were not equal. Thresholds were chosen as the smallest
integer meeting a target, so some indicators ended up well under it and one
above; an indicator allowed more false alarms will naturally alarm sooner.

The detection rates were not equal either, and they differ a lot -- from six
units in sixteen to fourteen. An indicator that only fires on the units where
damage is obvious can post an early median without being useful, because the
hard units are simply absent from its average.

Both are fixed here. The threshold for each indicator is tuned by bisection to
a common false-alarm rate rather than to a common integer, and indicators are
then compared at matched detection rate: for each pair, the alarm ages are taken
only over the units BOTH detected, which removes the selection entirely.
"""
import os
import json
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
EPS = 1e-30
BASE, PERSIST = 0.20, 0.02
TARGET_FA = 0.10
SEED = 60606


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


def block_shuffle(x, rng, frac=0.05):
    """Shuffle the record in blocks, so short-range correlation survives.

    A plain permutation destroys serial correlation entirely, which makes the
    surrogate easier than real noise for a rule that requires an excursion to
    persist: the calibrated threshold then comes out far too low. On PRONOSTIA
    the plain-permutation calibration returned thresholds below one robust
    deviation, at which every indicator alarms almost at once and none can be
    told apart. Blocks of 5% of the record keep the persistence the rule is
    sensitive to while removing the trend.
    """
    n = len(x)
    b = max(3, int(frac * n))
    parts = [x[i:i + b] for i in range(0, n, b)]
    rng.shuffle(parts)
    return np.concatenate(parts)[:n]


def fa_rate(series, k, rng, reps=8):
    fires = tot = 0
    for x in series:
        x = np.asarray(x, float)
        x = x[np.isfinite(x)]
        if len(x) < 60:
            continue
        for _ in range(reps):
            tot += 1
            if np.isfinite(alarm_age(block_shuffle(x, rng), k)):
                fires += 1
    return fires / tot if tot else np.nan


def tune(series, rng, target=TARGET_FA, lo=0.2, hi=40.0, iters=14):
    """Bisect the threshold to a common false-alarm rate, not a common k.

    The lower bound must sit below where every indicator already meets the
    target, or the bisection returns that bound for all of them and the
    comparison silently reverts to a common threshold. A first run at lo = 1.5
    did exactly that on PRONOSTIA.
    """
    if fa_rate(series, hi, rng) > target:
        return None, np.nan
    if fa_rate(series, lo, rng) <= target:
        # even the loosest threshold tried is below target: the bound is not
        # binding and the result would be a common threshold, not a common rate
        return lo, fa_rate(series, lo, rng)
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
    print(f"\n=== {rig} ===  thresholds bisected to a {100*TARGET_FA:.0f}% "
          f"false-alarm rate\n")
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

    # matched comparison: every distribution / amount pair, on the units both
    # detected, so neither benefits from having skipped the difficult ones
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
            g = float(np.median(vb["ages"][m] - va["ages"][m]))
            gains.append(dict(distribution=da, amount=am, units=int(m.sum()),
                              gain=g))
    if gains:
        gs = [g["gain"] for g in gains]
        won = sum(1 for g in gs if g > 0)
        print(f"\n{len(gains)} matched pairs, {won} won by the distribution "
              f"indicator")
        print(f"  median gain {np.median(gs):+.3f} of life, "
              f"range {min(gs):+.3f} to {max(gs):+.3f}")
        best = max(gains, key=lambda g: g["gain"])
        worst = min(gains, key=lambda g: g["gain"])
        print(f"  best  : {best['distribution']} over {best['amount']}"
              f"  {best['gain']:+.3f} on {best['units']} units")
        print(f"  worst : {worst['distribution']} over {worst['amount']}"
              f"  {worst['gain']:+.3f} on {worst['units']} units")
    out[rig] = dict(per={k: dict(kind=v["kind"], k=v["k"], fa=v["fa"],
                                 detected=int(np.isfinite(v["ages"]).sum()),
                                 units=len(v["ages"]),
                                 age=float(np.nanmedian(v["ages"])))
                         for k, v in per.items()},
                    pairs=gains)

print("\nA positive gain means the distribution indicator alarmed earlier on")
print("the units both detected, at the same false-alarm rate. Matching on both")
print("removes the two ways this comparison could otherwise be won unfairly.")
json.dump(out, open(os.path.join(HERE, "detection_matched.json"), "w"),
          indent=2, default=float)
