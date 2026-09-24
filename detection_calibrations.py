r"""All three alarm calibrations, run together and stored together.

WHY THIS EXISTS.  The manuscript reports the median detection advantage of
distribution indicators over amount indicators under three surrogates, six
numbers in one sentence.  Only the third surrogate's two numbers had a stored
result: detection_final.json.  The other four could not be reproduced from
anything in this directory, so four numbers in a paper whose stated guarantee is
that every quantity is checked against the run that produced it were not checked
against anything.  This script runs all three surrogates in one pass, with the
same estimator, the same measures, the same tuning and the same seed, and stores
every one of the six.

THE THREE SURROGATES, and why only the third is admissible.

  permutation   permute the whole record.  Destroys the serial correlation the
                persistence rule is sensitive to, so the surrogate is easier
                than real noise and the calibrated threshold comes out far too
                low.
  blocks        shuffle the whole record in blocks.  Keeps the correlation, but
                carries the large late-life excursions with it; moved early they
                look exactly like the fault, so the surrogate fires readily.
  healthy       resample each record's healthy opening in blocks to full length.
                Correlated, realistic, and trend-free by construction.

The criterion is stated before the answers are seen: the surrogate must contain
no degradation and must preserve the correlation the rule is sensitive to.  Only
the third does both.  Reporting all six numbers is what makes that a claim
rather than a preference, since the reader can see how much the choice moved the
answer.
"""
import io
import json
import os

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


# --- the three surrogates ----------------------------------------------------
def sur_permute(x, rng, block=0.05):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    return rng.permutation(x)


def sur_blocks(x, rng, block=0.05):
    """Blocks drawn from the whole record, late-life excursions included."""
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    n = len(x)
    b = max(3, int(block * n))
    if n < b + 2:
        return None
    out = []
    while sum(len(p) for p in out) < n:
        i = int(rng.integers(0, n - b))
        out.append(x[i:i + b])
    return np.concatenate(out)[:n]


def sur_healthy(x, rng, block=0.05):
    """Full-length record made only of the healthy stretch, blocks preserved."""
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


SURROGATES = [("permutation", sur_permute),
              ("blocks", sur_blocks),
              ("healthy", sur_healthy)]


def fa_rate(series, k, rng, sur, reps=10):
    fires = tot = 0
    for x in series:
        for _ in range(reps):
            s = sur(x, rng)
            if s is None:
                continue
            tot += 1
            if np.isfinite(alarm_age(s, k)):
                fires += 1
    return fires / tot if tot else np.nan


def tune(series, rng, sur, target=TARGET_FA, lo=0.2, hi=60.0, iters=14):
    if fa_rate(series, hi, rng, sur) > target:
        return None, np.nan                      # cannot be calibrated at all
    if fa_rate(series, lo, rng, sur) <= target:
        return lo, fa_rate(series, lo, rng, sur)  # the lower bound is not binding
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        if fa_rate(series, mid, rng, sur) > target:
            lo = mid
        else:
            hi = mid
    return hi, fa_rate(series, hi, rng, sur)


RIGS = {"PRONOSTIA": "fineband.npz", "XJTU": "fineband_xjtu.npz"}
out = {}
for sname, sur in SURROGATES:
    out[sname] = {}
    for rig, fn in RIGS.items():
        p = os.path.join(HERE, fn)
        if not os.path.exists(p):
            continue
        rng = np.random.default_rng(SEED)        # same seed for every surrogate
        z = np.load(p)
        f = z["centres"] / 1000.0
        BK = sorted(k for k in z.files if k != "centres")
        per = {}
        print("\n=== %s / %s ===" % (rig, sname))
        print("%-20s%-14s%>7s%13s%10s%11s".replace("%>7s", "%7s")
              % ("indicator", "kind", "k", "false alarm", "detected",
                 "alarm age"))
        print("-" * 76)
        for lab, kind, fun in MEASURES:
            series = [fun(z[b].astype(float), f) for b in BK]
            k, fa = tune(series, rng, sur)
            if k is None:
                print("%-20s%-14s     --   cannot reach the target rate"
                      % (lab, kind))
                continue
            ages = np.array([alarm_age(s, k) for s in series], float)
            per[lab] = dict(kind=kind, k=float(k), fa=float(fa), ages=ages)
            ok = np.isfinite(ages)
            print("%-20s%-14s%7.2f%12.0f%%%7d/%-3d%11.3f"
                  % (lab, kind, k, 100 * fa, ok.sum(), len(ages),
                     np.median(ages[ok]) if ok.any() else np.nan))
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
                gains.append(dict(distribution=da, amount=am,
                                  units=int(m.sum()),
                                  gain=float(np.median(vb["ages"][m]
                                                       - va["ages"][m]))))
        med = float(np.median([g["gain"] for g in gains])) if gains else None
        calibrated = len(per)
        print("  %d of %d indicators calibrated; %d matched pairs; "
              "median gain %s"
              % (calibrated, len(MEASURES), len(gains),
                 "%+.3f" % med if med is not None else "no pair"))
        out[sname][rig] = dict(
            per={k: dict(kind=v["kind"], k=v["k"], fa=v["fa"],
                         detected=int(np.isfinite(v["ages"]).sum()),
                         units=len(v["ages"]),
                         age=float(np.nanmedian(v["ages"])))
                 for k, v in per.items()},
            pairs=gains, median_gain=med,
            calibrated=calibrated, measures=len(MEASURES))

json.dump(out, io.open(os.path.join(HERE, "detection_calibrations.json"), "w",
                       encoding="utf-8"), indent=2, default=float)

print("\n" + "=" * 78)
print("the six numbers the manuscript quotes, all from one run")
print("=" * 78)
print("%-12s %14s %14s" % ("surrogate", "PRONOSTIA", "XJTU-SY"))
for sname, _ in SURROGATES:
    row = ["%+.3f" % out[sname][r]["median_gain"]
           if out[sname].get(r, {}).get("median_gain") is not None
           else "no pair" for r in ("PRONOSTIA", "XJTU")]
    print("%-12s %14s %14s" % (sname, row[0], row[1]))
print("""
Read across a row for what one calibration says and down a column for how much
the calibration choice moves it.  That spread is the finding: a quantity that
changes with a methodological choice which cannot be justified independently is
not a measurement of the indicators.
""")
