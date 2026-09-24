"""
The right task: when does an alarm first fire?

An attempt to turn the distribution finding into a demonstrated gain used RUL
regression and failed, because on these bearing fleets no indicator beats
predicting the fleet mean -- seventeen units with a twelvefold spread of life
cannot support the regression, whatever indicator is fed to it.

That was the wrong task. The claim is that distribution indicators become usable
EARLIER, which is a statement about detection, not about regression. Detection is
also what a condition-monitoring system actually does: a baseline is taken from
early life and an alarm fires when the indicator leaves it. Each unit is then its
own control, so no fleet is needed and the datasets stop being the obstacle.

For each unit and indicator: a baseline mean and robust scale are taken from the
first 20% of the record, and the alarm fires at the first age where the
indicator has stayed more than k robust deviations away for a sustained run.
Requiring persistence is what makes it a fair test -- a single excursion is not
a detection, and without that requirement a noisy indicator wins by accident.

The false-alarm rate is controlled and reported: the same rule is run on
surrogates with the trend removed, and any indicator that fires on those has its
threshold raised until it does not, so every indicator is compared at the same
false-alarm rate rather than at the same nominal k.
"""
import os
import json
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
EPS = 1e-30
BASE = 0.20          # baseline taken from the first fifth of life
PERSIST = 0.02       # an alarm must hold for this fraction of the record
SEED = 5150
TARGET_FA = 0.10     # tolerated false-alarm rate on trend-free surrogates


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
    ("spectral Gini", "distribution",
     lambda P, f: (2 * (np.sort(norm(P), axis=1)
                        * np.arange(1, P.shape[1] + 1)).sum(axis=1))
     / P.shape[1] - (P.shape[1] + 1) / P.shape[1]),
    ("high/low ratio", "distribution",
     lambda P, f: P[:, -8:].sum(axis=1) / np.clip(P[:, :8].sum(axis=1), EPS, None)),
    ("total power", "amount", lambda P, f: P.sum(axis=1)),
    ("peak band power", "amount", lambda P, f: P.max(axis=1)),
    ("high-band power", "amount", lambda P, f: P[:, -8:].sum(axis=1)),
    ("log total power", "amount",
     lambda P, f: np.log(np.clip(P.sum(axis=1), EPS, None))),
]


def alarm_age(x, k):
    """First age at which the indicator has left its baseline and stayed out."""
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


def false_alarm_rate(series, k, rng, reps=6):
    """How often the rule fires on records whose trend has been removed."""
    fires = tot = 0
    for x in series:
        x = np.asarray(x, float)
        x = x[np.isfinite(x)]
        if len(x) < 60:
            continue
        for _ in range(reps):
            tot += 1
            if np.isfinite(alarm_age(rng.permutation(x), k)):
                fires += 1
    return fires / tot if tot else np.nan


def calibrate(series, rng):
    """Smallest k whose false-alarm rate on surrogates is within target."""
    for k in (3, 4, 5, 6, 8, 10, 14, 20):
        if false_alarm_rate(series, k, rng) <= TARGET_FA:
            return k
    return None


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
    rows = []
    for lab, kind, fun in MEASURES:
        series = [fun(z[b].astype(float), f) for b in BK]
        k = calibrate(series, rng)
        if k is None:
            rows.append(dict(measure=lab, kind=kind, k=None, fa=np.nan,
                             detected=0, units=len(series), age=np.nan))
            continue
        ages = [alarm_age(s, k) for s in series]
        ok = [a for a in ages if np.isfinite(a)]
        rows.append(dict(measure=lab, kind=kind, k=k,
                         fa=false_alarm_rate(series, k, rng),
                         detected=len(ok), units=len(series),
                         age=float(np.median(ok)) if ok else np.nan))
    out[rig] = rows

for rig, rows in out.items():
    print(f"\n=== {rig} ===  alarm age, thresholds set to a common "
          f"false-alarm rate\n")
    print(f"{'indicator':<22}{'kind':<14}{'k':>4}{'false alarm':>13}"
          f"{'detected':>10}{'alarm age':>11}")
    print("-" * 74)
    for r in sorted(rows, key=lambda r: (r["age"] if np.isfinite(r["age"])
                                         else 9)):
        a = "     n/a" if not np.isfinite(r["age"]) else f"{r['age']:>11.3f}"
        kk = "  --" if r["k"] is None else f"{r['k']:>4}"
        fa = "     n/a" if not np.isfinite(r["fa"]) else f"{100*r['fa']:>12.0f}%"
        print(f"{r['measure']:<22}{r['kind']:<14}{kk}{fa}"
              f"{r['detected']:>7}/{r['units']:<3}{a}")
    print("-" * 74)
    for kind in ("distribution", "amount"):
        v = [r["age"] for r in rows
             if r["kind"] == kind and np.isfinite(r["age"])
             and r["detected"] >= 0.6 * r["units"]]
        if v:
            print(f"  {kind:<14} median alarm age {np.median(v):.3f}  "
                  f"({len(v)} indicators detecting in most units)")

print("\nAn indicator that raises the alarm earlier at the same false-alarm")
print("rate is better for the task condition monitoring actually performs.")
json.dump(out, open(os.path.join(HERE, "detection_time.json"), "w"), indent=2,
          default=float)
