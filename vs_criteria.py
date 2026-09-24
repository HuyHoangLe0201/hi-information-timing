"""
Does the earliest usable age say anything the standard criteria do not?

Health indicators are screened with monotonicity, trendability and
prognosability, in the forms introduced by Coble and Hines and used widely
since. If tau_min merely reproduces their ranking, the framework earns nothing:
practitioners already have three cheap numbers and would not want a fourth.

There is a specific reason to expect otherwise, and it is worth stating before
looking. The three criteria measure how TIDILY an indicator behaves -- whether
it rises consistently, correlates with time, and ends in the same place across
units. None of them refers to noise except through tidiness, and none says when
the indicator carries enough information to support a decision. An indicator can
rise very smoothly and still be useless until the last few percent of life,
because a smooth rise from nothing carries nothing.

So the prediction is not that the two are unrelated but that they are opposed:
root-mean-square amplitude should score well on all three and have the latest
tau_min of any bearing channel, while the high-frequency bands should score
badly and be usable earliest.

Definitions follow Coble and Hines (2009):
  monotonicity   |#(dx>0) - #(dx<0)| / (n-1), averaged over units
  trendability   the smallest |corr(x, t)| over the units
  prognosability exp( -std(x_failure) / mean|x_failure - x_start| )

They are computed on the smoothed indicator, not the raw one. Applied to raw
samples, monotonicity counts the signs of one-sample differences, and on a noisy
record those are near enough half positive and half negative whatever the trend
does: a first version of this comparison gave every bearing channel a score
between 0.010 and 0.019, which separates nothing and is not what practitioners
report. Smoothing is standard in every published application of these criteria.
The same Savitzky-Golay trend the framework uses is applied here, so neither
side gets a preprocessing advantage; the raw scores are printed alongside so
the size of that choice is visible.
"""
import os
import json
import numpy as np
from nonparam import info_curve, quantile
from scipy.signal import savgol_filter
from pipeline import _win

HERE = os.path.dirname(os.path.abspath(__file__))
SEED = 424242


def monotonicity(series):
    v = []
    for x in series:
        d = np.diff(np.asarray(x, float))
        d = d[np.isfinite(d)]
        if len(d) < 2:
            continue
        v.append(abs((d > 0).sum() - (d < 0).sum()) / len(d))
    return float(np.mean(v)) if v else np.nan


def trendability(series):
    v = []
    for x in series:
        x = np.asarray(x, float)
        t = np.arange(len(x), dtype=float)
        if len(x) < 4 or np.std(x) == 0:
            continue
        v.append(abs(float(np.corrcoef(x, t)[0, 1])))
    return float(np.min(v)) if v else np.nan


def prognosability(series):
    fail, start = [], []
    for x in series:
        x = np.asarray(x, float)
        if len(x) < 4:
            continue
        fail.append(float(np.median(x[-3:])))
        start.append(float(np.median(x[:max(3, len(x) // 20)])))
    if len(fail) < 3:
        return np.nan
    fail, start = np.array(fail), np.array(start)
    denom = float(np.mean(np.abs(fail - start)))
    if denom <= 0:
        return np.nan
    return float(np.exp(-np.std(fail, ddof=1) / denom))


def tau_of(series, rng, q=0.35):
    v = []
    for x in series:
        F = info_curve(np.asarray(x, float), "surrogate", rng)
        if F is not None:
            v.append(quantile(F, q))
    return float(np.median(v)) if v else np.nan


rng = np.random.default_rng(SEED)
SETS = {}
z = np.load(os.path.join(HERE, "feat_raw.npz"), allow_pickle=True)
FN = [str(s) for s in z["featnames"]]
I = {k: i for i, k in enumerate(FN)}
BK = sorted(k for k in z.files if k != "featnames")
for lab, parts in (("rms", ["rms"]), ("peak", ["peak"]), ("kurt", ["kurt"]),
                   ("0--2 kHz", ["b0_1", "b1_2"]), ("2--4 kHz", ["b2_4"]),
                   ("4--6 kHz", ["b4_6"]), ("6--8 kHz", ["b6_8"]),
                   ("8--10 kHz", ["b8_10"]), ("10--12.8 kHz", ["b10_12.8"])):
    SETS[lab] = [sum(z[b][:, I[p]] for p in parts) for b in BK]

def smooth(series):
    out = []
    for x in series:
        x = np.asarray(x, float)
        w = _win(len(x))
        out.append(savgol_filter(x, w, 2) if w >= 7 else x)
    return out


rows = []
for lab, series in SETS.items():
    sm = smooth(series)
    rows.append(dict(indicator=lab,
                     mon=monotonicity(sm), mon_raw=monotonicity(series),
                     tren=trendability(sm), tren_raw=trendability(series),
                     prog=prognosability(sm), prog_raw=prognosability(series),
                     tau=tau_of(series, rng)))
rows = [r for r in rows if np.isfinite(r["tau"])]
rows.sort(key=lambda r: r["tau"])

print("the three standard criteria against the earliest usable age\n")
print(f"{'indicator':<15}{'monotonic':>11}{'(raw)':>8}{'trendable':>11}"
      f"{'prognosable':>13}{'tau(0.35)':>11}")
print("-" * 69)
for r in rows:
    print(f"{r['indicator']:<15}{r['mon']:>11.3f}{r['mon_raw']:>8.3f}"
          f"{r['tren']:>11.3f}{r['prog']:>13.3f}{r['tau']:>11.3f}")
print("-" * 69)
print("(sorted by earliest usable age: the top row is usable soonest)\n")


def rank_corr(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    m = np.isfinite(a) & np.isfinite(b)
    ra = np.argsort(np.argsort(a[m])); rb = np.argsort(np.argsort(b[m]))
    return float(np.corrcoef(ra, rb)[0, 1])


tau = [r["tau"] for r in rows]
print("rank correlation with the earliest usable age")
print("(negative means a high score goes with an indicator usable EARLY,")
print(" positive means a high score goes with one usable only LATE)\n")
for key, lab in (("mon", "monotonicity"), ("tren", "trendability"),
                 ("prog", "prognosability")):
    c = rank_corr([r[key] for r in rows], tau)
    print(f"  {lab:<16}{c:+.2f}")

best_mon = max(rows, key=lambda r: r["mon"])
best_tau = min(rows, key=lambda r: r["tau"])
print(f"\nthe most monotonic indicator is {best_mon['indicator']} "
      f"(score {best_mon['mon']:.3f}), usable from tau = {best_mon['tau']:.2f}")
print(f"the earliest usable is {best_tau['indicator']} "
      f"(tau = {best_tau['tau']:.2f}), monotonicity {best_tau['mon']:.3f}")
print()
if rank_corr([r["mon"] for r in rows], tau) > 0.3:
    print("A positive correlation means the criteria favour exactly the")
    print("indicators that cannot be used until the end of life. They measure")
    print("how tidily an indicator behaves; a smooth rise from nothing is tidy")
    print("and carries nothing. The two answer different questions and the")
    print("earliest usable age is not recoverable from the three scores.")
else:
    print("The criteria and the earliest usable age rank the indicators")
    print("similarly, so the framework adds little to screening as practised.")
json.dump(rows, open(os.path.join(HERE, "vs_criteria.json"), "w"), indent=2,
          default=float)
