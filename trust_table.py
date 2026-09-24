"""
The earliest usable age for every real indicator, with a verdict on each.

Two things have been established about tau_min read off the information curve.
It is a genuine floor: over eighteen synthetic conditions the best possible
estimator -- one handed the exact degradation shape -- never beat it. And its
tightness is predicted by the signal fraction, the share of the raw density that
survives subtracting the noise floor. Where that fraction exceeds 90% the best
estimator sits a median 1.37x above the floor; below 90% the median is 2.57x and
the worst case 6.3x.

The signal fraction needs no ground truth, so it can be computed for every real
indicator and used to say which of these numbers is worth quoting. That is what
this produces: tau_min alongside the fraction, and a verdict.

An indicator whose fraction is low is not a bad indicator. It means the record is
too short or too noisy for the density to be separated from the noise floor, and
the reported tau_min is optimistic by an amount this experiment cannot pin down.
"""
import os
import json
import numpy as np
from nonparam import weighted_density, estimate_floor, quantile

HERE = os.path.dirname(os.path.abspath(__file__))
QS = (0.10, 0.35)
THRESH = 0.90
SEED = 1234


def profile(series, rng):
    Q = {q: [] for q in QS}
    SF, NS = [], []
    for raw in series:
        x = np.asarray(raw, float)
        dens, res = weighted_density(x)
        if dens is None:
            continue
        fl = estimate_floor(dens, res, "surrogate", rng)
        d = np.clip(dens - fl, 0.0, None)
        tot_raw, tot = float(dens.sum()), float(d.sum())
        if tot <= 0 or tot_raw <= 0:
            continue
        F = np.cumsum(d) / tot
        for q in QS:
            Q[q].append(quantile(F, q))
        # The calibration measured the signal fraction ACCUMULATED TO the age
        # in question, not over the whole record. Those differ, and by a lot:
        # the density is concentrated late while the floor is flat, so a
        # whole-record fraction flatters a channel whose early history is all
        # floor. The fraction is therefore taken to the age being quoted.
        k = int(round(np.median([quantile(F, 0.35)]) * len(d))) - 1
        k = max(1, min(k, len(d) - 1))
        rk, ok_ = float(dens[:k + 1].sum()), float(d[:k + 1].sum())
        SF.append(ok_ / rk if rk > 0 else np.nan)
        NS.append(len(d))
    SF = [v for v in SF if np.isfinite(v)]
    if not SF:
        return None
    return dict(units=len(SF), n=int(np.median(NS)),
                signal_frac=float(np.median(SF)),
                **{f"tau_{q:.2f}": float(np.median(Q[q])) for q in QS})


rng = np.random.default_rng(SEED)
SETS = {}

z = np.load(os.path.join(HERE, "feat_raw.npz"), allow_pickle=True)
FN = [str(s) for s in z["featnames"]]
I = {k: i for i, k in enumerate(FN)}
BK = sorted(k for k in z.files if k != "featnames")
for lab, parts in (("bearing rms", ["rms"]), ("bearing peak", ["peak"]),
                   ("bearing kurt", ["kurt"]),
                   ("bearing 0--2 kHz", ["b0_1", "b1_2"]),
                   ("bearing 2--4 kHz", ["b2_4"]),
                   ("bearing 4--10 kHz", ["b4_6", "b6_8", "b8_10"]),
                   ("bearing 10--12.8 kHz", ["b10_12.8"])):
    SETS[lab] = [sum(z[b][:, I[p]] for p in parts) for b in BK]

zt = np.load(os.path.join(HERE, "cmapss_FD001.npz"))
un = sorted({k.split("__")[0] for k in zt.files})[:60]
for c in (4, 9, 11, 15, 20):
    SETS[f"turbofan s{c}"] = [zt[f"{u}__sensors"][:, c - 1].astype(float)
                              for u in un]

zb = np.load(os.path.join(HERE, "battery_raw.npz"))
cl = sorted({k.split("__")[0] for k in zb.files})
for nm, lab in (("cap", "cell capacity"), ("Re", "cell R_e"),
                ("Rct", "cell R_ct")):
    SETS[lab] = [zb[f"{c}__{nm}"] for c in cl]

rows = []
for lab, series in SETS.items():
    p = profile(series, rng)
    if p:
        rows.append(dict(indicator=lab, **p))

rows.sort(key=lambda r: -r["signal_frac"])
print("earliest usable age, and whether the number can be trusted\n")
print(f"{'indicator':<22}{'units':>6}{'n':>6}{'signal':>9}"
      f"{'tau(0.10)':>11}{'tau(0.35)':>11}   verdict")
print("-" * 82)
for r in rows:
    ok = r["signal_frac"] >= THRESH
    verdict = ("floor is tight (~1.4x)" if ok
               else "floor is optimistic (2-6x)")
    print(f"{r['indicator']:<22}{r['units']:>6}{r['n']:>6}"
          f"{100*r['signal_frac']:>8.0f}%{r['tau_0.10']:>11.3f}"
          f"{r['tau_0.35']:>11.3f}   {verdict}")
print("-" * 82)
good = [r for r in rows if r["signal_frac"] >= THRESH]
bad = [r for r in rows if r["signal_frac"] < THRESH]
print(f"{len(good)} of {len(rows)} indicators clear the 90% signal threshold\n")
if good:
    print("quotable:")
    for r in good:
        print(f"  {r['indicator']:<22} cannot support a decision before "
              f"tau = {r['tau_0.35']:.2f}")
if bad:
    print("\nnot quotable as stated -- record too short or too noisy for the")
    print("density to be separated from the noise floor:")
    for r in bad:
        print(f"  {r['indicator']:<22} signal {100*r['signal_frac']:.0f}%, "
              f"n = {r['n']}")
json.dump(rows, open(os.path.join(HERE, "trust_table.json"), "w"), indent=2,
          default=float)
