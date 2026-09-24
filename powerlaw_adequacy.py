"""
Which indicators is the power-law summary entitled to describe?

The information curve G(tau) is non-parametric: it is a cumulative sum of
D'^2/sigma^2 and needs no functional form. The power law enters only as a
summary, through beta_eff, and everything that leans on beta_eff -- the closed
forms, tau_min = q*^{1/(2 beta - 1)}, the nuisance-parameter analysis -- inherits
whatever inadequacy that summary has.

A more flexible trend family cut the model-error discount by up to a factor of
1173 and collapsed its spread across bearing channels from 902x to 7.4x, so the
power law is demonstrably wrong for some of them. This measures the inadequacy
directly, on the object beta_eff actually summarises: the fitted power-law
cumulative against the measured one.

Reported per channel are the fit residual, and the tau_min taken two ways --
read straight off the measured curve, which needs no model, and computed from
the fitted exponent, which does. Their difference is the error the summary
introduces for that indicator.
"""
import os
import json
import numpy as np
from scipy.optimize import minimize_scalar
from pipeline import prepare

HERE = os.path.dirname(os.path.abspath(__file__))
QSTAR = 0.35


def assess(u, weighted):
    n = u["n"]
    d = (u["dtrend"] / np.clip(u["sig_loc"], 1e-12, None)) ** 2 if weighted \
        else u["dtrend"] ** 2
    if not np.all(np.isfinite(d)) or d.sum() <= 0:
        return None
    F = np.cumsum(d) / d.sum()
    tau = np.arange(1, n + 1) / n
    f = lambda b: float(np.mean((F - tau ** (2 * b - 1)) ** 2))
    r = minimize_scalar(f, bounds=(0.51, 30.0), method="bounded")
    beta = float(r.x)
    rms = float(np.sqrt(r.fun))                 # rms gap in cumulative units
    k = int(np.searchsorted(F, QSTAR))
    t_meas = (k + 1) / n if k < n else 1.0
    t_fit = QSTAR ** (1.0 / (2 * beta - 1))
    return dict(beta=beta, misfit=rms, tau_measured=t_meas, tau_fitted=t_fit,
                gap=abs(t_fit - t_meas))


z = np.load(os.path.join(HERE, "feat_raw.npz"), allow_pickle=True)
FN = [str(s) for s in z["featnames"]]
I = {k: i for i, k in enumerate(FN)}
BK = sorted(k for k in z.files if k != "featnames")
CH = {"rms": ["rms"], "peak": ["peak"], "kurt": ["kurt"],
      "0--2 kHz": ["b0_1", "b1_2"], "2--4 kHz": ["b2_4"],
      "4--10 kHz": ["b4_6", "b6_8", "b8_10"], "10--12.8 kHz": ["b10_12.8"]}

print("how well a power law describes the measured information curve\n")
print(f"{'channel':<15}{'units':>6}{'beta':>8}{'misfit':>9}"
      f"{'tau measured':>14}{'tau from beta':>15}{'gap':>8}")
print("-" * 75)
rows = []
for lab, parts in CH.items():
    A = []
    for b in BK:
        x = np.asarray(sum(z[b][:, I[p]] for p in parts), float)
        u, _ = prepare(x, f"{b}/{lab}")
        if u is None:
            continue
        a = assess(u, weighted=True)
        if a:
            A.append(a)
    if len(A) < 4:
        continue
    med = lambda k: float(np.median([a[k] for a in A]))
    r = dict(channel=lab, units=len(A), beta=med("beta"), misfit=med("misfit"),
             tau_measured=med("tau_measured"), tau_fitted=med("tau_fitted"),
             gap=med("gap"))
    rows.append(r)
    print(f"{lab:<15}{r['units']:>6}{r['beta']:>8.2f}{r['misfit']:>9.4f}"
          f"{r['tau_measured']:>14.3f}{r['tau_fitted']:>15.3f}{r['gap']:>8.3f}")
print("-" * 75)
print("misfit = rms gap between the fitted and measured cumulative curves;")
print("gap    = error the power-law summary introduces in the earliest usable")
print("         age, in fractions of life.\n")

good = [r for r in rows if r["gap"] <= 0.05]
bad = [r for r in rows if r["gap"] > 0.05]
print(f"the summary is within 0.05 of life for {len(good)} of {len(rows)} "
      f"channels:")
print(f"  {', '.join(r['channel'] for r in good)}")
if bad:
    print(f"and wrong by more than that for {len(bad)}:")
    for r in bad:
        print(f"  {r['channel']:<15} off by {r['gap']:.3f} of life "
              f"(measured {r['tau_measured']:.3f}, "
              f"summary {r['tau_fitted']:.3f})")
print()
print("tau_min read off the measured curve needs no power law and is available")
print("for every channel; the exponent is a convenience that costs accuracy")
print("exactly where the indicator does not happen to follow a power law.")
json.dump(rows, open(os.path.join(HERE, "powerlaw_adequacy.json"), "w"),
          indent=2)
