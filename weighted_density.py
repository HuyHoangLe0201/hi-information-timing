"""
Which density is the analysis actually entitled to use?

The Fisher information for Gaussian noise whose scale varies along the record is

    Gamma_W = sum_{tau_k in W}  D'(tau_k)^2 / sigma(tau_k)^2,

so the information density is D'^2 / sigma^2. The validation of the bound uses
exactly that. The exponent and the earliest usable age, however, were computed
from the unweighted D'^2, which is the same object only when sigma is constant
-- and the measured local scale spans up to a factor of five along a bearing
record.

The two therefore need not agree, and every exponent and every tau_min reported
from the unweighted density is suspect. This quantifies the difference, and then
asks which of the two satisfies the closed-form relation

    tau_min = (q*)^{1 / (2 beta - 1)}

that ties the exponent to the earliest usable age. That relation is a property
of a power-law density, so whichever density is closer to power-law will satisfy
it better; it is a consistency test, not proof of correctness, but a large gap
between the two would say the choice matters.
"""
import os
import json
import numpy as np
from scipy.optimize import minimize_scalar
from pipeline import prepare

HERE = os.path.dirname(os.path.abspath(__file__))
QSTAR = 0.35


def profile(dens, n):
    d = np.asarray(dens, float)
    if not np.all(np.isfinite(d)) or d.sum() <= 0:
        return np.nan, np.nan
    F = np.cumsum(d) / d.sum()
    tau = np.arange(1, n + 1) / n
    f = lambda b: float(np.mean((F - tau ** (2 * b - 1)) ** 2))
    beta = float(minimize_scalar(f, bounds=(0.51, 30.0), method="bounded").x)
    k = int(np.searchsorted(F, QSTAR))
    tmin = (k + 1) / n if k < n else 1.0
    return beta, tmin


def closure_err(beta, tmin):
    if not (np.isfinite(beta) and np.isfinite(tmin)) or beta <= 0.51:
        return np.nan
    return abs(QSTAR ** (1.0 / (2 * beta - 1)) - tmin)


z = np.load(os.path.join(HERE, "feat_raw.npz"), allow_pickle=True)
FN = [str(s) for s in z["featnames"]]
I = {k: i for i, k in enumerate(FN)}
BK = sorted(k for k in z.files if k != "featnames")

CH = {"rms": ["rms"], "peak": ["peak"], "kurt": ["kurt"],
      "0--2 kHz": ["b0_1", "b1_2"], "2--4 kHz": ["b2_4"],
      "4--10 kHz": ["b4_6", "b6_8", "b8_10"], "10--12.8 kHz": ["b10_12.8"]}

print(f"{'channel':<14}{'units':>6}{'scale span':>12}"
      f"{'beta unwt':>11}{'beta wtd':>10}{'tmin unwt':>11}{'tmin wtd':>10}")
print("-" * 74)
rows = []
for lab, parts in CH.items():
    BU, BW, TU, TW, SP, EU, EW = [], [], [], [], [], [], []
    for b in BK:
        x = sum(z[b][:, I[p]] for p in parts)
        u, _ = prepare(np.asarray(x, float), f"{b}/{lab}")
        if u is None:
            continue
        sl = np.clip(u["sig_loc"], 1e-12, None)
        bu, tu = profile(u["dtrend"] ** 2, u["n"])
        bw, tw = profile((u["dtrend"] / sl) ** 2, u["n"])
        if not np.isfinite(bu) or not np.isfinite(bw):
            continue
        BU.append(bu); BW.append(bw); TU.append(tu); TW.append(tw)
        SP.append(float(np.percentile(sl, 95) / np.percentile(sl, 5)))
        EU.append(closure_err(bu, tu)); EW.append(closure_err(bw, tw))
    if len(BU) < 4:
        continue
    r = dict(channel=lab, units=len(BU), span=float(np.median(SP)),
             beta_unweighted=float(np.median(BU)),
             beta_weighted=float(np.median(BW)),
             tmin_unweighted=float(np.median(TU)),
             tmin_weighted=float(np.median(TW)),
             closure_unweighted=float(np.nanmedian(EU)),
             closure_weighted=float(np.nanmedian(EW)))
    rows.append(r)
    print(f"{lab:<14}{r['units']:>6}{r['span']:>12.1f}"
          f"{r['beta_unweighted']:>11.2f}{r['beta_weighted']:>10.2f}"
          f"{r['tmin_unweighted']:>11.3f}{r['tmin_weighted']:>10.3f}")

print("-" * 74)
eu = float(np.median([r["closure_unweighted"] for r in rows]))
ew = float(np.median([r["closure_weighted"] for r in rows]))
print(f"closure error |q*^(1/(2b-1)) - tau_min|, median over channels:")
print(f"  unweighted density : {eu:.4f}")
print(f"  weighted density   : {ew:.4f}")
print()
dt = float(np.median([abs(r["tmin_weighted"] - r["tmin_unweighted"])
                      for r in rows]))
db = float(np.median([abs(r["beta_weighted"] - r["beta_unweighted"])
                      for r in rows]))
print(f"median shift in tau_min when the correct density is used: {dt:.3f}")
print(f"median shift in beta:                                     {db:.2f}")
print()
ordu = [r["channel"] for r in sorted(rows, key=lambda r: r["tmin_unweighted"])]
ordw = [r["channel"] for r in sorted(rows, key=lambda r: r["tmin_weighted"])]
print("channels ranked earliest-usable-first")
print(f"  unweighted : {', '.join(ordu)}")
print(f"  weighted   : {', '.join(ordw)}")
print("\nA changed ranking would mean the choice of density changes which")
print("indicator a designer should pick, not merely the numbers attached to it.")
json.dump(rows, open(os.path.join(HERE, "weighted_density.json"), "w"), indent=2)
