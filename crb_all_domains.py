"""
The central test, run in every domain.

Proposition 1 says the age-error variance of the windowed ML estimator equals
Gamma_w^-1. It has so far been checked on bearings only. Here the identical
procedure -- out-of-fold residuals, local noise scale, residuals resampled
inside each window -- is applied to turbofan and battery records as well. If
the bound is a property of the estimation problem rather than of one dataset,
the log-log slope should be 1 in all three.
"""
import os, json, sys
import numpy as np
import scipy.io as sio
from pipeline import prepare

HERE = os.path.dirname(os.path.abspath(__file__))
TAU0 = np.round(np.arange(0.30, 0.96, 0.05), 3)
WGRID = np.array([0.05, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50, 0.60])
N_MC = 300
RNG = np.random.default_rng(20260809)


def cells_for(units):
    lp, lr = [], []
    for u in units:
        n, dt, sl, res = u["n"], u["dtrend"], u["sig_loc"], u["res"]
        for t0 in TAU0:
            k0 = int(round(t0 * n)) - 1
            if k0 < 5 or k0 >= n:
                continue
            for w in WGRID:
                if w > t0:
                    continue
                klo = max(0, int(round((t0 - w) * n)) - 1)
                if k0 - klo < 10:
                    continue
                g, s = dt[klo:k0 + 1], sl[klo:k0 + 1]
                fisher = float(np.sum((g / s) ** 2))
                if fisher <= 0:
                    continue
                wt = g / s ** 2
                den = float(np.sum(wt * g))
                if den <= 0:
                    continue
                draws = RNG.choice(res[klo:k0 + 1], size=(N_MC, len(g)), replace=True)
                lp.append(np.log(1.0 / np.sqrt(fisher)))
                lr.append(np.log(float(np.std(draws @ wt / den, ddof=1))))
    if len(lp) < 30:
        return None
    lp, lr = np.asarray(lp), np.asarray(lr)
    r = np.exp(lr - lp)
    return dict(cells=len(lp), slope=float(np.polyfit(lp, lr, 1)[0]),
                corr=float(np.corrcoef(lp, lr)[0, 1]),
                median=float(np.median(r)),
                q25=float(np.percentile(r, 25)), q75=float(np.percentile(r, 75)))


def load_domain(tag):
    us = []
    if tag == "battery":
        z = np.load(os.path.join(HERE, "battery_raw.npz"))
        for cell in sorted({k.split("__")[0] for k in z.files}):
            for ind in ["cap", "Re", "Rct"]:
                u, _ = prepare(z[f"{cell}__{ind}"], f"{cell}/{ind}")
                if u: us.append(u)
    elif tag.startswith("turbofan"):
        sub = tag.split("_")[1]
        f = f"cmapss_{sub}_norm.npz" if sub == "FD004" else f"cmapss_{sub}.npz"
        z = np.load(os.path.join(HERE, f))
        names = sorted({k.split("__")[0] for k in z.files})[:60]   # cap the cost
        for nm in names:
            S = z[f"{nm}__sensors"]
            for c in (4, 11, 14):        # three standard prognostic channels
                u, _ = prepare(S[:, c - 1].astype(float), f"{nm}/s{c}")
                if u: us.append(u)
    return us


print(f"{'domain':<20}{'series':>7}{'cells':>7}{'slope':>8}{'r':>7}"
      f"{'median':>8}{'IQR':>16}")
print("-" * 74)
out = {}
for tag in ["battery", "turbofan_FD001", "turbofan_FD004"]:
    us = load_domain(tag)
    res = cells_for(us)
    if res is None:
        print(f"{tag:<20}  too few usable cells")
        continue
    out[tag] = dict(series=len(us), **res)
    print(f"{tag:<20}{len(us):>7}{res['cells']:>7}{res['slope']:>8.3f}"
          f"{res['corr']:>7.3f}{res['median']:>8.3f}"
          f"{'[%.2f, %.2f]' % (res['q25'], res['q75']):>16}")

# the bearing result already on disk, for the same table
prev = json.load(open(os.path.join(HERE, "oof_results.json")))["configs"]["A2"]
out["bearing"] = dict(series=15, cells=prev["cells"], slope=prev["slope"],
                      corr=prev["corr"], median=prev["median"],
                      q25=prev["q25"], q75=prev["q75"])
print(f"{'bearing (earlier)':<20}{15:>7}{prev['cells']:>7}{prev['slope']:>8.3f}"
      f"{prev['corr']:>7.3f}{prev['median']:>8.3f}"
      f"{'[%.2f, %.2f]' % (prev['q25'], prev['q75']):>16}")
print("-" * 74)
print("Proposition 1 predicts slope 1 and median 1 in every row.")
json.dump(out, open(os.path.join(HERE, "crb_all_domains.json"), "w"), indent=2)
