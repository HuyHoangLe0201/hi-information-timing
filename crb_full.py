"""
The central test, on every record rather than a subset.

crb_all_domains.py capped the turbofan sets at the first 60 units to keep the
cost down, and took the bearing row from an earlier run. Both are fixed here:
one script, one protocol, every unit in every domain, and the per-cell values
kept so the figure and the table cannot drift apart.

Proposition 1 says the age-error standard deviation of the windowed ML
estimator equals Gamma_W^{-1/2}. Each cell is one (tau_0, w) pair on one
unit-channel series: the predicted value comes from the estimated density and
local scale, the realised value from resampling that series' own out-of-fold
residuals inside the window. A log-log slope of 1 and a ratio of 1 are the
predictions.
"""
import os
import json
import numpy as np
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
                draws = RNG.choice(res[klo:k0 + 1], size=(N_MC, len(g)),
                                   replace=True)
                lp.append(np.log(1.0 / np.sqrt(fisher)))
                lr.append(np.log(float(np.std(draws @ wt / den, ddof=1))))
    if len(lp) < 30:
        return None, None, None
    lp, lr = np.asarray(lp), np.asarray(lr)
    r = np.exp(lr - lp)
    return (dict(cells=len(lp), slope=float(np.polyfit(lp, lr, 1)[0]),
                 corr=float(np.corrcoef(lp, lr)[0, 1]),
                 median=float(np.median(r)),
                 q25=float(np.percentile(r, 25)),
                 q75=float(np.percentile(r, 75))), lp, lr)


def load_domain(tag):
    """Every unit, three channels each, so no domain is over-weighted."""
    us = []
    if tag == "bearing":
        z = np.load(os.path.join(HERE, "feat_raw.npz"), allow_pickle=True)
        FN = [str(s) for s in z["featnames"]]
        I = {k: i for i, k in enumerate(FN)}
        for b in sorted(k for k in z.files if k != "featnames"):
            M = z[b]
            chans = {"rms": M[:, I["rms"]],
                     "lf": M[:, I["b0_1"]] + M[:, I["b1_2"]],
                     "hf": M[:, I["b4_6"]] + M[:, I["b6_8"]] + M[:, I["b8_10"]]}
            for nm, x in chans.items():
                u, _ = prepare(np.asarray(x, float), f"{b}/{nm}")
                if u:
                    us.append(u)
    elif tag == "battery":
        z = np.load(os.path.join(HERE, "battery_raw.npz"))
        for cell in sorted({k.split("__")[0] for k in z.files}):
            for ind in ("cap", "Re", "Rct"):
                u, _ = prepare(z[f"{cell}__{ind}"], f"{cell}/{ind}")
                if u:
                    us.append(u)
    elif tag.startswith("turbofan"):
        sub = tag.split("_")[1]
        f = f"cmapss_{sub}_norm.npz" if sub == "FD004" else f"cmapss_{sub}.npz"
        z = np.load(os.path.join(HERE, f))
        for nm in sorted({k.split("__")[0] for k in z.files}):
            S = z[f"{nm}__sensors"]
            for c in (4, 11, 14):        # three standard prognostic channels
                u, _ = prepare(S[:, c - 1].astype(float), f"{nm}/s{c}")
                if u:
                    us.append(u)
    return us


print(f"{'domain':<18}{'series':>7}{'cells':>8}{'slope':>8}{'r':>7}"
      f"{'median':>8}{'IQR':>16}")
print("-" * 72)
out, arrays = {}, {}
for tag in ("bearing", "turbofan_FD001", "turbofan_FD004", "battery"):
    us = load_domain(tag)
    res, lp, lr = cells_for(us)
    if res is None:
        print(f"{tag:<18}  too few usable cells")
        continue
    out[tag] = dict(series=len(us), **res)
    arrays[f"{tag}__lp"], arrays[f"{tag}__lr"] = lp, lr
    print(f"{tag:<18}{len(us):>7}{res['cells']:>8}{res['slope']:>8.3f}"
          f"{res['corr']:>7.3f}{res['median']:>8.3f}"
          f"{'[%.2f, %.2f]' % (res['q25'], res['q75']):>16}")

print("-" * 72)
print("Proposition 1 predicts slope 1 and median ratio 1 in every row.")
tot = sum(v["cells"] for v in out.values())
print(f"total evaluation cells: {tot}")

json.dump(out, open(os.path.join(HERE, "crb_full.json"), "w"), indent=2)
np.savez_compressed(os.path.join(HERE, "crb_cells.npz"), **arrays)
print("wrote crb_full.json and crb_cells.npz")
