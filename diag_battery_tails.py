"""
Is the battery failure caused by heavy tails, and does the obvious fix work?

The residual is white but wildly non-Gaussian (excess kurtosis ~255 on capacity),
because lithium cells recover capacity after rest and the record shows isolated
jumps. A robust scale describes the bulk of such a residual, not its variance,
so Gamma_w built on it over-states the information a window holds.

Three variants are compared on identical windows:
  robust   sigma from the MAD  (what the bearing analysis used)
  std      sigma from the actual standard deviation
  despiked isolated jumps removed first, then the robust scale
If the diagnosis is right, the last two should restore slope ~1.
"""
import os, json
import numpy as np
from pipeline import prepare, robust_scale, local_scale, LOCAL_BW

HERE = os.path.dirname(os.path.abspath(__file__))
TAU0 = np.round(np.arange(0.30, 0.96, 0.05), 3)
WGRID = np.array([0.10, 0.15, 0.20, 0.30, 0.40, 0.50, 0.60])
N_MC = 400


def despike(x, k=4.0):
    """Replace points whose first difference is an isolated outlier."""
    x = np.asarray(x, float).copy()
    d = np.diff(x, prepend=x[0])
    s = robust_scale(d)
    bad = np.abs(d - np.median(d)) > k * s
    idx = np.arange(len(x))
    if bad.any() and (~bad).sum() > 3:
        x[bad] = np.interp(idx[bad], idx[~bad], x[~bad])
    return x


def run(units, scale_mode):
    rng = np.random.default_rng(20260809)
    lp, lr = [], []
    for u in units:
        n, dt, res = u["n"], u["dtrend"], u["res"]
        half = max(5, int(LOCAL_BW * n))
        if scale_mode == "std":
            sl = np.empty(n)
            for i in range(n):
                lo, hi = max(0, i - half), min(n, i + half + 1)
                sl[i] = max(np.std(res[lo:hi]), 1e-9)
        else:
            sl = u["sig_loc"]
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
                dr = rng.choice(res[klo:k0 + 1], size=(N_MC, len(g)), replace=True)
                lp.append(np.log(1.0 / np.sqrt(fisher)))
                lr.append(np.log(float(np.std(dr @ wt / den, ddof=1))))
    if len(lp) < 30:
        return None
    lp, lr = np.asarray(lp), np.asarray(lr)
    return dict(cells=len(lp), slope=float(np.polyfit(lp, lr, 1)[0]),
                corr=float(np.corrcoef(lp, lr)[0, 1]),
                median=float(np.median(np.exp(lr - lp))))


zb = np.load(os.path.join(HERE, "battery_raw.npz"))
cells = sorted({k.split("__")[0] for k in zb.files})

print(f"{'variant':<26}{'series':>7}{'cells':>7}{'slope':>8}{'r':>7}{'median':>8}{'kurt':>8}")
print("-" * 71)
out = {}
for tag, mode, pre in [("robust scale (as before)", "robust", False),
                       ("standard deviation", "std", False),
                       ("despiked + robust", "robust", True)]:
    us = []
    kurts = []
    for c in cells:
        for ind in ["cap", "Re", "Rct"]:
            raw = zb[f"{c}__{ind}"]
            if pre:
                raw = despike(raw)
            u, _ = prepare(raw, f"{c}/{ind}")
            if u:
                us.append(u)
                r = u["res"] / u["sig_oof"]
                kurts.append(float(np.mean(r ** 4) - 3.0))
    res = run(us, mode)
    if res is None:
        print(f"{tag:<26} too few cells"); continue
    out[tag] = dict(series=len(us), kurt=float(np.median(kurts)), **res)
    print(f"{tag:<26}{len(us):>7}{res['cells']:>7}{res['slope']:>8.3f}"
          f"{res['corr']:>7.3f}{res['median']:>8.3f}{np.median(kurts):>8.1f}")
print("-" * 71)
print("Proposition 1 predicts slope 1, median 1.")
json.dump(out, open(os.path.join(HERE, "battery_tails.json"), "w"), indent=2)
