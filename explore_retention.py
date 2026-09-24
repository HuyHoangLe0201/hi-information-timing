"""
Does the trailing-vs-leading split survive on the real bearings?

For each indicator band we already have the empirical information density
Dhat'(tau)^2 per unit. The retention question is then purely a statement about
that density: for a budget of w, is more information carried by the most recent
w of life (a FIFO buffer) or by the earliest w (a fixed archive)?

Reported per band at tau0 = 0.7: median over units of the trailing mass, the
leading mass, and their ratio -- alongside beta_eff, which predicts the sign
(beta_eff > 1 => density increasing => trailing; < 1 => leading).
"""
import os
import json
import numpy as np
from scipy.signal import savgol_filter
from scipy.optimize import minimize_scalar

HERE = os.path.dirname(os.path.abspath(__file__))
fz = np.load(os.path.join(HERE, "feat_raw.npz"), allow_pickle=True)
FN = [str(s) for s in fz["featnames"]]
I = {n: k for k, n in enumerate(FN)}
BEARINGS = sorted(k for k in fz.files if k != "featnames")
SMOOTH_FRAC, SIGMA_MAX, TAU0 = 0.08, 0.5, 0.7

BANDS = {
    "rms": lambda M: M[:, I["rms"]],
    "kurtosis": lambda M: M[:, I["kurt"]],
    "lf 0-2kHz": lambda M: M[:, I["b0_1"]] + M[:, I["b1_2"]],
    "mf 2-6kHz": lambda M: M[:, I["b2_4"]] + M[:, I["b4_6"]],
    "hf 4-10kHz": lambda M: M[:, I["b4_6"]] + M[:, I["b6_8"]] + M[:, I["b8_10"]],
}


def density(x):
    n = len(x)
    base = np.median(x[:max(3, int(0.05 * n))]); end = np.median(x[-3:])
    if not np.isfinite(base) or not np.isfinite(end) or abs(end - base) < 1e-12:
        return None
    D = (x - base) / (end - base)
    win = max(7, int(SMOOTH_FRAC * n)); win += (win % 2 == 0)
    win = min(win, n - 1 if (n - 1) % 2 == 1 else n - 2)
    if win < 7:
        return None
    dt = savgol_filter(D, win, 2, deriv=1, delta=1.0 / n)
    res = D - savgol_filter(D, win, 2)
    sig = float(1.4826 * np.median(np.abs(res - np.median(res))))
    if not np.isfinite(sig) or sig > SIGMA_MAX:
        return None
    d = dt ** 2
    return (d / d.sum(), n) if d.sum() > 0 else None


def beta_eff(cum, n):
    tau = np.arange(1, n + 1) / n
    f = lambda b: float(np.mean((cum - tau ** (2 * b - 1)) ** 2))
    return float(minimize_scalar(f, bounds=(0.51, 30.0), method="bounded",
                                 options={"xatol": 1e-3}).x)


print(f"{'band':<14}{'units':>6}{'beta_eff':>10}{'w':>6}"
      f"{'trailing':>11}{'leading':>10}{'lead/trail':>13}{'better':>10}")
print("-" * 80)
out = {}
for name, fn in BANDS.items():
    per_w = {}
    betas = []
    for b in BEARINGS:
        r = density(fn(fz[b]))
        if r is None:
            continue
        d, n = r
        cum = np.cumsum(d)
        betas.append(beta_eff(cum, n))
        k0 = int(round(TAU0 * n)) - 1
        for w in (0.2, 0.3, 0.5):
            kw = int(round(w * n))
            klo = max(0, k0 - kw)
            tr = float(cum[k0] - cum[klo])
            ld = float(cum[min(kw, n - 1)])
            per_w.setdefault(w, []).append((tr, ld))
    if not betas:
        continue
    be = float(np.median(betas))
    out[name] = dict(beta_eff=be, units=len(betas), w={})
    for w, vals in per_w.items():
        tr = np.median([v[0] for v in vals])
        ld = np.median([v[1] for v in vals])
        ratio = ld / tr if tr > 0 else np.inf
        out[name]["w"][str(w)] = dict(trailing=float(tr), leading=float(ld),
                                      ratio=float(ratio))
        better = "archive" if ratio > 1 else "FIFO"
        print(f"{name:<14}{len(betas):>6}{be:>10.2f}{w:>6.2f}"
              f"{tr:>11.4f}{ld:>10.4f}{ratio:>13.2f}{better:>10}")
    print()

print("prediction: beta_eff > 1 => information density increasing => FIFO wins;")
print("            beta_eff < 1 => decreasing => early archive wins.")
agree = sum(1 for n, v in out.items()
            for w, d in v["w"].items()
            if (v["beta_eff"] > 1) == (d["ratio"] < 1))
total = sum(len(v["w"]) for v in out.values())
print(f"sign agreement with beta_eff: {agree}/{total} band-budget combinations")

with open(os.path.join(HERE, "retention.json"), "w") as fh:
    json.dump(dict(tau0=TAU0, bands=out), fh, indent=2)
