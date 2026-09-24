"""
What actually limits the achievable precision?

The information Gamma = sum D'^2/sigma^2 treats sigma as observation noise, and
on real bearing records it comes out around 10^7 -- a standard error on the
damage clock of 0.03% of life. That is not credible, and the nuisance-parameter
correction does not rescue it: allowing the amplitude and exponent to be unknown
costs a factor of two to four, leaving 10^5 or so.

The remaining gap has to be model error. sigma is estimated as the scatter about
a LOCAL smoother, which follows whatever the record does, so it measures
short-range noise only. The estimator that would attain the bound must instead
assume a trend from a fixed family, and the difference between a real record and
the best member of that family is a systematic error that no amount of averaging
removes. If that error is much larger than the local noise, it, and not sigma,
sets the achievable precision.

This measures both residuals on the same records:

    local   -- record minus a local (Savitzky-Golay) fit: short-range noise;
    family  -- record minus the best-fitting A tau^beta: noise plus whatever
               the family cannot represent.

Their ratio is how far the information estimate is out, and the autocorrelation
of the second says whether the excess is systematic or just a worse fit.
"""
import os
import json
import numpy as np
from scipy.optimize import curve_fit
from scipy.signal import savgol_filter
from pipeline import robust_scale, oof_trend, _win

HERE = os.path.dirname(os.path.abspath(__file__))


def acorr_len(r):
    r = r - r.mean()
    if np.allclose(r, 0):
        return np.nan
    a = np.correlate(r, r, "full")[len(r) - 1:]
    a = a / a[0]
    below = np.where(a < np.exp(-1))[0]
    return int(below[0]) if len(below) else len(r)


def analyse_one(x):
    n = len(x)
    s = robust_scale(x)
    if not np.isfinite(s) or s <= 0 or n < 100:
        return None
    D = (x - np.median(x)) / s
    w = _win(n)
    if w < 7:
        return None
    local_res = D - oof_trend(D, w // 2)
    s_local = robust_scale(local_res)
    tau = np.arange(1, n + 1) / n
    try:
        p, _ = curve_fit(lambda t, A, b, c: A * t ** b + c, tau, D,
                         p0=[1.0, 2.0, float(D[0])],
                         bounds=([1e-6, 0.51, -1e3], [1e4, 30.0, 1e3]),
                         maxfev=40000)
    except Exception:
        return None
    fam_res = D - (p[0] * tau ** p[1] + p[2])
    s_fam = robust_scale(fam_res)
    if not np.isfinite(s_local) or s_local <= 0 or not np.isfinite(s_fam):
        return None
    return dict(s_local=float(s_local), s_family=float(s_fam),
                ratio=float(s_fam / s_local),
                acorr_local=acorr_len(local_res),
                acorr_family=acorr_len(fam_res),
                beta=float(p[1]))


def run(name, series):
    R = [analyse_one(np.asarray(x, float)) for x in series]
    R = [r for r in R if r is not None]
    if len(R) < 3:
        return None
    med = lambda k: float(np.median([r[k] for r in R]))
    out = dict(indicator=name, units=len(R), s_local=med("s_local"),
               s_family=med("s_family"), ratio=med("ratio"),
               acorr_local=med("acorr_local"),
               acorr_family=med("acorr_family"))
    print(f"{name:<22}{len(R):>6}{out['s_local']:>10.4f}{out['s_family']:>11.4f}"
          f"{out['ratio']:>9.1f}{out['acorr_local']:>9.0f}"
          f"{out['acorr_family']:>10.0f}{out['ratio']**2:>12.0f}")
    return out


print("local noise against family misfit\n")
print(f"{'indicator':<22}{'units':>6}{'local sd':>10}{'family sd':>11}"
      f"{'ratio':>9}{'ac local':>9}{'ac family':>10}{'info factor':>12}")
print("-" * 89)
rows = []

z = np.load(os.path.join(HERE, "feat_raw.npz"), allow_pickle=True)
FN = [str(s) for s in z["featnames"]]
I = {k: i for i, k in enumerate(FN)}
BK = sorted(k for k in z.files if k != "featnames")
for lab, parts in (("bearing RMS", ["rms"]),
                   ("bearing 0--2 kHz", ["b0_1", "b1_2"]),
                   ("bearing 4--10 kHz", ["b4_6", "b6_8", "b8_10"])):
    r = run(lab, [sum(z[b][:, I[p]] for p in parts) for b in BK])
    if r:
        rows.append(r)

zt = np.load(os.path.join(HERE, "cmapss_FD001.npz"))
un = sorted({k.split("__")[0] for k in zt.files})[:60]
for c, lab in ((4, "turbofan s4"), (11, "turbofan s11")):
    r = run(lab, [zt[f"{u}__sensors"][:, c - 1].astype(float) for u in un])
    if r:
        rows.append(r)

zb = np.load(os.path.join(HERE, "battery_raw.npz"))
cl = sorted({k.split("__")[0] for k in zb.files})
r = run("cell capacity", [zb[f"{c}__cap"] for c in cl])
if r:
    rows.append(r)

print("-" * 89)
print("ratio       = family misfit / local noise")
print("info factor = ratio^2: how far the information built on the local")
print("              scale overstates what a fixed trend family can deliver")
print("ac          = autocorrelation length in samples; 1 is white")
print()
rs = [r["ratio"] for r in rows]
print(f"misfit exceeds local noise by {min(rs):.1f}x to {max(rs):.1f}x, so the")
print(f"information is overstated by {min(rs)**2:.0f}x to {max(rs)**2:.0f}x.")
al = [r["acorr_family"] for r in rows]
print(f"the misfit residual has autocorrelation length {min(al):.0f} to "
      f"{max(al):.0f} samples against {max(r['acorr_local'] for r in rows):.0f} "
      f"for the local one:")
print("it is structure the family cannot represent, not extra noise.")
json.dump(rows, open(os.path.join(HERE, "model_error.json"), "w"), indent=2)
