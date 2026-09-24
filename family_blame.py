"""
Is the discount the indicator's fault, or the trend family's?

Fitting A tau^beta + c leaves a residual whose scale and correlation length
imply a discount ranging from 1.5x to 1382x across bearing channels alone. Two
readings, with opposite consequences:

  the indicator's fault -- some indicators genuinely do not follow a smooth
      low-dimensional path, and any parametric estimator will pay for it;
  the family's fault    -- a power law is simply the wrong shape for those
      channels, and a better family would remove the penalty, in which case the
      discount measures my modelling choice rather than the data.

They are distinguishable by fitting families of increasing flexibility. If the
discount collapses as parameters are added, the family was to blame. If it
persists, the structure is in the indicator and no smooth family will capture it.

The parameter count is reported alongside, because a family flexible enough to
follow the noise would drive the residual down for no good reason: what matters
is whether the discount falls FASTER than the parameter count rises.
"""
import os
import json
import numpy as np
from scipy.optimize import curve_fit
from pipeline import robust_scale, oof_trend, _win

HERE = os.path.dirname(os.path.abspath(__file__))


def acorr_len(r):
    r = r - r.mean()
    if np.allclose(r, 0):
        return np.nan
    a = np.correlate(r, r, "full")[len(r) - 1:]
    a = a / a[0]
    b = np.where(a < np.exp(-1))[0]
    return int(b[0]) if len(b) else len(r)


def power_fit(tau, D):
    p, _ = curve_fit(lambda t, A, b, c: A * t ** b + c, tau, D,
                     p0=[1.0, 2.0, float(D[0])],
                     bounds=([1e-6, 0.51, -1e3], [1e4, 30.0, 1e3]),
                     maxfev=40000)
    return p[0] * tau ** p[1] + p[2]


def poly_fit(tau, D, deg):
    return np.polyval(np.polyfit(tau, D, deg), tau)


FAMILIES = [("power law", 3, None),
            ("poly deg 2", 3, 2), ("poly deg 3", 4, 3),
            ("poly deg 5", 6, 5), ("poly deg 8", 9, 8),
            ("poly deg 12", 13, 12)]


def discounts(x):
    n = len(x)
    s = robust_scale(x)
    if not np.isfinite(s) or s <= 0 or n < 150:
        return None
    D = (x - np.median(x)) / s
    w = _win(n)
    if w < 7:
        return None
    s_loc = robust_scale(D - oof_trend(D, w // 2))
    if not np.isfinite(s_loc) or s_loc <= 0:
        return None
    tau = np.arange(1, n + 1) / n
    out = {}
    for lab, npar, deg in FAMILIES:
        try:
            fit = power_fit(tau, D) if deg is None else poly_fit(tau, D, deg)
        except Exception:
            continue
        r = D - fit
        sf = robust_scale(r)
        ell = acorr_len(r)
        if not np.isfinite(sf) or not np.isfinite(ell):
            continue
        out[lab] = (sf / s_loc) ** 2 * max(ell, 1.0)
    return out


z = np.load(os.path.join(HERE, "feat_raw.npz"), allow_pickle=True)
FN = [str(s) for s in z["featnames"]]
I = {k: i for i, k in enumerate(FN)}
BK = sorted(k for k in z.files if k != "featnames")

# the channels that the power law fitted worst and best
WORST = ["b4_6", "b2_4", "b10_12.8", "rms"]
BEST = ["kurt", "peak", "b1_2"]

print("discount by trend family (bearing channels)\n")
hdr = "".join(f"{lab.split()[-1]:>9}" for lab, _, _ in FAMILIES)
print(f"{'channel':<12}{'params':>7}" + hdr)
print(f"{'':<12}{'':>7}" + "".join(f"{n:>9}" for _, n, _ in FAMILIES))
print("-" * (19 + 9 * len(FAMILIES)))
rows = {}
for nm in WORST + BEST:
    per = []
    for b in BK:
        d = discounts(np.asarray(z[b][:, I[nm]], float))
        if d and len(d) == len(FAMILIES):
            per.append(d)
    if len(per) < 4:
        continue
    med = {lab: float(np.median([p[lab] for p in per]))
           for lab, _, _ in FAMILIES}
    rows[nm] = med
    tag = "  (worst)" if nm in WORST else "  (best)"
    print(f"{nm:<12}{'':>7}" + "".join(f"{med[lab]:>9.1f}"
                                       for lab, _, _ in FAMILIES) + tag)
print("-" * (19 + 9 * len(FAMILIES)))

print()
print(f"{'channel':<12}{'power law':>11}{'best poly':>11}{'reduction':>11}")
print("-" * 45)
for nm, med in rows.items():
    pl = med["power law"]
    bp = min(v for k, v in med.items() if k != "power law")
    print(f"{nm:<12}{pl:>11.1f}{bp:>11.1f}{pl/bp:>10.1f}x")
print("-" * 45)

red = [rows[nm]["power law"] /
       min(v for k, v in rows[nm].items() if k != "power law")
       for nm in rows]
resid = [min(v for k, v in rows[nm].items() if k != "power law")
         for nm in rows]
print(f"a more flexible family reduces the discount by "
      f"{min(red):.1f}x to {max(red):.1f}x,")
print(f"leaving {min(resid):.1f} to {max(resid):.1f} -- a spread of "
      f"{max(resid)/min(resid):.1f}x, against {max(rows[nm]['power law'] for nm in rows) / min(rows[nm]['power law'] for nm in rows):.0f}x "
      f"under the power law.")
print()
print("If the residual spread collapses, the power law was the problem and the")
print("indicators are more alike than it suggested. If a large spread survives")
print("every family, the structure belongs to the indicators.")
json.dump(rows, open(os.path.join(HERE, "family_blame.json"), "w"), indent=2)
