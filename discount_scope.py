"""
Is the correction a property of the indicator, or of the domain?

The corrections for nuisance parameters, family misfit and misfit correlation
multiply out to a discount that ranges from 3x to 276x, and applying it reorders
the indicators. That would invalidate every comparison made from the
uncorrected information -- but only if the discount really varies indicator by
indicator. The four indicators measured so far hint otherwise: the two bearing
channels sit at 166x and 276x, the two turbofan channels at 3x and 3x. Within a
domain the discount looks nearly constant, and a constant factor cancels out of
any ranking.

This tests that on every channel available, because it decides which of the
earlier comparisons survive. If the discount is a domain constant, comparisons
between indicators on the same equipment stand as made and only cross-domain
statements need it carried; if it varies within a domain, none of them do.
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


def discount_one(x):
    """Family misfit factor and its correlation length for one record."""
    n = len(x)
    s = robust_scale(x)
    if not np.isfinite(s) or s <= 0 or n < 100:
        return None
    D = (x - np.median(x)) / s
    w = _win(n)
    if w < 7:
        return None
    s_loc = robust_scale(D - oof_trend(D, w // 2))
    tau = np.arange(1, n + 1) / n
    try:
        p, _ = curve_fit(lambda t, A, b, c: A * t ** b + c, tau, D,
                         p0=[1.0, 2.0, float(D[0])],
                         bounds=([1e-6, 0.51, -1e3], [1e4, 30.0, 1e3]),
                         maxfev=40000)
    except Exception:
        return None
    fam = D - (p[0] * tau ** p[1] + p[2])
    s_fam = robust_scale(fam)
    if not np.isfinite(s_loc) or s_loc <= 0 or not np.isfinite(s_fam):
        return None
    ell = acorr_len(fam)
    if not np.isfinite(ell):
        return None
    return (s_fam / s_loc) ** 2 * max(ell, 1.0)


def channel(series):
    v = [discount_one(np.asarray(x, float)) for x in series]
    v = [q for q in v if q is not None and np.isfinite(q)]
    return float(np.median(v)) if len(v) >= 3 else None


DOMAINS = {}

z = np.load(os.path.join(HERE, "feat_raw.npz"), allow_pickle=True)
FN = [str(s) for s in z["featnames"]]
I = {k: i for i, k in enumerate(FN)}
BK = sorted(k for k in z.files if k != "featnames")
DOMAINS["bearings"] = {nm: [z[b][:, I[nm]] for b in BK] for nm in FN}

zt = np.load(os.path.join(HERE, "cmapss_FD001.npz"))
un = sorted({k.split("__")[0] for k in zt.files})[:50]
NCH = zt[f"{un[0]}__sensors"].shape[1]
live = [c for c in range(NCH)
        if np.std(zt[f"{un[0]}__sensors"][:, c].astype(float)) > 0]
DOMAINS["turbofan"] = {f"s{c+1}": [zt[f"{u}__sensors"][:, c].astype(float)
                                   for u in un] for c in live}

zb = np.load(os.path.join(HERE, "battery_raw.npz"))
cl = sorted({k.split("__")[0] for k in zb.files})
DOMAINS["cells"] = {nm: [zb[f"{c}__{nm}"] for c in cl]
                    for nm in ("cap", "Re", "Rct")}

print("misfit discount, channel by channel\n")
res = {}
for dom, chans in DOMAINS.items():
    vals = {}
    for nm, series in chans.items():
        d = channel(series)
        if d is not None:
            vals[nm] = d
    if len(vals) < 2:
        continue
    res[dom] = vals
    v = np.array(list(vals.values()))
    print(f"{dom}  ({len(vals)} channels)")
    for nm, d in sorted(vals.items(), key=lambda kv: kv[1]):
        print(f"    {nm:<12}{d:>10.1f}")
    print(f"    {'spread':<12}{v.max()/v.min():>10.1f}x  "
          f"(median {np.median(v):.1f})\n")

print("-" * 58)
print(f"{'comparison':<34}{'spread':>10}")
print("-" * 58)
within = []
for dom, vals in res.items():
    v = np.array(list(vals.values()))
    within.append(v.max() / v.min())
    print(f"{'within ' + dom:<34}{v.max()/v.min():>9.1f}x")
meds = np.array([np.median(list(v.values())) for v in res.values()])
print(f"{'across domains (medians)':<34}{meds.max()/meds.min():>9.1f}x")
print("-" * 58)
print(f"worst within-domain spread : {max(within):.1f}x")
print(f"across-domain spread       : {meds.max()/meds.min():.1f}x")
print()
if max(within) < meds.max() / meds.min():
    print("The discount varies less within a domain than between domains, so")
    print("it behaves like a property of the equipment and the trend family,")
    print("not of the individual indicator. Comparisons between indicators on")
    print("the same equipment are therefore largely unaffected by it; a")
    print("comparison across domains is not, and must carry it explicitly.")
else:
    print("The discount varies as much within a domain as between domains, so")
    print("it cannot be treated as a common factor and every comparison made")
    print("from the uncorrected information has to be revisited.")
json.dump(res, open(os.path.join(HERE, "discount_scope.json"), "w"), indent=2)
