"""
The effective exponent of real indicators, and the nuisance factor it implies.

The closed forms derived in theory_exact.py are exact for the power-law class.
This study deliberately abandoned a power-law parameterisation, so applying the
correction 1/(16 beta^4) to real records requires beta to be MEASURED rather than
assumed, and requires knowing whether a single exponent describes a record well
enough for the correction to mean anything.

Two quantities are computed on every channel of every unit.

The local exponent, beta(tau) = tau D'(tau) / (D(tau) - D(0)), is what the
power-law family would have to be to reproduce the observed trend at that age.
It is reported at the median over the last half of life, where the trend is
established; the first half is excluded because D - D(0) is near zero there and
the ratio is dominated by noise.

The spread of beta(tau) over that same window says whether one exponent suffices.
A record whose local exponent wanders by an order of magnitude is not a power law
at all, and no single correction factor applies to it; the paper must say so
rather than quoting a number.

The trend used is the out-of-fold Savitzky-Golay trend the rest of the study
uses, so the exponent is measured on the same object the information curve is
built from and not on a separately smoothed series.
"""
import os
import json
import numpy as np

from nonparam import robust_scale, oof_trend, _win

HERE = os.path.dirname(os.path.abspath(__file__))
LATE = 0.5              # window over which the exponent is read
MIN_N = 60


def local_exponent(x):
    """beta(tau) = tau D'(tau) / (D(tau) - D(0)), on the out-of-fold trend."""
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    n = len(x)
    if n < MIN_N:
        return None
    s = robust_scale(x)
    if not np.isfinite(s) or s <= 0:
        return None
    D = (x - np.median(x)) / s
    w = _win(n)
    T = oof_trend(D, w // 2)
    if T is None or not np.isfinite(T).all():
        return None
    # orient so the trend rises, which the exponent definition presumes
    tau = (np.arange(n) + 1.0) / n
    if np.median(T[-n // 4:]) < np.median(T[:n // 4]):
        T = -T
    T0 = float(np.median(T[:max(5, n // 20)]))
    dT = np.gradient(T, tau)
    lo = int(LATE * n)
    num = tau[lo:] * dT[lo:]
    den = T[lo:] - T0
    ok = np.isfinite(num) & np.isfinite(den) & (den > 1e-9)
    if ok.sum() < 10:
        return None
    b = num[ok] / den[ok]
    b = b[np.isfinite(b) & (b > 0)]
    if len(b) < 10:
        return None
    q1, q3 = np.percentile(b, [25, 75])
    return dict(beta=float(np.median(b)), iqr_ratio=float(q3 / max(q1, 1e-9)),
                n=int(n))


def load(fn, key="featnames"):
    z = np.load(os.path.join(HERE, fn), allow_pickle=True)
    names = [str(s) for s in z[key]]
    return z, names


z, FN = load("feat_raw.npz")
units = sorted(k for k in z.files if k != "featnames")
I = {k: i for i, k in enumerate(FN)}

print(f"effective exponent on the late half of life, {len(units)} bearings, "
      f"{len(FN)} channels\n")
print(f"{'channel':<10}{'units':>7}{'median beta':>13}{'IQR of beta':>26}"
      f"{'r = 1/(16 b^4)':>17}")
print(f"{'':<10}{'':>7}{'':>13}{'25th':>9}{'75th':>9}{'q3/q1':>8}{'':>17}")
print("-" * 73)
rows = []
allb, allspread = [], []
for nm in FN:
    got = [local_exponent(np.asarray(z[u][:, I[nm]], float)) for u in units]
    got = [g for g in got if g is not None]
    if len(got) < 5:
        continue
    b = np.array([g["beta"] for g in got])
    sp = np.array([g["iqr_ratio"] for g in got])
    allb.append(b)
    allspread.append(sp)
    med = float(np.median(b))
    q1, q3 = np.percentile(b, [25, 75])
    rows.append(dict(channel=nm, units=len(got), beta=med, q1=float(q1),
                     q3=float(q3), spread=float(np.median(sp)),
                     r=1.0 / (16 * med ** 4)))
    print(f"{nm:<10}{len(got):>7}{med:>13.2f}{q1:>9.2f}{q3:>9.2f}"
          f"{np.median(sp):>8.1f}{1.0 / (16 * med ** 4):>17.2e}")
print("-" * 73)

B = np.concatenate(allb)
SP = np.concatenate(allspread)
bmed = float(np.median(B))
print(f"pooled over all channels and units: median beta {bmed:.2f}, "
      f"interquartile {np.percentile(B, 25):.2f} to {np.percentile(B, 75):.2f}")
print(f"pooled within-record spread of beta(tau): median q3/q1 "
      f"{np.median(SP):.1f}, upper quartile {np.percentile(SP, 75):.1f}")
print()

# --- does one exponent describe a record? ----------------------------------
tight = float((SP < 2.0).mean())
print(f"records whose local exponent varies by less than a factor two over the")
print(f"late half of life: {100 * tight:.0f}%")
print()
if tight < 0.5:
    print("A single exponent therefore does NOT describe most records, and the")
    print("closed-form correction must be read as an order of magnitude for the")
    print("size of the oracle gap rather than as a per-record adjustment.")
else:
    print("A single exponent describes most records over this window, so the")
    print("closed-form correction may be applied per record.")
print()

r_at = 1.0 / (16 * bmed ** 4)
delay = (1.0 / r_at) ** (1.0 / (2 * bmed - 1)) if bmed > 0.5 else np.nan
print(f"At the pooled median exponent {bmed:.2f}, the correction is")
print(f"r = {r_at:.2e}, an information loss of {1 / r_at:.0f}-fold, and a delay of the")
print(f"earliest usable age by a factor {delay:.2f}.")
print()
lo, hi = float(np.percentile(B, 25)), float(np.percentile(B, 75))
dlo = (16 * lo ** 4) ** (1.0 / (2 * lo - 1)) if lo > 0.5 else np.nan
dhi = (16 * hi ** 4) ** (1.0 / (2 * hi - 1)) if hi > 0.5 else np.nan
print(f"Across the interquartile range of exponents the delay factor runs")
print(f"{min(dlo, dhi):.2f} to {max(dlo, dhi):.2f}, so the conclusion that the gap is large in")
print("information and modest in design does not depend on the exact exponent.")

json.dump(dict(channels=rows, pooled_beta=bmed,
               beta_q1=lo, beta_q3=hi,
               spread_median=float(np.median(SP)),
               frac_single_exponent=tight,
               r_at_median=r_at, delay_at_median=float(delay),
               delay_range=[float(min(dlo, dhi)), float(max(dlo, dhi))]),
          open(os.path.join(HERE, "exponent_measure.json"), "w"), indent=2,
          default=float)
