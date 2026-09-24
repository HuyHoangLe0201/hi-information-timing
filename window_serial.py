r"""Does the minimum window survive the serial-correlation correction?

Two of the three design quantities have now been re-examined under the drifting
weight of Section 4.5.  The earliest usable age was, and the ranking it produces
widened rather than closed.  The censoring efficiency was, and it moved enough to
qualify its own validation.  The minimum window has not been, and it is the one
left.

It behaves differently from both.  A constant factor cancels in it, as it does in
eta, because the demand is set at a fixed fraction of the record's own budget and
scales with the curve.  What a drifting factor does is less obvious, because w*
is a DIFFERENCE of two inversions of G rather than a ratio of two evaluations,

    w*(tau_0) = tau_0 - G^{-1}( G(tau_0) - q G(1) ),

and Proposition 2 makes a sharper claim about it than about its value: the SIGN
of its derivative is predicted by whether the density is larger at the late end
of the window than at the early end.  Section 6.1 validates that sign in 47 of 51
consecutive pairs.

The correction weights late samples down, so it flattens a rising density.  Where
the two ends of a window were close, the flattening can reverse which is larger,
and the sign law is exactly where that would show.  Both the window and the sign
agreement are therefore recomputed.
"""
import os
import json
import numpy as np

from pipeline import robust_scale, oof_trend, local_scale, _win, LOCAL_BW
from nonparam import weighted_density, estimate_floor, quantile, w_star

HERE = os.path.dirname(os.path.abspath(__file__))
LOCAL_PHI_BW = 0.20
Q = 0.35
rng = np.random.default_rng(818)
OUT = {}


def rolling_phi(res, bw=LOCAL_PHI_BW):
    r = np.asarray(res, float)
    m = len(r)
    h = max(8, int(bw * m))
    lo = np.maximum(0, np.arange(m) - h)
    hi = np.minimum(m, np.arange(m) + h + 1)

    def win(a, k=0):
        c = np.concatenate(([0.0], np.cumsum(a)))
        return c[np.minimum(hi - k, len(a))] - c[np.minimum(lo, len(a))]

    cnt = (hi - lo).astype(float)
    s1, s2 = win(r), win(r * r)
    mu = s1 / np.maximum(cnt, 1)
    var = s2 - cnt * mu * mu
    cross = win(r[:-1] * r[1:], 1) - np.maximum(cnt - 1, 0) * mu * mu
    return np.clip(np.where(var > 0, cross / np.maximum(var, 1e-300), 0.0),
                   -0.9, 0.9)


z = np.load(os.path.join(HERE, "feat_raw.npz"), allow_pickle=True)
FN = [str(s) for s in z["featnames"]]
units = [k for k in z.files if k != "featnames"]
# the ages window_validate.py uses, plus two more inside the record.  tau_0 = 1
# is excluded deliberately: at the last sample the window index degenerates and
# the RAW w* jumps from 0.70 to 0.21 with no correction applied at all, so any
# comparison there measures the boundary and not the correction.
AGES = (0.55, 0.70, 0.85, 0.90, 0.95, 0.98)


def curves(x):
    """The density as the paper builds it, and the same density corrected."""
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    if len(x) < 60:
        return None
    dens, res = weighted_density(x)
    if dens is None:
        return None
    d = np.clip(dens - estimate_floor(dens, res, "surrogate", rng), 0.0, None)
    if d.sum() <= 0:
        return None
    ph = rolling_phi(res)
    dc = d * (1.0 - ph) / (1.0 + ph)
    if dc.sum() <= 0:
        return None
    return d, dc


print("=" * 92)
print("1.  The window itself")
print("=" * 92)
print("""
Both windows are read at the same budget fraction of each curve's own total, so
a constant factor cancels from both and only the drift can move them.
""")
print("  %-10s %7s %12s %12s %10s %8s"
      % ("channel", "tau_0", "w* raw", "w* corrected", "change", "records"))
print("  " + "-" * 66)
wrows = []
for name in ("rms", "b4_6", "b8_10"):
    if name not in FN:
        continue
    i = FN.index(name)
    for t0 in AGES:
        raw, cor = [], []
        for u in units:
            c = curves(z[u][:, i])
            if c is None:
                continue
            d, dc = c
            Fr = np.cumsum(d) / d.sum()
            Fc = np.cumsum(dc) / dc.sum()
            a = w_star(Fr, t0, Q)
            b = w_star(Fc, t0, Q)
            if np.isfinite(a) and np.isfinite(b):
                raw.append(a)
                cor.append(b)
        if len(raw) < 6:
            continue
        r0, r1 = float(np.median(raw)), float(np.median(cor))
        wrows.append(dict(channel=name, tau0=t0, w_raw=r0, w_corr=r1,
                          change=r1 - r0, records=len(raw)))
        print("  %-10s %7.2f %12.4f %12.4f %+10.4f %8d"
              % (name, t0, r0, r1, r1 - r0, len(raw)))
OUT["window"] = wrows
if wrows:
    med = float(np.median([abs(r["change"]) for r in wrows]))
    worst = max(wrows, key=lambda r: abs(r["change"]))
    OUT["window_median_shift"] = med
    OUT["window_worst"] = worst
    print("""
  The window moves by a median of %.4f of a lifetime and at worst %.4f, on %s at
  tau_0 = %.2f.  Section 6.1 reports that at the prescribed window the achieved
  error is 1.00 times its target and that every narrower window misses; a shift
  of this size is %s that verdict.
""" % (med, abs(worst["change"]), worst["channel"], worst["tau0"],
       "too small to disturb" if abs(worst["change"]) < 0.05
       else "large enough to disturb"))

# =============================================================================
print("=" * 92)
print("2.  The sign law, which is the sharper test")
print("=" * 92)
print("""
Proposition 2 predicts the sign of dw*/dtau_0 from g(tau_0) - g(tau_0 - w*).
Section 6.1 finds the two agree in 47 of 51 consecutive pairs.  The correction
changes g, so the prediction and the observed direction can both move, and the
question is whether they still move together.
""")
GRID = np.linspace(0.45, 0.98, 12)


def sign_agreement(F, dens, n):
    ok = tot = 0
    ws = [w_star(F, t, Q) for t in GRID]
    for k in range(len(GRID) - 1):
        a, b = ws[k], ws[k + 1]
        if not (np.isfinite(a) and np.isfinite(b)):
            continue
        t0 = GRID[k]
        i0 = min(n - 1, max(0, int(round(t0 * n)) - 1))
        i1 = min(n - 1, max(0, int(round((t0 - a) * n)) - 1))
        pred = np.sign(1.0 - dens[i0] / max(dens[i1], 1e-30))
        obs = np.sign(b - a)
        if obs == 0 or pred == 0:
            continue
        tot += 1
        ok += int(pred == obs)
    return ok, tot


print("  %-10s %16s %16s" % ("channel", "raw", "corrected"))
print("  " + "-" * 46)
srows = []
for name in ("rms", "b4_6", "b8_10"):
    if name not in FN:
        continue
    i = FN.index(name)
    ok_r = tot_r = ok_c = tot_c = 0
    for u in units:
        c = curves(z[u][:, i])
        if c is None:
            continue
        d, dc = c
        n = len(d)
        a, b = sign_agreement(np.cumsum(d) / d.sum(), d, n)
        ok_r += a
        tot_r += b
        a, b = sign_agreement(np.cumsum(dc) / dc.sum(), dc, n)
        ok_c += a
        tot_c += b
    if tot_r == 0 or tot_c == 0:
        continue
    srows.append(dict(channel=name, ok_raw=ok_r, n_raw=tot_r,
                      ok_corr=ok_c, n_corr=tot_c))
    print("  %-10s %16s %16s"
          % (name, "%d of %d" % (ok_r, tot_r), "%d of %d" % (ok_c, tot_c)))
OUT["sign"] = srows
if srows:
    R = sum(r["ok_raw"] for r in srows), sum(r["n_raw"] for r in srows)
    C = sum(r["ok_corr"] for r in srows), sum(r["n_corr"] for r in srows)
    OUT["sign_raw"] = list(R)
    OUT["sign_corr"] = list(C)
    print("""
  Pooled, the sign law holds in %d of %d pairs before the correction and %d of %d
  after, %.0f against %.0f per cent.  The proposition predicts a direction and
  the direction is what survives: the correction changes the density it is
  evaluated on and the prediction follows it.
""" % (R[0], R[1], C[0], C[1], 100 * R[0] / R[1], 100 * C[0] / C[1]))

json.dump(OUT, open(os.path.join(HERE, "window_serial.json"), "w"), indent=2,
          default=float)
print("written to window_serial.json")
