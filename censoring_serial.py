r"""Does the censoring efficiency survive the serial-correlation correction?

Section 4.5 established that the residuals carry a lag-one correlation between
0.067 and 0.399, that the information is divided by the integrated
autocorrelation time (1+phi)/(1-phi), and, decisively, that phi DRIFTS along
life: a median of about 0.03 early and 0.30 late.  A correction that drifts is
not a common factor, and the paper only argued that the ranking of indicators
survives it.

The censoring efficiency was never re-examined, and it is the one design
quantity that cannot be protected by the same argument.  It is a RATIO of the
curve to itself,

    eta(c) = G(c) / G(1),

so a factor common to the whole curve does cancel exactly -- but a factor that
falls along life does not.  Weighting the late part of the record down more than
the early part raises the share of the budget the early part holds, so eta must
increase, and the increase must grow with how far apart the early and late
weights are.  That is a directional prediction, and the size of it decides
whether the two per cent agreement Section 6.2 reports still stands.
"""
import os
import json
import numpy as np

from pipeline import robust_scale, oof_trend, local_scale, _win, LOCAL_BW
from nonparam import weighted_density, estimate_floor

HERE = os.path.dirname(os.path.abspath(__file__))
LOCAL_PHI_BW = 0.20
rng = np.random.default_rng(515)
OUT = {}


def rolling_phi(res, bw=LOCAL_PHI_BW):
    """Lag-one correlation in a rolling window, by cumulative sums.

    The same estimator Section 4.5 uses, restated rather than imported so that
    this file does not re-run that analysis on import.
    """
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

CUTS = (0.30, 0.50, 0.70, 0.90)
print("=" * 92)
print("The censoring efficiency, before and after the correction")
print("=" * 92)
print("""
eta(c) is a ratio of the curve to itself, so a constant correction cancels.  The
serial-correlation weight (1-phi)/(1+phi) is not constant: phi rises along life,
so the weight falls, and the early share of the budget rises with it.  The
column that matters is the last, because the paper's prediction is that the
achievable error rises by eta^(-1/2), and an error is what a practitioner meets.
""")
print("  %-22s %6s %10s %10s %9s %12s"
      % ("channel", "cut c", "eta raw", "eta corr", "ratio", "error factor"))
print("  " + "-" * 76)
rows = []
for name in ("rms", "b4_6", "b8_10"):
    if name not in FN:
        continue
    i = FN.index(name)
    for c in CUTS:
        raw_e, cor_e = [], []
        for u in units:
            x = np.asarray(z[u][:, i], float)
            x = x[np.isfinite(x)]
            if len(x) < 60:
                continue
            dens, res = weighted_density(x)
            if dens is None:
                continue
            d = np.clip(dens - estimate_floor(dens, res, "surrogate", rng),
                        0.0, None)
            if d.sum() <= 0:
                continue
            w = (1.0 - rolling_phi(res)) / (1.0 + rolling_phi(res))
            dc = d * w
            if dc.sum() <= 0:
                continue
            k = max(1, int(round(c * len(d))))
            raw_e.append(float(d[:k].sum() / d.sum()))
            cor_e.append(float(dc[:k].sum() / dc.sum()))
        if len(raw_e) < 6:
            continue
        r0, r1 = float(np.median(raw_e)), float(np.median(cor_e))
        rows.append(dict(channel=name, cut=c, eta_raw=r0, eta_corr=r1,
                         ratio=r1 / r0, err=(r1 / r0) ** -0.5,
                         records=len(raw_e)))
        print("  %-22s %6.2f %10.4f %10.4f %9.3f %12.4f"
              % (name, c, r0, r1, r1 / r0, (r1 / r0) ** -0.5))
OUT["rows"] = rows

if rows:
    up = sum(1 for r in rows if r["ratio"] > 1.0)
    worst = max(rows, key=lambda r: abs(r["err"] - 1.0))
    med = float(np.median([abs(r["err"] - 1.0) for r in rows]))
    OUT["cells_up"] = up
    OUT["cells"] = len(rows)
    OUT["worst_error_shift"] = abs(worst["err"] - 1.0)
    OUT["median_error_shift"] = med
    OUT["worst_cell"] = worst
    print("""
  The direction is as predicted in %d of %d cells: down-weighting the late part
  of a record raises the share the early part holds.  The size is what decides
  whether Section 6.2 stands.  Translated into the quantity that is actually
  validated, the achievable error at the cut, the correction moves it by a median
  of %.1f per cent and at worst %.1f per cent, on %s at c = %.2f.
""" % (up, len(rows), 100 * med, 100 * OUT["worst_error_shift"],
       worst["channel"], worst["cut"]))
    print("""  Section 6.2 reports agreement between predicted and measured
  truncation cost to two per cent.  A correction of %s that figure therefore
  %s the validation as reported.
""" % ("%.1f per cent" % (100 * med),
       "sits inside" if 100 * med < 2.0 else "exceeds, and so qualifies,"))

# --- which channel carries the paper's validation, and how far its phi drifts --
print("=" * 92)
print("Which channel the validation rests on")
print("=" * 92)
print("""
Table 6 quotes median eta of 0.015 at c = 0.30 and 0.017 at c = 0.50, which are
the root-mean-square channel's values and not the bands'.  The validation is
therefore carried by the one channel whose phi drifts furthest, so the two facts
have to be read together.
""")
print("  %-10s %11s %11s %10s" % ("channel", "phi early", "phi late", "drift"))
print("  " + "-" * 46)
drift = []
for nm in ("rms", "b4_6", "b8_10"):
    if nm not in FN:
        continue
    i = FN.index(nm)
    E, L = [], []
    for u in units:
        x = np.asarray(z[u][:, i], float)
        x = x[np.isfinite(x)]
        if len(x) < 60:
            continue
        sc = robust_scale(x)
        if not np.isfinite(sc) or sc <= 0:
            continue
        D = (x - np.median(x)) / sc
        w_ = _win(len(x))
        if w_ < 7 or w_ >= len(x):
            continue
        r = D - oof_trend(D, w_ // 2)
        sl = np.clip(local_scale(r, max(5, int(LOCAL_BW * len(r)))), 1e-12, None)
        ph = rolling_phi(r / sl)
        m = len(ph) // 4
        E.append(float(np.median(ph[:m])))
        L.append(float(np.median(ph[-m:])))
    if not E:
        continue
    e, l = float(np.median(E)), float(np.median(L))
    drift.append(dict(channel=nm, phi_early=e, phi_late=l, drift=l - e))
    print("  %-10s %11.3f %11.3f %10.3f" % (nm, e, l, l - e))
OUT["drift"] = drift
_rms = next((d for d in drift if d["channel"] == "rms"), None)
if _rms:
    print("""
  The root-mean-square channel drifts by %.2f, three to six times the bands.  It
  delivers its budget in the last few per cent of life, so down-weighting the end
  moves its eta more than any other channel's, and it is the channel the quoted
  medians come from.
""" % _rms["drift"])

json.dump(OUT, open(os.path.join(HERE, "censoring_serial.json"), "w"),
          indent=2, default=float)
print("written to censoring_serial.json")
