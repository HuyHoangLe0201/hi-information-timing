r"""How short a record may be, derived rather than conventional.

Two length screens are in use and the paper documents one of them.  Section 5
says records below sixty snapshots are excluded, "the sixty at which a curve can
be estimated", and adds that the only exclusions anywhere are for length and for
a feature that could not be extracted.  Every script behind the distortion
results of Section 2.4 -- the certificate, the contrast identity, the estimated
correction, the precision of eta -- instead requires three hundred.  On XJTU that
is seven records where the documented rule admits thirteen.

Neither number is derived.  This derives what can be derived and measures the
rest.

WHAT IS DERIVED.  The smoothing window is max(7, 0.08 n) samples, so below
n = 88 it stops being eight per cent of a life and becomes 7/n, and the same
happens to the noise-scale window below n = 63.  A record admitted at the
documented screen of sixty is therefore analysed at an effective bandwidth of
7/60, which is not the bandwidth the paper reports anywhere.  That is arithmetic
and it fixes one end of the question.

WHAT IS MEASURED.  The other end is precision, and decimation gives it without
synthetic data.  A record decimated by k has k different subsamples, one per
offset, all of the same length and all of the same bearing.  Their spread is the
sampling error of the measurement at that length, with the record held fixed --
which comparing different records of different lengths cannot separate.  The
screen should sit where that spread stops being small against the quantity being
read.

The confound is stated: decimation also thins the serial correlation, so the
serial correction is expected to DRIFT with k for a physical reason.  Drift and
spread are therefore reported separately, and the screen is read off the spread.
"""
import os
import json
import numpy as np
from scipy.signal import savgol_filter

from pipeline import robust_scale, _win, local_scale, LOCAL_BW, SMOOTH_FRAC
from nonparam import weighted_density, estimate_floor

HERE = os.path.dirname(os.path.abspath(__file__))
rng = np.random.default_rng(4242)
OUT = {}
MIN_SHARE = 0.02
H_HALF, WIDE = 0.04, 4.0
DISTORTIONS = ("serial", "tail", "derivative")
KS = (1, 2, 3, 4, 6, 8, 12)


# --- the machinery of the distortion block, unchanged except for the screen --
def rolling_phi(res, bw=0.20):
    r = np.asarray(res, float)
    m = len(r)
    h = max(8, int(bw * m))
    lo = np.maximum(0, np.arange(m) - h)
    hi = np.minimum(m, np.arange(m) + h + 1)

    def win(a, k=0):
        cc = np.concatenate(([0.0], np.cumsum(a)))
        return cc[np.minimum(hi - k, len(a))] - cc[np.minimum(lo, len(a))]

    cnt = (hi - lo).astype(float)
    s1, s2 = win(r), win(r * r)
    mu = s1 / np.maximum(cnt, 1)
    var = s2 - cnt * mu * mu
    cross = win(r[:-1] * r[1:], 1) - np.maximum(cnt - 1, 0) * mu * mu
    return np.clip(np.where(var > 0, cross / np.maximum(var, 1e-300), 0.0),
                   -0.9, 0.9)


def pieces(x, min_n):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    n = len(x)
    if n < min_n:
        return None
    dens, res = weighted_density(x)
    if dens is None:
        return None
    d = np.clip(dens - estimate_floor(dens, res, "surrogate", rng, reps=6),
                0.0, None)
    if d.sum() <= 0:
        return None
    s = robust_scale(x)
    D = (x - np.median(x)) / s
    w_ = _win(n)
    sl = np.clip(local_scale(res, max(5, int(LOCAL_BW * n))), 1e-12, None)
    out = {}
    ph = rolling_phi(res)
    out["serial"] = (1.0 - ph) / (1.0 + ph)
    h = max(8, int(LOCAL_BW * n))
    lo = np.maximum(0, np.arange(n) - h)
    hi = np.minimum(n, np.arange(n) + h + 1)
    cc = np.concatenate(([0.0], np.cumsum(res)))
    c2 = np.concatenate(([0.0], np.cumsum(res * res)))
    cnt = (hi - lo).astype(float)
    mu = (cc[hi] - cc[lo]) / cnt
    var = np.maximum((c2[hi] - c2[lo]) / cnt - mu * mu, 0.0)
    out["tail"] = 1.0 / np.clip((np.sqrt(var) / sl) ** 2, 1e-3, 1e3)
    wf = max(11, int(WIDE * SMOOTH_FRAC * n))
    wf += (wf % 2 == 0)
    if wf < n:
        dt = savgol_filter(D, w_, 2, deriv=1, delta=1.0 / n)
        d3 = savgol_filter(D, wf, 4, deriv=3, delta=1.0 / n)
        cor = ((dt - (H_HALF ** 2 / 10.0) * d3) / sl) ** 2
        with np.errstate(divide="ignore", invalid="ignore"):
            out["derivative"] = np.clip(
                np.where(dens > 0, cor / np.maximum(dens, 1e-300), 1.0),
                0.2, 5.0)
    return d, out


def log_contrast(w, g, ic):
    tot = float(np.sum(g))
    if tot <= 0:
        return np.nan
    if (float(np.sum(g[:ic + 1])) / tot < MIN_SHARE
            or float(np.sum(g[ic + 1:])) / tot < MIN_SHARE):
        return np.nan
    a = float(np.sum(w[:ic + 1] * g[:ic + 1]) / np.sum(g[:ic + 1]))
    b = float(np.sum(w[ic + 1:] * g[ic + 1:]) / np.sum(g[ic + 1:]))
    return np.log(a / b) if (a > 0 and b > 0) else np.nan


# =============================================================================
print("=" * 92)
print("1.  What the window floor does, which is arithmetic")
print("=" * 92)
print("""
The trend window is max(7, %.2f n) samples and the noise-scale window is
max(5, %.2f n).  Below the length at which each floor binds, the pipeline is no
longer running at the bandwidth the paper reports; it is running wider.
""" % (SMOOTH_FRAC, LOCAL_BW))
n_trend = int(np.ceil(7.0 / SMOOTH_FRAC))
n_scale = int(np.ceil(5.0 / LOCAL_BW))
OUT["floor_binds_trend"] = n_trend
OUT["floor_binds_scale"] = n_scale
print("  %-10s %14s %16s" % ("n", "trend window", "effective bandwidth"))
print("  " + "-" * 44)
eff = []
for n in (42, 52, 60, 88, 123, 161, 300, 500, 1000):
    w = _win(n)
    eff.append(dict(n=n, w=int(w), frac=float(w) / n))
    print("  %-10d %14d %15.1f%%" % (n, w, 100.0 * w / n))
OUT["effective_bandwidth"] = eff
OUT["bw_at_60"] = float(_win(60)) / 60.0
OUT["bw_at_42"] = float(_win(42)) / 42.0
print("""
  The trend window stops being %.0f per cent below n = %d and the noise-scale
  window below n = %d.  At the documented screen of sixty the effective trend
  bandwidth is %.1f per cent, and at the shortest XJTU record it is %.1f per
  cent: the two records the paper excludes for length would also have been
  analysed at twice the nominal bandwidth had they been kept.  The screen of
  sixty is therefore not arbitrary, but what it protects is the bandwidth and
  not the precision, and it protects it only loosely.
""" % (100 * SMOOTH_FRAC, n_trend, n_scale,
       100 * OUT["bw_at_60"], 100 * OUT["bw_at_42"]))

# =============================================================================
print("=" * 92)
print("2.  Precision against length, with the record held fixed")
print("=" * 92)
print("""
Each long bearing is decimated by k, giving k subsamples of equal length that
differ only in phase.  The spread of the measured log-contrast across those k is
the sampling error at that length; the shift of their mean away from the
undecimated value is drift, which for the serial correction is expected because
decimation thins the correlation it measures.
""")
z = np.load(os.path.join(HERE, "feat_raw.npz"), allow_pickle=True)
FN = [str(s) for s in z["featnames"]]
units = [k for k in z.files if k != "featnames"]
CH = [c for c in ("rms", "b4_6", "b8_10") if c in FN]
LONG = [u for u in units if z[u].shape[0] >= 1200]
OUT["long_records"] = len(LONG)
print("  %d bearings of at least 1200 samples: %s\n"
      % (len(LONG), ", ".join(LONG)))
print("  %-12s %8s %8s %12s %12s %10s"
      % ("distortion", "k", "length", "spread(log)", "drift(log)", "cells"))
print("  " + "-" * 66)
rows = []
base = {}
for u in LONG:
    for nm in CH:
        x = z[u][:, FN.index(nm)]
        p = pieces(x, 1)
        if p is None:
            continue
        g, ws = p
        ic = int(0.5 * len(g)) - 1
        for dist in DISTORTIONS:
            if dist in ws:
                v = log_contrast(np.clip(ws[dist], 1e-9, None), g, ic)
                if np.isfinite(v):
                    base[(u, nm, dist)] = v
for dist in DISTORTIONS:
    for k in KS:
        vals, drifts = [], []
        for u in LONG:
            for nm in CH:
                x = z[u][:, FN.index(nm)]
                got = []
                for off in range(k):
                    p = pieces(x[off::k], 1)
                    if p is None:
                        continue
                    g, ws = p
                    if dist not in ws:
                        continue
                    ic = int(0.5 * len(g)) - 1
                    v = log_contrast(np.clip(ws[dist], 1e-9, None), g, ic)
                    if np.isfinite(v):
                        got.append(v)
                if len(got) >= max(2, k) or (k == 1 and got):
                    if k > 1:
                        vals.append(float(np.std(got, ddof=1)))
                    b = base.get((u, nm, dist))
                    if b is not None:
                        drifts.append(float(np.mean(got) - b))
        if not drifts:
            continue
        ln = int(np.median([z[u].shape[0] for u in LONG]) / k)
        r = dict(distortion=dist, k=k, length=ln,
                 spread=float(np.median(vals)) if vals else 0.0,
                 drift=float(np.median(drifts)), cells=len(drifts))
        rows.append(r)
        print("  %-12s %8d %8d %12.4f %12.4f %10d"
              % (dist, k, ln, r["spread"], r["drift"], r["cells"]))
OUT["decimation"] = rows

# =============================================================================
print()
print("=" * 92)
print("3.  Where the spread overtakes the quantity")
print("=" * 92)
ref = json.load(open(os.path.join(HERE, "contrast_identity.json")))
typ = float(np.log(max(r["contrast"] for r in ref["corrections"])))
OUT["typical_log_contrast"] = typ
print("""
The largest log-contrast among the paper's three corrections is %.4f.  A screen
is defensible where the sampling spread of the measurement is a small fraction of
that; the table gives the shortest length at which the spread stays under a
quarter and under a half of it, per correction.
""" % typ)
print("  %-12s %22s %22s" % ("distortion", "spread < typ/4 above",
                             "spread < typ/2 above"))
print("  " + "-" * 60)
lim = {}
for dist in DISTORTIONS:
    rr = sorted([r for r in rows if r["distortion"] == dist and r["k"] > 1],
                key=lambda r: r["length"])
    out = []
    for frac in (0.25, 0.5):
        ok = [r["length"] for r in rr if r["spread"] < frac * typ]
        out.append(min(ok) if ok else None)
    lim[dist] = out
    print("  %-12s %22s %22s"
          % (dist, out[0] if out[0] else "never", out[1] if out[1] else "never"))
OUT["length_for_quarter"] = {k: v[0] for k, v in lim.items()}
OUT["length_for_half"] = {k: v[1] for k, v in lim.items()}
_q = [v for v in OUT["length_for_quarter"].values() if v]
OUT["derived_screen"] = int(max(_q)) if _q else None

json.dump(OUT, open(os.path.join(HERE, "length_screen.json"), "w"), indent=2,
          default=float)
print("\nwritten to length_screen.json")
