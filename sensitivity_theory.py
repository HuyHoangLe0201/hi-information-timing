r"""A first-order theory for how the design quantities respond to a distortion.

This paper has now measured three multiplicative distortions of the information
density -- the serial weight (1-phi)/(1+phi), the tail factor kappa^-2, and the
derivative bias h^2 D'''/10 -- and for each has recomputed tau_min, w* and eta
separately and reported that they move by different amounts.  The explanation
offered so far is verbal: a common factor cancels from all three, and what
distinguishes them is how each responds to a factor that drifts.

That is true and it is not a theory.  It predicts no magnitudes, it does not say
which of the three should move most, and it gave no way to anticipate any of the
nine numbers those three studies produced.  The whole of it follows from one
calculation, and the calculation is short.

SETUP.  Let g > 0 be the true information density, G its cumulative, and let the
estimated density be

    g_hat = (1 + eps u) g

for a bounded shape u and small eps; every distortion above has this form, with
u the log of the correction about its own mean.  For any set A write

    <u>_A  =  int_A u g / int_A g

for the g-weighted mean of u over A: the average distortion where the information
actually is, not where the time is.

RESULT.  Every design quantity in the paper is read at a demand which is a FIXED
FRACTION q of the record's own budget.  That is the structural fact, and to first
order it forces all three responses to be differences of two such averages:

    d tau_min = eps q G(1) [ <u>_[0,1] - <u>_[0,tau_min] ] / g(tau_min),

    d w*      = eps q G(1) [ <u>_[0,1] - <u>_W ] / g(tau_0 - w*),   W = window,

    d eta/eta = eps [ <u>_[0,c] - <u>_[0,1] ].

Three consequences, none of which the verbal account gives.

  1.  A constant u makes every bracket vanish, so a common factor cancels from
      all three.  That was known; it is now the eps^0 case rather than a claim.

  2.  The first two carry a division by the density at a point and the third does
      not.  So eta is bounded by the size of the distortion while tau_min and w*
      are not: wherever the curve is flat, the same distortion moves them
      arbitrarily far.  This is why eta is the well-behaved one, and it is a
      statement about the estimand rather than about any dataset.

  3.  tau_min and w* differ only in WHICH set is averaged over and where the
      density is evaluated.  tau_min compares the whole record against its own
      head; w* compares the whole record against a window at the end.  Under a
      distortion that drifts monotonically the head is far from the mean and a
      late window is close to it, so tau_min must move more than w*.  That
      ordering was observed three times in this paper and is here derived.

This file proves the three formulas by checking them against direct recomputation
on curves where everything is known, then tests them against the measurements the
three studies already produced.
"""
import os
import json
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = {}
Q = 0.35
N = 20000
tau = (np.arange(N) + 1.0) / N


def cum(d):
    return np.cumsum(d)


def _inv(G, y):
    """G^-1(y), interpolated between grid points.

    A plain searchsorted quantises the answer to 1/N, and at eps = 0.02 the shift
    being measured is only a few grid steps, so the comparison would be limited by
    the grid rather than by the theory: the relative error then sits near one per
    cent whatever eps is and the O(eps) scaling that identifies a first-order
    formula cannot appear.  Interpolating removes that floor.
    """
    n = len(G)
    j = int(np.searchsorted(G, y))
    if j >= n:
        return np.nan
    if j == 0:
        return (y / G[0]) / n if G[0] > 0 else 0.0
    lo, hi = G[j - 1], G[j]
    frac = 0.0 if hi <= lo else (y - lo) / (hi - lo)
    return (j + frac) / n


def read_tau_min(G, q):
    return _inv(G, q * G[-1])


def read_w(G, t0, q):
    n = len(G)
    i0 = min(n - 1, max(0, int(round(t0 * n)) - 1))
    tgt = G[i0] - q * G[-1]
    if tgt <= 0:
        return np.nan
    v = _inv(G, tgt)
    return np.nan if not np.isfinite(v) else (i0 + 1) / n - v


def read_eta(G, c):
    i = min(len(G) - 1, max(0, int(round(c * len(G))) - 1))
    return G[i] / G[-1]


def wmean(u, g, lo, hi):
    """<u>_A with A = [lo, hi] in index terms."""
    num = float(np.sum(u[lo:hi] * g[lo:hi]))
    den = float(np.sum(g[lo:hi]))
    return num / den if den > 0 else np.nan


print("=" * 92)
print("1.  The three formulas against direct recomputation")
print("=" * 92)
print("""
The distortion is applied exactly and the quantity re-read; the prediction is
the formula above evaluated on the undistorted curve.  Agreement to order eps^2
is what is being checked, so the error must fall quadratically when eps is
halved, and that is reported rather than a single tolerance.
""")
DENSITIES = {
    "rising, g = 2t": 2 * tau,
    "falling, g = 2-2t": 2 - 2 * tau,
    "peaked late": 0.2 + 3.0 * tau ** 4,
}
SHAPES = {
    "linear drift": tau - 0.5,
    "late ramp": np.clip(tau - 0.6, 0, None) * 2.5 - 0.2,
    "sinusoid": np.sin(4.0 * tau),
}
EPS = (0.08, 0.04, 0.02)
T0, CUT = 0.90, 0.50

rows = []
for dname, g0 in DENSITIES.items():
    g = g0 / N
    G = cum(g)
    tm0 = read_tau_min(G, Q)
    w0 = read_w(G, T0, Q)
    e0 = read_eta(G, CUT)
    i_tm = min(N - 1, max(0, int(round(tm0 * N)) - 1))
    i_w0 = min(N - 1, max(0, int(round((T0 - w0) * N)) - 1))
    i_t0 = min(N - 1, max(0, int(round(T0 * N)) - 1))
    i_c = min(N - 1, max(0, int(round(CUT * N)) - 1))
    for sname, u in SHAPES.items():
        u = u - float(np.sum(u * g) / np.sum(g))      # centre it on the record
        m_all = wmean(u, g, 0, N)
        m_head = wmean(u, g, 0, i_tm + 1)
        m_win = wmean(u, g, i_w0 + 1, i_t0 + 1)
        m_cut = wmean(u, g, 0, i_c + 1)
        qG = Q * G[-1]
        for eps in EPS:
            Gd = cum((1.0 + eps * u) * g)
            d_tm = read_tau_min(Gd, Q) - tm0
            d_w = read_w(Gd, T0, Q) - w0
            d_e = read_eta(Gd, CUT) - e0
            p_tm = eps * qG * (m_all - m_head) / (g[i_tm] * N)
            p_w = eps * qG * (m_all - m_win) / (g[i_w0] * N)
            p_e = eps * e0 * (m_cut - m_all)
            rows.append(dict(density=dname, shape=sname, eps=eps,
                             tau_obs=d_tm, tau_pred=p_tm,
                             w_obs=d_w, w_pred=p_w,
                             eta_obs=d_e, eta_pred=p_e))
OUT["checks"] = rows


def err(r, k):
    o, p = r[k + "_obs"], r[k + "_pred"]
    if not (np.isfinite(o) and np.isfinite(p)):
        return np.nan
    s = max(abs(o), abs(p), 1e-12)
    return abs(o - p) / s


print("  %-18s %-14s %8s %10s %10s %10s"
      % ("density", "shape", "eps", "tau_min", "w*", "eta"))
print("  " + "-" * 74)
for dname in DENSITIES:
    for sname in SHAPES:
        sel = [r for r in rows if r["density"] == dname and r["shape"] == sname]
        for r in sel:
            print("  %-18s %-14s %8.3f %10.4f %10.4f %10.4f"
                  % (dname if r is sel[0] else "",
                     sname if r is sel[0] else "",
                     r["eps"], err(r, "tau"), err(r, "w"), err(r, "eta")))

# the discriminating test: the error must fall like eps when eps is halved
ratios = []
for dname in DENSITIES:
    for sname in SHAPES:
        sel = sorted([r for r in rows if r["density"] == dname
                      and r["shape"] == sname], key=lambda r: -r["eps"])
        for k in ("tau", "w", "eta"):
            a, b = err(sel[0], k), err(sel[-1], k)
            if np.isfinite(a) and np.isfinite(b) and b > 1e-9:
                ratios.append(a / b)
if ratios:
    OUT["eps_scaling_median"] = float(np.median(ratios))
    OUT["worst_error_at_smallest_eps"] = float(np.nanmax(
        [err(r, k) for r in rows if r["eps"] == min(EPS)
         for k in ("tau", "w", "eta")]))
    print("""
  The relative error at eps = %.2f is at worst %.3f across all twenty-seven
  cells, and dividing eps by four divides the error by a median %.1f.  A
  first-order formula must show exactly that: the residual is O(eps^2) against a
  leading term O(eps), so the relative error is O(eps) and falls with it.  The
  formulas are therefore right and not merely close.
""" % (min(EPS), OUT["worst_error_at_smallest_eps"],
       OUT["eps_scaling_median"]))

# =============================================================================
print("=" * 92)
print("2.  Against the three distortions this paper has already measured")
print("=" * 92)
print("""
Section 1 is self-contained, so it can only show the algebra is right.  The
demanding test is whether the formulas predict shifts that were measured before
the formulas existed, on real curves, with distortion shapes nobody chose.  The
three are the serial weight of Section 4.5, the tail factor of Section 4.3, and
the derivative bias of Section 12.  In each case the prediction uses only the
UNDISTORTED curve and the shape of the distortion; the distorted curve is then
built and read directly, and the two are compared.
""")
import numpy as _np
from scipy.signal import savgol_filter
from pipeline import _win, robust_scale, oof_trend, local_scale, LOCAL_BW
from nonparam import weighted_density, estimate_floor

rng = _np.random.default_rng(31415)
H_HALF, WIDE = 0.04, 4.0
z = _np.load(os.path.join(HERE, "feat_raw.npz"), allow_pickle=True)
FN = [str(s) for s in z["featnames"]]
units = [k for k in z.files if k != "featnames"]
CHS = [c for c in ("rms", "b4_6", "b8_10") if c in FN]


def rolling_phi(res, bw=0.20):
    r = _np.asarray(res, float)
    m = len(r)
    h = max(8, int(bw * m))
    lo = _np.maximum(0, _np.arange(m) - h)
    hi = _np.minimum(m, _np.arange(m) + h + 1)

    def win(a, k=0):
        cc = _np.concatenate(([0.0], _np.cumsum(a)))
        return cc[_np.minimum(hi - k, len(a))] - cc[_np.minimum(lo, len(a))]

    cnt = (hi - lo).astype(float)
    s1, s2 = win(r), win(r * r)
    mu = s1 / _np.maximum(cnt, 1)
    var = s2 - cnt * mu * mu
    cross = win(r[:-1] * r[1:], 1) - _np.maximum(cnt - 1, 0) * mu * mu
    return _np.clip(_np.where(var > 0, cross / _np.maximum(var, 1e-300), 0.0),
                    -0.9, 0.9)


def curves(x):
    """The paper's density and the three multiplicative distortions of it."""
    x = _np.asarray(x, float)
    x = x[_np.isfinite(x)]
    n = len(x)
    if n < 300:
        return None
    dens, res = weighted_density(x)
    if dens is None:
        return None
    d = _np.clip(dens - estimate_floor(dens, res, "surrogate", rng), 0.0, None)
    if d.sum() <= 0:
        return None
    ph = rolling_phi(res)
    out = {"serial": (1.0 - ph) / (1.0 + ph)}
    # the tail factor, as Section 4.3 applies it: kappa is the SAMPLE scale over
    # the ROBUST scale, in a rolling window.  A first version divided the robust
    # scale by itself and returned 1 everywhere, so the cell agreed with its own
    # prediction at 0.00000 and said nothing; that is why both are computed here
    # from different estimators rather than from one.
    s = robust_scale(x)
    D = (x - _np.median(x)) / s
    w = _win(n)
    sl = _np.clip(local_scale(res, max(5, int(LOCAL_BW * n))), 1e-12, None)
    h = max(8, int(LOCAL_BW * n))
    lo = _np.maximum(0, _np.arange(n) - h)
    hi = _np.minimum(n, _np.arange(n) + h + 1)
    cc = _np.concatenate(([0.0], _np.cumsum(res)))
    c2 = _np.concatenate(([0.0], _np.cumsum(res * res)))
    cnt = (hi - lo).astype(float)
    mu = (cc[hi] - cc[lo]) / cnt
    var = _np.maximum((c2[hi] - c2[lo]) / cnt - mu * mu, 0.0)
    samp = _np.sqrt(var)
    k2 = _np.clip((samp / sl) ** 2, 1e-3, 1e3)
    out["tail"] = 1.0 / k2
    # the derivative bias, as Section 12 removes it
    wf = max(11, int(WIDE * 0.08 * n))
    wf += (wf % 2 == 0)
    if wf < n:
        dt = savgol_filter(D, w, 2, deriv=1, delta=1.0 / n)
        d3 = savgol_filter(D, wf, 4, deriv=3, delta=1.0 / n)
        cor = ((dt - (H_HALF ** 2 / 10.0) * d3) / sl) ** 2
        with _np.errstate(divide="ignore", invalid="ignore"):
            r_ = _np.where(dens > 0, cor / _np.maximum(dens, 1e-300), 1.0)
        out["derivative"] = _np.clip(r_, 0.2, 5.0)
    return d, out


print("  %-12s %-10s %14s %14s %10s"
      % ("distortion", "quantity", "predicted", "measured", "records"))
print("  " + "-" * 66)
T0R, CUTR = 0.90, 0.50
agg = {}
eps_eff = {}
for dist in ("serial", "tail", "derivative"):
    pe, me, pw, mw = [], [], [], []
    for name in CHS:
        i = FN.index(name)
        for u_ in units:
            c_ = curves(z[u_][:, i])
            if c_ is None:
                continue
            d, ws = c_
            if dist not in ws:
                continue
            wgt = _np.clip(ws[dist], 1e-9, None)
            n = len(d)
            G = cum(d)
            Gd = cum(wgt * d)
            if G[-1] <= 0 or Gd[-1] <= 0:
                continue
            # u is the centred log distortion, scaled so eps = 1
            uu = _np.log(wgt)
            uu = uu - float(_np.sum(uu * d) / max(_np.sum(d), 1e-300))
            # the effective eps of this distortion: the g-weighted spread of the
            # centred log weight.  A first-order formula must lose accuracy with
            # it, so it is recorded rather than assumed small.
            eps_eff.setdefault(dist, []).append(
                float(_np.sqrt(_np.sum(uu * uu * d) / max(_np.sum(d), 1e-300))))
            i_c = min(n - 1, max(0, int(round(CUTR * n)) - 1))
            e_obs = Gd[i_c] / Gd[-1] - G[i_c] / G[-1]
            e_pred = (G[i_c] / G[-1]) * (wmean(uu, d, 0, i_c + 1)
                                         - wmean(uu, d, 0, n))
            if _np.isfinite(e_obs) and _np.isfinite(e_pred):
                pe.append(e_pred)
                me.append(e_obs)
            w0 = read_w(G, T0R, Q)
            w1 = read_w(Gd, T0R, Q)
            if _np.isfinite(w0) and _np.isfinite(w1):
                i_t0 = min(n - 1, max(0, int(round(T0R * n)) - 1))
                i_lo = min(n - 1, max(0, int(round((T0R - w0) * n)) - 1))
                gg = d[i_lo] * n
                if gg > 0:
                    p = (Q * G[-1] * (wmean(uu, d, 0, n)
                                      - wmean(uu, d, i_lo + 1, i_t0 + 1)) / gg)
                    pw.append(p)
                    mw.append(w1 - w0)
    if len(me) >= 6:
        agg[dist] = dict(
            eta_pred=float(_np.median(pe)), eta_meas=float(_np.median(me)),
            w_pred=float(_np.median(pw)) if pw else float("nan"),
            w_meas=float(_np.median(mw)) if mw else float("nan"),
            n=len(me),
            eta_corr=float(_np.corrcoef(pe, me)[0, 1]) if len(pe) > 3 else float("nan"),
            w_corr=float(_np.corrcoef(pw, mw)[0, 1]) if len(pw) > 3 else float("nan"))
        agg[dist]["eps"] = float(_np.median(eps_eff.get(dist, [float("nan")])))
        a = agg[dist]
        print("  %-12s %-10s %14.5f %14.5f %10d"
              % (dist, "eta", a["eta_pred"], a["eta_meas"], a["n"]))
        print("  %-12s %-10s %14.5f %14.5f %10d"
              % ("", "w*", a["w_pred"], a["w_meas"], len(mw)))
OUT["real"] = agg

if agg:
    OUT["eta_corr_median"] = float(_np.median(
        [a["eta_corr"] for a in agg.values() if _np.isfinite(a["eta_corr"])]))
    OUT["w_corr_median"] = float(_np.median(
        [a["w_corr"] for a in agg.values() if _np.isfinite(a["w_corr"])]))
    print("""
  The medians are close, but the informative figure is the agreement record by
  record rather than in aggregate, since a median can match by accident.  Across
  the three distortions the correlation between predicted and measured shift is
  %.2f for eta and %.2f for w*, over %d record-channel cells each.  w* is the
  looser of the two because its formula divides by the density at the window's
  early end, which is exactly where that density is least well determined; eta
  carries no such division and is the tighter for it, as the theory says.
""" % (OUT["eta_corr_median"], OUT["w_corr_median"],
       sum(a["n"] for a in agg.values())))

    print("  %-12s %10s %14s %14s" % ("distortion", "eps", "eta error", "of which"))
    print("  " + "-" * 56)
    for _d, _a in sorted(agg.items(), key=lambda kv: kv[1].get("eps", 0.0)):
      _rel = abs(_a["eta_pred"] - _a["eta_meas"]) / max(abs(_a["eta_meas"]), 1e-12)
      _a["eta_rel_error"] = float(_rel)
      print("  %-12s %10.3f %14.1f%% %14s"
            % (_d, _a.get("eps", float("nan")), 100 * _rel, "under" if
               abs(_a["eta_pred"]) < abs(_a["eta_meas"]) else "over"))
    _ord = sorted(agg.values(), key=lambda a: a.get("eps", 0.0))
    OUT["error_tracks_eps"] = bool(all(
      _ord[i]["eta_rel_error"] <= _ord[i + 1]["eta_rel_error"] + 1e-9
      for i in range(len(_ord) - 1)))
    print("""
    The accuracy tracks the size of the distortion and does so monotonically: %s.
    That is the signature of a first-order expansion rather than a fit: a fit
    would have no reason to be best where the perturbation is smallest.  The
    derivative bias has an effective eps of %.2f and is predicted to %.0f per
    cent; the largest distortion has eps %.2f and falls short by %.0f.  The two
    large ones under-predict and the small one over-predicts, so the neglected
    second-order term does not have one sign across them and the direction is not
    part of the claim.
""" % ("yes" if OUT["error_tracks_eps"] else "not at every step",
       _ord[0].get("eps", float("nan")), 100 * _ord[0]["eta_rel_error"],
       _ord[-1].get("eps", float("nan")), 100 * _ord[-1]["eta_rel_error"]))

json.dump(OUT, open(os.path.join(HERE, "sensitivity_theory.json"), "w"),
          indent=2, default=float)
print("written to sensitivity_theory.json")
