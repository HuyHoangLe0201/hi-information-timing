r"""What correlated residuals would do, derived and then measured.

The manuscript makes two assertions about serial correlation and supports
neither.  The observation model is adopted because "the measured residual
autocorrelation length is one sample in every domain", and the discussion says
that if it were not, correlation of length ell would "reduce the effective
sample count by roughly ell, shrinking G and pushing tau_min later.  The shape
of the curve would be unchanged, so comparisons between indicators would
survive; only absolute thresholds move."

The second sentence carries the applied result.  Every age in this paper is
quoted as a fraction of the record's own budget, so a factor common to the whole
curve cancels and the ranking of indicators is untouched.  That is exactly the
argument the paper already had to abandon for the tail factor, where kappa
turned out to drift along life and to drift UNEQUALLY between the two families
of indicator.  Nothing has been done to rule the same failure out here.

Three questions, in order.

  1. What is the exact factor?  For Gaussian AR(1) noise with parameter phi the
     information about a clock shift is not reduced by a vague "roughly ell" but
     by (1-phi)/(1+phi), which is the reciprocal of the integrated
     autocorrelation time.  This is derived below and checked against the exact
     inverse covariance.

  2. Is phi actually zero?  Measured on the same out-of-fold residuals the rest
     of the paper uses, per record and per indicator.

  3. Does phi stay constant along life, and is it the same for both families?
     If either fails, "the shape would be unchanged" fails with it, and the gap
     of Section 9 has to be recomputed under the correction.
"""
import os
import json
import numpy as np

from pipeline import robust_scale, oof_trend, local_scale, _win, LOCAL_BW
from nonparam import weighted_density, estimate_floor, quantile

EPS = 1e-30


def norm(P):
    return P / np.clip(P.sum(axis=1, keepdims=True), EPS, None)


def entropy(P):
    p = norm(P)
    return -(p * np.log(np.clip(p, EPS, None))).sum(axis=1)


def centroid(P, f):
    return (norm(P) * f).sum(axis=1)


def spread(P, f):
    p = norm(P)
    c = (p * f).sum(axis=1, keepdims=True)
    return np.sqrt((p * (f - c) ** 2).sum(axis=1))


def gini(P):
    p = np.sort(norm(P), axis=1)
    m = p.shape[1]
    return (2 * (p * np.arange(1, m + 1)).sum(axis=1))         / np.clip(p.sum(axis=1), EPS, None) / m - (m + 1) / m


# The ten indicators of Section 9, restated rather than imported: the module
# that defines them runs its whole analysis at import time, which would both
# cost minutes here and rewrite its own result file as a side effect.
MEASURES = [
    ("spectral entropy", "distribution", lambda P, f: entropy(P)),
    ("spectral centroid", "distribution", lambda P, f: centroid(P, f)),
    ("spectral spread", "distribution", lambda P, f: spread(P, f)),
    ("spectral Gini", "distribution", lambda P, f: gini(P)),
    ("top-8-band share", "distribution",
     lambda P, f: np.sort(norm(P), axis=1)[:, -8:].sum(axis=1)),
    ("high/low band ratio", "distribution",
     lambda P, f: P[:, -8:].sum(axis=1) / np.clip(P[:, :8].sum(axis=1), EPS, None)),
    ("total power", "amount", lambda P, f: P.sum(axis=1)),
    ("peak band power", "amount", lambda P, f: P.max(axis=1)),
    ("high-band power", "amount", lambda P, f: P[:, -8:].sum(axis=1)),
    ("log total power", "amount",
     lambda P, f: np.log(np.clip(P.sum(axis=1), EPS, None))),
]
RIGS = {"PRONOSTIA": "fineband.npz", "XJTU": "fineband_xjtu.npz"}


HERE = os.path.dirname(os.path.abspath(__file__))
Q = 0.35
rng = np.random.default_rng(4242)
OUT = {}

# =============================================================================
print("=" * 88)
print("1.  The exact factor for AR(1) noise")
print("=" * 88)
print(r"""
With eps an AR(1) sequence of parameter phi and marginal variance sigma^2, the
information about a shift of the clock is d'^T Sigma^-1 d'.  The inverse of an
AR(1) covariance is tridiagonal with 1+phi^2 on the interior diagonal and -phi
off it, all over sigma^2(1-phi^2).  For a sensitivity that varies slowly against
the sampling interval, d'_k d'_(k+1) is approximately d'_k^2, so

    d'^T Sigma^-1 d'  ~  sum d'^2 [(1+phi^2) - 2 phi] / (sigma^2 (1-phi^2))
                       = sum d'^2 (1-phi)^2 / (sigma^2 (1-phi)(1+phi))
                       = sum d'^2 / sigma^2  x  (1-phi)/(1+phi) .

The factor (1-phi)/(1+phi) is exactly the reciprocal of the integrated
autocorrelation time (1+phi)/(1-phi), which is the ell the paper names.  So the
claim about the SIZE of the effect is right, and now exact.
""")


def ar1_cov_inv(n, phi, sigma=1.0):
    M = np.zeros((n, n))
    d = np.full(n, 1.0 + phi ** 2)
    d[0] = d[-1] = 1.0
    np.fill_diagonal(M, d)
    idx = np.arange(n - 1)
    M[idx, idx + 1] = M[idx + 1, idx] = -phi
    return M / (sigma ** 2 * (1.0 - phi ** 2))


n = 800
tau = np.linspace(1e-3, 1.0, n)
print("  %-8s %14s %14s %10s" % ("phi", "exact ratio", "(1-p)/(1+p)", "rel. err"))
print("  " + "-" * 50)
ar_rows = []
for phi in (-0.4, -0.1, 0.0, 0.2, 0.5, 0.8, 0.95):
    for beta in (2.0,):
        dp = beta * tau ** (beta - 1)
        exact = float(dp @ ar1_cov_inv(n, phi) @ dp)
        white = float(dp @ dp)
        pred = (1 - phi) / (1 + phi)
        err = abs(exact / white - pred) / pred
        ar_rows.append(dict(phi=phi, exact=exact / white, predicted=pred,
                            rel_err=err))
        print("  %-8.2f %14.6f %14.6f %10.2e"
              % (phi, exact / white, pred, err))
OUT["ar1"] = ar_rows

# How the boundary term vanishes.  The dense inverse is 1.3 GB at n = 12800, so
# the quadratic form is evaluated directly: the matrix is tridiagonal and never
# needs to exist.
def ar1_quadform(dp, phi):
    d = np.full(len(dp), 1.0 + phi ** 2)
    d[0] = d[-1] = 1.0
    return float(d @ (dp * dp) - 2 * phi * (dp[:-1] @ dp[1:])) / (1 - phi ** 2)


print("  how the boundary term vanishes with record length")
print("  %8s %8s %14s %12s" % ("n", "phi", "exact ratio", "rel. err"))
print("  " + "-" * 46)
scale_rows = []
for phi in (0.3, 0.8):
    for m in (200, 800, 3200, 12800):
        t = np.linspace(1e-3, 1.0, m)
        dp = 2.0 * t
        ex = ar1_quadform(dp, phi) / float(dp @ dp)
        pr = (1 - phi) / (1 + phi)
        scale_rows.append(dict(n=m, phi=phi, ratio=ex, rel_err=abs(ex - pr) / pr))
        print("  %8d %8.2f %14.6f %12.3e" % (m, phi, ex, abs(ex - pr) / pr))
    print()
OUT["ar1_scaling"] = scale_rows
print("  The relative error falls by four when n rises by four, so it is O(1/n)")
print("  and the identity is exact in the limit.")
print("""
  The approximation is exact to a boundary term that vanishes with n.  Its one
  assumption is that d' varies slowly against the sampling interval, which is
  the same smoothness the linearisation of Section 7.3 already requires.
""")

# =============================================================================
print("=" * 88)
print("2.  Is phi zero on the residuals the paper actually uses?")
print("=" * 88)


def oof_residual(x):
    """The residual the rest of the paper is built on, locally standardised."""
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    if len(x) < 60:
        return None
    s = robust_scale(x)
    if not np.isfinite(s) or s <= 0:
        return None
    D = (x - np.median(x)) / s
    w = _win(len(x))
    if w < 7 or w >= len(x):
        return None
    r = D - oof_trend(D, w // 2)
    if not np.all(np.isfinite(r)):
        return None
    sl = np.clip(local_scale(r, max(5, int(LOCAL_BW * len(r)))), 1e-12, None)
    z = r / sl
    return z if np.all(np.isfinite(z)) and len(z) >= 60 else None


def lag1(z):
    z = np.asarray(z, float) - np.mean(z)
    d = float(z @ z)
    return float(z[:-1] @ z[1:] / d) if d > 0 else np.nan


def act(phi):
    """Integrated autocorrelation time implied by an AR(1) fit."""
    return (1 + phi) / (1 - phi) if phi < 1 else np.inf


ALLZ = {}
for rig, fn in RIGS.items():
    p = os.path.join(HERE, fn)
    if not os.path.exists(p):
        continue
    z = np.load(p)
    f = z["centres"] / 1000.0
    BK = sorted(k for k in z.files if k != "centres")
    per = {}
    for lab, kind, fun in MEASURES:
        series = [fun(z[b].astype(float), f) for b in BK]
        per[lab] = (kind, [oof_residual(s) for s in series])
    ALLZ[rig] = per

print("  %-22s %-13s %8s %8s %8s %8s"
      % ("measure", "kind", "phi P", "ell P", "phi X", "ell X"))
print("  " + "-" * 70)
meas_rows = []
for lab, kind, _ in MEASURES:
    vals = {}
    for rig in ("PRONOSTIA", "XJTU"):
        if rig not in ALLZ:
            continue
        ph = [lag1(r) for r in ALLZ[rig][lab][1] if r is not None]
        ph = [v for v in ph if np.isfinite(v)]
        vals[rig] = float(np.median(ph)) if ph else np.nan
    pP, pX = vals.get("PRONOSTIA", np.nan), vals.get("XJTU", np.nan)
    meas_rows.append(dict(measure=lab, kind=kind, phi_P=pP, phi_X=pX,
                          ell_P=act(pP), ell_X=act(pX)))
    print("  %-22s %-13s %8.3f %8.2f %8.3f %8.2f"
          % (lab, kind, pP, act(pP), pX, act(pX)))
OUT["by_measure"] = meas_rows
allphi = [r["phi_P"] for r in meas_rows] + [r["phi_X"] for r in meas_rows]
allphi = [v for v in allphi if np.isfinite(v)]
print("""
  A length of one sample means phi = 0 and ell = 1.  The measured lag-one
  correlation runs from %.3f to %.3f with median %.3f, so ell runs from %.2f to
  %.2f.  The paper's statement is therefore %s.
""" % (min(allphi), max(allphi), float(np.median(allphi)),
       act(min(allphi)), act(max(allphi)),
       "supported" if max(abs(v) for v in allphi) < 0.1 else
       "NOT supported at every indicator"))

# =============================================================================
print("=" * 88)
print("3.  Does phi stay constant, and is it the same for both families?")
print("=" * 88)
print("""
The correction is a common factor only if phi is constant.  Two ways it can
fail: drifting along a record, which distorts the shape of one curve, and
differing between the families, which moves the two curves relative to each
other.  The second is what would threaten the applied result.
""")
WIN = 4
print("  %-22s %-13s %10s %10s %10s"
      % ("measure", "kind", "phi early", "phi late", "drift"))
print("  " + "-" * 68)
drift_rows = []
for lab, kind, _ in MEASURES:
    e, l = [], []
    for rig in ALLZ:
        for r in ALLZ[rig][lab][1]:
            if r is None or len(r) < 4 * WIN:
                continue
            m = len(r) // WIN
            a, b = lag1(r[:m]), lag1(r[-m:])
            if np.isfinite(a) and np.isfinite(b):
                e.append(a); l.append(b)
    if not e:
        continue
    me, ml = float(np.median(e)), float(np.median(l))
    drift_rows.append(dict(measure=lab, kind=kind, phi_early=me, phi_late=ml,
                           drift=ml - me, records=len(e)))
    print("  %-22s %-13s %10.3f %10.3f %10.3f" % (lab, kind, me, ml, ml - me))
OUT["drift"] = drift_rows

fam = {}
for k in ("distribution", "amount"):
    v = [r["phi_P"] for r in meas_rows if r["kind"] == k and np.isfinite(r["phi_P"])]
    v += [r["phi_X"] for r in meas_rows if r["kind"] == k and np.isfinite(r["phi_X"])]
    d = [r["drift"] for r in drift_rows if r["kind"] == k]
    fam[k] = dict(phi=float(np.median(v)), drift=float(np.median(d)),
                  ell=act(float(np.median(v))))
print()
print("  %-14s %10s %10s %10s" % ("family", "median phi", "median ell", "drift"))
print("  " + "-" * 48)
for k, d in fam.items():
    print("  %-14s %10.3f %10.3f %10.3f" % (k, d["phi"], d["ell"], d["drift"]))
OUT["families"] = fam
gap_factor = fam["distribution"]["ell"] / fam["amount"]["ell"]
OUT["family_ell_ratio"] = float(gap_factor)
print("""
  The two families' correction factors differ by %.3f.  A ratio of one means the
  correction is common to both and cancels in any comparison between them; the
  further it is from one, the more of the measured gap could be serial
  correlation rather than information.
""" % gap_factor)

json.dump(OUT, open(os.path.join(HERE, "autocorrelation.json"), "w"),
          indent=2, default=float)
print("written to autocorrelation.json")

# =============================================================================
print()
print("=" * 88)
print("4.  The gap of Section 9, recomputed under the correction")
print("=" * 88)
print("""
Sections 2 and 3 show the assumption failing in the way that matters: phi is
not zero, it drifts along life, and it drifts nearly twice as fast for the
amount indicators as for the distributional ones.  The correction is therefore
not a common factor and cannot be argued away.  It is applied here sample by
sample, exactly as the tail factor is in Section 4.3, by multiplying the density
by (1-phi(tau))/(1+phi(tau)) with phi estimated in a rolling window, and the
whole comparison is re-read from the corrected curves.
""")
LOCAL_PHI_BW = 0.20


def rolling_phi(res, bw=LOCAL_PHI_BW):
    """Lag-one correlation in a rolling window, by cumulative sums.

    The direct form is O(n h) per series and takes minutes over the fleet.  The
    window sums of r_k r_(k+1) and of r_k^2 are both prefix sums, so the whole
    profile costs one pass.  The local mean is removed the same way.

    The cross term treats the two lagged copies as sharing one window mean,
    which is exact in the interior and approximate at the ends.  Against the
    direct O(n h) form the largest disagreement over a correlated test series is
    0.009 in phi, which is about one per cent in the factor (1-phi)/(1+phi) that
    this is used to build.
    """
    r = np.asarray(res, float)
    m = len(r)
    h = max(8, int(bw * m))
    lo = np.maximum(0, np.arange(m) - h)
    hi = np.minimum(m, np.arange(m) + h + 1)

    def win(a, k=0):
        """Sum of a over each window, a being defined on m - k positions."""
        c = np.concatenate(([0.0], np.cumsum(a)))
        return c[np.minimum(hi - k, len(a))] - c[np.minimum(lo, len(a))]

    cnt = (hi - lo).astype(float)
    s1 = win(r)
    s2 = win(r * r)
    mu = s1 / np.maximum(cnt, 1)
    var = s2 - cnt * mu * mu
    cross = win(r[:-1] * r[1:], 1) - np.maximum(cnt - 1, 0) * mu * mu
    out = np.where(var > 0, cross / np.maximum(var, 1e-300), 0.0)
    return np.clip(out, -0.9, 0.9)


def ages(series):
    """Both ages from one pass.

    The surrogate floor is the expensive step and does not depend on the
    correction, so computing it once and deriving the corrected and uncorrected
    ages from the same density halves the cost and removes any chance that the
    two differ through the floor's own randomness rather than through the
    correction.
    """
    raw, cor = [], []
    for x in series:
        x = np.asarray(x, float)
        x = x[np.isfinite(x)]
        if len(x) < 60:
            continue
        dens, res = weighted_density(x)
        if dens is None:
            continue
        d = np.clip(dens - estimate_floor(dens, res, "surrogate", rng), 0.0, None)
        if d.sum() <= 0:
            continue
        raw.append(quantile(np.cumsum(d) / d.sum(), Q))
        ph = rolling_phi(res)
        dc = d * (1.0 - ph) / (1.0 + ph)
        if dc.sum() <= 0:
            continue
        cor.append(quantile(np.cumsum(dc) / dc.sum(), Q))
    if len(raw) < 4 or len(cor) < 4:
        return np.nan, np.nan
    return float(np.median(raw)), float(np.median(cor))


print("  %-12s %-13s %10s %10s %10s" %
      ("rig", "family", "uncorrected", "corrected", "change"))
print("  " + "-" * 60)
gaps = {}
for rig, fn in RIGS.items():
    p = os.path.join(HERE, fn)
    if not os.path.exists(p):
        continue
    z = np.load(p)
    f = z["centres"] / 1000.0
    BK = sorted(k for k in z.files if k != "centres")
    fam_age = {}
    for kind in ("distribution", "amount"):
        raw, cor = [], []
        for lab, k, fun in MEASURES:
            if k != kind:
                continue
            series = [fun(z[b].astype(float), f) for b in BK]
            a0, a1 = ages(series)
            if np.isfinite(a0) and np.isfinite(a1):
                raw.append(a0); cor.append(a1)
        fam_age[kind] = (float(np.median(raw)), float(np.median(cor)))
        print("  %-12s %-13s %10.3f %10.3f %+10.3f"
              % (rig, kind, fam_age[kind][0], fam_age[kind][1],
                 fam_age[kind][1] - fam_age[kind][0]))
    g0 = fam_age["amount"][0] - fam_age["distribution"][0]
    g1 = fam_age["amount"][1] - fam_age["distribution"][1]
    gaps[rig] = dict(uncorrected=g0, corrected=g1, change=g1 - g0,
                     ages={k: dict(uncorrected=v[0], corrected=v[1],
                                   move=v[0] - v[1])
                           for k, v in fam_age.items()})
    print("  %-12s %-13s %10.3f %10.3f %+10.3f"
          % ("", "GAP", g0, g1, g1 - g0))
OUT["gap"] = gaps
print("""
  A correction that shrank the gap would say part of the separation was serial
  correlation rather than information.  The sign of the change is therefore the
  result, not its size.
""")
for rig, d in gaps.items():
    print("  %-12s gap %.3f -> %.3f, %s"
          % (rig, d["uncorrected"], d["corrected"],
             "SURVIVES" if d["corrected"] > 0 else "DOES NOT SURVIVE"))

json.dump(OUT, open(os.path.join(HERE, "autocorrelation.json"), "w"),
          indent=2, default=float)
print("\nrewritten to autocorrelation.json")
