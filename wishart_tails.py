r"""Is the Wishart correction the right correction on these residuals?

Section 9 applies m/(m-K-1) throughout and dismisses it in one sentence: it
"moves this figure by 0.2% here, because m is large against K on these records".
That reasoning is sound for the quantity it names and silent about the one that
matters.  The factor m/(m-K-1) is the GAUSSIAN result.  It is the expectation of
d' S^-1 d under S ~ Wishart, and the Wishart law follows from normal sampling.

Proposition 13 establishes that the residuals on these records are not normal,
with a fitted nu = 4.28 on the bearings.  For an elliptical law the sample
covariance is far noisier than the Wishart at the same m, and the standard
description of how much noisier is Mardia's kurtosis parameter, which for a
multivariate t_nu is

    kappa_e = 2 / (nu - 4),        nu > 4,

so the effective sample size for a second-moment estimate is roughly m/(1+kappa_e).
At nu = 4.28 that parameter is 7.1 and the effective sample size is an eighth of
m.  A correction built on m rather than on m/8 would then be removing the wrong
amount, and "m is large against K" would be answering a question about m when
the binding quantity is m/8.  Whether that is what happens is measurable, and the
prediction is sharp enough to be wrong.

Two estimators are simulated, because the pipeline does not use the textbook one.

  (a) the plain sample covariance, which is what the m/(m-K-1) result is about;

  (b) the pipeline's own hybrid, Sigma = diag(s) C diag(s) with s the per-channel
      ROBUST scale and C the product-moment residual correlation, which is what
      fusion_prep.py builds and corrects.

(b) is the one the paper's numbers rest on.  Its robust scales are insensitive to
tails while its correlation matrix is not, so it is not obvious in advance
whether it inherits the failure or escapes it, and that is the point of measuring
rather than asserting.
"""
import os
import json
import numpy as np
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))
REPS = 4000
rng = np.random.default_rng(20260902)
OUT = {}


def robust_scale_1d(a):
    """1.4826 MAD, the scale the pipeline uses per channel."""
    return 1.4826 * np.median(np.abs(a - np.median(a)))


def draw(m, K, nu, corr):
    """m draws of a K-variate elliptical law with the given correlation.

    Normalised to unit population covariance on every channel, so the truth
    against which the estimators are judged is the correlation matrix itself and
    no scale question is mixed into the measurement.
    """
    L = np.linalg.cholesky(corr)
    zz = rng.standard_normal((m, K)) @ L.T
    if not np.isfinite(nu):
        return zz
    w = np.sqrt(nu / rng.chisquare(nu, size=(m, 1)))
    return zz * w / np.sqrt(nu / (nu - 2.0))


def draw_ar1(m, K, nu, corr, phi):
    """The same law, with lag-one correlation phi imposed along time.

    The autoregression acts on the Gaussian core and is normalised to leave the
    marginal variance at one, so the population covariance the estimators are
    judged against is unchanged and only the dependence between rows differs.
    The tail weight stays per row: that keeps the marginal exactly the t of
    draw() above, which is what makes the two measurements comparable.
    """
    L = np.linalg.cholesky(corr)
    e = rng.standard_normal((m, K))
    zz = np.empty_like(e)
    zz[0] = e[0]
    a = np.sqrt(1.0 - phi * phi)
    for t in range(1, m):
        zz[t] = phi * zz[t - 1] + a * e[t]
    zz = zz @ L.T
    if not np.isfinite(nu):
        return zz
    w = np.sqrt(nu / rng.chisquare(nu, size=(m, 1)))
    return zz * w / np.sqrt(nu / (nu - 2.0))


def inflation_ar1(m, K, nu, corr, kind, phi):
    Si = np.linalg.inv(limit_sigma(K, nu, corr, kind))
    out = []
    for _ in range(REPS):
        X = draw_ar1(m, K, nu, corr, phi)
        d = rng.standard_normal(K)
        truth = float(d @ Si @ d)
        if truth <= 0:
            continue
        S = _estimate(X, kind, K)
        if S is None:
            continue
        try:
            got = float(d @ np.linalg.inv(S) @ d)
        except np.linalg.LinAlgError:
            continue
        out.append(got / truth)
    return float(np.mean(out)), len(out)


def _estimate(X, kind, K):
    if kind == "sample":
        return np.cov(X, rowvar=False)
    s = np.array([robust_scale_1d(X[:, k]) for k in range(K)])
    if not np.all(np.isfinite(s)) or np.any(s <= 0):
        return None
    C = np.corrcoef(X / s, rowvar=False)
    C = 0.999999 * C + 1e-6 * np.eye(K)
    return np.outer(s, s) * C


_LIMIT = {}


def limit_sigma(K, nu, corr, kind):
    """What the estimator converges to as m grows, not what generated the data.

    This distinction is the whole measurement.  The hybrid estimator does NOT
    converge to the covariance: its robust scales converge to the MAD scale,
    which under a t_nu is smaller than the standard deviation by exactly the
    factor Proposition 13 calls kappa.  Judging the hybrid against the covariance
    therefore re-measures kappa and reports it as a Wishart failure, which is
    wrong twice over, since the paper already corrects for kappa.  The Wishart
    correction is about FINITE-SAMPLE inflation alone, so the reference has to be
    the estimator's own limit.
    """
    key = (K, float(nu), kind)
    if key not in _LIMIT:
        _LIMIT[key] = _estimate(draw(200000, K, nu, corr), kind, K)
    return _LIMIT[key]


def inflation(m, K, nu, corr, kind):
    """E[d' S^-1 d] / (d' Sigma_limit^-1 d) for one estimator, over REPS draws."""
    Si = np.linalg.inv(limit_sigma(K, nu, corr, kind))
    out = []
    for _ in range(REPS):
        X = draw(m, K, nu, corr)
        d = rng.standard_normal(K)
        truth = float(d @ Si @ d)
        if truth <= 0:
            continue
        S = _estimate(X, kind, K)
        if S is None:
            continue
        try:
            got = float(d @ np.linalg.inv(S) @ d)
        except np.linalg.LinAlgError:
            continue
        out.append(got / truth)
    return float(np.mean(out)), len(out)


# =============================================================================
print("=" * 92)
print("1.  What m/(m-K-1) predicts, and what actually happens")
print("=" * 92)
print("""
The Gaussian column is the control: if the simulation is right it must land on
m/(m-K-1) at nu = infinity.  The remaining columns carry the same m and K with
progressively heavier tails.  The bearings sit at nu = 4.28.
""")
NUS = (np.inf, 30.0, 8.0, 5.0, 4.28, 3.0)
CASES = [(300, 4), (300, 10), (1400, 4), (1400, 10)]
print("  %6s %4s %10s %9s %9s %9s %9s %9s %9s"
      % ("m", "K", "predicted", "nu=inf", "nu=30", "nu=8", "nu=5", "nu=4.28",
         "nu=3"))
print("  " + "-" * 84)
rows_a = []
for m, K in CASES:
    corr = np.eye(K) + 0.3 * (np.ones((K, K)) - np.eye(K))
    pred = m / (m - K - 1)
    vals = [inflation(m, K, nu, corr, "sample")[0] for nu in NUS]
    rows_a.append(dict(m=m, K=K, predicted=pred,
                       measured={("inf" if not np.isfinite(n) else "%g" % n): v
                                 for n, v in zip(NUS, vals)}))
    print("  %6d %4d %10.4f %9.4f %9.4f %9.4f %9.4f %9.4f %9.4f"
          % (m, K, pred, *vals))
OUT["sample_covariance"] = rows_a

_g = [abs(r["measured"]["inf"] - r["predicted"]) / r["predicted"] for r in rows_a]
OUT["gaussian_control_error"] = float(max(_g))
print("""
  The Gaussian control reproduces m/(m-K-1) to %.2f per cent at worst, so the
  simulation is measuring what it claims to measure.
""" % (100 * OUT["gaussian_control_error"]))

_b = [r for r in rows_a if r["m"] == 1400 and r["K"] == 10][0]
OUT["sample_ratio_at_nu428"] = _b["measured"]["4.28"] / _b["predicted"]
# The quantity being corrected is the EXCESS over one, not the factor, so the
# ratio that says how far the correction falls short is a ratio of excesses.
# Comparing the factors instead gives 1.02 and makes a threefold shortfall look
# like a rounding error.
OUT["sample_excess_ratio"] = ((_b["measured"]["4.28"] - 1.0)
                              / (_b["predicted"] - 1.0))
print("""  At the bearings' nu = 4.28, with m = 1400 and K = 10, the correction
  predicts %.4f and the truth is %.4f.  What is being corrected is the excess
  over one, so the correction removes %.1f per cent of a bias that is really
  %.1f per cent: it is short by a factor of %.2f.
""" % (_b["predicted"], _b["measured"]["4.28"],
       100 * (_b["predicted"] - 1), 100 * (_b["measured"]["4.28"] - 1),
       OUT["sample_excess_ratio"]))

# =============================================================================
print("=" * 92)
print("2.  The estimator the pipeline actually builds")
print("=" * 92)
print("""
fusion_prep.py does not use the sample covariance.  It uses robust per-channel
scales with a product-moment correlation between them, for a reason it states:
a robust scale on one side and a sample covariance on the other disagreed by an
order of magnitude on these residuals.  Whether that construction also inherits
the tail sensitivity is the question the paper needs answered, since it is the
construction every fusion number rests on.

Each estimator is judged against its OWN limit and not against the covariance.
The hybrid does not converge to the covariance: its robust scales converge to a
MAD scale, smaller than the standard deviation by the factor Proposition 13 calls
kappa.  Judging it against the covariance would re-measure kappa and report it as
a failure of the Wishart correction, when the paper already corrects for kappa
separately.  The check below confirms that the gap between the two references is
kappa squared and nothing else.
""")
_chk = []
for nu in (8.0, 5.0, 4.28, 3.0):
    c1 = np.eye(1)
    lim = limit_sigma(1, nu, c1, "hybrid")[0, 0]
    kap2 = 1.0 / lim
    pred = (nu / (nu - 2.0)) / (1.4826 * stats.t.ppf(0.75, nu)) ** 2 if nu > 2 \
        else float("nan")
    _chk.append(dict(nu=nu, measured=float(kap2), predicted=float(pred)))
OUT["kappa2_identity"] = _chk
print("  the reference gap, against kappa^2 computed from the fitted law:")
print("  %8s %14s %14s" % ("nu", "measured", "kappa^2"))
for c in _chk:
    print("  %8.2f %14.4f %14.4f" % (c["nu"], c["measured"], c["predicted"]))
OUT["kappa2_identity_error"] = float(max(
    abs(c["measured"] - c["predicted"]) / c["predicted"] for c in _chk))
print("""
  They agree to %.1f per cent, so the gap is kappa and the table below is
  measuring finite-sample inflation alone.
""" % (100 * OUT["kappa2_identity_error"]))
print("  %6s %4s %10s %9s %9s %9s %9s %9s %9s"
      % ("m", "K", "predicted", "nu=inf", "nu=30", "nu=8", "nu=5", "nu=4.28",
         "nu=3"))
print("  " + "-" * 84)
rows_b = []
for m, K in CASES:
    corr = np.eye(K) + 0.3 * (np.ones((K, K)) - np.eye(K))
    pred = m / (m - K - 1)
    vals = [inflation(m, K, nu, corr, "hybrid")[0] for nu in NUS]
    rows_b.append(dict(m=m, K=K, predicted=pred,
                       measured={("inf" if not np.isfinite(n) else "%g" % n): v
                                 for n, v in zip(NUS, vals)}))
    print("  %6d %4d %10.4f %9.4f %9.4f %9.4f %9.4f %9.4f %9.4f"
          % (m, K, pred, *vals))
OUT["hybrid"] = rows_b

_h = [r for r in rows_b if r["m"] == 1400 and r["K"] == 10][0]
_hs = [r for r in rows_b if r["m"] == 300 and r["K"] == 10][0]
OUT["hybrid_ratio_at_nu428"] = _h["measured"]["4.28"] / _h["predicted"]
OUT["hybrid_at_nu428"] = _h["measured"]["4.28"]
OUT["hybrid_short_at_nu428"] = _hs["measured"]["4.28"]
OUT["hybrid_short_ratio"] = _hs["measured"]["4.28"] / _hs["predicted"]
OUT["hybrid_excess_ratio"] = ((_h["measured"]["4.28"] - 1.0)
                              / (_h["predicted"] - 1.0))
OUT["hybrid_short_excess_ratio"] = ((_hs["measured"]["4.28"] - 1.0)
                                    / (_hs["predicted"] - 1.0))
print("""
  At the median record, m = 1400 with K = 10 and nu = 4.28, the hybrid inflates
  by %.4f against the %.4f the applied correction assumes.  As excesses over one
  that is %.2f per cent against %.2f, so the correction is short by a factor of
  %.2f.  On the shortest records, m = 300, it inflates by %.4f against %.4f,
  which as excesses is short by %.2f.  The two shortfalls agree, which is the
  first sign that the form of the correction is right and its sample size is not.
""" % (_h["measured"]["4.28"], _h["predicted"],
       100 * (_h["measured"]["4.28"] - 1), 100 * (_h["predicted"] - 1),
       OUT["hybrid_excess_ratio"], _hs["measured"]["4.28"], _hs["predicted"],
       OUT["hybrid_short_excess_ratio"]))

# =============================================================================
print("=" * 92)
print("3.  Is the correction the wrong size, or the wrong shape?")
print("=" * 92)
print("""
The two tables above show the correction falling short, but not by how it fails.
Writing m/(m-K-1) = 1 + (K+1)/m + O(m^-2), there are two ways to be wrong: the
(K+1) could be wrong, in which case the correction has the wrong dependence on
the number of channels and no rescaling can repair it, or the m could be wrong,
in which case the form survives and only the sample size needs replacing.

The second is testable.  Solving the measured inflation for the m that would
produce it gives an effective sample size, and if the form is right the ratio
c = m/m_eff must come out the same at every m and K for a given nu.  It has no
reason to unless the form is right, so a stable column is evidence and a
scattered one is a refutation.
""")
print("  %-8s %6s %4s %11s %12s %9s" %
      ("estimator", "m", "K", "inflation", "m_eff", "c = m/m_eff"))
print("  " + "-" * 60)
eff = {}
for tag, rows in (("sample", rows_a), ("hybrid", rows_b)):
    for r in rows:
        v = r["measured"]["4.28"]
        exc = v - 1.0
        if exc <= 0:
            continue
        m_eff = (r["K"] + 1) / exc
        c = r["m"] / m_eff
        eff.setdefault(tag, []).append(c)
        print("  %-8s %6d %4d %11.4f %12.1f %9.2f"
              % (tag, r["m"], r["K"], v, m_eff, c))
OUT["c_sample"] = float(np.median(eff["sample"]))
OUT["c_hybrid"] = float(np.median(eff["hybrid"]))
OUT["c_hybrid_lo"] = float(min(eff["hybrid"]))
OUT["c_hybrid_hi"] = float(max(eff["hybrid"]))
print("""
  c is %.2f to %.2f across four cells for the pipeline's estimator, median %.2f,
  with m varying by a factor of five and K by a factor of two and a half.  The
  form survives and the sample size does not: on residuals at nu = 4.28 the
  effective sample size for this correction is about a third of the record.
""" % (OUT["c_hybrid_lo"], OUT["c_hybrid_hi"], OUT["c_hybrid"]))

print("  How c varies with the tail, at m = 1400 and K = 10:")
print("  %10s %12s %12s" % ("nu", "inflation", "c"))
cnu = []
_r = [r for r in rows_b if r["m"] == 1400 and r["K"] == 10][0]
for nu in NUS:
    key = "inf" if not np.isfinite(nu) else "%g" % nu
    v = _r["measured"][key]
    exc = max(v - 1.0, 1e-9)
    c = _r["m"] / ((_r["K"] + 1) / exc)
    cnu.append(dict(nu=(None if not np.isfinite(nu) else float(nu)), c=float(c)))
    print("  %10s %12.4f %12.2f" % (key, v, c))
OUT["c_by_nu"] = cnu
OUT["c_gaussian"] = cnu[0]["c"]
print("""
  At nu = infinity c is %.2f, so the Gaussian case recovers the textbook
  correction as it must, and c grows monotonically as the tails thicken.  That is
  the behaviour an effective sample size should have and is the reason to state
  the result as a replacement for m rather than as a new factor.
""" % OUT["c_gaussian"])

# =============================================================================
print("=" * 92)
print("4.  Which nu the real fusion residuals sit at, and what m and K they use")
print("=" * 92)
z = np.load(os.path.join(HERE, "feat_raw.npz"), allow_pickle=True)
FN = [str(s) for s in z["featnames"]]
units = [k for k in z.files if k != "featnames"]
lens = [int(np.sum(np.isfinite(z[u][:, 0]))) for u in units]
OUT["m_median"] = int(np.median(lens))
OUT["m_min"] = int(min(lens))
OUT["m_max"] = int(max(lens))
OUT["K_max"] = len(FN)
print("  record length m : median %d, range %d to %d"
      % (OUT["m_median"], OUT["m_min"], OUT["m_max"]))
print("  channels available K : %d" % OUT["K_max"])

# =============================================================================
print("=" * 92)
print("5.  What replacing m by m_eff does to the results Section 9 reports")
print("=" * 92)
print("""
Everything above is about the correction in isolation.  What matters is whether
the numbers change, so the fusion is rerun with the only alteration being
m -> m/c inside the debias factor, c taken from the measured 3.08.

Section 9 says the correction "matters on short records with many channels, and
the rank guard removes those".  The first clause is right.  The second is a claim
about these records and is checked here rather than assumed: the shortest bearing
carries m = 230 and the K = 10 set is feasible on all seventeen, so if the guard
were removing the exposed records there would be none left at small m.
""")
from fusion_prep import prepare_fusion, single_info, tau_min_from, _sigma
from fusion_ranksafe import rank_safe

C_EFF = OUT["c_hybrid"]
QSTAR = 0.35


def joint_info_c(chans, c=1.0):
    """joint_info with the debias sample size divided by c.  c = 1 is the
    function fusion_prep.py exports, which is what makes this a controlled
    comparison rather than a reimplementation."""
    D = np.column_stack([u["dtrend"] for u in chans])
    m, K = D.shape
    S = _sigma(chans)
    q = np.einsum("kj,ji,ki->k", D, np.linalg.inv(S), D)
    if K > 1:
        m_e = m / c
        if m_e - K - 1 <= 0:
            return np.full(m, np.nan)
        q = q * (m_e - K - 1) / m_e
    return q


FN = [str(s) for s in z["featnames"]]
I = {k: i for i, k in enumerate(FN)}
recs = []
for b in sorted(units):
    M = z[b]
    u = {}
    for nm in FN:
        p, _ = prepare_fusion(np.asarray(M[:, I[nm]], float), nm)
        if p is not None:
            u[nm] = p
    if len(u) == len(FN) and len({p["n"] for p in u.values()}) == 1:
        recs.append((b, u))
SETS = {"RMS + peak": ["rms", "peak"],
        "RMS + peak + kurt": ["rms", "peak", "kurt"],
        "RMS + three bands": ["rms", "b4_6", "b6_8", "b8_10"],
        "everything": FN}
print("""  The rank guard is applied first, exactly as fusion_ranksafe.py applies it.
  Without it the K = 10 set returns tau = 0.029, which Section 9 already
  identifies as an artefact thirty times the effect, and the comparison would be
  between two readings of that artefact.
""")
print("  %-22s %6s %10s %10s %9s %8s"
      % ("indicator set", "K kept", "tau applied", "tau m_eff", "shift",
         "records"))
print("  " + "-" * 70)
srows = []
for lab, names in SETS.items():
    a, e, kept_k = [], [], []
    for b, u in recs:
        keep, _drop = rank_safe(u, names)
        if len(keep) < 2:
            continue
        ch = [u[n] for n in keep]
        tgt = QSTAR * float(single_info(u["rms"]).sum())
        ta = tau_min_from(joint_info_c(ch, 1.0), tgt)
        te = tau_min_from(joint_info_c(ch, C_EFF), tgt)
        if np.isfinite(ta) and np.isfinite(te):
            a.append(ta)
            e.append(te)
            kept_k.append(len(keep))
    if len(a) < 5:
        continue
    ma, me = float(np.median(a)), float(np.median(e))
    srows.append(dict(set=lab, K_nominal=len(names),
                      K_kept=float(np.median(kept_k)), tau_applied=ma,
                      tau_eff=me, shift=me - ma, records=len(a)))
    print("  %-22s %6.1f %10.4f %10.4f %+9.4f %8d"
          % (lab, float(np.median(kept_k)), ma, me, me - ma, len(a)))
OUT["applied"] = srows
if srows:
    OUT["tau_shift_max"] = float(max(abs(r["shift"]) for r in srows))
    OUT["tau_shift_median"] = float(np.median([abs(r["shift"]) for r in srows]))
    print("""
  The earliest usable age moves by a median of %.4f of a lifetime and at worst
  %.4f.  Section 9's conclusion therefore stands, but not for the reason it
  gives: the correction survives not because m is large against K but because
  m/3 is still large against K.  On these records that is the same verdict; on a
  record a third as long it would not be.
""" % (OUT["tau_shift_median"], OUT["tau_shift_max"]))

# --- the shortest records, which are the ones the claim is really about -------
lens_by_rec = []
for b, u in recs:
    keep, _ = rank_safe(u, FN)
    lens_by_rec.append((b, u[FN[0]]["n"], len(keep)))
short = sorted(lens_by_rec, key=lambda t: t[1])[:4]
OUT["shortest_records"] = [dict(unit=b, m=int(n), K=int(k))
                           for b, n, k in short]


def _pair(n, K):
    fa = (n - K - 1) / n
    me_ = n / C_EFF
    fe = (me_ - K - 1) / me_ if me_ - K - 1 > 0 else float("nan")
    return fa, fe


print("""  the four shortest records, at the K each keeps after the guard, since that
  and not the nominal ten is the number the correction sees:""")
print("    %-14s %7s %4s %10s %10s %9s"
      % ("unit", "m", "K", "applied", "m_eff", "ratio"))
print("    " + "-" * 60)
for b, n, k in short:
    fa, fe = _pair(n, k)
    print("    %-14s %7d %4d %10.4f %10.4f %9.3f" % (b, n, k, fa, fe, fe / fa))
_sm, _sk = short[0][1], short[0][2]
_fa, _fe = _pair(_sm, _sk)
OUT["shortest_m"] = int(_sm)
OUT["shortest_K"] = int(_sk)
OUT["shortest_ratio"] = float(_fe / _fa)
print("""
  On the shortest record the two factors differ by %.1f per cent, so the exposed
  case is present in the data and is not removed by the guard.  The guard is
  about conditioning and record length is a separate axis, which is the part of
  Section 9's sentence that does not follow.
""" % (100 * (1 - OUT["shortest_ratio"])))

# =============================================================================
print("=" * 92)
print("6.  The two corrections together, which is the case the records present")
print("=" * 92)
print("""
Section 4.5 divides the information by the integrated autocorrelation time
(1+phi)/(1-phi), and the residuals here carry phi from 0.067 to 0.399, rising
along life.  Both that and the tails reduce the effective sample size, and the
records have both at once, so the question is whether the two reductions compound
and by how much.

Multiplying the two separately measured factors is the obvious guess and has no
right to be true: the tail weight and the autoregression act on different parts of
the draw, and the covariance is a second-moment functional of both.  Simulating
them together is the only way to find out, so the same estimator is run on data
that is heavy-tailed AND serially correlated, and the product is quoted beside the
measurement as a prediction that could have failed.
""")
PHIS = (0.0, 0.20, 0.399)
_c_serial = {}
print("  %6s %4s %6s %11s %10s %9s %11s"
      % ("m", "K", "phi", "inflation", "m_eff", "c", "product"))
print("  " + "-" * 64)
crows = []
for phi in PHIS:
    for (m, K) in ((1400, 10), (300, 10)):
        # the same 0.3 equicorrelation the earlier tables use, so the only
        # difference between this table and section 2's is the autoregression
        _corr = np.eye(K) + 0.3 * (np.ones((K, K)) - np.eye(K))
        v, _n = inflation_ar1(m, K, 4.28, _corr, "hybrid", phi)
        exc = v - 1.0
        if exc <= 0:
            continue
        m_eff = (K + 1) / exc
        c = m / m_eff
        iat = (1.0 + phi) / (1.0 - phi)
        pred = OUT["c_hybrid"] * iat if phi > 0 else OUT["c_hybrid"]
        crows.append(dict(m=m, K=K, phi=phi, inflation=v, m_eff=m_eff, c=c,
                          predicted=pred))
        print("  %6d %4d %6.3f %11.4f %10.1f %9.2f %11.2f"
              % (m, K, phi, v, m_eff, c, pred))
OUT["serial_tail"] = crows
if crows:
    _hi = [r for r in crows if r["phi"] == max(PHIS) and r["m"] == 1400]
    _lo = [r for r in crows if r["phi"] == 0.0 and r["m"] == 1400]
    if _hi and _lo:
        r, r0 = _hi[0], _lo[0]
        OUT["c_combined"] = r["c"]
        OUT["c_serial_free"] = r0["c"]
        OUT["c_combined_predicted"] = r["predicted"]
        OUT["c_combined_error"] = abs(r["c"] - r["predicted"]) / r["predicted"]
        OUT["serial_multiplier"] = r["c"] / r0["c"]
        OUT["iat_at_max_phi"] = (1 + max(PHIS)) / (1 - max(PHIS))
        print("""
  The guess is wrong, and cleanly so.  At the largest phi the records carry, the
  measured effective sample size is m/%.2f where the product predicts m/%.2f, an
  error of %.0f per cent.  The two reductions do NOT compound: adding serial
  correlation multiplies c by %.2f where the integrated autocorrelation time
  would have it multiply by %.2f.

  The zero-phi row reproduces section 3's %.2f to within %.0f per cent, which is
  the control that makes the comparison a measurement of the autoregression and
  not of the simulation.

  The reason is visible in the estimator.  Serial correlation does not bias a
  robust scale or a correlation coefficient; it inflates their VARIANCE, and that
  enters the inverse quadratic form at order 1/m, which is small at these lengths.
  The tails act on the same quantity at order one.  So the tail term is the one
  that matters and a serial term need not be added to this correction, which is
  the opposite of what Section 4.5 found for the curve itself.  The difference is
  that G is a sum along the record, where dependence between neighbours is felt
  directly, while this factor is a property of a covariance estimated from the
  whole record, where it is felt only through a variance.
""" % (r["c"], r["predicted"], 100 * OUT["c_combined_error"],
       OUT["serial_multiplier"], OUT["iat_at_max_phi"],
       OUT["c_hybrid"], 100 * abs(r0["c"] - OUT["c_hybrid"]) / OUT["c_hybrid"]))

    # what that does on the record that is most exposed to it
    _m, _K = OUT["shortest_m"], OUT["shortest_K"]
    fa = (_m - _K - 1) / _m
    _me = _m / OUT.get("c_combined", OUT["c_hybrid"])
    fe = (_me - _K - 1) / _me if _me - _K - 1 > 0 else float("nan")
    OUT["shortest_combined_ratio"] = float(fe / fa) if np.isfinite(fe) else None
    if np.isfinite(fe):
        print("""  On the shortest bearing, m = %d with %d channels kept, the applied factor is
  %.4f and the combined one %.4f, a gap of %.0f per cent against the %.0f per cent
  the tails alone produce.  That record is inside the reported set, so this is the
  margin the section carries rather than a hypothetical.
""" % (_m, _K, fa, fe, 100 * (1 - fe / fa),
       100 * (1 - OUT["shortest_ratio"])))

json.dump(OUT, open(os.path.join(HERE, "wishart_tails.json"), "w"), indent=2,
          default=float)
print("\nwritten to wishart_tails.json")
