r"""Which scale gives a valid floor when the residuals are not Gaussian.

The paper carries two curves and treats the second of them as the truth.  G is
built with a robust scale; Proposition 5 observes that any estimator's variance
follows the second moment, so the robust curve overstates the information by
kappa^2 = (sigma / sigma_rob)^2, and divides that out.

That is exact only if the second-moment curve is itself the truth, which holds
when the noise is Gaussian.  On these records it is not: the bearing residuals
have kurtosis 48.  The quantity that actually enters the Cramer-Rao bound for a
location-type parameter is the LOCATION FISHER INFORMATION of the noise density,

    I_f = integral (f'/f)^2 f ,

which equals 1/sigma^2 only in the Gaussian case.  For every other density the
information inequality I_f sigma^2 >= 1 is strict, so a heavy-tailed record
carries MORE information about the damage clock than its variance suggests, not
less.  There are therefore three curves, not two:

    G_rob  = sum D'^2 / sigma_rob^2      what the paper computes
    G_2nd  = sum D'^2 / sigma^2          what Proposition 5 corrects to
    G_true = sum D'^2 I_f                what the bound actually is

and G_2nd <= G_true always, with equality only for Gaussian noise.

Two consequences follow, and they point in opposite directions.

  (a) G_2nd is a VALID floor for any noise distribution, being below G_true.
      The correction of Proposition 5 is therefore necessary for validity and
      not merely for accuracy, because G_rob can and does exceed G_true.

  (b) The corrected floor is CONSERVATIVE.  An estimator that uses the true
      likelihood rather than least squares can beat it, by the factor
      I_f sigma^2, which for a Student-t with nu degrees of freedom is
      nu(nu+1)/((nu+3)(nu-2)) and grows without bound as the tails thicken.

This script derives both, verifies them against simulation, and then measures
I_f on the paper's own out-of-fold residuals to say how much information the
Gaussian model leaves unclaimed on real records.
"""
import os
import json
import numpy as np
from scipy import stats, integrate, optimize
from scipy.special import gamma as gammafn

from pipeline import robust_scale, oof_trend, local_scale, _win, LOCAL_BW


def load_domain(name):
    """Yield (label, series) per record.  Copied rather than imported: the
    module that defines it runs its whole analysis at import time and would
    rewrite its own result file as a side effect of this one."""
    if name == "bearings":
        pth = os.path.join(HERE, "feat_raw.npz")
        if not os.path.exists(pth):
            return
        z = np.load(pth, allow_pickle=True)
        FN = [str(t) for t in z["featnames"]]
        i = FN.index("rms")
        for u in sorted(k for k in z.files if k != "featnames"):
            yield u, np.asarray(z[u][:, i], float)
    elif name == "battery":
        pth = os.path.join(HERE, "battery_raw.npz")
        if not os.path.exists(pth):
            return
        z = np.load(pth, allow_pickle=True)
        for u in sorted(z.files):
            a = np.asarray(z[u], float)
            yield u, (a if a.ndim == 1 else a[:, 0])
    else:
        pth = os.path.join(HERE, "cmapss_%s.npz" % name)
        if not os.path.exists(pth):
            return
        z = np.load(pth, allow_pickle=True)
        for u in sorted(k for k in z.files
                        if k not in ("featnames", "sensors"))[:120]:
            a = np.asarray(z[u], float)
            yield u, (a if a.ndim == 1 else a[:, 0])


HERE = os.path.dirname(os.path.abspath(__file__))
rng = np.random.default_rng(7)
OUT = {}

# =============================================================================
print("=" * 88)
print("1.  The information inequality, and the two scales, for Student-t noise")
print("=" * 88)
print("""
For t_nu with scale s:   I_f = (nu+1)/((nu+3) s^2),   Var = s^2 nu/(nu-2),
and the MAD-based robust scale is 1.4826 * s * q_nu with q_nu the 0.75 quantile
of the standard t_nu.  Both closed forms are checked against quadrature.
""")
print("  %-5s %10s %10s %10s %10s %10s" %
      ("nu", "I_f s^2", "quad", "I_f*Var", "kappa^2", "I_f*rob^2"))
print("  " + "-" * 62)
rows = []
for nu in (3.0, 4.0, 5.0, 8.0, 15.0, 50.0, 1e4):
    s = 1.0
    f = lambda x: stats.t.pdf(x, nu, scale=s)
    # score = -d/dx log f  ;  for t: (nu+1) x / (nu s^2 + x^2)
    score = lambda x: (nu + 1) * x / (nu * s ** 2 + x ** 2)
    quad = integrate.quad(lambda x: score(x) ** 2 * f(x), -np.inf, np.inf,
                          limit=400)[0]
    closed = (nu + 1) / ((nu + 3) * s ** 2)
    var = s ** 2 * nu / (nu - 2) if nu > 2 else np.inf
    q = stats.t.ppf(0.75, nu, scale=s)
    rob2 = (1.4826 * q) ** 2
    rows.append(dict(nu=nu, I_f=closed, I_var=closed * var,
                     kappa2=var / rob2, I_rob2=closed * rob2))
    print("  %-5g %10.5f %10.5f %10.5f %10.5f %10.5f"
          % (nu, closed, quad, closed * var, var / rob2, closed * rob2))
    assert abs(quad - closed) < 1e-6 * closed, "closed form disagrees with quadrature"
print("""
  Reading the table.  Column 'I_f*Var' is the factor by which the true
  information exceeds what the variance implies; it is 1 only in the Gaussian
  limit.  Column 'I_f*rob^2' is below 1 at every finite nu, which says the
  ROBUST curve claims more information than exists and is not a floor.  The
  second-moment curve, whose product is 1/(I_f*Var) <= 1, always is.
""")
OUT["student_t"] = rows

# =============================================================================
print("=" * 88)
print("2.  Simulation: three floors against what estimators actually achieve")
print("=" * 88)
print("""
A record with a known trend and t-distributed noise.  Three estimators: least
squares, which is the Gaussian maximum likelihood; the true maximum likelihood
under the t; and a Huber M-estimator as the practical middle.  Each is compared
against the three floors above.
""")
n = 1500
tau = np.linspace(1e-3, 1.0, n)
A, beta = 1.0, 2.0
Dp = A * beta * tau ** (beta - 1)
Dfun = lambda d: A * np.maximum(tau + d, 0) ** beta
SD = np.sum(Dp ** 2)
print("  %-5s %9s %9s %9s %9s %9s %9s"
      % ("nu", "floorRob", "floor2nd", "floorTrue", "sd(LS)", "sd(tML)", "sd(Hub)"))
print("  " + "-" * 66)
sim = []
for nu in (3.0, 4.0, 5.0, 30.0):
    s = 0.02
    var = s ** 2 * nu / (nu - 2)
    rob2 = (1.4826 * stats.t.ppf(0.75, nu, scale=s)) ** 2
    I_f = (nu + 1) / ((nu + 3) * s ** 2)
    f_rob, f_2nd, f_true = (rob2 / SD) ** .5, (var / SD) ** .5, (1 / (I_f * SD)) ** .5
    ls, tml, hub = [], [], []
    base = Dfun(0.0)
    for _ in range(3000):
        y = base + stats.t.rvs(nu, scale=s, size=n, random_state=rng)
        r = y - base
        ls.append(float(np.sum(Dp * r) / SD))
        # one Newton step of the t-likelihood from the least-squares point,
        # which is the efficient estimator to first order
        d0 = ls[-1]
        for _ in range(6):        # one step is not enough at nu = 3
            u = y - Dfun(d0)
            w = (nu + 1) / (nu * s ** 2 + u ** 2)
            d0 = d0 + float(np.sum(w * Dp * u) / np.sum(w * Dp ** 2))
        tml.append(d0)
        u = y - Dfun(ls[-1])
        c = 1.345 * s
        wh = np.clip(c / np.maximum(np.abs(u), 1e-12), None, 1.0)
        hub.append(d0 + float(np.sum(wh * Dp * u) / np.sum(wh * Dp ** 2)))
    e = lambda a: float(np.std(a, ddof=1))
    sim.append(dict(nu=nu, floor_rob=f_rob, floor_2nd=f_2nd, floor_true=f_true,
                    ls=e(ls), tml=e(tml), huber=e(hub)))
    print("  %-5g %9.3e %9.3e %9.3e %9.3e %9.3e %9.3e"
          % (nu, f_rob, f_2nd, f_true, e(ls), e(tml), e(hub)))
OUT["simulation"] = sim
print("""
  Least squares lands on the second-moment floor, as Proposition 5 says.  The
  t-likelihood estimator lands on the true floor, which is BELOW it, so the
  paper's corrected floor is not attainable-and-tight but conservative.  The
  robust floor sits below what least squares achieves, which is the failure
  Proposition 5 corrects.
""")
worst = max(abs(r["ls"] / r["floor_2nd"] - 1) for r in sim)
print("  least squares against the second-moment floor: worst deviation %.1f%%"
      % (100 * worst))
worst_t = max(abs(r["tml"] / r["floor_true"] - 1) for r in sim)
print("  t-likelihood against the true floor:           worst deviation %.1f%%"
      % (100 * worst_t))

# =============================================================================
print()
print("=" * 88)
print("3.  How much information does the Gaussian model leave on real records?")
print("=" * 88)
print("""
The residuals are the ones the tail factor is measured on: out-of-fold, then
locally standardised, so their scale is one and only their shape remains.  Two
densities are fitted to each record by maximum likelihood, a Student-t and a
generalised normal, and I_f is taken from each fit within its own family so that
the information inequality holds by construction.  A single family would only
report what that family can express; two families with different tail behaviour
agreeing is the evidence.  A kernel estimate of the score was tried first and
abandoned: it is strongly downward-biased at any usable bandwidth and returned
values below one, which the inequality forbids.
""")


def ged_fit(z):
    """Fit a generalised normal, exp(-|x/alpha|^p), by maximum likelihood.

    A second family is needed because a Student-t fit can only report what the
    t family can express: if the residuals are heavy in a way the t does not
    capture, the fitted nu absorbs the misfit and I_f inherits it.  Agreement
    between two families with different tail behaviour is evidence that the
    number is a property of the data.
    """
    z = np.asarray(z, float)
    z = z[np.isfinite(z)] - np.median(z)

    def nll(th):
        p, la = np.exp(th)
        if not (0.3 < p < 12) or not np.isfinite(la) or la <= 0:
            return 1e9
        return -np.sum(np.log(p / (2 * la)) - np.log(gammafn(1.0 / p))
                       - np.abs(z / la) ** p)

    best = None
    for p0 in (0.8, 1.2, 2.0, 3.0):
        r = optimize.minimize(nll, np.log([p0, np.std(z) + 1e-12]),
                              method="Nelder-Mead",
                              options=dict(xatol=1e-6, fatol=1e-6, maxiter=4000))
        if best is None or r.fun < best.fun:
            best = r
    p, la = np.exp(best.x)
    # I_f = (p/alpha)^2 Gamma(2 - 1/p)/Gamma(1/p);  Var = alpha^2 Gamma(3/p)/Gamma(1/p)
    if p <= 0.5:
        return p, np.nan
    I_f = (p / la) ** 2 * gammafn(2 - 1.0 / p) / gammafn(1.0 / p)
    var = la ** 2 * gammafn(3.0 / p) / gammafn(1.0 / p)
    return float(p), float(I_f * var)


def residuals(x):
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
    r = r[np.isfinite(r)]
    if len(r) < 40:
        return None
    sl = np.clip(local_scale(r, max(5, int(LOCAL_BW * len(r)))), 1e-12, None)
    z = r / sl
    z = z[np.isfinite(z)]
    return z if len(z) >= 60 else None


print("  %-10s %5s %8s %10s %10s %8s %10s"
      % ("domain", "recs", "nu_hat", "t: I_f*var", "I_f*rob^2", "GED p", "G: I_f*var"))
print("  " + "-" * 66)
dom_rows = []
for dom in ("bearings", "battery", "FD004"):
    nus, ivar, irob, gp, givar, heavy = [], [], [], [], [], 0
    per = []   # the decomposition must be formed per record, then summarised
    for lab, ser in load_domain(dom):
        z = residuals(ser)
        if z is None:
            continue
        z = z - np.median(z)
        try:
            nu, loc, sc = stats.t.fit(z, floc=0.0)
        except Exception:
            continue
        nus.append(float(nu))
        if nu <= 2.0:
            # the fitted law has no variance, so I_f * Var is not defined; the
            # record is counted and excluded from that column rather than
            # clipped to a value it does not have
            heavy += 1
        else:
            ivar.append(nu * (nu + 1) / ((nu + 3) * (nu - 2)))
        I_f = (nu + 1) / ((nu + 3) * sc ** 2)
        rob2 = robust_scale(z) ** 2
        irob.append(I_f * rob2)
        if nu > 2.0:
            over = 1.0 / (I_f * rob2)                 # G_rob / G_true
            ineff = nu * (nu + 1) / ((nu + 3) * (nu - 2))   # G_true / G_2nd
            per.append((over * ineff, over, ineff))
        pp, gv = ged_fit(z[:1500] if len(z) > 1500 else z)
        gp.append(pp)
        # the generalised normal has a location Fisher information only for
        # p > 1/2; below that the score is unbounded at the origin and the
        # integral diverges, so such a fit is counted, not averaged in
        if np.isfinite(gv) and pp > 0.5:
            givar.append(gv)
    if not nus:
        continue
    m = lambda a: float(np.median(a)) if len(a) else float("nan")
    q = lambda a: (float(np.percentile(a, 25)), float(np.percentile(a, 75)))         if len(a) >= 4 else (float("nan"), float("nan"))
    dom_rows.append(dict(domain=dom, records=len(nus), nu=m(nus),
                         I_var_t=m(ivar), I_rob2=m(irob), ged_p=m(gp),
                         I_var_ged=m(givar), no_variance=heavy,
                         kappa2=m([t[0] for t in per]),
                         overstatement=m([t[1] for t in per]),
                         inefficiency=m([t[2] for t in per]),
                         n_decomposed=len(per),
                         n_ged_valid=len(givar), n_var_valid=len(ivar),
                         I_var_t_q=q(ivar), ged_p_q=q(gp)))
    print("  %-10s %5d %8.2f %10.3f %10.3f %8.2f %10.3f"
          % (dom, len(nus), m(nus), m(ivar), m(irob), m(gp), m(givar)))
    if len(ivar) >= 4:
        lo, hi = q(ivar)
        print("             t-family quartiles %.3f to %.3f over %d records"
              % (lo, hi, len(ivar)))
    if heavy:
        print("             %d of %d records fit a t with nu <= 2, for which the "
              "variance does not exist" % (heavy, len(nus)))
    if len(givar) < len(gp):
        print("             %d of %d generalised-normal fits gave p <= 1/2, for "
              "which I_f is infinite" % (len(gp) - len(givar), len(gp)))
OUT["domains"] = dom_rows
print("""
  'I_f*var' is the factor by which the true information exceeds what the second
  moment implies, so it is what the paper's corrected floor gives away.  Both
  columns are computed within their fitted family, so each obeys the
  information inequality by construction and the comparison between them is
  the test: two families with different tails agreeing is evidence, one family
  alone is not.  'I_f*rob^2' below one says the uncorrected robust curve claims
  more information than exists.
""")

# =============================================================================
print()
print("=" * 88)
print("4.  What kappa^2 is actually made of")
print("=" * 88)
print("""
The three curves give an exact factorisation of the tail factor:

    kappa^2 = sigma^2 / sigma_rob^2
            = [ 1 / (sigma_rob^2 I_f) ] x [ I_f sigma^2 ]
            = ( G_rob / G_true )        x ( G_true / G_2nd ) .

The first factor is the amount by which the robust curve really does claim
information that does not exist.  The second is not a property of the curve at
all: it is the inefficiency of least squares under non-Gaussian noise.
Proposition 5 attributes the whole of kappa^2 to the first.  Measuring I_f
splits it.
""")
print("""  Each factor is formed on the record and then summarised, since the median
  of a product is not the product of the medians.
""")
print("  %-10s %9s %11s %11s %9s" %
      ("domain", "kappa^2", "G_rob/Gtrue", "Gtrue/G2nd", "records"))
print("  " + "-" * 54)
split = []
for r in dom_rows:
    if not r["n_decomposed"]:
        continue
    split.append(dict(domain=r["domain"], kappa2=r["kappa2"],
                      overstatement=r["overstatement"],
                      inefficiency=r["inefficiency"],
                      records=r["n_decomposed"]))
    print("  %-10s %9.3f %11.3f %11.3f %9d"
          % (r["domain"], r["kappa2"], r["overstatement"], r["inefficiency"],
             r["n_decomposed"]))
OUT["decomposition"] = split
_b = next((d for d in split if d["domain"] == "bearings"), None)
if _b:
    frac = np.log(_b["inefficiency"]) / np.log(_b["kappa2"])
    OUT["bearings_inefficiency_share"] = float(frac)
    print("""
  On the bearings %.0f per cent of kappa^2, measured on a log scale, is the
  inefficiency of least squares rather than an error in the curve.  The true
  bound lies between the two curves the paper already carries, and much nearer
  the robust one: the correction of Proposition 5 overshoots it by a factor
  %.2f.
""" % (100 * frac, _b["inefficiency"]))

if _b:
    emp = 1.3508785297893466 ** 2      # kappa measured in Section 4.2
    print("""  A caveat on the split itself.  The kappa^2 implied by the fitted law is
  %.2f against %.2f measured directly from the sample, because the sample
  second moment is itself an unstable estimator under tails this heavy: it is
  the quantity the fit says should not be trusted.  The direction of the split
  does not depend on which is used, since the overstatement factor stays near
  one either way, but the exact share does.
""" % (_b["kappa2"], emp))
    OUT["kappa2_empirical"] = emp

json.dump(OUT, open(os.path.join(HERE, "fisher_nongaussian.json"), "w"),
          indent=2, default=float)
print("written to fisher_nongaussian.json")
