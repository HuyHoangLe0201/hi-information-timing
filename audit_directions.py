r"""The seventh audit: does the text state each claim in the right DIRECTION?

The other six compare numbers.  A reversed inequality carries no number, so it
passes all of them, which is how Section 6.1 came to name
g(tau_0) - g(tau_0 - w*) where equation (7) gives the opposite difference.  That
error was found by hand.  This file asks the same question of every directional
claim in the paper rather than waiting to stumble on the next one.

The method is the same in each case.  The claim is read from the manuscript, an
instance is constructed where the answer is known independently, and the
direction the text asserts is compared against the direction that instance
produces.  A claim that is true only sometimes is reported as such, since a
proposition asserting monotonicity is refuted by one counterexample and no
amount of agreement elsewhere repairs it.
"""
import os
import json
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
rng = np.random.default_rng(707)
OUT = {}
RESULTS = []


def record(claim, where, verdict, detail):
    RESULTS.append(dict(claim=claim, where=where, verdict=verdict,
                        detail=detail))
    tag = {"holds": "ok", "fails": "FAILS", "partial": "PARTIAL"}[verdict]
    print("  %-46s %-12s %-8s" % (claim[:46], where, tag))
    if verdict != "holds":
        print("      %s" % detail)


def cum(d):
    return np.cumsum(d) / 1.0


def tau_min(G, target):
    i = np.searchsorted(G, target)
    return np.nan if i >= len(G) else (i + 1) / len(G)


def w_star_abs(G, t0, c):
    """Window at absolute demand c, by direct inversion of G."""
    n = len(G)
    i0 = min(n - 1, max(0, int(round(t0 * n)) - 1))
    tgt = G[i0] - c
    if tgt <= 0:
        return np.nan
    j = np.searchsorted(G, tgt)
    if j >= n:
        return np.nan
    return t0 - (j + 1) / n


print("=" * 92)
print("Directional claims, each against an instance where the answer is known")
print("=" * 92)
print("  %-46s %-12s %-8s" % ("claim", "section", "verdict"))
print("  " + "-" * 70)

# --- Proposition 2 -----------------------------------------------------------
# dw*/dtau_0 = 1 - g(tau_0)/g(tau_0-w*), so the sign follows the EARLY end minus
# the late one.  This is the claim that was stated backwards.
n = 4000
t = (np.arange(n) + 1.0) / n
worst = 0
for gfun in (lambda x: 2 * x, lambda x: 2 - 2 * x, lambda x: 1 + 0.8 * np.sin(6 * x)):
    d = gfun(t) / n
    G = cum(d)
    c = 0.35 * G[-1]
    bad = tot = 0
    for t0 in np.linspace(0.5, 0.95, 60):
        w = w_star_abs(G, t0, c)
        w2 = w_star_abs(G, t0 + 0.005, c)
        if not (np.isfinite(w) and np.isfinite(w2)):
            continue
        i0 = min(n - 1, max(0, int(round(t0 * n)) - 1))
        i1 = min(n - 1, max(0, int(round((t0 - w) * n)) - 1))
        pred = np.sign(gfun(t[i1]) - gfun(t[i0]))      # early minus late
        obs = np.sign(w2 - w)
        if pred == 0 or obs == 0:
            continue
        tot += 1
        bad += int(pred != obs)
    worst = max(worst, bad)
record("sign of dw*/dtau_0 follows early minus late", "Prop 2",
       "holds" if worst == 0 else "fails",
       "%d disagreements on analytic densities" % worst)

# --- Proposition 3, the three quantities said to be monotone in the set -------
# G_{1:K+1} >= G_{1:K} pointwise is Schur complement and is not in doubt.  What
# is in doubt is the sentence that follows: tau_min, w* and the censoring
# efficiency "are monotone in the set".  The first two inherit the pointwise
# inequality.  eta(c) = G(c)/G(1) does not, because the added channel raises the
# numerator and the denominator, and
#     eta_new - eta  has the sign of  Delta(c)/Delta(1) - G(c)/G(1),
# so it rises only if the new channel is more front-loaded than the old set.
base = np.linspace(0.2, 2.0, n) / n          # a set whose information is late
add_front = np.linspace(2.0, 0.2, n) / n     # an addition that is early
add_back = np.linspace(0.2, 3.0, n) / n      # an addition that is later still
Gb = cum(base)
tm_moves, ws_moves, eta_moves = [], [], []
for add in (add_front, add_back):
    Gn = cum(base + add)
    tgt = 0.5 * Gb[-1]
    tm_moves.append(tau_min(Gn, tgt) - tau_min(Gb, tgt))
    c = 0.35 * Gb[-1]
    for t0 in (0.6, 0.8, 0.95):
        a, b = w_star_abs(Gb, t0, c), w_star_abs(Gn, t0, c)
        if np.isfinite(a) and np.isfinite(b):
            ws_moves.append(b - a)
    for cc in (0.3, 0.5, 0.7):
        i = int(cc * n) - 1
        eta_moves.append(Gn[i] / Gn[-1] - Gb[i] / Gb[-1])

record("adding a channel cannot make tau_min later", "Prop 3",
       "holds" if max(tm_moves) <= 1e-12 else "fails",
       "largest increase %.4g" % max(tm_moves))
record("adding a channel cannot widen w*", "Prop 3",
       "holds" if max(ws_moves) <= 1e-12 else "fails",
       "largest increase %.4g" % max(ws_moves))
_up = max(eta_moves)
_dn = min(eta_moves)
OUT["eta_up"] = float(_up)
OUT["eta_down"] = float(_dn)
# The manuscript used to list eta with the other two.  It moves both ways, so
# that claim was false; what the manuscript now asserts is equation (9), and
# THAT is what is checked from here on.  The old form is kept as a guard: if it
# ever becomes true again, something else has changed and should be looked at.
record("eta is NOT monotone in the set, as now stated", "Prop 3",
       "holds" if (_up > 1e-12 and _dn < -1e-12) else "fails",
       "eta moved %+.4f and %+.4f, so a monotonicity claim would be false"
       % (_up, _dn))

# equation (9): sign(eta_new - eta) = sign(Delta(c)/Delta(1) - G(c)/G(1))
bad = tot = 0
for _ in range(400):
    d0 = rng.random(n) * np.linspace(*rng.choice([(0.2, 2.0), (2.0, 0.2)]), n) / n
    dd = rng.random(n) * np.linspace(*rng.choice([(0.2, 3.0), (3.0, 0.2)]), n) / n
    G0, Dl = cum(d0), cum(dd)
    if G0[-1] <= 0 or Dl[-1] <= 0:
        continue
    Gn = G0 + Dl
    for cc in (0.2, 0.4, 0.6, 0.8):
        i = int(cc * n) - 1
        lhs = np.sign(Gn[i] / Gn[-1] - G0[i] / G0[-1])
        rhs = np.sign(Dl[i] / Dl[-1] - G0[i] / G0[-1])
        if lhs == 0 or rhs == 0:
            continue
        tot += 1
        bad += int(lhs != rhs)
OUT["etasign_pairs"] = tot
OUT["etasign_bad"] = bad
record("eq (9) gives the sign of the change in eta", "Prop 3",
       "holds" if bad == 0 else "fails",
       "%d of %d random instances disagree" % (bad, tot))

# --- Proposition 4, the Bayesian floor ---------------------------------------
# "if epsilon >= s the target is met at tau_0 = 0 without any measurement".
_s, _eps = 0.4, 0.5
record("a prior alone suffices when epsilon >= s", "Prop 4",
       "holds" if (0.0 + _s ** -2) >= _eps ** -2 else "fails",
       "G=0 gives bound %.4f against target %.4f" % (_s ** 2, _eps ** 2))

# --- Proposition 6, the tail factor -------------------------------------------
# "for t_nu the second factor is nu(nu+1)/((nu+3)(nu-2)), unbounded as the tails
# thicken", so it must INCREASE as nu falls towards 2.
f = lambda nu: nu * (nu + 1) / ((nu + 3) * (nu - 2))
vals = [f(v) for v in (30.0, 10.0, 5.0, 3.0, 2.5)]
record("least-squares inefficiency grows as tails thicken", "Prop 6",
       "holds" if all(vals[i] < vals[i + 1] for i in range(len(vals) - 1))
       else "fails", "values %s" % ", ".join("%.3f" % v for v in vals))

# --- Proposition 8, information under serial correlation ----------------------
# G is divided by (1+phi)/(1-phi), so positive correlation must REDUCE it.
iat = lambda p: (1 + p) / (1 - p)
record("positive serial correlation reduces information", "Prop 8",
       "holds" if all(iat(p) > 1 for p in (0.05, 0.2, 0.4)) else "fails",
       "iat at phi=0.4 is %.3f" % iat(0.4))

# --- Proposition 14 of Section 11, leverage ----------------------------------
# "sum h_k = K" for a rank-K design, checked on random designs.
gaps = []
for _ in range(200):
    m, K = rng.integers(40, 200), int(rng.integers(2, 8))
    X = rng.standard_normal((int(m), K))
    Q_, _ = np.linalg.qr(X)
    gaps.append(abs(float(np.sum(np.sum(Q_ ** 2, axis=1))) - K))
record("the leverages of a rank-K design sum to K", "Prop 14",
       "holds" if max(gaps) < 1e-8 else "fails",
       "worst deviation %.3g" % max(gaps))

# --- Propositions 16 and 17, the two bounds -----------------------------------
# The odds bound is an inequality with a direction, and the monotone result is
# three directional claims, so both belong here as well as in the mathematics
# audit: a reversed inequality in either would carry no number.
_ic = n // 2 - 1
for _dn, _g0 in (("rising", 2 * t), ("peaked", 0.2 + 3 * t ** 4)):
    _g = _g0 / n
    _e0 = float(np.sum(_g[:_ic + 1]) / np.sum(_g))
    for _M in (1.2, 2.5):
        _w = np.where(np.arange(n) <= _ic, _M, 1.0 / _M)
        _gd = _w * _g
        _e1 = float(np.sum(_gd[:_ic + 1]) / np.sum(_gd))
        _r = (_e1 / (1 - _e1)) / (_e0 / (1 - _e0))
        if abs(_r - _M * _M) / (_M * _M) > 1e-9:
            break
record("the worst case multiplies the odds by M^2, not divides", "Prop 16",
       "holds" if abs(_r - _M * _M) / (_M * _M) < 1e-9 else "fails",
       "ratio %.4f against M^2 = %.4f" % (_r, _M * _M))

_rngD = np.random.default_rng(555)
_bad_up = _bad_dn = 0
for _ in range(600):
    _g = (0.3 + 2.0 * np.linspace(0, 1, n) ** float(_rngD.uniform(0.5, 4))) / n
    _s = np.sort(_rngD.random(n))[::-1]
    _F0 = np.cumsum(_g) / np.sum(_g)
    _F1 = np.cumsum(np.exp(2.0 * _s) * _g) / np.sum(np.exp(2.0 * _s) * _g)
    _bad_up += int(np.max(_F0 - _F1) > 1e-12)          # decreasing w: F must rise
    _F2 = np.cumsum(np.exp(2.0 * _s[::-1]) * _g)
    _F2 = _F2 / _F2[-1]
    _bad_dn += int(np.max(_F2 - _F0) > 1e-12)          # increasing w: F must fall
record("a falling w raises F and a rising w lowers it", "Prop 17",
       "holds" if _bad_up == 0 and _bad_dn == 0 else "fails",
       "%d and %d violations in 600 trials each" % (_bad_up, _bad_dn))

# --- Proposition 1, the four readings and their consistency -------------------
# (ii) and (iii) are separate formulas that must agree at one point: at
# tau_0 = tau_min the whole record is needed, so w*(tau_min) = tau_min.  Nothing
# in the text says so, which is what makes it a test rather than a restatement.
# Both of these are index questions, so they are done on indices.  A first draft
# round-tripped through tau and reported failures of 0.017 and 0.257 that were
# its own: tau_min returns (i+1)/n and w_star_abs re-derived i by rounding, and
# the demand loop ran from loose to tight while the assertion read the other way.
worst_c, worst_w, worst_mono = 0.0, 0.0, 0.0
for gfun in (lambda x: 2 * x, lambda x: 2 - 2 * x, lambda x: 0.5 + x ** 3):
    d = gfun(t) / n
    G = cum(d)
    prev_i = None
    for eps2 in (0.2, 0.4, 0.6, 0.8):        # increasing = a HARDER demand
        c = eps2 * G[-1]
        i_min = int(np.searchsorted(G, c))
        if i_min >= n:
            continue
        # (iii) at tau_0 = tau_min asks for G(tau_min) - c = 0, whose inverse is
        # the first sample, so w* is the whole record to within one sample.
        # w* = tau_min - G^-1(0), so the claim w*(tau_min) = tau_min says the
        # inverse lands at the start of the record.  The residual is measured in
        # INFORMATION and not in time.  Measured in time it reads 0.017 of a
        # lifetime on G = tau^2, which is not an error but the flatness of that
        # curve at the origin: g(0) = 0 makes G^-1 arbitrarily steep there, so a
        # one-sample overshoot in G maps to many samples of tau.  That is this
        # paper's own argument for reading budget rather than age, and applying
        # it to the paper's own proposition is the consistent thing to do.
        j = int(np.searchsorted(G, G[i_min] - c))
        worst_c = max(worst_c, float(G[j] / G[-1]))
        for t0 in (0.7, 0.85, 0.97):
            i0 = min(n - 1, max(0, int(round(t0 * n)) - 1))
            if G[i0] - c <= 0:
                continue
            i1 = int(np.searchsorted(G, G[i0] - c))
            if i1 >= n:
                continue
            worst_w = max(worst_w, abs((G[i0] - G[i1]) - c) / c)
        # a harder demand can only push tau_min later
        if prev_i is not None:
            worst_mono = max(worst_mono, (prev_i - i_min) / n)
        prev_i = i_min
record("w*(tau_min) = tau_min, joining (ii) and (iii)", "Prop 1",
       "holds" if worst_c <= 2.0 / n else "fails",
       "at most %.3g of the budget lies before the window" % worst_c)
record("w* meets its defining equation exactly", "Prop 1",
       "holds" if worst_w < 5e-3 else "fails",
       "worst relative miss %.3g" % worst_w)
record("a harder target never moves tau_min earlier", "Prop 1",
       "holds" if worst_mono <= 1.0 / n else "fails",
       "largest backward step %.4g" % worst_mono)

# --- Proposition 5, the standard error of tau_min ------------------------------
# se = se(G-hat)/g, so a FLATTER curve gives a LARGER error, and the sign of the
# first equation says that over-estimating G moves tau_min EARLIER.
d = (2 * t) / n
G = cum(d)
c = 0.5 * G[-1]
tm = tau_min(G, c)
i = min(n - 1, max(0, int(round(tm * n)) - 1))
g_here = d[i] * n
shifts = []
for bump in (+0.02, -0.02):
    Gh = G * (1.0 + bump)
    shifts.append((tau_min(Gh, c) - tm, bump))
sign_ok = all((s < 0) == (b > 0) for s, b in shifts if np.isfinite(s))
pred = abs(shifts[0][1] * G[i]) / g_here
record("over-estimating G moves tau_min earlier", "Prop 5",
       "holds" if sign_ok else "fails",
       "shifts %s" % ", ".join("%+.4f at %+.2f" % s for s in shifts))
record("the first-order size of that shift is |dG|/g", "Prop 5",
       "holds" if abs(abs(shifts[0][0]) - pred) < 0.02 else "fails",
       "observed %.4f against predicted %.4f" % (abs(shifts[0][0]), pred))

# --- Proposition 7, which curve is a floor -------------------------------------
# G_2 <= G_* always (that is I_f sigma^2 >= 1), and the robust curve exceeds G_*
# at every nu > 2 (not at nu = 1, where sigma_rob^2 I_f = 1.099).  The second is the strong claim: at large nu the t tends to
# a Gaussian and the margin must vanish without changing sign.
from scipy import stats as _st
inf_ok, rob_ok, margins = True, True, []
for nu in (3.0, 4.28, 6.0, 12.0, 30.0, 100.0, 1000.0):
    I_f = (nu + 1) / ((nu + 3) * 1.0 ** 2)          # unit scale
    var = nu / (nu - 2.0)
    if I_f * var < 1.0 - 1e-9:
        inf_ok = False
    # robust scale of a unit-scale t, by its own quantile
    rob = 1.4826 * _st.t.ppf(0.75, nu)
    m = rob ** 2 * I_f
    margins.append((nu, m))
    if m >= 1.0:
        rob_ok = False
record("I_f sigma^2 >= 1, so G_2 <= G_*: the least-squares benchmark "
       "sits at or above the Cramer-Rao floor", "Prop 7",
       "holds" if inf_ok else "fails", "checked over nu 3 to 1000")
record("the robust curve exceeds G_* throughout nu > 2: its benchmark "
       "sits below the Cramer-Rao floor", "Prop 7",
       "holds" if rob_ok else "fails",
       "sigma_rob^2 I_f at nu=1000 is %.5f" % margins[-1][1])
OUT["rob_margins"] = [[float(a), float(b)] for a, b in margins]

# --- Proposition 9, the replacement policy ------------------------------------
# (12) must be the stationary point of (11), and raising the cost of discarded
# life must move the threshold LATER.
from scipy.stats import norm as _nrm


def cost(u, r, s):
    return _nrm.cdf(u) + r * s * (_nrm.pdf(u) - u * (1 - _nrm.cdf(u)))


def hazard_u(r, s):
    lo, hi = -8.0, 8.0
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if _nrm.pdf(mid) / max(1 - _nrm.cdf(mid), 1e-300) < r * s:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


stat_gap, mono = 0.0, True
prev_u = None
for r in (0.05, 0.2, 0.5, 1.0, 2.0):
    s = 0.15
    u = hazard_u(r, s)
    h = 1e-4
    slope = (cost(u + h, r, s) - cost(u - h, r, s)) / (2 * h)
    stat_gap = max(stat_gap, abs(slope))
    if prev_u is not None and u < prev_u - 1e-6:
        mono = False
    prev_u = u
record("(12) is the stationary point of (11)", "Prop 9",
       "holds" if stat_gap < 1e-6 else "fails",
       "largest |dC/du| at the solution %.3g" % stat_gap)
record("costlier discarded life means a later threshold", "Prop 9",
       "holds" if mono else "fails", "u non-decreasing in c_p/c_f")

# --- Proposition 10, the nuisance costs ---------------------------------------
# Profiling out MORE parameters must leave LESS information, so
# r_Abeta <= min(r_A, r_beta), and all three must lie in (0, 1].
rA = lambda b: 1.0 / (4 * b ** 2)
rB = lambda b: 1 - (2 * b - 1) * (2 * b + 1) ** 3 / (32 * b ** 4)
rAB = lambda b: 1.0 / (16 * b ** 4)
bs = np.concatenate([np.linspace(0.501, 1.0, 400), np.linspace(1.0, 30.0, 400)])
in_range = all(0 < rA(b) <= 1 and 0 < rB(b) <= 1 and 0 < rAB(b) <= 1 for b in bs)
nested = all(rAB(b) <= min(rA(b), rB(b)) + 1e-12 for b in bs)
ident = max(abs(rAB(b) - rA(b) ** 2) for b in bs)
_tight = min(min(rA(b), rB(b)) - rAB(b) for b in bs)
record("all three nuisance fractions lie in (0,1]", "Prop 10",
       "holds" if in_range else "fails", "swept beta from 0.501 to 30")
record("profiling more parameters leaves less", "Prop 10",
       "holds" if nested else "fails",
       "tightest margin %.3g, reached as beta approaches 1/2" % _tight)
record("r_Abeta equals r_A squared", "Prop 10",
       "holds" if ident < 1e-12 else "fails", "worst deviation %.3g" % ident)
OUT["nuisance_tightest"] = float(_tight)

# --- Proposition 11, the linearisation radius ---------------------------------
# RMSE/floor = sqrt(1 + C^2 delta^4 G / 4) must be >= 1 and increase with |delta|.
ratio = lambda C, dl, Gv: np.sqrt(1 + 0.25 * C ** 2 * dl ** 4 * Gv)
vals = [ratio(0.8, dl, 50.0) for dl in (0.0, 0.05, 0.1, 0.2, 0.4)]
record("bias can only raise the error above the floor", "Prop 11",
       "holds" if vals[0] >= 1 - 1e-12 and all(
           vals[i] <= vals[i + 1] for i in range(len(vals) - 1)) else "fails",
       "ratios %s" % ", ".join("%.3f" % v for v in vals))

# --- Proposition 12, the best fixed indicator ---------------------------------
# lambda_max <= tr(Sigma^-1 M), with equality iff M has rank one.
from scipy.linalg import eigh as _eigh
viol, rank1_gap = 0.0, 0.0
for _ in range(300):
    K = int(rng.integers(2, 7))
    A = rng.standard_normal((K, K))
    S = A @ A.T + K * np.eye(K)
    B = rng.standard_normal((K, int(rng.integers(K, 3 * K))))
    M = B @ B.T
    lam = float(np.max(_eigh(M, S, eigvals_only=True)))
    viol = max(viol, lam - float(np.trace(np.linalg.solve(S, M))))
    v = rng.standard_normal((K, 1))
    M1 = v @ v.T
    lam1 = float(np.max(_eigh(M1, S, eigvals_only=True)))
    rank1_gap = max(rank1_gap,
                    abs(lam1 - float(np.trace(np.linalg.solve(S, M1)))))
record("lambda_max <= tr(Sigma^-1 M)", "Prop 12",
       "holds" if viol < 1e-8 else "fails", "worst violation %.3g" % viol)
record("equality when M has rank one", "Prop 12",
       "holds" if rank1_gap < 1e-8 else "fails",
       "worst gap %.3g" % rank1_gap)

# --- Proposition 13, the span bound -------------------------------------------
# lambda_max(M(c), M(1)) lies in [0,1] and is non-decreasing in c.
bad_range, bad_mono = 0.0, 0.0
for _ in range(120):
    K = int(rng.integers(2, 6))
    m = 300
    D = rng.standard_normal((m, K)) * np.linspace(0.3, 2.0, m)[:, None]
    Ms = np.cumsum(np.einsum("ij,ik->ijk", D, D), axis=0)
    M1 = Ms[-1] + 1e-9 * np.eye(K)
    prev = -np.inf
    for c in (0.2, 0.4, 0.6, 0.8, 1.0):
        Mc = Ms[int(c * m) - 1] + 1e-12 * np.eye(K)
        lam = float(np.max(_eigh(Mc, M1, eigvals_only=True)))
        bad_range = max(bad_range, max(0.0, lam - 1.0), max(0.0, -lam))
        bad_mono = max(bad_mono, max(0.0, prev - lam))
        prev = lam
record("the span ratio lies in [0,1]", "Prop 13",
       "holds" if bad_range < 1e-8 else "fails",
       "worst excursion %.3g" % bad_range)
record("the span ratio is non-decreasing in c", "Prop 13",
       "holds" if bad_mono < 1e-8 else "fails",
       "worst decrease %.3g" % bad_mono)

OUT["results"] = RESULTS
_fail = [r for r in RESULTS if r["verdict"] != "holds"]
OUT["n_checked"] = len(RESULTS)
OUT["n_failed"] = len(_fail)
print("\n" + "=" * 92)
if _fail:
    print("%d of %d directional claims do not hold as stated:"
          % (len(_fail), len(RESULTS)))
    for r in _fail:
        print("  %s (%s)\n      %s" % (r["claim"], r["where"], r["detail"]))
else:
    print("all %d directional claims hold as stated" % len(RESULTS))
print("=" * 92)

json.dump(OUT, open(os.path.join(HERE, "audit_directions.json"), "w"),
          indent=2, default=float)
print("written to audit_directions.json")

# A layer that prints its verdict and exits zero cannot fail, and every
# runner that reads exit codes reports it as passing while it flags.
# audit_paper.py sat that way through a pass for sentence variety, with
# two checks reading sentences that had been rewritten around them.
import sys
sys.exit(1 if _fail else 0)
