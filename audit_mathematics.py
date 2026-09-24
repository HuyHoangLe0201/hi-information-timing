r"""Verify every proposition independently of the code that produced the paper.

The three numerical audits compare the manuscript against stored results.  None
of them checks whether the propositions are true.  A proposition can be stated
wrongly, proved loosely, or be right in the code and mis-transcribed into the
paper, and all three audits would pass.

This re-derives each of the twelve propositions from its statement as printed,
using numerical quadrature and simulation rather than the closed forms the paper
gives, so that agreement is evidence and not a tautology.  Where the paper
asserts a closed form, the closed form is evaluated separately from the integral
it claims to equal.

Model throughout: X_k = D(tau_k + delta) + eps_k with eps independent and
sigma constant unless stated otherwise, so the score for the clock offset is
D'(tau) and the Fisher information is the integral of D'^2 / sigma^2.
"""
import io
import os
import re

import numpy as np
from scipy import integrate, optimize, stats

rng = np.random.default_rng(20260831)
FAIL = []

# --- proposition headings, numbered from the manuscript ----------------------
# Numbers are looked up in mssp.aux by title rather than typed, because they were
# typed once and drifted: the headings carried the numbering the paper had when
# each check was written, and the paper has been reordered several times since.
_PAPER = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                      "paper")
_AUX = os.path.join(_PAPER, "mssp.aux")
_SAUX = os.path.join(_PAPER, "supplement.aux")


def _paper_props():
    """Proposition titles to the numbers actually printed, from both documents.

    Several propositions moved to the supplement, where they are numbered S1,
    S2 and so on.  Reading only mssp.aux made those look like propositions that
    no longer exist: the checks on them printed "?" for the number and carried
    on, which is exactly the silence this layer exists to prevent.
    """
    out = {}
    for path, mark in ((_AUX, ""), (_SAUX, "S")):
        if not os.path.exists(path):
            continue
        t = io.open(path, encoding="utf-8", errors="replace").read()
        for m in re.finditer(
                r"\\newlabel\{(prop:[^}]*)\}\{\{S?(\d+)\}\{\d+\}\{([^}]*)\}", t):
            out[m.group(3)] = mark + m.group(2)
    return out


PAPER_PROPS = _paper_props()
COVERED = set()


def prop(*titles):
    """Head a block, printing the number the manuscript actually gives it."""
    nums = []
    for t in titles:
        if t in PAPER_PROPS:
            nums.append(str(PAPER_PROPS[t]))
            COVERED.add(t)
        else:
            nums.append("?")
            FAIL.append("no proposition titled %r in the manuscript" % t)
    word = "PROPOSITION" if len(titles) == 1 else "PROPOSITIONS"
    print("\n" + "=" * 92)
    print("%s %s  %s" % (word, " and ".join(nums), ";  ".join(titles)))
    print("=" * 92)



def check(name, got, want, tol=1e-6, rel=True):
    d = abs(got - want) / max(abs(want), 1e-300) if rel else abs(got - want)
    ok = d <= tol
    if not ok:
        FAIL.append(name)
    print("  %-58s %14.8g %14.8g  %s"
          % (name, got, want, "ok" if ok else "FAIL"))


def head(t):
    print("\n" + t)
    print("  %-58s %14s %14s" % ("", "derived", "as printed"))


# =============================================================================
prop('The curve is the design')
# Build an explicit record, form G by summation, and estimate the clock by
# maximum likelihood from simulated data.  Claim (i) is that the standard
# deviation is G(tau_0)^{-1/2}.
n = 4000
tau = np.linspace(1e-3, 1.0, n)
A, beta, sig = 1.0, 2.3, 0.02
D = lambda t: A * np.maximum(t, 0) ** beta
Dp = lambda t: A * beta * np.maximum(t, 1e-12) ** (beta - 1)
Gc = np.cumsum(Dp(tau) ** 2 / sig ** 2)          # G(tau_k), unnormalised

head("(i) the maximum-likelihood standard error meets the floor")
for frac in (0.25, 0.5, 1.0):
    m = int(frac * n)
    t, d = tau[:m], Dp(tau[:m])
    est = []
    for _ in range(4000):
        y = D(t) + rng.normal(0, sig, m)
        # linear ML for a small offset: delta_hat = sum(d*(y-D))/sum(d^2)
        est.append(np.sum(d * (y - D(t))) / np.sum(d * d))
    check("record to tau=%.2f: sd(delta_hat) x sqrt(G)" % tau[m - 1],
          float(np.std(est)) * np.sqrt(Gc[m - 1]), 1.0, tol=0.03)

head("(ii)-(iv) the design quantities are quantiles of the same G")
eps = 0.004
c = eps ** -2
i_min = int(np.searchsorted(Gc, c))
tau_min = tau[i_min]
# (ii) infeasible strictly before tau_min, feasible at it
check("G just below tau_min is short of the demand",
      float(Gc[i_min - 1] < c), 1.0, rel=False, tol=0)
check("G at tau_min meets the demand", float(Gc[i_min] >= c), 1.0,
      rel=False, tol=0)
# (iii) and (iv) are statements about the continuous curve, so they are checked
# against quadrature rather than against a grid, where searchsorted lands on a
# sample and the window is short by one step.
gC = lambda t: Dp(t) ** 2 / sig ** 2
GC = lambda t: integrate.quad(gC, 0, t, limit=300)[0]
t0 = 0.9
cw3 = 0.25 * GC(1.0)
root = optimize.brentq(lambda x: GC(x) - (GC(t0) - cw3), 1e-9, t0, xtol=1e-14)
check("window of the stated width supplies exactly the demand",
      float(GC(t0) - GC(root)), cw3, tol=1e-9)
check("any narrower window supplies less",
      float(GC(t0) - GC(root * 1.001) < cw3), 1.0, rel=False, tol=0)
cc = 0.6
check("censored budget fraction G(c)/G(1)", float(GC(cc) / GC(1.0)),
      float(cc ** (2 * beta - 1)), tol=1e-9)

# =============================================================================
prop('The buffer requirement falls with age')
# dw*/dtau0 = 1 - g(tau0)/g(tau0 - w*), checked by finite differences on the
# implicit definition G(tau0) - G(tau0-w*) = c.
head("dw*/dtau0 against 1 - g(tau0)/g(tau0-w*)")
g = lambda t: Dp(t) ** 2 / sig ** 2
Gfun = lambda t: integrate.quad(g, 0, t, limit=200)[0]


def wstar(t0, c):
    lo = Gfun(t0) - c
    if lo <= 0:
        return None
    r = optimize.brentq(lambda x: Gfun(x) - lo, 1e-9, t0)
    return t0 - r


cw = 0.05 * Gfun(1.0)
for t0 in (0.55, 0.75, 0.95):
    h = 1e-4
    num = (wstar(t0 + h, cw) - wstar(t0 - h, cw)) / (2 * h)
    ana = 1.0 - g(t0) / g(t0 - wstar(t0, cw))
    check("tau0=%.2f" % t0, float(num), float(ana), tol=2e-4, rel=False)
# and the sign law: for a rising density the buffer must shrink
check("buffer shrinks with age for a rising density",
      float(all(wstar(t, cw) > wstar(t + 0.05, cw) for t in (0.55, 0.7, 0.85))),
      1.0, rel=False, tol=0)

# =============================================================================
prop('Adding a channel cannot cost')
head("the Schur increment, on random positive-definite covariances")
worst_id, worst_neg = 0.0, 0.0
for _ in range(2000):
    K = int(rng.integers(1, 7))
    B = rng.normal(size=(K + 1, K + 1))
    Sp = B @ B.T + (K + 1) * np.eye(K + 1) * 1e-3
    dp = rng.normal(size=K + 1)
    S, sv, v = Sp[:K, :K], Sp[:K, K], Sp[K, K]
    d, gg = dp[:K], dp[K]
    lhs = dp @ np.linalg.solve(Sp, dp)
    base = d @ np.linalg.solve(S, d)
    incr = (gg - sv @ np.linalg.solve(S, d)) ** 2 / (v - sv @ np.linalg.solve(S, sv))
    worst_id = max(worst_id, abs(lhs - (base + incr)) / abs(lhs))
    worst_neg = min(worst_neg, incr)
check("worst relative error in the block-inversion identity",
      float(worst_id), 0.0, tol=1e-8, rel=False)
check("smallest increment observed (must be non-negative)",
      float(worst_neg), 0.0, tol=1e-12, rel=False)
# the vanishing condition
K = 4
B = rng.normal(size=(K + 1, K + 1))
Sp = B @ B.T + np.eye(K + 1)
S, sv, v = Sp[:K, :K], Sp[:K, K], Sp[K, K]
d = rng.normal(size=K)
gg = sv @ np.linalg.solve(S, d)          # exactly the redundant value
dp = np.append(d, gg)
check("increment when g = s'Sigma^-1 d",
      float(dp @ np.linalg.solve(Sp, dp) - d @ np.linalg.solve(S, d)),
      0.0, tol=1e-9, rel=False)

# =============================================================================
prop('Bayesian floor')
head("posterior mean attains 1/(G + s^-2)")
for G0, s in ((50.0, 0.3), (5000.0, 0.294), (2.0, 1.0)):
    N = 200000
    delta = rng.normal(0, s, N)
    ml = delta + rng.normal(0, G0 ** -0.5, N)
    post = G0 / (G0 + s ** -2) * ml
    check("G=%-8g s=%-5g  mse x (G+s^-2)" % (G0, s),
          float(np.mean((post - delta) ** 2) * (G0 + s ** -2)), 1.0, tol=0.02)
check("a demand looser than the prior needs no measurement: eps>=s at G=0",
      float(1.0 / (0.0 + 0.294 ** -2)) ** 0.5, 0.294, tol=1e-12)

# =============================================================================
prop('The tail factor')
head("G_hat = kappa^2 G and its three consequences")
kappa = 1.35
check("budget overstated by kappa^2", (kappa ** 2 * 100.0) / 100.0,
      kappa ** 2, tol=1e-12)
check("floor understates the error by kappa",
      (100.0) ** -0.5 / (kappa ** 2 * 100.0) ** -0.5, kappa, tol=1e-12)
# an age read at demand c is the age the true curve reaches at c/kappa^2
Ghat = Gc * kappa ** 2
c0 = 1e4
a_read = tau[int(np.searchsorted(Ghat, c0))]
a_true = tau[int(np.searchsorted(Gc, c0 / kappa ** 2))]
check("age read at c equals true age at c/kappa^2",
      float(a_read), float(a_true), tol=1e-9)
# a drifting kappa_k: the robust curve is kappa_eff^2(tau) G(tau) exactly, with
# kappa_eff^2 the information-weighted mean of kappa_k^2 up to tau, and its
# shape differs from G's, so a relative age moves (twelfth reviewer read)
_kk = 1.2 + 0.3 * tau
_gk = Dp(tau) ** 2 / sig ** 2
_Grob = np.cumsum(_kk ** 2 * _gk)
_keff2 = np.cumsum(_kk ** 2 * _gk) / Gc
check("drifting kappa: G_rob = kappa_eff^2(tau) G(tau) at every tau",
      float(np.max(np.abs(_Grob - _keff2 * Gc) / _Grob)), 0.0, rel=False,
      tol=1e-12)
_tq_true = tau[int(np.searchsorted(Gc / Gc[-1], 0.35))]
_tq_rob = tau[int(np.searchsorted(_Grob / _Grob[-1], 0.35))]
check("drifting kappa moves the relative age (rising kappa: later)",
      float(_tq_rob > _tq_true), 1.0, rel=False, tol=0)

# =============================================================================
prop('Standard error of the relative information age')
head("delta-method se(tau_min) = se(G(tau_min))/g(tau_min)")
# Perturb G by a known noise level and measure the induced spread in tau_min.
# g must be the density of G with respect to tau, not the per-sample increment:
# on a grid of n points spanning (0,1] the two differ by exactly n, which is
# what a first version of this check got wrong.
gden = float(g(tau_min)) * n
for seG in (0.02 * c, 0.05 * c):
    shifts = rng.normal(0, seG, 20000)
    taus = np.interp(c - shifts, Gc, tau)      # solve G_hat(tau)=c
    check("se(G)=%.4g" % seG, float(np.std(taus)), seG / gden, tol=0.05)

# =============================================================================
prop('Cost of a replacement policy')
head("the cost integral, by simulation against the closed form")
phi, Phi = stats.norm.pdf, stats.norm.cdf
for sg, r in ((0.20, 0.05), (0.05, 0.25), (0.294, 0.02)):
    a = 0.9
    u = (a - 1) / sg
    N = 2000000
    e = rng.normal(0, sg, N)
    failed = (e <= a - 1)                       # replacement at a-e is late
    discarded = np.maximum(1.0 - (a - e), 0.0)
    sim = np.mean(failed) + r * np.mean(discarded)
    closed = Phi(u) + r * sg * (phi(u) - u * (1 - Phi(u)))
    check("sigma=%.3f r=%.2f" % (sg, r), float(sim), float(closed), tol=0.01)

head("the optimum sets the normal hazard to c_p sigma / c_f")
for sg, r in ((0.20, 0.05), (0.05, 0.25)):
    f = lambda u: Phi(u) + r * sg * (phi(u) - u * (1 - Phi(u)))
    u_num = optimize.minimize_scalar(f, bounds=(-40, 10),
                                     method="bounded",
                                     options=dict(xatol=1e-10)).x
    hz = lambda u: phi(u) / (1 - Phi(u))
    u_haz = optimize.brentq(lambda u: hz(u) - r * sg, -40, 8, xtol=1e-12)
    check("sigma=%.2f r=%.2f  argmin vs hazard root" % (sg, r),
          float(u_num), float(u_haz), tol=1e-4, rel=False)
# the second-order condition the paper does not state
head("the stationary point is a minimum (h(u) > u for the normal)")
# The survival probability must come from norm.sf, not from 1 - cdf: at u = 8
# the subtraction loses the last digits and reports 6.66e-16 where the true
# value is 6.22e-16, which turns h(u) - u negative and fakes a failure.
us = np.linspace(-8, 12, 4001)
mn = float(np.min(phi(us) / stats.norm.sf(us) - us))
print("  %-58s %14.8g %14s  %s"
      % ("min of h(u)-u over [-8,8] (must be > 0)", mn, "> 0",
         "ok" if mn > 0 else "FAIL"))
if mn <= 0:
    FAIL.append("second-order condition")

# =============================================================================
prop('Exact nuisance cost')
# Fisher matrix for (delta, A, beta) with D = A tau^beta, sigma constant,
# every element by numerical quadrature.  No closed form is used here.
head("profiled information fractions, by quadrature")


def fisher(beta, A=1.0):
    d_delta = lambda t: A * beta * t ** (beta - 1)
    d_A = lambda t: t ** beta
    d_beta = lambda t: A * t ** beta * np.log(t)
    fs = [d_delta, d_A, d_beta]
    I = np.empty((3, 3))
    for i in range(3):
        for j in range(3):
            I[i, j] = integrate.quad(lambda t: fs[i](t) * fs[j](t),
                                     0, 1, limit=400)[0]
    return I


def remaining(I, keep=(0,), out=()):
    """Information on `keep` after profiling out `out` (Schur complement)."""
    k, o = list(keep), list(out)
    if not o:
        return I[np.ix_(k, k)][0, 0]
    return (I[np.ix_(k, k)]
            - I[np.ix_(k, o)] @ np.linalg.solve(I[np.ix_(o, o)], I[np.ix_(o, k)]))[0, 0]


for beta in (0.75, 1.0, 1.6, 2.3, 4.0):
    I = fisher(beta)
    base = I[0, 0]
    check("beta=%-5g  r_A            = 1/(4b^2)" % beta,
          remaining(I, (0,), (1,)) / base, 1.0 / (4 * beta ** 2), tol=1e-7)
    check("beta=%-5g  r_beta         = 1-(2b-1)(2b+1)^3/32b^4" % beta,
          remaining(I, (0,), (2,)) / base,
          1 - (2 * beta - 1) * (2 * beta + 1) ** 3 / (32 * beta ** 4), tol=1e-7)
    check("beta=%-5g  r_Abeta        = 1/(16b^4) = r_A^2" % beta,
          remaining(I, (0,), (1, 2)) / base, 1.0 / (16 * beta ** 4), tol=1e-6)

head("the stated properties of r_beta")
rb = lambda b: 1 - (2 * b - 1) * (2 * b + 1) ** 3 / (32 * b ** 4)
bs = np.linspace(0.5001, 60, 400001)
check("argmin of r_beta", float(bs[np.argmin(rb(bs))]), 1.0, tol=1e-4, rel=False)
check("r_beta at its minimum", float(rb(1.0)), 5.0 / 32.0, tol=1e-12)
check("r_beta limit as beta grows", float(rb(1e7)), 0.5, tol=1e-6)

head("K channels sharing the clock and exponent: cost independent of K")


def fisher_multi(beta, amps):
    """Parameters (delta, A_1..A_K, beta); one shared clock and exponent."""
    K = len(amps)
    p = 2 + K
    I = np.zeros((p, p))
    for jj, A in enumerate(amps):
        f = [lambda t, A=A: A * beta * t ** (beta - 1)]
        f += [(lambda t, jj2=jj2: t ** beta) if jj2 == jj else (lambda t: 0.0 * t)
              for jj2 in range(K)]
        f += [lambda t, A=A: A * t ** beta * np.log(t)]
        for i in range(p):
            for j in range(p):
                I[i, j] += integrate.quad(lambda t: f[i](t) * f[j](t), 0, 1,
                                          limit=400)[0]
    return I


for beta in (1.3, 2.3):
    for amps in ([1.0], [1.0, 2.0], [0.5, 1.0, 3.0, 0.2]):
        I = fisher_multi(beta, amps)
        out = tuple(range(1, 2 + len(amps)))
        check("beta=%-4g K=%d  r_Abeta" % (beta, len(amps)),
              remaining(I, (0,), out) / I[0, 0],
              1.0 / (16 * beta ** 4), tol=1e-6)

# =============================================================================
prop('Linearisation radius')
head("bias, RMSE and the radius, by simulation")
bt = 2.3
t = np.linspace(1e-3, 1, 3000)
dv = A * bt * t ** (bt - 1)
d2 = A * bt * (bt - 1) * t ** (bt - 2)
G0 = float(np.sum(dv ** 2) / sig ** 2)
C = float(np.sum(dv * d2 / sig ** 2) / G0)
# the closed form the paper gives for a power law
check("C for D = A tau^beta equals beta - 1/2", C, bt - 0.5, tol=2e-3)
for dl in (0.002, 0.005):
    N = 20000
    est = np.empty(N)
    for i in range(N):
        y = A * np.maximum(t + dl, 0) ** bt + rng.normal(0, sig, t.size)
        est[i] = np.sum(dv * (y - A * t ** bt) / sig ** 2) / G0
    # The bias here is a few parts in a million while the estimator's own
    # standard deviation is 3e-4, so at N draws the Monte-Carlo noise on the
    # mean is comparable to the bias itself.  A relative tolerance would be
    # measuring the sample size, not the proposition; the discrepancy is
    # therefore judged in units of that noise.
    mc = float(np.std(est)) / np.sqrt(N)
    z = abs(float(np.mean(est) - dl) - 0.5 * C * dl ** 2) / mc
    ok = z < 3.0
    if not ok:
        FAIL.append("bias at delta=%.4f" % dl)
    print("  %-58s %14.8g %14.8g  %s"
          % ("delta=%.4f  bias, discrepancy in MC sigmas = %.2f" % (dl, z),
             float(np.mean(est) - dl), 0.5 * C * dl ** 2,
             "ok" if ok else "FAIL"))
    check("delta=%.4f  RMSE/floor vs sqrt(1+C^2 d^4 G/4)" % dl,
          float(np.sqrt(np.mean((est - dl) ** 2)) * np.sqrt(G0)),
          float(np.sqrt(1 + 0.25 * C ** 2 * dl ** 4 * G0)), tol=0.03)
dmax = np.sqrt(2 / (abs(C) * np.sqrt(G0)))
check("degradation at delta_max equals sqrt(2)",
      float(np.sqrt(1 + 0.25 * C ** 2 * dmax ** 4 * G0)), np.sqrt(2), tol=1e-9)
check("at delta_max the bias equals the standard error",
      float(0.5 * abs(C) * dmax ** 2), float(G0 ** -0.5), tol=1e-9)

# =============================================================================
prop('Best fixed indicator')
head("generalised eigenvalue, the trace bound, and the rank-one case")
from scipy.linalg import eigh
for _ in range(200):
    K = int(rng.integers(2, 8))
    n2 = int(rng.integers(K + 2, 40))
    Dm = rng.normal(size=(n2, K))
    M = Dm.T @ Dm
    B = rng.normal(size=(K, K))
    S = B @ B.T + K * np.eye(K) * 1e-2
    lam = float(eigh(M, S, eigvals_only=True)[-1])
    # brute force over random directions must never exceed it
    a = rng.normal(size=(K, 4000))
    q = np.einsum("ik,ij,jk->k", a, M, a) / np.einsum("ik,ij,jk->k", a, S, a)
    if q.max() > lam * (1 + 1e-9):
        FAIL.append("Rayleigh quotient exceeds lambda_max")
    if lam > np.trace(np.linalg.solve(S, M)) * (1 + 1e-9):
        FAIL.append("lambda_max exceeds the trace")
_p10 = [f for f in FAIL if "Rayleigh" in f or "trace" in f]
print("  %-58s %14s %14s  %s"
      % ("random search never beats lambda_max, and lambda<=trace",
         "200 draws", "-", "ok" if not _p10 else "FAIL"))
# rank-one equality
K = 5
u = rng.normal(size=K)
M1 = np.outer(u, u)
B = rng.normal(size=(K, K)); S = B @ B.T + np.eye(K)
check("rank-one M: lambda_max equals trace(S^-1 M)",
      float(eigh(M1, S, eigvals_only=True)[-1]),
      float(np.trace(np.linalg.solve(S, M1))), tol=1e-9)

# =============================================================================
prop('Earliest age reachable by any linear indicator', 'Leverage bounds the span')
head("noise cancels, monotonicity, the QR identity and the trace bound")
for trial in range(300):
    K = int(rng.integers(2, 7))
    n2 = int(rng.integers(K + 5, 60))
    Dm = rng.normal(size=(n2, K))
    m = int(rng.integers(1, n2))
    M1_ = Dm.T @ Dm
    Mc = Dm[:m].T @ Dm[:m]
    lam = float(eigh(Mc, M1_, eigvals_only=True)[-1])
    Q, R = np.linalg.qr(Dm)
    lam_q = float(np.linalg.eigvalsh(Q[:m].T @ Q[:m])[-1])
    h = np.einsum("ij,jk,ik->i", Dm, np.linalg.inv(M1_), Dm)
    if abs(lam - lam_q) > 1e-8 * max(1.0, abs(lam)):
        FAIL.append("QR identity")
    if lam > min(1.0, h[:m].sum()) + 1e-9:
        FAIL.append("lambda_max exceeds min(1, H(c))")
    if lam < h[:m].sum() / K - 1e-9:
        FAIL.append("lambda_max below H(c)/K")
    if abs(h.sum() - K) > 1e-8:
        FAIL.append("sum of leverages is not K")
    # monotone in c
    lam2 = float(eigh(Dm[:min(m + 3, n2)].T @ Dm[:min(m + 3, n2)], M1_,
                      eigvals_only=True)[-1])
    if lam2 < lam - 1e-9:
        FAIL.append("lambda_max not non-decreasing in c")
print("  %-58s %14s %14s  %s"
      % ("300 random designs: identity, both bounds, sum h = K, monotone",
         "300", "-", "ok"))
# the noise covariance really does cancel in the ratio
K, n2 = 4, 30
Dm = rng.normal(size=(n2, K))
a = rng.normal(size=K)
for Sfac in (0.3, 7.0):
    B = rng.normal(size=(K, K)); S = (B @ B.T + np.eye(K)) * Sfac
    num = a @ (Dm[:10].T @ Dm[:10]) @ a
    den = a @ (Dm.T @ Dm) @ a
    check("ratio with covariance scaled by %.1f" % Sfac,
          float(num / den),
          float((a @ (Dm[:10].T @ Dm[:10]) @ a) / (a @ (Dm.T @ Dm) @ a)),
          tol=1e-12)

# =============================================================================
prop('Which curve is the floor')
head("the information inequality, on four families that are not Student-t")
# The paper fits a t, so reading the inequality off the t family would prove
# nothing about the inequality itself.  These four are checked by quadrature.


def loc_fisher(pdf, dpdf, lo=-60.0, hi=60.0):
    def integrand(x):
        p = pdf(x)
        return (dpdf(x) ** 2 / p) if p > 1e-300 else 0.0
    return integrate.quad(integrand, lo, hi, limit=600)[0]


_mix = lambda x: 0.9 * stats.norm.pdf(x) + 0.1 * stats.norm.pdf(x, scale=4.0)
_dmix = lambda x: (0.9 * (-x) * stats.norm.pdf(x)
                   + 0.1 * (-x / 16.0) * stats.norm.pdf(x, scale=4.0))
FAMS = [
    ("logistic", stats.logistic.pdf,
     lambda x: stats.logistic.pdf(x) * (1 - 2 * stats.logistic.cdf(x)),
     float(stats.logistic.var())),
    ("Laplace", stats.laplace.pdf,
     lambda x: -np.sign(x) * stats.laplace.pdf(x),
     float(stats.laplace.var())),
    ("Gaussian", stats.norm.pdf, lambda x: -x * stats.norm.pdf(x), 1.0),
    ("normal mixture", _mix, _dmix, 0.9 + 0.1 * 16.0),
]
for _name, _pdf, _dpdf, _var in FAMS:
    _I = loc_fisher(_pdf, _dpdf)
    _ok = _I * _var >= 1.0 - 1e-6
    if not _ok:
        FAIL.append("information inequality for the " + _name)
    print("  %-58s %14.8g %14s  %s"
          % ("I_f x var, " + _name + " (must be at least 1)", _I * _var,
             ">= 1", "ok" if _ok else "FAIL"))
check("equality holds exactly for the Gaussian",
      loc_fisher(stats.norm.pdf, lambda x: -x * stats.norm.pdf(x)), 1.0,
      tol=1e-8)

head("the factorisation of the tail factor, and the robust curve's excess")
for _nu in (3.0, 4.5, 8.0, 40.0):
    _I = (_nu + 1) / (_nu + 3)
    _var = _nu / (_nu - 2)
    _rob2 = (1.4826 * stats.t.ppf(0.75, _nu)) ** 2
    check("nu=%-5g  (G_rob/G*) x (G*/G_2) equals kappa^2" % _nu,
          (1.0 / (_rob2 * _I)) * (_I * _var), _var / _rob2, tol=1e-12)
    # G_rob > G* puts the robust floor BELOW the Cramer-Rao floor: valid but
    # optimistic.  G_2 < G* puts the second-moment floor ABOVE it: it binds
    # linear estimators (least squares attains it), not every estimator.  An
    # earlier label read "so it is not a floor", which had the direction of
    # the inequality on the floor backwards (twelfth reviewer read).
    check("nu=%-5g  G_rob exceeds G*: robust floor below the CRLB" % _nu,
          float(1.0 / (_rob2 * _I) > 1.0), 1.0, rel=False, tol=0)
    check("nu=%-5g  G_2 below G*: second-moment floor above the CRLB" % _nu,
          float(_I * _var > 1.0), 1.0, rel=False, tol=0)
check("second factor for t_5 equals nu(nu+1)/((nu+3)(nu-2))",
      (6.0 / 8.0) * (5.0 / 3.0), 5.0 * 6.0 / (8.0 * 3.0), tol=1e-12)

# =============================================================================
prop('Information under serial correlation')
head("the AR(1) factor, against the exact tridiagonal inverse")


def _ar_quad(dp, phi):
    """d' Sigma^-1 d' for an AR(1) covariance, without forming the matrix."""
    d = np.full(len(dp), 1.0 + phi ** 2)
    d[0] = d[-1] = 1.0
    return float(d @ (dp * dp) - 2 * phi * (dp[:-1] @ dp[1:])) / (1 - phi ** 2)


# A second, independent route: build the AR(1) covariance and invert it, so the
# tridiagonal form used above is itself checked rather than assumed.
_n = 300
_t = np.linspace(1e-3, 1.0, _n)
_dp = 2.0 * _t
for _phi in (-0.3, 0.0, 0.25, 0.6):
    # Marginal variance one, matching what _ar_quad assumes.  A first version
    # divided this by 1 - phi^2, giving the covariance of a unit-innovation
    # process instead, and the two then disagreed by exactly that factor.
    _S = _phi ** np.abs(np.subtract.outer(np.arange(_n), np.arange(_n)))
    _direct = float(_dp @ np.linalg.solve(_S, _dp))
    check("phi=%-5g  tridiagonal form against an explicit inverse" % _phi,
          _ar_quad(_dp, _phi), _direct, tol=1e-8)

head("the factor is (1-phi)/(1+phi), with an error falling as 1/n")
for _phi in (0.3, 0.8):
    errs = []
    for _m in (200, 800, 3200, 12800):
        _tt = np.linspace(1e-3, 1.0, _m)
        _d = 2.0 * _tt
        _r = _ar_quad(_d, _phi) / float(_d @ _d)
        errs.append(abs(_r - (1 - _phi) / (1 + _phi)) / ((1 - _phi) / (1 + _phi)))
    # each fourfold rise in n should divide the error by about four
    ratios = [errs[i] / errs[i + 1] for i in range(len(errs) - 1)]
    check("phi=%-5g  error ratio per fourfold n (should be 4)" % _phi,
          float(np.mean(ratios)), 4.0, tol=0.1)
check("the factor is the reciprocal of the integrated autocorrelation time",
      (1 - 0.3) / (1 + 0.3), 1.0 / ((1 + 0.3) / (1 - 0.3)), tol=1e-12)

head("the paper's own diagnostic is blind below phi = 1/e")
# acorr_len is the first lag at which phi^k < exp(-1); for AR(1) that is 1 for
# every phi below exp(-1), where the information has already halved.
_thr = float(np.exp(-1))
check("largest phi the diagnostic still calls one sample", _thr, 0.3679,
      tol=1e-3, rel=False)
check("information remaining at that phi",
      (1 - _thr) / (1 + _thr), 0.4621, tol=1e-3, rel=False)

# =============================================================================
prop('Response to a distorted density')
# The three sensitivity formulas are first-order, so the test is not a tolerance
# but a SCALING: the relative error must fall like eps.  It is re-derived here on
# a curve this file chooses, independently of sensitivity_theory.py, and the
# ratio of the errors at two values of eps must be the ratio of those eps.
_n = 40000
_t = (np.arange(_n) + 1.0) / _n
_g = (0.4 + 2.2 * _t ** 3) / _n
_G = np.cumsum(_g)
_u = np.sin(3.0 * _t) + 0.5 * _t
_u = _u - float(np.sum(_u * _g) / np.sum(_g))
_q = 0.35


def _inv_lin(G, y):
    j = int(np.searchsorted(G, y))
    if j >= len(G):
        return np.nan
    if j == 0:
        return (y / G[0]) / len(G)
    lo, hi = G[j - 1], G[j]
    return (j + (0.0 if hi <= lo else (y - lo) / (hi - lo))) / len(G)


def _wm(u, g, a, b):
    d = float(np.sum(g[a:b]))
    return float(np.sum(u[a:b] * g[a:b])) / d if d > 0 else np.nan


_errs = {}
for _eps in (0.04, 0.01):
    _Gd = np.cumsum((1.0 + _eps * _u) * _g)
    # tau_min
    _a0, _a1 = _inv_lin(_G, _q * _G[-1]), _inv_lin(_Gd, _q * _Gd[-1])
    _i = min(_n - 1, max(0, int(round(_a0 * _n)) - 1))
    _pred = (_eps * _q * _G[-1] * (_wm(_u, _g, 0, _n) - _wm(_u, _g, 0, _i + 1))
             / (_g[_i] * _n))
    _errs.setdefault("tau_min", []).append(
        abs((_a1 - _a0) - _pred) / max(abs(_a1 - _a0), 1e-15))
    # eta
    _ic = min(_n - 1, max(0, int(round(0.5 * _n)) - 1))
    _obs = _Gd[_ic] / _Gd[-1] - _G[_ic] / _G[-1]
    _pe = (_G[_ic] / _G[-1]) * _eps * (_wm(_u, _g, 0, _ic + 1)
                                       - _wm(_u, _g, 0, _n))
    _errs.setdefault("eta", []).append(abs(_obs - _pe) / max(abs(_obs), 1e-15))

head("the age formula is first order, not merely close")
# a fourfold reduction in eps must give a fourfold reduction in the relative
# error: that is what identifies a first-order formula rather than a fit
check("tau_min: error ratio when eps falls fourfold",
      _errs["tau_min"][0] / max(_errs["tau_min"][1], 1e-15), 4.0, tol=0.6,
      rel=False)
check("tau_min: relative error at eps = 0.01", _errs["tau_min"][1], 0.0,
      tol=0.02, rel=False)

head("the eta formula is EXACT once the distortion is centred")
# This began as the same scaling test and failed, because eta's error sits at
# machine precision and the ratio of two such numbers is noise.  The reason is
# worth more than the test was.  Writing A and B for the numerator integrals,
#     eta_eps - eta = eps [G(1)A - G(c)B] / [G(1)(G(1) + eps B)],
# and B = G(1) <u>_[0,1], which the normalisation sets to zero.  The denominator
# then loses its eps and the first-order formula is the whole answer.
check("eta: absolute error at eps = 0.04", _errs["eta"][0], 0.0, tol=1e-9,
      rel=False)
check("eta: absolute error at eps = 0.01", _errs["eta"][1], 0.0, tol=1e-9,
      rel=False)

prop('What a bounded distortion can do')
head("the odds bound is attained, so it cannot be improved in M alone")
_ic = _n // 2 - 1
_e0 = float(np.sum(_g[:_ic + 1]) / np.sum(_g))
for _M in (1.1, 1.8, 3.0):
    _w = np.where(np.arange(_n) <= _ic, _M, 1.0 / _M)
    _gd = _w * _g
    _e1 = float(np.sum(_gd[:_ic + 1]) / np.sum(_gd))
    _r = (_e1 / (1 - _e1)) / (_e0 / (1 - _e0))
    check("worst odds ratio at M = %.1f" % _M, _r, _M * _M, tol=1e-9)

head("and it is never exceeded by an arbitrary bounded distortion")
_rngL = np.random.default_rng(24601)
_worst = 0.0
for _ in range(1500):
    _M = float(_rngL.uniform(1.05, 4.0))
    _lw = np.convolve(_rngL.uniform(-np.log(_M), np.log(_M), _n),
                      np.ones(17) / 17, mode="same")
    _lw = np.clip(_lw, -np.log(_M), np.log(_M))
    _gd = np.exp(_lw) * _g
    _e1 = float(np.sum(_gd[:_ic + 1]) / np.sum(_gd))
    _r = (_e1 / (1 - _e1)) / (_e0 / (1 - _e0))
    _worst = max(_worst, _r / (_M * _M), 1.0 / (_r * _M * _M))
# The bound is one-sided: the requirement is that nothing EXCEEDS it, so the
# quantity to compare against one is max(worst, 1) and not min.  Written the
# other way round this check fails whenever the bound comfortably holds, which
# is how it was first written and what the harness caught.
check("largest excursion beyond the bound, clipped at one",
      max(_worst, 1.0), 1.0, tol=1e-9, rel=False)

prop('A monotone distortion signs two quantities of three')
head("a non-increasing w raises F everywhere and cannot lower eta")
_rngM = np.random.default_rng(13331)
_badF = _badE = _badT = 0.0
for _ in range(1200):
    _g2 = (0.3 + 2.0 * np.linspace(0, 1, _n) ** float(_rngM.uniform(0.5, 4))) / _n
    _w = np.exp(float(_rngM.uniform(0.2, 3.0)) * np.sort(_rngM.random(_n))[::-1])
    _F0 = np.cumsum(_g2) / np.sum(_g2)
    _F1 = np.cumsum(_w * _g2) / np.sum(_w * _g2)
    _badF = max(_badF, float(np.max(_F0 - _F1)))
    for _c in (0.25, 0.5, 0.75):
        _i = int(_c * _n) - 1
        _badE = max(_badE, float(_F0[_i] - _F1[_i]))
    _a = int(np.searchsorted(_F0, 0.35))
    _b = int(np.searchsorted(_F1, 0.35))
    _badT = max(_badT, float(_b - _a) / _n)
check("worst F below the undistorted F", _badF, 0.0, tol=1e-12, rel=False)
check("worst eta below the undistorted eta", _badE, 0.0, tol=1e-12, rel=False)
check("worst tau_min above the undistorted", _badT, 0.0, tol=1e-12, rel=False)

head("and the window genuinely has no sign under the same hypothesis")
_wu = _wd = 0
for _ in range(1200):
    _g2 = (0.3 + 2.0 * np.linspace(0, 1, _n) ** float(_rngM.uniform(0.5, 4))) / _n
    _w = np.exp(float(_rngM.uniform(0.2, 3.0)) * np.sort(_rngM.random(_n))[::-1])
    _F0 = np.cumsum(_g2) / np.sum(_g2)
    _F1 = np.cumsum(_w * _g2) / np.sum(_w * _g2)
    _t0 = float(_rngM.uniform(0.6, 0.95))
    _i0 = min(_n - 1, max(0, int(round(_t0 * _n)) - 1))
    _o = []
    for _F in (_F0, _F1):
        _t = _F[_i0] - 0.35
        if _t <= 0:
            _o.append(np.nan)
            continue
        _j = int(np.searchsorted(_F, _t))
        _o.append(np.nan if _j >= _n else _t0 - (_j + 1) / _n)
    if not all(np.isfinite(_o)):
        continue
    _d = _o[1] - _o[0]
    _wu += int(_d > 1e-9)
    _wd += int(_d < -1e-9)
check("cases where w* widened (must be positive)", float(_wu > 0), 1.0,
      tol=0, rel=False)
check("cases where w* narrowed (must be positive)", float(_wd > 0), 1.0,
      tol=0, rel=False)

prop('A certificate for the ordering')
# Only the SUFFICIENT direction is claimed and only it is checked: M below the
# certificate must never reverse the pair.  M above it does not imply reversal
# is possible, since the condition lets each indicator take its own worst case
# and a shared distortion cannot supply both, so the converse is not tested.
_rngC = np.random.default_rng(86420)
_m = 3000
_th = 0.35 / 0.65


def _A(F, x):
    _p = x / (1.0 + x)
    if _p >= 1.0:
        return np.inf
    _i = int(np.searchsorted(F, _p))
    return np.inf if _i >= len(F) else (_i + 1) / len(F)


def _cert(F1, F2):
    if not (_A(F1, _th) < _A(F2, _th)):
        return np.nan
    _lo, _hi = 1.0, 30.0
    for _ in range(50):
        _mid = 0.5 * (_lo + _hi)
        if _A(F1, _mid ** 2 * _th) < _A(F2, _th / _mid ** 2):
            _lo = _mid
        else:
            _hi = _mid
    return _lo


head("no distortion at or below the certificate reverses the pair")
_rev, _tested = 0, 0
for _ in range(60):
    _tt = (np.arange(_m) + 1.0) / _m
    _d1 = (0.2 + 3 * _tt ** float(_rngC.uniform(0.5, 2.0))) / _m
    _d2 = (0.05 + 3 * _tt ** float(_rngC.uniform(4.0, 9.0))) / _m
    _F1 = np.cumsum(_d1) / np.sum(_d1)
    _F2 = np.cumsum(_d2) / np.sum(_d2)
    _M = _cert(_F1, _F2)
    if not np.isfinite(_M) or _M <= 1.001:
        continue
    for _ in range(60):
        _k = int(_rngC.integers(1, 6))
        _cuts = np.sort(_rngC.random(_k))
        _lvl = _rngC.choice([np.log(_M), -np.log(_M)], size=_k + 1)
        _w = np.exp(_lvl[np.searchsorted(_cuts, np.arange(_m) / _m)])
        _G1 = np.cumsum(_w * _d1); _G1 /= _G1[-1]
        _G2 = np.cumsum(_w * _d2); _G2 /= _G2[-1]
        _tested += 1
        _rev += int(_A(_G1, _th) >= _A(_G2, _th))
check("reversals at or below the certificate", float(_rev), 0.0, tol=0,
      rel=False)
check("distortions actually tested (must be many)",
      float(_tested > 500), 1.0, tol=0, rel=False)

prop('The contrast identity')
# An identity is checked at machine precision, not at a tolerance, and against
# distortions with no smoothness, since that is what separates it from the
# expansion of Proposition 15.
head("odds(eta_hat)/odds(eta) equals the ratio of weighted means")
_rngI = np.random.default_rng(97531)
_bad = 0.0
for _ in range(2500):
    _p = float(_rngI.uniform(0.4, 5.0))
    _g3 = (0.15 + 3.0 * np.linspace(0, 1, _n) ** _p) / _n
    _mode = int(_rngI.integers(0, 3))
    if _mode == 0:
        _w3 = np.exp(_rngI.uniform(-2, 2) * np.sort(_rngI.random(_n))[::-1])
    elif _mode == 1:
        _w3 = np.exp(np.convolve(_rngI.normal(0, 1, _n),
                                 np.ones(31) / 31, "same"))
    else:
        _w3 = np.exp(_rngI.uniform(-3, 3, _n))       # independent per sample
    _ic3 = int(_rngI.uniform(0.05, 0.95) * _n)
    _e0 = float(np.sum(_g3[:_ic3 + 1]) / np.sum(_g3))
    _gd3 = _w3 * _g3
    _e1 = float(np.sum(_gd3[:_ic3 + 1]) / np.sum(_gd3))
    if not (0 < _e0 < 1 and 0 < _e1 < 1):
        continue
    _lhs = (_e1 / (1 - _e1)) / (_e0 / (1 - _e0))
    _a = np.sum(_w3[:_ic3 + 1] * _g3[:_ic3 + 1]) / np.sum(_g3[:_ic3 + 1])
    _b = np.sum(_w3[_ic3 + 1:] * _g3[_ic3 + 1:]) / np.sum(_g3[_ic3 + 1:])
    _bad = max(_bad, abs(_lhs - _a / _b) / max(abs(_a / _b), 1e-300))
check("worst relative discrepancy over 2500 cases", _bad, 0.0, tol=1e-10,
      rel=False)

head("and the three earlier results are its corollaries")
# M^2 bound, monotone sign, and the first-order form, each recovered from it
_g4 = (0.2 + 2.0 * np.linspace(0, 1, _n) ** 2) / _n
_ic4 = _n // 3
for _M4 in (1.3, 2.2):
    _w4 = np.where(np.arange(_n) <= _ic4, _M4, 1.0 / _M4)
    _a = np.sum(_w4[:_ic4 + 1] * _g4[:_ic4 + 1]) / np.sum(_g4[:_ic4 + 1])
    _b = np.sum(_w4[_ic4 + 1:] * _g4[_ic4 + 1:]) / np.sum(_g4[_ic4 + 1:])
    check("contrast at the extremal w equals M^2 (M=%.1f)" % _M4,
          float(_a / _b), _M4 ** 2, tol=1e-9)
_w5 = np.exp(2.0 * np.sort(_rngI.random(_n))[::-1])
_a = np.sum(_w5[:_ic4 + 1] * _g4[:_ic4 + 1]) / np.sum(_g4[:_ic4 + 1])
_b = np.sum(_w5[_ic4 + 1:] * _g4[_ic4 + 1:]) / np.sum(_g4[_ic4 + 1:])
check("contrast of a falling w is at least one",
      float(min(_a / _b, 1.0)), 1.0, tol=1e-12, rel=False)

prop('The range, charged where the information is')


def _trim_sets(g, alpha):
    """The samples carrying the top 1-alpha of THIS side's budget."""
    o = np.argsort(g, kind="stable")
    k = int(np.searchsorted(np.cumsum(g[o]), alpha * g.sum(), side="right"))
    A = np.ones(len(g), bool)
    A[o[:k]] = False
    return A if A.any() else np.ones(len(g), bool)


def _trim_bound(w, g, ic, alpha):
    """Equation (10) evaluated from its printed statement, not from the study."""
    M = float(max(w.max(), 1.0 / w.min()))
    wl, wh, gl, gh = w[:ic + 1], w[ic + 1:], g[:ic + 1], g[ic + 1:]
    A0, A1 = _trim_sets(gl, alpha), _trim_sets(gh, alpha)
    a = 1.0 - alpha
    S0, s0 = float(wl[A0].max()), float(wl[A0].min())
    S1, s1 = float(wh[A1].max()), float(wh[A1].min())
    return float(max((a * S0 + alpha * M) / (a * s1 + alpha / M),
                     (a * S1 + alpha * M) / (a * s0 + alpha / M)))


head("the bound holds against arbitrary distortions and trimming levels")
# Adversarial by construction: densities with a share of exactly-zero samples,
# because that is what the noise floor leaves and it is the case the statement
# is meant to exploit; distortions with no smoothness at all; and cuts anywhere.
# A single ratio above one would refute the proposition rather than loosen it.
_rngT = np.random.default_rng(8080)
_nT = 800
_worstT, _casesT = 0.0, 0
for _ in range(3000):
    _pT = float(_rngT.uniform(0.3, 6.0))
    _gT = (0.05 + 3.0 * np.linspace(0, 1, _nT) ** _pT) / _nT
    _gT[_rngT.random(_nT) < 0.15] = 0.0
    _icT = int(_rngT.uniform(0.1, 0.9) * _nT)
    if _gT[:_icT + 1].sum() <= 0 or _gT[_icT + 1:].sum() <= 0:
        continue
    _mT = int(_rngT.integers(0, 3))
    if _mT == 0:
        _wT = np.exp(_rngT.uniform(-1.6, 1.6, _nT))
    elif _mT == 1:
        _wT = np.exp(np.convolve(_rngT.normal(0, 1, _nT),
                                 np.ones(25) / 25, "same"))
    else:
        _wT = np.exp(_rngT.uniform(-2, 2) * np.sort(_rngT.random(_nT))[::-1])
    _aT = float(np.sum(_wT[:_icT + 1] * _gT[:_icT + 1])
                / np.sum(_gT[:_icT + 1]))
    _bT = float(np.sum(_wT[_icT + 1:] * _gT[_icT + 1:])
                / np.sum(_gT[_icT + 1:]))
    _RT = max(_aT / _bT, _bT / _aT)
    for _alT in (0.0, 0.01, 0.1, 0.4, 1.0):
        _worstT = max(_worstT, _RT / _trim_bound(_wT, _gT, _icT, _alT))
        _casesT += 1
check("never exceeded over %d combinations (worst ratio %.4f)"
      % (_casesT, _worstT), float(_worstT <= 1.0), 1.0, tol=0, rel=False)

head("its two endpoints are the two statements it interpolates")
_gU = np.ones(600) / 600.0
_wU = np.exp(_rngT.uniform(-1.0, 1.0, 600))
_MU = float(max(_wU.max(), 1.0 / _wU.min()))
check("alpha = 1 returns M^2, which is the odds bound",
      _trim_bound(_wU, _gU, 299, 1.0), _MU ** 2, tol=1e-12)
check("alpha = 0 returns the range over the support",
      _trim_bound(_wU, _gU, 299, 0.0),
      float(max(_wU[:300].max() / _wU[300:].min(),
                _wU[300:].max() / _wU[:300].min())), tol=1e-12)

head("and it is attained, so it cannot be improved in these quantities alone")
# The extremal distortion is constant on each side's retained set and sits at the
# global extreme on the trimmed share.  It is built FROM the sets the formula
# uses rather than from a sample count: placing exactly alpha*n/2 samples by hand
# put one of them back inside the retained set, floating-point rounding having
# moved the trimming boundary by a single index, and the bound then read 25
# against a contrast of 4.9.  The proposition was not at fault; the construction
# was, and the difference matters because a tightness check that is off by one
# sample cannot tell a sharp inequality from a loose one.
_alE, _ME, _S0E, _s1E = 0.1, 5.0, 2.0, 0.5
_nE = 1000
_gE = np.ones(_nE) / _nE
_A0E = _trim_sets(_gE[:_nE // 2], _alE)
_A1E = _trim_sets(_gE[_nE // 2:], _alE)
_wE = np.empty(_nE)
_wE[:_nE // 2] = np.where(_A0E, _S0E, _ME)
_wE[_nE // 2:] = np.where(_A1E, _s1E, 1.0 / _ME)
_betaE = 1.0 - float(_A0E.mean())          # uniform g, so shares are counts
_gammaE = 1.0 - float(_A1E.mean())
_aE = float(np.mean(_wE[:_nE // 2]))
_bE = float(np.mean(_wE[_nE // 2:]))
check("the split step is an equality at the extremal distortion",
      _aE / _bE,
      ((1 - _betaE) * _S0E + _betaE * _ME)
      / ((1 - _gammaE) * _s1E + _gammaE / _ME), tol=1e-12)
check("and relaxing beta to alpha only loosens it",
      float(_trim_bound(_wE, _gE, _nE // 2 - 1, _alE) >= _aE / _bE - 1e-12),
      1.0, tol=0, rel=False)

prop('Worst contrast under an energy budget')
head("the closed form is exact for the two-level distortion")


def _energy(u, g):
    _t = float(np.sum(g))
    _mu = float(np.sum(u * g)) / _t
    return float(np.sqrt(np.sum((u - _mu) ** 2 * g) / _t))


_rngE = np.random.default_rng(31337)
_worstE = 0.0
for _ in range(400):
    _pw = float(_rngE.uniform(0.4, 6.0))
    _g6 = (0.1 + 3.0 * np.linspace(0, 1, _n) ** _pw) / _n
    _ic6 = int(_rngE.uniform(0.1, 0.9) * _n)
    _p6 = float(np.sum(_g6[:_ic6 + 1]) / np.sum(_g6))
    if not (0.02 < _p6 < 0.98):
        continue
    _eps6 = float(_rngE.uniform(0.03, 0.5))
    _h6 = np.where(np.arange(_n) <= _ic6, 1.0 / _p6, -1.0 / (1.0 - _p6))
    _u6 = _eps6 * _h6 / _energy(_h6, _g6)
    _w6 = np.exp(_u6)
    _a6 = np.sum(_w6[:_ic6 + 1] * _g6[:_ic6 + 1]) / np.sum(_g6[:_ic6 + 1])
    _b6 = np.sum(_w6[_ic6 + 1:] * _g6[_ic6 + 1:]) / np.sum(_g6[_ic6 + 1:])
    _pred6 = _eps6 / np.sqrt(_p6 * (1.0 - _p6))
    _worstE = max(_worstE, abs(np.log(_a6 / _b6) - _pred6) / _pred6)
check("worst relative error over 400 random cells", _worstE, 0.0, tol=1e-10,
      rel=False)

head("and the bound is tight only at small budgets, as the paper says")
# The claim has two halves and both are checked: nothing beats the step at a
# small budget, and something does at a large one.  Checking only the first
# would let the manuscript's qualification rot unnoticed.
_g7 = (0.2 + 3.0 * np.linspace(0, 1, _n) ** 4) / _n
_ic7 = int(0.35 * _n)
_p7 = float(np.sum(_g7[:_ic7 + 1]) / np.sum(_g7))
for _eps7, _want_beat in ((0.15, False), (0.40, True)):
    _h7 = np.where(np.arange(_n) <= _ic7, 1.0 / _p7, -1.0 / (1.0 - _p7))
    _step7 = _eps7 / np.sqrt(_p7 * (1.0 - _p7))
    # a deliberate Jensen attack: spread u inside the early set at fixed mean
    _best7 = _step7
    for _sp in np.linspace(0.0, 3.0, 40):
        _u7 = _eps7 * _h7 / _energy(_h7, _g7)
        _mask = np.arange(_n) <= _ic7
        _u7 = _u7 + _sp * _mask * np.sin(np.linspace(0, 30, _n))
        _u7 = _u7 - float(np.sum(_u7 * _g7) / np.sum(_g7))
        _e7 = _energy(_u7, _g7)
        if _e7 <= 0:
            continue
        _u7 = _eps7 * _u7 / _e7
        _w7 = np.exp(_u7)
        _a7 = np.sum(_w7[:_ic7 + 1] * _g7[:_ic7 + 1]) / np.sum(_g7[:_ic7 + 1])
        _b7 = np.sum(_w7[_ic7 + 1:] * _g7[_ic7 + 1:]) / np.sum(_g7[_ic7 + 1:])
        _best7 = max(_best7, float(np.log(_a7 / _b7)))
    _beaten = (_best7 - _step7) / _step7 > 1e-3
    check("at eps = %.2f the step is beaten" % _eps7,
          float(_beaten), float(_want_beat), tol=0, rel=False)

prop('The shared distortion fills an ellipse')
head("attainable (log R1, log R2) lies inside x' Gamma^-1 x <= eps^2")
_rngS = np.random.default_rng(778899)
_nS = 2000
_tS = (np.arange(_nS) + 1.0) / _nS
_gA = (2 * _tS) / _nS
_gB = (0.2 + 3 * _tS ** 4) / _nS
_gA, _gB = _gA / _gA.sum(), _gB / _gB.sum()
_muS = np.ones(_nS) / _nS
_icS = int(0.45 * _nS)
_epsS = 0.05


def _psi(g, ic):
    _p = float(np.sum(g[:ic + 1]) / np.sum(g))
    _h = np.where(np.arange(len(g)) <= ic, 1.0 / _p, -1.0 / (1.0 - _p))
    return _h * g / float(np.sum(g))


_pA, _pB = _psi(_gA, _icS), _psi(_gB, _icS)
_Gam = np.array([[float(np.sum(a * b / _muS)) for b in (_pA, _pB)]
                 for a in (_pA, _pB)])
_Gin = np.linalg.inv(_Gam)


def _logR(u, g, ic):
    _w = np.exp(u)
    _a = np.sum(_w[:ic + 1] * g[:ic + 1]) / np.sum(g[:ic + 1])
    _b = np.sum(_w[ic + 1:] * g[ic + 1:]) / np.sum(g[ic + 1:])
    return float(np.log(_a / _b))


_out = 0.0
for _ in range(600):
    _k = int(_rngS.integers(0, 3))
    if _k == 0:
        _u = _rngS.normal(0, 1, _nS)
    elif _k == 1:
        _u = np.convolve(_rngS.normal(0, 1, _nS), np.ones(41) / 41, "same")
    else:
        _u = np.sort(_rngS.random(_nS))[::-1]
    _u = _u - float(np.sum(_u * _muS))
    _nr = float(np.sqrt(np.sum(_u * _u * _muS)))
    if _nr <= 0:
        continue
    _u = _epsS * _u / _nr
    _x = np.array([_logR(_u, _gA, _icS), _logR(_u, _gB, _icS)])
    _out = max(_out, float(_x @ _Gin @ _x) / _epsS ** 2)
check("largest x'Gamma^-1x/eps^2 reached, clipped at one",
      max(_out, 1.0), 1.0, tol=1e-9, rel=False)

head("and the boundary is attained, to first order in eps")
_reach = []
for _ang in np.linspace(0.0, np.pi, 5):
    _d = np.array([np.cos(_ang), np.sin(_ang)])
    _f = (_d[0] * _pA + _d[1] * _pB) / _muS
    _f = _f - float(np.sum(_f * _muS))
    _nr = float(np.sqrt(np.sum(_f * _f * _muS)))
    _u = _epsS * _f / _nr
    _x = np.array([_logR(_u, _gA, _icS), _logR(_u, _gB, _icS)])
    _reach.append(float(_x @ _Gin @ _x) / _epsS ** 2)
check("closest the aimed directions come to the boundary",
      float(min(_reach)), 1.0, tol=0.03, rel=False)

prop('An estimated correction')
# The claim is about two terms of DIFFERENT ORDERS, so the test is a pair of
# scalings and not a pair of tolerances: the variance must fall with n and the
# bias must not.
_rngQ = np.random.default_rng(24680)


def _lr(w, g, ic):
    _a = np.sum(w[:ic + 1] * g[:ic + 1]) / np.sum(g[:ic + 1])
    _b = np.sum(w[ic + 1:] * g[ic + 1:]) / np.sum(g[ic + 1:])
    return float(np.log(_a / _b))


head("unbiased noise in the correction costs a variance that falls like 1/n")
_sds = {}
for _nq in (500, 8000):
    _tq = (np.arange(_nq) + 1.0) / _nq
    _gq = (0.3 + 2.5 * _tq ** 3) / _nq
    _icq = int(0.5 * _nq) - 1
    _wq = np.exp(0.4 * (_tq - 0.5))
    _v = []
    for _ in range(300):
        _zq = 0.2 * _rngQ.standard_normal(_nq)
        _v.append(_lr(_wq * np.exp(_zq - 0.5 * 0.04), _gq, _icq))
    _sds[_nq] = float(np.std(_v))
check("sd ratio between n = 500 and n = 8000", _sds[500] / _sds[8000], 4.0,
      tol=0.5, rel=False)

head("a correlated error costs a bias that does not")
_bias = {}
for _nq in (500, 8000):
    _tq = (np.arange(_nq) + 1.0) / _nq
    _gq = (0.3 + 2.5 * _tq ** 3) / _nq
    _icq = int(0.5 * _nq) - 1
    _wq = np.exp(0.4 * (_tq - 0.5))
    _base = _lr(_wq, _gq, _icq)
    _v = []
    for _ in range(400):
        _d1 = _rngQ.standard_normal(_nq)
        _d2 = _rngQ.standard_normal(_nq)
        _cc = np.where(np.arange(_nq) <= _icq, 0.8, -0.4)
        _zq = 0.25 * (_cc * _d1 + np.sqrt(np.maximum(1 - _cc * _cc, 0)) * _d2)
        _gg = np.clip(_gq * (1.0 + 0.25 * _d1), 1e-12, None)
        _v.append(_lr(_wq * np.exp(_zq), _gg, _icq))
    _bias[_nq] = float(np.mean(_v)) - _base
check("bias ratio between n = 500 and n = 8000",
      _bias[500] / _bias[8000], 1.0, tol=0.15, rel=False)
check("bias against its closed form at n = 8000", _bias[8000],
      0.25 * 0.25 * (0.8 - (-0.4)), tol=0.05)

prop('Precision of the censoring efficiency')
head("the delta-method variance against repeated draws")
_rngV = np.random.default_rng(13579)


def _cl(x):
    v = np.asarray(x, float)
    v = v - v.mean()
    _d = float(np.sum(v * v))
    if _d <= 0:
        return 1.0
    _t, _k = 1.0, 1
    while _k < min(len(v) // 4, 300):
        _r = float(np.sum(v[:-_k] * v[_k:])) / _d
        if _r <= 0:
            break
        _t += 2.0 * _r
        _k += 1
    return max(_t, 1.0)


def _flat(seg, frac=0.15):
    _m = len(seg)
    _h = max(5, int(frac * _m))
    _k = np.ones(2 * _h + 1) / (2 * _h + 1)
    _l = np.convolve(seg, _k, "same") / np.maximum(
        np.convolve(np.ones(_m), _k, "same"), 1e-12)
    return seg / np.maximum(_l, 1e-12)


for _nv, _law in ((800, "chi2"), (3000, "lognormal")):
    _shape = 0.3 + 2.0 * ((np.arange(_nv) + 1.0) / _nv) ** 2
    _icv = _nv // 2 - 1
    _obs, _one = [], None
    for _r in range(500):
        _b = (_rngV.chisquare(1, _nv) if _law == "chi2"
              else _rngV.lognormal(0, 1.0, _nv))
        _dv = _shape * _b
        if _r == 0:
            _one = _dv.copy()
        _A, _B = float(np.sum(_dv[:_icv + 1])), float(np.sum(_dv[_icv + 1:]))
        if _B > 0:
            _obs.append(np.log(_A / _B))
    _tot = 0.0
    for _seg in (_one[:_icv + 1], _one[_icv + 1:]):
        _rr = _flat(_seg)
        _mu = float(np.mean(_rr))
        _tot += (float(np.var(_rr)) / (_mu * _mu)) / (len(_seg) / _cl(_rr))
    check("%s, n = %d: formula over repeated sd" % (_law, _nv),
          float(np.sqrt(_tot)) / float(np.std(_obs)), 1.0, tol=0.2, rel=False)

head("and detrending is what makes it work")
# taken from the raw density the correlation length reads the shape as
# correlation; the check is that the undetrended version is materially worse
_shape = 0.3 + 2.0 * ((np.arange(3000) + 1.0) / 3000) ** 2
_icv = 1499
_obs = []
for _ in range(400):
    _dv = _shape * _rngV.lognormal(0, 1.0, 3000)
    _A, _B = float(np.sum(_dv[:_icv + 1])), float(np.sum(_dv[_icv + 1:]))
    _obs.append(np.log(_A / _B))
_one = _shape * _rngV.lognormal(0, 1.0, 3000)
_raw = 0.0
for _seg in (_one[:_icv + 1], _one[_icv + 1:]):
    _mu = float(np.mean(_seg))
    _raw += (float(np.var(_seg)) / (_mu * _mu)) / (len(_seg) / _cl(_seg))
check("undetrended formula over the truth (must exceed 1.5)",
      float(min(np.sqrt(_raw) / float(np.std(_obs)), 1.5)), 1.5, tol=1e-9,
      rel=False)

prop('Precision of the family gap')
head("se(tau) = se(F_hat)/f, against repeated draws")
_rngG = np.random.default_rng(112358)
_ng = 2000
_tg = (np.arange(_ng) + 1.0) / _ng
for _pw, _lbl in ((2.0, "moderate slope"), (6.0, "steep")):
    _sh = 0.2 + 3.0 * _tg ** _pw
    _ages = []
    _one = None
    for _r in range(500):
        _dg = _sh * _rngG.chisquare(1, _ng)
        if _r == 0:
            _one = _dg.copy()
        _F = np.cumsum(_dg) / np.sum(_dg)
        _i = int(np.searchsorted(_F, 0.35))
        if _i < _ng:
            _ages.append((_i + 1) / _ng)
    _truth = float(np.std(_ages))
    # the formula, from the single record
    _r1 = _flat(_one)
    _mu1 = float(np.mean(_r1))
    _cd = float(np.std(_r1)) / _mu1
    _m1 = _ng / _cl(_r1)
    _seF = np.sqrt(0.35 * 0.65) * _cd / np.sqrt(_m1)
    _F1 = np.cumsum(_one) / np.sum(_one)
    _i1 = int(np.searchsorted(_F1, 0.35))
    _h1 = max(5, _ng // 50)
    _f1 = ((_F1[min(_ng - 1, _i1 + _h1)] - _F1[max(0, _i1 - _h1)])
           / ((min(_ng - 1, _i1 + _h1) - max(0, _i1 - _h1)) / _ng))
    check("%s: formula over repeated sd" % _lbl,
          float(_seF / _f1) / _truth, 1.0, tol=0.35, rel=False)

head("and the variance decomposition is an identity, not an approximation")
# Var(total) = Var(within) + Var(between) holds exactly when the two are
# independent; the check builds them that way and confirms the arithmetic
_bet = _rngG.normal(0, 0.25, 40000)
_wit = _rngG.normal(0, 0.05, 40000)
check("total variance equals the sum of its parts",
      float(np.var(_bet + _wit)), float(np.var(_bet) + np.var(_wit)),
      tol=0.01)

prop('The joint information is not submodular')


def _inf(d, S, Sig):
    if len(S) == 0:
        return 0.0
    _i = np.array(sorted(S))
    return float(d[_i] @ np.linalg.solve(Sig[np.ix_(_i, _i)], d[_i]))


head("the noise-cancelling reference: worthless alone, decisive when paired")
for _rho in (0.8, 0.95, 0.99):
    _S = np.array([[1.0, _rho], [_rho, 1.0]])
    _d = np.array([1.0, 0.0])
    _a = _inf(_d, {1}, _S) - _inf(_d, set(), _S)
    _b = _inf(_d, {0, 1}, _S) - _inf(_d, {0}, _S)
    check("rho=%.2f: increment alone" % _rho, _a, 0.0, tol=1e-12, rel=False)
    check("rho=%.2f: increment when paired" % _rho, _b,
          _rho ** 2 / (1 - _rho ** 2), tol=1e-9)

head("so submodularity fails, and the failure is unbounded")
# submodularity requires the increment to shrink; here it grows without limit
_g = [(_r ** 2 / (1 - _r ** 2)) for _r in (0.9, 0.99, 0.999)]
check("increments grow monotonically with rho",
      float(all(_g[i] < _g[i + 1] for i in range(len(_g) - 1))), 1.0,
      tol=0, rel=False)
check("and exceed any fixed bound by rho = 0.999",
      float(_g[-1] > 100.0), 1.0, tol=0, rel=False)

prop('The surviving guarantee')
head("gamma = 1 - rho^2, from the definition rather than from the closed form")
from itertools import combinations as _comb


def _gam(d, Sig, K):
    _best = np.inf
    _idx = list(range(K))
    for _sS in range(0, 2):
        for _S in _comb(_idx, _sS):
            _rest = [i for i in _idx if i not in _S]
            _base = _inf(d, set(_S), Sig)
            for _sT in range(1, len(_rest) + 1):
                for _T in _comb(_rest, _sT):
                    _num = sum(_inf(d, set(_S) | {j}, Sig) - _base for j in _T)
                    _den = _inf(d, set(_S) | set(_T), Sig) - _base
                    if _den > 1e-12:
                        _best = min(_best, _num / _den)
    return _best


for _r in (0.0, 0.4, 0.7, 0.9, 0.98):
    _S2 = np.array([[1.0, _r], [_r, 1.0]])
    check("rho=%.2f: gamma" % _r, _gam(np.array([1.0, 0.0]), _S2, 2),
          1.0 - _r * _r, tol=1e-9)

head("and the guarantee it gives cannot exceed 1 - 1/e")
check("the bound at gamma = 1", 1.0 - np.exp(-1.0), 0.6321205588, tol=1e-8)
check("gamma never exceeds one over the sweep",
      float(max(_gam(np.array([1.0, 0.0]),
                     np.array([[1.0, _r], [_r, 1.0]]), 2)
                for _r in (0.0, 0.3, 0.6, 0.9))), 1.0, tol=1e-9, rel=False)

prop('A nonparametric trend leaves no clock')
# A plain smoother is used here rather than the out-of-fold one, since the claim
# is about absorption and the in-fold case is the stronger and cheaper form of
# it; the out-of-fold figures are in the results file.
from scipy.signal import savgol_filter as _sg

head("a local quadratic absorbs a global clock offset")
_rngN = np.random.default_rng(20260904)
_nN, _bN, _sN = 600, 3.0, 0.02
_tN = (np.arange(_nN) + 1.0) / _nN
_DpN = _bN * np.maximum(_tN, 1e-12) ** (_bN - 1)
_wN = max(7, int(0.08 * _nN))
_wN += (_wN % 2 == 0)
_means = {}
for _dl in (-0.02, 0.02):
    _acc = []
    for _ in range(200):
        _y = np.maximum(_tN + _dl, 1e-12) ** _bN + _rngN.normal(0, _sN, _nN)
        _r = _y - _sg(_y, _wN, 2)
        _acc.append(float(np.sum(_DpN * _r) / np.sum(_DpN * _DpN)))
    _means[_dl] = float(np.mean(_acc))
_slope = (_means[0.02] - _means[-0.02]) / 0.04
check("sensitivity of delta-hat to a real offset", abs(_slope), 0.0, tol=0.02,
      rel=False)

head("while against the true trend the same estimator is unbiased")
_acc = []
for _ in range(200):
    _dl = 0.02
    _y = np.maximum(_tN + _dl, 1e-12) ** _bN + _rngN.normal(0, _sN, _nN)
    _r = _y - _tN ** _bN
    _acc.append(float(np.sum(_DpN * _r) / np.sum(_DpN * _DpN)))
check("slope against the known trend", float(np.mean(_acc)) / 0.02, 1.0,
      tol=0.1)

prop('The identifiable part')
from pipeline import oof_trend

head("the smoother is linear, which is what makes (I-S)D' one evaluation")
# linearity is the load-bearing assumption: if S were not linear, (I-S)D' would
# not be computable as D' minus S applied to D'
_rngP = np.random.default_rng(31173)
_np2 = 500
_h2 = max(3, int(0.08 * _np2) // 2)
_a = _rngP.standard_normal(_np2)
_b = _rngP.standard_normal(_np2)
_c1, _c2 = 0.7, -1.3
_lhs = oof_trend(_c1 * _a + _c2 * _b, _h2)
_rhs = _c1 * oof_trend(_a, _h2) + _c2 * oof_trend(_b, _h2)
check("worst departure from linearity",
      float(np.max(np.abs(_lhs - _rhs))), 0.0, tol=1e-9, rel=False)

head("and kappa_S is near zero for a smooth derivative, as Proposition 27 needs")
_t3 = (np.arange(_np2) + 1.0) / _np2
for _bx, _lab in ((2.0, "quadratic trend"), (4.0, "quartic trend")):
    _dp = _bx * np.maximum(_t3, 1e-12) ** (_bx - 1)
    _sur = _dp - oof_trend(_dp, _h2)
    _k = float(np.sum(_sur ** 2) / np.sum(_dp ** 2))
    check("%s: kappa_S below 0.05" % _lab, float(min(_k, 0.05)), _k,
          tol=1e-12, rel=False)

prop('What a fleet recovers')
head("the invariance is exact: shifting D and every clock together changes nothing")
# This is the load-bearing claim and it is algebraic, so it is checked as an
# identity on the generated data rather than through any estimator.
_nF, _bF = 400, 3.0
_tF = (np.arange(_nF) + 1.0) / _nF
_rngF = np.random.default_rng(556677)
_dl = _rngF.normal(0, 0.02, 5)
_cF = 0.037
_worstF = 0.0
for _i in range(5):
    _a = np.maximum(_tF + _dl[_i], 1e-12) ** _bF
    # D shifted by c, clock shifted by -c: the same record
    _b = np.maximum((_tF + _cF) + (_dl[_i] - _cF), 1e-12) ** _bF
    _worstF = max(_worstF, float(np.max(np.abs(_a - _b))))
check("worst departure over five units", _worstF, 0.0, tol=1e-12, rel=False)

head("so N-1 combinations are identifiable and exactly one is not")
# the invariant direction is the all-ones vector; its orthogonal complement has
# dimension N-1, which is the count the proposition states
for _N in (2, 5, 12):
    _ones = np.ones((_N, 1))
    _rank = int(np.linalg.matrix_rank(np.eye(_N) - _ones @ _ones.T / _N))
    check("N=%d: dimension of the identifiable space" % _N, float(_rank),
          float(_N - 1), tol=0, rel=False)

prop('The price of flexibility')
head("projecting out the power-law tangent space recovers 1/(16 beta^4)")
# Proposition 10 was derived analytically; this reaches the same number by
# projection, which is a different route and so is evidence rather than an echo.
_nX = 4000
_tX = (np.arange(_nX) + 1.0) / _nX
for _bx in (2.3, 3.4, 5.7):
    _dpx = _bx * np.maximum(_tX, 1e-12) ** (_bx - 1)
    _B = np.column_stack([_tX ** _bx,
                          (_tX ** _bx) * np.log(np.maximum(_tX, 1e-12))])
    _co, *_ = np.linalg.lstsq(_B, _dpx, rcond=None)
    _res = _dpx - _B @ _co
    _kx = float(np.sum(_res ** 2) / np.sum(_dpx ** 2))
    check("beta=%.1f: projection against the closed form" % _bx, _kx,
          1.0 / (16.0 * _bx ** 4), tol=2e-4, rel=True)

head("and a richer basis can only remove more, whatever its shape")
# nesting: enlarging the basis cannot raise the surviving fraction
_dpx = 3.4 * np.maximum(_tX, 1e-12) ** 2.4
_prev = np.inf
_ok = True
for _deg in (2, 4, 8, 16):
    _B = np.column_stack([_tX ** _k for _k in range(_deg + 1)])
    _co, *_ = np.linalg.lstsq(_B, _dpx, rcond=None)
    _r = _dpx - _B @ _co
    _k2 = float(np.sum(_r ** 2) / np.sum(_dpx ** 2))
    _ok = _ok and (_k2 <= _prev + 1e-12)
    _prev = _k2
check("nested polynomial bases never raise kappa", float(_ok), 1.0, tol=0,
      rel=False)

prop('The estimand is a band')
head("a fast clock pattern survives the smoother where a constant does not")
# The projection form is enough here and avoids the estimator: what survives is
# (I - S) applied to the perturbation the clock produces.
_nB = 600
_tB = (np.arange(_nB) + 1.0) / _nB
_hB = max(3, int(0.08 * _nB) // 2)
_DpB = 3.0 * np.maximum(_tB, 1e-12) ** 2.0
_keep = {}
for _f in (0.0, 6.0, 25.0, 50.0):
    _pat = np.sin(2 * np.pi * _f * _tB) if _f > 0 else np.ones(_nB)
    _pert = _pat * _DpB
    _sur = _pert - oof_trend(_pert, _hB)
    _keep[_f] = float(np.sum(_sur ** 2) / np.sum(_pert ** 2))
check("a constant clock leaves almost nothing", float(min(_keep[0.0], 0.05)),
      _keep[0.0], tol=1e-12, rel=False)
check("a clock at f = 50 leaves order one of it",
      float(max(min(_keep[50.0], 1.5), 0.7)), _keep[50.0], tol=1e-12,
      rel=False)
# Monotonicity is NOT claimed and does not hold: the surviving fraction reads
# 0.000, 0.052, 1.378, 1.337, 1.130 at f = 6, 12, 25, 50, 100, overshooting one
# near the cutoff.  An out-of-fold prediction at high frequency is out of phase
# with the truth, so subtracting it ADDS rather than removes.  That amplification
# acts on the noise as well, which is why the efficiency of Proposition 31 --
# sensitivity squared over variance -- does not overshoot while this ratio does.
# The two are different quantities and only the efficiency supports the claim.
check("the ratio overshoots one above the cutoff, as a filter does",
      float(max(_keep[25.0], _keep[50.0]) > 1.0), 1.0, tol=0, rel=False)

# =============================================================================
# The manuscript claims this audit re-derives EVERY proposition.  That claim was
# prose until now: nothing checked that a proposition added to the paper had a
# block added here.  Comparing the titles covered against the titles in mssp.aux
# turns it into a check that fails when it stops being true.
_missing = sorted(set(PAPER_PROPS) - COVERED)
print("\n" + "=" * 92)
print("coverage: %d of the paper's %d propositions re-derived above,"
      " counting the manuscript and the supplement together"
      % (len(COVERED), len(PAPER_PROPS)))
if _missing:
    for t in _missing:
        FAIL.append("proposition %s, %r, has no check here"
                    % (PAPER_PROPS[t], t))
        print("   NOT COVERED: Proposition %s, %s" % (PAPER_PROPS[t], t))

print("\n" + "=" * 92)
if FAIL:
    print("%d CHECKS FAILED" % len(FAIL))
    for f in sorted(set(FAIL)):
        print("   " + f)
else:
    print("every proposition verified independently; no discrepancy found")
print("=" * 92)

# A layer that prints its verdict and exits zero cannot fail, and every
# runner that reads exit codes reports it as passing while it flags.
# audit_paper.py sat that way through a pass for sentence variety, with
# two checks reading sentences that had been rewritten around them.
import sys
sys.exit(1 if FAIL else 0)
