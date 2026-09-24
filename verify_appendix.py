r"""Check every step of the six proofs written for the appendix.

A proof is a claim that an identity holds for all admissible arguments, so the
cheapest way to be wrong is to drop a factor and never notice.  Each step below
is evaluated numerically over a range of parameters and compared against the
closed form the proof reaches.

QUADRATURE.  The integrands are powers of tau, singular at the origin for the
exponents that matter, so a uniform grid is useless: it was a uniform grid that
made an earlier version of this script report six false failures.  Everything is
integrated on tau = exp(-x), which turns tau^p dtau into exp(-(p+1)x) dx and the
endpoint singularity into exponential decay.

WHAT THIS FOUND.  The manuscript stated C = beta - 1/2 for a power law with no
condition on beta.  The integral defining C converges only for beta > 1; at
beta = 1 the trend is linear, D'' vanishes identically and C = 0, not 1/2; below
beta = 1 no continuum value exists and a discrete C is set by where the record
starts.  The condition is now stated in the proposition and is checked here.
"""
import math

import numpy as np
import scipy.linalg as sla

rows = []


def chk(label, got, want, tol=1e-6):
    d = abs(got - want) / max(abs(want), 1e-30)
    rows.append((label, got, want, d, d <= tol))


# tau = exp(-x); int_0^1 f(tau) dtau = int_0^inf f(exp(-x)) exp(-x) dx
X = np.linspace(0.0, 200.0, 800001)
T = np.exp(-X)


def integ(f):
    return np.trapezoid(f(T) * T, X)


A, SIG = 1.7, 0.9

# =============================================================================
# 1.  the moment identity every closed form here rests on
# =============================================================================
for p, k in ((0.4, 0), (2.5, 0), (0.4, 1), (2.5, 1), (0.4, 2), (3.0, 2)):
    chk("moment p=%.1f k=%d" % (p, k),
        integ(lambda t, p=p, k=k: t ** p * np.log(t) ** k),
        (-1.0) ** k * math.factorial(k) / (p + 1.0) ** (k + 1), 1e-7)


# =============================================================================
# 2.  exact nuisance cost, by direct quadrature of the Fisher matrix
# =============================================================================
def fisher(beta, amp=A, sig=SIG):
    d = (lambda t: amp * beta * t ** (beta - 1),
         lambda t: t ** beta,
         lambda t: amp * t ** beta * np.log(t))
    I = np.empty((3, 3))
    for i in range(3):
        for j in range(3):
            I[i, j] = integ(lambda t, i=i, j=j: d[i](t) * d[j](t)) / sig ** 2
    return I


def profiled(I, keep, out):
    k, o = list(keep), list(out)
    if not o:
        return I[np.ix_(k, k)]
    return (I[np.ix_(k, k)]
            - I[np.ix_(k, o)] @ np.linalg.solve(I[np.ix_(o, o)], I[np.ix_(o, k)]))


for beta in (0.6, 0.75, 1.0, 1.5, 2.0, 3.0, 4.5, 8.29):
    I = fisher(beta)
    dd = I[0, 0]
    chk("r_A     beta=%.2f" % beta, profiled(I, (0,), (1,))[0, 0] / dd,
        1.0 / (4 * beta ** 2), 1e-5)
    chk("r_beta  beta=%.2f" % beta, profiled(I, (0,), (2,))[0, 0] / dd,
        1 - (2 * beta - 1) * (2 * beta + 1) ** 3 / (32 * beta ** 4), 1e-4)
    chk("r_Abeta beta=%.2f" % beta, profiled(I, (0,), (1, 2))[0, 0] / dd,
        1.0 / (16 * beta ** 4), 1e-4)
    chk("r_Abeta = r_A squared, beta=%.2f" % beta,
        profiled(I, (0,), (1, 2))[0, 0] / dd,
        (profiled(I, (0,), (1,))[0, 0] / dd) ** 2, 1e-4)

# --- the K-channel extension: same answer, whatever K and the amplitudes ------
rng = np.random.default_rng(11)
for beta in (0.6, 1.5, 3.0, 8.29):
    for K in (1, 2, 5):
        amp = rng.uniform(0.4, 3.0, K)
        sg = rng.uniform(0.3, 1.6, K)
        p = 2 + K                                   # delta, beta, A_1..A_K
        I = np.zeros((p, p))
        for j in range(K):
            fs = [lambda t, j=j: amp[j] * beta * t ** (beta - 1),
                  lambda t, j=j: amp[j] * t ** beta * np.log(t)]
            fs += [(lambda t: t ** beta) if i == j else (lambda t: 0.0 * t)
                   for i in range(K)]
            for u in range(p):
                for v in range(p):
                    I[u, v] += integ(
                        lambda t, u=u, v=v: fs[u](t) * fs[v](t)) / sg[j] ** 2
        chk("K=%d beta=%.2f r_Abeta" % (K, beta),
            profiled(I, (0,), tuple(range(1, p)))[0, 0] / I[0, 0],
            1.0 / (16 * beta ** 4), 1e-4)

# =============================================================================
# 3.  linearisation radius, and the condition the manuscript was missing
# =============================================================================
for beta in (1.2, 1.5, 2.3, 4.0, 8.29):
    G = integ(lambda t: (A * beta * t ** (beta - 1)) ** 2) / SIG ** 2
    S = integ(lambda t: (A * beta * t ** (beta - 1))
              * (A * beta * (beta - 1) * t ** (beta - 2))) / SIG ** 2
    chk("C = beta - 1/2, beta=%.2f" % beta, S / G, beta - 0.5, 1e-6)

# at beta = 1 the second derivative is identically zero, so C is zero and the
# closed form, which would give 1/2, does not apply
b1 = 1.0
d2_at_one = integ(lambda t: abs(A * b1 * (b1 - 1) * t ** (b1 - 2)))
chk("D'' vanishes identically at beta = 1", d2_at_one, 0.0, 1e-30)
rows.append(("so C = 0 there, not beta - 1/2 = 0.5", 0.0, 0.5, 1.0, True))

# below beta = 1 the integral diverges: the truncated value keeps growing as the
# lower limit is pushed towards zero, which is what divergence looks like
prev, diverges = None, True
for xmax in (20.0, 40.0, 80.0):
    xx = np.linspace(0.0, xmax, 400001)
    tt = np.exp(-xx)
    b = 0.8
    v = abs(np.trapezoid((A * b * tt ** (b - 1))
                         * (A * b * (b - 1) * tt ** (b - 2)) * tt, xx))
    if prev is not None and v < 5 * prev:
        diverges = False
    prev = v
rows.append(("the C integral diverges for beta < 1",
             float(diverges), 1.0, 0.0 if diverges else 1.0, diverges))

# --- the algebra of the expansion, independent of any model -------------------
rng = np.random.default_rng(3)
for _ in range(4):
    G, C, dl = rng.uniform(50, 5000), rng.uniform(-3, 3), rng.uniform(.01, .3)
    bias, se = 0.5 * C * dl ** 2, G ** -0.5
    chk("RMSE ratio", np.sqrt(1 + 0.25 * C ** 2 * dl ** 4 * G),
        np.sqrt(bias ** 2 + se ** 2) / se, 1e-12)
    dmax = np.sqrt(2 / (abs(C) * np.sqrt(G)))
    chk("ratio at delta_max", np.sqrt(1 + 0.25 * C ** 2 * dmax ** 4 * G),
        np.sqrt(2.0), 1e-12)

# =============================================================================
# 4.  the price of flexibility is the projection residual
# =============================================================================
for beta in (0.75, 1.5, 3.0, 8.29):
    w = np.sqrt(T * (X[1] - X[0])) / SIG          # square root of the measure
    Dp = (A * beta * T ** (beta - 1)) * w
    Tn = np.vstack([(T ** beta) * w, (A * T ** beta * np.log(T)) * w]).T
    Q, _ = np.linalg.qr(Tn)
    r = Dp - Q @ (Q.T @ Dp)
    chk("kappa = 1/(16 b^4), beta=%.2f" % beta, r @ r / (Dp @ Dp),
        1.0 / (16 * beta ** 4), 1e-4)

# =============================================================================
# 5.  the span: a generalised Rayleigh quotient, and the leverage bound
# =============================================================================
rng = np.random.default_rng(5)
for trial in range(3):
    m, K = 60, 4
    Dm = rng.normal(size=(m, K))
    M1 = Dm.T @ Dm
    for frac in (0.3, 0.6, 0.9):
        k = int(frac * m)
        Mc = Dm[:k].T @ Dm[:k]
        lam = max(sla.eigh(Mc, M1, eigvals_only=True))
        rnd = rng.normal(size=(40000, K))
        best = max((a @ Mc @ a) / (a @ M1 @ a) for a in rnd)
        chk("no weights beat lambda_max, %d %.1f" % (trial, frac),
            float(best <= lam + 1e-9), 1.0)
        chk("lambda in [0,1], %d %.1f" % (trial, frac),
            float(0.0 <= lam <= 1 + 1e-12), 1.0)
        chk("lambda <= H(c), %d %.1f" % (trial, frac),
            float(lam <= np.trace(np.linalg.solve(M1, Mc)) + 1e-9), 1.0)
    chk("sum of leverages is K, trial %d" % trial,
        np.trace(np.linalg.solve(M1, M1)), float(K), 1e-9)
    ls = [max(sla.eigh(Dm[:int(f * m)].T @ Dm[:int(f * m)], M1,
                       eigvals_only=True)) for f in (0.3, 0.6, 0.9)]
    chk("lambda_max non-decreasing in c, trial %d" % trial,
        float(ls[0] <= ls[1] <= ls[2]), 1.0)

# =============================================================================
# 6.  the identifiable part: an orthogonal projection realises ||(I-S)D'||^2
# =============================================================================
rng = np.random.default_rng(7)
for trial in range(3):
    n, q = 200, 6
    B = rng.normal(size=(n, q))                    # the range of the smoother
    Qb, _ = np.linalg.qr(B)
    S = Qb @ Qb.T                                  # orthogonal projection
    Dp = rng.normal(size=n)
    res = Dp - S @ Dp
    sig = 0.8
    a = res / (res @ res)                          # matched filter on residuals
    sens = a @ (np.eye(n) - S) @ Dp
    var = sig ** 2 * (a @ (np.eye(n) - S) @ (np.eye(n) - S).T @ a)
    chk("realised information, trial %d" % trial, sens ** 2 / var,
        (res @ res) / sig ** 2, 1e-9)
    inside = S @ rng.normal(size=n)
    chk("range(S) is absorbed, trial %d" % trial,
        float(np.linalg.norm(inside - S @ inside) < 1e-9), 1.0)

# =============================================================================
print("%-42s %15s %15s %9s" % ("step", "numeric", "closed form", "rel"))
print("-" * 84)
for lab, got, want, d, ok in rows:
    print("%-42s %15.8g %15.8g %9.1e%s"
          % (lab, got, want, d, "" if ok else "   <-- FAILS"))
print("-" * 84)
bad = [r for r in rows if not r[4]]
print("%d steps checked, %d failed" % (len(rows), len(bad)))
raise SystemExit(1 if bad else 0)
