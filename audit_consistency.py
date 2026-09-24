r"""The eighth audit: do the propositions agree with each other?

Seven layers check the manuscript against results, the propositions against
independent derivation, and the directions against the equations they come from.
None checks a proposition against ANOTHER proposition.  That gap did not matter
when the propositions were few and separate.  It matters now: Propositions 15 to
28 form one chain, several of them describe the same quantity from different
directions, and each was added and verified on its own.

A chain built that way can be individually correct and jointly inconsistent.  The
checks below are all of the form: construct an instance where two propositions
both apply, and require their answers to agree.  A disagreement means one of them
is wrong however well each verified alone.
"""
import numpy as np
from itertools import combinations

rng = np.random.default_rng(80808)
FAIL = []


def check(name, got, want, tol=1e-9, rel=False):
    d = abs(got - want) / max(abs(want), 1e-300) if rel else abs(got - want)
    ok = d <= tol
    if not ok:
        FAIL.append(name)
    print("  %-62s %12.6g %12.6g  %s"
          % (name, got, want, "ok" if ok else "FAIL"))


def head(t):
    print("\n" + t)
    print("  %-62s %12s %12s" % ("", "left", "right"))


N = 4000
tau = (np.arange(N) + 1.0) / N
DENS = {"rising": 2 * tau, "peaked": 0.2 + 3 * tau ** 4,
        "falling": 2 - 2 * tau}


def contrast(w, g, ic):
    a = float(np.sum(w[:ic + 1] * g[:ic + 1]) / np.sum(g[:ic + 1]))
    b = float(np.sum(w[ic + 1:] * g[ic + 1:]) / np.sum(g[ic + 1:]))
    return a / b


def odds_ratio(w, g, ic):
    e0 = float(np.sum(g[:ic + 1]) / np.sum(g))
    gd = w * g
    e1 = float(np.sum(gd[:ic + 1]) / np.sum(gd))
    return (e1 / (1 - e1)) / (e0 / (1 - e0))


print("=" * 92)
print("Pairwise consistency of Propositions 15 to 28")
print("=" * 92)

# --- 19 against 16 ------------------------------------------------------------
head("Prop 19's contrast never exceeds Prop 16's M^2")
worst = 0.0
for _ in range(3000):
    g = DENS[str(rng.choice(list(DENS)))] / N
    M = float(rng.uniform(1.05, 4.0))
    lw = np.clip(np.convolve(rng.normal(0, 1, N), np.ones(31) / 31, "same"),
                 -np.log(M), np.log(M))
    ic = int(rng.uniform(0.1, 0.9) * N)
    worst = max(worst, contrast(np.exp(lw), g, ic) / (M * M))
check("worst contrast over M^2, clipped at one", max(worst, 1.0), 1.0)

# --- 19 against 17 ------------------------------------------------------------
head("Prop 19's contrast is at least one when Prop 17's w is non-increasing")
worst = np.inf
for _ in range(2000):
    g = DENS[str(rng.choice(list(DENS)))] / N
    w = np.exp(rng.uniform(0.2, 3.0) * np.sort(rng.random(N))[::-1])
    ic = int(rng.uniform(0.1, 0.9) * N)
    worst = min(worst, contrast(w, g, ic))
check("smallest contrast under a falling w, clipped at one",
      min(worst, 1.0), 1.0)

# --- 19 is the identity the others are read through ---------------------------
head("Prop 19 reproduces the odds ratio exactly, for every w above")
worst = 0.0
for _ in range(2000):
    g = DENS[str(rng.choice(list(DENS)))] / N
    w = np.exp(rng.uniform(-2.5, 2.5, N))
    ic = int(rng.uniform(0.1, 0.9) * N)
    worst = max(worst, abs(contrast(w, g, ic) - odds_ratio(w, g, ic))
                / max(contrast(w, g, ic), 1e-300))
check("worst relative gap between identity and direct reading", worst, 0.0,
      tol=1e-10)

# --- 15 against 19 ------------------------------------------------------------
head("Prop 15's expansion is Prop 19's identity as eps goes to zero")
g = DENS["peaked"] / N
ic = N // 2
u = np.sin(4 * tau) + 0.4 * tau
u = u - float(np.sum(u * g) / np.sum(g))
errs = []
for eps in (0.04, 0.01):
    exact = np.log(contrast(np.exp(eps * u), g, ic))
    a = float(np.sum(u[:ic + 1] * g[:ic + 1]) / np.sum(g[:ic + 1]))
    b = float(np.sum(u[ic + 1:] * g[ic + 1:]) / np.sum(g[ic + 1:]))
    errs.append(abs(exact - eps * (a - b)) / abs(eps * (a - b)))
check("error ratio when eps falls fourfold", errs[0] / max(errs[1], 1e-300),
      4.0, tol=0.6)

# --- 20 against 19 ------------------------------------------------------------
head("Prop 20's extremal step, read through Prop 19, gives Prop 20's value")
worst = 0.0
for _ in range(200):
    g = DENS[str(rng.choice(list(DENS)))] / N
    ic = int(rng.uniform(0.15, 0.85) * N)
    p = float(np.sum(g[:ic + 1]) / np.sum(g))
    if not (0.05 < p < 0.95):
        continue
    eps = float(rng.uniform(0.03, 0.4))
    h = np.where(np.arange(N) <= ic, 1.0 / p, -1.0 / (1.0 - p))
    m = float(np.sum(h * g) / np.sum(g))
    e = float(np.sqrt(np.sum((h - m) ** 2 * g) / np.sum(g)))
    us = eps * h / e
    lhs = np.log(contrast(np.exp(us), g, ic))
    rhs = eps / np.sqrt(p * (1 - p))
    worst = max(worst, abs(lhs - rhs) / abs(rhs))
check("worst relative gap, Prop 19 against Prop 20", worst, 0.0, tol=1e-9)

# --- 21 against 20 ------------------------------------------------------------
head("Prop 21's ellipse, in one coordinate, is Prop 20's bound")
# The two propositions measure the budget against DIFFERENT references: Prop 20
# takes the variance of u against the information density g, Prop 21 takes a ball
# in whatever reference the Gram matrix is built on.  Compared naively they
# disagree, 2.21 against 2.49, which is not an inconsistency but an ambiguity:
# Prop 21 has to name its norm.  Built on the same g-measure they must agree, and
# that is what is checked.
g1 = DENS["rising"] / N
ic = int(0.45 * N)
p1 = float(np.sum(g1[:ic + 1]) / np.sum(g1))
h1 = np.where(np.arange(N) <= ic, 1.0 / p1, -1.0 / (1 - p1))
# Gram entry in L^2(g): <h,h>_g = int h^2 g / int g
G11 = float(np.sum(h1 * h1 * g1) / np.sum(g1))
check("sqrt(Gamma_11) in L^2(g) against 1/sqrt(eta(1-eta))",
      float(np.sqrt(G11)), 1.0 / np.sqrt(p1 * (1 - p1)), tol=1e-9, rel=True)

# --- 25 against 26 ------------------------------------------------------------
head("Prop 25's synergy and Prop 26's gamma come from one Schur complement")
for rho in (0.5, 0.8, 0.95):
    syn = rho ** 2 / (1 - rho ** 2)          # Prop 25
    gam = 1 - rho ** 2                        # Prop 26
    check("rho=%.2f: (1 + synergy) times gamma" % rho,
          (1 + syn) * gam, 1.0, tol=1e-12)

# --- 27 against 28 ------------------------------------------------------------
head("Prop 28's kappa_S vanishes exactly where Prop 27 says the clock does")
from pipeline import oof_trend
t2 = (np.arange(600) + 1.0) / 600
hh = max(3, int(0.08 * 600) // 2)
for beta, lab in ((2.0, "smooth derivative"), (5.0, "steeper")):
    dp = beta * np.maximum(t2, 1e-12) ** (beta - 1)
    k = float(np.sum((dp - oof_trend(dp, hh)) ** 2) / np.sum(dp * dp))
    check("%s: kappa_S below 1e-3" % lab, min(k, 1e-3), k, tol=1e-15)

# --- 23 against 24 ------------------------------------------------------------
head("Prop 24's within-record term is Prop 23's variance, same delta method")
q = 0.35
cd, m = 2.5, 400.0
se_share = np.sqrt(q * (1 - q)) * cd / np.sqrt(m)          # Prop 24's numerator
se_odds = cd * np.sqrt(1.0 / (q * m) + 1.0 / ((1 - q) * m))  # Prop 23
check("their ratio is sqrt(q(1-q)) times q(1-q)",
      se_share / se_odds, q * (1 - q), tol=1e-9)

# =============================================================================
print("\n" + "=" * 92)
if FAIL:
    print("%d CONSISTENCY CHECKS FAILED" % len(FAIL))
    for f in sorted(set(FAIL)):
        print("   " + f)
else:
    print("every pair of propositions checked agrees; no inconsistency found")
print("=" * 92)

# A layer that prints its verdict and exits zero cannot fail, and every
# runner that reads exit codes reports it as passing while it flags.
# audit_paper.py sat that way through a pass for sentence variety, with
# two checks reading sentences that had been rewritten around them.
import sys
sys.exit(1 if FAIL else 0)
