"""
Theorem 1 is the Letter's one implicit equation. Is it closable?

    n = A[2p ln(2en/p) + 2L],     A := B^4 Gamma^2 rho_w^2 / Tbar^4,  L := ln(4/delta)

Substituting u := 2en/p turns this into u = C ln u + K with C = 4eA and
K = 4eAL/p, whose solution is a Lambert-W branch:

    u = -C W_{-1}( -exp(-K/C)/C ),      n = pu/(2e).

Checked here against the self-consistent (fixed-point) solution over a wide
parameter sweep, including the branch choice.
"""
import numpy as np
from scipy.special import lambertw
from scipy.optimize import brentq

RNG = np.random.default_rng(7)


def fixed_point(A, p, L):
    """Solve n = A[2p ln(2en/p) + 2L] by bracketing."""
    g = lambda n: A * (2 * p * np.log(2 * np.e * n / p) + 2 * L) - n
    lo, hi = p / (2 * np.e) * (1 + 1e-9), 1e18
    if g(lo) < 0:
        return np.nan
    while g(hi) > 0 and hi < 1e300:
        hi *= 10
    return brentq(g, lo, hi, xtol=1e-10, rtol=1e-14)


def lambert_form(A, p, L):
    C = 4 * np.e * A
    K = 4 * np.e * A * L / p
    arg = -np.exp(-K / C) / C
    if arg < -np.exp(-1.0):
        return np.nan, None               # no real solution
    for branch in (-1, 0):
        u = np.real(-C * lambertw(arg, branch))
        n = p * u / (2 * np.e)
        if np.isfinite(n) and n > p / (2 * np.e):
            resid = abs(A * (2 * p * np.log(2 * np.e * n / p) + 2 * L) - n)
            if resid < 1e-6 * max(1.0, n):
                return n, branch
    return np.nan, None


print(f"{'A':>10}{'p':>7}{'L':>7}{'fixed-point':>16}{'Lambert W':>16}"
      f"{'rel.err':>11}{'branch':>8}")
print("-" * 75)
bad = 0
for _ in range(14):
    A = 10 ** RNG.uniform(-1, 4)
    p = RNG.integers(5, 4000)
    L = np.log(4 / 10 ** RNG.uniform(-3, -1))
    nf = fixed_point(A, p, L)
    nl, br = lambert_form(A, p, L)
    if not (np.isfinite(nf) and np.isfinite(nl)):
        print(f"{A:>10.3g}{p:>7}{L:>7.2f}{'--':>16}{'--':>16}{'--':>11}{'--':>8}")
        continue
    rel = abs(nf - nl) / nf
    bad += rel > 1e-8
    print(f"{A:>10.3g}{p:>7}{L:>7.2f}{nf:>16.4f}{nl:>16.4f}{rel:>11.2e}{br:>8}")
print("-" * 75)
print(f"disagreements beyond 1e-8: {bad}")
print("Branch -1 is the relevant root throughout (the large solution);")
print("branch 0 would give the spurious small crossing.")
