r"""What guarantee survives, and whether the rank guard already supplies it.

Proposition 25 refutes submodularity and so removes the 1 - 1/e bound from the
greedy rule this paper uses.  It does not leave nothing.  For a function that is
merely monotone there is the SUBMODULARITY RATIO

    gamma = min over S, T disjoint  [ sum_{j in T} Delta(j | S) ]
                                     / [ G(S union T) - G(S) ],

and greedy attains at least (1 - e^{-gamma}) of the optimum.  gamma is one when
the function is submodular and falls towards zero as synergy grows, so it is the
exact quantity that measures how far Proposition 25's counterexample is from the
channels actually used.  It is computable, so the guarantee becomes
data-dependent rather than absent.

There is also a tempting possibility worth testing rather than asserting.  The
failure mode of Proposition 25 is high correlation, and this paper already
imposes a rank guard that caps the condition number of the residual correlation
matrix at 1e3.  If that cap bounded gamma away from zero, the guard introduced
for numerical reasons would turn out to underwrite the greedy rule as well, which
would be a tidy result.  Whether it does is arithmetic and is checked below.
"""
import os
import json
import numpy as np
from itertools import combinations

HERE = os.path.dirname(os.path.abspath(__file__))
rng = np.random.default_rng(1414213)
OUT = {}


def info(d, S, Sig):
    if len(S) == 0:
        return 0.0
    i = np.array(sorted(S))
    return float(d[i] @ np.linalg.solve(Sig[np.ix_(i, i)], d[i]))


def gamma_of(d, Sig, maxS=3, maxT=3):
    """Submodularity ratio, minimised over sets up to the given sizes."""
    K = len(d)
    idx = list(range(K))
    g = np.inf
    for sS in range(0, maxS + 1):
        for S in combinations(idx, sS):
            rest = [i for i in idx if i not in S]
            base = info(d, set(S), Sig)
            for sT in range(1, min(maxT, len(rest)) + 1):
                for T in combinations(rest, sT):
                    num = sum(info(d, set(S) | {j}, Sig) - base for j in T)
                    den = info(d, set(S) | set(T), Sig) - base
                    if den > 1e-12:
                        g = min(g, num / den)
    return float(g)


print("=" * 92)
print("1.  The guarantee as a function of correlation")
print("=" * 92)
print("""
The two-channel case of Proposition 25 has gamma in closed form, so the bound can
be traced against the correlation that drives the synergy.  This is the exact
quantity the rank guard would have to control.
""")
print("  %8s %14s %16s %16s"
      % ("rho", "gamma", "1 - exp(-gamma)", "cond(C)"))
print("  " + "-" * 60)
rows = []
for rho in (0.0, 0.5, 0.8, 0.9, 0.95, 0.99, 0.998):
    S2 = np.array([[1.0, rho], [rho, 1.0]])
    d2 = np.array([1.0, 0.0])
    g = gamma_of(d2, S2, maxS=1, maxT=2)
    cond = (1 + rho) / (1 - rho) if rho < 1 else np.inf
    rows.append(dict(rho=rho, gamma=g, bound=float(1 - np.exp(-g)),
                     cond=float(cond)))
    print("  %8.3f %14.4f %16.4f %16.1f"
          % (rho, g, 1 - np.exp(-g), cond))
OUT["by_rho"] = rows
_closed = max(abs(r["gamma"] - (1 - r["rho"] ** 2)) for r in rows)
OUT["closed_form_error"] = float(_closed)
print("""
  The column is not a table but a closed form: gamma = 1 - rho^2, matched here to
  %.1e.  So the guarantee is 1 - exp(-(1 - rho^2)), it is one minus one over e at
  zero correlation and nothing else can exceed that, since gamma cannot exceed
  one.  The last row is the correlation the paper's own rank guard still admits,
  a two-channel correlation matrix at rho having condition number
  (1+rho)/(1-rho) against the guard's cap of 1e3.
""" % _closed)

# --- so: does the guard supply a useful bound? --------------------------------
CAP = 1e3
rho_max = (CAP - 1) / (CAP + 1)
S2 = np.array([[1.0, rho_max], [rho_max, 1.0]])
g_at_cap = gamma_of(np.array([1.0, 0.0]), S2, maxS=1, maxT=2)
OUT["cap"] = CAP
OUT["rho_at_cap"] = float(rho_max)
OUT["gamma_at_cap"] = float(g_at_cap)
OUT["bound_at_cap"] = float(1 - np.exp(-g_at_cap))
print("""  At the guard's cap of %.0e the admissible correlation is %.4f, giving
  gamma = %.4f and a guarantee of %.4f of the optimum.  That is not a useful
  bound: the guard was set to keep an inverse from exploding and is two orders of
  magnitude too loose to say anything about selection.
""" % (CAP, rho_max, g_at_cap, OUT["bound_at_cap"]))

# what cap would be needed for a stated guarantee.  The ceiling is 1 - 1/e at
# gamma = 1, so anything above it is unattainable at any correlation and must be
# reported as such rather than as a rho of zero.
CEIL = 1.0 - np.exp(-1.0)
OUT["ceiling"] = float(CEIL)
print("  %-22s %14s %14s %12s"
      % ("guarantee wanted", "rho allowed", "cond cap", "attainable?"))
print("  " + "-" * 66)
need = []
for target in (0.3, 0.5, 0.632, 0.8):
    if target > CEIL + 1e-9:
        need.append(dict(target=target, rho=None, cond=None,
                         attainable=False))
        print("  %-22.3f %14s %14s %12s"
              % (target, "-", "-", "NO"))
        continue
    lo, hi = 0.0, 0.999999
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        if 1 - np.exp(-(1 - mid * mid)) >= target:
            lo = mid
        else:
            hi = mid
    need.append(dict(target=target, rho=float(lo),
                     cond=float((1 + lo) / (1 - lo)), attainable=True))
    print("  %-22.3f %14.4f %14.1f %12s"
          % (target, lo, (1 + lo) / (1 - lo), "yes"))
OUT["needed"] = need
_half = next(r for r in need if abs(r["target"] - 0.5) < 1e-9)
print("""
  The ceiling is %.4f, reached only at zero correlation, so any guarantee above
  it is unattainable however the guard is set: this bound cannot beat what
  submodularity would have given, it can only approach it.  Even half the optimum
  requires the condition number capped at %.1f, against the paper's 1000.

  That is a concrete design consequence.  A guard chosen to keep an inverse from
  exploding does not double as a guard for selection; the two thresholds differ
  by more than two orders of magnitude, and the paper should not be read as
  having bought the second when it set the first.
""" % (CEIL, _half["cond"]))

json.dump(OUT, open(os.path.join(HERE, "submodularity_ratio.json"), "w"),
          indent=2, default=float)
print("written to submodularity_ratio.json")
