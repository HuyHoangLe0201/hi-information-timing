"""
Does the normalisation to F cost us something?

F(tau) = int_0^tau D'^2 / int_0^1 D'^2 is a shape: it says WHEN information
arrives but discards HOW MUCH. That was convenient for comparing indicators,
but it breaks as soon as two are combined, because informations add and shapes
do not.

Work instead with the unnormalised accumulation

    G(tau) := (f_s / sigma^2) int_0^tau D'(s)^2 ds  =  Gamma * F(tau),

the actual Fisher information the record carries up to tau. G lives on [0,1]
only: a record does not exist past failure, so G(1) = Gamma is the entire budget
and a target above it is unreachable, full stop. That boundary is what the q*
formulation hides inside the condition q* > 1.
"""
import numpy as np

INFEASIBLE = np.inf


def G_pow(tau, beta, Gamma):
    return Gamma * np.clip(np.asarray(tau, float), 0, 1) ** (2 * beta - 1)


def tau_min_single(beta, Gamma, eps):
    target = 1.0 / eps ** 2
    if Gamma < target:                      # whole record is not enough
        return INFEASIBLE
    return (target / Gamma) ** (1.0 / (2 * beta - 1))


def tau_min_fused(betas, Gammas, eps, m=200_001):
    target = 1.0 / eps ** 2
    if sum(Gammas) < target:
        return INFEASIBLE
    t = np.linspace(0.0, 1.0, m)
    tot = sum(G_pow(t, b, G) for b, G in zip(betas, Gammas))
    return float(t[int(np.searchsorted(tot, target))])


def fmt(x):
    return "  infeas." if not np.isfinite(x) else f"{x:>9.4f}"


print("claim 1 -- G is additive; F is not\n")
b1, b2, G1, G2 = 3.0, 0.6, 400.0, 90.0
print(f"{'tau':>6}{'G1(t)':>10}{'G2(t)':>10}{'sum':>10}"
      f"{'fused F':>11}{'mean of F1,F2':>15}")
for x in (0.2, 0.5, 0.8):
    g1, g2 = G_pow(x, b1, G1), G_pow(x, b2, G2)
    print(f"{x:>6.1f}{g1:>10.2f}{g2:>10.2f}{g1+g2:>10.2f}"
          f"{(g1+g2)/(G1+G2):>11.4f}"
          f"{0.5*(x**(2*b1-1) + x**(2*b2-1)):>15.4f}")
print("\n  the fused shape is a budget-weighted mixture, not an average of")
print("  shapes: normalising first destroys the weights that do the adding.\n")

print("claim 2 -- the free parameter q* is not needed\n")
print(f"{'beta':>6}{'Gamma':>8}{'eps':>7}{'q*':>8}{'via q*':>11}{'via G':>11}")
print("-" * 51)
for beta, Gam, eps in [(3.0, 400., 0.15), (0.6, 90., 0.20),
                       (1.0, 250., 0.10), (3.0, 400., 0.04)]:
    q = 1.0 / (eps ** 2 * Gam)
    tq = q ** (1.0 / (2 * beta - 1)) if q <= 1.0 else INFEASIBLE
    tg = tau_min_single(beta, Gam, eps)
    print(f"{beta:>6.1f}{Gam:>8.0f}{eps:>7.2f}{q:>8.3f}{fmt(tq)}{fmt(tg)}")
print("\n  identical, including the infeasible case, which under q* is the")
print("  side condition q* > 1 and under G is simply Gamma < 1/eps^2.\n")

print("claim 3 -- fusion can never delay the earliest usable age\n")
eps = 0.15
print(f"{'case':<32}{'alone A':>10}{'alone B':>10}{'fused':>10}{'gain':>9}")
print("-" * 71)
CASES = [
    ("late + early, similar budgets", (3.0, 400.), (0.6, 90.)),
    ("late dominates the budget",     (3.0, 4000.), (0.6, 90.)),
    ("early dominates the budget",    (3.0, 400.), (0.6, 900.)),
    ("two late indicators",           (3.0, 400.), (5.0, 400.)),
    ("second carries nothing",        (3.0, 400.), (0.6, 1e-6)),
    ("neither alone is feasible",     (3.0, 30.), (0.6, 25.)),
]
viol = 0
for lab, (ba, Ga), (bb, Gb) in CASES:
    ta, tb = tau_min_single(ba, Ga, eps), tau_min_single(bb, Gb, eps)
    tf = tau_min_fused([ba, bb], [Ga, Gb], eps)
    best = min(ta, tb)
    if np.isfinite(best) and tf > best + 1e-4:
        viol += 1
    gain = (best - tf) if np.isfinite(best) and np.isfinite(tf) else np.nan
    gs = "   --  " if not np.isfinite(gain) else f"{gain:>9.4f}"
    print(f"{lab:<32}{fmt(ta)}{fmt(tb)}{fmt(tf)}{gs}")
print("-" * 71)
print(f"cases where fusion delayed tau_min: {viol}")
print("\nStructural, not numerical: tau_min solves sum_j G_j(tau) = 1/eps^2 with")
print("every G_j non-negative and increasing, so an extra term can only move the")
print("crossing earlier. Note the last row -- two indicators each individually")
print("infeasible become feasible together, which the F formulation cannot even")
print("express, since both have F(1) = 1 by construction.")
