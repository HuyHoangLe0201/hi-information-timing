"""
Further structure in F. Three candidate results, each checked numerically.

(A) F IS the censoring efficiency of the parent paper.
    [1, Thm. 4] defines eta(c,D) = fraction of cumulative SNR retained when a
    TRAINING trajectory is observed only on [0,c]. Its closed forms should be
    exactly F(c). If so, one function governs both papers: the training-side
    window is anchored at 0 (eta = F(c) - F(0)) and the deployment-side window
    at tau0 (rho_w = F(tau0) - F(tau0-w)). The Letter's Remark 2 currently says
    this in words; it would become an identity.

(B) Is a TRAILING buffer even the right architecture?
    Proposition 1 gives the information of a retained set S as proportional to
    int_S D'^2. For a retention budget |S| = w drawn from [0,tau0], the optimal
    S is the super-level set of the information density (bathtub principle).
    D'^2 is monotone for all three classes, so the optimum is always an END
    interval -- either the most recent w (FIFO buffer) or the earliest w (fixed
    archive). Which one is decided by the sign of the trend in D'^2.

(C) Under the phase model of Prop. 1 the density is D'^2; if instead the
    uncertainty is in the lifetime scale T, [1, Prop. 1] gives tau^2 D'^2.
    Reported here as a robustness check, since it can move the optimum inside.
"""
import numpy as np
from scipy.special import gammainc

ALPHA, BETA, ALPHA_B, B_STR = 2.0, 3.0, 3.0, 0.7
GRID = np.linspace(1e-6, 1.0, 200001)


def dens_exp(t):
    return np.exp(-2 * ALPHA * t)


def dens_pow(t):
    return t ** (2 * BETA - 2)


def dens_str(t):
    return t ** (2 * B_STR - 2) * np.exp(-2 * ALPHA_B * t ** B_STR)


def dens_lin(t):
    return np.ones_like(t)


CLASSES = [("linear", dens_lin), ("exponential", dens_exp),
           ("power-law", dens_pow), ("stretched-exp", dens_str)]


def F_numeric(dens, tau):
    m = GRID <= tau
    return np.trapezoid(dens(GRID[m]), GRID[m]) / np.trapezoid(dens(GRID), GRID)


# ------------------------------------------------------------------ (A)
print("=== (A) F(c) vs the censoring efficiency eta(c) of the parent paper ===")
print("    [1, Thm. 4]: eta_exp=(1-e^-2ac)/(1-e^-2a), eta_pow=c^(2b-1),")
print("    eta_str=int_0^c tau^2(b-1) e^-2ab tau^b / int_0^1, eta_lin=c\n")
print(f"{'c':>6}" + "".join(f"{n:>16}" for n, _ in CLASSES))
print(f"{'':>6}" + "".join(f"{'|F - eta|':>16}" for _ in CLASSES))
print("-" * 70)
for c in [0.2, 0.4, 0.6, 0.8, 1.0]:
    row = f"{c:>6.1f}"
    for name, dens in CLASSES:
        F = F_numeric(dens, c)
        if name == "exponential":
            eta = (1 - np.exp(-2 * ALPHA * c)) / (1 - np.exp(-2 * ALPHA))
        elif name == "power-law":
            eta = c ** (2 * BETA - 1)
        elif name == "linear":
            eta = c
        else:
            s = 2 - 1.0 / B_STR
            eta = gammainc(s, 2 * ALPHA_B * c ** B_STR) / gammainc(s, 2 * ALPHA_B)
        row += f"{abs(F - eta):>16.2e}"
    print(row)
print("\n  -> identical: eta(c) = F(c). The censoring efficiency of [1] and the")
print("     windowing efficiency here are the same measure, differently anchored.")


# ------------------------------------------------------------------ (B)
def mass(dens, lo, hi):
    m = (GRID >= lo) & (GRID <= hi)
    return np.trapezoid(dens(GRID[m]), GRID[m]) / np.trapezoid(dens(GRID), GRID)


def best_mass(dens, tau0, w):
    """Bathtub: the w-measure subset of [0,tau0] carrying the most information."""
    m = GRID <= tau0
    x, d = GRID[m], dens(GRID[m])
    order = np.argsort(d)[::-1]
    dx = x[1] - x[0]
    keep = order[:max(1, int(round(w / dx)))]
    return float(np.sum(d[keep]) * dx / np.trapezoid(dens(GRID), GRID))


print("\n=== (B) trailing buffer vs earliest archive vs unconstrained optimum ===")
print("    information captured for a retention budget w, at tau0 = 0.7\n")
tau0 = 0.7
print(f"{'class':<16}{'w':>6}{'trailing':>11}{'leading':>10}{'optimal':>10}"
      f"{'lead/trail':>12}{'monotone D^2':>15}")
print("-" * 80)
for name, dens in CLASSES:
    trend = "increasing" if dens(np.array([0.65]))[0] > dens(np.array([0.05]))[0] \
        else ("flat" if abs(dens(np.array([0.65]))[0] - dens(np.array([0.05]))[0]) < 1e-12
              else "decreasing")
    for w in [0.2, 0.3, 0.5]:
        tr = mass(dens, tau0 - w, tau0)
        ld = mass(dens, 0.0, w)
        op = best_mass(dens, tau0, w)
        print(f"{name:<16}{w:>6.2f}{tr:>11.4f}{ld:>10.4f}{op:>10.4f}"
              f"{ld / tr:>12.2f}{trend:>15}")
    print()
print("  -> the optimum coincides with whichever END interval the density favours,")
print("     confirming the bathtub solution is an end interval whenever D'^2 is")
print("     monotone -- which it is for all three classes.")


# ------------------------------------------------------------------ (C)
print("\n=== (C) robustness: if the uncertainty is in the lifetime scale T ===")
print("    [1, Prop. 1] then gives density tau^2 D'^2 instead of D'^2\n")
print(f"{'class':<16}{'argmax of tau^2 D^2':>22}{'implication':>28}")
print("-" * 68)
for name, dens in CLASSES:
    d2 = GRID ** 2 * dens(GRID)
    am = GRID[int(np.argmax(d2))]
    if am > 0.995:
        imp = "trailing still optimal"
    elif am < 0.005:
        imp = "leading still optimal"
    else:
        imp = "interior optimum"
    print(f"{name:<16}{am:>22.3f}{imp:>28}")
print("\n  -> under a scale (rather than phase) uncertainty the exponential and")
print("     stretched-exponential optima move inside; the phase model of Prop. 1")
print("     is the one under which the end-interval statement holds.")
