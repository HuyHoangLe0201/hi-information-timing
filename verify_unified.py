"""
Check the proposed reformulation against the formulas currently in the Letter.

Claim: every row of Table I, both closed forms of Proposition 2, the
infeasibility condition and the saturation ceiling are increments and inverses
of one increasing function

    F(tau) := int_0^tau D'(s)^2 ds / int_0^1 D'(s)^2 ds        (F(0)=0, F(1)=1)

the normalized cumulative Fisher information, so that

    rho_w(tau0)  = F(tau0) - F((tau0-w)_+)
    rho_w        <= F(tau0)                      (ceiling at w = tau0)
    w*(tau0)     = tau0 - Finv((F(tau0)-q*)_+)   (one formula, all classes)
    feasible     <=> F(tau0) >= q*
    tau_min      = Finv(q*)

Nothing here is asserted analytically: F and its inverse are compared against
the existing expressions, and against brute-force numerical integration of
D'^2 straight from the degradation functions D themselves.
"""
import numpy as np
from scipy.special import gammainc, gammaincinv
from scipy.integrate import quad
from scipy.optimize import brentq

ALPHA, BETA, ALPHA_B, B_STR = 2.0, 3.0, 3.0, 0.7


# ----------------------------------------------- the degradation functions D
def D_exp(t, a=ALPHA):
    return (1 - np.exp(-a * t)) / (1 - np.exp(-a))


def D_pow(t, b=BETA):
    return t ** b


def D_str(t, ab=ALPHA_B, b=B_STR):
    return (1 - np.exp(-ab * t ** b)) / (1 - np.exp(-ab))


def D_lin(t):
    return t


def dD(D, t, h=1e-6):
    t = min(max(t, h), 1 - h)
    return (D(t + h) - D(t - h)) / (2 * h)


def F_numeric(D, tau):
    """Brute force: normalized cumulative integral of D'^2, no closed form used."""
    num = quad(lambda s: dD(D, s) ** 2, 0, tau, limit=200)[0]
    den = quad(lambda s: dD(D, s) ** 2, 0, 1, limit=200)[0]
    return num / den


# ------------------------------------------------- proposed closed forms of F
def F_exp(t, a=ALPHA):
    return (1 - np.exp(-2 * a * t)) / (1 - np.exp(-2 * a))


def F_pow(t, b=BETA):
    return t ** (2 * b - 1)


def F_str(t, ab=ALPHA_B, b=B_STR):
    s = 2 - 1.0 / b
    return gammainc(s, 2 * ab * t ** b) / gammainc(s, 2 * ab)


def F_lin(t):
    return t


def Finv_exp(u, a=ALPHA):
    return -np.log(1 - u * (1 - np.exp(-2 * a))) / (2 * a)


def Finv_pow(u, b=BETA):
    return u ** (1.0 / (2 * b - 1))


def Finv_str(u, ab=ALPHA_B, b=B_STR):
    s = 2 - 1.0 / b
    return (gammaincinv(s, u * gammainc(s, 2 * ab)) / (2 * ab)) ** (1.0 / b)


# --------------------------------- the expressions currently printed in the Letter
def rho_exp_letter(t0, w, a=ALPHA):
    lo = max(t0 - w, 0.0)
    return (np.exp(-2 * a * lo) - np.exp(-2 * a * t0)) / (1 - np.exp(-2 * a))


def rho_pow_letter(t0, w, b=BETA):
    lo = max(t0 - w, 0.0)
    return t0 ** (2 * b - 1) - lo ** (2 * b - 1)


def rho_str_letter(t0, w, ab=ALPHA_B, b=B_STR):
    s = 2 - 1.0 / b
    lo = max(t0 - w, 0.0)
    return (gammainc(s, 2 * ab * t0 ** b) - gammainc(s, 2 * ab * lo ** b)) \
        / gammainc(s, 2 * ab)


def wstar_exp_letter(t0, q, a=ALPHA):
    return np.log(1 + q * np.exp(2 * a * t0) * (1 - np.exp(-2 * a))) / (2 * a)


def wstar_pow_letter(t0, q, b=BETA):
    tgt = t0 ** (2 * b - 1) - q
    return np.nan if tgt < 0 else t0 - tgt ** (1.0 / (2 * b - 1))


CLASSES = [
    ("exponential", D_exp, F_exp, Finv_exp, rho_exp_letter, wstar_exp_letter),
    ("power-law", D_pow, F_pow, Finv_pow, rho_pow_letter, wstar_pow_letter),
    ("stretched-exp", D_str, F_str, Finv_str, rho_str_letter, None),
    ("linear", D_lin, F_lin, lambda u: u, None, None),
]

print("=== 1. proposed F(tau) vs brute-force integration of D'^2 ===")
for name, D, F, _, _, _ in CLASSES:
    err = max(abs(F(t) - F_numeric(D, t)) for t in np.linspace(0.05, 1.0, 20))
    print(f"  {name:<15} max |F_closed - F_numeric| = {err:.3e}")

print("\n=== 2. rho_w = F(tau0) - F((tau0-w)_+) vs the Table I entries ===")
for name, D, F, _, rho_letter, _ in CLASSES:
    if rho_letter is None:
        continue
    err = 0.0
    for t0 in np.linspace(0.2, 1.0, 9):
        for w in np.linspace(0.01, t0, 12):
            unified = F(t0) - F(max(t0 - w, 0.0))
            err = max(err, abs(unified - rho_letter(t0, w)))
    print(f"  {name:<15} max |unified - Table I| = {err:.3e}")

print("\n=== 3. saturation ceiling: rho_w(tau0) <= F(tau0), equality at w=tau0 ===")
for name, D, F, _, rho_letter, _ in CLASSES:
    if rho_letter is None:
        continue
    gap = max(abs(rho_letter(t0, t0) - F(t0)) for t0 in np.linspace(0.2, 1.0, 9))
    over = max(rho_letter(t0, w) - F(t0)
               for t0 in np.linspace(0.2, 1.0, 9)
               for w in np.linspace(0.01, t0, 12))
    print(f"  {name:<15} |rho(w=tau0) - F(tau0)| = {gap:.3e}, "
          f"max excess over ceiling = {over:.3e}")

print("\n=== 4. w* = tau0 - Finv((F(tau0)-q*)_+) vs Prop. 2 closed forms ===")
for q in [0.15, 0.35, 0.50]:
    for name, D, F, Finv, _, w_letter in CLASSES:
        if w_letter is None:
            continue
        errs, infeas = [], 0
        for t0 in np.linspace(0.2, 1.0, 17):
            tgt = F(t0) - q
            if tgt < 0:
                infeas += 1
                if not np.isnan(w_letter(t0, q)) and name == "power-law":
                    print(f"    MISMATCH: unified infeasible but Letter feasible")
                continue
            unified = t0 - Finv(tgt)
            errs.append(abs(unified - w_letter(t0, q)))
        print(f"  q*={q:.2f} {name:<15} max diff = {max(errs) if errs else float('nan'):.3e} "
              f"({infeas} infeasible ages agree)")

print("\n=== 5. w* by direct root-finding on rho_w (no closed form at all) ===")
for name, D, F, Finv, _, _ in CLASSES:
    errs = []
    for q in [0.15, 0.35]:
        for t0 in np.linspace(0.3, 1.0, 8):
            if F(t0) < q:
                continue
            g = lambda w: (F(t0) - F(max(t0 - w, 0.0))) - q
            root = brentq(g, 1e-9, t0)
            errs.append(abs(root - (t0 - Finv(F(t0) - q))))
    print(f"  {name:<15} max |root-find - Finv formula| = {max(errs):.3e}")

print("\n=== 6. tau_min = Finv(q*): earliest age at which the target is reachable ===")
for q in [0.15, 0.35, 0.50]:
    row = []
    for name, D, F, Finv, _, _ in CLASSES:
        tmin = Finv(q)
        ok = abs(F(tmin) - q) < 1e-9
        row.append(f"{name}={tmin:.3f}{'' if ok else ' BAD'}")
    print(f"  q*={q:.2f}: " + ", ".join(row))
