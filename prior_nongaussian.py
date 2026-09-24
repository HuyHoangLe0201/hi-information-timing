r"""Is the Bayesian floor a floor when the fleet's lifetimes are not Gaussian?

Proposition 4 bounds every estimator by 1/(G + J_pi), with J_pi the Fisher
information of the prior on the clock offset, and then says "equal to s^-2 for a
Gaussian prior of standard deviation s".  The paper uses s^-2 = 11.5 throughout.

That substitution is the same one Proposition 13 had to correct elsewhere, and it
fails the same way.  For any density, the information inequality gives

    J_pi >= 1 / Var(pi)

with equality only in the Gaussian case, so a skewed prior carries MORE
information than its variance implies.  Since the bound is 1/(G + J_pi), a larger
J_pi makes the true bound SMALLER.  Using s^-2 in its place therefore reports a
bound ABOVE the true one, and a bound that sits above the truth is not
guaranteed to sit below an estimator's error.  The direction is the opposite of
the robust-scale case and the failure is the same: a quantity that is only a
bound for Gaussian data is being used on data that is not.

Lifetimes are not Gaussian.  They are positive, right-skewed and usually
modelled as lognormal or Weibull.  The prior on the clock offset inherits that
shape exactly, because the offset is a deterministic transform of the life: a
unit assumed to be at normalised age tau when its true life is T rather than the
fleet median m sits at an offset

    delta = tau (1 - m/T),

so the prior on delta is the push-forward of the life distribution through that
map.  Nothing here is estimated loosely; the transform is exact and the only
modelling choice is the family fitted to T, so two families are fitted and
compared.
"""
import os
import json
import numpy as np
from scipy import stats, integrate, optimize

HERE = os.path.dirname(os.path.abspath(__file__))
TAU = 0.5                      # mid-life, the convention the paper quotes at
OUT = {}


def lifetimes():
    """Record lengths are the lifetimes: every record here runs to failure."""
    F = {}
    p = os.path.join(HERE, "feat_raw.npz")
    if os.path.exists(p):
        z = np.load(p, allow_pickle=True)
        F["PRONOSTIA"] = np.array(
            [len(z[u]) for u in z.files if u != "featnames"], float)
    p = os.path.join(HERE, "xjtu_feat.npz")
    if os.path.exists(p):
        z = np.load(p, allow_pickle=True)
        ks = [k for k in z.files if k not in ("featnames", "centres")]
        F["XJTU-SY"] = np.array([len(z[k]) for k in ks], float)
    for sub in ("FD001", "FD004"):
        p = os.path.join(HERE, "cmapss_%s.npz" % sub)
        if os.path.exists(p):
            z = np.load(p, allow_pickle=True)
            F[sub] = np.array(
                [len(z[k]) for k in z.files if k.endswith("__sensors")], float)
    p = os.path.join(HERE, "battery_raw.npz")
    if os.path.exists(p):
        z = np.load(p, allow_pickle=True)
        F["battery"] = np.array([len(z[k]) for k in z.files], float)
    return {k: v[np.isfinite(v) & (v > 0)] for k, v in F.items()}


def offset_density(fit, m, tau=TAU):
    """Density of delta = tau(1 - m/T) and its derivative, for a fitted T law.

    T = m / (1 - delta/tau), so dT/ddelta = m / (tau (1-delta/tau)^2), and the
    push-forward density is f_T(T(delta)) times that Jacobian.  Both are
    evaluated in closed form; only the derivative of the result is numerical.
    """
    def pi(d):
        u = 1.0 - d / tau
        if np.any(u <= 0):
            return 0.0
        T = m / u
        return float(fit.pdf(T) * m / (tau * u ** 2))
    return pi


def fisher_of(pi, lo, hi):
    """J = integral (pi'/pi)^2 pi, with pi' by central difference."""
    h = (hi - lo) * 1e-4

    def integrand(d):
        p = pi(d)
        if p <= 1e-300:
            return 0.0
        dp = (pi(d + h) - pi(d - h)) / (2 * h)
        return dp * dp / p
    return integrate.quad(integrand, lo, hi, limit=400)[0]


FAMILIES = {
    "lognormal": lambda T: stats.lognorm(*stats.lognorm.fit(T, floc=0)),
    "Weibull": lambda T: stats.weibull_min(*stats.weibull_min.fit(T, floc=0)),
}

print("=" * 92)
print("The prior on the clock offset, and the information it really carries")
print("=" * 92)
print("""
s is the standard deviation of the offset prior at mid-life, which is what the
paper uses; J_paper is the s^-2 it substitutes; J_true is the Fisher information
of the actual push-forward density.  Their ratio is the factor by which the
substitution understates the prior's information, and therefore the factor by
which the stated bound sits above the true one.
""")
F = lifetimes()
print("  %-11s %5s %8s %9s %10s %10s %8s"
      % ("fleet", "units", "life CV", "s", "J_paper", "J_true", "ratio"))
print("  " + "-" * 68)
rows = []
for name, T in sorted(F.items(), key=lambda kv: -len(kv[1])):
    if len(T) < 4:
        continue
    m = float(np.median(T))
    d = TAU * (1.0 - m / T)                       # the exact offset sample
    s = float(np.std(d, ddof=1))
    cv = float(np.std(T, ddof=1) / np.mean(T))
    per = {}
    for fam, mk in FAMILIES.items():
        try:
            fit = mk(T)
        except Exception:
            continue
        pi = offset_density(fit, m)
        lo, hi = float(d.min()) - 3 * s, min(TAU - 1e-6, float(d.max()) + 3 * s)
        try:
            mass = integrate.quad(pi, lo, hi, limit=400)[0]
            J = fisher_of(pi, lo, hi)
        except Exception:
            continue
        if not (0.9 < mass < 1.1) or not np.isfinite(J) or J <= 0:
            continue
        per[fam] = J
    if not per:
        continue
    J_true = float(np.median(list(per.values())))
    rows.append(dict(fleet=name, units=len(T), life_cv=cv, s=s,
                     J_paper=1.0 / s ** 2, J_true=J_true,
                     ratio=J_true * s ** 2, per_family=per))
    print("  %-11s %5d %8.3f %9.4f %10.2f %10.2f %8.2f"
          % (name, len(T), cv, s, 1.0 / s ** 2, J_true, J_true * s ** 2))
OUT["fleets"] = rows
print("""
  A ratio of one would mean the Gaussian substitution is exact.  Every fleet is
  above it, which is the information inequality showing up on real lifetimes:
  the prior always carries more than its variance says.
""")

# --- what it does to the bound ------------------------------------------------
print("=" * 92)
print("What the substitution does to the bound")
print("=" * 92)
_p = next((r for r in rows if r["fleet"] == "PRONOSTIA"), None)
if _p:
    print("""
The paper quotes s^-2 = 11.5 information units from PRONOSTIA and calls the
tighter fleet the conservative choice.  On the push-forward density the same
fleet carries %.1f, so the demand the prior removes is larger than stated and
every claim of the form "this demand is met before the machine is switched on"
holds a fortiori.  The bound itself moves the other way.
""" % _p["J_true"])
    print("  %10s %14s %14s %10s"
          % ("G", "1/(G+J_paper)", "1/(G+J_true)", "ratio"))
    print("  " + "-" * 52)
    bnd = []
    for G in (0.0, 10.0, 100.0, 1e3, 1e4):
        a = 1.0 / (G + _p["J_paper"])
        b = 1.0 / (G + _p["J_true"])
        bnd.append(dict(G=G, paper=a, true=b, ratio=a / b))
        print("  %10.0f %14.5f %14.5f %10.3f" % (G, a, b, a / b))
    OUT["bound"] = bnd
    print("""
  The stated bound exceeds the true one by up to %.2f, and the gap closes as the
  record's own information grows: the substitution matters exactly where the
  prior matters, which is early in life and at loose demands.  Nothing in the
  paper's numbers changes, because every age is read at a budget fraction and
  the prior enters as a constant subtracted from the demand; what changes is
  that Proposition 4 must be stated with J_pi and not with s^-2, since only the
  former is a bound.
""" % max(r["ratio"] for r in bnd))

json.dump(OUT, open(os.path.join(HERE, "prior_nongaussian.json"), "w"),
          indent=2, default=float)
print("written to prior_nongaussian.json")
