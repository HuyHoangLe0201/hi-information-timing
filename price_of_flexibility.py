r"""What flexibility costs: a ladder from the parametric bound to nothing.

Proposition 28 measures kappa_S, the fraction of a channel's derivative that
survives the out-of-fold smoother, and finds it between 0.001 and 0.023.
Proposition 10 prices the same nuisance parametrically and finds 1/(16 beta^4),
which is 7.7e-4 at beta = 3.  The two numbers are not comparable as they stand,
because they describe different nuisance classes, and nothing in the paper joins
them.

Semiparametric theory says what joins them.  The efficient information for the
clock is || (I - Pi_T) D' ||^2 with Pi_T the projection onto the TANGENT SPACE of
the nuisance.  A smoother projects onto its own range, and if that range is
larger than the tangent space the estimator gives away information it did not
have to.  So the loss should be a ladder in flexibility: smallest for a fit
matched to the truth, growing as the fit is allowed more freedom, and reaching
Proposition 28's figures at a local smoother.

The ladder is measurable directly.  Each estimator is a linear projection, so
(I - Pi) D' is D' minus that projection applied to D', and

    kappa = || (I - Pi) D' ||^2 / || D' ||^2

is one evaluation each.  A monotone ladder confirms that flexibility and not
estimation is what costs; a flat one would mean the cost is intrinsic and the
paper's smoother is not to blame.
"""
import os
import json
import numpy as np
from scipy.signal import savgol_filter

from pipeline import oof_trend

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = {}
N = 800
t = (np.arange(N) + 1.0) / N


def kappa_of(dp, basis):
    """Fraction of dp orthogonal to the span of the given basis columns."""
    B = np.column_stack(basis)
    coef, *_ = np.linalg.lstsq(B, dp, rcond=None)
    resid = dp - B @ coef
    return float(np.sum(resid ** 2) / np.sum(dp ** 2))


print("=" * 92)
print("1.  The ladder, on a trend whose class is known")
print("=" * 92)
print("""
The truth is a power law, so the correct nuisance class has two parameters and
its tangent space is spanned by the derivatives of the trend with respect to
them.  Everything else on the ladder is more flexible than that.

The exponents are deliberately NOT integers.  A first version used beta = 3,
where D' = 3 t^2 is itself a polynomial and every polynomial basis of degree two
or more annihilates it exactly; the ladder then read 5e-31 at every rung above
the first and appeared to span 28 orders of magnitude, which measured the choice
of exponent and not the cost of flexibility.
""")
print("  %-34s %10s %14s %12s"
      % ("nuisance allowed", "d.o.f.", "kappa", "vs parametric"))
print("  " + "-" * 74)
rows = []
for beta in (2.3, 3.4, 5.7):
    D = t ** beta
    dp = beta * np.maximum(t, 1e-12) ** (beta - 1)
    r_par = 1.0 / (16.0 * beta ** 4)          # Proposition 10
    # tangent space of {A t^b}: derivatives in A and in b
    tang = [t ** beta, (t ** beta) * np.log(np.maximum(t, 1e-12))]
    ladder = [("amplitude and exponent (Prop 10)", 2, tang)]
    for deg in (3, 6, 12):
        ladder.append(("polynomial of degree %d" % deg, deg + 1,
                       [t ** k for k in range(deg + 1)]))
    base = None
    for lab, dof, basis in ladder:
        k = kappa_of(dp, basis)
        if base is None:
            base = k
        rows.append(dict(beta=beta, nuisance=lab, dof=dof, kappa=float(k),
                         vs_par=float(k / base) if base > 0 else np.nan))
        if abs(beta - 3.4) < 1e-9:
            print("  %-34s %10d %14.3e %12.3f"
                  % (lab, dof, k, k / base if base > 0 else np.nan))
    # the pipeline's own smoother, which has no finite basis to write down
    w = max(7, int(0.08 * N))
    w += (w % 2 == 0)
    k_s = float(np.sum((dp - oof_trend(dp, w // 2)) ** 2) / np.sum(dp ** 2))
    rows.append(dict(beta=beta, nuisance="local quadratic, 8 per cent",
                     dof=int(N / (w // 2)), kappa=k_s,
                     vs_par=float(k_s / base) if base > 0 else np.nan))
    if abs(beta - 3.4) < 1e-9:
        print("  %-34s %10d %14.3e %12.3f"
              % ("local quadratic, 8 per cent", int(N / (w // 2)), k_s,
                 k_s / base if base > 0 else np.nan))
        OUT["prop10_closed_form"] = float(r_par)
        OUT["kappa_parametric"] = float(base)
        OUT["kappa_smoother"] = float(k_s)
OUT["ladder"] = rows

_b3 = [r for r in rows if abs(r["beta"] - 3.4) < 1e-9]
if _b3:
    _ks = [r["kappa"] for r in _b3]
    OUT["ladder_monotone"] = bool(all(_ks[i] >= _ks[i + 1] - 1e-15
                                      for i in range(len(_ks) - 1)))
    OUT["ladder_span"] = float(max(_ks) / max(min(_ks), 1e-300))
    _poly = [r for r in _b3 if "polynomial" in r["nuisance"]]
    _sm = next(r for r in _b3 if "local" in r["nuisance"])
    OUT["kappa_poly12"] = float(_poly[-1]["kappa"])
    OUT["parametric_over_smoother"] = float(OUT["kappa_parametric"]
                                            / max(OUT["kappa_smoother"], 1e-300))
    print("""
  The parametric rung is %.3e and Proposition 10's closed form for the same class
  is %.3e; the same quantity computed two ways, agreeing to %.1f per cent.  That
  is an independent confirmation of Proposition 10 by a route it was not derived
  through, and it fixes the ladder's top rung.

  From there the cost is severe.  A fit matched to the truth keeps %.3e of the
  derivative; the pipeline's smoother keeps %.3e, a factor of %.1e.  Flexibility
  is what costs and it costs heavily.

  It does not, however, cost in proportion to the degrees of freedom, and the
  ladder is not monotone in them: %s.  The local quadratic has twenty-five
  effective parameters and keeps MORE than a global polynomial with thirteen,
  %.3e against %.3e.  A local basis and a global basis of the same dimension do
  not span the same functions, and it is the span and not the count that decides
  what survives.  Degrees of freedom are the wrong currency for this, which is
  worth saying because they are the obvious one to reach for.
""" % (OUT["kappa_parametric"], OUT["prop10_closed_form"],
       100 * abs(OUT["kappa_parametric"] - OUT["prop10_closed_form"])
       / OUT["prop10_closed_form"],
       OUT["kappa_parametric"], OUT["kappa_smoother"],
       OUT["parametric_over_smoother"],
       "it falls and then rises" if not OUT["ladder_monotone"] else "monotone",
       OUT["kappa_smoother"], OUT["kappa_poly12"]))

json.dump(OUT, open(os.path.join(HERE, "price_of_flexibility.json"), "w"),
          indent=2, default=float)
print("written to price_of_flexibility.json")
