"""
What the information is worth, in the currency a maintainer uses.

This paper says when an indicator becomes usable at a stated precision.  A
maintenance decision is not a precision target: it is a trade between replacing
early, which throws away life, and replacing late, which risks failure.  Nothing
here has connected the two, and that connection is what the framework is
ultimately for.

The connection needs only the standard error of the age estimate, not the
absolute calibration of G that Section 6.5 shows to be unusable.  Let a unit fail
at tau = 1, let the estimate of its age carry error eps ~ N(0, sigma^2), and let
the policy replace when the ESTIMATED age reaches a.  Replacement then happens at
true age a - eps, which precedes failure exactly when eps > a - 1.  Writing
u = (a - 1)/sigma, with c_f the cost of a failure and c_p the cost per unit of
life discarded,

    C(u, sigma) = c_f Phi(u) + c_p sigma [ phi(u) - u (1 - Phi(u)) ] ,

and differentiating,

    dC/du = c_f phi(u) - c_p sigma (1 - Phi(u)) = 0
        =>  phi(u) / (1 - Phi(u))  =  c_p sigma / c_f ,

so the optimal threshold sets the standard normal's HAZARD to the ratio of the
cost of discarded life to the cost of failure.  The minimised cost C*(sigma) is
then a function of the measurement error alone.

The value of an indicator is the cost it saves against measuring nothing.  With
no indicator the best a fleet can do is replace on elapsed time, whose error is
the fleet prior s of Proposition 2 -- so the baseline is C*(s), and the value is
C*(s) - C*(sigma).  Both ends of that are quantities this paper has measured.
"""
import os
import json
import numpy as np
from scipy.stats import norm
from scipy.optimize import brentq

HERE = os.path.dirname(os.path.abspath(__file__))
SIGMAS = (0.005, 0.01, 0.02, 0.05, 0.10, 0.20, 0.294)
RATIOS = (0.02, 0.05, 0.10, 0.25)          # c_p / c_f


def cost(u, sigma, ratio):
    """C / c_f, so the scale is failures avoided per unit."""
    return norm.cdf(u) + ratio * sigma * (norm.pdf(u)
                                          - u * (1.0 - norm.cdf(u)))


def optimal(sigma, ratio):
    """The hazard condition phi(u)/(1-Phi(u)) = ratio * sigma."""
    target = ratio * sigma

    def f(u):
        s = 1.0 - norm.cdf(u)
        return (norm.pdf(u) / s if s > 1e-300 else np.inf) - target

    # the normal hazard is increasing and unbounded above, ~0 far below
    lo, hi = -40.0, 10.0
    if f(lo) > 0:
        u = lo
    elif f(hi) < 0:
        u = hi
    else:
        u = brentq(f, lo, hi, xtol=1e-10)
    return float(u), float(cost(u, sigma, ratio))


print("expected cost of a replacement policy, in units of the failure cost\n")
print("the estimate's error is sigma; the policy threshold is set optimally\n")
print(f"{'sigma':>8}" + "".join(f"{f'c_p/c_f={r}':>15}" for r in RATIOS))
print("-" * (8 + 15 * len(RATIOS)))
grid = []
for s in SIGMAS:
    line = []
    for r in RATIOS:
        u, c = optimal(s, r)
        grid.append(dict(sigma=s, ratio=r, u=u, cost=c))
        line.append(c)
    print(f"{s:>8.3f}" + "".join(f"{c:>15.4f}" for c in line))
print("-" * (8 + 15 * len(RATIOS)))
print("The last row is the fleet prior of Proposition 2: what a policy costs")
print("when it replaces on elapsed time and measures nothing.\n")

# --- the value of an indicator ----------------------------------------------
PRIOR = 0.294
print("value of an indicator: the cost it saves against measuring nothing\n")
print(f"{'sigma':>8}" + "".join(f"{f'c_p/c_f={r}':>15}" for r in RATIOS))
print("-" * (8 + 15 * len(RATIOS)))
value = []
base = {r: optimal(PRIOR, r)[1] for r in RATIOS}
for s in SIGMAS[:-1]:
    line = []
    for r in RATIOS:
        c = optimal(s, r)[1]
        v = base[r] - c
        value.append(dict(sigma=s, ratio=r, saved=v,
                          fraction=v / max(base[r], 1e-30)))
        line.append(v)
    print(f"{s:>8.3f}" + "".join(f"{v:>15.4f}" for v in line))
print("-" * (8 + 15 * len(RATIOS)))
print("Entries are failures avoided per unit, so 0.01 means one failure")
print("prevented per hundred units against replacing on elapsed time.\n")

# --- where the paper's own estimators sit -----------------------------------
print("where this paper's estimators fall on that scale\n")
try:
    OM = json.load(open(os.path.join(HERE, "oracle_vs_model.json")))
    s_oracle = float(min(r["oracle"] for r in OM))
    s_ridge = float(min(r["ridge"] for r in OM))
except Exception:
    s_oracle = s_ridge = None

rows = []
if s_oracle is not None:
    print(f"{'estimator':<28}{'sigma':>9}" +
          "".join(f"{f'r={r}':>11}" for r in RATIOS))
    print("-" * (37 + 11 * len(RATIOS)))
    for lab, s in (("oracle, trend known", s_oracle),
                   ("practical regression", s_ridge),
                   ("fleet prior, no indicator", PRIOR)):
        line = []
        for r in RATIOS:
            c = optimal(min(s, PRIOR), r)[1]
            line.append(c)
        rows.append(dict(estimator=lab, sigma=float(s),
                         costs={str(r): c for r, c in zip(RATIOS, line)}))
        print(f"{lab:<28}{s:>9.4f}" + "".join(f"{c:>11.4f}" for c in line))
    print("-" * (37 + 11 * len(RATIOS)))
    print()
    r0 = RATIOS[1]
    co = rows[0]["costs"][str(r0)]
    cr = rows[1]["costs"][str(r0)]
    cp = rows[2]["costs"][str(r0)]
    print(f"At a cost ratio of {r0}, the oracle of Section 6 costs {co:.4f} per")
    print(f"unit against {cp:.4f} for replacing on elapsed time -- it removes")
    print(f"{100 * (1 - co / cp):.0f} per cent of the cost.  The practical regression costs")
    print(f"{cr:.4f}, removing {100 * (1 - cr / cp):.0f} per cent.  The factor of thirty-four")
    print(f"between their errors is therefore a factor of "
          f"{(cp - co) / max(cp - cr, 1e-12):.1f} in what they are")
    print("worth, which is the number a maintainer would be shown.")

json.dump(dict(sigmas=list(SIGMAS), ratios=list(RATIOS), prior=PRIOR,
               grid=grid, value=value, estimators=rows,
               sigma_oracle=s_oracle, sigma_ridge=s_ridge),
          open(os.path.join(HERE, "decision_value.json"), "w"), indent=2,
          default=float)
