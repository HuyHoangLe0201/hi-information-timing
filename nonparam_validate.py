"""
Does reading the quantiles beat fitting an exponent?

Dropping the power law is only right if the non-parametric reading is at least
as accurate where a power law is true, and clearly better where it is not. Both
halves are tested here against synthetic records whose information curve is
known exactly.

The comparator is the route the power-law summary takes: fit beta to the
measured cumulative, then compute tau_min = q^{1/(2 beta - 1)}. The
non-parametric route reads tau_min off the measured cumulative directly, after
subtracting the noise floor.

The shapes are chosen to include the ones a power law cannot represent:
a plateau followed by a rise, which is what a bearing that runs in before it
degrades actually looks like; and an S-curve, which is what a process that
saturates looks like. If the power law only loses on those, the decision is
about which shapes matter; if it loses everywhere, the decision is simpler.
"""
import os
import json
import numpy as np
from scipy.optimize import minimize_scalar
from nonparam import info_curve, quantile

HERE = os.path.dirname(os.path.abspath(__file__))
QS = (0.10, 0.25, 0.50, 0.75, 0.90)
REPS = 20
N = 1000


def true_curve(dfun, n, h=1e-5):
    """Exact F for a known trend, constant noise: cumulative of D'^2."""
    tau = np.arange(1, n + 1) / n
    d = (dfun(np.clip(tau + h, 0, 1)) - dfun(np.clip(tau - h, 0, 1))) / (2 * h)
    dens = d ** 2
    return np.cumsum(dens) / dens.sum()


def beta_route(x, qs):
    """Fit an exponent to the measured cumulative, then use the closed form."""
    from nonparam import weighted_density
    dens, _ = weighted_density(x)
    if dens is None or dens.sum() <= 0:
        return {q: np.nan for q in qs}
    n = len(dens)
    F = np.cumsum(dens) / dens.sum()
    tau = np.arange(1, n + 1) / n
    f = lambda b: float(np.mean((F - tau ** (2 * b - 1)) ** 2))
    b = float(minimize_scalar(f, bounds=(0.51, 30.0), method="bounded").x)
    return {q: q ** (1.0 / (2 * b - 1)) for q in qs}


SHAPES = {
    "power law b=3": lambda t: t ** 3,
    "power law b=8": lambda t: t ** 8,
    "linear": lambda t: t,
    "exponential": lambda t: (1 - np.exp(-3 * t)) / (1 - np.exp(-3.0)),
    "plateau then rise": lambda t: np.clip((t - 0.6) / 0.4, 0, 1) ** 2,
    "S-curve": lambda t: 1 / (1 + np.exp(-12 * (t - 0.5))),
    "two-stage": lambda t: 0.3 * t + 0.7 * np.clip((t - 0.75) / 0.25, 0, 1) ** 2,
}

rng = np.random.default_rng(4242)
tau = np.arange(1, N + 1) / N

print(f"quantile recovery, {REPS} records each, sigma = 0.05\n")
print(f"{'shape':<20}{'route':<14}" + "".join(f"{q:>8.2f}" for q in QS)
      + f"{'mean err':>10}")
print("-" * 90)
rows = []
for lab, dfun in SHAPES.items():
    Ftrue = true_curve(dfun, N)
    tq = {q: quantile(Ftrue, q) for q in QS}
    clean = dfun(tau)
    NP, BE = {q: [] for q in QS}, {q: [] for q in QS}
    for _ in range(REPS):
        x = clean + rng.normal(0, 0.05, N)
        F = info_curve(x, floor="surrogate", rng=rng)
        for q in QS:
            NP[q].append(quantile(F, q))
        br = beta_route(x, QS)
        for q in QS:
            BE[q].append(br[q])
    npm = {q: float(np.median(NP[q])) for q in QS}
    bem = {q: float(np.median(BE[q])) for q in QS}
    e_np = float(np.mean([abs(npm[q] - tq[q]) for q in QS]))
    e_be = float(np.mean([abs(bem[q] - tq[q]) for q in QS]))
    rows.append(dict(shape=lab, truth=tq, nonparam=npm, beta=bem,
                     err_nonparam=e_np, err_beta=e_be))
    print(f"{lab:<20}{'truth':<14}" + "".join(f"{tq[q]:>8.3f}" for q in QS))
    print(f"{'':<20}{'non-param':<14}" + "".join(f"{npm[q]:>8.3f}" for q in QS)
          + f"{e_np:>10.3f}")
    print(f"{'':<20}{'via beta':<14}" + "".join(f"{bem[q]:>8.3f}" for q in QS)
          + f"{e_be:>10.3f}")
    print()

print("-" * 90)
pl = [r for r in rows if r["shape"].startswith("power law")]
ot = [r for r in rows if not r["shape"].startswith("power law")]
print(f"{'group':<26}{'non-param':>12}{'via beta':>11}{'ratio':>9}")
print("-" * 58)
for tag, grp in (("power-law shapes", pl), ("everything else", ot),
                 ("all shapes", rows)):
    a = float(np.mean([r["err_nonparam"] for r in grp]))
    b = float(np.mean([r["err_beta"] for r in grp]))
    print(f"{tag:<26}{a:>12.3f}{b:>11.3f}{b/a:>9.2f}x")
print("-" * 58)
print("error is the mean absolute quantile error, in fractions of life.")
print("A ratio above 1 means the non-parametric reading is the better one.")
json.dump(rows, open(os.path.join(HERE, "nonparam_validate.json"), "w"),
          indent=2, default=float)
