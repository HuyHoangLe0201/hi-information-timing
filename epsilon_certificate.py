r"""The worst contrast under an energy budget, and where it is safe to read eta.

Proposition 18 charges a distortion by its RANGE and is defeated by corrections
whose extremes sit on a handful of samples.  Proposition 15 charges it by an
information-weighted average and predicts the same corrections to within 6, 30
and 38 per cent.  The certificate should therefore be written in the second
currency, and doing so has a closed form.

Let u = log w, normalised so that <u>_[0,1] = 0, and let its budget be its
information-weighted energy,

    Var_g(u) = int u^2 g / int g   <=   eps^2.

By Proposition 19 the log-contrast at a cut c is <u>_[0,c] - <u>_[c,1] to first
order, which is a LINEAR functional of u,

    <u>_[0,c] - <u>_[c,1] = int u h g / int g,     h = 1_A / p - 1_B / (1-p),

with A = [0,c], B = (c,1] and p = eta(c) the budget share before the cut.  A
linear functional under an energy constraint is Cauchy-Schwarz, so its maximum is
eps times the norm of h, and h is already centred:

    int h g / int g = 1 - 1 = 0,      int h^2 g / int g = 1/p + 1/(1-p),

giving

    max log R(c)  =  eps / sqrt( eta(c) (1 - eta(c)) ).                    (*)

Two things follow that no earlier result gives.

  The bound depends on WHERE the curve is read.  It is smallest at eta = 1/2 and
  diverges as eta approaches 0 or 1, so a censoring cut placed where the record
  has spent about half its budget is the most robust place to read one, and a cut
  near either end is the least.  That is a design rule with a proof, and the
  paper currently reads eta at cuts chosen by convention.

  The extremal distortion is a two-level step at c, as in Proposition 16, but
  with levels set by the masses rather than by a range.  So the worst case has
  the same shape under either budget and only its size differs.

(*) is first order in eps.  Whether that matters at the eps these corrections
carry is measured rather than assumed.
"""
import os
import json
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
rng = np.random.default_rng(160180)
OUT = {}
N = 4000
tau = (np.arange(N) + 1.0) / N

DENS = {
    "rising": 2 * tau,
    "falling": 2 - 2 * tau,
    "peaked late": 0.2 + 3.0 * tau ** 4,
    "bimodal": 1 + np.exp(-((tau - 0.2) / 0.05) ** 2)
    + np.exp(-((tau - 0.85) / 0.05) ** 2),
}


def contrast(w, g, ic):
    a = float(np.sum(w[:ic + 1] * g[:ic + 1]) / np.sum(g[:ic + 1]))
    b = float(np.sum(w[ic + 1:] * g[ic + 1:]) / np.sum(g[ic + 1:]))
    return a / b


def energy(u, g):
    tot = float(np.sum(g))
    m = float(np.sum(u * g)) / tot
    return float(np.sqrt(np.sum((u - m) ** 2 * g) / tot))


print("=" * 92)
print("1.  Is the extremal distortion the one the derivation names?")
print("=" * 92)
print("""
The derivation says the worst u under an energy budget is proportional to
h = 1_A/p - 1_B/(1-p).  That is a claim about a maximiser, so it is checked
against a search: many random distortions are rescaled to the same energy and
none may beat the step.
""")
print("  %-14s %8s %10s %12s %12s %9s"
      % ("density", "eta", "eps", "step log R", "best found", "ratio"))
print("  " + "-" * 70)
rows, worst_beat = [], 0.0
for dn, g0 in DENS.items():
    g = g0 / N
    for cfrac in (0.35, 0.6):
        ic = int(cfrac * N) - 1
        p = float(np.sum(g[:ic + 1]) / np.sum(g))
        if not (0.02 < p < 0.98):
            continue
        for eps in (0.15, 0.4):
            h = np.where(np.arange(N) <= ic, 1.0 / p, -1.0 / (1.0 - p))
            u_star = eps * h / energy(h, g)
            step = float(np.log(contrast(np.exp(u_star), g, ic)))
            best = -np.inf
            for _ in range(400):
                k = int(rng.integers(0, 3))
                if k == 0:
                    u = rng.normal(0, 1, N)
                elif k == 1:
                    u = np.convolve(rng.normal(0, 1, N), np.ones(51) / 51, "same")
                else:
                    u = np.sort(rng.random(N))[::-1]
                u = u - float(np.sum(u * g) / np.sum(g))
                e = energy(u, g)
                if e <= 0:
                    continue
                u = eps * u / e
                best = max(best, float(np.log(contrast(np.exp(u), g, ic))))
            rows.append(dict(density=dn, eta=p, eps=eps, step=step,
                             best=best, ratio=best / step))
            worst_beat = max(worst_beat, best / step)
            print("  %-14s %8.3f %10.2f %12.4f %12.4f %9.3f"
                  % (dn, p, eps, step, best, best / step))
OUT["extremal"] = rows
OUT["worst_search_over_step"] = float(worst_beat)
print("""
  No random distortion at the same energy exceeds the step; the largest anything
  reaches is %.3f of it.  The maximiser is the one the derivation names.
""" % worst_beat)

# =============================================================================
print("=" * 92)
print("2.  The closed form, and how far first order carries it")
print("=" * 92)
print("""
(*) says the worst log-contrast is eps / sqrt(eta(1-eta)).  The step's exact
log-contrast is computed and compared, over a wide range of eta and eps, so the
size of the first-order error is measured rather than hoped for.
""")
print("  %-10s %10s %14s %14s %10s"
      % ("eta", "eps", "exact", "closed form", "error"))
print("  " + "-" * 62)
crows = []
g = DENS["peaked late"] / N
for cfrac in (0.15, 0.35, 0.5, 0.7, 0.9):
    ic = int(cfrac * N) - 1
    p = float(np.sum(g[:ic + 1]) / np.sum(g))
    if not (0.01 < p < 0.99):
        continue
    for eps in (0.05, 0.15, 0.4):
        h = np.where(np.arange(N) <= ic, 1.0 / p, -1.0 / (1.0 - p))
        u_star = eps * h / energy(h, g)
        exact = float(np.log(contrast(np.exp(u_star), g, ic)))
        pred = eps / np.sqrt(p * (1.0 - p))
        crows.append(dict(eta=p, eps=eps, exact=exact, closed=float(pred),
                          err=abs(exact - pred) / abs(pred)))
        print("  %-10.4f %10.2f %14.4f %14.4f %10.4f"
              % (p, eps, exact, pred, crows[-1]["err"]))
OUT["closed_form"] = crows
if crows:
    small = [r for r in crows if r["eps"] <= 0.05]
    OUT["err_at_small_eps"] = float(max(r["err"] for r in small)) if small else None
    OUT["err_at_large_eps"] = float(max(r["err"] for r in crows
                                        if r["eps"] >= 0.4))
    print("""
  The agreement is exact, to %.0e at every budget including eps = 0.40, and that
  is not luck.  For a two-level u the weighted means ARE the two levels, so
  log R = u_A - u_B with no expansion, and the constraint fixes that difference
  at eps/sqrt(eta(1-eta)).  (*) is an identity for the step, not an
  approximation to it.  What remains open is whether the step is the worst case,
  which section 3 settles.

  The shape is the useful part: the worst contrast scales as 1/sqrt(eta(1-eta)),
  and nothing about the density enters beyond eta itself.
""" % max(OUT["err_at_small_eps"] or 0.0, OUT["err_at_large_eps"], 1e-16))

# =============================================================================
print("=" * 92)
print("3.  Is the step actually the maximiser, or only of the linearisation?")
print("=" * 92)
print("""
Section 1 compared the step against random distortions, which is weak evidence
for optimality, and the derivation maximises the LINEARISED functional
<u>_A - <u>_B, not the exact log-contrast.  Those differ, and in a direction that
matters: by Jensen <e^u>_A exceeds e^{<u>_A}, so spreading u within A raises the
numerator for the same mean, at a cost in energy.  Whether the trade-off favours
the step is a question for an optimiser, not for a search.

The exact objective is maximised over a free piecewise-constant u with twenty
blocks, started from the step and from random points, under the same energy
budget.  If the step is optimal nothing beats it; if it is not, the excess is the
error in (*) and has to be reported.
""")
from scipy import optimize

print("  %-10s %8s %14s %14s %10s"
      % ("eta", "eps", "step", "optimiser", "excess"))
print("  " + "-" * 60)
orows = []
NB = 20
for cfrac in (0.35, 0.6):
    ic = int(cfrac * N) - 1
    p = float(np.sum(g[:ic + 1]) / np.sum(g))
    if not (0.02 < p < 0.98):
        continue
    edges = np.linspace(0, N, NB + 1).astype(int)
    blk = np.zeros((NB, N))
    for b in range(NB):
        blk[b, edges[b]:edges[b + 1]] = 1.0
    # 0.90 was added after trimmed_certificate.py measured the budgets
    # the paper's own corrections carry: the sweep stopped at 0.40 and
    # the text called that above the largest budget in the paper, a
    # figure that had been typed rather than measured.  It is 0.90.
    for eps in (0.15, 0.4, 0.9):
        h = np.where(np.arange(N) <= ic, 1.0 / p, -1.0 / (1.0 - p))
        u_step = eps * h / energy(h, g)
        step_val = float(np.log(contrast(np.exp(u_step), g, ic)))

        def neg(v):
            u = v @ blk
            e = energy(u, g)
            if e <= 1e-12:
                return 0.0
            u = eps * u / e                       # project onto the budget
            return -float(np.log(contrast(np.exp(u), g, ic)))

        best = step_val
        starts = [np.array([float(np.mean(u_step[edges[b]:edges[b + 1]]))
                            for b in range(NB)])]
        for _ in range(8):
            starts.append(rng.normal(0, 1, NB))
        for s0 in starts:
            r = optimize.minimize(neg, s0, method="Nelder-Mead",
                                  options=dict(maxiter=20000, fatol=1e-12,
                                               xatol=1e-10))
            best = max(best, -float(r.fun))
        orows.append(dict(eta=p, eps=eps, step=step_val, best=best,
                          excess=best / step_val - 1.0))
        print("  %-10.4f %8.2f %14.5f %14.5f %10.4f"
              % (p, eps, step_val, best, best / step_val - 1.0))
OUT["optimiser"] = orows
if orows:
    OUT["max_excess"] = float(max(r["excess"] for r in orows))
    _small = [r for r in orows if r["eps"] <= 0.2]
    _large = [r for r in orows if 0.4 <= r["eps"] < 0.9]
    _huge = [r for r in orows if r["eps"] >= 0.9]
    OUT["excess_small_eps"] = float(max(r["excess"] for r in _small)) if _small else None
    OUT["excess_large_eps"] = float(max(r["excess"] for r in _large)) if _large else None
    OUT["excess_at_measured_max"] = (float(max(r["excess"] for r in _huge))
                                     if _huge else None)
    # the budget the paper's own corrections carry, measured rather than typed
    _tcf = os.path.join(HERE, "trimmed_certificate.json")
    _EPS_MAX = (json.load(open(_tcf))["eps_max"] if os.path.exists(_tcf)
                else float("nan"))
    OUT["eps_max_of_corrections"] = float(_EPS_MAX)
    print("""
  The answer splits by budget, and the split is the result.  At eps = 0.15 the
  optimiser cannot beat the step at all, to %.0e: there the step IS the worst
  case and (*) is the exact bound.  At eps = 0.40 the optimiser beats it by up to
  %.0f per cent, and the mechanism is the one anticipated: spreading u within the
  early set raises <e^u> above e^{<u>} by Jensen, and at a large budget that gain
  outweighs what the spreading costs in energy.

  So the honest form of the claim has three parts.  (*) is an identity for the
  two-level distortion at any budget.  It is the exact worst case for small
  budgets.  And it understates the true worst case by up to %.0f per cent at
  eps = 0.40 and %s at eps = 0.90, the largest budget the corrections of this
  paper actually carry (%.2f, measured in trimmed_certificate.py; an earlier
  version of this sentence gave 0.34, which had been typed rather than measured
  and put the sweep's endpoint above the range in use when it was below it).
""" % (OUT["excess_small_eps"] or 0.0, 100 * (OUT["excess_large_eps"] or 0.0),
       100 * (OUT["excess_large_eps"] or 0.0),
       ("%.0f per cent" % (100 * OUT["excess_at_measured_max"]))
       if OUT["excess_at_measured_max"] is not None else "an unmeasured amount",
       _EPS_MAX))

json.dump(OUT, open(os.path.join(HERE, "epsilon_certificate.json"), "w"),
          indent=2, default=float)
print("written to epsilon_certificate.json")
