"""
Testing the claim that survives: feasibility, not performance.

The end-to-end test showed the bound governs an estimator that knows the
degradation trend -- the oracle tracked the predicted -1/2 scaling -- while a
practical ridge model left a factor of thirty-four unclaimed and did not
improve as noise fell. So the framework cannot promise which indicator will give
the better model. What it can still promise is a floor: no estimator whatever
resolves the damage clock to precision eps before the age at which the record
has accumulated eps^{-2} of information.

That is checked here in its strongest form. Rather than the scaling alone, the
whole curve is tested pointwise:

    oracle standard error at age tau_0   ==   G(tau_0)^{-1/2}

If the two agree across ages, tau_min follows immediately as the age where the
prediction crosses eps, and the feasibility claim is exact rather than
conservative. If the oracle beats the prediction anywhere, the claim is false.
If it is far above everywhere, the bound is true but loose and tau_min would be
optimistic.

The oracle uses everything up to tau_0, which is what tau_min assumes: the
question is whether the target is reachable at all, with no retention limit.
"""
import os
import json
import numpy as np
from nonparam import weighted_density, estimate_floor

HERE = os.path.dirname(os.path.abspath(__file__))
N_UNITS, N_MEAN = 40, 800
LIFE_CV, BETA_SD = 0.20, 0.25
AGES = (0.25, 0.40, 0.55, 0.70, 0.85, 0.95)
NOISE = (0.02, 0.08, 0.30)
SEED = 606


def make_fleet(rng):
    """Each unit is a shape function of normalised age, sampled over its life.

    The shape is kept as a callable so the oracle can evaluate it at candidate
    lives rather than only at the sampled points.
    """
    lives = np.clip(rng.normal(N_MEAN, LIFE_CV * N_MEAN, N_UNITS),
                    0.4 * N_MEAN, 2.0 * N_MEAN).astype(int)
    shapes, series = [], []
    for L in lives:
        b = float(np.clip(rng.normal(2.5, BETA_SD), 1.2, 6.0))
        a = float(np.exp(rng.normal(0.0, BETA_SD)))
        shapes.append(lambda t, a=a, b=b: a * np.asarray(t, float) ** b)
        series.append(shapes[-1](np.arange(1, L + 1) / L))
    return shapes, series


def oracle_sd(shapes, noisy, t0, grid=81):
    """Standard error of the ML estimate of the damage clock, trend shape known.

    A first version slid the observed stretch along the unit's own curve and
    took the best-matching end position. That search could only place the
    stretch at or after its true position -- the observations run from index 0,
    so no earlier alignment exists -- and the resulting one-sided error had a
    standard deviation well below the floor, which is how a Cramer-Rao bound
    came to be "beaten" in eleven of eighteen cases.

    The estimator is reformulated as the problem it actually is. The shape of
    the degradation, as a function of normalised age, is known; the unknown is
    the unit's life L, and hence where on that shape the record currently sits.
    Candidate lives are scored by least squares against the observations, and
    the estimate of tau_0 follows as k_0 / L_hat. The error is then two-sided
    and the estimator is the ML one for a known shape.
    """
    err, edge = [], 0
    for shape, x in zip(shapes, noisy):
        n = len(x)
        k0 = int(round(t0 * n)) - 1
        if k0 < 10:
            continue
        idx = np.arange(k0 + 1)
        obs = x[:k0 + 1]

        def sse(L):
            d = obs - shape(np.clip(idx / L, 0.0, 1.0))
            return float(d @ d)

        # coarse pass over lives bracketing the truth on both sides, then a
        # refinement: a coarse grid alone makes every unit snap to the same
        # node when the noise is low, which reports a standard deviation of
        # exactly zero and looks like the floor being broken
        lo, hi = 0.55 * n, 2.2 * n
        for _ in range(3):
            cand = np.linspace(lo, hi, grid)
            vals = [sse(L) for L in cand]
            j = int(np.argmin(vals))
            if j == 0 or j == len(cand) - 1:
                edge += 1
            step = cand[1] - cand[0]
            lo, hi = cand[j] - step, cand[j] + step
        bL = float(cand[j])
        err.append((k0 + 1) / bL - (k0 + 1) / n)
    if len(err) <= 2:
        return np.nan, 0
    return float(np.std(err, ddof=1)), edge


def G_at(noisy, rng, t0):
    G = []
    for raw in noisy:
        dens, res = weighted_density(np.asarray(raw, float))
        if dens is None:
            continue
        d = np.clip(dens - estimate_floor(dens, res, "surrogate", rng, reps=6),
                    0.0, None)
        k0 = int(round(t0 * len(d))) - 1
        if k0 > 1:
            G.append(float(d[:k0 + 1].sum()))
    return float(np.median(G)) if G else np.nan


rng = np.random.default_rng(SEED)
shapes, fleet = make_fleet(rng)
print(f"fleet: {N_UNITS} units, mean life {N_MEAN}\n")
print(f"{'noise':>7}{'age':>7}{'G(age)':>12}{'predicted sd':>14}"
      f"{'oracle sd':>12}{'ratio':>8}")
print("-" * 62)
rows = []
for lv in NOISE:
    noisy = [s + rng.normal(0, lv, len(s)) for s in fleet]
    for t0 in AGES:
        g = G_at(noisy, rng, t0)
        o, edge = oracle_sd(shapes, noisy, t0)
        if not (np.isfinite(g) and g > 0 and np.isfinite(o)):
            continue
        pred = 1.0 / np.sqrt(g)
        rows.append(dict(noise=lv, age=t0, G=g, predicted=pred, oracle=o,
                         ratio=o / pred, edge_hits=edge))
        flag = "  <- search hit a bracket edge" if edge else ""
        print(f"{lv:>7.2f}{t0:>7.2f}{g:>12.3g}{pred:>14.5f}{o:>12.5f}"
              f"{o/pred:>8.2f}{flag}")
    print()
print("-" * 62)
r = np.array([x["ratio"] for x in rows])
below = int((r < 0.95).sum())
print(f"ratio of achieved to predicted standard error: "
      f"median {np.median(r):.2f}, range {r.min():.2f}-{r.max():.2f}")
print(f"cases where the oracle beat the floor: {below} of {len(r)}")
print()
if below == 0:
    print("The floor is never broken, so the feasibility claim holds: no")
    print("estimator reaches the target before the record has the information.")
    if np.median(r) < 3:
        print(f"It is also tight -- the best estimator sits within "
              f"{np.median(r):.1f}x of it -- so tau_min read off the curve is")
        print("close to the true earliest usable age, not merely a lower bound.")
    else:
        print(f"But it is loose by {np.median(r):.0f}x, so tau_min read off the")
        print("curve is optimistic: the real earliest usable age is later.")
else:
    print("The floor is broken, which would mean the information estimate is")
    print("too high -- most likely the noise floor is under-subtracted.")
json.dump(rows, open(os.path.join(HERE, "feasibility.json"), "w"), indent=2,
          default=float)
