"""
How loose is the floor, and can the looseness be predicted?

The feasibility test found the bound never broken but not exact: the best
possible estimator sat between 1.1x and 2.8x above it, and the gap widened with
noise. A floor that is loose by an unknown factor is still a floor, but tau_min
read off it is optimistic by an unknown amount, which is awkward to use.

The gap has a standard explanation. The Cramer-Rao bound is asymptotic: it
describes the curvature of the log-likelihood at the true parameter, and an
estimator attains it only when the likelihood is close to quadratic over the
region the estimate actually explores. When the predicted error is itself large
-- a substantial fraction of the parameter's range -- the estimate wanders into
regions where that approximation fails, and the achieved error exceeds the
bound.

If that is the mechanism, the excess should be a function of the PREDICTED error
and not of the noise, the record length or the age separately, since those enter
only through it. That is a strong prediction and an easily falsified one: the
same predicted error reached by different routes must show the same excess.

A correction follows if it holds, turning tau_min from a floor into a calibrated
estimate.
"""
import os
import json
import numpy as np
from nonparam import weighted_density, estimate_floor

HERE = os.path.dirname(os.path.abspath(__file__))
N_UNITS = 30
SEED = 8128
LENGTHS = (300, 800)
NOISES = (0.01, 0.04, 0.12, 0.40)
AGES = (0.30, 0.50, 0.70, 0.90)


def make_fleet(rng, n_mean, n_units=N_UNITS, life_cv=0.20, beta_sd=0.25):
    lives = np.clip(rng.normal(n_mean, life_cv * n_mean, n_units),
                    0.4 * n_mean, 2.0 * n_mean).astype(int)
    shapes, series = [], []
    for L in lives:
        b = float(np.clip(rng.normal(2.5, beta_sd), 1.2, 6.0))
        a = float(np.exp(rng.normal(0.0, beta_sd)))
        shapes.append(lambda t, a=a, b=b: a * np.asarray(t, float) ** b)
        series.append(shapes[-1](np.arange(1, L + 1) / L))
    return shapes, series


def oracle_sd(shapes, noisy, t0, grid=81):
    err, edge = [], 0
    for shape, x in zip(shapes, noisy):
        n = len(x)
        k0 = int(round(t0 * n)) - 1
        if k0 < 10:
            continue
        idx = np.arange(k0 + 1)
        obs = x[:k0 + 1]
        sse = lambda L: float(np.sum((obs - shape(np.clip(idx / L, 0, 1))) ** 2))
        lo, hi = 0.45 * n, 3.0 * n           # wider bracket than before
        for _ in range(3):
            cand = np.linspace(lo, hi, grid)
            vals = [sse(L) for L in cand]
            j = int(np.argmin(vals))
            if j in (0, len(cand) - 1):
                edge += 1
            step = cand[1] - cand[0]
            lo, hi = cand[j] - step, cand[j] + step
        err.append((k0 + 1) / float(cand[j]) - (k0 + 1) / n)
    if len(err) <= 2:
        return np.nan, 1
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
rows = []
print("collecting (predicted error, achieved error) over many conditions\n")
print(f"{'n':>6}{'noise':>7}{'age':>6}{'predicted':>11}{'achieved':>11}"
      f"{'excess':>8}")
print("-" * 49)
for n_mean in LENGTHS:
    shapes, fleet = make_fleet(rng, n_mean)
    for lv in NOISES:
        noisy = [s + rng.normal(0, lv, len(s)) for s in fleet]
        for t0 in AGES:
            g = G_at(noisy, rng, t0)
            o, edge = oracle_sd(shapes, noisy, t0)
            if not (np.isfinite(g) and g > 0 and np.isfinite(o) and o > 0):
                continue
            pred = 1.0 / np.sqrt(g)
            rows.append(dict(n=n_mean, noise=lv, age=t0, G=g, predicted=pred,
                             achieved=o, excess=o / pred, edge=int(edge)))
            mark = " *" if edge else ""
            print(f"{n_mean:>6}{lv:>7.2f}{t0:>6.2f}{pred:>11.5f}{o:>11.5f}"
                  f"{o/pred:>8.2f}{mark}")
print("-" * 49)
print("* the search reached its bracket; those points are unreliable\n")

good = [r for r in rows if not r["edge"]]
print(f"{len(good)} clean points of {len(rows)}\n")


def rank_corr(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    ra, rb = np.argsort(np.argsort(a)), np.argsort(np.argsort(b))
    return float(np.corrcoef(ra, rb)[0, 1])


ex = [r["excess"] for r in good]
print("what does the excess track?")
for key in ("predicted", "noise", "n", "age", "G"):
    print(f"  {key:<12} rank correlation {rank_corr([r[key] for r in good], ex):+.2f}")

# the prediction: excess is a function of the predicted error alone
lp = np.log([r["predicted"] for r in good])
le = np.log(ex)
slope, inter = np.polyfit(lp, le, 1)
pred_ex = np.exp(inter) * np.array([r["predicted"] for r in good]) ** slope
resid = np.array(ex) / pred_ex
print(f"\nfitting excess = c * (predicted error)^p")
print(f"  p = {slope:+.3f}, c = {np.exp(inter):.3f}, "
      f"r = {float(np.corrcoef(lp, le)[0, 1]):+.2f}")
print(f"  after correction the residual spread is "
      f"{resid.max()/resid.min():.2f}x, against "
      f"{max(ex)/min(ex):.2f}x uncorrected")
print()
print("If the excess depends only on the predicted error, the same predicted")
print("error reached at different noise and length must show the same excess.")
by_pred = sorted(good, key=lambda r: r["predicted"])
print("\n  closest pairs from different conditions:")
shown = 0
for i in range(len(by_pred) - 1):
    a, b = by_pred[i], by_pred[i + 1]
    if a["n"] != b["n"] or a["noise"] != b["noise"]:
        if abs(np.log(a["predicted"] / b["predicted"])) < 0.15:
            print(f"    pred {a['predicted']:.5f} (n={a['n']}, s={a['noise']}) "
                  f"excess {a['excess']:.2f}   vs "
                  f"pred {b['predicted']:.5f} (n={b['n']}, s={b['noise']}) "
                  f"excess {b['excess']:.2f}")
            shown += 1
            if shown >= 5:
                break
json.dump(dict(rows=rows, slope=float(slope), c=float(np.exp(inter))),
          open(os.path.join(HERE, "tightness.json"), "w"), indent=2, default=float)
