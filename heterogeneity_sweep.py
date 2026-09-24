"""
What actually limits RUL error: measurement noise, or fleet heterogeneity?

On a synthetic fleet with realistic unit-to-unit variation, raising the
information by a factor of 119 lowered the RUL error by 1.4x, against the 10.9x
the Cramer-Rao floor predicts. The fitted slope was -0.075 against -0.500. Over
most of that range the error did not move at all: G fell 78-fold between the two
lowest noise levels and the error changed by 8%.

A first attempt blamed heterogeneity in the trend SHAPE and swept the spread of
the exponent across the fleet. It was the wrong nuisance: at zero shape
heterogeneity, with every unit on the identical curve, the slope was still
-0.141.

The spread of LIFE was held at a quarter throughout, and that is the nuisance
that matters. G is information about the normalised damage clock tau, but the
target is absolute remaining life, (1 - tau) times the unit's life L. Knowing
tau perfectly still leaves the error of not knowing L, and no amount of
measurement precision removes it. The bound governs the first term only.

So the sweep is over the fleet's spread of life. The prediction: as that spread
shrinks the error becomes dominated by locating the unit on its own curve, which
is exactly what the bound governs, and the slope should approach -1/2.
"""
import os
import json
import numpy as np
from nonparam import weighted_density, estimate_floor

HERE = os.path.dirname(os.path.abspath(__file__))
N_UNITS, N_MEAN, LIFE_CV = 60, 700, 0.25
NFEAT, RIDGE, N_POS = 12, 1e-3, 16
LEVELS = (0.01, 0.03, 0.08, 0.20, 0.50, 1.20)
HETERO = (0.02, 0.05, 0.12, 0.25, 0.50)   # CV of life across the fleet
BETA_SD = 0.25                            # shape spread, now held fixed
SEED = 90210


def make_fleet(rng, life_cv):
    lives = np.clip(rng.normal(N_MEAN, life_cv * N_MEAN, N_UNITS),
                    0.3 * N_MEAN, 2.5 * N_MEAN).astype(int)
    out = []
    for L in lives:
        tau = np.arange(1, L + 1) / L
        beta = float(np.clip(rng.normal(2.5, BETA_SD), 1.2, 6.0))
        amp = float(np.exp(rng.normal(0.0, BETA_SD)))
        out.append(amp * tau ** beta)
    return out


def baseline_norm(x, frac=0.05):
    x = np.asarray(x, float)
    k = max(3, int(frac * len(x)))
    b = np.median(x[:k])
    s = np.median(np.abs(x[:k] - b)) * 1.4826
    if not np.isfinite(s) or s <= 0:
        s = float(np.std(x)) or 1.0
    return (x - b) / s


def features(seg):
    g = np.interp(np.linspace(0, 1, NFEAT), np.linspace(0, 1, len(seg)), seg)
    t = np.linspace(0, 1, len(seg))
    return np.concatenate([g, [np.polyfit(t, seg, 1)[0], seg.std()]])


def build(series, kw, mean_n):
    X, y, u = [], [], []
    for i, raw in enumerate(series):
        x = baseline_norm(raw)
        n = len(x)
        if n <= kw + 4:
            continue
        for k0 in np.unique(np.linspace(kw, n - 2, N_POS).astype(int)):
            seg = x[k0 - kw + 1:k0 + 1]
            if len(seg) < 8 or not np.all(np.isfinite(seg)):
                continue
            f = features(seg)
            if np.all(np.isfinite(f)):
                X.append(f); y.append((n - 1 - k0) / mean_n); u.append(i)
    return np.array(X), np.array(y), np.array(u)


def loo(X, y, u):
    em, eb = [], []
    for i in np.unique(u):
        tr, te = u != i, u == i
        if tr.sum() < 8 or te.sum() == 0:
            continue
        A, b = X[tr], y[tr]
        mu, sd = A.mean(0), A.std(0)
        sd[sd <= 0] = 1.0
        A2 = np.column_stack([(A - mu) / sd, np.ones(tr.sum())])
        w = np.linalg.solve(A2.T @ A2 + RIDGE * np.eye(A2.shape[1]), A2.T @ b)
        B = np.column_stack([(X[te] - mu) / sd, np.ones(te.sum())])
        em.append(B @ w - y[te]); eb.append(y[te] - b.mean())
    rms = lambda v: float(np.sqrt(np.mean(np.concatenate(v) ** 2)))
    return (rms(em), rms(eb)) if em else (np.nan, np.nan)


def info(series, rng, t0=0.60):
    G = []
    for raw in series:
        dens, res = weighted_density(np.asarray(raw, float))
        if dens is None:
            continue
        d = np.clip(dens - estimate_floor(dens, res, "surrogate", rng, reps=6),
                    0.0, None)
        k0 = int(round(t0 * len(d))) - 1
        if k0 > 1:
            G.append(float(d[:k0 + 1].sum()))
    return float(np.median(G)) if G else np.nan


print("does fleet heterogeneity explain the collapsed slope?\n")
print(f"{'CV of life':>11}{'points':>8}{'G span':>10}{'slope':>10}"
      f"{'+/-':>8}{'r':>8}{'best skill':>12}")
print("-" * 67)
out = []
for het in HETERO:
    rng = np.random.default_rng(SEED + int(1000 * het))
    fleet = make_fleet(rng, het)
    lens = [len(s) for s in fleet]
    kw = max(8, int(0.5 * min(lens)))
    mean_n = float(np.mean(lens))
    rows = []
    for lv in LEVELS:
        noisy = [s + rng.normal(0, lv, len(s)) for s in fleet]
        X, y, u = build(noisy, kw, mean_n)
        if len(X) < 20:
            continue
        e, b = loo(X, y, u)
        g = info(noisy, rng)
        if all(np.isfinite(v) and v > 0 for v in (e, b, g)):
            rows.append(dict(level=lv, G=g, rmse=e, baseline=b, skill=1 - e / b))
    fit = [r for r in rows if r["skill"] > 0.05]
    if len(fit) < 4:
        print(f"{het:>11.2f}   too few points with skill")
        out.append(dict(hetero=het, rows=rows))
        continue
    a = np.log([r["G"] for r in fit]); c = np.log([r["rmse"] for r in fit])
    slope, inter = np.polyfit(a, c, 1)
    resid = c - (slope * a + inter)
    se = float(np.sqrt((resid ** 2).sum() / (len(fit) - 2) /
                       ((a - a.mean()) ** 2).sum()))
    rr = float(np.corrcoef(a, c)[0, 1])
    span = max(r["G"] for r in fit) / min(r["G"] for r in fit)
    best = max(r["skill"] for r in fit)
    out.append(dict(hetero=het, rows=rows, slope=float(slope), se=se, r=rr,
                    span=float(span), best_skill=float(best)))
    print(f"{het:>11.2f}{len(fit):>8}{span:>9.0f}x{slope:>10.3f}"
          f"{1.96*se:>8.3f}{rr:>8.2f}{best:>12.2f}")
print("-" * 67)
ok = [o for o in out if "slope" in o]
if ok:
    print("\nCramer-Rao predicts -0.500 at every heterogeneity level.")
    for o in ok:
        inside = abs(o["slope"] + 0.5) < 1.96 * o["se"]
        print(f"  CV of life = {o['hetero']:.2f}: slope {o['slope']:+.3f}, "
              f"{'consistent with' if inside else 'far from'} the prediction")
    print("\nIf the slope recovers as heterogeneity falls, the bound is exact")
    print("for what it bounds and the shortfall on a real fleet is the trend")
    print("the model must also infer -- not a defect in the bound.")
json.dump(out, open(os.path.join(HERE, "heterogeneity_sweep.json"), "w"),
          indent=2, default=float)
