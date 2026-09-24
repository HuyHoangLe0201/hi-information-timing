"""
The end-to-end test, on a fleet built to make it answerable.

Neither public dataset can carry it. Bearing records are long enough that G is
96-99% signal, but seventeen units with a twelve-fold spread of life cannot
support the regression -- no indicator beat a predictor that ignores the
indicator. Turbofan fleets are large and their lifetimes tight enough for the
regression to work, but at 194 samples per record G is only about half signal,
so adding noise does not move it and the sweep has nothing to sweep.

Both conditions can be met at once in simulation. The fleet here has sixty
units, records long enough for the density to be measurable, and a spread of
life wide enough to make remaining life worth predicting but not so wide that
the fleet mean is uninformative. Units differ in rate and in exponent, as a real
fleet does, so the model faces genuine unit-to-unit variation rather than one
curve repeated.

The prediction under test is the Cramer-Rao floor, applied across a noise sweep
with everything else held fixed:

    log(RUL error)  =  const  -  (1/2) log G(tau_0).

Only points where the model retains real skill are fitted: once the error has
saturated at the no-indicator baseline it can fall no further, and the relation
must break there for reasons that have nothing to do with the bound.
"""
import os
import json
import numpy as np
from nonparam import weighted_density, estimate_floor

HERE = os.path.dirname(os.path.abspath(__file__))
N_UNITS = 60
N_MEAN = 700
LIFE_CV = 0.25
NFEAT, RIDGE, N_POS = 12, 1e-3, 16
LEVELS = (0.01, 0.02, 0.04, 0.08, 0.16, 0.32, 0.64, 1.28)
SEED = 20260827


def make_fleet(rng):
    """Units differing in life, in rate and in exponent."""
    lives = np.clip(rng.normal(N_MEAN, LIFE_CV * N_MEAN, N_UNITS),
                    0.4 * N_MEAN, 2.0 * N_MEAN).astype(int)
    fleet = []
    for L in lives:
        tau = np.arange(1, L + 1) / L
        beta = float(np.clip(rng.normal(2.5, 0.5), 1.2, 5.0))
        amp = float(np.exp(rng.normal(0.0, 0.25)))
        fleet.append(amp * tau ** beta)
    return fleet


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
    G, S = [], []
    for raw in series:
        dens, res = weighted_density(np.asarray(raw, float))
        if dens is None:
            continue
        fl = estimate_floor(dens, res, "surrogate", rng, reps=6)
        d = np.clip(dens - fl, 0.0, None)
        k0 = int(round(t0 * len(d))) - 1
        if k0 > 1:
            G.append(float(d[:k0 + 1].sum()))
            tot = float(dens[:k0 + 1].sum())
            S.append(G[-1] / tot if tot > 0 else np.nan)
    return (float(np.median(G)) if G else np.nan,
            float(np.median(S)) if S else np.nan)


rng = np.random.default_rng(SEED)
fleet = make_fleet(rng)
lens = [len(s) for s in fleet]
kw = max(8, int(0.5 * min(lens)))
mean_n = float(np.mean(lens))
L = np.array(lens, float)
print(f"synthetic fleet: {N_UNITS} units, life {int(L.min())}-{int(L.max())} "
      f"({L.max()/L.min():.1f}x), CV {L.std(ddof=1)/L.mean():.2f}")
print(f"window {kw} samples\n")
print(f"{'noise':>7}{'G(0.60)':>12}{'signal %':>10}{'RUL RMSE':>11}"
      f"{'baseline':>11}{'skill':>8}")
print("-" * 59)
rows = []
for lv in LEVELS:
    noisy = [s + rng.normal(0, lv, len(s)) for s in fleet]
    X, y, u = build(noisy, kw, mean_n)
    if len(X) < 20:
        continue
    e, b = loo(X, y, u)
    g, sf = info(noisy, rng)
    if not all(np.isfinite(v) and v > 0 for v in (e, b, g)):
        continue
    rows.append(dict(level=lv, G=g, signal_frac=sf, rmse=e, baseline=b,
                     skill=1 - e / b))
    print(f"{lv:>7.2f}{g:>12.3g}{100*sf:>9.0f}%{e:>11.4f}{b:>11.4f}"
          f"{1-e/b:>8.2f}")
print("-" * 59)

fit = [r for r in rows if r["skill"] > 0.05]
if len(fit) >= 4:
    a = np.log([r["G"] for r in fit]); c = np.log([r["rmse"] for r in fit])
    slope, inter = np.polyfit(a, c, 1)
    rr = float(np.corrcoef(a, c)[0, 1])
    n = len(fit)
    resid = c - (slope * a + inter)
    se = float(np.sqrt((resid ** 2).sum() / (n - 2) /
                       ((a - a.mean()) ** 2).sum()))
    span = max(r["G"] for r in fit) / min(r["G"] for r in fit)
    print(f"\n{n} points with real skill, G spanning {span:.0f}x")
    print(f"slope {slope:+.3f} +/- {1.96*se:.3f}   r {rr:+.3f}")
    print(f"Cramer-Rao predicts -0.500 -- "
          f"{'inside' if abs(slope + 0.5) < 1.96 * se else 'outside'} "
          f"the interval")
    json.dump(dict(rows=rows, slope=float(slope), se=se, r=rr, span=span),
              open(os.path.join(HERE, "predicts_synth.json"), "w"), indent=2)
else:
    print("\ntoo few points with real skill to fit")
    json.dump(dict(rows=rows), open(os.path.join(HERE, "predicts_synth.json"),
                                    "w"), indent=2)
