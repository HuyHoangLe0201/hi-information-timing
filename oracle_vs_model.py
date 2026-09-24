"""
Is the shortfall in the bound, or in the estimator?

Raising the information 119-fold cut the RUL error by 1.4x, not the 10.9x the
Cramer-Rao floor predicts. Two explanations have been tested and eliminated:
heterogeneity in the trend shape (the slope was -0.141 even with every unit on
an identical curve) and spread of life across the fleet (the slope was -0.06 to
-0.11 at every spread from 0.02 to 0.50).

The remaining candidate is the estimator. The bound constrains the maximum
likelihood estimator for a KNOWN trend. A ridge regression on window features is
not that: its error contains its own approximation bias, and bias does not fall
when observation noise does. If so, the bound would be exact for what it bounds
and simply not predictive of what a practical model achieves -- which is a
different, and more useful, statement than either.

The discriminating experiment gives one estimator the trend for free. The oracle
matches the noisy window against the unit's own clean trajectory and reports the
position minimising squared error: this is the ML estimator of the damage clock
with the trend known, the very quantity the bound governs. If the oracle tracks
-1/2 while the ridge does not, the gap is the estimator's, not the theory's.
"""
import os
import json
import numpy as np
from nonparam import weighted_density, estimate_floor

HERE = os.path.dirname(os.path.abspath(__file__))
N_UNITS, N_MEAN, LIFE_CV, BETA_SD = 60, 700, 0.25, 0.25
NFEAT, RIDGE, N_POS = 12, 1e-3, 16
LEVELS = (0.01, 0.03, 0.08, 0.20, 0.50, 1.20)
SEED = 5150


def make_fleet(rng):
    lives = np.clip(rng.normal(N_MEAN, LIFE_CV * N_MEAN, N_UNITS),
                    0.3 * N_MEAN, 2.5 * N_MEAN).astype(int)
    out = []
    for L in lives:
        tau = np.arange(1, L + 1) / L
        b = float(np.clip(rng.normal(2.5, BETA_SD), 1.2, 6.0))
        a = float(np.exp(rng.normal(0.0, BETA_SD)))
        out.append(a * tau ** b)
    return out


def norm(x, frac=0.05):
    x = np.asarray(x, float)
    k = max(3, int(frac * len(x)))
    b = np.median(x[:k])
    s = np.median(np.abs(x[:k] - b)) * 1.4826
    if not np.isfinite(s) or s <= 0:
        s = float(np.std(x)) or 1.0
    return (x - b) / s


def feats(seg):
    g = np.interp(np.linspace(0, 1, NFEAT), np.linspace(0, 1, len(seg)), seg)
    t = np.linspace(0, 1, len(seg))
    return np.concatenate([g, [np.polyfit(t, seg, 1)[0], seg.std()]])


def positions(n, kw):
    return np.unique(np.linspace(kw, n - 2, N_POS).astype(int))


def ridge_error(noisy, kw, mean_n):
    X, y, u = [], [], []
    for i, raw in enumerate(noisy):
        x = norm(raw); n = len(x)
        if n <= kw + 4:
            continue
        for k0 in positions(n, kw):
            seg = x[k0 - kw + 1:k0 + 1]
            f = feats(seg)
            if np.all(np.isfinite(f)):
                X.append(f); y.append((n - 1 - k0) / mean_n); u.append(i)
    X, y, u = np.array(X), np.array(y), np.array(u)
    em, eb = [], []
    for i in np.unique(u):
        tr, te = u != i, u == i
        if tr.sum() < 8 or te.sum() == 0:
            continue
        A, b = X[tr], y[tr]
        mu, sd = A.mean(0), A.std(0); sd[sd <= 0] = 1.0
        A2 = np.column_stack([(A - mu) / sd, np.ones(tr.sum())])
        w = np.linalg.solve(A2.T @ A2 + RIDGE * np.eye(A2.shape[1]), A2.T @ b)
        B = np.column_stack([(X[te] - mu) / sd, np.ones(te.sum())])
        em.append(B @ w - y[te]); eb.append(y[te] - b.mean())
    rms = lambda v: float(np.sqrt(np.mean(np.concatenate(v) ** 2)))
    return rms(em), rms(eb)


def oracle_error(clean, noisy, kw, mean_n):
    """ML position estimate with the unit's own trend known exactly.

    Both series must be in the SAME units. An earlier version normalised each
    with norm(), which scales by the spread of the first 5% of the record: on a
    clean power law that stretch is almost constant, the robust scale collapses
    to zero and the fallback picks the whole-series standard deviation instead.
    The clean curve and the noisy one were then divided by wildly different
    numbers and the match compared incomparable scales, which is why the oracle
    came out worse than predicting the fleet mean. Matching in raw units removes
    the problem; a self-check below confirms the error vanishes at zero noise.
    """
    err = []
    for c, x in zip(clean, noisy):
        n = len(c)
        if n <= kw + 4:
            continue
        for k0 in positions(n, kw):
            seg = x[k0 - kw + 1:k0 + 1]
            # slide the window over every admissible position on the clean curve
            best, bk = np.inf, k0
            for j in range(kw - 1, n):
                d = seg - c[j - kw + 1:j + 1]
                s = float(d @ d)
                if s < best:
                    best, bk = s, j
            err.append(((n - 1 - bk) - (n - 1 - k0)) / mean_n)
    return float(np.sqrt(np.mean(np.square(err)))) if err else np.nan


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


rng = np.random.default_rng(SEED)
fleet = make_fleet(rng)
lens = [len(s) for s in fleet]
kw = max(8, int(0.5 * min(lens)))
mean_n = float(np.mean(lens))

# self-check: with no noise at all the oracle must recover the position exactly
chk = oracle_error(fleet, fleet, kw, mean_n)
print(f"oracle self-check at zero noise: error {chk:.2e} "
      f"({'passes' if chk < 1e-9 else 'FAILS -- the matching is broken'})\n")

print(f"fleet: {N_UNITS} units, window {kw} samples\n")
print(f"{'noise':>7}{'G(0.60)':>12}{'oracle':>11}{'ridge':>10}"
      f"{'baseline':>11}")
print("-" * 51)
rows = []
for lv in LEVELS:
    noisy = [s + rng.normal(0, lv, len(s)) for s in fleet]
    r, b = ridge_error(noisy, kw, mean_n)
    o = oracle_error(fleet, noisy, kw, mean_n)     # both in raw units
    g = info(noisy, rng)
    if not all(np.isfinite(v) and v > 0 for v in (r, b, o, g)):
        continue
    rows.append(dict(level=lv, G=g, oracle=o, ridge=r, baseline=b))
    print(f"{lv:>7.2f}{g:>12.3g}{o:>11.4f}{r:>10.4f}{b:>11.4f}")
print("-" * 51)


def fit(key):
    v = [r for r in rows if r[key] > 0]
    if len(v) < 4:
        return None
    a = np.log([r["G"] for r in v]); c = np.log([r[key] for r in v])
    s, i0 = np.polyfit(a, c, 1)
    res = c - (s * a + i0)
    se = float(np.sqrt((res ** 2).sum() / (len(v) - 2) /
                       ((a - a.mean()) ** 2).sum()))
    return float(s), se, float(np.corrcoef(a, c)[0, 1])


print()
for key, lab in (("oracle", "oracle, trend known"), ("ridge", "ridge on windows")):
    f = fit(key)
    if f:
        s, se, rr = f
        ok = abs(s + 0.5) < 1.96 * se
        print(f"  {lab:<22} slope {s:+.3f} +/- {1.96*se:.3f}  r {rr:+.2f}"
              f"   {'consistent with -0.5' if ok else 'far from -0.5'}")
print("\nIf the oracle tracks -1/2 and the ridge does not, the bound is exact")
print("for the estimator it bounds, and the gap on a practical model is that")
print("model's own inefficiency -- which the bound was never claiming to cover.")
json.dump(rows, open(os.path.join(HERE, "oracle_vs_model.json"), "w"),
          indent=2, default=float)
