"""
The decisive form of the test: vary the information and nothing else.

Comparing indicators cannot settle whether the information curve predicts model
error. Where the model works -- turbofan sensors -- G spans a factor of 1.6, and
a slope fitted over that range is mostly noise. Where G spans a factor of forty
-- bearing channels -- the model barely beats a predictor that ignores the
indicator, so its errors carry no signal to correlate against. Indicators also
differ in trend shape, noise structure and fleet variability, any of which could
drive the comparison.

Adding noise to one indicator removes all of that. The trend, the fleet, the
model and the evaluation are held fixed; only the noise level moves, and it
moves G over orders of magnitude. The Cramer-Rao floor then predicts

    log(RUL error)  =  const  -  (1/2) log G,

with no confounds left to explain a deviation.

Two caveats are built in. At high noise the model degenerates to predicting the
fleet mean, so the error saturates at the baseline and the relation must break;
only points with real skill are fitted. And G is measured on the same noisy
record the model sees, not on the clean one, so both sides of the relation refer
to the same object.
"""
import os
import json
import numpy as np
from nonparam import weighted_density, estimate_floor

HERE = os.path.dirname(os.path.abspath(__file__))
NFEAT, RIDGE, N_POS = 12, 1e-3, 14
LEVELS = (0.0, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0)
REPS = 3
SEED = 31337


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
    if not em:
        return np.nan, np.nan
    rms = lambda v: float(np.sqrt(np.mean(np.concatenate(v) ** 2)))
    return rms(em), rms(eb)


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


def sweep(name, series, rng):
    lens = [len(s) for s in series]
    kw = max(8, int(0.5 * min(lens)))
    mean_n = float(np.mean(lens))
    scales = [float(np.median(np.abs(np.diff(np.asarray(s, float)))) * 1.4826)
              for s in series]
    print(f"\n=== {name} ===   window {kw} samples, "
          f"{len(series)} units\n")
    print(f"{'noise x':>9}{'G(0.60)':>12}{'RUL RMSE':>11}{'baseline':>11}"
          f"{'skill':>8}")
    print("-" * 51)
    rows = []
    for lv in LEVELS:
        Gs, Es, Bs = [], [], []
        for _ in range(REPS if lv > 0 else 1):
            noisy = [np.asarray(s, float) +
                     rng.normal(0, lv * sc, len(s))
                     for s, sc in zip(series, scales)]
            X, y, u = build(noisy, kw, mean_n)
            if len(X) < 20:
                continue
            e, b = loo(X, y, u)
            g = info(noisy, rng)
            if all(np.isfinite(v) and v > 0 for v in (e, b, g)):
                Gs.append(g); Es.append(e); Bs.append(b)
        if not Gs:
            continue
        g, e, b = (float(np.median(v)) for v in (Gs, Es, Bs))
        rows.append(dict(level=lv, G=g, rmse=e, baseline=b, skill=1 - e / b))
        print(f"{lv:>9.1f}{g:>12.3g}{e:>11.4f}{b:>11.4f}{1-e/b:>8.2f}")
    print("-" * 51)
    fit = [r for r in rows if r["skill"] > 0.05]
    if len(fit) >= 4:
        a = np.log([r["G"] for r in fit]); c = np.log([r["rmse"] for r in fit])
        s = float(np.polyfit(a, c, 1)[0])
        rr = float(np.corrcoef(a, c)[0, 1])
        span = max(r["G"] for r in fit) / min(r["G"] for r in fit)
        print(f"{len(fit)} points with real skill, G spanning {span:.0f}x")
        print(f"slope {s:+.3f}  (Cramer-Rao predicts -0.500)   r {rr:+.3f}")
        return dict(name=name, rows=rows, slope=s, r=rr, span=span,
                    n_fit=len(fit))
    print("too few points with real skill to fit")
    return dict(name=name, rows=rows)


rng = np.random.default_rng(SEED)
out = []

zt = np.load(os.path.join(HERE, "cmapss_FD001.npz"))
un = sorted({k.split("__")[0] for k in zt.files})[:60]
for c, lab in ((15, "turbofan s15"), (4, "turbofan s4")):
    out.append(sweep(lab, [zt[f"{u}__sensors"][:, c - 1].astype(float)
                           for u in un], rng))

z = np.load(os.path.join(HERE, "feat_raw.npz"), allow_pickle=True)
FN = [str(s) for s in z["featnames"]]
I = {k: i for i, k in enumerate(FN)}
BK = sorted(k for k in z.files if k != "featnames")
out.append(sweep("bearing 0--2k",
                 [z[b][:, I["b0_1"]] + z[b][:, I["b1_2"]] for b in BK], rng))

json.dump(out, open(os.path.join(HERE, "predicts_sweep.json"), "w"), indent=2,
          default=float)
print("\nWith the trend, the fleet, the model and the evaluation all held")
print("fixed, a slope away from -0.5 cannot be blamed on differences between")
print("indicators. It would mean the bound describes information the model")
print("cannot convert into accuracy.")
