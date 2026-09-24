"""
Does the information curve predict what a trained model actually achieves?

The whole point of the framework is that indicators can be ranked before any
RUL model exists. That has not been tested. It is testable, and it is sharp: the
Cramer-Rao floor says the achievable standard error on the damage clock goes as
G(tau_0)^{-1/2}, so across indicators

    log(RUL error)  =  const  -  (1/2) log G(tau_0),

a line of slope -1/2. A slope near -1/2 validates the framework end to end. A
slope near zero would mean the information curve describes something real but
not what a model can do with it, and the framework would be elegant and useless.

The model is deliberately fixed and simple -- ridge on the trailing window,
leave-one-unit-out -- so that what varies between conditions is the indicator
and not the estimator. A strong learner would confound the two: it could recover
information a weak one misses, and the comparison would then be about model
capacity.

Each unit contributes windows at several ages. The target is remaining life in
units of the fleet's mean life, so errors are comparable across datasets whose
absolute lifetimes differ by orders of magnitude.
"""
import os
import json
import numpy as np
from nonparam import weighted_density, estimate_floor

HERE = os.path.dirname(os.path.abspath(__file__))
TAU0 = (0.30, 0.45, 0.60, 0.75)
W = 0.25                 # trailing window, as a fraction of life
NFEAT = 12               # window resampled to this many points
RIDGE = 1e-3
SEED = 606


def baseline_norm(x, frac=0.05):
    x = np.asarray(x, float)
    k = max(3, int(frac * len(x)))
    b = np.median(x[:k])
    s = np.median(np.abs(x[:k] - b)) * 1.4826
    if not np.isfinite(s) or s <= 0:
        s = np.std(x) or 1.0
    return (x - b) / s


def features(x, k0, kw):
    seg = x[max(0, k0 - kw + 1):k0 + 1]
    if len(seg) < 4:
        return None
    g = np.interp(np.linspace(0, 1, NFEAT), np.linspace(0, 1, len(seg)), seg)
    slope = np.polyfit(np.linspace(0, 1, len(seg)), seg, 1)[0]
    return np.concatenate([g, [slope, seg.std()]])


def build(series, n_pos=8):
    """Windows at absolute positions, with unit labels for the split.

    An earlier version placed the windows at fixed FRACTIONS of each unit's
    life and made the target the remaining fraction. That leaks: the target
    then takes one value per fraction and is fixed by where the window was cut,
    so the model only has to recognise which of a few ages it is looking at --
    and the fraction is not knowable at prediction time anyway, since it needs
    the life the model is trying to predict.

    Here the window ends at an absolute sample index and the target is the
    absolute remaining life, expressed in units of the fleet's mean life so
    that errors compare across datasets. The spread of lifetimes within the
    fleet is what makes the problem non-trivial, exactly as in deployment.
    """
    lens = [len(s) for s in series]
    mean_n = float(np.mean(lens))
    kw = max(6, int(round(W * mean_n)))          # one window length for all
    X, y, u = [], [], []
    for i, raw in enumerate(series):
        x = baseline_norm(raw)
        n = len(x)
        if n <= kw + 4:
            continue
        for k0 in np.unique(np.linspace(kw, n - 2, n_pos).astype(int)):
            f = features(x, int(k0), kw)
            if f is None or not np.all(np.isfinite(f)):
                continue
            X.append(f)
            y.append((n - 1 - k0) / mean_n)      # remaining life, fleet units
            u.append(i)
    return np.array(X), np.array(y), np.array(u)


def loo_error(X, y, u):
    """Leave-one-unit-out ridge; error in units of the fleet's mean life."""
    err = []
    for i in np.unique(u):
        tr, te = u != i, u == i
        if tr.sum() < 8 or te.sum() == 0:
            continue
        A, b = X[tr], y[tr]
        mu, sd = A.mean(0), A.std(0)
        sd[sd <= 0] = 1.0
        A = (A - mu) / sd
        A = np.column_stack([A, np.ones(len(A))])
        w = np.linalg.solve(A.T @ A + RIDGE * np.eye(A.shape[1]), A.T @ b)
        B = (X[te] - mu) / sd
        B = np.column_stack([B, np.ones(len(B))])
        err.append(B @ w - y[te])
    if not err:
        return np.nan
    e = np.concatenate(err)
    return float(np.sqrt(np.mean(e ** 2)))


def information(series, rng, t0=0.60):
    """G accumulated by tau_0, floor removed; median over units."""
    G = []
    for raw in series:
        dens, res = weighted_density(np.asarray(raw, float))
        if dens is None:
            continue
        d = np.clip(dens - estimate_floor(dens, res, "surrogate", rng), 0.0, None)
        n = len(d)
        k0 = int(round(t0 * n)) - 1
        if k0 < 1:
            continue
        G.append(float(d[:k0 + 1].sum()))
    return float(np.median(G)) if G else np.nan


rng = np.random.default_rng(SEED)
SETS = {}

z = np.load(os.path.join(HERE, "feat_raw.npz"), allow_pickle=True)
FN = [str(s) for s in z["featnames"]]
I = {k: i for i, k in enumerate(FN)}
BK = sorted(k for k in z.files if k != "featnames")
for lab, parts in (("bearing rms", ["rms"]), ("bearing peak", ["peak"]),
                   ("bearing kurt", ["kurt"]),
                   ("bearing 0--2k", ["b0_1", "b1_2"]),
                   ("bearing 2--4k", ["b2_4"]),
                   ("bearing 4--10k", ["b4_6", "b6_8", "b8_10"]),
                   ("bearing 10--12.8k", ["b10_12.8"])):
    SETS[lab] = [sum(z[b][:, I[p]] for p in parts) for b in BK]

zt = np.load(os.path.join(HERE, "cmapss_FD001.npz"))
un = sorted({k.split("__")[0] for k in zt.files})[:60]
for c in (4, 9, 11, 15, 20, 21):
    SETS[f"turbofan s{c}"] = [zt[f"{u}__sensors"][:, c - 1].astype(float)
                              for u in un]

def constant_baseline(y, u):
    """What a predictor that ignores the indicator entirely achieves.

    It answers the fleet's mean remaining life every time. Any indicator whose
    error does not beat this carries nothing the model could use, and its place
    in the ranking would be meaningless.
    """
    err = []
    for i in np.unique(u):
        tr, te = u != i, u == i
        if tr.sum() < 8 or te.sum() == 0:
            continue
        err.append(y[te] - y[tr].mean())
    return float(np.sqrt(np.mean(np.concatenate(err) ** 2))) if err else np.nan


print("model error against the information available, at tau_0 = 0.60")
print("(the baseline predicts the fleet mean and uses no indicator)\n")
print(f"{'indicator':<20}{'units':>6}{'windows':>9}{'G(0.60)':>12}"
      f"{'RUL RMSE':>11}{'baseline':>11}{'skill':>8}")
print("-" * 77)
rows = []
for lab, series in SETS.items():
    X, y, u = build(series)
    if len(X) < 20:
        continue
    e = loo_error(X, y, u)
    g = information(series, rng)
    if not (np.isfinite(e) and np.isfinite(g) and g > 0):
        continue
    c = constant_baseline(y, u)
    rows.append(dict(indicator=lab, units=int(len(np.unique(u))),
                     windows=int(len(X)), G=g, rmse=e, baseline=c,
                     skill=1.0 - e / c if c > 0 else np.nan))
    print(f"{lab:<20}{len(np.unique(u)):>6}{len(X):>9}{g:>12.3g}{e:>11.4f}"
          f"{c:>11.4f}{1-e/c:>8.2f}")
print("-" * 77)

lg = np.log([r["G"] for r in rows])
le = np.log([r["rmse"] for r in rows])
slope, intercept = np.polyfit(lg, le, 1)
r = float(np.corrcoef(lg, le)[0, 1])
print(f"\nlog(RUL error) against log G:  slope {slope:+.3f}, r {r:+.3f}")
print(f"the Cramer-Rao floor predicts a slope of -0.500\n")

# within domain, where the trend family and the noise structure are shared
for dom in ("bearing", "turbofan"):
    sub = [x for x in rows if x["indicator"].startswith(dom)]
    if len(sub) < 4:
        continue
    a = np.log([x["G"] for x in sub]); b = np.log([x["rmse"] for x in sub])
    s = np.polyfit(a, b, 1)[0]
    rr = float(np.corrcoef(a, b)[0, 1])
    print(f"  {dom:<10} {len(sub)} indicators: slope {s:+.3f}, r {rr:+.3f}")

print("\nrank agreement between the two orderings:")
ri = np.argsort(np.argsort([-x["G"] for x in rows]))
re_ = np.argsort(np.argsort([x["rmse"] for x in rows]))
print(f"  {float(np.corrcoef(ri, re_)[0, 1]):+.2f}  "
      f"(most informative first, against lowest error first)")
json.dump(dict(rows=rows, slope=float(slope), r=r),
          open(os.path.join(HERE, "predicts_rul.json"), "w"), indent=2)
