"""
Does the information curve predict what a trained model achieves? (corrected)

Two faults in the first attempt had to be fixed before the question could be
asked at all.

The window was a fixed fraction of the FLEET's mean life, so on PRONOSTIA --
where lifetimes span 230 to 2803 samples -- it did not fit the short units,
which were silently dropped, and covered only a tenth of the long ones. Every
bearing indicator then scored worse than a predictor that ignores the indicator
entirely, which is a broken experiment rather than a finding. The window is now
a fixed absolute length that fits the shortest unit in the set.

And the slope was fitted across domains, where it came out +0.219 -- the wrong
sign. That is Simpson's paradox: bearings have both larger G and larger error
than turbofans, for reasons that have nothing to do with the relationship being
tested, so pooling measures the gap between domains. Slopes are now fitted
within each domain only.

The prediction stands: log(RUL error) = const - (1/2) log G(tau_0), a slope of
-1/2 within each domain, and only among indicators whose model beats the
no-indicator baseline. An indicator the model cannot exploit at all carries no
information about the framework's claim.
"""
import os
import json
import numpy as np
from nonparam import weighted_density, estimate_floor

HERE = os.path.dirname(os.path.abspath(__file__))
NFEAT = 12
RIDGE = 1e-3
N_POS = 14
SEED = 4242


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
    slope = np.polyfit(t, seg, 1)[0]
    return np.concatenate([g, [slope, seg.std()]])


def build(series):
    """Windows of one absolute length that every unit can supply."""
    lens = [len(s) for s in series]
    kw = max(8, int(0.5 * min(lens)))        # fits the shortest unit
    mean_n = float(np.mean(lens))
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
            if not np.all(np.isfinite(f)):
                continue
            X.append(f)
            y.append((n - 1 - k0) / mean_n)
            u.append(i)
    return np.array(X), np.array(y), np.array(u), kw


def loo(X, y, u):
    """Leave-one-unit-out ridge, and the no-indicator baseline on the same split."""
    em, eb = [], []
    for i in np.unique(u):
        tr, te = u != i, u == i
        if tr.sum() < 8 or te.sum() == 0:
            continue
        A, b = X[tr], y[tr]
        mu, sd = A.mean(0), A.std(0)
        sd[sd <= 0] = 1.0
        A = np.column_stack([(A - mu) / sd, np.ones(tr.sum())])
        w = np.linalg.solve(A.T @ A + RIDGE * np.eye(A.shape[1]), A.T @ b)
        B = np.column_stack([(X[te] - mu) / sd, np.ones(te.sum())])
        em.append(B @ w - y[te])
        eb.append(y[te] - b.mean())
    if not em:
        return np.nan, np.nan
    rms = lambda v: float(np.sqrt(np.mean(np.concatenate(v) ** 2)))
    return rms(em), rms(eb)


def information(series, rng, t0=0.60):
    G = []
    for raw in series:
        dens, res = weighted_density(np.asarray(raw, float))
        if dens is None:
            continue
        d = np.clip(dens - estimate_floor(dens, res, "surrogate", rng), 0.0, None)
        k0 = int(round(t0 * len(d))) - 1
        if k0 > 1:
            G.append(float(d[:k0 + 1].sum()))
    return float(np.median(G)) if G else np.nan


rng = np.random.default_rng(SEED)
DOMAINS = {}

z = np.load(os.path.join(HERE, "feat_raw.npz"), allow_pickle=True)
FN = [str(s) for s in z["featnames"]]
I = {k: i for i, k in enumerate(FN)}
BK = sorted(k for k in z.files if k != "featnames")
DOMAINS["bearing"] = {
    lab: [sum(z[b][:, I[p]] for p in parts) for b in BK]
    for lab, parts in (("rms", ["rms"]), ("peak", ["peak"]), ("kurt", ["kurt"]),
                       ("0--2k", ["b0_1", "b1_2"]), ("2--4k", ["b2_4"]),
                       ("4--10k", ["b4_6", "b6_8", "b8_10"]),
                       ("10--12.8k", ["b10_12.8"]))}

zt = np.load(os.path.join(HERE, "cmapss_FD001.npz"))
un = sorted({k.split("__")[0] for k in zt.files})[:60]
DOMAINS["turbofan"] = {f"s{c}": [zt[f"{u}__sensors"][:, c - 1].astype(float)
                                 for u in un] for c in (2, 3, 4, 7, 9, 11, 12,
                                                        15, 17, 20, 21)}

out = {}
for dom, sets in DOMAINS.items():
    print(f"\n=== {dom} ===\n")
    rows = []
    kw_used = None
    for lab, series in sets.items():
        X, y, u, kw = build(series)
        kw_used = kw
        if len(X) < 20:
            continue
        e, base = loo(X, y, u)
        g = information(series, rng)
        if not all(np.isfinite(v) and v > 0 for v in (e, base, g)):
            continue
        rows.append(dict(indicator=lab, units=int(len(np.unique(u))),
                         G=g, rmse=e, baseline=base, skill=1 - e / base))
    if not rows:
        continue
    print(f"window: {kw_used} samples, fitting the shortest unit\n")
    print(f"{'indicator':<12}{'units':>6}{'G(0.60)':>12}{'RUL RMSE':>11}"
          f"{'baseline':>11}{'skill':>8}")
    print("-" * 60)
    for r in sorted(rows, key=lambda r: -r["skill"]):
        print(f"{r['indicator']:<12}{r['units']:>6}{r['G']:>12.3g}"
              f"{r['rmse']:>11.4f}{r['baseline']:>11.4f}{r['skill']:>8.2f}")
    print("-" * 60)
    useful = [r for r in rows if r["skill"] > 0]
    print(f"{len(useful)} of {len(rows)} indicators beat the no-indicator "
          f"baseline")
    if len(useful) >= 4:
        a = np.log([r["G"] for r in useful])
        b = np.log([r["rmse"] for r in useful])
        s = float(np.polyfit(a, b, 1)[0])
        rr = float(np.corrcoef(a, b)[0, 1])
        ri = np.argsort(np.argsort([-r["G"] for r in useful]))
        re_ = np.argsort(np.argsort([r["rmse"] for r in useful]))
        rk = float(np.corrcoef(ri, re_)[0, 1])
        print(f"among those: slope {s:+.3f} (predicted -0.500), r {rr:+.3f}, "
              f"rank agreement {rk:+.2f}")
        out[dom] = dict(rows=rows, slope=s, r=rr, rank=rk, n_useful=len(useful))
    else:
        print("too few usable indicators to fit a slope")
        out[dom] = dict(rows=rows, n_useful=len(useful))

json.dump(out, open(os.path.join(HERE, "predicts_rul2.json"), "w"), indent=2,
          default=float)
print("\nA slope near -0.5 would validate the framework end to end; near zero")
print("would mean the information curve describes the record but not what a")
print("model can extract from it.")
