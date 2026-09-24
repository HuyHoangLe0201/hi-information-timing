"""
Does choosing a distribution indicator actually improve RUL prediction?

The framework says distribution indicators carry their information earlier. That
is a statement about the record, not about any model, and the paper is careful
to say so. But it implies something a model can be asked: predicting remaining
life from an EARLY stretch of the record should work better with a distribution
indicator than with a level one, and the advantage should shrink or vanish late,
when the level indicators have finally accumulated their own information.

If it holds, the finding stops being descriptive. If the advantage is absent, or
present late as well as early, then the earliest-usable-age ordering does not
translate into anything a practitioner gains by acting on it, and the paper
should say that too.

The setup is the leave-one-unit-out ridge used earlier, unchanged, so what
varies between conditions is the indicator and not the estimator. Windows are
drawn from an early or a late half of each record, and the same no-indicator
baseline -- predict the fleet's mean remaining life -- is computed on the same
split so the skill scores compare.
"""
import os
import json
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
NFEAT, RIDGE, N_POS = 12, 1e-3, 12
EPS = 1e-30
SEED = 4242


def norm(P):
    return P / np.clip(P.sum(axis=1, keepdims=True), EPS, None)


def entropy(P):
    p = norm(P)
    return -(p * np.log(np.clip(p, EPS, None))).sum(axis=1)


def spread(P, f):
    p = norm(P)
    c = (p * f).sum(axis=1, keepdims=True)
    return np.sqrt((p * (f - c) ** 2).sum(axis=1))


def gini(P):
    p = np.sort(norm(P), axis=1)
    n = p.shape[1]
    return (2 * (p * np.arange(1, n + 1)).sum(axis=1)) \
        / np.clip(p.sum(axis=1), EPS, None) / n - (n + 1) / n


MEASURES = [
    ("spectral entropy", "distribution", lambda P, f: entropy(P)),
    ("spectral centroid", "distribution", lambda P, f: (norm(P) * f).sum(axis=1)),
    ("spectral spread", "distribution", lambda P, f: spread(P, f)),
    ("spectral Gini", "distribution", lambda P, f: gini(P)),
    ("high/low ratio", "distribution",
     lambda P, f: P[:, -8:].sum(axis=1) / np.clip(P[:, :8].sum(axis=1), EPS, None)),
    ("total power", "amount", lambda P, f: P.sum(axis=1)),
    ("peak band power", "amount", lambda P, f: P.max(axis=1)),
    ("high-band power", "amount", lambda P, f: P[:, -8:].sum(axis=1)),
    ("log total power", "amount",
     lambda P, f: np.log(np.clip(P.sum(axis=1), EPS, None))),
]


def baseline_norm(x, frac=0.05):
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


def build(series, half):
    """Windows drawn from the early or the late half of each record."""
    lens = [len(s) for s in series]
    kw = max(8, int(0.25 * min(lens)))
    mean_n = float(np.mean(lens))
    X, y, u = [], [], []
    for i, raw in enumerate(series):
        x = baseline_norm(raw)
        n = len(x)
        if n <= kw + 6:
            continue
        lo, hi = (kw, n // 2) if half == "early" else (n // 2, n - 2)
        if hi - lo < 4:
            continue
        for k0 in np.unique(np.linspace(lo, hi, N_POS).astype(int)):
            seg = x[k0 - kw + 1:k0 + 1]
            if len(seg) < 8 or not np.all(np.isfinite(seg)):
                continue
            f = feats(seg)
            if np.all(np.isfinite(f)):
                X.append(f)
                y.append((n - 1 - k0) / mean_n)
                u.append(i)
    return np.array(X), np.array(y), np.array(u)


def skill(X, y, u):
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
        em.append(B @ w - y[te])
        eb.append(y[te] - b.mean())
    if not em:
        return np.nan
    rms = lambda v: float(np.sqrt(np.mean(np.concatenate(v) ** 2)))
    a, c = rms(em), rms(eb)
    return 1.0 - a / c if c > 0 else np.nan


RIGS = {"PRONOSTIA": "fineband.npz", "XJTU": "fineband_xjtu.npz"}
out = {}
for rig, fn in RIGS.items():
    p = os.path.join(HERE, fn)
    if not os.path.exists(p):
        continue
    z = np.load(p)
    f = z["centres"] / 1000.0
    BK = sorted(k for k in z.files if k != "centres")
    rows = []
    for lab, kind, fun in MEASURES:
        series = [fun(z[b].astype(float), f) for b in BK]
        r = dict(measure=lab, kind=kind)
        for half in ("early", "late"):
            X, y, u = build(series, half)
            r[half] = skill(X, y, u) if len(X) >= 20 else np.nan
        rows.append(r)
    out[rig] = rows

for rig, rows in out.items():
    print(f"\n=== {rig} ===  skill against a predictor that ignores the "
          f"indicator\n")
    print(f"{'indicator':<22}{'kind':<14}{'early half':>12}{'late half':>12}")
    print("-" * 60)
    for r in sorted(rows, key=lambda r: -(r["early"] if np.isfinite(r["early"])
                                          else -9)):
        e = "     n/a" if not np.isfinite(r["early"]) else f"{r['early']:>12.2f}"
        l = "     n/a" if not np.isfinite(r["late"]) else f"{r['late']:>12.2f}"
        print(f"{r['measure']:<22}{r['kind']:<14}{e}{l}")
    print("-" * 60)
    for half in ("early", "late"):
        d = [r[half] for r in rows
             if r["kind"] == "distribution" and np.isfinite(r[half])]
        a = [r[half] for r in rows
             if r["kind"] == "amount" and np.isfinite(r[half])]
        if d and a:
            print(f"  {half:<6} median: distribution {np.median(d):+.2f}, "
                  f"amount {np.median(a):+.2f}, "
                  f"advantage {np.median(d)-np.median(a):+.2f}")

print("\nA positive skill means the indicator beat predicting the fleet mean.")
print("The framework predicts an advantage for distribution indicators in the")
print("early half; if it also appears in the late half, the effect is not")
print("about early usability but about something else.")
json.dump(out, open(os.path.join(HERE, "rul_by_kind.json"), "w"), indent=2,
          default=float)
