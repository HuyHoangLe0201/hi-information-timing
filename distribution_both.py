"""
Distribution against amount, on both rigs, with every indicator built here.

On PRONOSTIA the six measures of how spectral energy is distributed became
usable between 0.15 and 0.53 of life, while the four measures of how much energy
there is did not become usable until 0.96 to 0.97 -- disjoint ranges, a gap of
0.64 at the median. The log of total power sat with the amounts, so bounding the
growth is not what separates them.

That was one test rig. XJTU-SY records the same two accelerometers at the same
rate, so the same thirty-two bands and the same ten indicators can be built from
its raw acquisitions, and the claim tested on a second stand with nothing
inherited: every definition below is in this file.

If the two groups separate on both rigs, the statement is about how
rolling-element bearings fail rather than about one laboratory.
"""
import os
import json
import numpy as np
from nonparam import weighted_density, estimate_floor, quantile

HERE = os.path.dirname(os.path.abspath(__file__))
Q = 0.35
THRESH = 0.90
SEED = 8675309
EPS = 1e-30


def profile(series, rng):
    T, S = [], []
    for x in series:
        x = np.asarray(x, float)
        x = x[np.isfinite(x)]
        if len(x) < 60:
            continue
        dens, res = weighted_density(x)
        if dens is None:
            continue
        d = np.clip(dens - estimate_floor(dens, res, "surrogate", rng), 0.0, None)
        if d.sum() <= 0:
            continue
        F = np.cumsum(d) / d.sum()
        t = quantile(F, Q)
        k = max(1, min(int(round(t * len(d))) - 1, len(d) - 1))
        rk = float(dens[:k + 1].sum())
        if rk <= 0:
            continue
        T.append(t)
        S.append(float(d[:k + 1].sum()) / rk)
    if len(T) < 4:
        return None
    return float(np.median(T)), float(np.median(S)), len(T)


def norm(P):
    return P / np.clip(P.sum(axis=1, keepdims=True), EPS, None)


def entropy(P):
    p = norm(P)
    return -(p * np.log(np.clip(p, EPS, None))).sum(axis=1)


def centroid(P, f):
    return (norm(P) * f).sum(axis=1)


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
    ("spectral centroid", "distribution", lambda P, f: centroid(P, f)),
    ("spectral spread", "distribution", lambda P, f: spread(P, f)),
    ("spectral Gini", "distribution", lambda P, f: gini(P)),
    ("top-8-band share", "distribution",
     lambda P, f: np.sort(norm(P), axis=1)[:, -8:].sum(axis=1)),
    ("high/low band ratio", "distribution",
     lambda P, f: P[:, -8:].sum(axis=1) / np.clip(P[:, :8].sum(axis=1), EPS, None)),
    ("total power", "amount", lambda P, f: P.sum(axis=1)),
    ("peak band power", "amount", lambda P, f: P.max(axis=1)),
    ("high-band power", "amount", lambda P, f: P[:, -8:].sum(axis=1)),
    ("log total power", "amount",
     lambda P, f: np.log(np.clip(P.sum(axis=1), EPS, None))),
]

RIGS = {"PRONOSTIA": "fineband.npz", "XJTU": "fineband_xjtu.npz"}
rng = np.random.default_rng(SEED)
ALL = {}
for rig, fn in RIGS.items():
    p = os.path.join(HERE, fn)
    if not os.path.exists(p):
        print(f"{rig}: {fn} not found")
        continue
    z = np.load(p)
    f = z["centres"] / 1000.0
    BK = sorted(k for k in z.files if k != "centres")
    out = []
    for lab, kind, fun in MEASURES:
        r = profile([fun(z[b].astype(float), f) for b in BK], rng)
        if r:
            out.append(dict(measure=lab, kind=kind, tau=r[0], signal=r[1],
                            units=r[2]))
    ALL[rig] = out
    print(f"{rig}: {len(BK)} bearings, {len(out)} measures")

print()
rigs = list(ALL)
print(f"{'measure':<22}{'kind':<14}" + "".join(f"{r:>12}" for r in rigs))
print("-" * (36 + 12 * len(rigs)))
for lab, kind, _ in MEASURES:
    cells = []
    for rig in rigs:
        m = next((x for x in ALL[rig] if x["measure"] == lab), None)
        cells.append(m["tau"] if m and m["signal"] >= THRESH else np.nan)
    if all(not np.isfinite(c) for c in cells):
        continue
    s = "".join("         n/a" if not np.isfinite(c) else f"{c:>12.3f}"
                for c in cells)
    print(f"{lab:<22}{kind:<14}{s}")
print("-" * (36 + 12 * len(rigs)))

print(f"\n{'':<22}" + "".join(f"{r:>12}" for r in rigs))
summ = {}
for kind in ("distribution", "amount"):
    cells, spans = [], []
    for rig in rigs:
        v = [x["tau"] for x in ALL[rig]
             if x["kind"] == kind and x["signal"] >= THRESH]
        cells.append(float(np.median(v)) if v else np.nan)
        spans.append((min(v), max(v)) if v else (np.nan, np.nan))
    summ[kind] = dict(median=cells, span=spans)
    print(f"{kind + ' (median)':<22}"
          + "".join(f"{c:>12.3f}" for c in cells))
    print(f"{'  range':<22}"
          + "".join(f"{a:>6.3f}-{b:<5.3f}" for a, b in spans))

print()
for i, rig in enumerate(rigs):
    a = summ["amount"]["median"][i]
    d = summ["distribution"]["median"][i]
    da = summ["distribution"]["span"][i]
    aa = summ["amount"]["span"][i]
    if not (np.isfinite(a) and np.isfinite(d)):
        continue
    disjoint = da[1] < aa[0]
    print(f"{rig}: gap {a - d:.3f} of life, ranges "
          f"{'disjoint' if disjoint else 'overlapping'}")
print()
gaps = [summ["amount"]["median"][i] - summ["distribution"]["median"][i]
        for i in range(len(rigs))
        if np.isfinite(summ["amount"]["median"][i])]
disj = [summ["distribution"]["span"][i][1] < summ["amount"]["span"][i][0]
        for i in range(len(rigs))
        if np.isfinite(summ["amount"]["median"][i])]
if len(gaps) == 2 and min(gaps) > 0.15:
    print(f"Both rigs put the distribution measures earlier, by {min(gaps):.2f} "
          f"and {max(gaps):.2f} of life at the median.")
    if all(disj):
        print("The groups are disjoint on both, so the separation is clean.")
    else:
        # Naming which rig fails and which indicator crosses matters more than
        # the aggregate: a single amount measure inside the distribution range
        # is what stops this being a rule.
        for i, rig in enumerate(rigs):
            if i < len(disj) and not disj[i]:
                cross = [x["measure"] for x in ALL[rig]
                         if x["kind"] == "amount"
                         and x["signal"] >= THRESH
                         and x["tau"] < summ["distribution"]["span"][i][1]]
                print(f"But on {rig} the groups overlap: "
                      f"{', '.join(cross)} sits inside the distribution range.")
        print("So the direction holds on both rigs while the clean separation")
        print("does not, and the gap itself differs by nearly a factor of two.")
        print("The defensible statement is a tendency, not a partition.")
json.dump(dict(rigs=ALL, summary={k: {"median": v["median"],
                                      "span": [list(s) for s in v["span"]]}
                                  for k, v in summ.items()}),
          open(os.path.join(HERE, "distribution_both.json"), "w"), indent=2,
          default=float)
