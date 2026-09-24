"""
Distribution against amount, on indicators built here rather than inherited.

Three things are now known. Rescaling an indicator does not move the earliest
usable age (0.000). Dividing the level by its own slow average does not either
(0.973 to 0.976), so being a bounded ratio is not the mechanism. But the ratio
of the two accelerometers -- carrying no spectral information -- is early
(0.501), as are the spectral shape measures.

The refined claim is therefore about neither spectra nor ratios: what an
indicator says about how vibration is DISTRIBUTED, over frequency or over
direction, is available long before what it says about how MUCH there is.

That claim is tested here on indicators constructed from the thirty-two narrow
bands extracted from the raw acquisitions, so every definition is known. Six
measures of distribution and four of amount are built from the same spectra,
and nothing else differs between them.
"""
import os
import json
import numpy as np
from nonparam import weighted_density, estimate_floor, quantile

HERE = os.path.dirname(os.path.abspath(__file__))
Q = 0.35
THRESH = 0.90
SEED = 606


def profile(series, rng):
    T, S = [], []
    for x in series:
        x = np.asarray(x, float)
        m = np.isfinite(x)
        x = x[m]
        if len(x) < 80:
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


EPS = 1e-30


def norm(P):
    """Row-normalised spectrum: the distribution, with the amount divided out."""
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
    idx = np.arange(1, n + 1)
    return (2 * (p * idx).sum(axis=1)) / np.clip(p.sum(axis=1), EPS, None) \
        / n - (n + 1) / n


def top_share(P, k=8):
    p = norm(P)
    return np.sort(p, axis=1)[:, -k:].sum(axis=1)


def hi_lo(P):
    lo = P[:, :8].sum(axis=1)
    hi = P[:, -8:].sum(axis=1)
    return hi / np.clip(lo, EPS, None)


MEASURES = [
    ("spectral entropy", "distribution", lambda P, f: entropy(P)),
    ("spectral centroid", "distribution", lambda P, f: centroid(P, f)),
    ("spectral spread", "distribution", lambda P, f: spread(P, f)),
    ("spectral Gini", "distribution", lambda P, f: gini(P)),
    ("top-8-band share", "distribution", lambda P, f: top_share(P)),
    ("high/low band ratio", "distribution", lambda P, f: hi_lo(P)),
    ("total power", "amount", lambda P, f: P.sum(axis=1)),
    ("peak band power", "amount", lambda P, f: P.max(axis=1)),
    ("high-band power", "amount", lambda P, f: P[:, -8:].sum(axis=1)),
    ("log total power", "amount",
     lambda P, f: np.log(np.clip(P.sum(axis=1), EPS, None))),
]

z = np.load(os.path.join(HERE, "fineband.npz"))
f = z["centres"] / 1000.0
BK = sorted(k for k in z.files if k != "centres")
print(f"{len(BK)} bearings, {len(f)} bands, all measures built from the same "
      f"spectra\n")

rng = np.random.default_rng(SEED)
print(f"{'measure':<22}{'kind':<14}{'units':>6}{'signal':>9}{'tau':>9}")
print("-" * 60)
rows = []
for lab, kind, fn in MEASURES:
    series = [fn(z[b].astype(float), f) for b in BK]
    p = profile(series, rng)
    if not p:
        print(f"{lab:<22}{kind:<14}   not measurable")
        continue
    t, s, n = p
    rows.append(dict(measure=lab, kind=kind, tau=t, signal=s, units=n))
    flag = "" if s >= THRESH else "  (below threshold)"
    print(f"{lab:<22}{kind:<14}{n:>6}{100*s:>8.0f}%{t:>9.3f}{flag}")
print("-" * 60)

ok = [r for r in rows if r["signal"] >= THRESH]
D = [r["tau"] for r in ok if r["kind"] == "distribution"]
A = [r["tau"] for r in ok if r["kind"] == "amount"]
if D and A:
    print(f"\ndistribution ({len(D)}): {min(D):.3f} to {max(D):.3f}, "
          f"median {np.median(D):.3f}")
    print(f"amount       ({len(A)}): {min(A):.3f} to {max(A):.3f}, "
          f"median {np.median(A):.3f}")
    print(f"gap at the median: {np.median(A) - np.median(D):.3f} of life")
    overlap = max(D) > min(A)
    print(f"\nranges {'overlap' if overlap else 'are disjoint'}")
    print()
    if np.median(A) - np.median(D) > 0.15 and not overlap:
        print("Every measure of how the energy is distributed is usable before")
        print("every measure of how much there is, on indicators whose")
        print("definitions are all known and which come from the same spectra.")
        print("The log of total power is included as a control: taking a")
        print("logarithm bounds the growth without saying anything about the")
        print("distribution, and it stays with the amounts.")
    else:
        print("The two groups are not cleanly separated, so the distinction")
        print("does not survive on indicators built to order.")
json.dump(rows, open(os.path.join(HERE, "distribution_test.json"), "w"),
          indent=2, default=float)
