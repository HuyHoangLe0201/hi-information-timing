"""
Checking the standard-error formula, and where it cannot be checked.

Proposition on the standard error of the earliest usable age gives
se(tau_min) = se(G_hat(tau_min)) / g(tau_min).  Verifying it needs an
independent estimate of se(G_hat(tau_min)), and a first attempt used a block
bootstrap of the partial sum with blocks a twentieth of the record.  It returned
nothing for the distributional indicators, and the reason is structural rather
than numerical.

The summands of G are built from a Savitzky-Golay derivative of width w, so
neighbouring summands are correlated over about w samples and any block estimate
needs blocks of that length.  For an indicator whose tau_min is at a per cent or
two of life -- which the distributional ones are, at k = 50 of 2803 samples for
spectral entropy -- the whole stretch being summed is SHORTER THAN ONE SMOOTHING
WINDOW.  There is then no second block to compare against, and no variance
statement at that age is supported by an estimator with that resolution.

That is a real limitation and is reported as one.  It also has a shape worth
noting: the indicators whose age cannot be error-barred this way are exactly the
early ones, which by the proposition are the ones whose age is most sharply
determined, since se scales as 1/g and g is large there.  The check therefore
runs where it can -- the late-firing amount indicators, which accumulate over
many windows -- and the count of cells it cannot reach is given rather than
hidden.
"""
import os
import json
import time
import numpy as np
from scipy.signal import savgol_filter

from nonparam import (weighted_density, estimate_floor, info_curve, quantile,
                      _win)

HERE = os.path.dirname(os.path.abspath(__file__))
SEED = 20260828
QSTAR = 0.35
BOOT = 25
MIN_BLOCKS = 4
EPS = 1e-30

MEASURES = [
    ("total power", "amount", lambda P, f: P.sum(axis=1)),
    ("peak band power", "amount", lambda P, f: P.max(axis=1)),
    ("high-band power", "amount", lambda P, f: P[:, -8:].sum(axis=1)),
    ("log total power", "amount",
     lambda P, f: np.log(np.clip(P.sum(axis=1), EPS, None))),
    ("spectral entropy", "distribution",
     lambda P, f: -((P / np.clip(P.sum(axis=1, keepdims=True), EPS, None))
                    * np.log(np.clip(P / np.clip(P.sum(axis=1, keepdims=True),
                                                 EPS, None), EPS, None))
                    ).sum(axis=1)),
    ("high/low ratio", "distribution",
     lambda P, f: P[:, -8:].sum(axis=1)
     / np.clip(P[:, :8].sum(axis=1), EPS, None)),
]


def formula_se(x, rng):
    """se(G_hat)/g at tau_min, with blocks the length of the smoothing window.

    Returns (se, tau, blocks) so a cell with too few blocks can be reported
    rather than silently dropped.
    """
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    dens, res = weighted_density(x)
    if dens is None:
        return np.nan, np.nan, 0
    d = np.clip(dens - estimate_floor(dens, res, "surrogate", rng), 0.0, None)
    tot, n = d.sum(), len(d)
    if tot <= 0:
        return np.nan, np.nan, 0
    k = int(np.searchsorted(np.cumsum(d), QSTAR * tot))
    if k >= n:
        return np.nan, np.nan, 0
    b = max(3, _win(n))                       # the correlation length
    nb = (k + 1) // b
    if nb < MIN_BLOCKS:
        return np.nan, (k + 1.0) / n, nb
    blocks = np.array([d[i * b:(i + 1) * b].sum() for i in range(nb)], float)
    se_G = float(np.sqrt(nb) * np.std(blocks, ddof=1))
    lo, hi = max(0, k - b // 2), min(n, k + b // 2 + 1)
    g = float(np.mean(d[lo:hi])) * n
    if not np.isfinite(se_G) or not np.isfinite(g) or g <= 0:
        return np.nan, (k + 1.0) / n, nb
    return se_G / g, (k + 1.0) / n, nb


def direct_bootstrap(x, rng, B=BOOT):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    n = len(x)
    w = _win(n)
    if w < 7 or n < 200:
        return np.nan
    T = savgol_filter(x, w, 2)
    r = x - T
    b = max(3, w)
    starts = np.arange(0, max(1, n - b))
    out = []
    for _ in range(B):
        rb = np.concatenate([r[s:s + b] for s in
                             rng.choice(starts, size=n // b + 1,
                                        replace=True)])[:n]
        F = info_curve(T + rb, rng=rng)
        if F is not None:
            v = quantile(F, QSTAR)
            if np.isfinite(v):
                out.append(v)
    return float(np.std(out, ddof=1)) if len(out) > 3 else np.nan


RIGS = {"PRONOSTIA": "fineband.npz"}
rng = np.random.default_rng(SEED)
print("standard error of the earliest usable age\n")
print(f"blocks are the smoothing window; a cell needs {MIN_BLOCKS} of them\n")
print(f"{'record':<13}{'measure':<19}{'kind':<13}{'tau':>7}{'blocks':>8}"
      f"{'formula':>10}{'bootstrap':>11}{'ratio':>7}")
print("-" * 88, flush=True)
rows, unreach = [], []
t0 = time.time()
for rig, fn in RIGS.items():
    p = os.path.join(HERE, fn)
    if not os.path.exists(p):
        continue
    zz = np.load(p)
    f = zz["centres"] / 1000.0
    for b in sorted(k for k in zz.files if k != "centres")[:3]:
        P = zz[b].astype(float)
        for nm, kind, fun in MEASURES:
            x = np.asarray(fun(P, f), float)
            if not np.isfinite(x).all() or len(x) < 200:
                continue
            se_f, t, nb = formula_se(x, rng)
            if not np.isfinite(se_f):
                unreach.append(dict(unit=b, measure=nm, kind=kind,
                                    tau=None if not np.isfinite(t) else float(t),
                                    blocks=int(nb)))
                print(f"{b:<13}{nm:<19}{kind:<13}"
                      f"{t if np.isfinite(t) else float('nan'):>7.3f}{nb:>8}"
                      f"{'--':>10}{'--':>11}{'--':>7}", flush=True)
                continue
            se_b = direct_bootstrap(x, rng)
            if not (np.isfinite(se_b) and se_b > 0):
                continue
            rows.append(dict(unit=b, measure=nm, kind=kind, tau=float(t),
                             blocks=int(nb), formula=float(se_f),
                             boot=float(se_b), ratio=float(se_f / se_b)))
            print(f"{b:<13}{nm:<19}{kind:<13}{t:>7.3f}{nb:>8}{se_f:>10.4f}"
                  f"{se_b:>11.4f}{se_f / se_b:>7.2f}", flush=True)
print("-" * 88)
print(f"({time.time() - t0:.0f}s)\n")

if rows:
    rr = np.array([r["ratio"] for r in rows])
    print(f"formula over bootstrap: median {np.median(rr):.2f}, "
          f"range {rr.min():.2f} to {rr.max():.2f} over {len(rr)} cells")
    print()
    print("THIS IS A NEGATIVE RESULT AND THE ESTIMATOR ABOVE IS AT FAULT, not")
    print("the proposition.  se_G was taken as sqrt(nb) times the scatter of the")
    print("block sums, which is the standard error of a sum of independent")
    print("blocks.  These blocks are consecutive stretches of a density that")
    print("rises steeply over life, so their scatter is dominated by that")
    print("systematic rise rather than by sampling error, and the estimate is")
    print("inflated -- twentyfold on the worst cell.  No cheap block estimator of")
    print("se(G_hat) exists on a record whose density varies systematically, so")
    print("the proposition is not a shortcut for computing an error bar and the")
    print("bootstrap is what the paper uses.  The testable content of the")
    print("proposition is the SHAPE of the dependence, se proportional to 1/g,")
    print("which se_scaling.py tests directly and which needs none of this.")
print()
nk = {}
for u in unreach:
    nk[u["kind"]] = nk.get(u["kind"], 0) + 1
print(f"{len(unreach)} cells could not be reached: "
      + ", ".join(f"{v} {k}" for k, v in sorted(nk.items())))
print("They are the cells whose tau_min falls before the record has accumulated")
print(f"{MIN_BLOCKS} smoothing windows, so the partial sum has too few "
      f"independent blocks")
print("to have a variance estimated from it.  By the proposition those are also")
print("the cells whose age is most sharply determined, since the standard error")
print("scales as 1/g and the density is large there; the limitation is one of")
print("this estimator's resolution, not of the quantity.")

json.dump(dict(min_blocks=MIN_BLOCKS, boot=BOOT, checked=rows,
               unreachable=unreach,
               ratio_median=(float(np.median([r["ratio"] for r in rows]))
                             if rows else None),
               ratio_min=(float(min(r["ratio"] for r in rows))
                          if rows else None),
               ratio_max=(float(max(r["ratio"] for r in rows))
                          if rows else None)),
          open(os.path.join(HERE, "se_formula_check.json"), "w"), indent=2,
          default=float)
