r"""Freeze the pipeline's choices on one set of units, measure on another.

A reviewer's objection to the applied result is not that the controls are
missing -- the supplement has many -- but that almost every control was run on
the same corpus the choices were explored on.  Sweeping a setting and finding
the conclusion survives shows the conclusion is not fragile; it does not show
the setting was not chosen, however implicitly, with the answer in view.

The test that does separate the two is ordinary out-of-sample practice.  Split
the units.  Pick the choices on the development half by the most favourable
criterion available -- the one that MAXIMISES the family gap, which is the
worst case for the paper.  Freeze them.  Read the gap on the held-out half,
which had no part in the choice.  If tuning buys nothing out of sample, the
result is not resting on it.

Three splits are run, in increasing severity:

  within-rig     random halves of one rig's bearings, 200 repetitions
  across-rig     develop on PRONOSTIA, hold out XJTU-SY, and the reverse
  across-domain  develop on the bearings, hold out the turbofan fleets, whose
                 channels carry no spectrum and whose families are defined by
                 a different construction entirely

The grid is the three bandwidth-like settings the body already sweeps: the
Savitzky-Golay window as a fraction of life, the local-scale bandwidth, and the
demand q.  The density does not depend on q, so F is computed once per
(unit, measure, window, bandwidth) and the three demands are read off it.

    python frozen_split.py   ->  frozen_split.json
"""
import itertools
import json
import os

import numpy as np
from scipy.signal import savgol_filter

from pipeline import robust_scale, oof_trend, local_scale

HERE = os.path.dirname(os.path.abspath(__file__))
EPS = 1e-30
SEED = 20260923
REPS = 8
SPLITS = 200

FRACS = (0.05, 0.08, 0.12)
BWS = (0.05, 0.08, 0.12)
QS = (0.20, 0.35, 0.50)
DEFAULT = (0.08, 0.08, 0.35)


# ------------------------------------------------- the pipeline, parameterised --
def win_of(n, frac):
    w = max(7, int(frac * n))
    w += (w % 2 == 0)
    return min(w, n - 1 if (n - 1) % 2 == 1 else n - 2)


def curve(x, frac, bw, rng):
    """Normalised cumulative information F, noise floor removed.

    Identical to nonparam.info_curve except that the two bandwidths are
    arguments rather than module constants.
    """
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    n = len(x)
    if n < 60:
        return None
    s = robust_scale(x)
    if not np.isfinite(s) or s <= 0:
        s = float(np.std(x))
    if not np.isfinite(s) or s <= 0:
        return None
    D = (x - np.median(x)) / s
    w = win_of(n, frac)
    if w < 7 or n < w + 2:
        return None
    h = max(5, int(bw * n))
    dt = savgol_filter(D, w, 2, deriv=1, delta=1.0 / n)
    res = D - oof_trend(D, w // 2)
    sl = np.clip(local_scale(res, h), 1e-12, None)
    dens = (dt / sl) ** 2

    acc = []
    for _ in range(REPS):
        z = res[rng.integers(0, n, n)]
        sc = robust_scale(z)
        if not np.isfinite(sc) or sc <= 0:
            continue
        Dz = (z - np.median(z)) / sc
        dz = savgol_filter(Dz, w, 2, deriv=1, delta=1.0 / n)
        r2 = Dz - savgol_filter(Dz, w, 2)
        s2 = np.clip(local_scale(r2, h), 1e-12, None)
        acc.append((dz / s2) ** 2)
    fl = float(np.median(np.concatenate(acc))) if acc else 0.0

    d = np.clip(dens - fl, 0.0, None)
    if d.sum() <= 0:
        return None
    return np.cumsum(d) / d.sum()


def tau_at(F, q):
    k = int(np.searchsorted(F, q))
    return min((k + 1) / len(F), 1.0)


# ------------------------------------------------------------------ measures --
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
    m = p.shape[1]
    return (2 * (p * np.arange(1, m + 1)).sum(axis=1)) \
        / np.clip(p.sum(axis=1), EPS, None) / m - (m + 1) / m


MEASURES = [
    ("spectral entropy", "distribution", lambda P, f: entropy(P)),
    ("spectral centroid", "distribution", lambda P, f: centroid(P, f)),
    ("spectral spread", "distribution", lambda P, f: spread(P, f)),
    ("spectral Gini", "distribution", lambda P, f: gini(P)),
    ("top-8-band share", "distribution",
     lambda P, f: np.sort(norm(P), axis=1)[:, -8:].sum(axis=1)),
    ("high/low band ratio", "distribution",
     lambda P, f: P[:, -8:].sum(axis=1)
     / np.clip(P[:, :8].sum(axis=1), EPS, None)),
    ("total power", "amount", lambda P, f: P.sum(axis=1)),
    ("peak band power", "amount", lambda P, f: P.max(axis=1)),
    ("high-band power", "amount", lambda P, f: P[:, -8:].sum(axis=1)),
    ("log total power", "amount",
     lambda P, f: np.log(np.clip(P.sum(axis=1), EPS, None))),
]
RIGS = {"PRONOSTIA": "fineband.npz", "XJTU": "fineband_xjtu.npz"}


# ----------------------------------------------------- the turbofan families --
# Mirrors turbofan_distribution.py: each sensor standardised on the unit's own
# healthy head, absolute deviations taken as a non-negative profile across
# sensors, and only order-invariant functionals admitted, the sensor index
# carrying no ordering.  Kept as a copy rather than an import because that
# module runs its whole analysis when imported.
BASE = 0.20
MIN_TURBO_N = 100
MAX_PER_FLEET = 40


def prof_norm(A):
    return A / np.clip(A.sum(axis=1, keepdims=True), EPS, None)


TURBO_MEASURES = [
    ("deviation entropy", "distribution",
     lambda A, f: -(prof_norm(A) * np.log(np.clip(prof_norm(A), EPS, None))
                    ).sum(axis=1)),
    ("deviation Gini", "distribution",
     lambda A, f: (2.0 * (np.arange(1, A.shape[1] + 1)
                          * np.sort(prof_norm(A), axis=1)).sum(axis=1)
                   - (A.shape[1] + 1.0)) / A.shape[1]),
    ("top-five share", "distribution",
     lambda A, f: np.sort(prof_norm(A), axis=1)[:, -5:].sum(axis=1)),
    ("participation ratio", "distribution",
     lambda A, f: 1.0 / np.clip((prof_norm(A) ** 2).sum(axis=1), EPS, None)),
    ("total deviation", "amount", lambda A, f: A.sum(axis=1)),
    ("max deviation", "amount", lambda A, f: A.max(axis=1)),
    ("mean deviation", "amount", lambda A, f: A.mean(axis=1)),
    ("log total deviation", "amount",
     lambda A, f: np.log(np.clip(A.sum(axis=1), EPS, None))),
]


def deviation_profile(S, base=BASE):
    """|z| of each sensor against the unit's own healthy head."""
    S = np.asarray(S, float)
    n = S.shape[0]
    h = S[:max(5, int(base * n))]
    mu = np.median(h, axis=0)
    sd = np.median(np.abs(h - mu), axis=0) * 1.4826
    keep = np.isfinite(sd) & (sd > 0)
    if keep.sum() < 8:
        return None
    return np.abs((S[:, keep] - mu[keep]) / sd[keep])


def turbofan_units():
    """The first MAX_PER_FLEET long-enough units of each fleet, by unit name."""
    out = {}
    for sub in ("FD001", "FD004"):
        path = os.path.join(HERE, "cmapss_%s.npz" % sub)
        if not os.path.exists(path):
            continue
        z = np.load(path, allow_pickle=True)
        taken = 0
        for nm in sorted({k.split("__")[0] for k in z.files}):
            key = "%s__sensors" % nm
            if key not in z.files:
                continue
            S = np.asarray(z[key], float)
            if S.ndim != 2 or S.shape[0] < MIN_TURBO_N:
                continue
            A = deviation_profile(S)
            if A is None:
                continue
            out["%s/%s" % (sub, nm)] = A
            taken += 1
            if taken >= MAX_PER_FLEET:
                break
    return out


# --------------------------------------------------------------- precompute --
def table(units, measures, freqs):
    """tau[unit][measure][(frac, bw, q)] for every grid point."""
    T = {}
    for uid, (P, f) in units.items():
        T[uid] = {}
        for lab, kind, fun in measures:
            x = np.asarray(fun(P, f), float)
            x = x[np.isfinite(x)]
            if len(x) < 60:
                continue
            cell = {}
            for frac, bw in itertools.product(FRACS, BWS):
                F = curve(x, frac, bw, np.random.default_rng(SEED))
                if F is None:
                    continue
                for q in QS:
                    cell[(frac, bw, q)] = tau_at(F, q)
            if cell:
                T[uid][lab] = (kind, cell)
    return T


def gap(T, uids, point):
    """Family gap over a set of units at one grid point."""
    fam = {"distribution": [], "amount": []}
    for lab in {l for u in uids for l in T.get(u, {})}:
        per = [T[u][lab][1][point] for u in uids
               if lab in T.get(u, {}) and point in T[u][lab][1]]
        if not per:
            continue
        kind = next(T[u][lab][0] for u in uids if lab in T.get(u, {}))
        fam[kind].append(float(np.median(per)))
    if not fam["distribution"] or not fam["amount"]:
        return None
    return float(np.median(fam["amount"]) - np.median(fam["distribution"]))


def best_point(T, uids):
    """The grid point that most favours the paper on these units."""
    best, bg = None, -np.inf
    for point in itertools.product(FRACS, BWS, QS):
        g = gap(T, uids, point)
        if g is not None and g > bg:
            best, bg = point, g
    return best, bg



# --------------------------------------------------------------------- cache --
# The grid costs an hour of CPU and does not depend on anything downstream, so
# it is written once and reused.  Delete the file to recompute.
def cached(name, build):
    path = os.path.join(HERE, name)
    if os.path.exists(path):
        raw = json.load(open(path))
        print("  %s: %d units from cache" % (name, len(raw)))
        return {u: {lab: (kind, {tuple(float(x) for x in k.split("|")): v
                                 for k, v in cell.items()})
                    for lab, (kind, cell) in m.items()}
                for u, m in raw.items()}
    T = build()
    json.dump({u: {lab: (kind, {"%g|%g|%g" % k: v for k, v in cell.items()})
                   for lab, (kind, cell) in m.items()}
               for u, m in T.items()}, open(path, "w"))
    print("  %s: written" % name)
    return T



BOOT = 2000              # unit-level resamples behind every held-out interval


def gap_ci(T, uids, point, rng, reps=BOOT):
    """Percentile interval for the held-out gap, resampling UNITS.

    A reviewer asked for this at the point where the paper first states a
    train/test protocol, and the request is the right one: a held-out estimate
    read off sixteen bearings carries the uncertainty of sixteen bearings, and
    a point estimate hides it.  Units are the resampling unit throughout this
    paper, never evaluation cells, so they are the resampling unit here.
    """
    n = len(uids)
    if n < 3:
        return None, None
    draws = []
    for _ in range(reps):
        pick = [uids[i] for i in rng.integers(0, n, n)]
        g = gap(T, pick, point)
        if g is not None:
            draws.append(g)
    if len(draws) < reps // 4:
        return None, None
    return float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))


# ------------------------------------------------------------------- load it --
print("loading and precomputing the grid ...")
units, freqs = {}, {}
for rig, fn in RIGS.items():
    path = os.path.join(HERE, fn)
    if not os.path.exists(path):
        print("  %s: %s not found" % (rig, fn))
        continue
    z = np.load(path)
    f = z["centres"] / 1000.0
    for b in sorted(k for k in z.files if k != "centres"):
        units["%s/%s" % (rig, b)] = (z[b].astype(float), f)
print("  %d bearings" % len(units))

T = cached("frozen_grid_bearings.json",
           lambda: table(units, MEASURES, freqs))
print("  %d bearings with at least one measure" % sum(1 for u in T if T[u]))

results = {}

# ------------------------------------------------------- 1. within-rig halves --
print()
print("=" * 78)
print("1  RANDOM HALVES WITHIN A RIG")
print("=" * 78)
print("%-12s%8s  %-22s %-22s"
      % ("rig", "splits", "paper's settings", "frozen choice"))
print("-" * 78)
rng = np.random.default_rng(SEED)
for rig in RIGS:
    ids = sorted(u for u in T if u.startswith(rig + "/") and T[u])
    if len(ids) < 6:
        continue
    tuned, default = [], []
    for _ in range(SPLITS):
        perm = rng.permutation(len(ids))
        half = len(ids) // 2
        dev = [ids[i] for i in perm[:half]]
        hold = [ids[i] for i in perm[half:]]
        pt, _ = best_point(T, dev)
        if pt is None:
            continue
        a, b = gap(T, hold, pt), gap(T, hold, DEFAULT)
        if a is not None and b is not None:
            tuned.append(a)
            default.append(b)
    if not tuned:
        continue
    # Across 200 random halves the spread of the held-out gap IS its
    # uncertainty: each split is a fresh draw of which units are held out.
    results.setdefault("within_rig", {})[rig] = dict(
        splits=len(tuned), tuned=float(np.median(tuned)),
        default=float(np.median(default)),
        tuned_lo=float(np.percentile(tuned, 2.5)),
        tuned_hi=float(np.percentile(tuned, 97.5)),
        default_lo=float(np.percentile(default, 2.5)),
        default_hi=float(np.percentile(default, 97.5)),
        tuned_q05=float(np.percentile(tuned, 5)),
        default_q05=float(np.percentile(default, 5)))
    print("%-12s%8d  %.3f [%.3f,%.3f]   %.3f [%.3f,%.3f]"
          % (rig, len(tuned), np.median(default),
             np.percentile(default, 2.5), np.percentile(default, 97.5),
             np.median(tuned),
             np.percentile(tuned, 2.5), np.percentile(tuned, 97.5)))

# ------------------------------------------------------------- 2. across rig --
print()
print("=" * 78)
print("2  DEVELOP ON ONE RIG, HOLD OUT THE OTHER")
print("=" * 78)
print("%-11s  %-11s %-16s %-22s %-22s"
      % ("develop", "hold out", "frozen point",
         "paper's settings", "frozen choice"))
print("-" * 78)
for dev_rig, hold_rig in (("PRONOSTIA", "XJTU"), ("XJTU", "PRONOSTIA")):
    dev = sorted(u for u in T if u.startswith(dev_rig + "/") and T[u])
    hold = sorted(u for u in T if u.startswith(hold_rig + "/") and T[u])
    if not dev or not hold:
        continue
    pt, dg = best_point(T, dev)
    hg, dfg = gap(T, hold, pt), gap(T, hold, DEFAULT)
    _r = np.random.default_rng(SEED)
    hlo, hhi = gap_ci(T, hold, pt, _r)
    dlo, dhi = gap_ci(T, hold, DEFAULT, np.random.default_rng(SEED))
    results.setdefault("across_rig", {})["%s->%s" % (dev_rig, hold_rig)] = dict(
        point=list(pt), dev_gap=dg, held_gap=hg, default_gap=dfg,
        units=len(hold), held_lo=hlo, held_hi=hhi,
        default_lo=dlo, default_hi=dhi)
    print("%-11s->%-11s %s  %.3f [%.3f,%.3f]   %.3f [%.3f,%.3f]"
          % (dev_rig, hold_rig, "%.2f/%.2f/%.2f" % pt,
             dfg, dlo, dhi, hg, hlo, hhi))

# ---------------------------------------------------------- 3. across domain --
print()
print("=" * 78)
print("3  DEVELOP ON THE BEARINGS, HOLD OUT THE TURBOFAN FLEETS")
print("=" * 78)
TU = {k: (P, None) for k, P in turbofan_units().items()}
print("  %d turbofan units" % len(TU))
TT = cached("frozen_grid_turbofan.json",
            lambda: table(TU, TURBO_MEASURES, {})) if TU else {}
if TT and any(TT[u] for u in TT):
    dev = sorted(u for u in T if T[u])
    pt, dg = best_point(T, dev)
    for sub in ("FD001", "FD004"):
        hold = sorted(u for u in TT if u.startswith(sub + "/") and TT[u])
        if not hold:
            continue
        hg, dfg = gap(TT, hold, pt), gap(TT, hold, DEFAULT)
        if hg is None:
            continue
        hlo, hhi = gap_ci(TT, hold, pt, np.random.default_rng(SEED))
        dlo, dhi = gap_ci(TT, hold, DEFAULT, np.random.default_rng(SEED))
        results.setdefault("across_domain", {})[sub] = dict(
            point=list(pt), units=len(hold), held_gap=hg, default_gap=dfg,
            held_lo=hlo, held_hi=hhi, default_lo=dlo, default_hi=dhi)
        print("  %-8s %3d units  %.3f [%.3f,%.3f]   %.3f [%.3f,%.3f]"
              % (sub, len(hold), dfg, dlo, dhi, hg, hlo, hhi))

json.dump(dict(results=results, grid=dict(fracs=list(FRACS), bws=list(BWS),
                                          qs=list(QS), default=list(DEFAULT)),
               splits=SPLITS, boot=BOOT, seed=SEED),
          open(os.path.join(HERE, "frozen_split.json"), "w"), indent=1)
print()
print("written to frozen_split.json")
