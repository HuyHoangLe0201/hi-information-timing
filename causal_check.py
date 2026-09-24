r"""Does the applied result survive a strictly causal estimator?

A reviewer asked the right question.  The density the whole paper is built on
uses a Savitzky-Golay derivative with a CENTRED window, an out-of-fold trend
fitted with interleaved folds spanning the whole record, and a local noise
scale on a centred window.  All three look forward.  For characterising what a
completed record contains that is exactly right, and it is what "the
information a record carries" means.  But the phrase "the earliest age at
which an indicator becomes usable" invites a deployment reading, and under that
reading a two-sided estimator is optimistic: at age tau it has seen data from
after tau.

This rebuilds the applied result with an estimator that has seen nothing after
tau, and asks whether the ordering survives.

  derivative    quadratic fitted on the trailing window [i-w+1, i] and
                differentiated at its right edge, so the estimate at i uses i
                and earlier only.
  trend         quadratic fitted on [i-w, i-1] and extrapolated to i: a genuine
                one-step-ahead prediction, causal and out-of-sample, replacing
                the interleaved K-fold trend.
  local scale   robust scale of the causal residual over the trailing window.

Before the first full window there is no estimate, and the causal density is
zero there rather than extrapolated -- an operator has no derivative before
there is data to take one from.

  noise floor   the level subtracted at age tau is estimated from the causal
                residuals available at tau, re-estimated at forty checkpoints
                along the record and held until the next one.  The first
                version took one level from the residuals of the whole record,
                which a reviewer rightly noted is not causal; that arm is kept
                as `causal_record_floor` so the difference is visible.  Before
                ten residuals exist no floor can be estimated and the density
                is left at zero.

What stays non-causal is stated in Definition 2: the age axis t/T, the
normalisation by the completed record's total, and the window widths, which are
fractions of the completed life as the age axis is.

    python causal_check.py   ->  causal_check.json
"""
import json
import os

import numpy as np
from scipy.signal import savgol_filter

HERE = os.path.dirname(os.path.abspath(__file__))
Q = 0.35
SMOOTH_FRAC = 0.08
LOCAL_BW = 0.08
EPS = 1e-30
SEED = 8675309


def robust_scale(x):
    x = np.asarray(x, float)
    m = np.median(x)
    return 1.4826 * np.median(np.abs(x - m))


def _win(n):
    w = max(7, int(SMOOTH_FRAC * n))
    w += (w % 2 == 0)
    return min(w, n - 1 if (n - 1) % 2 == 1 else n - 2)


def _edge_filters(w):
    """Least-squares quadratic on t = -(w-1)..0; value and slope at t = 0."""
    t = np.arange(-(w - 1), 1, dtype=float)
    X = np.vstack([np.ones_like(t), t, t ** 2]).T
    P = np.linalg.pinv(X)            # rows: constant, linear, quadratic
    return P[0], P[1]                # value at t=0, derivative at t=0


def causal_density(x):
    """(d/sigma)^2 using only the past, and the residual it is built on."""
    x = np.asarray(x, float)
    n = len(x)
    s = robust_scale(x)
    if not np.isfinite(s) or s <= 0:
        s = float(np.std(x))
    if not np.isfinite(s) or s <= 0:
        return None, None
    D = (x - np.median(x)) / s
    w = _win(n)
    if w < 7 or n < w + 2:
        return None, None
    c_val, c_der = _edge_filters(w)

    dt = np.zeros(n)
    pred = np.full(n, np.nan)
    for i in range(w - 1, n):
        seg = D[i - w + 1:i + 1]
        dt[i] = float(c_der @ seg) * n          # per unit of normalised age
    # one-step-ahead trend: fit on the w samples ending at i-1, predict at i
    for i in range(w, n):
        seg = D[i - w:i]
        t = np.arange(-w, 0, dtype=float)
        a = np.polyfit(t, seg, 2)
        pred[i] = np.polyval(a, 0.0)
    res = D - pred                               # NaN before the first window

    h = max(5, int(LOCAL_BW * n))
    sl = np.full(n, np.nan)
    for i in range(n):
        lo = max(0, i - h + 1)
        seg = res[lo:i + 1]
        seg = seg[np.isfinite(seg)]
        if len(seg) >= 5:
            sl[i] = robust_scale(seg)
    sl = np.where(np.isfinite(sl) & (sl > 0), sl, np.nan)

    dens = np.zeros(n)
    ok = np.isfinite(sl) & (np.arange(n) >= w - 1)
    dens[ok] = (dt[ok] / sl[ok]) ** 2
    if dens.sum() <= 0:
        return None, None
    return dens, res


def centred_density(x):
    """The paper's own density, for the side-by-side."""
    from nonparam import weighted_density
    return weighted_density(x)


def tau_of(dens, res, rng):
    """tau_min after the same noise-floor subtraction the paper applies.

    Without it the centred arm does not reproduce the published gap, and a
    comparison whose control does not match the paper is worth nothing.
    """
    from nonparam import estimate_floor
    d = np.clip(dens - estimate_floor(dens, res, "surrogate", rng), 0.0, None)
    if d.sum() <= 0:
        return None
    F = np.cumsum(d) / d.sum()
    k = int(np.searchsorted(F, Q))
    return min((k + 1) / len(d), 1.0)


FLOOR_REPS = 8          # as nonparam.FLOOR_REPS
MIN_POOL = 10           # residuals needed before a floor is estimated
CHECKPOINTS = 40        # floor re-estimated this many times along a record


def fast_local_scale(r, half):
    """pipeline.local_scale, vectorised in the interior; identical values."""
    n = len(r)
    out = np.empty(n)
    if n > 2 * half:
        W = np.lib.stride_tricks.sliding_window_view(r, 2 * half + 1)
        med = np.median(W, axis=1)
        out[half:n - half] = 1.4826 * np.median(np.abs(W - med[:, None]),
                                                axis=1)
        edges = list(range(half)) + list(range(n - half, n))
    else:
        edges = range(n)
    for i in edges:
        lo, hi = max(0, i - half), min(n, i + half + 1)
        out[i] = robust_scale(r[lo:hi])
    return np.maximum(out, 1e-9)


def floor_level(pool, n, rng, reps=FLOOR_REPS):
    """nonparam.estimate_floor('surrogate') with the residual pool given."""
    w = _win(n)
    acc = []
    for _ in range(reps):
        sr = pool[rng.integers(0, len(pool), n)]
        sc = robust_scale(sr)
        if not np.isfinite(sc) or sc <= 0:
            continue
        Dz = (sr - np.median(sr)) / sc
        dt = savgol_filter(Dz, w, 2, deriv=1, delta=1.0 / n)
        r2 = Dz - savgol_filter(Dz, w, 2)
        sl = np.clip(fast_local_scale(r2, max(5, int(LOCAL_BW * n))), 1e-12,
                     None)
        acc.append((dt / sl) ** 2)
    return float(np.median(np.concatenate(acc))) if acc else 0.0


def tau_of_strict(dens, res, rng):
    """tau_min with a noise floor estimated from the past residuals only."""
    n = len(dens)
    fin = np.isfinite(res)
    idx = np.flatnonzero(fin)
    if len(idx) < MIN_POOL:
        return None
    i0 = int(idx[MIN_POOL - 1])
    step = max(1, (n - i0) // CHECKPOINTS)
    floor = np.full(n, np.inf)
    for c in range(i0, n, step):
        pool = res[:c + 1][fin[:c + 1]]
        floor[c:c + step] = floor_level(pool, n, rng)
    d = np.clip(dens - floor, 0.0, None)
    if d.sum() <= 0:
        return None
    F = np.cumsum(d) / d.sum()
    k = int(np.searchsorted(F, Q))
    return min((k + 1) / len(d), 1.0)


# ---------------------------------------------------------------- measures --
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

def one_measure(job):
    """Ages of one measure on one rig, under all three arms."""
    rig, fn, lab, kind = job
    fun = dict((m[0], m[2]) for m in MEASURES)[lab]
    z = np.load(os.path.join(HERE, fn))
    f = z["centres"] / 1000.0
    BK = sorted(k for k in z.files if k != "centres")
    taus = {"causal": [], "causal_record_floor": [], "centred": []}
    for b in BK:
        x = np.asarray(fun(z[b].astype(float), f), float)
        x = x[np.isfinite(x)]
        if len(x) < 60:
            continue
        for scheme, fn_ in (("causal", causal_density),
                            ("centred", centred_density)):
            d, r = fn_(x)
            if d is None or r is None or np.sum(d) <= 0:
                continue
            d = np.asarray(d, float)
            r = np.asarray(r, float)
            rr = np.where(np.isfinite(r), r, 0.0)
            t = tau_of(d, rr, np.random.default_rng(SEED))
            if scheme == "centred":
                if t is not None:
                    taus["centred"].append(t)
                continue
            if t is not None:
                taus["causal_record_floor"].append(t)
            t2 = tau_of_strict(d, r, np.random.default_rng(SEED))
            if t2 is not None:
                taus["causal"].append(t2)
    if not all(taus.values()):
        return None
    return rig, dict(measure=lab, kind=kind,
                     **{k: float(np.median(v)) for k, v in taus.items()})


if __name__ == "__main__":
    from multiprocessing import Pool
    jobs = [(rig, fn, m[0], m[1]) for rig, fn in RIGS.items()
            if os.path.exists(os.path.join(HERE, fn)) for m in MEASURES]
    with Pool(min(10, len(jobs))) as pool:
        got = pool.map(one_measure, jobs)
    out = {rig: [] for rig in RIGS}
    for g_ in got:
        if g_ is not None:
            out[g_[0]].append(g_[1])
    for rig, rows in out.items():
        print("%s: %d measures" % (rig, len(rows)))
    print()
    print("%-22s %-13s %9s %9s %8s" % ("measure", "kind", "centred", "causal",
                                       "shift"))
    print("-" * 66)
    summary = {}
    for rig, rows in out.items():
        print(rig)
        for r in rows:
            print("  %-20s %-13s %9.3f %9.3f %+8.3f"
                  % (r["measure"], r["kind"], r["centred"], r["causal"],
                     r["causal"] - r["centred"]))
        g = {}
        for scheme in ("centred", "causal", "causal_record_floor"):
            med = {k: float(np.median([r[scheme] for r in rows if r["kind"] == k]))
                   for k in ("distribution", "amount")}
            g[scheme] = dict(median=med, gap=med["amount"] - med["distribution"])
        summary[rig] = g
        print("  %-34s gap %.3f (centred)  %.3f (causal)  %.3f (record floor)"
              % ("", g["centred"]["gap"], g["causal"]["gap"],
                 g["causal_record_floor"]["gap"]))
        print()

    json.dump(dict(per_measure=out, summary=summary),
              open(os.path.join(HERE, "causal_check.json"), "w"), indent=1)
    print("written to causal_check.json")
