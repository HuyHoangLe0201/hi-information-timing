r"""Is the noise scale a function of observation age or of damage-clock age?

A reviewer put the question precisely.  The paper writes

    X_k = D(tau_k + delta) + e_k,   e_k ~ N(0, sigma(tau_k)^2),

so the clock offset shifts the mean trend while the noise scale stays indexed
by the age at which the sample was taken.  If sigma is a property of the
measurement -- sensor noise, quantisation, the operating point the machine is
run at -- that is right.  If it is a property of the damage state, then a unit
running ahead on its damage clock should show the scale of the age it has
reached, sigma(tau_k + delta), and the Fisher information is no longer
D'^2/sigma^2 alone.

For X ~ N(mu(theta), sigma(theta)^2) the information about theta is

    I = mu'^2 / sigma^2  +  2 sigma'^2 / sigma^2,

so indexing the scale by the damage clock adds the non-negative term
2 (sigma'/sigma)^2 at every sample.  Two things follow and both are
consequences, not opinions.

  (1) The reported G is then a LOWER bound on the information, so G^{-1/2} is
      an over-estimate of the floor.  Proposition 1(i) -- that no unbiased
      estimator beats G^{-1/2} -- is a statement under the stated model and
      would not survive the substitution.  The assumption is material and the
      paper has to say so.

  (2) The RANKING moves only if the neglected term is a different fraction of
      the total for one family of indicators than for the other.

(2) is measurable, and so is the size of the term.  This script measures both.

The scale term is differentiated noise like any other, so it carries the same
spurious density the paper already subtracts from the mean term, and it is
given the same treatment: a surrogate floor built by resampling the record's
own residual, measured through the identical estimator, and subtracted before
anything is summed.  Without that the answer would be a measurement of the
estimator rather than of the record.

    python clock_scale.py   ->  clock_scale.json
"""
import json
import os

import numpy as np
from scipy.signal import savgol_filter

from pipeline import robust_scale, oof_trend, local_scale, _win, LOCAL_BW, prepare
from nonparam import estimate_floor

HERE = os.path.dirname(os.path.abspath(__file__))
Q = 0.35
EPS = 1e-30
SEED = 20260923
REPS = 8
SL_MIN = 1e-6            # a local scale below this is degenerate, not small


# ------------------------------------------------------------- the two terms --
def terms(x):
    """(mean term, scale term, residual) at every sample of one record.

    mean term   D'(tau)^2 / sigma(tau)^2      -- what the paper accumulates
    scale term  2 sigma'(tau)^2 / sigma(tau)^2 -- what a clock-indexed scale adds

    sigma is the pipeline's own rolling robust scale, and sigma' is taken from
    it with the same Savitzky-Golay window the derivative of the trend uses, so
    the two terms are estimated at one bandwidth and are comparable.
    """
    x = np.asarray(x, float)
    n = len(x)
    if n < 60 or not np.all(np.isfinite(x)):
        return None, None, None
    s = robust_scale(x)
    if not np.isfinite(s) or s <= 0:
        s = float(np.std(x))
    if not np.isfinite(s) or s <= 0:
        return None, None, None
    D = (x - np.median(x)) / s
    w = _win(n)
    if w < 7:
        return None, None, None
    dt = savgol_filter(D, w, 2, deriv=1, delta=1.0 / n)
    res = D - oof_trend(D, w // 2)
    sl = np.clip(local_scale(res, max(5, int(LOCAL_BW * n))), 1e-12, None)
    dsl = savgol_filter(sl, w, 2, deriv=1, delta=1.0 / n)
    # Where the residual has no dispersion at all -- quantised channels with
    # runs of identical values, which several battery channels have -- the
    # rolling scale returns zero and is clipped to 1e-12.  Neither term is
    # estimable there: the mean term divides by the clip and the scale term
    # divides by it twice.  D is normalised to unit robust scale, so a local
    # scale below SL_MIN of that is degenerate rather than small, and both
    # terms are set to zero at those samples.
    ok = sl > SL_MIN
    m = np.where(ok, (dt / sl) ** 2, 0.0)
    sc = np.where(ok, 2.0 * (dsl / sl) ** 2, 0.0)
    return m, sc, res


def scale_floor(res, rng, reps=REPS):
    """What the scale term reads on a record with no degradation in it.

    The surrogate is the record's own residual resampled, so it keeps the noise
    distribution and destroys the trend.  The level returned is the median over
    every sample of every surrogate, matching nonparam.estimate_floor.
    """
    n = len(res)
    w = _win(n)
    h = max(5, int(LOCAL_BW * n))
    acc = []
    for _ in range(reps):
        z = res[rng.integers(0, n, n)]
        sc = robust_scale(z)
        if not np.isfinite(sc) or sc <= 0:
            continue
        Dz = (z - np.median(z)) / sc
        r2 = Dz - savgol_filter(Dz, w, 2)
        sl = np.clip(local_scale(r2, h), 1e-12, None)
        dsl = savgol_filter(sl, w, 2, deriv=1, delta=1.0 / n)
        acc.append(np.where(sl > SL_MIN, 2.0 * (dsl / sl) ** 2, 0.0))
    if not acc:
        return 0.0
    return float(np.median(np.concatenate(acc)))


def both(x, rng):
    """Floor-corrected mean and scale terms, or None."""
    m, sc, res = terms(x)
    if m is None:
        return None
    m = np.clip(m - estimate_floor(m, res, "surrogate", rng), 0.0, None)
    sc = np.clip(sc - scale_floor(res, rng), 0.0, None)
    if m.sum() <= 0:
        return None
    return m, sc


def tau_of(d):
    if d.sum() <= 0:
        return None
    F = np.cumsum(d) / d.sum()
    k = int(np.searchsorted(F, Q))
    return min((k + 1) / len(d), 1.0)


# ------------------------------------------------------- part 1: how large --
def load_domain(tag):
    """The CRB corpus, three channels per unit, as crb_full.py loads it."""
    us = []
    if tag == "bearing":
        z = np.load(os.path.join(HERE, "feat_raw.npz"), allow_pickle=True)
        FN = [str(s) for s in z["featnames"]]
        I = {k: i for i, k in enumerate(FN)}
        for b in sorted(k for k in z.files if k != "featnames"):
            M = z[b]
            for nm, v in (("rms", M[:, I["rms"]]),
                          ("lf", M[:, I["b0_1"]] + M[:, I["b1_2"]]),
                          ("hf", M[:, I["b4_6"]] + M[:, I["b6_8"]]
                           + M[:, I["b8_10"]])):
                u, _ = prepare(np.asarray(v, float), "%s/%s" % (b, nm))
                if u:
                    us.append(u)
    elif tag == "battery":
        z = np.load(os.path.join(HERE, "battery_raw.npz"))
        for cell in sorted({k.split("__")[0] for k in z.files}):
            for ind in ("cap", "Re", "Rct"):
                u, _ = prepare(z["%s__%s" % (cell, ind)], "%s/%s" % (cell, ind))
                if u:
                    us.append(u)
    elif tag.startswith("turbofan"):
        sub = tag.split("_")[1]
        f = "cmapss_%s_norm.npz" % sub if sub == "FD004" else "cmapss_%s.npz" % sub
        z = np.load(os.path.join(HERE, f))
        for nm in sorted({k.split("__")[0] for k in z.files}):
            S = z["%s__sensors" % nm]
            for c in (4, 11, 14):
                u, _ = prepare(S[:, c - 1].astype(float), "%s/s%d" % (nm, c))
                if u:
                    us.append(u)
    return us


def series_of(u):
    """prepare() returns a dict; D is the record normalised once."""
    if isinstance(u, dict):
        return np.asarray(u["D"], float) if "D" in u else None
    return np.asarray(u, float)


def main():
    print("=" * 78)
    print("PART 1  the neglected term as a share of the total information")
    print("=" * 78)
    print("%-16s%8s%12s%12s%12s" % ("domain", "series", "median", "q75", "q90"))
    print("-" * 78)

    rng = np.random.default_rng(SEED)
    part1 = {}
    for tag in ("bearing", "turbofan_FD001", "turbofan_FD004", "battery"):
        ratios = []
        for u in load_domain(tag):
            x = series_of(u)
            if x is None:
                continue
            r = both(x, rng)
            if r is None:
                continue
            m, sc = r
            tot = m.sum() + sc.sum()
            if tot <= 0:
                continue
            ratios.append(float(sc.sum() / tot))
        if not ratios:
            continue
        a = np.asarray(ratios)
        part1[tag] = dict(series=len(a), median=float(np.median(a)),
                          q75=float(np.percentile(a, 75)),
                          q90=float(np.percentile(a, 90)))
        print("%-16s%8d%12.4f%12.4f%12.4f"
              % (tag, len(a), np.median(a), np.percentile(a, 75),
                 np.percentile(a, 90)))

    allr = [v["median"] for v in part1.values()]
    print("-" * 78)
    print("the neglected term is a median %.1f%% to %.1f%% of the total"
          % (100 * min(allr), 100 * max(allr)))

    # ------------------------------------------- part 2: does the ordering move --
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

    print()
    print("=" * 78)
    print("PART 2  does adding it move the family ordering?")
    print("=" * 78)

    part2 = {}
    for rig, fn in RIGS.items():
        path = os.path.join(HERE, fn)
        if not os.path.exists(path):
            print("%s: %s not found" % (rig, fn))
            continue
        z = np.load(path)
        f = z["centres"] / 1000.0
        BK = sorted(k for k in z.files if k != "centres")
        rows = []
        for lab, kind, fun in MEASURES:
            mean_only, with_scale, frac = [], [], []
            for b in BK:
                x = np.asarray(fun(z[b].astype(float), f), float)
                x = x[np.isfinite(x)]
                if len(x) < 60:
                    continue
                r = both(x, np.random.default_rng(SEED))
                if r is None:
                    continue
                m, sc = r
                t0, t1 = tau_of(m), tau_of(m + sc)
                if t0 is None or t1 is None:
                    continue
                mean_only.append(t0)
                with_scale.append(t1)
                frac.append(float(sc.sum() / max(m.sum() + sc.sum(), EPS)))
            if mean_only:
                rows.append(dict(measure=lab, kind=kind,
                                 mean_only=float(np.median(mean_only)),
                                 with_scale=float(np.median(with_scale)),
                                 scale_share=float(np.median(frac))))
        g = {}
        for scheme in ("mean_only", "with_scale"):
            med = {k: float(np.median([r[scheme] for r in rows if r["kind"] == k]))
                   for k in ("distribution", "amount")}
            g[scheme] = dict(median=med, gap=med["amount"] - med["distribution"])
        share = {k: float(np.median([r["scale_share"] for r in rows
                                     if r["kind"] == k]))
                 for k in ("distribution", "amount")}
        part2[rig] = dict(rows=rows, summary=g, share=share)

        print(rig)
        print("  %-22s%-14s%10s%12s%10s"
              % ("measure", "family", "mean only", "with scale", "share"))
        for r in rows:
            print("  %-22s%-14s%10.3f%12.3f%10.3f"
                  % (r["measure"], r["kind"], r["mean_only"], r["with_scale"],
                     r["scale_share"]))
        print("  %-36s gap %.3f -> %.3f" % ("", g["mean_only"]["gap"],
                                            g["with_scale"]["gap"]))
        print("  scale share of the total: distribution %.3f, amount %.3f"
              % (share["distribution"], share["amount"]))
        print()

    json.dump(dict(corpus=part1, rigs=part2, q=Q, reps=REPS, seed=SEED),
              open(os.path.join(HERE, "clock_scale.json"), "w"), indent=1)
    print("written to clock_scale.json")


if __name__ == "__main__":
    main()
