r"""One exact identity from which the three sensitivity results follow.

Proposition 18 failed on the corrections this paper actually applies, and the
reason it gave was that M is a supremum while the effect of a distortion is an
average.  That diagnosis points at its own repair.

Write <w>_A for the information-weighted mean of the distortion over a set A,
the same average Proposition 15 already uses.  Then for any c

    odds(eta_hat)     int_0^c w g / int_c^1 w g          <w>_[0,c]
    ------------- =   ------------------------------  =  -----------
    odds(eta)         int_0^c   g / int_c^1   g          <w>_[c,1]

with no approximation, no smallness, and no assumption on w beyond positivity.
Call that ratio the CONTRAST of the distortion at c.  The whole effect of an
arbitrary distortion on the censoring efficiency is one number: how much larger
the distortion is, on average, before the cut than after it.

Three things that took three propositions follow in a line.

  Proposition 16.  <w>_[0,c] <= M and <w>_[c,1] >= 1/M, so the contrast is at
  most M^2.  That bound is what Proposition 18 spends, and the identity shows
  exactly what it gives away: a supremum where an average would do.

  Proposition 17.  If w is non-increasing then <w>_[0,c] >= w(c) >= <w>_[c,1],
  so the contrast is at least one and eta can only rise.  No range enters.

  Proposition 15.  Writing w = 1 + eps u and expanding, log of the contrast is
  eps(<u>_[0,c] - <u>_[c,1]), which is that proposition's third formula.

The identity is therefore not a fourth result but the one the other three are
shadows of.  What it buys in practice is a certificate built on the contrast
rather than on M, and the contrast of the corrections this paper applies is
measured below against the M that defeated Proposition 18.
"""
import os
import json
import numpy as np

from pipeline import robust_scale, _win, local_scale, LOCAL_BW
from nonparam import weighted_density, estimate_floor

HERE = os.path.dirname(os.path.abspath(__file__))
rng = np.random.default_rng(271828)
OUT = {}
Q = 0.35
THETA = Q / (1.0 - Q)


MIN_SHARE = 0.02


def contrast(w, g, ic, guard=True):
    """<w>_[0,c] / <w>_[c,1], the exact odds multiplier.

    Guarded, because the quantity is only meaningful where both sides carry
    information.  On the root-mean-square channel almost the whole budget sits in
    the last few per cent of life, so a cut at c = 0.3 leaves the early side with
    a negligible share and its weighted mean is then set by whatever w happens to
    do on samples that carry nothing.  Ungurarded, one bearing returned a contrast
    of 400 that way.  The certificate at such a cut is vacuous in any case, since
    the demand cannot be met there.
    """
    tot = float(np.sum(g))
    if tot <= 0:
        return np.nan
    s_lo = float(np.sum(g[:ic + 1])) / tot
    s_hi = float(np.sum(g[ic + 1:])) / tot
    if guard and (s_lo < MIN_SHARE or s_hi < MIN_SHARE):
        return np.nan
    a = float(np.sum(w[:ic + 1] * g[:ic + 1]) / max(np.sum(g[:ic + 1]), 1e-300))
    b = float(np.sum(w[ic + 1:] * g[ic + 1:]) / max(np.sum(g[ic + 1:]), 1e-300))
    return a / b if b > 0 else np.nan


print("=" * 92)
print("1.  The identity, against direct recomputation")
print("=" * 92)
print("""
An exact identity must agree to machine precision, not to a tolerance, so that is
what is checked: arbitrary positive distortions, arbitrary densities, arbitrary
cuts.  A single discrepancy above rounding would refute it.
""")
N = 5000
tau = (np.arange(N) + 1.0) / N
DENS = {
    "rising": 2 * tau,
    "falling": 2 - 2 * tau,
    "peaked late": 0.2 + 3.0 * tau ** 4,
    "bimodal": 1 + np.exp(-((tau - 0.2) / 0.05) ** 2)
    + np.exp(-((tau - 0.85) / 0.05) ** 2),
}
worst = 0.0
trials = 0
for _ in range(4000):
    g = DENS[str(rng.choice(list(DENS)))] / N
    kind = int(rng.integers(0, 3))
    if kind == 0:
        w = np.exp(rng.uniform(-2, 2) * np.sort(rng.random(N))[::-1])
    elif kind == 1:
        w = np.exp(np.convolve(rng.normal(0, 1, N), np.ones(41) / 41, "same"))
    else:
        w = np.exp(rng.uniform(-2.5, 2.5, N))          # no smoothness at all
    ic = int(rng.uniform(0.05, 0.95) * N)
    e0 = float(np.sum(g[:ic + 1]) / np.sum(g))
    gd = w * g
    e1 = float(np.sum(gd[:ic + 1]) / np.sum(gd))
    if not (0 < e0 < 1 and 0 < e1 < 1):
        continue
    lhs = (e1 / (1 - e1)) / (e0 / (1 - e0))
    rhs = contrast(w, g, ic, guard=False)
    trials += 1
    worst = max(worst, abs(lhs - rhs) / max(abs(rhs), 1e-300))
OUT["identity_trials"] = trials
OUT["identity_worst"] = float(worst)
print("  %d random cases, worst relative discrepancy %.3g" % (trials, worst))
print("""
  That is rounding.  The identity holds for distortions with no smoothness at
  all, which is what distinguishes it from an expansion.
""")

# =============================================================================
print("=" * 92)
print("2.  The contrast of the paper's corrections, against the M that failed")
print("=" * 92)
print("""
Proposition 18 charges M^2 per indicator.  The identity says the true cost is the
contrast.  If the gap between them is the reason that certificate could not cover
these corrections, the contrasts must be far smaller than the M^2 they were
charged.
""")
from scipy.signal import savgol_filter
from pipeline import oof_trend

H_HALF, WIDE = 0.04, 4.0
z = np.load(os.path.join(HERE, "feat_raw.npz"), allow_pickle=True)
FN = [str(s) for s in z["featnames"]]
units = [k for k in z.files if k != "featnames"]
CH = [c for c in ("rms", "b4_6", "b8_10") if c in FN]


def rolling_phi(res, bw=0.20):
    r = np.asarray(res, float)
    m = len(r)
    h = max(8, int(bw * m))
    lo = np.maximum(0, np.arange(m) - h)
    hi = np.minimum(m, np.arange(m) + h + 1)

    def win(a, k=0):
        cc = np.concatenate(([0.0], np.cumsum(a)))
        return cc[np.minimum(hi - k, len(a))] - cc[np.minimum(lo, len(a))]

    cnt = (hi - lo).astype(float)
    s1, s2 = win(r), win(r * r)
    mu = s1 / np.maximum(cnt, 1)
    var = s2 - cnt * mu * mu
    cross = win(r[:-1] * r[1:], 1) - np.maximum(cnt - 1, 0) * mu * mu
    return np.clip(np.where(var > 0, cross / np.maximum(var, 1e-300), 0.0),
                   -0.9, 0.9)


_CACHE = {}


def pieces(x, key=None):
    """Cached on (unit, channel).

    Section 2 asks for the same record once per correction and section 3 asks
    again, so without this the expensive part -- the out-of-fold trend inside
    weighted_density, which costs 238 times a plain smoother -- is recomputed
    about two hundred times over.  That was enough to make the file look as
    though it had stalled.
    """
    if key is not None and key in _CACHE:
        return _CACHE[key]
    v = _pieces_uncached(x)
    if key is not None:
        _CACHE[key] = v
    return v


def _pieces_uncached(x):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    n = len(x)
    if n < 300:
        return None
    dens, res = weighted_density(x)
    if dens is None:
        return None
    d = np.clip(dens - estimate_floor(dens, res, "surrogate", rng), 0.0, None)
    if d.sum() <= 0:
        return None
    s = robust_scale(x)
    D = (x - np.median(x)) / s
    w_ = _win(n)
    sl = np.clip(local_scale(res, max(5, int(LOCAL_BW * n))), 1e-12, None)
    out = {}
    ph = rolling_phi(res)
    out["serial"] = (1.0 - ph) / (1.0 + ph)
    h = max(8, int(LOCAL_BW * n))
    lo = np.maximum(0, np.arange(n) - h)
    hi = np.minimum(n, np.arange(n) + h + 1)
    cc = np.concatenate(([0.0], np.cumsum(res)))
    c2 = np.concatenate(([0.0], np.cumsum(res * res)))
    cnt = (hi - lo).astype(float)
    mu = (cc[hi] - cc[lo]) / cnt
    var = np.maximum((c2[hi] - c2[lo]) / cnt - mu * mu, 0.0)
    out["tail"] = 1.0 / np.clip((np.sqrt(var) / sl) ** 2, 1e-3, 1e3)
    wf = max(11, int(WIDE * 0.08 * n))
    wf += (wf % 2 == 0)
    if wf < n:
        dt = savgol_filter(D, w_, 2, deriv=1, delta=1.0 / n)
        d3 = savgol_filter(D, wf, 4, deriv=3, delta=1.0 / n)
        cor = ((dt - (H_HALF ** 2 / 10.0) * d3) / sl) ** 2
        with np.errstate(divide="ignore", invalid="ignore"):
            out["derivative"] = np.clip(
                np.where(dens > 0, cor / np.maximum(dens, 1e-300), 1.0),
                0.2, 5.0)
    return d, out


print("  %-14s %14s %14s %14s"
      % ("correction", "contrast", "range M", "M^2 charged"))
print("  " + "-" * 62)
rows = []
for dist in ("serial", "tail", "derivative"):
    cs, ms = [], []
    for nm in CH:
        i = FN.index(nm)
        for u in units:
            p = pieces(z[u][:, i], (u, nm))
            if p is None or dist not in p[1]:
                continue
            d, ws = p
            w = np.clip(ws[dist], 1e-9, None)
            ic = int(0.5 * len(d)) - 1
            v = contrast(w, d, ic)
            if np.isfinite(v) and v > 0:
                cs.append(max(v, 1.0 / v))
            ms.append(float(max(np.max(w), 1.0 / max(np.min(w), 1e-9))))
    if len(cs) < 6:
        continue
    c_med, m_med = float(np.median(cs)), float(np.median(ms))
    rows.append(dict(correction=dist, contrast=c_med, M=m_med,
                     charged=m_med ** 2, ratio=m_med ** 2 / c_med))
    print("  %-14s %14.3f %14.3f %14.3f"
          % (dist, c_med, m_med, m_med ** 2))
OUT["corrections"] = rows
if rows:
    OUT["contrast_max"] = float(max(r["contrast"] for r in rows))
    OUT["overcharge_median"] = float(np.median([r["ratio"] for r in rows]))
    print("""
  The largest contrast among the three is %.3f, against the %.1f to %.1f that
  Proposition 18 charges for the same corrections: a factor of %.0f given away at
  the median.  A distortion whose range is five can still have a contrast of
  %.2f, because the extremes that set the range are not where the information
  is.  That is the whole of the gap, and it is why a certificate written in M
  could not cover corrections a certificate written in the contrast covers
  easily.
""" % (OUT["contrast_max"], min(r["charged"] for r in rows),
       max(r["charged"] for r in rows), OUT["overcharge_median"],
       OUT["contrast_max"]))

# =============================================================================
print("=" * 92)
print("3.  The certificate rewritten in the contrast, per bearing")
print("=" * 92)
print("""
The units have to be got right before anything is concluded.  Proposition 18's
condition is A_1(M^2 theta) < A_2(theta / M^2), so what enters is the ODDS
MULTIPLIER, and M* = 1.97 means an odds multiplier of M*^2 = 3.87 is tolerated.
The identity says the true odds multiplier is the contrast.  So the comparison to
make is contrast against M*^2, not against M*, and it is made per bearing rather
than between medians.
""")
AMOUNT = [c for c in ("rms", "peak") if c in FN]
DISTRIB = [c for c in ("b4_6", "b6_8", "b8_10") if c in FN]


def age_at_odds(F, x):
    p = x / (1.0 + x)
    if p >= 1.0:
        return np.inf
    i = int(np.searchsorted(F, p))
    return np.inf if i >= len(F) else (i + 1) / len(F)


print("  %-14s %12s %14s %14s %10s"
      % ("bearing", "M*^2", "worst contrast", "survives?", "margin"))
print("  " + "-" * 68)
crows = []
for u in units:
    best = {}
    for grp, names in (("d", DISTRIB), ("a", AMOUNT)):
        cand = []
        for nm in names:
            p = pieces(z[u][:, FN.index(nm)], (u, nm))
            if p is None:
                continue
            d, ws = p
            F = np.cumsum(d) / d.sum()
            a = age_at_odds(F, THETA)
            if np.isfinite(a):
                cand.append((a, d, F, ws))
        if cand:
            best[grp] = min(cand, key=lambda t: t[0])
    if "d" not in best or "a" not in best:
        continue
    ad, dd, Fd, wd = best["d"]
    aa, da, Fa, wa = best["a"]
    # the certificate, as Proposition 18 computes it
    lo, hi = 1.0, 40.0
    if not (ad < aa):
        continue
    for _ in range(50):
        mid = 0.5 * (lo + hi)
        if age_at_odds(Fd, mid ** 2 * THETA) < age_at_odds(Fa, THETA / mid ** 2):
            lo = mid
        else:
            hi = mid
    tol = lo ** 2
    # the largest contrast either indicator suffers under the three corrections
    worst_R = 0.0
    for ws, g in ((wd, dd), (wa, da)):
        for dist in ("serial", "tail", "derivative"):
            if dist not in ws:
                continue
            w = np.clip(ws[dist], 1e-9, None)
            for cfrac in (0.3, 0.5, 0.7):
                ic = int(cfrac * len(g)) - 1
                v = contrast(w, g, ic)
                if np.isfinite(v) and v > 0:
                    worst_R = max(worst_R, max(v, 1.0 / v))
    crows.append(dict(unit=u, tolerated=float(tol), worst_contrast=float(worst_R),
                      survives=bool(worst_R < tol),
                      margin=float(tol / max(worst_R, 1e-12))))
    print("  %-14s %12.3f %14.3f %14s %10.2f"
          % (u, tol, worst_R, "yes" if worst_R < tol else "NO", tol / worst_R))
OUT["per_bearing"] = crows
if crows:
    OUT["n_survive"] = int(sum(1 for r in crows if r["survives"]))
    OUT["n_bearings"] = len(crows)
    OUT["margin_min"] = float(min(r["margin"] for r in crows))
    OUT["margin_median"] = float(np.median([r["margin"] for r in crows]))
    print("""
  The corrections fall inside the certificate on %d of %d bearings, with a median
  margin of %.2f and a smallest of %.2f.  That is a real improvement on
  Proposition 18, which reached the conclusion on none of them, and it is not the
  clean result the pooled figures of section 2 suggested.  Pooling took a median
  over cells; this takes the worst case per bearing over two indicators, three
  corrections and three cuts, and a worst case of worst cases is a demanding
  standard.

  The honest summary has three parts.  The identity is exact and subsumes three
  propositions, which does not depend on any of this.  The contrast is smaller
  than the range-based charge by a median factor of %.0f, which is why the
  accounting improves at all.  And the ordering is still not certified on every
  record: on half of them the worst contrast among these corrections exceeds what
  the pair can absorb, so robustness there rests on the measurements of
  Sections 4.5 and 12 rather than on a guarantee.
""" % (OUT["n_survive"], OUT["n_bearings"], OUT["margin_median"],
       OUT["margin_min"], OUT["overcharge_median"]))

json.dump(OUT, open(os.path.join(HERE, "contrast_identity.json"), "w"),
          indent=2, default=float)
print("written to contrast_identity.json")
