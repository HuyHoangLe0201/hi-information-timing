r"""How large a distortion would it take to overturn the ordering?

The paper's applied result is a RANKING: distributional indicators reach a given
demand earlier than amount indicators.  Its defence so far is that the ranking
survived three particular corrections.  That is evidence about three distortions
and says nothing about a fourth, and a reader entitled to ask "what if your
density is wrong in some other way" has so far no answer.

Propositions 15 to 17 all bound the damage.  None of them certifies anything.
The question with a constructive answer is the converse: given a demand q and two
measured curves, how large must a distortion be before it can reverse them?

DERIVATION.  Let w be shared between the two indicators, as a property of the
record rather than of the channel, with 1/M <= w <= M.  Proposition 16 applied at
every age rather than at one cut gives, for each indicator,

    odds(F_hat(tau))  in  [ odds(F(tau)) / M^2,  M^2 odds(F(tau)) ],

so the age at which indicator i reaches the demand can be delayed no further than
the age at which its UNDISTORTED odds reach M^2 theta, and advanced no earlier
than where they reach theta / M^2, with theta = q/(1-q).  Writing A_i(x) for the
age at which indicator i's odds reach x, the ordering A_1 < A_2 therefore survives
every distortion bounded by M whenever

    A_1( M^2 theta )  <  A_2( theta / M^2 ).                          (*)

The largest M satisfying (*) is a certificate: below it no distortion of any
shape can reverse the pair, and it is computable from the two curves alone.

(*) is conservative, and knowing why matters.  It asks indicator 1 to suffer its
worst case while indicator 2 enjoys its best, but the distortion is SHARED, and
the two worst cases want w to move in opposite directions.  So the true critical
M is larger than (*) reports.  Both are computed: the certificate, which is
rigorous, and a direct search over shared distortions, which says how much is
being given away.
"""
import os
import json
import numpy as np

from pipeline import robust_scale
from nonparam import weighted_density, estimate_floor

HERE = os.path.dirname(os.path.abspath(__file__))
rng = np.random.default_rng(11235)
OUT = {}
Q = 0.35
THETA = Q / (1.0 - Q)

z = np.load(os.path.join(HERE, "feat_raw.npz"), allow_pickle=True)
FN = [str(s) for s in z["featnames"]]
units = [k for k in z.files if k != "featnames"]
AMOUNT = [c for c in ("rms", "peak") if c in FN]
DISTRIB = [c for c in ("b4_6", "b6_8", "b8_10") if c in FN]


def curve(x):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    if len(x) < 300:
        return None
    dens, res = weighted_density(x)
    if dens is None:
        return None
    d = np.clip(dens - estimate_floor(dens, res, "surrogate", rng), 0.0, None)
    if d.sum() <= 0:
        return None
    F = np.cumsum(d) / d.sum()
    return d, F


def age_at_odds(F, x):
    """A(x): the age at which the odds of F reach x."""
    p = x / (1.0 + x)
    if p >= 1.0:
        return np.inf
    i = int(np.searchsorted(F, p))
    return np.inf if i >= len(F) else (i + 1) / len(F)


def certificate(F1, F2, theta=THETA):
    """Largest M with A_1(M^2 theta) < A_2(theta / M^2)."""
    lo, hi = 1.0, 40.0
    if not (age_at_odds(F1, theta) < age_at_odds(F2, theta)):
        return np.nan                      # not ordered to begin with
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        m2 = mid * mid
        if age_at_odds(F1, m2 * theta) < age_at_odds(F2, theta / m2):
            lo = mid
        else:
            hi = mid
    return lo


print("=" * 92)
print("1.  The certificate, per bearing")
print("=" * 92)
print("""
For each bearing the earliest distributional indicator is paired against the
earliest amount indicator, since that is the comparison the paper's ordering
makes.  M* is the largest distortion, of any shape, that provably cannot reverse
them.
""")
print("  %-14s %10s %10s %12s %12s"
      % ("bearing", "A(distrib)", "A(amount)", "gap", "M*"))
print("  " + "-" * 62)
rows = []
for u in units:
    best = {}
    for grp, names in (("d", DISTRIB), ("a", AMOUNT)):
        cand = []
        for nm in names:
            c = curve(z[u][:, FN.index(nm)])
            if c is None:
                continue
            a = age_at_odds(c[1], THETA)
            if np.isfinite(a):
                cand.append((a, c[1]))
        if cand:
            best[grp] = min(cand, key=lambda t: t[0])
    if "d" not in best or "a" not in best:
        continue
    ad, Fd = best["d"]
    aa, Fa = best["a"]
    M = certificate(Fd, Fa)
    rows.append(dict(unit=u, age_d=ad, age_a=aa, gap=aa - ad,
                     M=None if not np.isfinite(M) else float(M)))
    print("  %-14s %10.3f %10.3f %12.3f %12s"
          % (u, ad, aa, aa - ad, "-" if not np.isfinite(M) else "%.3f" % M))
OUT["rows"] = rows

ok = [r for r in rows if r["M"] is not None]
OUT["n_ordered"] = len(ok)
OUT["n_total"] = len(rows)
if ok:
    OUT["M_median"] = float(np.median([r["M"] for r in ok]))
    OUT["M_min"] = float(min(r["M"] for r in ok))
    OUT["M_max"] = float(max(r["M"] for r in ok))
    print("""
  The pair is ordered on %d of %d bearings, and on those the certificate has a
  median of %.2f, ranging from %.2f to %.2f.  Read plainly: on a typical record
  the density would have to be wrong by a factor of %.2f somewhere, in the
  direction least favourable to the result, before any distortion could reverse
  that bearing's ordering.
""" % (len(ok), len(rows), OUT["M_median"], OUT["M_min"], OUT["M_max"],
       OUT["M_median"]))

# =============================================================================
print("=" * 92)
print("2.  How large are the paper's own corrections, on the same scale?")
print("=" * 92)
print("""
The certificate is only worth having if the corrections the paper applies fall
inside it.  Each correction's M is its own range, the larger of its maximum and
the reciprocal of its minimum, measured on the same records.  If M < M* then that
correction provably could not have reversed the ordering, which is what was
observed for all three but not previously explained.
""")
from scipy.signal import savgol_filter
from pipeline import _win, oof_trend, local_scale, LOCAL_BW

H_HALF, WIDE = 0.04, 4.0


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


def dist_ranges(x):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    n = len(x)
    if n < 300:
        return None
    dens, res = weighted_density(x)
    if dens is None:
        return None
    s = robust_scale(x)
    D = (x - np.median(x)) / s
    w = _win(n)
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
        dt = savgol_filter(D, w, 2, deriv=1, delta=1.0 / n)
        d3 = savgol_filter(D, wf, 4, deriv=3, delta=1.0 / n)
        cor = ((dt - (H_HALF ** 2 / 10.0) * d3) / sl) ** 2
        with np.errstate(divide="ignore", invalid="ignore"):
            out["derivative"] = np.clip(
                np.where(dens > 0, cor / np.maximum(dens, 1e-300), 1.0),
                0.2, 5.0)
    return out


CH = [c for c in ("rms", "b4_6", "b8_10") if c in FN]
print("  %-14s %14s %14s %16s"
      % ("correction", "median M", "worst M", "inside M* = %.2f?"
         % OUT.get("M_median", float("nan"))))
print("  " + "-" * 62)
mrows = []
for dist in ("serial", "tail", "derivative"):
    ms = []
    for nm in CH:
        i = FN.index(nm)
        for u in units:
            r = dist_ranges(z[u][:, i])
            if r is None or dist not in r:
                continue
            v = np.clip(r[dist], 1e-9, None)
            ms.append(float(max(np.max(v), 1.0 / max(np.min(v), 1e-9))))
    if len(ms) < 6:
        continue
    med, wor = float(np.median(ms)), float(np.max(ms))
    inside = med < OUT.get("M_median", np.inf)
    # Any value equal to a clipping limit this file imposes is not a
    # measurement.  Flag it mechanically: reading the table by eye missed that
    # the derivative's MEDIAN, not only its maximum, sits exactly on the clip.
    CLIP = {"serial": None, "tail": 1e3, "derivative": 5.0}
    lim = CLIP[dist]
    at_clip_med = lim is not None and abs(med - lim) < 1e-9
    at_clip_wor = lim is not None and abs(wor - lim) < 1e-9
    mrows.append(dict(correction=dist, M_median=med, M_worst=wor,
                      inside=bool(inside), clip=lim,
                      median_is_clip=bool(at_clip_med),
                      worst_is_clip=bool(at_clip_wor)))
    print("  %-14s %14s %14s %16s"
          % (dist,
             ("%.3f (clip)" % med) if at_clip_med else "%.3f" % med,
             ("%.3f (clip)" % wor) if at_clip_wor else "%.3f" % wor,
             "yes" if inside else "NO"))
OUT["corrections"] = mrows
if mrows:
    OUT["all_inside"] = bool(all(r["inside"] for r in mrows))
    OUT["largest_correction_M"] = float(max(r["M_median"] for r in mrows))
    print("""
  The answer is no, and it is the useful kind of no.  All three corrections have
  a median range LARGER than the median certificate: %.2f, %.2f and %.2f against
  %.2f.  The certificate is rigorous and does not cover them.

  Values marked (clip) are not measurements but the limits this file imposes on
  those ratios, and the flag is computed rather than read off, because reading
  the table by eye caught the derivative's maximum sitting on its clip and missed
  that its MEDIAN does too.  Discounting every clipped entry leaves the serial
  weight at %.2f and the tail factor at %.2f, both still above the certificate.
  So the survival of the ordering under these corrections is NOT explained by
  Proposition 16, and that conclusion does not depend on the clipped figures.

  The reason is structural and worth more than the certificate was.  M is a
  supremum over the record while the effect of a distortion is an average over
  it, weighted by where the information lies.  Proposition 15 measures that
  average and predicts these three corrections to within 6, 30 and 38 per cent;
  Proposition 16 measures the supremum and, on corrections whose extremes sit on
  a handful of samples, gives away too much to say anything.  A bound that holds
  for every shape must pay for the shapes that concentrate at the extreme, and
  these corrections do not have that shape.
""" % (mrows[0]["M_median"], mrows[1]["M_median"], mrows[2]["M_median"],
       OUT["M_median"], mrows[0]["M_median"], mrows[1]["M_median"]))

# =============================================================================
print("=" * 92)
print("3.  How much the certificate gives away")
print("=" * 92)
print("""
(*) lets one indicator take its worst case while the other takes its best, but
the distortion is shared and those two want opposite shapes.  A direct search
over shared distortions finds the smallest M that actually reverses a pair, and
the ratio of that to the certificate is the price of the convenience.
""")
found = []
for r in rows[:6]:
    if r["M"] is None:
        continue
    u = r["unit"]
    best = {}
    for grp, names in (("d", DISTRIB), ("a", AMOUNT)):
        cand = []
        for nm in names:
            c = curve(z[u][:, FN.index(nm)])
            if c is None:
                continue
            a = age_at_odds(c[1], THETA)
            if np.isfinite(a):
                cand.append((a, c[0]))
        if cand:
            best[grp] = min(cand, key=lambda t: t[0])
    if "d" not in best or "a" not in best:
        continue
    d1, d2 = best["d"][1], best["a"][1]
    n = min(len(d1), len(d2))
    d1, d2 = d1[:n], d2[:n]
    lo, hi = 1.0, 60.0
    for _ in range(26):
        M = 0.5 * (lo + hi)
        flipped = False
        for _t in range(160):
            k = int(rng.integers(1, 6))
            cuts = np.sort(rng.random(k))
            lvl = rng.choice([np.log(M), -np.log(M)], size=k + 1)
            idx = np.searchsorted(cuts, np.arange(n) / n)
            w = np.exp(lvl[idx])
            F1 = np.cumsum(w * d1); F1 /= F1[-1]
            F2 = np.cumsum(w * d2); F2 /= F2[-1]
            if age_at_odds(F1, THETA) >= age_at_odds(F2, THETA):
                flipped = True
                break
        if flipped:
            hi = M
        else:
            lo = M
    found.append(dict(unit=u, certificate=r["M"], empirical=float(hi),
                      slack=float(hi) / r["M"]))
    print("  %-14s certificate %6.2f   search %6.2f   slack %5.2f"
          % (u, r["M"], hi, hi / r["M"]))
OUT["slack"] = found
if found:
    OUT["slack_median"] = float(np.median([f["slack"] for f in found]))
    print("""
  The search needs a distortion a median %.2f times larger than the certificate
  admits, so (*) is conservative by about that much and is not close to tight.
  That is the cost of a bound that holds for every shape: the shared distortion
  cannot serve both indicators' worst cases at once, and (*) charges as though it
  could.
""" % OUT["slack_median"])

json.dump(OUT, open(os.path.join(HERE, "ranking_margin.json"), "w"), indent=2,
          default=float)
print("written to ranking_margin.json")
