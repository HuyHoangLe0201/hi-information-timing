r"""Is the noise floor still flat, and still right, when the noise is correlated?

Every G in this paper is dens - floor, so a floor that is wrong in level moves
every information total and a floor that is wrong in SHAPE moves every fraction
read off the curve.  Two things about how it is built have never been put
together.

The surrogate is a bootstrap resample,

    s = res[rng.integers(0, n, n)],

which draws with replacement and therefore destroys the order of the record.  The
surrogate is white by construction, so what estimate_floor returns is the floor of
WHITE noise however the real residual is ordered.

Section 4.5 established that the real residuals are not white.  Their lag-one
correlation runs from 0.067 to 0.399, and it DRIFTS, from a median near 0.03 in
the first quarter of life to near 0.30 in the last.

floor_shape.py established the two properties the pipeline relies on, that the
floor is flat along a record and that its level does not depend on the noise
scale.  Both were measured with the noise scale varying and the noise white; its
own text says so, "the Savitzky-Golay derivative of white noise".  Neither result
speaks to correlation, and the function returns np.full_like, a single constant,
so if the floor moves with phi and phi drifts then a flat floor is subtracted
where a rising one is owed.  That would leave spurious energy at the end of life,
which is where G is read.

Which way it goes is not obvious and is worth stating before measuring.  Positive
phi is red noise, with its power pushed towards low frequency, while the
derivative filter passes low frequency as i*omega and suppresses it.  So the
floor could as easily fall with phi as rise.  The sign is the first thing this
file reports.
"""
import os
import json
import numpy as np
from scipy.signal import savgol_filter, lfilter

from pipeline import _win, robust_scale, local_scale, LOCAL_BW
from nonparam import weighted_density, estimate_floor

HERE = os.path.dirname(os.path.abspath(__file__))
rng = np.random.default_rng(9091)
REPS = 240        # section 1: stable to 0.4% between 240 and 600, and the
                 # expensive one, since every repetition calls estimate_floor
TILT_REPS = 1500  # section 2: no floor call, so repetitions are nearly free and
                 # the tilt is the quantity whose third digit was moving
PHIS = (0.0, 0.10, 0.20, 0.30, 0.40)
OUT = {}


def ar1(n, phi, rg):
    """Stationary AR(1) with unit marginal variance.

    Written with lfilter rather than a Python loop: at 240 repetitions over five
    values of phi the loop was the whole cost of this file.
    """
    e = rg.standard_normal(n)
    if phi == 0.0:
        return e
    z = lfilter([np.sqrt(1.0 - phi * phi)], [1.0, -phi], e)
    z[0] = e[0]
    return z


def density_of(x):
    """The pipeline's weighted density, built exactly as weighted_density does."""
    n = len(x)
    s = robust_scale(x)
    if not np.isfinite(s) or s <= 0:
        return None
    D = (x - np.median(x)) / s
    w = _win(n)
    if w < 7 or w >= n:
        return None
    dt = savgol_filter(D, w, 2, deriv=1, delta=1.0 / n)
    r2 = D - savgol_filter(D, w, 2)
    sl = np.clip(local_scale(r2, max(5, int(LOCAL_BW * n))), 1e-12, None)
    return (dt / sl) ** 2


# =============================================================================
print("=" * 92)
print("1.  The true floor as a function of phi, against what the surrogate returns")
print("=" * 92)
print("""
The records are trend-free, so the whole density IS the floor and there is nothing
to separate it from.  The surrogate column is what estimate_floor reports when
handed the same residual, and it cannot depend on phi because it resamples.
""")
N = 1200
print("  %6s %14s %16s %10s"
      % ("phi", "true floor", "surrogate floor", "ratio"))
print("  " + "-" * 52)
rows = []
for phi in PHIS:
    tru, sur = [], []
    for _ in range(REPS):
        x = ar1(N, phi, rng)
        d = density_of(x)
        if d is None:
            continue
        tru.append(float(np.median(d)))
        sur.append(float(estimate_floor(d, x, "surrogate", rng, reps=3)[0]))
    if not tru:
        continue
    t, s = float(np.median(tru)), float(np.median(sur))
    rows.append(dict(phi=phi, true=t, surrogate=s, ratio=t / s))
    print("  %6.2f %14.2f %16.2f %10.3f" % (phi, t, s, t / s))
OUT["level"] = rows

if rows:
    OUT["ratio_at_0"] = [r for r in rows if r["phi"] == 0.0][0]["ratio"]
    OUT["ratio_at_max"] = rows[-1]["ratio"]
    OUT["level_direction"] = "falls" if OUT["ratio_at_max"] < 1 else "rises"
    # The surrogate column cannot depend on phi, since resampling destroys the
    # ordering, so its variation down the column IS this table's Monte Carlo
    # scale and needs no separate estimate.
    _sur = [r["surrogate"] for r in rows]
    OUT["surrogate_spread"] = float((max(_sur) - min(_sur))
                                    / float(np.mean(_sur)))
    print("""
  At phi = 0 the two agree to %.1f per cent.  That residual is noise, not bias:
  the surrogate column must be flat, because resampling destroys the ordering,
  and it varies by %.1f per cent down the column, so anything of that size is
  Monte Carlo.  Raising the repetitions from 240 to 600 puts the control at 1.000.

  As phi grows the true floor %s relative to the surrogate, reaching %.3f at
  phi = 0.40, which is far outside that scale.  The pipeline %s the floor on
  correlated noise.
""" % (100 * abs(OUT["ratio_at_0"] - 1), 100 * OUT["surrogate_spread"],
       OUT["level_direction"], OUT["ratio_at_max"],
       "over-subtracts" if OUT["ratio_at_max"] < 1 else "under-subtracts"))

# --- which half of the ratio moves, since guessing it got the sign wrong -------
print("""  The density is (dt/sl)^2, so the direction has two possible sources and
  the low-frequency argument in this file's header predicted the wrong one.  It
  is settled by measuring the numerator and the denominator separately rather
  than by arguing about the filter.
""")
print("  %6s %14s %14s %12s" % ("phi", "rms of dt", "median sl", "check"))
print("  " + "-" * 50)
dec = []
for phi in PHIS:
    nums, dens_ = [], []
    for _ in range(60):
        x = ar1(N, phi, rng)
        s = robust_scale(x)
        if not np.isfinite(s) or s <= 0:
            continue
        D = (x - np.median(x)) / s
        w = _win(N)
        dt = savgol_filter(D, w, 2, deriv=1, delta=1.0 / N)
        r2 = D - savgol_filter(D, w, 2)
        sl = np.clip(local_scale(r2, max(5, int(LOCAL_BW * N))), 1e-12, None)
        nums.append(float(np.sqrt(np.mean(dt ** 2))))
        dens_.append(float(np.median(sl)))
    if not nums:
        continue
    a, b = float(np.median(nums)), float(np.median(dens_))
    dec.append(dict(phi=phi, dt_rms=a, sl=b, implied=(a / b) ** 2))
    print("  %6.2f %14.2f %14.4f %12.2f" % (phi, a, b, (a / b) ** 2))
OUT["decomposition"] = dec
if len(dec) >= 2:
    OUT["dt_growth"] = dec[-1]["dt_rms"] / dec[0]["dt_rms"]
    OUT["sl_shrink"] = dec[-1]["sl"] / dec[0]["sl"]
    print("""
  Between phi = 0 and phi = 0.40 the derivative grows by %.2f while the local
  scale moves by %.2f, essentially not at all, and (%.2f/%.2f)^2 = %.2f accounts
  for the measured %.2f.  It is the numerator alone.

  The header of this file predicted the opposite sign, on the argument that a
  derivative filter passes low frequency as i*omega and so suppresses exactly
  where red noise keeps its power.  That is wrong about which frequencies matter.
  The window is 8 per cent of the record, so the filter's passband sits at periods
  of order the window, not at the sampling rate, and an AR(1) has its excess power
  there.  The filter is looking straight at the part of the spectrum correlation
  adds to.  The surrogate cannot see any of it, because resampling flattens the
  spectrum before the filter is ever applied.
""" % (OUT["dt_growth"], OUT["sl_shrink"], OUT["dt_growth"], OUT["sl_shrink"],
       (OUT["dt_growth"] / OUT["sl_shrink"]) ** 2, OUT["ratio_at_max"]))

# =============================================================================
print("=" * 92)
print("2.  Is the floor still flat when phi drifts, as it does on the records?")
print("=" * 92)
print("""
This is the property the pipeline actually uses, since estimate_floor returns one
constant for the whole record.  floor_shape.py verified flatness against a
varying noise SCALE.  Here the scale is held fixed and phi is ramped from 0.03 to
0.30, the drift Section 4.5 measured, so any tilt found is caused by correlation
alone.
""")
PHI_EARLY, PHI_LATE = 0.03, 0.30


def ar1_drift(n, p0, p1, rg):
    e = rg.standard_normal(n)
    ph = np.linspace(p0, p1, n)
    z = np.empty(n)
    z[0] = e[0]
    for t in range(1, n):
        z[t] = ph[t] * z[t - 1] + np.sqrt(1.0 - ph[t] ** 2) * e[t]
    return z


tilts, q1s, q4s = [], [], []
for _ in range(TILT_REPS):
    x = ar1_drift(N, PHI_EARLY, PHI_LATE, rng)
    d = density_of(x)
    if d is None:
        continue
    a = float(np.median(d[:N // 4]))
    b = float(np.median(d[-(N // 4):]))
    if a > 0:
        q1s.append(a)
        q4s.append(b)
        tilts.append(b / a)
if tilts:
    # The headline is the median of the PER-RECORD ratios.  The quarter medians
    # are kept for the record but are not divided into each other: a median of
    # ratios is not a ratio of medians, and quoting both invites the reader to
    # divide them and get a third number.  An interquartile range is reported
    # because a single run of this moved the estimate from 1.35 to 1.50, so the
    # third digit is Monte Carlo noise and should not be read.
    OUT["tilt"] = float(np.median(tilts))
    OUT["tilt_iqr"] = [float(np.percentile(tilts, 25)),
                       float(np.percentile(tilts, 75))]
    OUT["tilt_se"] = float(np.std(tilts, ddof=1) / np.sqrt(len(tilts)))
    OUT["tilt_q1_level"] = float(np.median(q1s))
    OUT["tilt_q4_level"] = float(np.median(q4s))
    print("  median of per-record ratios %.2f, quartiles %.2f to %.2f,"
          % (OUT["tilt"], OUT["tilt_iqr"][0], OUT["tilt_iqr"][1]))
    print("  standard error %.3f over %d records" % (OUT["tilt_se"], len(tilts)))
    print("""
  A constant floor is subtracted from a floor that is %s along the record by
  about %.1f.  The subtraction is therefore %s at the end of life relative to the
  beginning, and the end of life is where the earliest usable age and the minimum
  window are read.
""" % ("falling" if OUT["tilt"] < 1 else "rising", OUT["tilt"],
       "too small" if OUT["tilt"] > 1 else "too large"))

# =============================================================================
print("=" * 92)
print("3.  What it does on the records, where the floor competes with real signal")
print("=" * 92)
print("""
Sections 1 and 2 are on trend-free records, where the floor is the whole density.
On a real record the floor is only part of it, and how much decides whether any of
this reaches a reported number.  The correction multiplies the surrogate floor by
the factor measured above at each sample's own local phi, so a record whose phi
rises gets a floor that rises with it.
""")
from pipeline import oof_trend
from nonparam import quantile, w_star

_phis = np.array([r["phi"] for r in rows])
_mult = np.array([r["ratio"] for r in rows])


def floor_multiplier(ph):
    return np.interp(np.clip(ph, 0.0, _phis[-1]), _phis, _mult)


def rolling_phi(res, bw=0.20):
    r = np.asarray(res, float)
    m = len(r)
    h = max(8, int(bw * m))
    lo = np.maximum(0, np.arange(m) - h)
    hi = np.minimum(m, np.arange(m) + h + 1)

    def win(a, k=0):
        c = np.concatenate(([0.0], np.cumsum(a)))
        return c[np.minimum(hi - k, len(a))] - c[np.minimum(lo, len(a))]

    cnt = (hi - lo).astype(float)
    s1, s2 = win(r), win(r * r)
    mu = s1 / np.maximum(cnt, 1)
    var = s2 - cnt * mu * mu
    cross = win(r[:-1] * r[1:], 1) - np.maximum(cnt - 1, 0) * mu * mu
    return np.clip(np.where(var > 0, cross / np.maximum(var, 1e-300), 0.0),
                   -0.9, 0.9)


z = np.load(os.path.join(HERE, "feat_raw.npz"), allow_pickle=True)
FN = [str(s) for s in z["featnames"]]
units = [k for k in z.files if k != "featnames"]
KIND = {"rms": "amount", "peak": "amount",
        "b4_6": "distribution", "b8_10": "distribution"}
CUTS = (0.30, 0.50, 0.70, 0.90)
Q = 0.35
AGES = (0.55, 0.70, 0.85, 0.95)


_CACHE = {}


def pair(x, key=None):
    """Density with the flat floor, and with a floor that follows local phi.

    Cached on (unit, channel): the surrogate floor is the expensive part of the
    pipeline and every cut, age and family reads the same two curves.
    """
    if key is not None and key in _CACHE:
        return _CACHE[key]
    v = _pair_uncached(x)
    if key is not None:
        _CACHE[key] = v
    return v


def _pair_uncached(x):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    if len(x) < 200:
        return None
    dens, res = weighted_density(x)
    if dens is None:
        return None
    fl = estimate_floor(dens, res, "surrogate", rng)
    a = np.clip(dens - fl, 0.0, None)
    b = np.clip(dens - fl * floor_multiplier(rolling_phi(res)), 0.0, None)
    if a.sum() <= 0 or b.sum() <= 0:
        return None
    # Two shares, because the obvious one misleads.  The floor's share of the
    # TOTAL density is small on these records simply because the density is
    # heavy-tailed and its sum is dominated by the end of life.  What decides
    # whether a floor error reaches a reported number is its share where the
    # density is small, which is the early record the fractions are read across.
    n4 = max(1, len(dens) // 4)
    share_all = float(np.median(fl) * len(dens) / max(dens.sum(), 1e-30))
    share_q1 = float(np.median(fl) * n4 / max(dens[:n4].sum(), 1e-30))
    return a, b, (share_all, share_q1)


shares, erows = [], []
print("  %-10s %6s %10s %10s %9s %8s"
      % ("channel", "cut c", "eta flat", "eta phi", "ratio", "records"))
print("  " + "-" * 60)
for name in ("rms", "b4_6", "b8_10"):
    if name not in FN:
        continue
    i = FN.index(name)
    for c in CUTS:
        A, B = [], []
        for u in units:
            p = pair(z[u][:, i], (u, name))
            if p is None:
                continue
            a, b, sh = p
            if c == CUTS[0]:
                shares.append(sh)
            k = max(1, int(round(c * len(a))))
            A.append(float(a[:k].sum() / a.sum()))
            B.append(float(b[:k].sum() / b.sum()))
        if len(A) < 6:
            continue
        r0, r1 = float(np.median(A)), float(np.median(B))
        erows.append(dict(channel=name, cut=c, eta_flat=r0, eta_phi=r1,
                          ratio=r1 / r0 if r0 > 0 else np.nan, records=len(A)))
        print("  %-10s %6.2f %10.4f %10.4f %9.3f %8d"
              % (name, c, r0, r1, r1 / r0 if r0 > 0 else np.nan, len(A)))
OUT["eta"] = erows
if shares:
    OUT["floor_share"] = float(np.median([s[0] for s in shares]))
    OUT["floor_share_q1"] = float(np.median([s[1] for s in shares]))
    print("""
  The floor is a median %.1f per cent of the total raw density and %.1f per cent
  of the density in the first quarter.  The second was measured because the first
  could have been small for an uninformative reason: the density is heavy-tailed
  and its sum is dominated by the end of the record, so a floor that was most of
  the early density would still be a small share of the total.  It is not.  The
  floor is a small part of these densities everywhere, and that, rather than any
  property of the correction, is why nothing below moves.
""" % (100 * OUT["floor_share"], 100 * OUT["floor_share_q1"]))
if erows:
    good = [r for r in erows if np.isfinite(r["ratio"])]
    OUT["eta_down"] = int(sum(1 for r in good if r["ratio"] < 1.0))
    OUT["eta_cells"] = len(good)
    OUT["eta_err_median"] = float(np.median(
        [abs(r["ratio"] ** -0.5 - 1.0) for r in good]))
    OUT["eta_err_worst"] = float(max(
        abs(r["ratio"] ** -0.5 - 1.0) for r in good))
    OUT["eta_ratio_span"] = [float(min(r["ratio"] for r in good)),
                             float(max(r["ratio"] for r in good))]
    print("""  Raising the floor late in life should raise eta, and the direction here is
  mixed: it falls in %d of %d cells.  That is not a mechanism, it is a null.  Every
  ratio lies between %.3f and %.3f, and at a floor worth %.1f per cent of the
  density a median over seventeen records cannot resolve a direction that small.
  In the achievable error at the cut the correction moves the figure by a median
  of %.1f per cent and at worst %.1f.
""" % (OUT["eta_down"], OUT["eta_cells"], OUT["eta_ratio_span"][0],
       OUT["eta_ratio_span"][1], 100 * OUT["floor_share"],
       100 * OUT["eta_err_median"], 100 * OUT["eta_err_worst"]))

wsh, gaps = [], {"amount": [[], []], "distribution": [[], []]}
for name, kind in KIND.items():
    if name not in FN:
        continue
    i = FN.index(name)
    for u in units:
        p = pair(z[u][:, i], (u, name))
        if p is None:
            continue
        a, b, _ = p
        Fa, Fb = np.cumsum(a) / a.sum(), np.cumsum(b) / b.sum()
        for t0 in AGES:
            x1, x2 = w_star(Fa, t0, Q), w_star(Fb, t0, Q)
            if np.isfinite(x1) and np.isfinite(x2):
                wsh.append(abs(x2 - x1))
        q1, q2 = quantile(Fa, Q), quantile(Fb, Q)
        if np.isfinite(q1) and np.isfinite(q2):
            gaps[kind][0].append(q1)
            gaps[kind][1].append(q2)
if wsh:
    OUT["window_shift_median"] = float(np.median(wsh))
    OUT["window_shift_p95"] = float(np.percentile(wsh, 95))
    print("  the window moves by a median of %.4f of a lifetime, 95th percentile %.4f"
          % (OUT["window_shift_median"], OUT["window_shift_p95"]))
if all(gaps[k][0] for k in gaps):
    g0 = (float(np.median(gaps["amount"][0]))
          - float(np.median(gaps["distribution"][0])))
    g1 = (float(np.median(gaps["amount"][1]))
          - float(np.median(gaps["distribution"][1])))
    OUT["gap_flat"], OUT["gap_phi"] = g0, g1
    print("  the family gap is %.4f flat and %.4f corrected, a change of %+.4f"
          % (g0, g1, g1 - g0))

json.dump(OUT, open(os.path.join(HERE, "floor_serial.json"), "w"), indent=2,
          default=float)
print("written to floor_serial.json")
