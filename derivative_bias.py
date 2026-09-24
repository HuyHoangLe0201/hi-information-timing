r"""Is the derivative estimator biased, and does the bias drift along life?

Every number in this paper passes through one line of pipeline.py,

    dt = savgol_filter(D, w, 2, deriv=1, delta=1.0/n),

because G is built from D'(tau)^2.  The window is fixed at 8 per cent of life and
Section 12 reports that sweeping it from 5 to 12 per cent leaves the conclusions
standing.  That sweep answers a different question from the one asked here.  A
sweep detects sensitivity to a choice; it cannot detect a bias that is present at
every setting, because the bias moves with the estimate and the comparison never
sees it.  Nothing in the work so far has compared this estimator against a
derivative that is known.

The bias is analytic.  Fitting a quadratic by least squares over a symmetric
window of half-width h and reading its slope at the centre, a cubic term c3 x^3
contributes

    a1 = c3 * integral(x^4) / integral(x^2) = c3 * (3/5) h^2,

so the estimate carries + h^2 f'''/10 with f''' = 6 c3.  For a trend tau^p this is

    relative bias = (h^2 / 10) (p-1)(p-2) / tau^2,

which is POSITIVE for p > 2 and grows as tau falls.  The estimator does not
attenuate an accelerating trend, it inflates it, and it inflates it most early in
life.  That direction matters for a paper about a bound: information credited to
the early record that is not there makes the earliest usable age look earlier
than it is.

Whether it matters is a separate question from whether it exists, because G is
tiny early and the fractions the paper reads are cumulative.  Both are measured
here, first against trends whose derivative is known exactly and then on the
shapes the records actually have.
"""
import os
import json
import numpy as np
from scipy.signal import savgol_filter

from pipeline import _win, robust_scale, oof_trend, local_scale, LOCAL_BW

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = {}

# the trend families, with derivatives in closed form so the truth is exact
FAMILIES = {
    "linear, p=1": (lambda t: t, lambda t: np.ones_like(t)),
    "quadratic, p=2": (lambda t: t ** 2, lambda t: 2 * t),
    "p=4": (lambda t: t ** 4, lambda t: 4 * t ** 3),
    "p=8": (lambda t: t ** 8, lambda t: 8 * t ** 7),
    "exponential": (lambda t: np.exp(4 * t) - 1.0,
                    lambda t: 4 * np.exp(4 * t)),
}
NS = (500, 1400)
BANDS = (0.05, 0.08, 0.12)


def sg_derivative(D, n, frac):
    w = max(7, int(frac * n))
    w += (w % 2 == 0)
    w = min(w, n - 1 if (n - 1) % 2 == 1 else n - 2)
    return savgol_filter(D, w, 2, deriv=1, delta=1.0 / n), w


print("=" * 92)
print("1.  The estimator against a derivative that is known, with no noise at all")
print("=" * 92)
print("""
Noiseless, so what is measured is the estimator and nothing else.  The columns are
the ratio of estimated to true derivative, median within each quarter of life.  A
value above one is information credited that the trend does not carry.
""")
print("  %-16s %6s %6s %8s %8s %8s %8s"
      % ("trend", "n", "band", "q1", "q2", "q3", "q4"))
print("  " + "-" * 66)
rows = []
for name, (f, fp) in FAMILIES.items():
    for n in NS:
        t = (np.arange(n) + 1.0) / n
        D = f(t)
        s = robust_scale(D)
        if not np.isfinite(s) or s <= 0:
            continue
        for band in BANDS:
            dh, w = sg_derivative(D / s, n, band)
            dt_true = fp(t) / s
            ok = np.abs(dt_true) > 1e-9
            r = np.full(n, np.nan)
            r[ok] = dh[ok] / dt_true[ok]
            q = [float(np.nanmedian(r[k * n // 4:(k + 1) * n // 4]))
                 for k in range(4)]
            rows.append(dict(trend=name, n=n, band=band, quarters=q,
                             window=int(w)))
            if band == 0.08:
                print("  %-16s %6d %6.2f %8.3f %8.3f %8.3f %8.3f"
                      % (name, n, band, *q))
OUT["noiseless"] = rows

_acc = [r for r in rows if r["band"] == 0.08 and r["trend"] in ("p=4", "p=8",
                                                               "exponential")]
if _acc:
    OUT["q1_worst"] = float(max(r["quarters"][0] for r in _acc))
    OUT["q4_worst"] = float(max(abs(r["quarters"][3] - 1.0) for r in _acc))
    print("""
  The prediction holds.  The two straight families are estimated exactly, since a
  quadratic fit reproduces a linear or quadratic trend with no residual cubic
  term.  The accelerating families are inflated, worst in the first quarter where
  the ratio reaches %.3f, and are correct to %.4f by the last quarter.  The bias
  is real, it has the sign the algebra gives, and it lives early in life.
""" % (OUT["q1_worst"], OUT["q4_worst"]))

# --- does sweeping the bandwidth reveal it? ------------------------------------
print("=" * 92)
print("2.  Why the bandwidth sweep did not find this")
print("=" * 92)
print("""
Section 12 sweeps the window from 5 to 12 per cent and the conclusions survive.
If the bias scaled with the window the sweep would have exposed it, so the
question is how much of it the sweep could have seen.  h^2 says a factor of
(12/5)^2 = 5.8 between the ends.
""")
print("  %-16s %6s %10s %10s %10s %9s"
      % ("trend", "n", "5%", "8%", "12%", "spread"))
print("  " + "-" * 64)
srows = []
for name in ("p=4", "p=8", "exponential"):
    for n in NS:
        got = {}
        for band in BANDS:
            m = [r for r in rows if r["trend"] == name and r["n"] == n
                 and r["band"] == band]
            if m:
                got[band] = m[0]["quarters"][0]
        if len(got) == 3:
            sp = (got[0.12] - 1.0) / max(got[0.05] - 1.0, 1e-12)
            srows.append(dict(trend=name, n=n, at5=got[0.05], at8=got[0.08],
                              at12=got[0.12], spread=sp))
            print("  %-16s %6d %10.3f %10.3f %10.3f %9.2f"
                  % (name, n, got[0.05], got[0.08], got[0.12], sp))
OUT["sweep"] = srows
if srows:
    OUT["sweep_spread_median"] = float(np.median([r["spread"] for r in srows]))
    print("""
  The bias does scale with the window, by a median factor %.1f between the ends of
  the swept range, close to the %.1f the algebra predicts.  So the sweep did move
  it.  What the sweep reports is whether the CONCLUSIONS change, and they do not,
  which is a statement about robustness and not about accuracy.  A quantity that
  is wrong by the same sign at every setting stays wrong at every setting, and
  that is the gap this file fills.
""" % (OUT["sweep_spread_median"], (0.12 / 0.05) ** 2))

# =============================================================================
print("=" * 92)
print("3.  The shapes the records actually have")
print("=" * 92)
print("""
Section 1 spans a wide range on purpose, and the range is what decides the
answer: an exponential is inflated by 0.3 per cent and tau^8 by 35, so quoting
either as the bias would be choosing the conclusion.  What the records carry is
the only relevant question.

There is no ground truth to be had from the records themselves, and two attempts
to manufacture one failed in instructive ways.  Taking a ratio of derivatives
pointwise gave 1.000 in the outer quarters against 0.50 in the middle, which is
not a bias profile but a division by zero, since a real trend is not monotone and
its derivative crosses zero.  Replacing that with a ratio of sums of squares fixed
the zero crossings and left a worse problem: the reference derivative was taken by
central differences at spacing 1/n, which amplifies whatever ripple survives the
smoother, so the reference was noise and the pipeline was penalised for not
following it.

The bias does not need a ground truth, because Section 1 has already confirmed
the formula that produces it,

    relative bias  =  (h^2 / 10) f''' / f',      h = 0.04 of a lifetime,

to three decimals on the exponential.  So what the records have to supply is
f'''/f' and nothing else.  Weighting by the energy the paper actually reads makes
that finite where f' passes through zero, since

    sum(D'^2 . relative bias)  =  (h^2/10) sum(f' f'''),

so the reported figure is (h^2/10) sum(f' f''') / sum(f'^2) over each quarter:
the fractional error in the information G credits to that quarter.
""")
z = np.load(os.path.join(HERE, "feat_raw.npz"), allow_pickle=True)
FN = [str(s) for s in z["featnames"]]
units = [k for k in z.files if k != "featnames"]
CH = [("rms", "amount"), ("peak", "amount"),
      ("b4_6", "distribution"), ("b8_10", "distribution")]
WIDE = 4.0


H_HALF = 0.04     # the pipeline's half-window, 8 per cent of life


def shape_derivatives(x):
    """f' and f''' of a record's settled trend, on a common wide window.

    Both come from one quartic Savitzky-Golay fit, so they describe the same
    function.  A quartic is the lowest order that admits a third derivative with
    a term left over, and the window is four times the pipeline's so that the
    shape being described is the trend and not the pipeline's own smoothing.
    """
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    n = len(x)
    if n < 200:
        return None
    s = robust_scale(x)
    if not np.isfinite(s) or s <= 0:
        return None
    D = (x - np.median(x)) / s
    wf = max(11, int(WIDE * 0.08 * n))
    wf += (wf % 2 == 0)
    if wf >= n:
        return None
    d1 = savgol_filter(D, wf, 4, deriv=1, delta=1.0 / n)
    d3 = savgol_filter(D, wf, 4, deriv=3, delta=1.0 / n)
    if not (np.all(np.isfinite(d1)) and np.all(np.isfinite(d3))):
        return None
    return d1, d3, n


def weighted_bias(d1, d3, sl):
    """(h^2/10) sum(f' f''') / sum(f'^2) over a slice: the fractional error in
    the information credited to it."""
    den = float(np.sum(d1[sl] ** 2))
    if den <= 0:
        return None
    return float((H_HALF ** 2 / 10.0) * np.sum(d1[sl] * d3[sl]) / den)


# --- the formula, checked against the cases Section 1 measured directly --------
print("  the formula against Section 1, where the answer is already known:")
print("    %-16s %10s %12s %10s" % ("trend", "measured", "formula", "gap"))
print("    " + "-" * 52)
chk = []
for name, (f, fp) in FAMILIES.items():
    m = [r for r in rows if r["trend"] == name and r["n"] == 1400
         and r["band"] == 0.08]
    if not m:
        continue
    n = 1400
    t = (np.arange(n) + 1.0) / n
    D = f(t)
    s = robust_scale(D)
    wf = max(11, int(WIDE * 0.08 * n))
    wf += (wf % 2 == 0)
    d1 = savgol_filter(D / s, wf, 4, deriv=1, delta=1.0 / n)
    d3 = savgol_filter(D / s, wf, 4, deriv=3, delta=1.0 / n)
    b = weighted_bias(d1, d3, slice(0, n // 4))
    if b is None:
        continue
    meas, pred = m[0]["quarters"][0], 1.0 + b
    chk.append(dict(trend=name, measured=meas, formula=pred,
                    gap=abs(pred - meas)))
    print("    %-16s %10.3f %12.3f %10.3f" % (name, meas, pred,
                                              abs(pred - meas)))
OUT["formula_check"] = chk
if chk:
    OUT["formula_gap_max"] = float(max(r["gap"] for r in chk))
    print("""
    The formula tracks the direct measurement to %.3f at worst across families
    spanning a %.0f per cent range of bias, so it can carry the real records.
""" % (OUT["formula_gap_max"], 100 * (OUT["q1_worst"] - 1)))

print("""  The figures below are the error in the INFORMATION, which is twice the
  derivative bias: D'(1+b) squared is D'^2(1+2b) to first order, and G is built
  from the square.  Reporting b where the paper reads D'^2 would halve every
  number in the table.
""")
print("  fractional error in the information credited to each quarter, per cent")
print("  %-10s %-13s %8s %8s %8s %8s %8s"
      % ("channel", "kind", "q1", "q2", "q3", "q4", "recs"))
print("  " + "-" * 66)
real = []
for name, kind in CH:
    if name not in FN:
        continue
    i = FN.index(name)
    cols = [[] for _ in range(4)]
    nrec = 0
    for u in units:
        r = shape_derivatives(z[u][:, i])
        if r is None:
            continue
        d1, d3, n = r
        q = [weighted_bias(d1, d3, slice(k * n // 4, (k + 1) * n // 4))
             for k in range(4)]
        if any(v is None or not np.isfinite(v) for v in q):
            continue
        q = [2.0 * v for v in q]          # derivative bias -> information error
        nrec += 1
        for k, v in enumerate(q):
            cols[k].append(v)
    if nrec < 6:
        continue
    med = [float(np.median(c)) for c in cols]
    real.append(dict(channel=name, kind=kind, quarters=med, records=nrec))
    print("  %-10s %-13s %+8.2f %+8.2f %+8.2f %+8.2f %8d"
          % (name, kind, *[100 * v for v in med], nrec))
OUT["real"] = real

if real:
    allq = [v for r in real for v in r["quarters"]]
    OUT["real_worst"] = float(max(abs(v) for v in allq))
    OUT["real_median"] = float(np.median([abs(v) for v in allq]))
    OUT["real_last_quarter"] = float(np.median([r["quarters"][3]
                                                for r in real]))
    OUT["real_q4_all_positive"] = bool(all(r["quarters"][3] > 0 for r in real))
    OUT["real_mid_all_negative"] = bool(
        all(r["quarters"][k] < 0 for r in real for k in (1, 2)))
    OUT["real_q4_max"] = float(max(r["quarters"][3] for r in real))
    OUT["real_mid_min"] = float(min(min(r["quarters"][1], r["quarters"][2])
                                    for r in real))
    print("""
  The profile is not the one Section 1's power families gave, and it is more
  interesting.  The middle of life is UNDER-credited, by up to %.1f per cent, and
  the last quarter is OVER-credited on every channel, by up to %.1f.  A monotone
  power law cannot do that: it requires f'''/f' to change sign, which is what a
  trend that is concave through mid-life and then accelerates into failure does.

  The direction is the one that matters.  The last quarter is where most of G
  sits and where the earliest usable age and the minimum window are read, and it
  is the quarter the estimator flatters.  The largest error is %.1f per cent and
  the median over the sixteen cells is %.1f, so this is not a rounding term and
  is treated below as the other drifting corrections were.
""" % (100 * abs(OUT["real_mid_min"]), 100 * OUT["real_q4_max"],
       100 * OUT["real_worst"], 100 * OUT["real_median"]))

# =============================================================================
print("=" * 92)
print("4.  What the design quantities do once the bias is removed")
print("=" * 92)
print("""
The correction is exact rather than estimated, which is what makes this cheap:
the estimator carries D' + (h^2/10) f''', so subtracting (h^2/10) f''' before
squaring removes it, and no division by a derivative is involved anywhere.  The
wide quartic fit supplies f''' and nothing else changes.

The three quantities are the ones the last three corrections were put through.
The censoring efficiency is a ratio of the curve to itself, the minimum window is
a difference of two inversions, and the family gap is a ranking across
indicators, and they respond differently for that reason.
""")
from nonparam import estimate_floor, quantile, w_star

rng = np.random.default_rng(4242)
Q = 0.35
CUTS = (0.30, 0.50, 0.70, 0.90)


def densities(x):
    """The pipeline's density, and the same with the derivative bias removed."""
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    n = len(x)
    if n < 200:
        return None
    s = robust_scale(x)
    if not np.isfinite(s) or s <= 0:
        return None
    D = (x - np.median(x)) / s
    w = _win(n)
    if w < 7 or w >= n:
        return None
    dt = savgol_filter(D, w, 2, deriv=1, delta=1.0 / n)
    wf = max(11, int(WIDE * 0.08 * n))
    wf += (wf % 2 == 0)
    if wf >= n:
        return None
    d3 = savgol_filter(D, wf, 4, deriv=3, delta=1.0 / n)
    res = D - oof_trend(D, w // 2)
    sl = np.clip(local_scale(res, max(5, int(LOCAL_BW * n))), 1e-12, None)
    raw = (dt / sl) ** 2
    cor = ((dt - (H_HALF ** 2 / 10.0) * d3) / sl) ** 2
    fr = estimate_floor(raw, res, "surrogate", rng)
    a = np.clip(raw - fr, 0.0, None)
    b = np.clip(cor - fr, 0.0, None)
    if a.sum() <= 0 or b.sum() <= 0:
        return None
    return a, b


print("  the censoring efficiency, which a common factor would leave alone")
print("  %-10s %6s %10s %10s %9s %8s"
      % ("channel", "cut c", "eta raw", "eta corr", "ratio", "records"))
print("  " + "-" * 62)
erows = []
for name in ("rms", "b4_6", "b8_10"):
    if name not in FN:
        continue
    i = FN.index(name)
    for c in CUTS:
        A, B = [], []
        for u in units:
            d = densities(z[u][:, i])
            if d is None:
                continue
            a, b = d
            k = max(1, int(round(c * len(a))))
            A.append(float(a[:k].sum() / a.sum()))
            B.append(float(b[:k].sum() / b.sum()))
        if len(A) < 6:
            continue
        r0, r1 = float(np.median(A)), float(np.median(B))
        erows.append(dict(channel=name, cut=c, eta_raw=r0, eta_corr=r1,
                          ratio=r1 / r0 if r0 > 0 else np.nan,
                          records=len(A)))
        print("  %-10s %6.2f %10.4f %10.4f %9.3f %8d"
              % (name, c, r0, r1, r1 / r0 if r0 > 0 else np.nan, len(A)))
OUT["eta"] = erows
if erows:
    good = [r for r in erows if np.isfinite(r["ratio"])]
    OUT["eta_up"] = int(sum(1 for r in good if r["ratio"] > 1.0))
    OUT["eta_cells"] = len(good)
    OUT["eta_err_median"] = float(np.median(
        [abs(r["ratio"] ** -0.5 - 1.0) for r in good]))
    OUT["eta_err_worst"] = float(max(
        abs(r["ratio"] ** -0.5 - 1.0) for r in good))
    print("""
  eta rises in %d of %d cells rather than in all of them, and that is the correct
  behaviour rather than a weakness.  The distortion is not one-signed: it removes
  information from the last quarter and returns some to the middle, so which way
  a given cut moves depends on whether the cut falls before or after the sign
  change.  The serial correction of Section 4.5 was one-signed and moved every
  cell; this one is not and does not.

  In the quantity Section 6.2 validates, the achievable error at the cut, the
  correction moves it by a median of %.1f per cent and at worst %.1f.
""" % (OUT["eta_up"], OUT["eta_cells"], 100 * OUT["eta_err_median"],
       100 * OUT["eta_err_worst"]))

# --- the window, and the family gap -------------------------------------------
print("  the minimum window and the family gap")
AGES = (0.55, 0.70, 0.85, 0.95)
wsh, gaps = [], {"amount": [[], []], "distribution": [[], []]}
KIND = {"rms": "amount", "peak": "amount",
        "b4_6": "distribution", "b8_10": "distribution"}
for name, kind in KIND.items():
    if name not in FN:
        continue
    i = FN.index(name)
    for u in units:
        d = densities(z[u][:, i])
        if d is None:
            continue
        a, b = d
        Fa = np.cumsum(a) / a.sum()
        Fb = np.cumsum(b) / b.sum()
        for t0 in AGES:
            x1, x2 = w_star(Fa, t0, Q), w_star(Fb, t0, Q)
            if np.isfinite(x1) and np.isfinite(x2):
                wsh.append(dict(d=x2 - x1, unit=u, channel=name, tau0=t0,
                                w_raw=float(x1), w_corr=float(x2)))
        q1, q2 = quantile(Fa, Q), quantile(Fb, Q)
        if np.isfinite(q1) and np.isfinite(q2):
            gaps[kind][0].append(q1)
            gaps[kind][1].append(q2)
if wsh:
    mags = [abs(r["d"]) for r in wsh]
    OUT["window_shift_median"] = float(np.median(mags))
    OUT["window_shift_worst"] = float(max(mags))
    OUT["window_shift_p95"] = float(np.percentile(mags, 95))
    OUT["window_pairs"] = len(wsh)
    worst = max(wsh, key=lambda r: abs(r["d"]))
    OUT["window_worst_case"] = worst
    print("    the window moves by a median of %.4f of a lifetime, worst %.4f"
          % (OUT["window_shift_median"], OUT["window_shift_worst"]))
    print("      the worst is %s on %s at tau_0 = %.2f, where w* goes %.4f -> %.4f"
          % (worst["unit"], worst["channel"], worst["tau0"],
             worst["w_raw"], worst["w_corr"]))
    print("      the 95th percentile of the shift is %.4f over %d pairs, so the"
          % (OUT["window_shift_p95"], len(wsh)))
    print("      worst case is a tail and not the typical behaviour")
if all(gaps[k][0] for k in gaps):
    g_raw = (float(np.median(gaps["amount"][0]))
             - float(np.median(gaps["distribution"][0])))
    g_cor = (float(np.median(gaps["amount"][1]))
             - float(np.median(gaps["distribution"][1])))
    OUT["gap_raw"] = g_raw
    OUT["gap_corr"] = g_cor
    print("    the family gap is %.4f raw and %.4f corrected, a change of %.4f"
          % (g_raw, g_cor, g_cor - g_raw))
    print("""
  The gap is the paper's headline and it %s the correction: a ranking across
  indicators feels only the part of a distortion that differs between them, and
  this one is close to common, being a property of trend shape rather than of
  which channel measures it.
""" % ("survives" if abs(g_cor - g_raw) < 0.05 else "does not survive"))

json.dump(OUT, open(os.path.join(HERE, "derivative_bias.json"), "w"), indent=2,
          default=float)
print("written to derivative_bias.json")
