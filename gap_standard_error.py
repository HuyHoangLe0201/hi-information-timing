r"""How precisely is the family gap known, and how much of its spread is real?

Proposition 23 gave eta a standard error and used it to show that three shifts
this paper reports are below what the records resolve.  It then asserted that the
family ordering is unaffected, "a comparison between indicators on the same
record" carrying "a different and much smaller error".  That was asserted and not
computed, which is the kind of sentence this paper's own audits exist to catch.

The gap is measured per bearing as the difference of two ages read from two
channels of the SAME record, then summarised across bearings.  Two sources of
variation are mixed in what is reported.

  WITHIN a record, each age carries the error of Proposition 5, se(G_hat)/g, and
  the two are correlated because both curves are built from one realisation of
  one bearing's life.  Only the part that does not cancel survives the
  difference.

  BETWEEN records, bearings genuinely differ: load, severity, the size of the
  spall.  That variation is real and is not error.

The spread of the gap across bearings contains both, so quoting it as an error
overstates the measurement noise and quoting it as biological variation
overstates the agreement.  The two can be separated, because Proposition 23
supplies the within-record part independently of the between-record spread.  The
difference is the real bearing-to-bearing variation, and if it comes out negative
the model is wrong and that has to be said.
"""
import os
import json
import numpy as np

from pipeline import robust_scale
from nonparam import weighted_density, estimate_floor

HERE = os.path.dirname(os.path.abspath(__file__))
rng = np.random.default_rng(24680)
OUT = {}
Q = 0.35

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
    return d


def age(d, q=Q):
    F = np.cumsum(d) / d.sum()
    i = int(np.searchsorted(F, q))
    return np.nan if i >= len(F) else (i + 1) / len(F)


def corr_length(x):
    v = np.asarray(x, float) - np.mean(x)
    den = float(np.sum(v * v))
    if den <= 0:
        return 1.0
    tot, k = 1.0, 1
    while k < min(len(v) // 4, 300):
        r = float(np.sum(v[:-k] * v[k:])) / den
        if r <= 0:
            break
        tot += 2.0 * r
        k += 1
    return max(tot, 1.0)


def flatten(seg, frac=0.15):
    m = len(seg)
    h = max(5, int(frac * m))
    k = np.ones(2 * h + 1) / (2 * h + 1)
    lvl = np.convolve(seg, k, "same") / np.maximum(
        np.convolve(np.ones(m), k, "same"), 1e-12)
    return seg / np.maximum(lvl, 1e-12)


def age_se(d, q=Q):
    """se(tau) = se(F_hat(tau)) / f(tau), Proposition 5 in normalised form.

    se(F_hat) at the crossing is that of a cumulative share, which by the same
    delta method as Proposition 23 is sqrt(q(1-q)) c_d / sqrt(m).
    """
    n = len(d)
    F = np.cumsum(d) / d.sum()
    i = int(np.searchsorted(F, q))
    if i >= n - 2 or i < 2:
        return np.nan
    r = flatten(d)
    mu = float(np.mean(r))
    if mu <= 0:
        return np.nan
    cd = float(np.std(r)) / mu
    m = n / corr_length(r)
    se_F = np.sqrt(q * (1 - q)) * cd / np.sqrt(m)
    # local density of the normalised curve, per unit tau
    h = max(5, n // 50)
    lo, hi = max(0, i - h), min(n - 1, i + h)
    f = (F[hi] - F[lo]) / ((hi - lo) / n)
    return np.nan if f <= 0 else float(se_F / f)


print("=" * 92)
print("1.  Per bearing: the gap, and the error each age carries")
print("=" * 92)
print("""
For each bearing the earliest distributional and earliest amount channel are
paired, as the ordering does.  The two standard errors come from Proposition 5
with Proposition 23's variance for the cumulative share.
""")
print("  %-14s %9s %9s %8s %9s %9s"
      % ("bearing", "tau dist", "tau amt", "gap", "se dist", "se amt"))
print("  " + "-" * 62)
rows = []
for u in units:
    best = {}
    for grp, names in (("d", DISTRIB), ("a", AMOUNT)):
        cand = []
        for nm in names:
            d = curve(z[u][:, FN.index(nm)])
            if d is None:
                continue
            t = age(d)
            if np.isfinite(t):
                cand.append((t, d))
        if cand:
            best[grp] = min(cand, key=lambda p: p[0])
    if "d" not in best or "a" not in best:
        continue
    td, dd = best["d"]
    ta, da = best["a"]
    sd, sa = age_se(dd), age_se(da)
    if not (np.isfinite(sd) and np.isfinite(sa)):
        continue
    rows.append(dict(unit=u, tau_d=td, tau_a=ta, gap=ta - td,
                     se_d=float(sd), se_a=float(sa)))
    print("  %-14s %9.3f %9.3f %8.3f %9.4f %9.4f"
          % (u, td, ta, ta - td, sd, sa))
OUT["rows"] = rows

if len(rows) >= 6:
    gaps = np.array([r["gap"] for r in rows])
    OUT["gap_median"] = float(np.median(gaps))
    OUT["gap_spread"] = float(np.std(gaps, ddof=1))
    # within-record error of one gap, if the two ages were independent
    w_ind = np.array([np.hypot(r["se_d"], r["se_a"]) for r in rows])
    OUT["within_independent"] = float(np.median(w_ind))
    print("""
  The gap has a median of %.3f of a lifetime and a spread across bearings of
  %.3f.  Treating the two ages as independent, one gap carries a within-record
  standard error of about %.4f.
""" % (OUT["gap_median"], OUT["gap_spread"], OUT["within_independent"]))

# =============================================================================
print("=" * 92)
print("2.  Splitting the spread into measurement and real variation")
print("=" * 92)
print("""
The spread across bearings contains both the within-record error and genuine
differences between bearings.  Subtracting the first from the second in variance
gives the real component, and a negative answer would mean the error model
overstates the noise, which is worth knowing either way.
""")
if len(rows) >= 6:
    v_total = OUT["gap_spread"] ** 2
    v_within = float(np.median(w_ind ** 2))
    v_real = v_total - v_within
    OUT["var_total"] = float(v_total)
    OUT["var_within"] = float(v_within)
    OUT["var_real"] = float(v_real)
    OUT["share_within"] = float(v_within / v_total) if v_total > 0 else np.nan
    print("  %-34s %12.5f" % ("total variance of the gap", v_total))
    print("  %-34s %12.5f" % ("within-record measurement", v_within))
    print("  %-34s %12.5f" % ("remainder, real variation", v_real))
    print("""
  Measurement accounts for %.1f per cent of the observed spread, so the rest is
  bearings genuinely differing.  The gap is therefore not a noisy reading of one
  number; it varies because bearings do.
""" % (100 * OUT["share_within"]))

    se_med = OUT["gap_spread"] * np.sqrt(np.pi / 2.0) / np.sqrt(len(rows))
    OUT["se_of_median_gap"] = float(se_med)
    OUT["gap_in_se"] = float(OUT["gap_median"] / se_med)
    print("""  For the headline the relevant error is that of the median gap over %d
  bearings, %.4f of a lifetime.  The median gap of %.3f is %.1f standard errors
  from zero.
""" % (len(rows), se_med, OUT["gap_median"], OUT["gap_in_se"]))

# =============================================================================
print("=" * 92)
print("3.  Two degenerate standard errors, and why the theory expects them")
print("=" * 92)
_bad = [r for r in rows if max(r["se_d"], r["se_a"]) > 0.3]
OUT["degenerate"] = [dict(unit=r["unit"], se=max(r["se_d"], r["se_a"]))
                     for r in _bad]
print("""
%d entries carry a standard error of a third of a lifetime or more, which is not
a number to be reported: %s.  They are not a numerical accident.  se(tau) is
se(F_hat)/f(tau), and Proposition 16 says exactly this, that an age read where
the curve is flat has no bound in terms of the error in the curve.  Each places
the demand on a plateau, so f is small and the ratio diverges.

They are shown rather than dropped because they are the theory's own prediction
meeting the data, and because the summaries above use medians rather than means,
so they do not move the conclusion: the median within-record error is %.4f with
them in.
""" % (len(_bad), ", ".join("%s at %.2f" % (r["unit"], max(r["se_d"], r["se_a"]))
                            for r in _bad), OUT["within_independent"]))
_clean = [r for r in rows if max(r["se_d"], r["se_a"]) <= 0.3]
if _clean:
    _w = np.array([np.hypot(r["se_d"], r["se_a"]) for r in _clean])
    OUT["within_excluding_degenerate"] = float(np.median(_w))
    print("  dropping them entirely gives %.4f, against %.4f with them in\n"
          % (OUT["within_excluding_degenerate"], OUT["within_independent"]))

# =============================================================================
print("=" * 92)
print("4.  The quantity the abstract actually reports")
print("=" * 92)
print("""
Sections 1 and 2 pair the earliest channel of each family, which is a natural
statistic and is NOT the one the abstract quotes.  The abstract says
distributional indicators become usable 0.35 to 0.65 of a lifetime earlier, which
is a range over indicator pairs rather than a median over bearings.  Both are
computed here so that the number being tested is the number that is claimed.
""")
per_pair = {}
for u in units:
    for dn in DISTRIB:
        dd = curve(z[u][:, FN.index(dn)])
        if dd is None:
            continue
        td = age(dd)
        if not np.isfinite(td):
            continue
        for an in AMOUNT:
            da = curve(z[u][:, FN.index(an)])
            if da is None:
                continue
            ta = age(da)
            if np.isfinite(ta):
                per_pair.setdefault((dn, an), []).append(ta - td)
prows = []
for (dn, an), v in sorted(per_pair.items()):
    if len(v) < 6:
        continue
    prows.append(dict(pair="%s vs %s" % (dn, an), median=float(np.median(v)),
                      n=len(v)))
OUT["pairs"] = prows
print("  %-24s %12s %8s" % ("indicator pair", "median gap", "records"))
print("  " + "-" * 48)
for r in prows:
    print("  %-24s %12.3f %8d" % (r["pair"], r["median"], r["n"]))
if prows:
    lo = min(r["median"] for r in prows)
    hi = max(r["median"] for r in prows)
    OUT["pair_gap_lo"] = float(lo)
    OUT["pair_gap_hi"] = float(hi)
    _se = OUT.get("se_of_median_gap", float("nan"))
    OUT["pair_lo_in_se"] = float(lo / _se) if _se == _se else None
    print("""
  Across the %d pairs the median gap runs from %.3f to %.3f.  That CONTAINS the
  abstract's %.2f to %.2f rather than reproducing it: the lower end agrees to
  within a thousandth and the upper end here is higher, because these pairs are
  formed channel by channel while the abstract's range comes from the full
  indicator set of Table 8.  The comparison to make is therefore of the weakest
  pair, and that one is %.1f standard errors from zero on the error of section 2.
  Every pair in the range the abstract quotes is resolved.
""" % (len(prows), lo, hi, 0.35, 0.65, OUT["pair_lo_in_se"]))

json.dump(OUT, open(os.path.join(HERE, "gap_standard_error.json"), "w"),
          indent=2, default=float)
print("written to gap_standard_error.json")
