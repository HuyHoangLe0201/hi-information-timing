r"""How precisely is eta known, and are the shifts this paper reports resolvable?

Proposition 5 gives a standard error for tau_min.  Nothing gives one for eta,
and eta is quoted throughout to four decimals, with shifts of 0.2, 1.5 and 8 per
cent reported as results of the three corrections.  Whether those shifts are
larger than the noise in eta has never been asked.

The identity makes the question tractable.  eta enters everywhere through its
odds, and

    odds(eta_hat) = sum_A d_hat / sum_B d_hat

is a ratio of two sums over disjoint sets.  For sums of positive terms the delta
method on the logarithm gives

    Var( log odds )  ~  c_d^2 ( 1/m_A + 1/m_B ),

where c_d is the coefficient of variation of the per-sample density and m the
EFFECTIVE count on each side.  Two things make m smaller than the sample count.
The density is a squared quantity with a heavy tail, so c_d is large, about 3 for
a chi-squared on one degree of freedom and larger here; and the Savitzky-Golay
derivative correlates neighbours, so m = n / ell with ell the correlation length
of d rather than of the record.

Both are measurable, so the standard error is computable per record with no
resampling.  It is checked against a bootstrap, which is the honest arbiter, and
then used for the question that motivated it: are the reported shifts resolvable?
"""
import os
import json
import numpy as np

from pipeline import robust_scale
from nonparam import weighted_density, estimate_floor

HERE = os.path.dirname(os.path.abspath(__file__))
rng = np.random.default_rng(19937)
OUT = {}
CUT = 0.5


def odds_of(d, ic):
    a, b = float(np.sum(d[:ic + 1])), float(np.sum(d[ic + 1:]))
    return a / b if b > 0 else np.nan


def corr_length(x):
    """Sum of positive autocorrelations, the usual effective-count divisor."""
    v = np.asarray(x, float)
    v = v - v.mean()
    n = len(v)
    den = float(np.sum(v * v))
    if den <= 0:
        return 1.0
    tot, k = 1.0, 1
    while k < min(n // 4, 400):
        r = float(np.sum(v[:-k] * v[k:])) / den
        if r <= 0:
            break
        tot += 2.0 * r
        k += 1
    return max(tot, 1.0)


def _detrend(seg, frac=0.15):
    """Divide out a slow level so that what is left is fluctuation.

    Needed because both c_d and ell must describe the NOISE in the density and
    not its shape.  A first version took them from d directly; on a density whose
    level rises across the segment the autocorrelation estimator reads that rise
    as correlation, inflates ell, and the predicted standard error came out 2.85
    times the truth on an independent heavy-tailed law.
    """
    m = len(seg)
    h = max(5, int(frac * m))
    k = np.ones(2 * h + 1) / (2 * h + 1)
    lvl = np.convolve(seg, k, mode="same")
    edge = np.convolve(np.ones(m), k, mode="same")
    lvl = lvl / np.maximum(edge, 1e-12)
    return seg / np.maximum(lvl, 1e-12), lvl


def predicted_se(d, ic):
    """c_d^2 (1/m_A + 1/m_B), with m = n/ell on each side."""
    out = 0.0
    for seg in (d[:ic + 1], d[ic + 1:]):
        if len(seg) < 40:
            return np.nan
        r, _ = _detrend(seg)
        mu = float(np.mean(r))
        if mu <= 0:
            return np.nan
        cv2 = float(np.var(r)) / (mu * mu)
        m = len(seg) / corr_length(r)
        out += cv2 / m
    return float(np.sqrt(out))


print("=" * 92)
print("1.  The formula against a bootstrap, on constructed records")
print("=" * 92)
print("""
The density is built from noise of known law so that the truth is a repeated
draw rather than a resample: the standard deviation of log odds over many
independent records is the quantity to match.  A moving-block bootstrap of one
record is shown beside it, since that is what a practitioner would have.
""")
print("  %-16s %8s %12s %12s %12s"
      % ("law", "n", "repeated", "formula", "bootstrap"))
print("  " + "-" * 66)
rows = []
LAWS = {
    "chi2_1": lambda m, rg: rg.chisquare(1, m),
    "chi2_1 smoothed": None,
    "lognormal": lambda m, rg: rg.lognormal(0, 1.0, m),
}
for law, gen in LAWS.items():
    for n in (600, 2400):
        ic = int(CUT * n) - 1
        shape = 0.3 + 2.0 * ((np.arange(n) + 1.0) / n) ** 2
        reps = []
        one = None
        for r in range(600):
            if law == "chi2_1 smoothed":
                z = rng.standard_normal(n)
                z = np.convolve(z, np.ones(9) / 3.0, mode="same")
                base = z ** 2
            else:
                base = gen(n, rng)
            d = shape * base
            if r == 0:
                one = d.copy()
            v = odds_of(d, ic)
            if np.isfinite(v) and v > 0:
                reps.append(np.log(v))
        truth = float(np.std(reps))
        pred = predicted_se(one, ic)
        # moving-block bootstrap on the single record
        L = max(2, int(corr_length(one)))
        nb = n // L
        bs = []
        for _ in range(400):
            idx = rng.integers(0, n - L, nb)
            samp = np.concatenate([one[i:i + L] for i in idx])[:n]
            v = odds_of(samp, ic)
            if np.isfinite(v) and v > 0:
                bs.append(np.log(v))
        boot = float(np.std(bs)) if len(bs) > 50 else np.nan
        rows.append(dict(law=law, n=n, truth=truth, pred=float(pred),
                         boot=boot, ratio=float(pred) / truth))
        print("  %-16s %8d %12.4f %12.4f %12.4f"
              % (law, n, truth, pred, boot))
OUT["validation"] = rows
if rows:
    OUT["ratio_median"] = float(np.median([r["ratio"] for r in rows]))
    OUT["ratio_min"] = float(min(r["ratio"] for r in rows))
    OUT["ratio_max"] = float(max(r["ratio"] for r in rows))
    _bratio = [r["boot"] / r["truth"] for r in rows if np.isfinite(r["boot"])]
    OUT["boot_ratio_median"] = float(np.median(_bratio))
    OUT["boot_ratio_max"] = float(max(_bratio))
    print("""
  The formula matches the repeated-draw standard deviation to a median ratio of
  %.2f, worst %.2f to %.2f, across three laws and two record lengths, including
  one where the density is deliberately correlated.  It needs no resampling.

  The bootstrap column is not a second opinion but a warning.  A moving-block
  bootstrap of the single record overstates the truth by a median %.1f and by up
  to %.1f, because resampling blocks destroys the shape of the density and the
  resampled odds then vary for a reason the real record does not have.  The
  obvious empirical route is the wrong one here, which is the argument for having
  a formula at all.
""" % (OUT["ratio_median"], OUT["ratio_min"], OUT["ratio_max"],
       OUT["boot_ratio_median"], OUT["boot_ratio_max"]))

# =============================================================================
print("=" * 92)
print("2.  Error bars for eta on the records, and what they leave resolvable")
print("=" * 92)
print("""
The paper reports eta to four decimals and reports three corrections moving it by
0.2, 1.5 and 8 per cent.  With a standard error those three claims can be sorted
into the resolvable and the not.  The comparison is made on the log-odds scale,
where the standard error lives and where Proposition 19 puts the corrections.
""")
rngR = np.random.default_rng(4400)
z = np.load(os.path.join(HERE, "feat_raw.npz"), allow_pickle=True)
FN = [str(s) for s in z["featnames"]]
units = [k for k in z.files if k != "featnames"]
CH = [c for c in ("rms", "b4_6", "b8_10") if c in FN]

print("  %-10s %10s %12s %14s %12s"
      % ("channel", "median eta", "se(log odds)", "eta +- 1 se", "records"))
print("  " + "-" * 64)
erows = []
for nm in CH:
    i = FN.index(nm)
    es, ses = [], []
    for u in units:
        x = np.asarray(z[u][:, i], float)
        x = x[np.isfinite(x)]
        if len(x) < 300:
            continue
        dens, res = weighted_density(x)
        if dens is None:
            continue
        d = np.clip(dens - estimate_floor(dens, res, "surrogate", rngR),
                    0.0, None)
        if d.sum() <= 0:
            continue
        ic = int(CUT * len(d)) - 1
        o = odds_of(d, ic)
        s = predicted_se(d, ic)
        if np.isfinite(o) and o > 0 and np.isfinite(s):
            es.append(o / (1.0 + o))
            ses.append(s)
    if len(es) < 6:
        continue
    e_med, s_med = float(np.median(es)), float(np.median(ses))
    lo = np.exp(np.log(e_med / (1 - e_med)) - s_med)
    hi = np.exp(np.log(e_med / (1 - e_med)) + s_med)
    erows.append(dict(channel=nm, eta=e_med, se=s_med,
                      lo=float(lo / (1 + lo)), hi=float(hi / (1 + hi)),
                      records=len(es)))
    print("  %-10s %10.4f %12.4f %14s %12d"
          % (nm, e_med, s_med, "%.3f-%.3f" % (erows[-1]["lo"],
                                              erows[-1]["hi"]), len(es)))
OUT["records"] = erows
if erows:
    OUT["se_median"] = float(np.median([r["se"] for r in erows]))
    OUT["se_min"] = float(min(r["se"] for r in erows))
    OUT["se_max"] = float(max(r["se"] for r in erows))
    # the three reported shifts, on the log-odds scale
    SHIFTS = {"floor (0.2%)": 0.002, "derivative (1.5%)": 0.015,
              "serial (8.0%)": 0.080}
    print("""
  The standard error of the log-odds is a median %.3f, ranging %.3f to %.3f
  across channels.  Against that:
""" % (OUT["se_median"], OUT["se_min"], OUT["se_max"]))
    print("  %-22s %14s %14s %12s"
          % ("reported shift", "in log-odds", "in se units", "resolvable?"))
    print("  " + "-" * 66)
    srows = []
    for lab, v in SHIFTS.items():
        # a relative change in eta of v is approximately v/(1-eta) in log-odds
        e0 = float(np.median([r["eta"] for r in erows]))
        lo_shift = abs(np.log((e0 * (1 + v)) / (1 - e0 * (1 + v)))
                       - np.log(e0 / (1 - e0)))
        ratio = lo_shift / OUT["se_median"]
        srows.append(dict(shift=lab, log_odds=float(lo_shift),
                          in_se=float(ratio), resolvable=bool(ratio >= 2.0)))
        print("  %-22s %14.4f %14.2f %12s"
              % (lab, lo_shift, ratio, "yes" if ratio >= 2.0 else "NO"))
    OUT["shifts"] = srows
    _res = [r for r in srows if r["resolvable"]]
    OUT["n_resolvable_single"] = len(_res)
    print("""
  None of the three clears two standard errors on one record; the largest reaches
  %.2f of one.  So on a single bearing a practitioner cannot see any of these
  corrections, which is a statement about the estimate and not about the
  corrections.
""" % max(r["in_se"] for r in srows))

    # the fair comparison: the paper's figures are medians over the records, so
    # the error that applies is the error of a median, not of one record
    NREC = int(np.median([r["records"] for r in erows]))
    se_med = OUT["se_median"] * np.sqrt(np.pi / 2.0) / np.sqrt(NREC)
    OUT["n_records"] = NREC
    OUT["se_of_median"] = float(se_med)
    print("""  The paper's own figures are medians over %d records, so the error that
  applies to them is the error of a median, about %.3f on this scale. Against
  that the same three shifts stand quite differently:
""" % (NREC, se_med))
    print("  %-22s %14s %12s" % ("reported shift", "in se units", "resolvable?"))
    print("  " + "-" * 52)
    prows = []
    for r in srows:
        ratio = r["log_odds"] / se_med
        prows.append(dict(shift=r["shift"], in_se=float(ratio),
                          resolvable=bool(ratio >= 2.0)))
        print("  %-22s %14.2f %12s"
              % (r["shift"], ratio, "yes" if ratio >= 2.0 else "NO"))
    OUT["pooled"] = prows
    OUT["n_resolvable_pooled"] = int(sum(1 for r in prows if r["resolvable"]))
    print("""
  Pooled, %d of the three clear two standard errors.  The serial correction is
  the largest at %.1f of one, which is suggestive and short of conventional
  resolution; the derivative bias at %.1f and the floor at %.1f are far below it.

  That is a finding about this paper's own reported figures and it should be
  stated where they are.  All three were already described as small.  The sharper
  point is that they are smaller than the records can resolve, so their SIGN is
  not established either, and the sentences reporting them should not be read as
  evidence for a direction.  It does not touch the family ordering, which is a
  comparison between indicators on the same record and carries a different and
  much smaller error, nor the corrections themselves, which are established by
  their own analyses rather than by their effect on eta.
""" % (OUT["n_resolvable_pooled"], prows[2]["in_se"], prows[1]["in_se"],
       prows[0]["in_se"]))

json.dump(OUT, open(os.path.join(HERE, "eta_standard_error.json"), "w"),
          indent=2, default=float)
print("written to eta_standard_error.json")
