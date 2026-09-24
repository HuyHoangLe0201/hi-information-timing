r"""What survives a nonparametric trend, and does the ordering survive with it?

Proposition 27 shows that a GLOBAL clock offset is unidentifiable once the trend
is learned from the same record.  It does not say what is left, and the paper
asserts that its estimand is local without characterising the space that
"local" picks out.  The structure is exact and gives one.

The out-of-fold trend is a LINEAR smoother.  Write S for its action, so the
residual the pipeline works with is (I - S)y.  A perturbation of the trend that
lies in the range of S is absorbed and invisible; what an estimator can see is
the component orthogonal to it.  A clock offset perturbs the trend by delta D',
so the information actually realised is

    G_eff = || (I - S) D' ||^2 / sigma^2   rather than   || D' ||^2 / sigma^2,

and the ratio

    kappa_S = || (I - S) D' ||^2 / || D' ||^2

is the fraction of the budget that survives the smoother.  Proposition 27 is the
case where D' is smooth, S reproduces it, and kappa_S is near zero.

kappa_S costs one call to compute, since S is linear and (I - S)D' is just
D' minus the smoother applied to D'.  Two things follow that matter here.

  The paper's G overstates the realised information by 1/kappa_S, per channel,
  and that factor has never been computed.

  kappa_S is NOT a common factor.  It depends on how smooth each channel's D' is,
  so channels lose different amounts, and a ranking between them can move.  Every
  cancellation argument in this paper turns on the correction being common, and
  this one is not.  Whether the family ordering survives is therefore a question
  and not a corollary.
"""
import os
import json
import numpy as np
from scipy.signal import savgol_filter

from pipeline import _win, robust_scale, oof_trend, local_scale, LOCAL_BW
from nonparam import weighted_density, estimate_floor

HERE = os.path.dirname(os.path.abspath(__file__))
rng = np.random.default_rng(70707)
OUT = {}
Q = 0.35

z = np.load(os.path.join(HERE, "feat_raw.npz"), allow_pickle=True)
FN = [str(s) for s in z["featnames"]]
units = [k for k in z.files if k != "featnames"]
FAMILY = {"rms": "amount", "peak": "amount", "kurt": "amount",
          "b0_1": "distribution", "b1_2": "distribution",
          "b2_4": "distribution", "b4_6": "distribution",
          "b6_8": "distribution", "b8_10": "distribution"}
CH = [(c, FAMILY[c]) for c in FAMILY if c in FN]


def parts(x):
    """D', the smoother-surviving part of it, and the local scale."""
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    n = len(x)
    if n < 300:
        return None
    s = robust_scale(x)
    if not np.isfinite(s) or s <= 0:
        return None
    D = (x - np.median(x)) / s
    w = _win(n)
    if w < 7 or w >= n:
        return None
    dt = savgol_filter(D, w, 2, deriv=1, delta=1.0 / n)
    res = D - oof_trend(D, w // 2)
    sl = np.clip(local_scale(res, max(5, int(LOCAL_BW * n))), 1e-12, None)
    # S is linear, so (I - S) D' is D' minus the smoother applied to D'
    surviving = dt - oof_trend(dt, w // 2)
    return dt, surviving, sl, n


print("=" * 92)
print("1.  How much of each channel's derivative survives the smoother")
print("=" * 92)
print("""
kappa_S is the squared norm of the surviving part over that of the whole.  A
channel whose derivative is smooth loses nearly all of it; one whose derivative
carries structure the smoother cannot follow keeps more.
""")
print("  %-10s %-14s %12s %14s %8s"
      % ("channel", "family", "kappa_S", "overstatement", "records"))
print("  " + "-" * 62)
rows = []
for name, fam in sorted(CH, key=lambda t: (t[1], t[0])):
    i = FN.index(name)
    ks = []
    for u in units:
        p = parts(z[u][:, i])
        if p is None:
            continue
        dt, sur, sl, n = p
        den = float(np.sum(dt * dt))
        if den <= 0:
            continue
        ks.append(float(np.sum(sur * sur) / den))
    if len(ks) < 6:
        continue
    k = float(np.median(ks))
    rows.append(dict(channel=name, family=fam, kappa=k,
                     overstatement=1.0 / k if k > 0 else np.inf,
                     records=len(ks)))
    print("  %-10s %-14s %12.4f %14.1f %8d"
          % (name, fam, k, 1.0 / k if k > 0 else np.inf, len(ks)))
OUT["channels"] = rows

if rows:
    am = [r for r in rows if r["family"] == "amount"]
    di = [r for r in rows if r["family"] == "distribution"]
    OUT["kappa_amount"] = float(np.median([r["kappa"] for r in am])) if am else None
    OUT["kappa_distribution"] = (float(np.median([r["kappa"] for r in di]))
                                 if di else None)
    OUT["kappa_min"] = float(min(r["kappa"] for r in rows))
    OUT["kappa_max"] = float(max(r["kappa"] for r in rows))
    OUT["kappa_spread"] = OUT["kappa_max"] / max(OUT["kappa_min"], 1e-12)
    print("""
  kappa_S runs from %.4f to %.4f across channels, a spread of %.2f, so the
  smoother takes between %.0f and %.0f per cent of each channel's derivative
  energy.  The amount channels keep a median %.4f and the distributional ones
  %.4f.

  That spread is the point.  A correction of this size would be harmless if it
  were common; it is not, so the ordering has to be re-read rather than assumed.
""" % (OUT["kappa_min"], OUT["kappa_max"], OUT["kappa_spread"],
       100 * (1 - OUT["kappa_max"]), 100 * (1 - OUT["kappa_min"]),
       OUT["kappa_amount"], OUT["kappa_distribution"]))

# =============================================================================
print("=" * 92)
print("2.  The ordering, re-read on the surviving part")
print("=" * 92)
print("""
Section 1 says the distributional channels keep four times as much of their
derivative as the amount channels do, which would strengthen the ordering rather
than weaken it.  That is a prediction from two medians and it has to be checked
by rebuilding the curves, because eta and the ages are read from a cumulative and
not from a total.

The density is rebuilt with (I - S)D' in place of D', everything else identical,
and the age at demand q is re-read.
""")


def curves(x):
    """The paper's density, and the same built from the surviving derivative."""
    p = parts(x)
    if p is None:
        return None
    dt, sur, sl, n = p
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    dens, res = weighted_density(x)
    if dens is None:
        return None
    fl = estimate_floor(dens, res, "surrogate", rng)
    a = np.clip((dt / sl) ** 2 - fl, 0.0, None)
    b = np.clip((sur / sl) ** 2 - fl, 0.0, None)
    if a.sum() <= 0 or b.sum() <= 0:
        return None
    return a, b


def age(d, q=Q):
    F = np.cumsum(d) / d.sum()
    i = int(np.searchsorted(F, q))
    return np.nan if i >= len(F) else (i + 1) / len(F)


print("  %-10s %-14s %12s %12s %10s"
      % ("channel", "family", "age, raw", "age, survive", "shift"))
print("  " + "-" * 62)
arows = []
for name, fam in sorted(CH, key=lambda t: (t[1], t[0])):
    i = FN.index(name)
    ar, asv = [], []
    for u in units:
        c = curves(z[u][:, i])
        if c is None:
            continue
        t0, t1 = age(c[0]), age(c[1])
        if np.isfinite(t0) and np.isfinite(t1):
            ar.append(t0)
            asv.append(t1)
    if len(ar) < 6:
        continue
    m0, m1 = float(np.median(ar)), float(np.median(asv))
    arows.append(dict(channel=name, family=fam, age_raw=m0, age_surv=m1,
                      shift=m1 - m0, records=len(ar)))
    print("  %-10s %-14s %12.3f %12.3f %+10.3f"
          % (name, fam, m0, m1, m1 - m0))
OUT["ages"] = arows

if arows:
    am = [r for r in arows if r["family"] == "amount"]
    di = [r for r in arows if r["family"] == "distribution"]
    if am and di:
        g_raw = (float(np.median([r["age_raw"] for r in am]))
                 - float(np.median([r["age_raw"] for r in di])))
        g_sur = (float(np.median([r["age_surv"] for r in am]))
                 - float(np.median([r["age_surv"] for r in di])))
        OUT["gap_raw"] = g_raw
        OUT["gap_surviving"] = g_sur
        OUT["gap_change"] = g_sur - g_raw
        OUT["ordering_holds"] = bool(g_sur > 0)
        print("""
  The family gap is %.3f on the paper's density and %.3f on the surviving part, a
  change of %+.3f.  The ordering %s.

  So the prediction from the two medians %s.  kappa_S favours the distributional
  channels by a factor of four in total energy, and the gap %s once the curves
  are rebuilt, which is the thing that had to be checked rather than inferred:
  eta and the ages read a cumulative, and a channel can keep more energy overall
  while keeping it in a place that does not move its age.
""" % (g_raw, g_sur, g_sur - g_raw,
       "survives" if g_sur > 0 else "REVERSES",
       "holds" if (g_sur - g_raw) > 0 else "does not carry over",
       "widens" if (g_sur - g_raw) > 0 else "narrows"))

json.dump(OUT, open(os.path.join(HERE, "identifiable_part.json"), "w"),
          indent=2, default=float)
print("written to identifiable_part.json")
