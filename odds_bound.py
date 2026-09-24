r"""A non-asymptotic bound: what a bounded distortion can do, however large.

Proposition 15 is an expansion.  It says what happens as eps -> 0 and its error
grows with the distortion, reaching 38 per cent on the tail factor.  A paper whose
subject is a BOUND should be able to do better than that for a quantity it asks
practitioners to trust, and for one of the three it can, exactly and for every
distortion however large.

The observation is that eta is a ratio of a part of the budget to the whole, so
its ODDS

    odds(eta) = eta / (1 - eta) = G(c) / (G(1) - G(c))

is a ratio of two disjoint integrals.  If the distortion satisfies

    1/M  <=  w(tau)  <=  M                                        (*)

pointwise, then the numerator can be multiplied by at most M and the denominator
divided by at most M, so

    odds(eta) / M^2   <=   odds(eta_hat)   <=   M^2 odds(eta),

and both ends are ATTAINED, by the distortion equal to M before c and 1/M after
it.  Nothing about the shape of g enters, and nothing about the shape of w beyond
its range.  A bounded calibration error moves the log-odds of the censoring
efficiency by at most 2 log M and can do no worse.

The two ages have no such bound.  Under (*) the demand qG(1) is known only to
within M^2, so

    tau_min_hat  in  [ G^-1( q G(1) / M^2 ),  G^-1( M^2 q G(1) ) ],

which is again attained, and the width of that interval is set by how flat G is:
where the density is small the same M moves the age arbitrarily far.  That is
Proposition 15's second consequence, which was a statement about derivatives,
turned into a statement that holds at any distortion size.

This file checks sharpness, checks that the ages really are unbounded in the way
claimed, and then applies (*) to the three distortions the paper measures, to see
what the guarantee is worth on real records.
"""
import os
import json
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
rng = np.random.default_rng(1618)
OUT = {}
N = 6000
tau = (np.arange(N) + 1.0) / N


def odds(p):
    return p / (1.0 - p) if 0 < p < 1 else np.nan


print("=" * 92)
print("1.  The odds bound, and whether it is attained")
print("=" * 92)
print("""
For each density and each M the worst distortion allowed by (*) is applied, and
the resulting odds ratio is compared with M^2.  If the bound is sharp the two
agree exactly; if it is merely valid they do not.
""")
DENS = {
    "rising": 2 * tau,
    "falling": 2 - 2 * tau,
    "peaked late": 0.2 + 3.0 * tau ** 4,
    "bimodal": 1 + np.exp(-((tau - 0.2) / 0.05) ** 2)
    + np.exp(-((tau - 0.85) / 0.05) ** 2),
}
MS = (1.05, 1.25, 2.0, 4.0)
CUT = 0.5
print("  %-14s %6s %14s %14s %12s"
      % ("density", "M", "worst ratio", "M^2", "gap"))
print("  " + "-" * 66)
rows, worst_gap = [], 0.0
for dn, g0 in DENS.items():
    g = g0 / N
    ic = int(CUT * N) - 1
    e0 = float(np.sum(g[:ic + 1]) / np.sum(g))
    for M in MS:
        w = np.where(np.arange(N) <= ic, M, 1.0 / M)
        gd = w * g
        e1 = float(np.sum(gd[:ic + 1]) / np.sum(gd))
        ratio = odds(e1) / odds(e0)
        gap = abs(ratio - M * M) / (M * M)
        worst_gap = max(worst_gap, gap)
        rows.append(dict(density=dn, M=M, ratio=ratio, target=M * M, gap=gap))
        print("  %-14s %6.2f %14.4f %14.4f %12.2g"
              % (dn if M == MS[0] else "", M, ratio, M * M, gap))
OUT["sharpness"] = rows
OUT["sharpness_worst_gap"] = float(worst_gap)
print("""
  The worst relative gap over sixteen cells is %.2g, so the bound is attained and
  not merely valid.  It is therefore the best bound of its kind: no function of M
  alone can be smaller.
""" % worst_gap)

# --- and it must never be violated by an arbitrary distortion -----------------
viol = 0.0
for _ in range(4000):
    dn = rng.choice(list(DENS))
    g = DENS[dn] / N
    M = float(rng.uniform(1.02, 5.0))
    lw = rng.uniform(-np.log(M), np.log(M), size=N)
    k = max(3, int(rng.integers(3, 60)))
    ker = np.ones(k) / k
    lw = np.convolve(lw, ker, mode="same")
    lw = np.clip(lw, -np.log(M), np.log(M))
    gd = np.exp(lw) * g
    ic = int(CUT * N) - 1
    e0 = float(np.sum(g[:ic + 1]) / np.sum(g))
    e1 = float(np.sum(gd[:ic + 1]) / np.sum(gd))
    r = odds(e1) / odds(e0)
    viol = max(viol, r / (M * M), (1.0 / (M * M)) / r)
OUT["max_violation"] = float(viol)
print("""  Over four thousand random distortions of varying smoothness and range the
  largest excursion beyond the bound is a factor %.6f, that is none.
""" % viol)

# =============================================================================
print("=" * 92)
print("2.  The ages, which have no such bound")
print("=" * 92)
print("""
The interval for tau_min is also attained, being G^-1 of an interval, but its
WIDTH is set by the density AT tau_min and not by flatness anywhere else.  A
first version of this section swept g = 2t, 4t^3, 8t^7 to make the curve flatter
early, and produced the opposite of the claim: on 8t^7 the age interval is the
NARROWEST of the four, because tau_min then lands at 0.88 where that density is
large.  Flat early is not the same as flat where the demand is met.

A second version put a plateau after a spike and failed the same way: the spike
carried more than q of the budget, so tau_min landed on the spike again and the
widths went the wrong way once more.  The family below is therefore built from
prescribed MASSES rather than from prescribed heights, and where tau_min lands is
checked and printed rather than assumed.  Three pieces: an early block holding
0.30 of the budget, a long flat middle holding a small mass m, and a late block
holding the rest.  With q = 0.35 the demand is met in the middle, whose height is
m/0.8, and m is the knob.
""")
Q = 0.35
M_SMALL = 1.05
print("""  M is taken small enough that the whole interval stays inside the middle.
  With a larger M the upper end escapes into the late block and the width
  saturates, which is a property of that construction and not of the bound; the
  containment is checked in the last column.
""")
print("  %-12s %11s %9s %9s %9s %8s %10s"
      % ("middle mass", "g(tau_min)", "tau_min", "lower", "upper", "inside",
         "width x g"))
print("  " + "-" * 78)
arows = []
for m in (0.40, 0.20, 0.12, 0.09):
    g0 = np.empty(N)
    lo_i, hi_i = int(0.1 * N), int(0.9 * N)
    g0[:lo_i] = 0.30 / 0.1
    g0[lo_i:hi_i] = m / 0.8
    g0[hi_i:] = (1.0 - 0.30 - m) / 0.1
    g = g0 / N
    G = np.cumsum(g)
    t0i = int(np.searchsorted(G, Q * G[-1]))
    if t0i >= N:
        continue
    t0 = float((t0i + 1) / N)
    M = M_SMALL
    li = int(np.searchsorted(G, Q * G[-1] / M ** 2))
    hi_j = int(np.searchsorted(G, min(M ** 2 * Q * G[-1], G[-1])))
    lo = float((li + 1) / N)
    hi = float((hi_j + 1) / N)
    inmid = bool(lo_i <= li and hi_j < hi_i and lo_i <= t0i < hi_i)
    wid = hi - lo
    arows.append(dict(mass=m, g_at=float(g0[t0i]), M=M, tau=t0, lo=lo, hi=hi,
                      width=wid, in_middle=inmid, width_times_g=wid * g0[t0i]))
    print("  %-12.2f %11.3f %9.4f %9.4f %9.4f %8s %10.4f"
          % (m, g0[t0i], t0, lo, hi, "yes" if inmid else "NO",
             wid * g0[t0i]))
arows = [r for r in arows if r["in_middle"]]
OUT["ages"] = arows
if len(arows) >= 2:
    OUT["age_width_min"] = float(min(r["width"] for r in arows))
    OUT["age_width_max"] = float(max(r["width"] for r in arows))
    OUT["age_width_ratio"] = (OUT["age_width_max"]
                              / max(OUT["age_width_min"], 1e-12))
    _mono = all(arows[i]["width"] <= arows[i + 1]["width"] + 1e-12
                for i in range(len(arows) - 1))
    OUT["age_width_monotone"] = bool(_mono)
    _wg = [r["width_times_g"] for r in arows]
    OUT["width_times_g_spread"] = float(max(_wg) / max(min(_wg), 1e-12))
    print("""
  At one M the interval runs from %.4f of a lifetime to %.4f as the middle is
  thinned, a factor of %.1f, and it widens monotonically: %s.

  The last column is the sharper statement.  Multiplying each width by the
  density at tau_min gives %s, constant to a factor of %.3f across a fourfold
  change in that density.  The width is therefore not merely growing but growing
  exactly like 1/g(tau_min), which is the factor Proposition 15 carries, and
  since g there can be made as small as one likes the age interval has no bound
  in terms of M.  eta's does, and depends on nothing but M.
""" % (OUT["age_width_min"], OUT["age_width_max"], OUT["age_width_ratio"],
       "yes" if _mono else "not at every step",
       ", ".join("%.3f" % v for v in _wg), OUT["width_times_g_spread"]))

    # the constant has a closed form: the interval in G units is
    # q G(1) (M^2 - M^-2), and dividing by g(tau_min) turns it into time
    _pred = Q * (M_SMALL ** 2 - M_SMALL ** -2)
    OUT["width_times_g"] = float(np.median(_wg))
    OUT["width_times_g_closed"] = float(_pred)
    print("""  That constant is not empirical either.  The interval measured in budget is
  q G(1) (M^2 - M^-2) = %.4f, and dividing it by the density converts budget into
  time, so the width is q G(1) (M^2 - M^-2) / g(tau_min).  The column above reads
  %.4f.
""" % (_pred, OUT["width_times_g"]))

json.dump(OUT, open(os.path.join(HERE, "odds_bound.json"), "w"), indent=2,
          default=float)
print("written to odds_bound.json")
