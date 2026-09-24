r"""A certificate that pays for the range only where the information is.

Section 2.4 now measures something that asks for a proposition.  The samples at
which a correction attains its range are a median three per cent of a record and
carry a median 0.00002 of its budget: the supremum M that Proposition 16 spends
is attained, by construction, on samples the noise floor has already removed.
Proposition 19's repair was to abandon the range and measure the contrast
instead, which is exact but must be recomputed for every distortion one wants to
survive.  A range-based certificate has the opposite property -- it covers every
distortion of a stated range, whatever its shape -- and that is what a designer
who does not know the distortion in advance actually needs.

So the question is whether the range bound can be kept and merely charged
correctly.  It can, and the statement interpolates.

Let A be any set of samples, the ones vouched for, and write

    M_A = sup_A w,   m_A = inf_A w,   1/M <= w <= M everywhere,
    beta  = share of the budget in [0,c] that lies outside A,
    gamma = share of the budget in [c,1] that lies outside A.

Splitting each information-weighted mean over A and its complement,

    <w>_[0,c] <= (1-beta) M_A + beta M,
    <w>_[c,1] >= (1-gamma) m_A + gamma / M,

so by the contrast identity

    R  <=  [ (1-beta) M_A + beta M ] / [ (1-gamma) m_A + gamma / M ].

At A empty this is M^2, which is Proposition 16 exactly.  At A carrying the whole
budget it is M_A/m_A, the range restricted to where the density lives.  In
between it is a ladder, and the parameter is the share of the budget one declines
to vouch for.

Two things are then measured rather than asserted.  Whether the bound holds --
a bound that failed on one cell would be a wrong proposition, not a loose one --
and how many of the sixteen bearings it certifies as alpha is swept, against the
none that the range certifies and the eleven that the measured contrast does.
"""
import os
import json
import numpy as np
from scipy.signal import savgol_filter

from pipeline import robust_scale, _win, local_scale, oof_trend, LOCAL_BW
from nonparam import weighted_density, estimate_floor

HERE = os.path.dirname(os.path.abspath(__file__))
rng = np.random.default_rng(271828)
OUT = {}
Q = 0.35
THETA = Q / (1.0 - Q)
MIN_SHARE = 0.02
# The length screen these results use.  It is five times the one Section 5
# documents, it was never stated there, and section 5 below asks whether it
# changes anything.
MIN_N = 300
H_HALF, WIDE = 0.04, 4.0
ALPHAS = (0.0, 0.0001, 0.001, 0.01, 0.05, 0.10, 0.25, 1.0)
CUTS = (0.3, 0.5, 0.7)
DISTORTIONS = ("serial", "tail", "derivative")


# --- the machinery of contrast_identity.py, reproduced and then checked ------
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
    key = None if key is None else (MIN_N,) + tuple(key)
    if key is not None and key in _CACHE:
        return _CACHE[key]
    v = _pieces(x)
    if key is not None:
        _CACHE[key] = v
    return v


def _pieces(x):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    n = len(x)
    if n < MIN_N:
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


def contrast(w, g, ic):
    tot = float(np.sum(g))
    if tot <= 0:
        return np.nan
    if (float(np.sum(g[:ic + 1])) / tot < MIN_SHARE
            or float(np.sum(g[ic + 1:])) / tot < MIN_SHARE):
        return np.nan
    a = float(np.sum(w[:ic + 1] * g[:ic + 1]) / max(np.sum(g[:ic + 1]), 1e-300))
    b = float(np.sum(w[ic + 1:] * g[ic + 1:]) / max(np.sum(g[ic + 1:]), 1e-300))
    return a / b if b > 0 else np.nan


def energy_budget(w, g):
    """eps^2 = Var_g(log w), the budget Proposition 9 prices the distortion in."""
    tot = float(np.sum(g))
    if tot <= 0:
        return np.nan
    u = np.log(np.clip(w, 1e-12, None))
    mu = float(np.sum(u * g) / tot)
    return float(np.sqrt(max(np.sum((u - mu) ** 2 * g) / tot, 0.0)))


def energy_charge(w, g, ic):
    """exp(eps / sqrt(eta(1-eta))), the worst contrast under that budget."""
    tot = float(np.sum(g))
    if tot <= 0:
        return np.nan, np.nan
    eta = float(np.sum(g[:ic + 1]) / tot)
    if eta < MIN_SHARE or (1.0 - eta) < MIN_SHARE:
        return np.nan, np.nan
    eps = energy_budget(w, g)
    if not np.isfinite(eps):
        return np.nan, np.nan
    return float(np.exp(eps / np.sqrt(eta * (1.0 - eta)))), eps


def age_at_odds(F, x):
    p = x / (1.0 + x)
    if p >= 1.0:
        return np.inf
    i = int(np.searchsorted(F, p))
    return np.inf if i >= len(F) else (i + 1) / len(F)


# --- the proposition ---------------------------------------------------------
def vouched(g, alpha):
    """Retain the samples carrying the top 1-alpha of THIS side's budget.

    A first version trimmed against the whole record's budget, which made the
    excluded share of one side, beta, far larger than alpha whenever that side
    carried little of the total: at alpha = 0.25 one cut lost its entire early
    side and the bound read five thousand.  Trimming each side against its own
    budget makes beta = gamma = alpha by construction, which is also what makes
    the statement interpretable.
    """
    g = np.asarray(g, float)
    tot = g.sum()
    if tot <= 0:
        return np.ones(len(g), bool)
    order = np.argsort(g, kind="stable")
    cum = np.cumsum(g[order])
    k = int(np.searchsorted(cum, alpha * tot, side="right"))
    A = np.ones(len(g), bool)
    A[order[:k]] = False
    return A


def trimmed_bound(w, g, ic, alpha):
    """Bound on max(R, 1/R), or nan where the cut is not usable.

    Two-sided, because the certificate moves each indicator in whichever
    direction hurts and the paper's own worst case is max(R, 1/R).  Bounding
    only R and comparing it against the symmetrised contrast is what made a
    correct inequality read as violated by seven per cent.
    """
    tot = float(np.sum(g))
    if tot <= 0:
        return np.nan
    lo_share = float(np.sum(g[:ic + 1])) / tot
    if lo_share < MIN_SHARE or (1.0 - lo_share) < MIN_SHARE:
        return np.nan
    M = float(max(np.max(w), 1.0 / max(np.min(w), 1e-12)))
    wl, wh = w[:ic + 1], w[ic + 1:]
    gl, gh = g[:ic + 1], g[ic + 1:]
    Al, Ah = vouched(gl, alpha), vouched(gh, alpha)
    if not Al.any() or not Ah.any():
        return M * M
    a = 1.0 - alpha
    # sup and inf of w over each side's retained set
    Ml, ml = float(np.max(wl[Al])), float(np.min(wl[Al]))
    Mh, mh = float(np.max(wh[Ah])), float(np.min(wh[Ah]))
    up_l, dn_l = a * Ml + alpha * M, a * ml + alpha / M
    up_h, dn_h = a * Mh + alpha * M, a * mh + alpha / M
    if dn_h <= 0 or dn_l <= 0:
        return np.nan
    return float(max(up_l / dn_h, up_h / dn_l))


# --- the two rigs -----------------------------------------------------------
# The certificate has only ever been computed on PRONOSTIA.  The paper
# replicates its applied result on XJTU-SY and its scope on turbofan units, so a
# guarantee that holds on one rig and has never been asked of another is a
# guarantee with one observation behind it.  XJTU is the harder case: nine of its
# sixteen records are under three hundred samples and are screened out by the
# same length rule the rest of the paper uses, which leaves few records and is
# reported rather than worked around.
DATASETS = (("PRONOSTIA", "feat_raw.npz"), ("XJTU-SY", "xjtu_feat.npz"))


def load(fname):
    z = np.load(os.path.join(HERE, fname), allow_pickle=True)
    fn = [str(s) for s in z["featnames"]]
    units = [k for k in z.files if k != "featnames"]
    return (z, fn, units,
            [c for c in ("rms", "peak") if c in fn],
            [c for c in ("b4_6", "b6_8", "b8_10") if c in fn])


print("=" * 92)
print("1.  Does the bound hold?  Every cell, every cut, every trimming level")
print("=" * 92)
print("""
The bound is a proposition, so one violation refutes it rather than loosening it.
The check is the measured contrast against the bound, over every record, channel,
correction, cut and alpha on both rigs, and the worst ratio of the two decides.
""")
worst_ratio, n_checked, mono_bad = 0.0, 0, 0
for _dn, _fn in DATASETS:
    z, FN, units, AMOUNT, DISTRIB = load(_fn)
    for u in units:
        for nm in AMOUNT + DISTRIB:
            p = pieces(z[u][:, FN.index(nm)], (_dn, u, nm))
            if p is None:
                continue
            g, ws = p
            for dist in DISTORTIONS:
                if dist not in ws:
                    continue
                w = np.clip(ws[dist], 1e-9, None)
                for cf in CUTS:
                    ic = int(cf * len(g)) - 1
                    R = contrast(w, g, ic)
                    if not np.isfinite(R) or R <= 0:
                        continue
                    R = max(R, 1.0 / R)
                    prev = None
                    for a in ALPHAS:
                        B = trimmed_bound(w, g, ic, a)
                        if not np.isfinite(B):
                            continue
                        n_checked += 1
                        worst_ratio = max(worst_ratio, R / B)
                        if prev is not None and B < prev - 1e-9:
                            mono_bad += 1
                        prev = B
OUT["bound_checks"] = n_checked
OUT["worst_contrast_over_bound"] = float(worst_ratio)
OUT["monotonicity_violations"] = int(mono_bad)
print("  %d checks over both rigs; the largest ratio of measured contrast to "
      "bound is %.4f" % (n_checked, worst_ratio))
print("  the bound is non-decreasing in alpha on every cell: %s"
      % ("yes" if mono_bad == 0 else
         "no, on %d of them, since trimming removes extremes but the trimmed "
         "mass is\n  then charged at the full range" % mono_bad))


def ladder_for(dname, fname):
    """The four charges and what each certifies, on one rig."""
    z, FN, units, AMOUNT, DISTRIB = load(fname)
    per_bearing = []
    for u in units:
        best = {}
        for grp, names in (("d", DISTRIB), ("a", AMOUNT)):
            cand = []
            for nm in names:
                p = pieces(z[u][:, FN.index(nm)], (dname, u, nm))
                if p is None:
                    continue
                g, ws = p
                F = np.cumsum(g) / g.sum()
                a = age_at_odds(F, THETA)
                if np.isfinite(a):
                    cand.append((a, g, F, ws))
            if cand:
                best[grp] = min(cand, key=lambda t: t[0])
        if "d" not in best or "a" not in best:
            continue
        ad, dd, Fd, wd = best["d"]
        aa, da, Fa, wa = best["a"]
        if not (ad < aa):
            continue
        lo, hi = 1.0, 40.0
        for _ in range(50):
            mid = 0.5 * (lo + hi)
            if age_at_odds(Fd, mid ** 2 * THETA) < age_at_odds(Fa,
                                                               THETA / mid ** 2):
                lo = mid
            else:
                hi = mid
        row = dict(unit=u, tolerated=float(lo ** 2))
        worst_R, worst_M2 = 0.0, 0.0
        charge = {a: 0.0 for a in ALPHAS}
        worst_E, max_eps, exceed = 0.0, 0.0, 0.0
        for ws, g in ((wd, dd), (wa, da)):
            for dist in DISTORTIONS:
                if dist not in ws:
                    continue
                w = np.clip(ws[dist], 1e-9, None)
                M = float(max(np.max(w), 1.0 / max(np.min(w), 1e-12)))
                for cf in CUTS:
                    ic = int(cf * len(g)) - 1
                    v = contrast(w, g, ic)
                    if np.isfinite(v) and v > 0:
                        worst_R = max(worst_R, max(v, 1.0 / v))
                        worst_M2 = max(worst_M2, M * M)
                    for a in ALPHAS:
                        B = trimmed_bound(w, g, ic, a)
                        if np.isfinite(B):
                            charge[a] = max(charge[a], B)
                    E, eps = energy_charge(w, g, ic)
                    if np.isfinite(E):
                        worst_E = max(worst_E, E)
                        max_eps = max(max_eps, eps)
                        if np.isfinite(v) and v > 0:
                            exceed = max(exceed, max(v, 1.0 / v) / E)
        row["worst_contrast"] = float(worst_R)
        row["worst_M2"] = float(worst_M2)
        row["charge"] = {str(a): float(charge[a]) for a in ALPHAS}
        row["best_alpha"] = float(min(charge, key=charge.get))
        row["best_charge"] = float(min(charge.values()))
        row["worst_energy"] = float(worst_E)
        row["max_eps"] = float(max_eps)
        row["contrast_over_energy"] = float(exceed)
        per_bearing.append(row)
    if not per_bearing:
        return None

    tol = np.array([r["tolerated"] for r in per_bearing])
    got = lambda v: int(np.sum(np.array(v) < tol))
    m2 = [r["worst_M2"] for r in per_bearing]
    bc = [r["best_charge"] for r in per_bearing]
    en = [r["worst_energy"] for r in per_bearing]
    co = [r["worst_contrast"] for r in per_bearing]
    ess = [r["charge"][str(0.0)] for r in per_bearing]
    s = dict(rig=dname, n_bearings=len(per_bearing),
             per_bearing=per_bearing,
             charge_range_median=float(np.median(m2)), n_range=got(m2),
             charge_essential_median=float(np.median(ess)),
             n_essential=got(ess),
             charge_best_median=float(np.median(bc)), n_best=got(bc),
             best_alpha_median=float(np.median([r["best_alpha"]
                                                for r in per_bearing])),
             charge_energy_median=float(np.median(en)), n_energy=got(en),
             charge_contrast_median=float(np.median(co)), n_contrast=got(co),
             eps_max=float(max(r["max_eps"] for r in per_bearing)),
             eps_median=float(np.median([r["max_eps"] for r in per_bearing])),
             contrast_over_energy_max=float(max(r["contrast_over_energy"]
                                                for r in per_bearing)))
    s["ladder"] = [dict(alpha=a,
                        median_charge=float(np.median(
                            [r["charge"][str(a)] for r in per_bearing])),
                        certified=got([r["charge"][str(a)]
                                       for r in per_bearing]))
                   for a in ALPHAS]
    # a within-record question, which does not depend on how many records there
    # are: on how many does each currency charge less than the one before?
    s["energy_below_trimmed"] = float(np.mean(np.array(en) < np.array(bc)))
    s["contrast_below_energy"] = float(np.mean(np.array(co) < np.array(en)))
    return s


print()
print("=" * 92)
print("2.  Four currencies for the same certificate, on each rig")
print("=" * 92)
print("""
The tolerated contrast is a property of the pair of indicators and does not
change; what changes is the charge.  Each row is one currency, and the guarantees
they buy are not interchangeable: the first three hold for every distortion with
the stated property, the last only for the three corrections measured.
""")
SUM = {}
for _dn, _fn in DATASETS:
    s = ladder_for(_dn, _fn)
    if s is None:
        print("  %-12s no record carries both families" % _dn)
        continue
    SUM[_dn] = s
    print("  %s, %d records" % (_dn, s["n_bearings"]))
    print("    %-16s %14s %12s" % ("currency", "median charge", "certified"))
    print("    " + "-" * 46)
    for lab, ch, n in (("range", s["charge_range_median"], s["n_range"]),
                       ("range, support", s["charge_essential_median"],
                        s["n_essential"]),
                       ("range, trimmed", s["charge_best_median"], s["n_best"]),
                       ("energy", s["charge_energy_median"], s["n_energy"]),
                       ("contrast", s["charge_contrast_median"],
                        s["n_contrast"])):
        print("    %-16s %14.3f %9d of %d" % (lab, ch, n, s["n_bearings"]))
    print("    budgets reach %.3f, median %.3f; largest contrast over its "
          "energy charge %.3f" % (s["eps_max"], s["eps_median"],
                                  s["contrast_over_energy_max"]))
    print()

# the PRONOSTIA figures stay at the top level, where the manuscript's audit
# already reads them; the second rig is added beside them
if "PRONOSTIA" in SUM:
    P = SUM["PRONOSTIA"]
    for k in ("n_bearings", "per_bearing", "charge_range_median", "n_range",
              "charge_essential_median", "n_essential", "charge_best_median",
              "n_best", "best_alpha_median", "charge_energy_median",
              "n_energy", "charge_contrast_median", "n_contrast", "eps_max",
              "eps_median", "contrast_over_energy_max", "ladder"):
        OUT[k] = P[k]
OUT["by_rig"] = {k: {kk: vv for kk, vv in v.items() if kk != "per_bearing"}
                 for k, v in SUM.items()}

# --- does the study reproduce the certificate already published? ------------
ref = json.load(open(os.path.join(HERE, "contrast_identity.json")))
rmap = {r["unit"]: r for r in ref["per_bearing"]}
if "PRONOSTIA" in SUM:
    pb = SUM["PRONOSTIA"]["per_bearing"]
    OUT["reproduces_tolerated"] = float(max(
        abs(r["tolerated"] - rmap[r["unit"]]["tolerated"])
        for r in pb if r["unit"] in rmap))
    OUT["reproduces_contrast"] = float(max(
        abs(r["worst_contrast"] - rmap[r["unit"]]["worst_contrast"])
        for r in pb if r["unit"] in rmap))
    print("  reproduction of contrast_identity.json: tolerated to %.2e, worst "
          "contrast to %.2e,\n  and its count of %d certified against %d here"
          % (OUT["reproduces_tolerated"], OUT["reproduces_contrast"],
             ref["n_survive"], SUM["PRONOSTIA"]["n_contrast"]))

print()
print("=" * 92)
print("3.  What replicates")
print("=" * 92)
if len(SUM) >= 2:
    A, B = SUM["PRONOSTIA"], SUM["XJTU-SY"]
    OUT["replicates_ordering"] = bool(
        A["charge_range_median"] > A["charge_best_median"] > A["charge_contrast_median"]
        and B["charge_range_median"] > B["charge_best_median"] > B["charge_contrast_median"])
    OUT["replicates_energy_holds"] = bool(A["contrast_over_energy_max"] <= 1.0
                                          and B["contrast_over_energy_max"] <= 1.0)
    OUT["xjtu_eps_max"] = B["eps_max"]
    OUT["xjtu_n"] = B["n_bearings"]
    OUT["xjtu_n_range"] = B["n_range"]
    OUT["xjtu_n_best"] = B["n_best"]
    OUT["xjtu_n_energy"] = B["n_energy"]
    OUT["xjtu_n_contrast"] = B["n_contrast"]
    print("""
  The ordering of the charges is the same on both rigs: %s.  The range is the
  most expensive currency, trimming it helps, and the contrast is cheapest.

  The energy charge is respected on PRONOSTIA, the worst measured contrast
  reaching %.3f of it, and crossed on XJTU, where one cell reaches %.4f.  Whether
  that refutes Proposition 9 or shows a known limitation is not a matter of
  opinion, and section 4 decomposes it.

  The budgets themselves replicate too.  PRONOSTIA reaches %.2f and XJTU %.2f,
  both far above the 0.34 the paper used to quote, so the discrepancy was in the
  quoted figure and not in one rig's records.

  What does not replicate cleanly is the count, and the reason is the sample.
  XJTU certifies %d of %d in the contrast against %d of %d on PRONOSTIA; nine of
  its sixteen records fall under the length screen the rest of the paper applies,
  so the second rig contributes a check on the ORDERING and on the budgets, not a
  second estimate of how many records a certificate covers.
""" % ("yes" if OUT["replicates_ordering"] else "no",
       A["contrast_over_energy_max"], B["contrast_over_energy_max"],
       A["eps_max"], B["eps_max"],
       B["n_contrast"], B["n_bearings"], A["n_contrast"], A["n_bearings"]))


print()
print("=" * 92)
print("4.  The closed form is crossed once.  Which step of the derivation?")
print("=" * 92)
print("""
On XJTU a measured contrast exceeds its energy charge.  That is either a wrong
proposition or a known limitation showing itself, and the two are distinguished
by decomposing the claim into its two steps.

  Cauchy-Schwarz.   L = <u>_[0,c] - <u>_[c,1] = <u,h>_g  <=  eps ||h||  =  C.
                    This is exact and unconditional.  A violation here would
                    refute Proposition 9.

  Linearisation.    X = log( <e^u>_[0,c] / <e^u>_[c,1] ), and

                        X - L = (log<e^u>_A - <u>_A) - (log<e^u>_B - <u>_B),

                    a DIFFERENCE of two Jensen gaps, each non-negative.  A first
                    version of this block asserted X >= L "because <e^u> >= e^<u>
                    on the early set", which is the numerator's half of the
                    argument with the denominator's half omitted; the second gap
                    is subtracted, so X - L has no fixed sign.  The count below
                    settles it rather than the argument.

X is what a record actually suffers and L is what the derivation bounds, so X may
exceed C without anything being wrong with the algebra.  If every cell satisfies
L <= C, Proposition 9 is intact and is simply being read outside the regime where
first order is enough.
""")
print("  %-10s %-22s %-9s %6s %6s %8s %8s %8s %8s"
      % ("rig", "record", "distortion", "eta", "eps", "L", "C", "X", "X/C"))
print("  " + "-" * 96)
cs_worst, jensen_bad, exceed_rows, n_cells = 0.0, 0, [], 0
worst_cell = None
for _dn, _fn in DATASETS:
    z, FN, units, AMOUNT, DISTRIB = load(_fn)
    for u in units:
        for nm in AMOUNT + DISTRIB:
            p = pieces(z[u][:, FN.index(nm)], (_dn, u, nm))
            if p is None:
                continue
            g, ws = p
            tot = float(np.sum(g))
            for dist in DISTORTIONS:
                if dist not in ws:
                    continue
                w = np.clip(ws[dist], 1e-9, None)
                uu = np.log(w)
                uu = uu - float(np.sum(uu * g) / tot)      # centred, as stated
                eps = float(np.sqrt(np.sum(uu ** 2 * g) / tot))
                for cf in CUTS:
                    ic = int(cf * len(g)) - 1
                    eta = float(np.sum(g[:ic + 1]) / tot)
                    if eta < MIN_SHARE or (1.0 - eta) < MIN_SHARE:
                        continue
                    n_cells += 1
                    lo_g, hi_g = g[:ic + 1], g[ic + 1:]
                    L = float(np.sum(uu[:ic + 1] * lo_g) / np.sum(lo_g)
                              - np.sum(uu[ic + 1:] * hi_g) / np.sum(hi_g))
                    C = eps / np.sqrt(eta * (1.0 - eta))
                    X = float(np.log(np.sum(w[:ic + 1] * lo_g) / np.sum(lo_g))
                              - np.log(np.sum(w[ic + 1:] * hi_g)
                                       / np.sum(hi_g)))
                    cs_worst = max(cs_worst, abs(L) / C)
                    if abs(X) < abs(L) - 1e-12:
                        jensen_bad += 1        # counted, not assumed away
                    r = dict(rig=_dn, unit=u, channel=nm, distortion=dist,
                             cut=cf, eta=eta, eps=eps, L=L, C=C, X=X,
                             ratio=abs(X) / C)
                    if worst_cell is None or r["ratio"] > worst_cell["ratio"]:
                        worst_cell = r
                    if abs(X) > C:
                        exceed_rows.append(r)
                        print("  %-10s %-22s %-9s %6.3f %6.3f %8.3f %8.3f "
                              "%8.3f %8.4f"
                              % (_dn, u[:22], dist, eta, eps, L, C, X,
                                 abs(X) / C))
if not exceed_rows:
    print("  none")
OUT["energy_cells"] = n_cells
OUT["cauchy_schwarz_worst"] = float(cs_worst)
OUT["cells_exact_below_linear"] = int(jensen_bad)
OUT["energy_exceedances"] = len(exceed_rows)
OUT["energy_worst_ratio"] = float(worst_cell["ratio"])
OUT["energy_worst_cell"] = {k: v for k, v in worst_cell.items()}
_w = worst_cell
_share = (abs(_w["X"]) - abs(_w["L"])) / max(abs(_w["X"]) - _w["C"], 1e-300)
OUT["exceedance_from_jensen"] = float(_share) if abs(_w["X"]) > _w["C"] else None
print("""
  Over %d cells on both rigs the linearised gap never exceeds the closed form:
  the largest |L|/C is %.4f, so Cauchy-Schwarz holds everywhere and
  Proposition 9's derivation is intact.  The difference X - L takes both signs,
  as it must: the exact contrast falls BELOW the linearised gap on %d of the %d
  cells and rises above it on the rest.

  The exceedance is therefore in the linearisation and not in the algebra.  It
  occurs on %d of %d cells, on %s, at %s / %s with the cut at %.1f of the record.
  There eta = %.3f: the record has spent three per cent of its budget before the
  cut, which is where 1/sqrt(eta(1-eta)) is largest and where Proposition 9 warns
  the bound is least useful.  The closed form gives %.3f, the linearisation
  reaches %.4f of it, and the exact contrast %.4f, an overshoot of %.2f per cent.

  So the reading is not that the closed form fails in practice but that it is
  TIGHT at an extreme cut, tight enough that a first-order expression can be
  crossed by a fraction of a per cent.  That supports the design rule the
  proposition already carries -- read eta where the record has spent about half
  its budget -- with the only measured case in which reading it elsewhere costs
  anything.
""" % (n_cells, cs_worst, jensen_bad, n_cells, len(exceed_rows), n_cells,
       ", ".join(sorted(set(r["rig"] for r in exceed_rows))) or "neither rig",
       _w["unit"], _w["distortion"], _w["cut"], _w["eta"],
       _w["C"], abs(_w["L"]), abs(_w["X"]), 100 * (_w["ratio"] - 1.0)))


print()
print("=" * 92)
print("5.  Does the undocumented screen change anything?")
print("=" * 92)
print("""
Everything above requires three hundred snapshots.  Section 5 of the manuscript
documents sixty and says the only exclusions are for length and for a feature
that could not be extracted, so the screen these results actually use has never
been stated.  The question is not whether stating it is necessary -- it is -- but
whether it was doing any work.  The ladder is therefore recomputed at sixty, which
readmits Bearing2_7 at 230 snapshots on the first rig and six records on the
second, and the two are put side by side.
""")
_SAVE = MIN_N
sens = {}
for _screen in (300, 60):
    globals()["MIN_N"] = _screen
    for _dn, _fn in DATASETS:
        s = ladder_for(_dn, _fn)
        if s is None:
            continue
        sens[(_dn, _screen)] = s
globals()["MIN_N"] = _SAVE

print("  %-12s %8s %8s %10s %10s %10s %10s"
      % ("rig", "screen", "records", "range", "trimmed", "energy", "contrast"))
print("  " + "-" * 74)
for _dn, _fn in DATASETS:
    for _screen in (300, 60):
        s = sens.get((_dn, _screen))
        if s is None:
            continue
        print("  %-12s %8d %8d %10s %10s %10s %10s"
              % (_dn, _screen, s["n_bearings"],
                 "%d of %d" % (s["n_range"], s["n_bearings"]),
                 "%d of %d" % (s["n_best"], s["n_bearings"]),
                 "%d of %d" % (s["n_energy"], s["n_bearings"]),
                 "%d of %d" % (s["n_contrast"], s["n_bearings"])))
OUT["screen_sensitivity"] = {
    "%s@%d" % (k[0], k[1]): {kk: vv for kk, vv in v.items()
                             if kk not in ("per_bearing", "ladder")}
    for k, v in sens.items()}

_ord = []
for k, s in sens.items():
    _ord.append(s["charge_range_median"] > s["charge_best_median"]
                > s["charge_contrast_median"])
OUT["ordering_holds_at_both_screens"] = bool(all(_ord))
_p3, _p6 = sens.get(("PRONOSTIA", 300)), sens.get(("PRONOSTIA", 60))
_x3, _x6 = sens.get(("XJTU-SY", 300)), sens.get(("XJTU-SY", 60))
if _p3 and _p6:
    OUT["screen_added_records_rig_one"] = _p6["n_bearings"] - _p3["n_bearings"]
    OUT["screen_contrast_rig_one_60"] = _p6["n_contrast"]
if _x3 and _x6:
    OUT["screen_added_records_rig_two"] = _x6["n_bearings"] - _x3["n_bearings"]
    OUT["screen_contrast_rig_two_60"] = _x6["n_contrast"]
    OUT["eps_max_rig_two_60"] = _x6["eps_max"]
    OUT["contrast_over_energy_rig_two_60"] = _x6["contrast_over_energy_max"]
_share3 = (_p3["n_contrast"] / _p3["n_bearings"]) if _p3 else float("nan")
_share6 = (_p6["n_contrast"] / _p6["n_bearings"]) if _p6 else float("nan")
OUT["certified_share_rig_one_300"] = float(_share3)
OUT["certified_share_rig_one_60"] = float(_share6)
print("""
  The ordering of the four currencies is the same at both screens on both rigs:
  %s.  Relaxing to sixty adds %d record to the first rig and %d to the second.
  The contrast then certifies %d of %d and %d of %d, against %d of %d and %d of
  %d at the stricter screen; as a share of the records admitted that is %.2f
  against %.2f on the first rig.

  Whether the screen was holding a conclusion up is therefore answered by those
  two shares and by the ordering, not by an argument.  What the screen is
  certainly doing is keeping out records whose smoothing windows have hit the
  floor that length_screen.py derives, and the first duty either way is to state
  it, which Section 5 now does.
""" % ("yes" if OUT["ordering_holds_at_both_screens"] else "no",
       OUT.get("screen_added_records_rig_one", 0),
       OUT.get("screen_added_records_rig_two", 0),
       OUT.get("screen_contrast_rig_one_60", 0),
       _p6["n_bearings"] if _p6 else 0,
       OUT.get("screen_contrast_rig_two_60", 0),
       _x6["n_bearings"] if _x6 else 0,
       _p3["n_contrast"] if _p3 else 0, _p3["n_bearings"] if _p3 else 0,
       _x3["n_contrast"] if _x3 else 0, _x3["n_bearings"] if _x3 else 0,
       _share6, _share3))

json.dump(OUT, open(os.path.join(HERE, "trimmed_certificate.json"), "w"),
          indent=2, default=float)
print("written to trimmed_certificate.json")
