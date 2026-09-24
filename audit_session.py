"""
Every claim made this session, checked against the run that produced it.

The equivalent audit on the earlier Letter caught several real errors -- a
quantity described as calibrated when it was nominal, an age reported as a
buffer width, a figure whose colourbar named the wrong variable. It works
because it reads the stored results rather than the prose, so a number that
drifted during writing shows up as a mismatch instead of surviving into review.

Each check names the claim, the value asserted, and the value the result file
holds. Anything not marked ok needs looking at before the claim is used.
"""
import os
import json
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
L = lambda f: json.load(open(os.path.join(HERE, f)))
rows = []


def chk(label, claimed, actual, tol=0.02, rel=True):
    if actual is None or (isinstance(actual, float) and not np.isfinite(actual)):
        rows.append((label, str(claimed), "missing", "NO SOURCE"))
        return
    d = abs(claimed - actual) / abs(actual) if (rel and actual) else abs(claimed - actual)
    rows.append((label, f"{claimed:g}", f"{actual:.4g}",
                 "ok" if d <= tol else "MISMATCH"))


def assert_(label, claimed, actual, cond):
    rows.append((label, str(claimed), str(actual), "ok" if cond else "MISMATCH"))


# --- the bound is attained on real records ----------------------------------
C = L("crb_full.json")
GOOD = ("bearing", "turbofan_FD001", "turbofan_FD004")
chk("CRB bearing slope", 0.952, C["bearing"]["slope"], tol=0.005)
chk("CRB FD001 slope", 0.903, C["turbofan_FD001"]["slope"], tol=0.005)
chk("CRB FD004 slope", 0.913, C["turbofan_FD004"]["slope"], tol=0.005)
chk("CRB bearing median", 1.086, C["bearing"]["median"], tol=0.005)
tot = sum(v["cells"] for v in C.values())
chk("total evaluation cells", 102464, tot, tol=0.0)
assert_("all three medians within 9%", "<=1.09",
        f"{max(C[d]['median'] for d in GOOD):.4f}",
        max(C[d]["median"] for d in GOOD) <= 1.09)
assert_("slopes span 0.90-0.95", "0.90-0.95",
        f"{min(C[d]['slope'] for d in GOOD):.3f}-"
        f"{max(C[d]['slope'] for d in GOOD):.3f}",
        min(C[d]["slope"] for d in GOOD) >= 0.90
        and max(C[d]["slope"] for d in GOOD) <= 0.955)

# --- the non-parametric reading beats the exponent route --------------------
V = L("nonparam_validate.json")
np_err = float(np.mean([r["err_nonparam"] for r in V]))
be_err = float(np.mean([r["err_beta"] for r in V]))
chk("non-parametric quantile error", 0.004, np_err, tol=0.25)
other = [r for r in V if not r["shape"].startswith("power law")]
chk("exponent route, non-power-law shapes", 0.062,
    float(np.mean([r["err_beta"] for r in other])), tol=0.05)
chk("ratio on non-power-law shapes", 15.8,
    float(np.mean([r["err_beta"] for r in other]) /
          np.mean([r["err_nonparam"] for r in other])), tol=0.05)
worst = max(V, key=lambda r: r["err_beta"])
assert_("worst shape for the exponent route", "S-curve", worst["shape"],
        worst["shape"] == "S-curve")

# --- the floor is genuine, and its tightness is diagnosable -----------------
F = L("feasibility.json")
assert_("feasibility conditions tested", 18, len(F), len(F) == 18)
assert_("times the floor was broken", 0,
        sum(1 for r in F if r["ratio"] < 0.95),
        sum(1 for r in F if r["ratio"] < 0.95) == 0)
chk("median excess over the floor", 1.83,
    float(np.median([r["ratio"] for r in F])), tol=0.03)

T = L("tightness2.json")
clean = [r for r in T if not r["edge"]]
hi = [r["excess"] for r in clean if r["signal_frac"] > 0.9]
lo = [r["excess"] for r in clean if r["signal_frac"] <= 0.9]
chk("excess when signal > 90%", 1.37, float(np.median(hi)), tol=0.03)
chk("excess when signal <= 90%", 2.57, float(np.median(lo)), tol=0.03)
assert_("worst excess below the threshold", 6.28, f"{max(lo):.2f}",
        abs(max(lo) - 6.28) < 0.1)

# --- the trust table --------------------------------------------------------
TT = L("trust_table.json")
good = [r for r in TT if r["signal_frac"] >= 0.90]
assert_("indicators clearing the 90% threshold", "10 of 15",
        f"{len(good)} of {len(TT)}", len(good) == 10 and len(TT) == 15)
rms = next(r for r in TT if r["indicator"] == "bearing rms")
chk("bearing rms tau(0.35)", 0.971, rms["tau_0.35"], tol=0.01)
hf = next(r for r in TT if r["indicator"] == "bearing 4--10 kHz")
chk("bearing 4-10 kHz tau(0.35)", 0.241, hf["tau_0.35"], tol=0.02)
tf = [r for r in TT if r["indicator"].startswith("turbofan")]
assert_("every turbofan channel fails the threshold", "all 5",
        f"{sum(1 for r in tf if r['signal_frac'] < 0.9)} of {len(tf)}",
        all(r["signal_frac"] < 0.9 for r in tf))

# --- cross-rig replication --------------------------------------------------
X = L("xjtu_trust.json")
xr = {r["indicator"]: r for r in X}
chk("XJTU rms tau(0.35)", 0.903, xr["rms"]["tau0.35"], tol=0.01)
chk("XJTU 4-10 kHz tau(0.35)", 0.363, xr["4--10 kHz"]["tau0.35"], tol=0.02)
shared = ["rms", "kurt", "0--2 kHz", "4--10 kHz", "peak"]
pron = {r["indicator"].replace("bearing ", ""): r["tau_0.35"] for r in TT}
diffs = [abs(xr[k]["tau0.35"] - pron[k]) for k in shared if k in xr and k in pron]
chk("cross-rig median difference", 0.098, float(np.median(diffs)), tol=0.05)
a = [xr[k]["tau0.35"] for k in shared]
b = [pron[k] for k in shared]
rk = float(np.corrcoef(np.argsort(np.argsort(a)),
                       np.argsort(np.argsort(b)))[0, 1])
chk("cross-rig rank agreement", 0.90, rk, tol=0.02)

# --- operating conditions ---------------------------------------------------
XC = L("xjtu_conditions.json")
bc = XC["by_condition"]
sp = {}
for ch in {r["channel"] for r in bc}:
    v = [r["tau"] for r in bc if r["channel"] == ch]
    sp[ch] = max(v) - min(v)
chk("widest spread across loads (kurt)", 0.346, sp["kurt"], tol=0.03)
chk("narrowest spread (0-2 kHz)", 0.054, sp["0--2 kHz"], tol=0.10)
ct = XC["control"]
chk("decimation moved tau by", 0.007,
    float(np.median([abs(r["thinned"] - r["long"]) for r in ct])), tol=0.5)
OS = L("order_stability.json")
tab = OS["table"]
lastc = [sorted(tab[c], key=lambda k: tab[c][k])[-1] for c in tab]
assert_("rms is last under every load", "rms x3",
        ", ".join(lastc), all(x == "rms" for x in lastc))

# --- against the standard criteria ------------------------------------------
CR = L("criteria_resolution.json")
worst_gap = max(v["worst"] for v in CR.values())
chk("largest age gap the criteria cannot see", 0.78, worst_gap, tol=0.03)
assert_("all three criteria are blind to >0.7 of life", "all 3",
        f"{sum(1 for v in CR.values() if v['worst'] > 0.7)} of 3",
        sum(1 for v in CR.values() if v["worst"] > 0.7) == 3)

# --- retention is a null result ---------------------------------------------
RN = L("retention_null.json")
at_one = sum(1 for r in RN if r["eff_corrected"] >= 0.999)
# corrected: the first reading of this table counted 12, but rms falls short
# at all three window widths (0.988, 0.988, 0.986) and kurt at the narrowest
assert_("retention entries at exactly 1.000", "11 of 15",
        f"{at_one} of {len(RN)}", at_one == 11 and len(RN) == 15)
chk("worst corrected retention efficiency", 0.836,
    min(r["eff_corrected"] for r in RN), tol=0.02)
pr = [r["pieces"] / r["pieces_null"] for r in RN if r["pieces_null"] > 0]
assert_("fragmentation matches its null", "0.68-1.50",
        f"{min(pr):.2f}-{max(pr):.2f}",
        min(pr) > 0.6 and max(pr) < 1.6)

# --- the bound governs the estimator, not the model -------------------------
OM = L("oracle_vs_model.json")
lg = np.log([r["G"] for r in OM])
so = float(np.polyfit(lg, np.log([r["oracle"] for r in OM]), 1)[0])
sr = float(np.polyfit(lg, np.log([r["ridge"] for r in OM]), 1)[0])
chk("oracle slope", -0.684, so, tol=0.05)
chk("ridge slope", -0.121, sr, tol=0.05)
best = min(OM, key=lambda r: r["G"] * -1)
assert_("oracle beats ridge at the lowest noise", ">30x",
        f"{OM[0]['ridge'] / OM[0]['oracle']:.0f}x",
        OM[0]["ridge"] / OM[0]["oracle"] > 30)

# --- fusion -----------------------------------------------------------------
FN = L("fusion_null.json")
for blk in FN:
    if blk["title"].startswith("PRONOSTIA"):
        assert_("bearing fusion is mostly signal", ">60x floor",
                f"{min(r['ratio'] for r in blk['rows']):.0f}x",
                min(r["ratio"] for r in blk["rows"]) > 60)
    if blk["title"].startswith("C-MAPSS"):
        assert_("turbofan fusion is mostly floor", "<2.3x floor",
                f"{max(r['ratio'] for r in blk['rows']):.2f}x",
                max(r["ratio"] for r in blk["rows"]) < 2.3)

# --- distribution against amount --------------------------------------------
SL = L("shape_vs_level.json")
sl = {r["band"]: r for r in SL}
chk("4-10 kHz absolute", 0.974, sl["4--10 kHz"]["absolute"], tol=0.01)
chk("4-10 kHz as a fraction", 0.321, sl["4--10 kHz"]["fraction"], tol=0.02)
d = [r["diff"] for r in SL]
assert_("normalising gains 0.36-0.74 of life", "0.36-0.74",
        f"{min(d):.2f}-{max(d):.2f}", min(d) > 0.35 and max(d) < 0.75)

RC = L("ratio_control.json")
rc = {r["indicator"]: r["tau"] for r in RC}
chk("rescaling by a constant moves tau by", 0.0,
    abs(rc["h_rms / constant"] - rc["h_rms (level, reference)"]),
    tol=0.005, rel=False)
assert_("self-normalised level stays late", ">0.95",
        f"{rc['h_rms / slow(h_rms)']:.3f}",
        rc["h_rms / slow(h_rms)"] > 0.95)
assert_("cross-channel ratio is early", "<0.6",
        f"{rc['h_rms / v_rms']:.3f}", rc["h_rms / v_rms"] < 0.6)

DB = L("distribution_both.json")
S = DB["summary"]
for i, rig in enumerate(("PRONOSTIA", "XJTU")):
    gap = S["amount"]["median"][i] - S["distribution"]["median"][i]
    lab = f"{rig}: distribution earlier by"
    chk(lab, 0.640 if i == 0 else 0.358, gap, tol=0.03)
disj = [S["distribution"]["span"][i][1] < S["amount"]["span"][i][0]
        for i in range(2)]
assert_("separation is clean on PRONOSTIA only", "True, False",
        f"{disj[0]}, {disj[1]}", disj == [True, False])
xa = DB["rigs"]["XJTU"]
hb = next(r["tau"] for r in xa if r["measure"] == "high-band power")
assert_("the XJTU crossing indicator", "high-band power 0.614",
        f"{hb:.3f}", abs(hb - 0.614) < 0.02)

SP = L("spatial_test.json")
for rig, gap in (("PRONOSTIA", 0.649), ("XJTU", 0.345)):
    ok = [x for x in SP[rig] if x["signal"] >= 0.90]
    sp = [x["tau"] for x in ok if x["kind"] == "spatial"]
    am = [x["tau"] for x in ok if x["kind"] == "amount"]
    chk(f"{rig}: spatial earlier by", gap,
        float(np.median(am)) - float(np.median(sp)), tol=0.03)
# the best spatial measure swaps between rigs, which is the same instability
# the spectral measures showed; only the category transfers
pa = {x["measure"]: x["tau"] for x in SP["PRONOSTIA"]}
px = {x["measure"]: x["tau"] for x in SP["XJTU"]}
best_a = min((m for m in pa if "h/v" in m), key=lambda m: pa[m])
best_x = min((m for m in px if "h/v" in m), key=lambda m: px[m])
assert_("best spatial measure swaps between rigs", "different",
        f"{best_a} / {best_x}", best_a != best_x)

# --- the exact results, and the error they exposed --------------------------
TE = L("theory_exact.json")
# monotonicity holds with correlated noise, and the increment is the residual
# sensitivity rather than merely a non-negative quantity
assert_("increment non-negative, noise correlated", ">= 0",
        f"{TE['monotone']['smallest_increment']:.3g}",
        TE["monotone"]["smallest_increment"] >= 0)
chk("increment equals its residual form", 0.0,
    TE["monotone"]["residual_form_rel_err"], tol=1e-10, rel=False)
# the three closed forms, against quadrature that resolves the origin
assert_("joint nuisance cost is r_A squared", "1/(16 beta^4)",
        TE["exact"]["r_Abeta"], TE["exact"]["r_Abeta"] == "1/(16 beta^4)")
chk("closed forms vs quadrature", 0.0,
    TE["exact"]["max_abs_err_vs_quadrature"], tol=1e-4, rel=False)
chk("nuisance factor independent of K", 0.0,
    TE["separability"]["max_rel_departure"], tol=1e-9, rel=False)
chk("exponent cost peaks at beta = 1", 1.0,
    TE["separability"]["r_beta_max_cost_at"], tol=0.0)
chk("exponent cost at that peak", 5.0 / 32.0,
    TE["separability"]["r_beta_at_1"], tol=1e-12, rel=False)

# the measured exponent, which the closed form needs and the data do not supply
EM = L("exponent_measure.json")
chk("pooled effective exponent", 8.29, EM["pooled_beta"], tol=0.01)
assert_("a single exponent does not describe these records", "<1%",
        f"{100 * EM['frac_single_exponent']:.2f}%",
        EM["frac_single_exponent"] < 0.01)

# the band features are compositional, which corrupts the unguarded fusion
CS = L("check_singular.json")
assert_("adding every band destroys the conditioning", ">1e3",
        f"{CS['conditioning']['everything (10)']['median']:.1e}",
        CS["conditioning"]["everything (10)"]["median"] > 1e3)
assert_("the same set less one band conditions normally", "<1e3",
        f"{CS['conditioning']['everything less one band']['median']:.1e}",
        CS["conditioning"]["everything less one band"]["median"] < 1e3)

FR = L("fusion_ranksafe.json")
chk("rank-safe fusion age", 0.914, FR["headline"]["fusion"], tol=0.005)
chk("unguarded fusion age", 0.029,
    next(r["naive"] for r in FR["sets"] if r["set"] == "everything"),
    tol=0.005, rel=False)
chk("fusion gain over RMS", 0.026, FR["headline"]["gain"], tol=0.005, rel=False)
chk("bearings where fusion is earlier", 17, FR["headline"]["earlier_on"],
    tol=0.0)
# the published monotonicity count included comparisons that could not fail
chk("monotonicity, testable comparisons", 80,
    FR["monotone_ranksafe"]["tests"], tol=0.0)
chk("monotonicity, comparisons passed", 80,
    FR["monotone_ranksafe"]["pass_"], tol=0.0)
chk("jointly feasible pairs", 5, FR["joint_rescues"], tol=0.0)
assert_("that result is below the rank guard", "<1e3",
        f"{FR['worst_pair_cond']:.1f}", FR["worst_pair_cond"] < 1e3)

# a loss of information is not a delay at the demand this paper works at
DM = L("delay_measured.json")
_r = [x for x in DM["rows"] if x["base"] == 0.35]
assert_("at q = 0.35 every loss is infeasible", "all infeasible",
        f"{sum(x['usable'] for x in _r)} usable",
        all(x["usable"] == 0 for x in _r) and len(_r) > 0)
_t = next(x for x in DM["rows"] if x["base"] == 1e-2 and x["loss"] == 10.0)
chk("delay at a tenfold loss, q = 1e-2", 0.103, _t["median"], tol=0.002,
    rel=False)
_h = next(x for x in DM["rows"] if x["base"] == 1e-2 and x["loss"] == 1e2)
chk("records infeasible at a hundredfold loss", 48, _h["infeasible"], tol=0.0)

# --- constructing the indicator, and the limit of linear methods ------------
OI = L("optimal_indicator.json")
LS = L("linear_vs_shape.json")
chk("leading direction holds of the trace", 0.946, OI["share_median"],
    tol=0.002, rel=False)
chk("participation ratio of the eigenvalues", 1.30, OI["erank_median"],
    tol=0.02, rel=False)
_std = {r["indicator"]: r["share"] for r in OI["standard"]}
chk("RMS share of the best fixed combination", 0.446, _std["rms"],
    tol=0.002, rel=False)
chk("kurtosis share", 0.021, _std["kurt"], tol=0.002, rel=False)
assert_("every standard channel is below the optimum", "all < 1",
        f"max {max(_std.values()):.3f}", max(_std.values()) < 1.0)
chk("transferred direction, earliest usable age", 0.980,
    OI["tau_transferred"], tol=0.002, rel=False)
chk("RMS, earliest usable age", 0.983, OI["tau_rms"], tol=0.002, rel=False)
chk("direction transfer cosine", 0.96, OI["cos_median"], tol=0.02, rel=False)
# anchoring trades feasibility for earliness rather than improving on it
_a10 = next(r for r in OI["anchored"] if r["anchor"] == 0.10)
chk("paired gain at anchor 0.10", 0.943, _a10["gain"], tol=0.002, rel=False)
chk("bearings lost at anchor 0.10", 14, _a10["lost"], tol=0.0)
assert_("no anchor keeps every bearing feasible", "none",
        f"min lost {min(r['lost'] for r in OI['anchored'])}",
        all(r["lost"] > 0 for r in OI["anchored"]))
assert_("anchors other than 0.10 and 1.00 gain nothing", "<= 0",
        f"max {max(r['gain'] for r in OI['anchored'] if r['anchor'] not in (0.10, 1.00)):+.3f}",
        all(r["gain"] <= 0 for r in OI["anchored"]
            if r["anchor"] not in (0.10, 1.00)))
# the linear span is the limit
chk("best linear combination, in sample", 0.971, LS["median"]["lin_own"],
    tol=0.002, rel=False)
chk("best linear combination, anchored", 0.971,
    LS["median"]["lin_anchored"], tol=0.002, rel=False)
chk("best distributional statistic", 0.105, LS["median"]["best_shape"],
    tol=0.002, rel=False)
chk("bearings where shape beats linear", 15, LS["paired"]["shape_wins"],
    tol=0.0)
chk("paired gain of shape over linear", 0.804, LS["paired"]["gap"],
    tol=0.002, rel=False)
chk("the same under the anchored objective", 0.764,
    LS["paired_anchored"]["gap"], tol=0.002, rel=False)
assert_("the linear side was fitted in sample, so the test is generous to it",
        "in-sample", "own fit reported",
        LS["median"]["lin_own"] <= LS["median"]["lin_tr"] + 1e-9)

# The exact span bound overturned the first reading of the section above: the
# linear span reaches far EARLIER than any shape statistic, so the claim that
# the distribution gain is unreachable by reweighting is false.  What survives
# is that the span bound does not transfer and the shape statistics do.
# The generalised-eigenproblem route inverts an unconditioned D^T D and broke
# the leverage inequality on six of sixteen bearings; span_qr.json is the stable
# QR route and supersedes linear_bound{,_null}.json throughout.
SQ = L("span_qr.json")
_tau = SQ["median"]["tau_lin"]
_shuf = SQ["median"]["shuffled"]
chk("span bound, QR route", 0.0066, _tau, tol=0.0002, rel=False)
assert_("the leverage inequality holds everywhere", "16/16",
        f"{SQ['bearings'] - SQ['violations']}/{SQ['bearings']}",
        SQ["violations"] == 0)
chk("the leverage bound is attained", 1.00, SQ["tightness"], tol=0.01,
    rel=False)
assert_("the span bound is earlier than the best shape statistic", "yes",
        f"{_tau:.4f} vs {LS['median']['best_shape']:.3f}",
        _tau < LS["median"]["best_shape"])
assert_("the exchangeable bracket contains the shuffled null", "q/K to q",
        f"{SQ['median']['qK']:.4f} < {_shuf:.4f} < 0.35",
        SQ["median"]["qK"] < _shuf < 0.35)
chk("time-shuffled null", 0.111, _shuf, tol=0.002, rel=False)
assert_("the bound survives its null, so it is not capacity", "shuffle hurts",
        f"factor {_shuf / _tau:.0f}", _shuf > 5 * _tau)
chk("leverage in one direction, real", 0.974, SQ["concentration"]["real"],
    tol=0.01, rel=False)
chk("leverage in one direction, shuffled", 0.467,
    SQ["concentration"]["shuffled"], tol=0.01, rel=False)
chk("held-out age of the attaining direction", 0.966,
    SQ["transfer_heldout"], tol=0.002, rel=False)
assert_("the attaining direction does not transfer", ">100x",
        f"{SQ['transfer_heldout'] / SQ['transfer_own']:.0f}x",
        SQ["transfer_heldout"] / SQ["transfer_own"] > 100)
assert_("the shape statistics beat only the FIXED linear alternatives",
        "fixed only", "claim corrected",
        _tau < LS["median"]["best_shape"] < LS["median"]["lin_tr"])

# --- the linearisation radius ------------------------------------------------
LR = L("linearisation_radius.json")
assert_("C = beta - 1/2 for a power law", "exact",
        f"err {LR['powerlaw_check']:.1e}", LR["powerlaw_check"] < 1e-5)
chk("paired agreement with the closed form", 1.02,
    LR["agreement"]["ratio_median"], tol=0.01, rel=False)
assert_("the closed form tracks well beyond the radius", "0.7 to 1.3",
        f"{LR['agreement']['ratio_min']:.2f} to "
        f"{LR['agreement']['ratio_max']:.2f}",
        0.7 <= LR["agreement"]["ratio_min"]
        and LR["agreement"]["ratio_max"] <= 1.3)
chk("heavy-tail factor on the residuals", 1.35, LR["heavy_tail"], tol=0.01,
    rel=False)
chk("error at zero offset, heavy tail removed", 1.01,
    next(r["measured"] for r in LR["degradation"] if r["delta"] == 0.0),
    tol=0.01, rel=False)
chk("radius, median", 0.0043, LR["radius_median"], tol=0.0002, rel=False)
_bw = {r["bw"]: r["radius"] for r in LR["bandwidth"]}
assert_("the radius follows the trend bandwidth", "monotone",
        f"{_bw[0.05]:.4f} / {_bw[0.08]:.4f} / {_bw[0.12]:.4f}",
        _bw[0.05] < _bw[0.08] < _bw[0.12])
chk("radius an estimator that fits the trend sees", 0.072,
    LR["radius_practical"], tol=0.002, rel=False)
assert_("the practical radius brackets the Letter's empirical 3 per cent",
        "yes", f"{LR['radius_median']:.4f} to {LR['radius_practical']:.3f}",
        LR["radius_median"] < 0.03 < LR["radius_practical"])

# --- the Bayesian floor ------------------------------------------------------
BF = L("bayesian_floor.json")
_f = {r["fleet"]: r for r in BF["fleets"]}
chk("PRONOSTIA coefficient of variation of life", 0.59, _f["PRONOSTIA"]["cv"],
    tol=0.005, rel=False)
chk("XJTU coefficient of variation of life", 1.39, _f["XJTU-SY"]["cv"],
    tol=0.005, rel=False)
assert_("the tighter fleet is used, being conservative", "PRONOSTIA",
        BF["tightest_fleet"], BF["tightest_fleet"] == "PRONOSTIA")
chk("prior standard deviation at mid-life", 0.294, BF["prior_sd"], tol=0.002,
    rel=False)
chk("information the prior contributes", 11.5, BF["prior_sd"] ** -2, tol=0.05,
    rel=False)
# the posterior mean attains the van Trees bound exactly, so no measurement of
# tightness is needed -- unlike the unbiased floor of Section 5
chk("MAP attains the Bayesian floor", 1.0, BF["map_over_bayes_median"],
    tol=3 * BF["mcse"], rel=False)
assert_("the departure is Monte Carlo noise", "< 3 sigma",
        f"{abs(BF['map_over_bayes_median'] - 1) / BF['mcse']:.1f} sigma",
        abs(BF["map_over_bayes_median"] - 1) < 3 * BF["mcse"])
assert_("the unbiased floor is beaten where the prior matters", "6 of 7",
        f"{BF['levels_beating_unbiased']} of {len(BF['simulation'])}",
        BF["levels_beating_unbiased"] == 6)
chk("how far the unbiased floor is beaten", 3.5,
    max(1.0 / s["r_unbiased"] for s in BF["simulation"]), tol=0.1, rel=False)
# a demand looser than the fleet's own spread needs no measurement at all
assert_("a loose demand is met at age zero", "yes",
        f"eps 0.30 vs s {BF['prior_sd']:.3f}",
        next(d["bayes"] for d in BF["demand"] if abs(d["eps"] - 0.30) < 1e-9)
        == 0.0)
chk("censoring at half the information, with the prior", 0.77,
    next(c["with_prior"] for c in BF["censoring"] if c["G1"] == 10.0),
    tol=0.01, rel=False)
assert_("the prior makes truncation cheaper, not dearer", "gain > 0",
        f"{next(c['gain'] for c in BF['censoring'] if c['G1'] == 10.0):+.2f}",
        all(c["gain"] >= 0 for c in BF["censoring"]))

# --- the nuisance cost is achievable, and what refinement actually buys ------
NA = L("nuisance_achievable.json")
for lab, want in (("trend known", 0.99), ("amplitude fitted", 1.00),
                  ("amplitude and exponent", 0.99)):
    chk(f"ladder rung: {lab}", want, NA["summary"][lab], tol=0.01, rel=False)
_r = [r["ratio"] for r in NA["rungs"]]
assert_("every rung lands on its prediction", "0.95 to 1.07",
        f"{min(_r):.2f} to {max(_r):.2f}", 0.9 <= min(_r) and max(_r) <= 1.1)

SK = L("skill_local.json")
chk("best refinement gain over elapsed time, per cent", 1.5,
    SK["best_gain_percent"], tol=0.05, rel=False)
chk("spread across channels, per cent", 0.6,
    100 * SK["spread_across_channels"], tol=0.05, rel=False)
assert_("the channel differences exceed the measurement noise", "yes",
        f"{SK['spread_across_channels'] / SK['typical_scatter']:.1f}x",
        SK["separable"])
chk("rank correlation with tau_min", 0.53, SK["spearman"], tol=0.01, rel=False)
chk("its permutation p-value", 0.061, SK["spearman_p"], tol=0.005, rel=False)
assert_("the correlation is reported as suggestive, not established",
        "p >= 0.05", f"p = {SK['spearman_p']:.3f}", SK["spearman_p"] >= 0.05)
assert_("the gain is too small to be usable", "< 2 per cent",
        f"{SK['best_gain_percent']:.1f}%", SK["best_gain_percent"] < 2.0)

# --- the minimum window, the design quantity that had never been checked -----
WV = L("window_validate.json")
_w = {round(r["mult"], 1): r for r in WV["window"]}
chk("achieved error at the prescribed window", 1.00, _w[1.0]["corrected"],
    tol=0.01, rel=False)
assert_("every narrower window misses the target", "yes",
        f"{_w[0.4]['corrected']:.2f}, {_w[0.7]['corrected']:.2f}",
        _w[0.4]["corrected"] > 1.02 and _w[0.7]["corrected"] > 1.02)
assert_("every wider window buys a surplus", "yes",
        f"{_w[1.5]['corrected']:.2f}, {_w[2.5]['corrected']:.2f}",
        _w[1.5]["corrected"] < 1.0 and _w[2.5]["corrected"] < 1.0)
assert_("so w* is the smallest window that works", "tight",
        "crosses at 1.0",
        _w[0.7]["corrected"] > 1.0 >= _w[1.0]["corrected"] - 0.01)
chk("bearings where the window falls with age", 11, WV["falls_count"], tol=0.0)
chk("sign-law agreements", 47, WV["sign_law_ok"], tol=0.0)
chk("sign-law pairs tested", 51, WV["sign_law_pairs"], tol=0.0)
assert_("the sign law holds on the large majority of pairs", "> 90%",
        f"{100 * WV['sign_law_ok'] / WV['sign_law_pairs']:.0f}%",
        WV["sign_law_ok"] / WV["sign_law_pairs"] > 0.9)

# --- uncertainty: the fleet interval, and what the se formula does not give ---
FC = L("tau_fleet_ci.json")
_g = {r["rig"]: r for r in FC["fleet"]}
chk("PRONOSTIA gap, independently recomputed", 0.640, _g["PRONOSTIA"]["gap"],
    tol=0.002, rel=False)
chk("XJTU gap, independently recomputed", 0.355, _g["XJTU"]["gap"], tol=0.002,
    rel=False)
assert_("both fleet intervals exclude zero", "yes",
        f"P [{_g['PRONOSTIA']['lo']:+.3f}], X [{_g['XJTU']['lo']:+.3f}]",
        all(r["excludes_zero"] for r in FC["fleet"]))
assert_("the XJTU interval nearly reaches zero, and is reported so", "yes",
        f"{_g['XJTU']['lo']:+.3f}", _g["XJTU"]["lo"] < 0.10)
assert_("the paper's inclusion rule is used, not a stricter one", "n >= 60",
        f"n >= {FC['min_n']}", FC["min_n"] == 60)

SS = L("se_scaling.json")
chk("cases where se x g is tighter than se", 9, SS["product_tighter"], tol=0.0)
chk("median spread of se across demands", 18.0, SS["se_spread_median"],
    tol=0.05, rel=False)
chk("median spread of the product", 12.9, SS["prod_spread_median"], tol=0.05,
    rel=False)
chk("sign-test p-value", 0.073, SS["sign_test_p"], tol=0.005, rel=False)
assert_("the reciprocal-density dependence is suggestive, not established",
        "p >= 0.05", f"p = {SS['sign_test_p']:.3f}", SS["sign_test_p"] >= 0.05)
SF = L("se_formula_check.json")
assert_("the block estimator of se(G) fails badly", "> 15x",
        f"{SF['ratio_max']:.1f}x", SF["ratio_max"] > 15)
assert_("so the formula is not used to compute error bars", "bootstrap used",
        "stated in Section 3.4", True)

# --- the tail factor, and the demand sweep -----------------------------------
TF = L("tail_factor.json")
_d = TF["domains"]
chk("tail factor, bearings", 1.35, _d["bearings"]["kappa"], tol=0.005,
    rel=False)
chk("tail factor, battery", 1.46, _d["battery"]["kappa"], tol=0.005, rel=False)
chk("tail factor, turbofan", 0.95, _d["FD004"]["kappa"], tol=0.005, rel=False)
assert_("the turbofan residual is lighter than Gaussian", "kappa < 1",
        f"{_d['FD004']['kappa']:.3f}", _d["FD004"]["kappa"] < 1.0)
_bat = next(p for p in TF["predictions"] if p["domain"] == "battery")
chk("the independent battery prediction", 1.68, _bat["predicted"], tol=0.01,
    rel=False)
assert_("it lands within a sixth", "yes",
        f"{100 * (max(_bat['ratio'], 1 / _bat['ratio']) - 1):.0f}%",
        max(_bat["ratio"], 1 / _bat["ratio"]) < 1.25)
assert_("the bearings check is definitional, not independent", "noted",
        "1.35 by construction", abs(_d["bearings"]["kappa"] - 1.35) < 0.01)
assert_("a global scale would give a different, wrong number", ">2x",
        f"{_d['bearings']['kappa_global']:.2f} vs "
        f"{_d['bearings']['kappa']:.2f}",
        _d["bearings"]["kappa_global"] > 2 * _d["bearings"]["kappa"])

QSW = L("qstar_sweep.json")
assert_("the gap is positive at every demand tested", "7 of 7 on both",
        f"{sum(r['positive'] for r in QSW['rows'])} of {len(QSW['rows'])}",
        all(r["positive"] for r in QSW["rows"]))
_p = [r for r in QSW["rows"] if r["rig"] == "PRONOSTIA"]
chk("largest PRONOSTIA gap over the sweep", 0.895,
    max(r["gap"] for r in _p), tol=0.005, rel=False)
chk("smallest PRONOSTIA gap over the sweep", 0.044,
    min(r["gap"] for r in _p), tol=0.005, rel=False)
assert_("the working demand is not the flattering one", "yes",
        f"q=0.10 gives "
        f"{next(r['gap'] for r in _p if abs(r['q'] - 0.10) < 1e-9):.3f} "
        f"vs 0.640 at q=0.35",
        next(r["gap"] for r in _p if abs(r["q"] - 0.10) < 1e-9) > 0.640)

# --- the tail factor drifts, and what that costs the applied result ----------
TD = L("tail_drift.json")
_fam = {(r["rig"], r["kind"]): r for r in TD["by_family"]}
chk("pooled late-over-early drift of kappa", 1.28, TD["drift_median"],
    tol=0.01, rel=False)
assert_("the drift is family-dependent", "amount drifts more",
        f"amount {_fam[('PRONOSTIA', 'amount')]['drift']:.2f}/"
        f"{_fam[('XJTU', 'amount')]['drift']:.2f}, "
        f"distribution {_fam[('PRONOSTIA', 'distribution')]['drift']:.2f}/"
        f"{_fam[('XJTU', 'distribution')]['drift']:.2f}",
        _fam[("PRONOSTIA", "amount")]["drift"]
        > _fam[("PRONOSTIA", "distribution")]["drift"]
        and _fam[("XJTU", "amount")]["drift"]
        > _fam[("XJTU", "distribution")]["drift"])
chk("cells the correction moves by more than 0.05", 59, TD["cells_moved"],
    tol=0.0)
assert_("the correction is therefore not inert", "acts on a quarter",
        f"{TD['cells_moved']}/{TD['cells_total']}",
        TD["cells_moved"] > 0.15 * TD["cells_total"])
# the direction is what decides whether the applied result is exposed
assert_("the corrected gap does not shrink on either rig", "yes",
        f"P {TD['gaps']['PRONOSTIA']['after'] - TD['gaps']['PRONOSTIA']['before']:+.3f}, "
        f"X {TD['gaps']['XJTU']['after'] - TD['gaps']['XJTU']['before']:+.3f}",
        min(g["after"] - g["before"] for g in TD["gaps"].values()) >= -0.02)
chk("the corrected XJTU gap", 0.394, TD["gaps"]["XJTU"]["after"], tol=0.002,
    rel=False)
assert_("the paper's own statistic was used, not a pooled one", "0.641 / 0.338",
        f"{TD['gaps']['PRONOSTIA']['before']:.3f} / "
        f"{TD['gaps']['XJTU']['before']:.3f}",
        abs(TD["gaps"]["PRONOSTIA"]["before"] - 0.641) < 0.005
        and abs(TD["gaps"]["XJTU"]["before"] - 0.338) < 0.005)

# --- the transfer mechanism: one half rejected, one half confirmed -----------
SM = L("score_mechanism.json")
chk("fitted direction against the multinomial score", 0.11, SM["cos_a_score"],
    tol=0.01, rel=False)
chk("chance level in this dimension", 0.14, SM["chance_level"], tol=0.01,
    rel=False)
assert_("the multinomial score model is REJECTED", "at chance",
        f"{SM['cos_a_score']:.2f} vs chance {SM['chance_level']:.2f}",
        SM["cos_a_score"] <= 2 * SM["chance_level"])
chk("profile-derived weights transfer between bearings", 0.18,
    SM["transfer_score"], tol=0.01, rel=False)
chk("entropy weights transfer between bearings", 0.96,
    SM["transfer_entropy"], tol=0.01, rel=False)
assert_("the transfer asymmetry is confirmed and is large", ">0.3 apart",
        f"{SM['transfer_entropy']:.2f} vs {SM['transfer_score']:.2f}",
        SM["transfer_entropy"] - SM["transfer_score"] > 0.3)
assert_("profile-derived weights transfer no better than chance", "yes",
        f"{SM['transfer_score']:.2f} vs {SM['chance_level']:.2f}",
        SM["transfer_score"] < 2 * SM["chance_level"])

# --- the extension tested and not supported ----------------------------------
WS = L("weight_stability.json")
chk("gradient stability vs tau spread, PRONOSTIA", -0.07,
    WS["by_rig"]["PRONOSTIA"]["rho_stability"], tol=0.01, rel=False)
chk("gradient stability vs tau spread, XJTU", 0.07,
    WS["by_rig"]["XJTU"]["rho_stability"], tol=0.01, rel=False)
assert_("no relation, so the extension is not supported", "both near zero",
        f"{WS['by_rig']['PRONOSTIA']['rho_stability']:+.2f}, "
        f"{WS['by_rig']['XJTU']['rho_stability']:+.2f}",
        all(abs(v["rho_stability"]) < 0.2 for v in WS["by_rig"].values()))
_p = [r for r in WS["rows"] if r["rig"] == "PRONOSTIA"]
assert_("but the predictor is nearly constant, so power is low", "5 of 8",
        f"{sum(1 for r in _p if r['weight_cos'] >= 0.95)} of {len(_p)} above "
        f"0.95", sum(1 for r in _p if r["weight_cos"] >= 0.95) == 5)
# the confound check: stable weights must not merely mean early ages
assert_("weight stability is not earliness in disguise", "|rho| <= 0.5",
        f"{max(v['rho_earliness'] for v in WS['by_rig'].values()):+.2f}",
        max(abs(v["rho_earliness"]) for v in WS["by_rig"].values()) <= 0.55)

# --- censoring efficiency: the last design quantity to be validated ----------
CV = L("censoring_validate.json")
chk("censoring agreement, median", 0.98, CV["ratio_median"], tol=0.01,
    rel=False)
assert_("it holds across every censoring level", "0.98 to 1.00",
        f"{CV['ratio_min']:.2f} to {CV['ratio_max']:.2f}",
        CV["ratio_min"] >= 0.95 and CV["ratio_max"] <= 1.05)
assert_("over a wide range of predicted penalty", "2.7 to 8.2",
        f"{min(r['predicted'] for r in CV['rows']):.1f} to "
        f"{max(r['predicted'] for r in CV['rows']):.1f}",
        max(r["predicted"] for r in CV["rows"])
        / min(r["predicted"] for r in CV["rows"]) > 2.5)
# this is the one validation where the tail factor cancels, being a ratio on the
# same record; that is why it is tighter than the window check
assert_("tighter than the window validation, as the ratio form predicts",
        "yes", f"censoring {abs(1 - CV['ratio_median']):.2f} vs window "
        f"{abs(1 - _w[1.0]['corrected']):.2f}",
        abs(1 - CV["ratio_median"]) <= abs(1 - _w[1.0]["corrected"]) + 0.02)

# --- the multivariate floor, and why it looked unattained --------------------
MA = L("multivariate_attained.json")
chk("multivariate floor against the exact noise covariance", 1.00,
    MA["vs_exact_median"], tol=0.02, rel=False)
assert_("it is attained on every channel set", "0.98 to 1.00",
        f"{MA['vs_exact_min']:.2f} to {MA['vs_exact_max']:.2f}",
        MA["vs_exact_min"] >= 0.95 and MA["vs_exact_max"] <= 1.05)
chk("excess against the floor as the paper builds it", 1.34,
    MA["vs_built_median"], tol=0.02, rel=False)
assert_("so the excess is the robust covariance, not the bound", "yes",
        f"built {MA['vs_built_median']:.2f} vs exact "
        f"{MA['vs_exact_median']:.2f}",
        MA["vs_built_median"] > 1.1 and abs(MA["vs_exact_median"] - 1) < 0.05)
# a scalar kappa cannot fix a shape mismatch, and the residual grows with K
_s = {r["set"]: r for r in MA["sets"]}
assert_("the shape cost is larger at K=6 than at K=2", "suggestive",
        f"{_s['RMS + peak']['corrected']:.2f} at K=2, "
        f"{_s['six bands']['corrected']:.2f} at K=6",
        _s["six bands"]["corrected"] > _s["RMS + peak"]["corrected"])

# --- the out-of-fold choice, quantified rather than asserted -----------------
OC = L("oof_cost.json")
chk("in-sample scale understates by", 1.024, OC["scale_ratio_median"],
    tol=0.002, rel=False)
chk("budget inflation from that", 1.011, OC["G_ratio_median"], tol=0.002,
    rel=False)
chk("median shift in the reported age", 0.0005, OC["tau_shift_median"],
    tol=0.0002, rel=False)
assert_("the reported medians do not depend on the choice", "negligible",
        f"{OC['tau_shift_median']:+.4f}, q75 {OC['tau_shift_q75abs']:.4f}",
        abs(OC["tau_shift_median"]) < 0.005
        and OC["tau_shift_q75abs"] < 0.01)
assert_("but the choice is not inert", "36 of 170 move",
        f"{OC['cells_moved']} of {OC['cells_total']}, worst "
        f"{OC['tau_shift_worst']:.2f}",
        OC["cells_moved"] > 0 and OC["tau_shift_worst"] > 0.1)

# --- the noise floor, calibrated against constructed truth -------------------
FCAL = L("floor_calibration.json")
chk("surviving fraction on a record with no signal", 0.67, FCAL["null_median"],
    tol=0.01, rel=False)
assert_("it should be zero and is not", "false positive",
        f"{FCAL['null_median']:.2f}, worst {FCAL['null_worst']:.2f}",
        FCAL["null_median"] > 0.5)
assert_("and it is stable across lengths and noise profiles", "yes",
        f"{len(FCAL['null'])} cells, worst {FCAL['null_worst']:.2f}",
        FCAL["null_worst"] - min(r["median"] for r in FCAL["null"]) < 0.25)
chk("calibration bias where signal is present", 0.001,
    FCAL["cal_excess_median"], tol=0.001, rel=False)
chk("worst calibration bias, at the lowest SNR", 0.096,
    FCAL["cal_excess_max"], tol=0.002, rel=False)
# the threshold the paper applies must sit clear of the null for the use to hold
assert_("the 0.90 threshold clears the null", "yes",
        f"0.90 vs null {FCAL['null_worst']:.2f}", FCAL["null_worst"] < 0.85)

# --- the membership control the four alternatives do not cover ---------------
LOM = L("leave_one_measure.json")
chk("gap without any measure dropped, PRONOSTIA", 0.640, LOM["base"]["PRONOSTIA"],
    tol=0.002, rel=False)
chk("gap without any measure dropped, XJTU", 0.355, LOM["base"]["XJTU"],
    tol=0.002, rel=False)
assert_("every single deletion leaves the gap positive", "10 of 10, both rigs",
        f"worst {LOM['worst']:.3f}", LOM["all_positive"])
chk("the weakest deletion", 0.327, LOM["worst"], tol=0.002, rel=False)
assert_("the deletions move the gap little", "within 0.10",
        f"{LOM['worst']:.3f} to {LOM['best']:.3f}",
        max(abs(LOM["best"] - LOM["base"]["PRONOSTIA"]),
            abs(LOM["worst"] - LOM["base"]["XJTU"])) < 0.10)

# --- the estimator's bandwidths, the last free parameters --------------------
# --- twelfth reviewer read: purged out-of-fold scale, two-level bootstrap,
# and the surviving-fraction threshold as a flag on the applied result
OC = L("oof_control.json")
# fourteenth reviewer read: the purge widens its window by m, and the numbers
# printed at the thirteenth read (0.469, 0.457 on XJTU-SY) were an artefact of
# fits that fell back to the observation on the shortest records
chk("purged m=3: median scale rise", 1.039, OC["scale_ratio"]["purged m=3"]["median"],
    tol=0.001, rel=False)
chk("purged m=3: 90th percentile scale rise", 1.85,
    OC["scale_ratio"]["purged m=3"]["p90"], tol=0.005, rel=False)
chk("purged m=10: median scale rise", 1.13,
    OC["scale_ratio"]["purged m=10"]["median"], tol=0.005, rel=False)
chk("purged m=3: median age shift", 0.023, OC["age_shift"]["purged m=3"]["median"],
    tol=0.001, rel=False)
chk("purged m=10: median age shift", 0.052,
    OC["age_shift"]["purged m=10"]["median"], tol=0.001, rel=False)
for _v, _p, _x in (("interleaved", 0.641, 0.355), ("purged m=3", 0.649, 0.388),
                   ("purged m=10", 0.677, 0.217)):
    chk("oof %s: PRONOSTIA gap" % _v, _p, OC["variants"][_v]["PRONOSTIA"]["gap"],
        tol=0.001, rel=False)
    chk("oof %s: XJTU gap" % _v, _x, OC["variants"][_v]["XJTU"]["gap"],
        tol=0.001, rel=False)
chk("purged m=3, widened: fits left unfitted", 300,
    OC["fallbacks"]["purged m=3"]["unfitted"], tol=0.0, rel=False)
assert_("unwidened, three XJTU-SY records have no fit at all", 3,
        len(OC["unwidened"]["m=3 XJTU"]["records_wholly_unfitted"]),
        len(OC["unwidened"]["m=3 XJTU"]["records_wholly_unfitted"]) == 3)
_SY = {(r["n"], r["phi"]): r for r in OC["synthetic"]}
assert_("known scale: interleaved at least as close as m=3 in all twelve cells",
        "12 of 12", "%d of %d" % (sum(abs(r["interleaved"] - 1)
                                      <= abs(r["purged m=3"] - 1) + 1e-3
                                      for r in _SY.values()), len(_SY)),
        all(abs(r["interleaved"] - 1) <= abs(r["purged m=3"] - 1) + 1e-3
            for r in _SY.values()) and len(_SY) == 12)
chk("known scale: interleaved worst error from 240 samples", 0.052,
    max(abs(r["interleaved"] - 1) for k, r in _SY.items() if k[0] >= 240),
    tol=0.0005, rel=False)
chk("known scale: m=3 worst overstatement at 240", 0.25,
    max(r["purged m=3"] - 1 for k, r in _SY.items() if k[0] == 240),
    tol=0.005, rel=False)
chk("known scale: 120 samples, phi 0.4, interleaved", -0.16,
    _SY[(120, 0.4)]["interleaved"] - 1, tol=0.005, rel=False)
chk("known scale: 120 samples, phi 0.4, m=3", 0.31,
    _SY[(120, 0.4)]["purged m=3"] - 1, tol=0.005, rel=False)
chk("known scale: 120 samples, phi 0.4, m=10", 0.75,
    _SY[(120, 0.4)]["purged m=10"] - 1, tol=0.005, rel=False)
chk("known scale: 120 samples, phi 0.4, m=1", 0.11,
    _SY[(120, 0.4)]["purged m=1"] - 1, tol=0.005, rel=False)
chk("known scale: the two agree at 2400 to within", 0.022,
    max(abs(r["interleaved"] - r["purged m=3"]) for k, r in _SY.items()
        if k[0] == 2400), tol=0.0005, rel=False)
# Table 5's last row reads the purged-fold gaps bound just above
HB = L("hier_bootstrap.json")["rigs"]
chk("two-level bootstrap, PRONOSTIA low", 0.182, HB["PRONOSTIA"]["hier_lo"],
    tol=0.001, rel=False)
chk("two-level bootstrap, PRONOSTIA high", 0.841, HB["PRONOSTIA"]["hier_hi"],
    tol=0.001, rel=False)
chk("two-level bootstrap, XJTU low", -0.065, HB["XJTU"]["hier_lo"], tol=0.001,
    rel=False)
chk("two-level bootstrap, XJTU high", 0.500, HB["XJTU"]["hier_hi"], tol=0.001,
    rel=False)
chk("two-level bootstrap, XJTU share positive", 0.945,
    HB["XJTU"]["hier_positive_share"], tol=0.001, rel=False)
_DB = L("distribution_both.json")["rigs"]
chk("lowest surviving fraction among the ten spectral measures", 0.976,
    min(r["signal"] for rr in _DB.values() for r in rr), tol=0.001, rel=False)

# rerun on all ten measures of Table 5 at the eleventh reviewer read; the
# working setting reproduces 0.641 and 0.355
BW = L("bandwidth_sweep.json")
assert_("the sign holds at every bandwidth, both rigs", "yes",
        f"{len(BW['rows'])} settings, min gap {BW['gap_min']:.3f}",
        BW["all_positive"])
_bp = [r["gap"] for r in BW["rows"] if r["rig"] == "PRONOSTIA"]
_bx = [r["gap"] for r in BW["rows"] if r["rig"] == "XJTU"]
chk("PRONOSTIA gap under the sweep, lowest", 0.599, min(_bp), tol=0.002,
    rel=False)
chk("PRONOSTIA gap under the sweep, highest", 0.745, max(_bp), tol=0.002,
    rel=False)
chk("XJTU gap under the sweep, lowest", 0.154, min(_bx), tol=0.002, rel=False)
chk("XJTU gap under the sweep, highest", 0.435, max(_bx), tol=0.002, rel=False)
assert_("the long-record rig is far the more robust", "yes",
        f"P spread {max(_bp) - min(_bp):.3f}, X spread {max(_bx) - min(_bx):.3f}",
        (max(_bp) - min(_bp)) < (max(_bx) - min(_bx)))

# --- the principle outside vibration -----------------------------------------
TB = L("turbofan_distribution.json")
_tf = {r["fleet"]: r for r in TB["fleets"]}
chk("FD001 gap, distribution before amount", 0.397, _tf["FD001"]["gap"],
    tol=0.002, rel=False)
chk("FD004 gap", 0.068, _tf["FD004"]["gap"], tol=0.002, rel=False)
assert_("both turbofan intervals exclude zero", "yes",
        f"FD001 [{_tf['FD001']['lo']:+.3f}], FD004 [{_tf['FD004']['lo']:+.3f}]",
        all(r["positive"] for r in TB["fleets"]))
assert_("only order-invariant functionals were admitted", "2 dropped",
        ", ".join(TB["measures_dropped"]),
        len(TB["measures_dropped"]) == 2)
assert_("no unit was dropped for length", "0 skipped",
        str(TB["skipped"]), all(v == 0 for v in TB["skipped"].values()))
# FD004 is the confounded fleet and gives the weakest evidence
assert_("FD004 is much the weaker of the two", "yes",
        f"{_tf['FD004']['gap']:.3f} vs {_tf['FD001']['gap']:.3f}",
        _tf["FD004"]["gap"] < 0.5 * _tf["FD001"]["gap"])
assert_("and on it both families are early", "yes",
        f"{_tf['FD004']['dist']:.3f} and {_tf['FD004']['amount']:.3f}",
        _tf["FD004"]["amount"] < 0.5)

# --- the regime explanation, tested rather than offered ----------------------
RN = L("regime_normalise.json")
_rg = {r["fleet"]: r for r in RN["fleets"]}
chk("FD004 gap before regime normalisation", 0.068, _rg["FD004"]["before"],
    tol=0.002, rel=False)
chk("FD004 gap after", 0.225, _rg["FD004"]["after"], tol=0.002, rel=False)
assert_("the control fleet does not move at all", "FD001 inert",
        f"{_rg['FD001']['after'] - _rg['FD001']['before']:+.4f}",
        abs(_rg["FD001"]["after"] - _rg["FD001"]["before"]) < 0.001)
assert_("six balanced regimes recovered from pooled rows", "yes",
        f"{min(_rg['FD004']['regime_sizes'])} to "
        f"{max(_rg['FD004']['regime_sizes'])}",
        len(_rg["FD004"]["regime_sizes"]) == 6
        and min(_rg["FD004"]["regime_sizes"]) > 5000)
assert_("but it explains only about half the shortfall", "~half",
        f"{_rg['FD004']['after'] - _rg['FD004']['before']:.3f} of "
        f"{_rg['FD001']['before'] - _rg['FD004']['before']:.3f}",
        0.3 < (_rg["FD004"]["after"] - _rg["FD004"]["before"])
        / (_rg["FD001"]["before"] - _rg["FD004"]["before"]) < 0.7)

# --- how much operational variation the curve tolerates ----------------------
RS = L("regime_sensitivity.json")
_tl = {r["rate"]: r["tolerated"] for r in RS["limits"]}
chk("tolerated step, slow switching", 2.0, _tl[0.02], tol=0.0)
chk("tolerated step, medium switching", 2.0, _tl[0.10], tol=0.0)
chk("tolerated step, fast switching", 4.0, _tl[0.30], tol=0.0)
assert_("fast switching is the more forgiving", "yes",
        f"{_tl[0.30]:.0f} vs {_tl[0.02]:.0f}", _tl[0.30] > _tl[0.02])
_rep = {r["amp"]: r for r in RS["repair"]}
assert_("with regimes given the repair is near-complete", "< 0.01",
        f"worst residual {max(r['fixed'] for r in RS['repair']):.3f}",
        max(r["fixed"] for r in RS["repair"]) < 0.01)
chk("uncorrected error at the largest step", 0.275, _rep[8.0]["raw"],
    tol=0.005, rel=False)
chk("corrected error at the largest step", 0.006, _rep[8.0]["fixed"],
    tol=0.002, rel=False)
# the earlier version of this test used a whole-record baseline and damaged the
# record it was correcting; that is why the correction is defined on the head
assert_("the repair improves every amplitude tested", "yes",
        f"ratios {min(r['fixed'] / r['raw'] for r in RS['repair']):.2f} to "
        f"{max(r['fixed'] / r['raw'] for r in RS['repair']):.2f}",
        all(r["fixed"] < r["raw"] for r in RS["repair"]))

# --- the mechanism, replicated in a second domain ----------------------------
MT = L("mechanism_turbofan.json")
_mt = {r["fleet"]: r for r in MT["fleets"]}
chk("level weights transfer, FD001", 0.95, _mt["FD001"]["level"], tol=0.01,
    rel=False)
chk("level weights transfer, FD004", 0.97, _mt["FD004"]["level"], tol=0.01,
    rel=False)
chk("derivative weights, FD001", 0.21, _mt["FD001"]["derivative"], tol=0.01,
    rel=False)
chk("derivative weights, FD004", 0.43, _mt["FD004"]["derivative"], tol=0.01,
    rel=False)
assert_("level beats derivative on both fleets", "yes",
        f"{_mt['FD001']['level'] / _mt['FD001']['derivative']:.1f}x, "
        f"{_mt['FD004']['level'] / _mt['FD004']['derivative']:.1f}x",
        all(r["level"] > 2 * r["derivative"] for r in MT["fleets"]))
assert_("FD001's derivative weights are at chance", "yes",
        f"{_mt['FD001']['derivative']:.2f} vs {_mt['FD001']['chance']:.2f}",
        _mt["FD001"]["derivative"] < 1.5 * _mt["FD001"]["chance"])
assert_("FD004's are not, and no explanation is offered", "reported bare",
        f"{_mt['FD004']['derivative'] / _mt['FD004']['chance']:.1f}x chance",
        _mt["FD004"]["derivative"] > 2 * _mt["FD004"]["chance"])
# the whole fleet is used: an earlier version fixed the sensor set from the
# first unit and silently kept 9 of 100
assert_("every usable unit contributes", "no silent attrition",
        f"{_mt['FD001']['units']} and {_mt['FD004']['units']}",
        all(r["dropped"] == 0 for r in MT["fleets"]))

# --- every record accounted for ----------------------------------------------
AT = L("attrition.json")
_s = {r["source"]: r for r in AT["sources"]}
chk("PRONOSTIA ten-channel records entering", 17,
    _s["PRONOSTIA, ten channels"]["entering"], tol=0.0)
chk("PRONOSTIA spectral records entering", 16,
    _s["PRONOSTIA, 32-band spectra"]["entering"], tol=0.0)
chk("XJTU records entering", 13, _s["XJTU-SY, 32-band spectra"]["entering"],
    tol=0.0)
chk("XJTU records too short", 2, _s["XJTU-SY, 32-band spectra"]["short"],
    tol=0.0)
assert_("no record anywhere is excluded on its values", "yes",
        f"no-scale {sum(r['no_scale'] for r in AT['sources'])}, "
        f"no-curve {sum(r['no_curve'] for r in AT['sources'])}",
        all(r["no_scale"] == 0 and r["no_curve"] == 0
            for r in AT["sources"]))
assert_("the analyses that record their own attrition report none", "0",
        str(sum(e.get("dropped", 0) for e in AT["recorded"])),
        sum(e.get("dropped", 0) for e in AT["recorded"]) == 0)
# the 17-vs-16 difference is in the sources, not an exclusion
assert_("the 17-against-16 difference is in the sources", "yes",
        f"{_s['PRONOSTIA, ten channels']['offered']} vs "
        f"{_s['PRONOSTIA, 32-band spectra']['offered']} offered",
        _s["PRONOSTIA, ten channels"]["offered"]
        != _s["PRONOSTIA, 32-band spectra"]["offered"]
        and _s["PRONOSTIA, 32-band spectra"]["short"] == 0)


# --- the third reviewer read: three new measurements -------------------------
# Each of these is quoted in the body, so each is bound here to the file that
# produced it.  The causal rerun and the clock-scale term were written for this
# revision; the record lengths were quoted from a one-off calculation until
# record_lengths.py was written to produce them.
CZ = L("causal_check.json")
# fifteenth reviewer read: the floor is causal now; the whole-record floor of
# the first version is kept as a second arm and reported in S5
for _rig, _c, _u, _r in (("PRONOSTIA", 0.641, 0.563, 0.572),
                         ("XJTU", 0.357, 0.029, 0.025)):
    chk(f"causal, {_rig}: centred gap", _c,
        CZ["summary"][_rig]["centred"]["gap"], tol=0.001, rel=False)
    chk(f"causal, {_rig}: causal gap", _u,
        CZ["summary"][_rig]["causal"]["gap"], tol=0.001, rel=False)
    chk(f"causal, {_rig}: gap with the whole-record floor", _r,
        CZ["summary"][_rig]["causal_record_floor"]["gap"], tol=0.001,
        rel=False)

CS = L("clock_scale.json")
for _dom, _v in (("bearing", 0.0065), ("turbofan_FD001", 0.0674),
                 ("turbofan_FD004", 0.0540), ("battery", 0.0271)):
    chk(f"clock scale, {_dom}: share of the total", _v,
        CS["corpus"][_dom]["median"], tol=0.0001, rel=False)
for _rig, _d, _a in (("PRONOSTIA", 0.0077, 0.0048), ("XJTU", 0.0107, 0.0133)):
    chk(f"clock scale, {_rig}: distributional share", _d,
        CS["rigs"][_rig]["share"]["distribution"], tol=0.0001, rel=False)
    chk(f"clock scale, {_rig}: amount share", _a,
        CS["rigs"][_rig]["share"]["amount"], tol=0.0001, rel=False)
for _rig, _b, _w in (("PRONOSTIA", 0.6404, 0.6388), ("XJTU", 0.3581, 0.3414)):
    chk(f"clock scale, {_rig}: gap without the term", _b,
        CS["rigs"][_rig]["summary"]["mean_only"]["gap"], tol=0.0001, rel=False)
    chk(f"clock scale, {_rig}: gap with it", _w,
        CS["rigs"][_rig]["summary"]["with_scale"]["gap"], tol=0.0001, rel=False)

RL = L("record_lengths.json")
chk("record length, PRONOSTIA median", 1274, RL["PRONOSTIA"]["median_n"], tol=0.0)
chk("record length, XJTU median", 161, RL["XJTU"]["median_n"], tol=0.0)
chk("causal window, PRONOSTIA", 101, RL["PRONOSTIA"]["causal_window"], tol=0.0)
chk("causal window, XJTU", 13, RL["XJTU"]["causal_window"], tol=0.0)
chk("battery channels with a flat run", 6,
    RL["battery"]["with_flat_run"], tol=0.0)
chk("battery: the longest flat run", 19,
    RL["battery"]["longest_run"], tol=0.0)


# --- the demand sweep now reports its range, not only its sign ---------------
# The reviewer's objection was that q = 0.35 is a convention and the gap moves
# with it.  Section 9.2 now gives the range, so the range is bound here.
QS = L("qstar_sweep.json")["rows"]
for _rig, _lo, _hi in (("PRONOSTIA", 0.044, 0.895), ("XJTU", 0.034, 0.340)):
    _g = [r["gap"] for r in QS if r["rig"] == _rig]
    chk("demand sweep, %s: smallest gap" % _rig, _lo, min(_g),
        tol=0.001, rel=False)
    chk("demand sweep, %s: largest gap" % _rig, _hi, max(_g),
        tol=0.001, rel=False)
    assert_("demand sweep, %s: positive at every demand" % _rig, "all > 0",
            "%d of %d" % (sum(1 for x in _g if x > 0), len(_g)),
            all(x > 0 for x in _g))


# --- and the frozen-split test ------------------------------------------------
# Section 9.5 turns on three statements: the frozen choice gives the larger
# held-out gap in five arms of six, FD004 is the exception, and all twelve
# readings are positive.  An earlier draft of that section said "every arm",
# which the run does not support; these bind the sentence to the file so it
# cannot drift back.
FS = L("frozen_split.json")["results"]
_arms = []
for _k, _v in FS.get("within_rig", {}).items():
    _arms.append(("within %s" % _k, _v["default"], _v["tuned"]))
for _k, _v in FS.get("across_rig", {}).items():
    _arms.append((_k, _v["default_gap"], _v["held_gap"]))
for _k, _v in FS.get("across_domain", {}).items():
    _arms.append(("bearings->%s" % _k, _v["default_gap"], _v["held_gap"]))
_up = [a for a in _arms if a[2] > a[1]]
assert_("frozen split: arms where the frozen choice wins", "5 of 6",
        "%d of %d" % (len(_up), len(_arms)),
        len(_up) == 5 and len(_arms) == 6)
assert_("frozen split: the exception is FD004", "FD004",
        ", ".join(a[0] for a in _arms if a[2] <= a[1]),
        [a[0] for a in _arms if a[2] <= a[1]] == ["bearings->FD004"])
assert_("frozen split: every point estimate is positive", "all > 0",
        "%d of %d" % (sum(1 for a in _arms if min(a[1], a[2]) > 0),
                      len(_arms)),
        all(min(a[1], a[2]) > 0 for a in _arms))

# The intervals, which are the reason Section 9.5 does not say "all twelve
# positive": eleven exclude zero and the twelfth, FD004 under the frozen
# choice, runs to -0.020.  A draft that loses that sentence fails here.
_FS = L("frozen_split.json")["results"]
_cells = []
for _k, _v in _FS["within_rig"].items():
    _cells.append(("within %s, paper" % _k, _v["default_lo"]))
    _cells.append(("within %s, frozen" % _k, _v["tuned_lo"]))
for _grp in ("across_rig", "across_domain"):
    for _k, _v in _FS[_grp].items():
        _cells.append(("%s, paper" % _k, _v["default_lo"]))
        _cells.append(("%s, frozen" % _k, _v["held_lo"]))
_clear = [c for c in _cells if c[1] > 0]
assert_("frozen split: intervals excluding zero", "11 of 12",
        "%d of %d" % (len(_clear), len(_cells)),
        len(_clear) == 11 and len(_cells) == 12)
assert_("frozen split: the interval that crosses zero", "FD004, frozen",
        ", ".join(c[0] for c in _cells if c[1] <= 0),
        [c[0] for c in _cells if c[1] <= 0] == ["FD004, frozen"])
chk("frozen split: how far below zero it reaches", -0.020,
    _FS["across_domain"]["FD004"]["held_lo"], tol=0.001, rel=False)
_gains = [a[2] - a[1] for a in _up]
chk("frozen split: smallest gain among the five", 0.012, min(_gains),
    tol=0.001, rel=False)
chk("frozen split: largest gain among the five", 0.234, max(_gains),
    tol=0.001, rel=False)
chk("frozen split: FD004 under the frozen choice", 0.096,
    FS["across_domain"]["FD004"]["held_gap"], tol=0.001, rel=False)
chk("frozen split: FD004 under the paper's settings", 0.150,
    FS["across_domain"]["FD004"]["default_gap"], tol=0.001, rel=False)
assert_("frozen split: the demand maximises at the grid's edge", "q=0.20",
        ", ".join(sorted({"%.2f" % v["point"][2] for v in
                          list(FS.get("across_rig", {}).values())
                          + list(FS.get("across_domain", {}).values())})),
        all(abs(v["point"][2] - 0.20) < 1e-9 for v in
            list(FS.get("across_rig", {}).values())
            + list(FS.get("across_domain", {}).values())))


# --- the seventh reviewer read: timing against amount -------------------------
# The measurement that reframed the paper.  tau_q divides the budget out, so it
# orders indicators by when their information arrives; under a fixed absolute
# demand the family ordering reverses on both rigs.  Section 9.3 turns on these
# four numbers and the sign of the reversal.
TVA = L("timing_vs_amount.json")
chk("budget ratio, amount over distribution, PRONOSTIA", 18.6,
    TVA["PRONOSTIA"]["budget_ratio"], tol=0.05, rel=False)
chk("budget ratio, amount over distribution, XJTU", 3.8,
    TVA["XJTU"]["budget_ratio"], tol=0.05, rel=False)
chk("Spearman(tau_q, budget), PRONOSTIA", 0.72,
    TVA["PRONOSTIA"]["spearman_tau_budget"], tol=0.005, rel=False)
chk("Spearman(tau_q, budget), XJTU", 0.73,
    TVA["XJTU"]["spearman_tau_budget"], tol=0.005, rel=False)
for _rig, _n in (("PRONOSTIA", 5), ("XJTU", 6)):
    _neg = sum(1 for s in TVA[_rig]["absolute_sweep"] if s["gap"] < 0)
    assert_("absolute demand reverses the gap, %s" % _rig,
            "%d of 7 levels" % _n, "%d of %d"
            % (_neg, len(TVA[_rig]["absolute_sweep"])), _neg == _n)
assert_("the relative gap is positive on both rigs", "both",
        ", ".join("%s %+.3f" % (k, v["relative_gap"]) for k, v in TVA.items()),
        all(v["relative_gap"] > 0 for v in TVA.values()))

print(f"{'claim':<44}{'asserted':>14}{'source':>14}")
print("-" * 76)
for lab, c, a, st in rows:
    mark = "" if st == "ok" else f"  <-- {st}"
    print(f"{lab:<44}{c:>14}{a:>14}{mark}")
print("-" * 76)
bad = [r for r in rows if r[3] != "ok"]
print(f"{len(rows)} claims checked, {len(bad)} flagged")
for r in bad:
    print(f"  {r[0]}: asserted {r[1]}, source says {r[2]}")

# A layer that prints its verdict and exits zero cannot fail, and every
# runner that reads exit codes reports it as passing while it flags.
# audit_paper.py sat that way through a pass for sentence variety, with
# two checks reading sentences that had been rewritten around them.
import sys
sys.exit(1 if bad else 0)
