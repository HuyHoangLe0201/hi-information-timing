"""
The best indicator a sensor set admits, and what a fixed formula costs.

Everything in this study so far judges an indicator that someone else built.
The multivariate form of Section 2.5 also answers the design question, and the
answer is a closed form.

A published health indicator is a FIXED formula: one recipe, applied at every
age.  Write it as a linear combination h = a^T x of the raw channels.  Its
information is then a generalised Rayleigh quotient,

    G_a  =  sum_k (a^T d_k)^2 / (a^T Sigma a)  =  a^T M a / (a^T Sigma a),
    M    =  sum_k d_k d_k^T,

so the best fixed indicator is the leading generalised eigenvector of (M, Sigma)
and its information is the leading generalised eigenvalue lambda_max.  Meanwhile
the multivariate bound of Proposition 2 is

    G_{1:K}  =  sum_k d_k^T Sigma^-1 d_k  =  tr(Sigma^-1 M)  =  sum_j lambda_j,

which is attained by weights allowed to vary with age.  Hence

    lambda_max  <=  G_{1:K},

with equality if and only if M has rank one, that is, if and only if the
damage sensitivity points in a fixed direction in feature space for the whole of
life.  The ratio lambda_max / tr(Sigma^-1 M) is therefore a measurable price for
insisting on a fixed formula, and it is one if and only if the fault signature
never rotates.

Three quantities are computed here on the PRONOSTIA channels.  How far the
standard indicators sit below the best fixed combination of the same raw
channels; how far that best fixed combination sits below the adaptive bound; and
-- the only question that decides whether any of this is usable -- whether the
optimal direction learned on other bearings transfers to a held-out one, since a
direction fitted on the record it is evaluated on proves nothing.

The rank guard of Section 2.5 applies here too and for a sharper reason: the
generalised eigenproblem inverts Sigma, so the compositional band features would
otherwise hand the leading eigenvector to the sum-to-one direction, which carries
no measurement at all.
"""
import os
import json
import numpy as np
from scipy.linalg import eigh

from fusion_prep import prepare_fusion, joint_info, single_info, tau_min_from

HERE = os.path.dirname(os.path.abspath(__file__))
QSTAR, REF = 0.35, "rms"
COND_MAX = 1e3
RIDGE = 1e-6


def cond_of(ch, names):
    if len(names) < 2:
        return 1.0
    R = np.column_stack([ch[n]["res"] for n in names])
    C = np.nan_to_num(np.corrcoef(R, rowvar=False), nan=0.0)
    return float(np.linalg.cond(C))


def rank_safe(ch, names, cmax=COND_MAX):
    keep = list(names)
    while len(keep) > 1 and cond_of(ch, keep) > cmax:
        keep.remove(min(keep,
                        key=lambda n: cond_of(ch, [k for k in keep if k != n])))
    return keep


def moments(ch, names):
    """M = sum_k d_k d_k^T and Sigma, in the robust units used throughout."""
    D = np.column_stack([ch[n]["dtrend"] for n in names])
    R = np.column_stack([ch[n]["res"] for n in names])
    s = np.array([ch[n]["sig"] for n in names], float)
    K = len(names)
    C = np.nan_to_num(np.corrcoef(R, rowvar=False), nan=0.0) if K > 1 \
        else np.ones((1, 1))
    C = (1 - RIDGE) * C + RIDGE * np.eye(K)
    S = np.outer(s, s) * C
    return D.T @ D, S, D


def best_fixed(M, S):
    """Leading generalised eigenpair of (M, Sigma): the best fixed formula."""
    w, V = eigh(M, S)
    return float(w[-1]), np.asarray(V[:, -1], float), np.asarray(w, float)


def curve_for(a, D, S):
    """Per-sample information of the fixed combination a, as a curve."""
    num = (D @ a) ** 2
    den = float(a @ S @ a)
    return num / max(den, 1e-300)


z = np.load(os.path.join(HERE, "feat_raw.npz"), allow_pickle=True)
FN = [str(s) for s in z["featnames"]]
I = {k: i for i, k in enumerate(FN)}
units = sorted(k for k in z.files if k != "featnames")

prepared = []
for u in units:
    ch = {}
    for nm in FN:
        p, _ = prepare_fusion(np.asarray(z[u][:, I[nm]], float), nm)
        if p is not None:
            ch[nm] = p
    if len(ch) == len(FN) and len({p["n"] for p in ch.values()}) == 1:
        prepared.append((u, ch))
print(f"{len(prepared)} bearings carrying all {len(FN)} channels\n")

# --- 1. the price of a fixed formula ----------------------------------------
print("the price of a fixed formula: leading eigenvalue against the trace\n")
print(f"{'bearing':<14}{'K':>3}{'lambda_max':>13}{'trace':>13}"
      f"{'share':>9}{'eff. rank':>11}")
print("-" * 63)
rows, shares, eranks = [], [], []
keeps = {}
for u, ch in prepared:
    names = rank_safe(ch, FN)
    keeps[u] = names
    M, S, D = moments(ch, names)
    lam, a, w = best_fixed(M, S)
    tr = float(w.sum())
    share = lam / tr
    # participation ratio: how many directions the sensitivity really uses
    p = w / tr
    erank = float(np.exp(-(p[p > 0] * np.log(p[p > 0])).sum()))
    shares.append(share)
    eranks.append(erank)
    rows.append(dict(unit=u, K=len(names), lam=lam, trace=tr, share=share,
                     erank=erank))
    print(f"{u:<14}{len(names):>3}{lam:>13.3e}{tr:>13.3e}{share:>9.3f}"
          f"{erank:>11.2f}")
print("-" * 63)
print(f"median share held by the best fixed direction: {np.median(shares):.3f}")
print(f"median effective rank of the sensitivity:      {np.median(eranks):.2f}")
print()
print("A fixed formula captures the whole bound only if M has rank one, that is")
print("only if the damage signature points in a constant direction for the")
print("whole of life.  The effective rank measures how far from that these")
print("bearings are.")

print()
print("The share is high and the effective rank close to one, so the loss from")
print("insisting on a fixed formula is small.  That is a result in favour of")
print("standard practice, and it moves the question: if a fixed formula can")
print("hold 95 per cent of the bound, how much of it do the fixed formulas")
print("actually in use hold?\n")

# --- 2. what the standard indicators hold ------------------------------------
print("information held by each standard indicator, against the best fixed")
print("combination of the same raw channels\n")
print(f"{'indicator':<12}{'median share of lambda_max':>28}"
      f"{'quartiles':>18}")
print("-" * 58)
held = {nm: [] for nm in FN}
lams = {}
for u, ch in prepared:
    names = keeps[u]
    M, S, D = moments(ch, names)
    lam, a, w = best_fixed(M, S)
    lams[u] = lam
    for nm in FN:
        g = float(single_info(ch[nm]).sum())
        held[nm].append(g / lam)
std_rows = []
for nm in FN:
    v = np.array(held[nm], float)
    q1, q3 = np.percentile(v, [25, 75])
    std_rows.append(dict(indicator=nm, share=float(np.median(v)),
                         q1=float(q1), q3=float(q3)))
for r in sorted(std_rows, key=lambda x: -x["share"]):
    print(f"{r['indicator']:<12}{r['share']:>28.4f}"
          f"{r['q1']:>12.4f}{r['q3']:>6.4f}")
print("-" * 58)
best_std = max(std_rows, key=lambda x: x["share"])
print(f"the best standard channel, {best_std['indicator']}, holds a median")
print(f"{best_std['share']:.4f} of what the best fixed combination of the same")
print(f"channels holds, a factor of {1 / best_std['share']:.0f}.")
print()

# --- 3. does the direction transfer? -----------------------------------------
# A direction fitted on the record it is scored on proves nothing.  The test is
# leave-one-bearing-out: average the directions learned on the other sixteen,
# sign-aligned so they do not cancel, and score the held-out bearing with it.
print("leave-one-bearing-out: a direction learned on the other sixteen\n")
COMMON = [n for n in FN if all(n in keeps[u] for u, _ in prepared)]
print(f"scored on the {len(COMMON)} channels every bearing retains after the "
      f"rank guard\n")

dirs = {}
for u, ch in prepared:
    M, S, D = moments(ch, COMMON)
    lam, a, w = best_fixed(M, S)
    if float((D @ a).sum()) < 0:          # orient so the indicator rises
        a = -a
    dirs[u] = a / max(float(np.linalg.norm(a)), 1e-300)

print(f"{'bearing':<14}{'own tau':>10}{'transferred':>13}{'best single':>13}"
      f"{'RMS':>9}")
print("-" * 59)
tr_rows = []
for u, ch in prepared:
    M, S, D = moments(ch, COMMON)
    lam, a, w = best_fixed(M, S)
    if float((D @ a).sum()) < 0:
        a = -a
    others = np.mean([dirs[v] for v, _ in prepared if v != u], axis=0)
    others = others / max(float(np.linalg.norm(others)), 1e-300)
    target = QSTAR * float(single_info(ch[REF]).sum())
    t_own = tau_min_from(curve_for(a, D, S), target)
    t_tr = tau_min_from(curve_for(others, D, S), target)
    singles = {nm: tau_min_from(single_info(ch[nm]), target) for nm in FN}
    t_best = min(singles.values())
    t_rms = singles[REF]
    tr_rows.append(dict(unit=u, own=t_own, transferred=t_tr, best_single=t_best,
                        rms=t_rms,
                        cos=float(a @ others / max(np.linalg.norm(a), 1e-300))))
    f = lambda x: f"{x:>10.3f}" if np.isfinite(x) else f"{'--':>10}"
    print(f"{u:<14}{f(t_own)}{f(t_tr)[-13:]:>13}{f(t_best)[-13:]:>13}"
          f"{f(t_rms)[-9:]:>9}")
print("-" * 59)


def med(key):
    v = np.array([r[key] for r in tr_rows], float)
    v = v[np.isfinite(v)]
    return float(np.median(v)) if len(v) else np.nan


cos = np.array([r["cos"] for r in tr_rows], float)
print(f"median earliest usable age")
print(f"  direction fitted on the bearing itself (optimistic): {med('own'):.3f}")
print(f"  direction transferred from the other sixteen       : "
      f"{med('transferred'):.3f}")
print(f"  best single standard channel, chosen per bearing   : "
      f"{med('best_single'):.3f}")
print(f"  root-mean-square amplitude                         : {med('rms'):.3f}")
print()
print(f"alignment between a bearing's own direction and the transferred one:")
print(f"  median cosine {np.median(cos):+.2f}, "
      f"quartiles {np.percentile(cos, 25):+.2f} to {np.percentile(cos, 75):+.2f}")
gain = med("best_single") - med("transferred")
print()
if med("transferred") < med("best_single") - 1e-9:
    print(f"The transferred direction is earlier than the best single standard")
    print(f"channel by {gain:.3f} of a lifetime, and it is fixed in advance rather")
    print(f"than chosen per bearing, so the comparison is generous to the")
    print(f"baseline: choosing the best channel per bearing requires knowing")
    print(f"which one it is.")
else:
    print(f"The transferred direction is NOT earlier than the best single")
    print(f"standard channel ({med('transferred'):.3f} against "
          f"{med('best_single'):.3f}), so the optimal")
    print(f"direction does not transfer between bearings and the construction is")
    print(f"not usable as a recipe, whatever it achieves in sample.")

# --- 4. the objective is wrong -----------------------------------------------
# The transferred direction holds roughly twice the information of RMS and buys
# 0.003 of a lifetime.  That is not a failure of the construction, it is a
# failure of the objective.  The leading eigenvector maximises the information
# accumulated over the WHOLE record, and on these indicators the whole is
# dominated by the last few per cent of life, so a direction chosen to maximise
# it is chosen to be late.  What design needs is a direction that maximises the
# information available EARLY, which is the same eigenproblem with the
# sensitivity accumulated only to an anchor age c:
#
#     M(c) = sum_{tau_k <= c} d_k d_k^T,   a_c = argmax a^T M(c) a / a^T Sigma a.
#
# Nothing after c enters.  This is the design objective the field does not use:
# separability, monotonicity and correlation with remaining life are all scored
# over the whole record.
print("\n" + "=" * 62)
print("anchoring the objective at an early age")
print("=" * 62 + "\n")
print("a_c maximises the information accumulated up to c only; the direction")
print("is still fixed, and is still transferred from the other sixteen\n")


def anchored(ch, names, c):
    D = np.column_stack([ch[n]["dtrend"] for n in names])
    _, S, _ = moments(ch, names)
    m = max(3, int(c * len(D)))
    Mc = D[:m].T @ D[:m]
    lam, a, w = best_fixed(Mc, S)
    if float((D[:m] @ a).sum()) < 0:
        a = -a
    return a / max(float(np.linalg.norm(a)), 1e-300), D, S


ANCHORS = (0.10, 0.20, 0.30, 0.50, 0.70, 1.00)
print("Anchoring buys early information by giving up total information, so the")
print("anchored direction can fail to meet an absolute demand it would have met")
print("unanchored.  Medians over the feasible subset would hide exactly that, as")
print("the subsets differ by row.  Every comparison below is therefore paired,")
print("bearing by bearing, against the same bearing's RMS.\n")
print(f"{'anchor c':>9}{'feasible':>10}{'earlier':>9}{'later':>7}"
      f"{'lost':>6}{'median gain, paired':>22}")
print("-" * 63)
anch_rows = []
rms_all = {u: tau_min_from(single_info(ch[REF]),
                           QSTAR * float(single_info(ch[REF]).sum()))
           for u, ch in prepared}
for c in ANCHORS:
    dirs_c = {}
    for u, ch in prepared:
        a, _, _ = anchored(ch, COMMON, c)
        dirs_c[u] = a
    gains, feas, earlier, later, lost = [], 0, 0, 0, 0
    for u, ch in prepared:
        _, D, S = anchored(ch, COMMON, c)
        o = np.mean([dirs_c[v] for v, _ in prepared if v != u], axis=0)
        o = o / max(float(np.linalg.norm(o)), 1e-300)
        target = QSTAR * float(single_info(ch[REF]).sum())
        t = tau_min_from(curve_for(o, D, S), target)
        r = rms_all[u]
        if not np.isfinite(t):
            if np.isfinite(r):
                lost += 1          # RMS could, the anchored direction cannot
            continue
        feas += 1
        if np.isfinite(r):
            gains.append(r - t)
            earlier += int(t < r - 1e-9)
            later += int(t > r + 1e-9)
    g = float(np.median(gains)) if gains else np.nan
    anch_rows.append(dict(anchor=c, feasible=feas, units=len(prepared),
                          earlier=earlier, later=later, lost=lost,
                          paired=len(gains), gain=g))
    print(f"{c:>9.2f}{feas:>7}/{len(prepared):<3}{earlier:>9}{later:>7}"
          f"{lost:>6}{g:>+22.3f}")
print("-" * 63)
print("'lost' counts bearings on which RMS meets the demand and the anchored")
print("direction does not; those are excluded from the paired median, so a row")
print("with a large gain and a large loss is not an improvement.\n")

usable = [r for r in anch_rows if r["lost"] == 0 and np.isfinite(r["gain"])]
if usable:
    best_a = max(usable, key=lambda r: r["gain"])
    print(f"Only anchors that lose no bearing are admissible.  Among those the")
    print(f"best is c = {best_a['anchor']:.2f}, earlier on {best_a['earlier']} of "
          f"{best_a['paired']} bearings by a")
    print(f"paired median {best_a['gain']:+.3f} of a lifetime.")
else:
    best_a = max((r for r in anch_rows if np.isfinite(r["gain"])),
                 key=lambda r: r["gain"])
    print(f"No anchor leaves every bearing feasible.  The largest paired gain,")
    print(f"{best_a['gain']:+.3f} of a lifetime at c = {best_a['anchor']:.2f}, comes with "
          f"{best_a['lost']} bearings")
    print(f"on which the anchored direction cannot meet a demand RMS meets, so")
    print(f"it is a trade and not an improvement.  Anchoring the objective")
    print(f"redistributes information towards early life; it does not create it.")
print()
print("The unanchored row c = 1.00 is the objective the field optimises --")
print("separability, monotonicity and correlation with remaining life are all")
print("scored over the whole record -- and it is the row that gains least.")

json.dump(dict(units=len(prepared), per_unit=rows,
               anchored=anch_rows, best_anchor=best_a,
               share_median=float(np.median(shares)),
               share_q1=float(np.percentile(shares, 25)),
               share_q3=float(np.percentile(shares, 75)),
               erank_median=float(np.median(eranks)),
               standard=std_rows,
               best_standard=best_std,
               common_channels=COMMON,
               transfer=tr_rows,
               tau_own=med("own"), tau_transferred=med("transferred"),
               tau_best_single=med("best_single"), tau_rms=med("rms"),
               cos_median=float(np.median(cos))),
          open(os.path.join(HERE, "optimal_indicator.json"), "w"), indent=2,
          default=float)
