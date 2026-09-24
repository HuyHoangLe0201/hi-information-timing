"""
Rank-safe fusion, and the correction it forces to the additivity result.

check_singular.py established that the ten-channel result reported earlier is an
artefact.  The seven band features are fractions of total spectrum power over
bands that tile the spectrum, so they sum to one up to spectral leakage; the
residual correlation matrix of any set containing all seven conditions at 1e5
against 1e2 for a set of the same size that does not, and the earliest usable age
it reports collapses from 0.957 to 0.029.

The mechanism is the one the Schur-complement identity predicts.  Along the
sum-to-one direction the noise is near zero, so the quadratic form divides by
nearly nothing and attributes enormous information to a direction that carries no
measurement at all.  This is not a real noiseless channel: it is an identity of
the feature construction.  Any fusion that inverts a covariance will do the same
whenever the feature set contains an exact or near-exact linear dependency, which
is common -- band fractions, normalised histograms, compositional features and
one-hot encodings all have it.

The remedy is to work in a basis that spans the same measurements without the
degenerate direction.  A dependency is detected from the residual correlation
matrix alone, so no ground truth and no knowledge of how the features were built
is required: channels are dropped greedily, each time the one whose removal most
improves the condition number, until the matrix conditions below a threshold.
The threshold is fixed in advance at 1e3, two orders below the 1e5 the degenerate
sets reach and an order above the 1e2 the well-conditioned ones do.

Everything the paper claims from fusion is then recomputed on this protocol.
"""
import os
import json
import numpy as np

from fusion_prep import prepare_fusion, joint_info, single_info, tau_min_from

HERE = os.path.dirname(os.path.abspath(__file__))
QSTAR, REF = 0.35, "rms"
COND_MAX = 1e3

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
print(f"{len(prepared)} bearings carrying all {len(FN)} channels")
print(f"condition threshold {COND_MAX:.0e}, fixed before looking at any age\n")


def cond_of(ch, names):
    if len(names) < 2:
        return 1.0
    R = np.column_stack([ch[n]["res"] for n in names])
    C = np.nan_to_num(np.corrcoef(R, rowvar=False), nan=0.0)
    return float(np.linalg.cond(C))


def rank_safe(ch, names, cmax=COND_MAX):
    """Greedily drop the channel whose removal best conditions the set."""
    keep, dropped = list(names), []
    while len(keep) > 1 and cond_of(ch, keep) > cmax:
        best = min(keep, key=lambda n: cond_of(ch, [k for k in keep if k != n]))
        keep = [k for k in keep if k != best]
        dropped.append(best)
    return keep, dropped


SETS = {
    "RMS": ["rms"],
    "RMS + peak": ["rms", "peak"],
    "RMS + peak + kurt": ["rms", "peak", "kurt"],
    "three bands": ["b4_6", "b6_8", "b8_10"],
    "RMS + three bands": ["rms", "b4_6", "b6_8", "b8_10"],
    "six bands": ["b0_1", "b1_2", "b2_4", "b4_6", "b6_8", "b8_10"],
    "everything": FN,
}
CHAINS = [["RMS", "RMS + peak", "RMS + peak + kurt", "everything"],
          ["three bands", "RMS + three bands", "everything"],
          ["six bands", "everything"]]

# --- which sets are degenerate, and what is dropped -------------------------
print("rank guard, per set (dropped channels listed where the guard fires)\n")
print(f"{'set':<22}{'K':>3}{'cond before':>14}{'cond after':>13}{'kept':>6}"
      f"  dropped")
print("-" * 78)
guard = {}
for lab, names in SETS.items():
    cb = [cond_of(ch, names) for _, ch in prepared]
    keeps, drops, ca = [], [], []
    for _, ch in prepared:
        k, d = rank_safe(ch, names)
        keeps.append(k)
        drops.append(tuple(sorted(d)))
        ca.append(cond_of(ch, k))
    guard[lab] = keeps
    common = sorted({x for d in drops for x in d})
    nk = int(np.median([len(k) for k in keeps]))
    print(f"{lab:<22}{len(names):>3}{np.median(cb):>14.1e}{np.median(ca):>13.1e}"
          f"{nk:>6}  {', '.join(common) if common else '--'}")
print("-" * 78)
print("The guard fires only on the sets that contain every band, and removes")
print("one band from each; it never fires on a set of comparable size that is")
print("free of the dependency, so it is not simply penalising large sets.\n")

# --- the corrected table ----------------------------------------------------
per_naive, per_safe = {}, {}
for i, (u, ch) in enumerate(prepared):
    target = QSTAR * float(single_info(ch[REF]).sum())
    per_naive[u] = {lab: tau_min_from(joint_info([ch[n] for n in names]), target)
                    for lab, names in SETS.items()}
    per_safe[u] = {lab: tau_min_from(joint_info([ch[n] for n in guard[lab][i]]),
                                     target)
                   for lab in SETS}

print("earliest usable age, at one fixed absolute target per bearing\n")
print(f"{'set':<22}{'K':>3}{'feasible':>11}{'as published':>15}"
      f"{'rank-safe':>12}")
print("-" * 65)
rows = []
for lab, names in SETS.items():
    vn = np.array([per_naive[u][lab] for u in per_naive], float)
    vs = np.array([per_safe[u][lab] for u in per_safe], float)
    on, os_ = np.isfinite(vn), np.isfinite(vs)
    mn = float(np.median(vn[on])) if on.any() else np.nan
    ms = float(np.median(vs[os_])) if os_.any() else np.nan
    rows.append(dict(set=lab, K=len(names), feasible=int(os_.sum()),
                     units=len(vs), naive=mn, safe=ms))
    print(f"{lab:<22}{len(names):>3}{os_.sum():>8}/{len(vs):<3}{mn:>15.3f}"
          f"{ms:>12.3f}")
print("-" * 65)
print("Rows differ in which bearings are feasible, so the medians are not")
print("comparable across rows; only the per-bearing tests below are.\n")

# --- monotonicity, on the guarded protocol ----------------------------------
def monotone(per):
    bad = tests = 0
    for chain in CHAINS:
        for u in per:
            seq = [per[u][c] for c in chain]
            for i in range(len(seq) - 1):
                a, c = seq[i], seq[i + 1]
                if np.isfinite(a) and np.isfinite(c):
                    tests += 1
                    if c > a + 1e-9:
                        bad += 1
    return tests, bad


tn, bn = monotone(per_naive)
ts, bs = monotone(per_safe)
print(f"monotonicity over nested sets, as published: {tn - bn}/{tn}")
print(f"monotonicity over nested sets, rank-safe   : {ts - bs}/{ts}")
print("The published figure was not evidence: the degenerate sets report")
print("spuriously large information, which satisfies monotonicity")
print("automatically.  The rank-safe figure is a real test.\n")

# --- the joint-feasibility case, which uses pairs only ----------------------
PAIRS = [("peak", "kurt"), ("b0_1", "b1_2"), ("b6_8", "b8_10")]
resc, pc = 0, []
for u, ch in prepared:
    target = QSTAR * float(single_info(ch[REF]).sum())
    for a, c in PAIRS:
        pc.append(cond_of(ch, [a, c]))
        ta = tau_min_from(joint_info([ch[a]]), target)
        tc = tau_min_from(joint_info([ch[c]]), target)
        tj = tau_min_from(joint_info([ch[a], ch[c]]), target)
        if not np.isfinite(ta) and not np.isfinite(tc) and np.isfinite(tj):
            resc += 1
print(f"pairs infeasible alone but feasible together: {resc}")
print(f"worst conditioning among those pairs: {max(pc):.1f}, far below the")
print(f"threshold, so this result is untouched by the correction.\n")

# --- the honest headline ----------------------------------------------------
ev = np.array([per_safe[u]["everything"] for u in per_safe], float)
rm = np.array([per_naive[u]["RMS"] for u in per_naive], float)
m = np.isfinite(ev) & np.isfinite(rm)
print(f"rank-safe fusion of all channels against RMS alone, "
      f"per bearing ({m.sum()} units):")
print(f"  median age {np.median(ev[m]):.3f} against {np.median(rm[m]):.3f}, "
      f"median gain {np.median(rm[m] - ev[m]):+.3f} of a lifetime")
print(f"  fusion is earlier on {int((ev[m] < rm[m] - 1e-9).sum())} of "
      f"{int(m.sum())} bearings")

json.dump(dict(units=len(prepared), cond_max=COND_MAX, sets=rows,
               monotone_published=dict(tests=tn, pass_=tn - bn),
               monotone_ranksafe=dict(tests=ts, pass_=ts - bs),
               joint_rescues=resc, worst_pair_cond=float(max(pc)),
               headline=dict(units=int(m.sum()),
                             fusion=float(np.median(ev[m])),
                             rms=float(np.median(rm[m])),
                             gain=float(np.median(rm[m] - ev[m])),
                             earlier_on=int((ev[m] < rm[m] - 1e-9).sum()))),
          open(os.path.join(HERE, "fusion_ranksafe.json"), "w"), indent=2,
          default=float)
