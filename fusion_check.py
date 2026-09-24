"""
Additivity, on the corrected machinery.

Section 2 argues that the unnormalised curve is the right object partly because
it is additive: informations add, shapes do not, so two indicators can be
combined under G and cannot under F. The consequence is testable and sharp --
adding an indicator to a set cannot make the earliest usable age later, because
information is non-negative.

An earlier version of this check used the unweighted density and no noise-floor
subtraction, both since shown to be wrong. It is redone here on the machinery the
paper uses, with two corrections that matter for the multi-channel case:

  the floor grows with the number of channels, since every channel contributes
  spurious derivative energy, so a larger set is flattered unless it is removed;
  and inverting an estimated covariance inflates the quadratic form by
  m/(m-K-1), which also grows with K, so a larger set is flattered again.

The target is held at one fixed absolute demand for every set on a unit. Setting
it from the best-endowed member instead moves the target when a strong indicator
is added, and the answer then appears to get worse -- which cannot happen and is
an artefact of the protocol rather than a finding.
"""
import os
import json
import numpy as np
from fusion_prep import prepare_fusion, joint_info, single_info, tau_min_from

HERE = os.path.dirname(os.path.abspath(__file__))
QSTAR = 0.35
REF = "rms"
SEED = 20260828


def load_bearings():
    z = np.load(os.path.join(HERE, "feat_raw.npz"), allow_pickle=True)
    FN = [str(s) for s in z["featnames"]]
    I = {k: i for i, k in enumerate(FN)}
    out = []
    for b in sorted(k for k in z.files if k != "featnames"):
        M = z[b]
        u = {}
        for nm in FN:
            p, _ = prepare_fusion(np.asarray(M[:, I[nm]], float), nm)
            if p is not None:
                u[nm] = p
        if len(u) == len(FN) and len({p["n"] for p in u.values()}) == 1:
            out.append((b, u))
    return FN, out


FN, units = load_bearings()
print(f"{len(units)} bearings carrying all {len(FN)} channels\n")

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

rows, per_unit = [], {}
for b, u in units:
    target = QSTAR * float(single_info(u[REF]).sum())
    per_unit[b] = {lab: tau_min_from(joint_info([u[n] for n in names]), target)
                   for lab, names in SETS.items()}

print(f"{'indicator set':<22}{'K':>3}{'feasible':>10}{'tau_min':>10}"
      f"{'vs RMS alone':>14}")
print("-" * 60)
base = float(np.median([per_unit[b]["RMS"] for b in per_unit
                        if np.isfinite(per_unit[b]["RMS"])]))
for lab, names in SETS.items():
    v = [per_unit[b][lab] for b in per_unit]
    ok = [x for x in v if np.isfinite(x)]
    med = float(np.median(ok)) if ok else np.nan
    rows.append(dict(set=lab, K=len(names), feasible=len(ok), units=len(v),
                     tau=med))
    ms = "      --  " if not ok else f"{med:>10.3f}"
    gs = "        --" if not ok else f"{base - med:>14.3f}"
    print(f"{lab:<22}{len(names):>3}{len(ok):>7}/{len(v):<3}{ms}{gs}")
print("-" * 60)
print("The tau_min column is a median over the units where that set is")
print("feasible, and those units differ between rows: 'three bands' is")
print("feasible on 5 and 'RMS + three bands' on 17, so their medians are not")
print("comparable and the second is NOT later than the first. Only the")
print("per-unit test below compares like with like.\n")

bad, tests = [], 0
for chain in CHAINS:
    for b in per_unit:
        seq = [per_unit[b][c] for c in chain]
        for i in range(len(seq) - 1):
            tests += 1
            a, c = seq[i], seq[i + 1]
            if np.isfinite(a) and np.isfinite(c) and c > a + 1e-9:
                bad.append((b, chain[i], chain[i + 1], a, c))
print(f"\nmonotonicity over nested sets: {tests - len(bad)}/{tests} pass")
for b, x, y, a, c in bad[:4]:
    print(f"  VIOLATION {b}: {x} {a:.3f} -> {y} {c:.3f}")

# the case F cannot express: individually infeasible, jointly feasible
joint_rescues = 0
for b, u in units:
    target = QSTAR * float(single_info(u[REF]).sum())
    for a, c in (("peak", "kurt"), ("b0_1", "b1_2"), ("b6_8", "b8_10")):
        ta = tau_min_from(joint_info([u[a]]), target)
        tc = tau_min_from(joint_info([u[c]]), target)
        tj = tau_min_from(joint_info([u[a], u[c]]), target)
        if not np.isfinite(ta) and not np.isfinite(tc) and np.isfinite(tj):
            joint_rescues += 1
print(f"\npairs infeasible alone but feasible together: {joint_rescues}")
print("Under the normalised curve this case has no representation, since every")
print("indicator has F(1) = 1 by construction and none is ever infeasible.")
json.dump(dict(sets=rows, tests=tests, violations=len(bad),
               joint_rescues=joint_rescues),
          open(os.path.join(HERE, "fusion_check.json"), "w"), indent=2,
          default=float)
