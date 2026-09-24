"""
Does fusion buy earlier usable age on real records?

Two corrections to earlier versions of this experiment are folded in.

The precision target must be one fixed absolute demand across every indicator
set on a unit. Setting it from the best-endowed member instead makes the answer
appear to worsen when a strong indicator is added, which cannot happen.

And channels must not be screened on their individual signal-to-noise. That
screen is right for a single indicator and wrong here -- a channel that carries
little alone still contributes -- and requiring all ten bearing channels to pass
it left 6 of 17 units. fusion_prep uses a normalisation that cannot degenerate,
which the joint information is invariant to.

The sets are nested, so the monotonicity implied by information being
non-negative is directly checkable rather than assumed.
"""
import os
import json
import numpy as np
from fusion_prep import (prepare_fusion, joint_info, single_info, tau_min_from)

HERE = os.path.dirname(os.path.abspath(__file__))
QSTAR = 0.35


def analyse(units, sets, chains, ref, title):
    per_unit, dropped = {}, 0
    for name, chans in units:
        u = {}
        for nm, x in chans.items():
            p, _ = prepare_fusion(x, nm)
            if p is not None:
                u[nm] = p
        need = {n for names in sets.values() for n in names}
        if not need <= set(u) or len({p["n"] for p in u.values()}) != 1:
            dropped += 1
            continue
        target = QSTAR * float(single_info(u[ref]).sum())
        per_unit[name] = {lab: tau_min_from(joint_info([u[n] for n in names]),
                                            target)
                          for lab, names in sets.items()}
        per_unit[name]["__gam"] = {
            lab: float(joint_info([u[n] for n in names]).sum())
            for lab, names in sets.items()}
        per_unit[name]["__sing"] = {
            lab: float(max(single_info(u[n]).sum() for n in names))
            for lab, names in sets.items()}
        per_unit[name]["__sum"] = {
            lab: float(sum(single_info(u[n]).sum() for n in names))
            for lab, names in sets.items()}

    print(f"\n=== {title} === "
          f"({len(per_unit)} units analysed, {dropped} dropped)\n")
    print(f"{'indicator set':<22}{'K':>3}{'feasible':>10}{'tau_min':>9}"
          f"{'earlier':>9}{'K_eff':>8}{'overcount':>11}")
    print("-" * 72)
    base = float(np.median([per_unit[b][ref.upper()] for b in per_unit
                            if ref.upper() in per_unit[b]
                            and np.isfinite(per_unit[b][ref.upper()])])) \
        if ref.upper() in sets else np.nan
    rows = []
    for lab, names in sets.items():
        v = [per_unit[b][lab] for b in per_unit]
        ok = [x for x in v if np.isfinite(x)]
        med = float(np.median(ok)) if ok else np.nan
        keff = float(np.median([per_unit[b]["__gam"][lab] /
                                per_unit[b]["__sing"][lab] for b in per_unit]))
        over = float(np.median([per_unit[b]["__sum"][lab] /
                                per_unit[b]["__gam"][lab] for b in per_unit]))
        rows.append(dict(set=lab, K=len(names), feasible=len(ok),
                         units=len(v), tau_min=med, k_eff=keff,
                         overcount=over))
        ms = "     --  " if not ok else f"{med:>9.3f}"
        gs = "     --  " if not ok or not np.isfinite(base) \
            else f"{base - med:>9.3f}"
        print(f"{lab:<22}{len(names):>3}{len(ok):>7}/{len(v):<3}{ms}{gs}"
              f"{keff:>8.2f}{over:>11.2f}")
    print("-" * 72)

    bad, tests = [], 0
    for chain in chains:
        for b in per_unit:
            seq = [per_unit[b][c] for c in chain]
            for i in range(len(seq) - 1):
                tests += 1
                a, c = seq[i], seq[i + 1]
                if np.isfinite(a) and np.isfinite(c) and c > a + 1e-9:
                    bad.append((b, chain[i], chain[i + 1], a, c))
    print(f"monotonicity over nested sets: {tests - len(bad)}/{tests} pass")
    for b, x, y, a, c in bad[:4]:
        print(f"  VIOLATION {b}: {x} {a:.3f} -> {y} {c:.3f}")
    print("K_eff     = joint budget / best single member's budget")
    print("overcount = what additivity would predict / the truth "
          "(>1 means the members repeat each other)")
    return dict(title=title, units=len(per_unit), rows=rows,
                violations=len(bad), tests=tests)


out = []

# ---- bearings --------------------------------------------------------------
z = np.load(os.path.join(HERE, "feat_raw.npz"), allow_pickle=True)
FN = [str(s) for s in z["featnames"]]
I = {k: i for i, k in enumerate(FN)}
BUNITS = [(b, {nm: z[b][:, I[nm]] for nm in FN})
          for b in sorted(k for k in z.files if k != "featnames")]
BSETS = {
    "RMS": ["rms"],
    "RMS + peak + kurt": ["rms", "peak", "kurt"],
    "4--10 kHz": ["b4_6", "b6_8", "b8_10"],
    "RMS + 4--10 kHz": ["rms", "b4_6", "b6_8", "b8_10"],
    "six bands": ["b0_1", "b1_2", "b2_4", "b4_6", "b6_8", "b8_10"],
    "everything": FN,
}
out.append(analyse(BUNITS, BSETS,
                   [["RMS", "RMS + peak + kurt", "everything"],
                    ["RMS", "RMS + 4--10 kHz", "everything"],
                    ["4--10 kHz", "RMS + 4--10 kHz", "everything"],
                    ["six bands", "everything"]],
                   "rms", "PRONOSTIA bearings, 10 channels"))

# ---- turbofan: 21 sensor channels ------------------------------------------
zt = np.load(os.path.join(HERE, "cmapss_FD001.npz"))
names = sorted({k.split("__")[0] for k in zt.files})
TU = []
for nm in names[:60]:
    S = zt[f"{nm}__sensors"].astype(float)
    TU.append((nm, {f"s{c+1}": S[:, c] for c in range(S.shape[1])}))
ALL = [f"s{c+1}" for c in range(zt[f"{names[0]}__sensors"].shape[1])]
TSETS = {
    "S11": ["s11"],
    "three standard": ["s4", "s11", "s15"],
    "eight informative": ["s2", "s3", "s4", "s7", "s11", "s12", "s15", "s21"],
    "all sensors": ALL,
}
out.append(analyse(TU, TSETS,
                   [["S11", "three standard", "eight informative",
                     "all sensors"]],
                   "s11", "C-MAPSS FD001, 21 sensor channels"))

json.dump(out, open(os.path.join(HERE, "fusion_taumin.json"), "w"), indent=2)
print("\nwrote fusion_taumin.json")
