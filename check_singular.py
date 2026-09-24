"""
Are the band channels linearly dependent, and does that corrupt the fusion?

channel_corr.py reported seven band channels each contributing the identical
increment 3.19e+09, several hundred times their own information.  Correlated
noise genuinely can make a channel worth more in company than alone, so a large
increment is not by itself an error, but an identical one across seven distinct
channels is not a physical effect.

The suspicion is structural.  The band features are fractions of total spectrum
power, so if the bands tile the spectrum they sum to one identically, the ten
channels span a nine-dimensional space, and the covariance is singular.  The
Schur-complement identity says the increment of a redundant channel is zero over
zero; the ridge that keeps the inverse finite then sets it to something large and
arbitrary rather than to nothing.  That is the failure mode the identity
predicts, and if it is present the paper's ten-channel result is an artefact.

This script establishes three things in order: whether the bands sum to a
constant, what that does to the conditioning, and whether the reported
ten-channel age survives dropping one band to break the dependency.
"""
import os
import json
import numpy as np

from fusion_prep import prepare_fusion, joint_info, single_info, tau_min_from

HERE = os.path.dirname(os.path.abspath(__file__))
QSTAR, REF = 0.35, "rms"

z = np.load(os.path.join(HERE, "feat_raw.npz"), allow_pickle=True)
FN = [str(s) for s in z["featnames"]]
I = {k: i for i, k in enumerate(FN)}
units = sorted(k for k in z.files if k != "featnames")
BANDS = [n for n in FN if n.startswith("b")]
print(f"channels: {FN}")
print(f"bands   : {BANDS}\n")

# --- 1. do the bands sum to a constant? ------------------------------------
print("sum of the band channels, per bearing")
print(f"{'bearing':<16}{'mean':>10}{'sd':>12}{'min':>10}{'max':>10}")
print("-" * 58)
sds = []
for u in units[:6]:
    S = np.asarray(z[u][:, [I[b] for b in BANDS]], float).sum(axis=1)
    S = S[np.isfinite(S)]
    sds.append(float(np.std(S)))
    print(f"{u:<16}{S.mean():>10.4f}{np.std(S):>12.2e}{S.min():>10.4f}"
          f"{S.max():>10.4f}")
allsd = []
for u in units:
    S = np.asarray(z[u][:, [I[b] for b in BANDS]], float).sum(axis=1)
    allsd.append(float(np.std(S[np.isfinite(S)])))
print("-" * 58)
print(f"over all {len(units)} bearings the standard deviation of the sum is at "
      f"most {max(allsd):.2e}")
dependent = max(allsd) < 1e-6
print("The bands tile the spectrum and their fractions sum to one identically."
      if dependent else "The bands do not sum to a constant.")
print()

# --- 2. what that does to the conditioning ---------------------------------
prepared = []
for u in units:
    ch = {}
    for nm in FN:
        p, _ = prepare_fusion(np.asarray(z[u][:, I[nm]], float), nm)
        if p is not None:
            ch[nm] = p
    if len(ch) == len(FN) and len({p["n"] for p in ch.values()}) == 1:
        prepared.append((u, ch))
print(f"{len(prepared)} bearings carrying all channels\n")


def cond_of(ch, names):
    R = np.column_stack([ch[n]["res"] for n in names])
    C = np.corrcoef(R, rowvar=False)
    return float(np.linalg.cond(np.nan_to_num(C, nan=0.0)))


SETS = {
    "three bands": ["b4_6", "b6_8", "b8_10"],
    "six bands": ["b0_1", "b1_2", "b2_4", "b4_6", "b6_8", "b8_10"],
    "all seven bands": BANDS,
    "everything (10)": FN,
    "everything less one band": [n for n in FN if n != BANDS[-1]],
}
print("conditioning of the residual correlation matrix")
print(f"{'set':<28}{'K':>4}{'median cond':>16}{'worst cond':>14}")
print("-" * 62)
conds = {}
for lab, names in SETS.items():
    c = [cond_of(ch, names) for _, ch in prepared]
    conds[lab] = dict(median=float(np.median(c)), worst=float(max(c)),
                      K=len(names))
    print(f"{lab:<28}{len(names):>4}{np.median(c):>16.2e}{max(c):>14.2e}")
print("-" * 62)
print("A correlation matrix of full rank conditions at order ten; the sets")
print("containing every band do not, which is the signature of the dependency.")
print()

# --- 3. does the reported age survive breaking the dependency? -------------
print("earliest usable age, at a target fixed per bearing from RMS\n")
print(f"{'set':<28}{'K':>4}{'feasible':>11}{'median tau_min':>17}")
print("-" * 62)
rows = []
per = {}
for lab, names in SETS.items():
    vals = []
    for u, ch in prepared:
        target = QSTAR * float(single_info(ch[REF]).sum())
        vals.append(tau_min_from(joint_info([ch[n] for n in names]), target))
    per[lab] = np.array(vals, float)
    ok = np.isfinite(per[lab])
    med = float(np.median(per[lab][ok])) if ok.any() else np.nan
    rows.append(dict(set=lab, K=len(names), feasible=int(ok.sum()),
                     units=len(vals), tau=med,
                     cond=conds[lab]["median"]))
    print(f"{lab:<28}{len(names):>4}{ok.sum():>8}/{len(vals):<3}{med:>17.3f}")
print("-" * 62)

a = per["everything (10)"]
b = per["everything less one band"]
m = np.isfinite(a) & np.isfinite(b)
print(f"\nten channels against the same set with one band removed, "
      f"per bearing ({m.sum()} units):")
print(f"  median age {np.median(a[m]):.3f} against {np.median(b[m]):.3f}, "
      f"median difference {np.median(b[m] - a[m]):+.3f}")
print(f"  the removed channel is redundant by construction, so the identity")
print(f"  requires the difference to be zero; anything else is the ridge")
print(f"  standing in for an increment that is genuinely zero over zero.")

json.dump(dict(bands=BANDS, max_sd_of_band_sum=float(max(allsd)),
               dependent=bool(dependent), conditioning=conds, sets=rows,
               ten_vs_nine=dict(
                   units=int(m.sum()),
                   ten=float(np.median(a[m])), nine=float(np.median(b[m])),
                   diff=float(np.median(b[m] - a[m])))),
          open(os.path.join(HERE, "check_singular.json"), "w"), indent=2,
          default=float)
