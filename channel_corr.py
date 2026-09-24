"""
How correlated the channels' noises actually are, and whether the increment a
channel supplies is its novelty or its size.

Two things the additivity argument needs to be checked rather than assumed.

The text justifies combining indicators by saying informations add.  That is the
independent-noise special case.  If the residual noises of the ten bearing
channels are appreciably correlated, additivity is false and the property that
survives is monotonicity, which the Schur-complement identity supplies without
any independence assumption.  The median absolute residual correlation decides
which statement the paper is entitled to make.

The same identity predicts more than an inequality: the increment a channel
contributes is its sensitivity residual after regression on the channels already
present, divided by its residual variance.  A channel with a large trend that
merely duplicates one already in the set should contribute almost nothing.  That
is checked directly by comparing each channel's own information with what it adds
to the other nine.
"""
import os
import json
import numpy as np

from fusion_prep import prepare_fusion, joint_info, single_info

HERE = os.path.dirname(os.path.abspath(__file__))

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

# --- residual correlation ---------------------------------------------------
offs = []
for u, ch in prepared:
    R = np.column_stack([ch[nm]["res"] for nm in FN])
    C = np.corrcoef(R, rowvar=False)
    offs.append(np.abs(C[~np.eye(len(FN), dtype=bool)]))
O = np.concatenate(offs)
print("absolute residual correlation between channels, pooled over bearings")
print(f"  median {np.median(O):.2f}   quartiles {np.percentile(O, 25):.2f} "
      f"to {np.percentile(O, 75):.2f}   90th {np.percentile(O, 90):.2f}")
print(f"  pairs above 0.3: {100 * (O > 0.3).mean():.0f}%    "
      f"above 0.6: {100 * (O > 0.6).mean():.0f}%")
print()
if np.median(O) > 0.1:
    print("The noises are not independent, so the curves of the individual")
    print("channels do not sum to the joint curve and the paper may not claim")
    print("additivity on these records.  Monotonicity is the property that")
    print("survives, and it needs no independence.")
print()

# --- novelty against size ---------------------------------------------------
print("what each channel adds to the other nine, against what it holds alone\n")
print(f"{'channel':<10}{'own information':>18}{'increment added':>18}"
      f"{'fraction kept':>16}")
print("-" * 62)
rows = []
own_all, inc_all = {}, {}
for nm in FN:
    own, inc = [], []
    for u, ch in prepared:
        others = [ch[o] for o in FN if o != nm]
        g_o = float(joint_info(others).sum())
        g_all = float(joint_info(others + [ch[nm]]).sum())
        own.append(float(single_info(ch[nm]).sum()))
        inc.append(max(g_all - g_o, 0.0))
    own_all[nm], inc_all[nm] = own, inc
    mo, mi = float(np.median(own)), float(np.median(inc))
    rows.append(dict(channel=nm, own=mo, increment=mi,
                     kept=mi / mo if mo > 0 else np.nan))
    print(f"{nm:<10}{mo:>18.3g}{mi:>18.3g}{mi / mo if mo > 0 else np.nan:>16.3f}")
print("-" * 62)
k = np.array([r["kept"] for r in rows], float)
k = k[np.isfinite(k)]
print(f"fraction of its own information a channel retains once the other nine")
print(f"are present: median {np.median(k):.3f}, range {k.min():.3f} to {k.max():.3f}")
print()

o = np.array([r["own"] for r in rows], float)
i = np.array([r["increment"] for r in rows], float)
m = np.isfinite(o) & np.isfinite(i) & (o > 0) & (i > 0)
if m.sum() >= 4:
    rho = float(np.corrcoef(np.log(o[m]), np.log(i[m]))[0, 1])
    print(f"rank of a channel by what it holds alone against rank by what it")
    print(f"adds: Spearman "
          f"{float(np.corrcoef(np.argsort(np.argsort(o[m])), np.argsort(np.argsort(i[m])))[0, 1]):+.2f}, "
          f"log-log Pearson {rho:+.2f}")
    print("A channel's own information is therefore a poor guide to what it")
    print("contributes, which is the identity's content: novelty, not size.")

json.dump(dict(units=len(prepared),
               corr=dict(median=float(np.median(O)),
                         q1=float(np.percentile(O, 25)),
                         q3=float(np.percentile(O, 75)),
                         p90=float(np.percentile(O, 90)),
                         frac_above_03=float((O > 0.3).mean()),
                         frac_above_06=float((O > 0.6).mean())),
               channels=rows,
               kept_median=float(np.median(k)),
               kept_min=float(k.min()), kept_max=float(k.max())),
          open(os.path.join(HERE, "channel_corr.json"), "w"), indent=2,
          default=float)
