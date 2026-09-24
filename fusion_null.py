"""
How much of a fused set's information is real?

Correcting the inverse-covariance inflation is not enough. On pure noise the
corrected joint information still grows roughly linearly with the number of
channels, because the Savitzky-Golay derivative of noise is not zero: every
channel contributes spurious derivative energy that the density counts as
information. Any comparison of a large indicator set against a small one is
therefore rigged in favour of the large one unless that floor is measured.

The floor is measured per set, per unit, by surrogates that keep everything
except the degradation: each channel's own out-of-fold residual is resampled,
and the residual ROWS are resampled jointly so the cross-channel correlation
structure is preserved exactly. The surrogate has no trend by construction, so
whatever information the pipeline reports on it is floor.
"""
import os
import json
import numpy as np
from fusion_prep import prepare_fusion, joint_info, single_info

HERE = os.path.dirname(os.path.abspath(__file__))
REPS = 24


def surrogate_info(chans, rng, reps=REPS):
    """Information the pipeline reports when the degradation is removed."""
    R = np.column_stack([u["res"] for u in chans])
    n, K = R.shape
    tot = []
    for _ in range(reps):
        idx = rng.integers(0, n, n)          # rows together: correlation kept
        S = R[idx, :]
        ch = [prepare_fusion(S[:, j])[0] for j in range(K)]
        if any(c is None for c in ch):
            continue
        v = joint_info(ch).sum()
        if np.isfinite(v):
            tot.append(float(v))
    return float(np.median(tot)) if tot else np.nan


def run(units, sets, title, rng):
    print(f"\n=== {title} ===\n")
    print(f"{'indicator set':<22}{'K':>3}{'units':>6}{'observed':>12}"
          f"{'floor':>12}{'obs/floor':>11}{'excess/chan':>13}")
    print("-" * 77)
    rows = []
    for lab, names in sets.items():
        obs, nul = [], []
        for uname, chans in units:
            ch = []
            for nm in names:
                p, _ = prepare_fusion(chans[nm], nm)
                if p is None:
                    ch = None
                    break
                ch.append(p)
            if ch is None or len({c["n"] for c in ch}) != 1:
                continue
            o = float(joint_info(ch).sum())
            f = surrogate_info(ch, rng)
            if np.isfinite(o) and np.isfinite(f) and f > 0:
                obs.append(o); nul.append(f)
        if not obs:
            print(f"{lab:<22}{len(names):>3}   no unit carries every channel")
            continue
        obs, nul = np.array(obs), np.array(nul)
        ratio = float(np.median(obs / nul))
        excess = float(np.median(obs - nul))
        rows.append(dict(set=lab, K=len(names), units=len(obs),
                         observed=float(np.median(obs)),
                         floor=float(np.median(nul)),
                         ratio=ratio, excess=excess,
                         excess_per_channel=excess / len(names)))
        r = rows[-1]
        print(f"{lab:<22}{r['K']:>3}{r['units']:>6}{r['observed']:>12.3g}"
              f"{r['floor']:>12.3g}{ratio:>11.2f}"
              f"{r['excess_per_channel']:>13.3g}")
    print("-" * 77)
    print("excess = observed - floor: the information attributable to")
    print("degradation rather than to differentiating noise.")
    return dict(title=title, rows=rows)


rng = np.random.default_rng(404)
out = []

z = np.load(os.path.join(HERE, "feat_raw.npz"), allow_pickle=True)
FN = [str(s) for s in z["featnames"]]
I = {k: i for i, k in enumerate(FN)}
BU = [(b, {nm: z[b][:, I[nm]] for nm in FN})
      for b in sorted(k for k in z.files if k != "featnames")]
out.append(run(BU, {
    "RMS": ["rms"],
    "RMS + peak + kurt": ["rms", "peak", "kurt"],
    "4--10 kHz": ["b4_6", "b6_8", "b8_10"],
    "six bands": ["b0_1", "b1_2", "b2_4", "b4_6", "b6_8", "b8_10"],
    "everything": FN,
}, "PRONOSTIA bearings", rng))

zt = np.load(os.path.join(HERE, "cmapss_FD001.npz"))
names = sorted({k.split("__")[0] for k in zt.files})
NCH = zt[f"{names[0]}__sensors"].shape[1]
TU = []
for nm in names[:40]:
    S = zt[f"{nm}__sensors"].astype(float)
    TU.append((nm, {f"s{c+1}": S[:, c] for c in range(NCH)}))
# channels that are constant in FD001 carry nothing and cannot be normalised
live = [f"s{c+1}" for c in range(NCH)
        if np.std(zt[f"{names[0]}__sensors"][:, c].astype(float)) > 0]
print(f"\n(FD001: {len(live)} of {NCH} sensor channels are non-constant)")
out.append(run(TU, {
    "s11": ["s11"],
    "three standard": ["s4", "s11", "s15"],
    "eight informative": ["s2", "s3", "s4", "s7", "s11", "s12", "s15", "s21"],
    "all live sensors": live,
}, "C-MAPSS FD001", rng))

json.dump(out, open(os.path.join(HERE, "fusion_null.json"), "w"), indent=2)
print("\nwrote fusion_null.json")
