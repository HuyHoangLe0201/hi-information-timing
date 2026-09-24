"""
Does the flat lookup agree with the bootstrap it replaces?

The surrogate floor resamples each record's own residual and measures what the
pipeline reports; it assumes nothing, and costs 24 pipeline runs per record. The
lookup uses one number read off record length, valid only because the weighted
floor was measured to be flat and free of the noise scale. This checks the two
against each other on the real records, where the residuals are neither
Gaussian nor stationary and the lookup has no right to be exact.
"""
import os
import json
import numpy as np
from nonparam import weighted_density, estimate_floor, flat_level, quantile, info_curve

HERE = os.path.dirname(os.path.abspath(__file__))
rng = np.random.default_rng(31)

z = np.load(os.path.join(HERE, "feat_raw.npz"), allow_pickle=True)
FN = [str(s) for s in z["featnames"]]
I = {k: i for i, k in enumerate(FN)}
BK = sorted(k for k in z.files if k != "featnames")[:5]
CH = {"rms": ["rms"], "kurt": ["kurt"], "0--2 kHz": ["b0_1", "b1_2"],
      "4--10 kHz": ["b4_6", "b6_8", "b8_10"]}

print(f"{'channel':<14}{'n':>6}{'bootstrap':>11}{'lookup':>9}{'ratio':>8}"
      f"{'q35 boot':>10}{'q35 look':>10}{'gap':>7}")
print("-" * 75)
rows = []
for lab, parts in CH.items():
    B, L, QB, QL = [], [], [], []
    for b in BK:
        x = np.asarray(sum(z[b][:, I[p]] for p in parts), float)
        dens, res = weighted_density(x)
        if dens is None:
            continue
        fb = float(estimate_floor(dens, res, "surrogate", rng, reps=8)[0])
        fl = flat_level(len(dens))
        B.append(fb); L.append(fl)
        db = np.clip(dens - fb, 0, None)
        QB.append(quantile(np.cumsum(db)/db.sum(), 0.35))
        QL.append(quantile(info_curve(x, "flat"), 0.35))
    if not B:
        continue
    n = int(np.median([len(z[b]) for b in BK]))
    r = dict(channel=lab, n=n, boot=float(np.median(B)), look=float(np.median(L)),
             q_boot=float(np.median(QB)), q_look=float(np.median(QL)))
    r["ratio"] = r["boot"] / r["look"]
    r["gap"] = abs(r["q_boot"] - r["q_look"])
    rows.append(r)
    print(f"{lab:<14}{n:>6}{r['boot']:>11.2f}{r['look']:>9.2f}{r['ratio']:>8.2f}"
          f"{r['q_boot']:>10.3f}{r['q_look']:>10.3f}{r['gap']:>7.3f}")
print("-" * 75)
g = [r["gap"] for r in rows]
rt = [r["ratio"] for r in rows]
print(f"floor levels agree to {min(rt):.2f}-{max(rt):.2f}x;")
print(f"the quantile they produce differs by at most {max(g):.3f} of life.")
print("\nThe lookup is the cheaper route and is used where the difference is")
print("immaterial; the bootstrap remains available for records whose residual")
print("structure is in doubt.")
json.dump(rows, open(os.path.join(HERE, "floor_check.json"), "w"), indent=2)
