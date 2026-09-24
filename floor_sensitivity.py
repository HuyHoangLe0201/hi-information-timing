"""
How much does the answer depend on the floor estimate?

The floor is subtracted with a clip at zero, so an over-estimate silently
deletes the early part of the record and pushes the early quantiles late; an
under-estimate leaves noise in and pulls them early. It is estimated from eight
bootstrap repetitions and varies by about 1.3x across seeds, so the question is
whether that variation reaches the answer.

The floor is deliberately mis-set by factors of one half and two -- far beyond
its own sampling variation -- and the quantiles recomputed. A quantity that
barely moves is safe; one that moves a lot has to be reported with the floor
that produced it.
"""
import os
import json
import numpy as np
from nonparam import weighted_density, estimate_floor, quantile

HERE = os.path.dirname(os.path.abspath(__file__))
QS = (0.10, 0.35, 0.50, 0.90)
MULT = (0.5, 1.0, 2.0)
rng = np.random.default_rng(99)

z = np.load(os.path.join(HERE, "feat_raw.npz"), allow_pickle=True)
FN = [str(s) for s in z["featnames"]]
I = {k: i for i, k in enumerate(FN)}
BK = sorted(k for k in z.files if k != "featnames")
CH = {"rms": ["rms"], "kurt": ["kurt"], "0--2 kHz": ["b0_1", "b1_2"],
      "2--4 kHz": ["b2_4"], "4--10 kHz": ["b4_6", "b6_8", "b8_10"]}

print("quantiles when the floor is deliberately mis-set\n")
print(f"{'indicator':<14}{'q':>6}" + "".join(f"{m:>10.1f}x" for m in MULT)
      + f"{'swing':>9}")
print("-" * 55)
rows = []
for lab, parts in CH.items():
    per = {m: {q: [] for q in QS} for m in MULT}
    for b in BK:
        x = np.asarray(sum(z[b][:, I[p]] for p in parts), float)
        dens, res = weighted_density(x)
        if dens is None:
            continue
        f0 = estimate_floor(dens, res, "surrogate", rng)
        for m in MULT:
            d = np.clip(dens - m * f0, 0.0, None)
            if d.sum() <= 0:
                continue
            F = np.cumsum(d) / d.sum()
            for q in QS:
                per[m][q].append(quantile(F, q))
    for q in QS:
        vals = [float(np.median(per[m][q])) for m in MULT if per[m][q]]
        if len(vals) != len(MULT):
            continue
        swing = max(vals) - min(vals)
        rows.append(dict(indicator=lab, q=q,
                         values={str(m): v for m, v in zip(MULT, vals)},
                         swing=swing))
        print(f"{lab:<14}{q:>6.2f}" + "".join(f"{v:>11.3f}" for v in vals)
              + f"{swing:>9.3f}")
    print()

print("-" * 55)
sw = [r["swing"] for r in rows]
big = [r for r in rows if r["swing"] > 0.05]
print(f"a fourfold change in the floor moves the quantiles by "
      f"{np.median(sw):.3f} at the median, up to {max(sw):.3f}.")
if big:
    print(f"{len(big)} of {len(rows)} entries move by more than 0.05:")
    for r in sorted(big, key=lambda r: -r["swing"])[:6]:
        print(f"  {r['indicator']:<14} q={r['q']:.2f}  swing {r['swing']:.3f}")
else:
    print("no entry moves by more than 0.05 of life.")
print("\nThe floor's own sampling variation is about 1.3x, well inside the")
print("fourfold range tested here, so its contribution to the reported")
print("quantiles is smaller than the swings above.")
json.dump(rows, open(os.path.join(HERE, "floor_sensitivity.json"), "w"),
          indent=2)
