"""
Does the applied result rest on any one measure?

The distribution-before-amount gap is the difference of two medians over
measures, and which measures went into each family was a choice.  Section 8.1
defends the CATEGORIES against four alternative explanations -- scale,
boundedness, ratio arithmetic, spectral content -- but not the membership: if one
unusually early distributional statistic or one unusually late amount statistic
carried the whole effect, every control in that section would still pass.

The test is immediate from the per-unit ages already computed.  Each measure is
dropped in turn and the statistic rebuilt from the remaining nine.  The set is
the ten of Table 8, six on the distribution side and four on the amount side,
so a median survives every deletion and the comparison stays the paper's own.
It ran on eight of the ten until the fourth reviewer read, which is how the
interval quoted beside Table 8 came to be computed on a different set from
the table itself.

Reported: the gap without each measure, and the worst case over all ten.  If
the result survives every deletion it does not rest on any single member.
"""
import os
import json
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.join(HERE, "tau_percell_v3.json")

DIST = ["spectral entropy", "spectral centroid", "spectral spread",
        "spectral Gini", "top-8-band share", "high/low band ratio"]
AMT = ["total power", "peak band power", "high-band power", "log total power"]

per = json.load(open(CACHE))
print(", ".join(f"{k}: {len(v)} bearings" for k, v in per.items()) + "\n")


def gap(rows, dist, amt):
    def med_measure(nm):
        v = [r[nm] for r in rows if r.get(nm) is not None]
        return float(np.median(v)) if v else np.nan

    d = [v for v in (med_measure(n) for n in dist) if np.isfinite(v)]
    a = [v for v in (med_measure(n) for n in amt) if np.isfinite(v)]
    if not d or not a:
        return np.nan
    return float(np.median(a) - np.median(d))


base = {rig: gap(rows, DIST, AMT) for rig, rows in per.items()}
print("the gap with every measure present")
for rig, g in base.items():
    print(f"  {rig:<12}{g:>8.3f}")
print()

print("the gap with each measure dropped in turn\n")
print(f"{'dropped':<20}{'family':<14}" + "".join(f"{r:>13}" for r in per))
print("-" * (34 + 13 * len(per)))
rows_out = []
for nm in DIST + AMT:
    fam = "distribution" if nm in DIST else "amount"
    d2 = [x for x in DIST if x != nm]
    a2 = [x for x in AMT if x != nm]
    gs = {rig: gap(rows, d2, a2) for rig, rows in per.items()}
    rows_out.append(dict(dropped=nm, family=fam,
                         **{k: float(v) for k, v in gs.items()}))
    print(f"{nm:<20}{fam:<14}"
          + "".join(f"{gs[r]:>13.3f}" for r in per))
print("-" * (34 + 13 * len(per)))

for rig in per:
    v = np.array([r[rig] for r in rows_out], float)
    v = v[np.isfinite(v)]
    print(f"{rig}: gap runs {v.min():.3f} to {v.max():.3f} across the {len(v)} "
          f"deletions, base {base[rig]:.3f}")
print()

allv = np.array([r[rig] for r in rows_out for rig in per], float)
allv = allv[np.isfinite(allv)]
if (allv > 0).all():
    worst = float(allv.min())
    who = min(((r[rig], r["dropped"], rig) for r in rows_out for rig in per
               if np.isfinite(r[rig])), key=lambda t: t[0])
    print("The gap stays positive under every single deletion, on both rigs.")
    print(f"The weakest case is {worst:.3f}, on {who[2]} without "
          f"{who[1]},")
    print("so the applied result does not rest on any one measure.  It is a")
    print("statement about the two families and not about a favourable member")
    print("of either.")
else:
    bad = [(r["dropped"], rig, r[rig]) for r in rows_out for rig in per
           if np.isfinite(r[rig]) and r[rig] <= 0]
    print(f"The gap fails to stay positive when {bad} is dropped, so the result")
    print("does rest on that measure and the paper must say so.")

json.dump(dict(base=base, deletions=rows_out,
               worst=float(allv.min()), best=float(allv.max()),
               all_positive=bool((allv > 0).all())),
          open(os.path.join(HERE, "leave_one_measure.json"), "w"), indent=2,
          default=float)
