"""
Not whether the criteria rank the same, but whether they can tell things apart.

Rank correlation was the wrong question. With nine indicators its standard error
is about 0.35, so the measured -0.43, -0.33 and -0.52 against the earliest usable
age are not distinguishable from zero, and a prediction that the two would be
OPPOSED was simply wrong -- they agree weakly.

The useful question is resolution. Two indicators can score alike on a criterion
and still differ by most of a lifetime in when they become usable, and if that
happens the criterion cannot be used to make the choice the framework makes.
Root-mean-square amplitude scores 0.130 for monotonicity and the 8-10 kHz band
scores 0.126 -- indistinguishable -- while their earliest usable ages are 0.97
and 0.20.

So every pair of indicators is examined: how far apart are they on each
criterion, and how far apart in earliest usable age. A criterion that separates
what matters will not have pairs that are close on it and far apart in age.
"""
import os
import json
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
rows = json.load(open(os.path.join(HERE, "vs_criteria.json")))
KEYS = (("mon", "monotonicity"), ("tren", "trendability"),
        ("prog", "prognosability"))

for key, lab in KEYS:
    v = np.array([r[key] for r in rows], float)
    rng_ = v.max() - v.min()
    print(f"{lab}: scores span {v.min():.3f} to {v.max():.3f}")
print()

print("pairs that a criterion cannot separate, and how far apart they really are\n")
out = {}
for key, lab in KEYS:
    v = np.array([r[key] for r in rows], float)
    span = v.max() - v.min()
    pairs = []
    for i in range(len(rows)):
        for j in range(i + 1, len(rows)):
            d_crit = abs(v[i] - v[j]) / span            # as a share of the range
            d_tau = abs(rows[i]["tau"] - rows[j]["tau"])
            pairs.append((d_crit, d_tau, rows[i]["indicator"],
                          rows[j]["indicator"]))
    close = [p for p in pairs if p[0] < 0.10]           # within a tenth of range
    out[key] = dict(n_pairs=len(pairs), n_close=len(close),
                    worst=max((p[1] for p in close), default=0.0))
    print(f"{lab}  ({len(close)} of {len(pairs)} pairs within 10% of the "
          f"score range)")
    for d_crit, d_tau, a, b in sorted(close, key=lambda p: -p[1])[:4]:
        print(f"    {a:<14} vs {b:<14} scores differ by "
              f"{100*d_crit:>4.1f}% of range, ages by {d_tau:.2f} of life")
    print()

print("-" * 70)
worst = max(out[k]["worst"] for k, _ in KEYS)
print(f"the largest age gap between two indicators a criterion cannot "
      f"separate: {worst:.2f} of life\n")

# and the converse: does a criterion ever separate what the age does not?
tau = np.array([r["tau"] for r in rows], float)
conv = []
for key, lab in KEYS:
    v = np.array([r[key] for r in rows], float)
    span = v.max() - v.min()
    for i in range(len(rows)):
        for j in range(i + 1, len(rows)):
            if abs(tau[i] - tau[j]) < 0.05 and abs(v[i] - v[j]) / span > 0.5:
                conv.append((lab, rows[i]["indicator"], rows[j]["indicator"],
                             abs(v[i] - v[j]) / span, abs(tau[i] - tau[j])))
print("the converse -- pairs the age cannot separate but a criterion does:")
if conv:
    for lab, a, b, dc, dt in conv:
        print(f"  {lab}: {a} vs {b}, scores differ {100*dc:.0f}% of range, "
              f"ages by {dt:.2f}")
else:
    print("  none")
print()
print("A criterion blind to a gap of most of a lifetime is not measuring the")
print("same thing, whatever its rank correlation. The two are complementary:")
print("the criteria say whether an indicator behaves tidily enough to model,")
print("the age says whether there is anything to model yet.")
json.dump(out, open(os.path.join(HERE, "criteria_resolution.json"), "w"),
          indent=2, default=float)
