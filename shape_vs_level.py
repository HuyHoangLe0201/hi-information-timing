"""
The early information is in the spectral shape, not the energy level.

A 0.73-of-a-lifetime discrepancy between two nominally identical 4-10 kHz
channels traced to one line in the older extraction: the stored bands are
divided by the total spectrum power, so they measure the FRACTION of energy in a
band, while the bands computed here are absolute.

Those are different physical quantities. A fraction rises when energy moves
toward a band even if the overall vibration level is unchanged, which is what
early spalling does -- it redistributes energy upward before it makes the
bearing louder. An absolute band energy only responds once the level itself
grows, which happens late.

This is not a failure of the invariance the framework relies on: dividing by the
total spectrum is not a monotone transform of the band energy, it is a different
function of the spectrum.

The test is direct. Both forms are computed from the same raw acquisitions, so
nothing differs but the normalisation.
"""
import os
import json
import numpy as np
from nonparam import weighted_density, estimate_floor, quantile

HERE = os.path.dirname(os.path.abspath(__file__))
Q = 0.35
rng = np.random.default_rng(31415)


def tau_of(x):
    dens, res = weighted_density(np.asarray(x, float))
    if dens is None:
        return np.nan
    d = np.clip(dens - estimate_floor(dens, res, "surrogate", rng, reps=6),
                0.0, None)
    if d.sum() <= 0:
        return np.nan
    return quantile(np.cumsum(d) / d.sum(), Q)


z = np.load(os.path.join(HERE, "fineband.npz"))
centres = z["centres"] / 1000.0
BK = sorted(k for k in z.files if k != "centres")
SETS = {"0--2 kHz": slice(0, 5), "2--4 kHz": slice(5, 10),
        "4--10 kHz": slice(10, 25), "10--12.8 kHz": slice(25, 32),
        "full band": slice(0, 32)}

print("the same bands, absolute against fraction of total spectrum power\n")
print(f"{'band':<16}{'units':>6}{'absolute':>11}{'fraction':>11}"
      f"{'difference':>13}")
print("-" * 57)
rows = []
for lab, sl in SETS.items():
    A, R = [], []
    for b in BK:
        M = z[b]
        tot = M.sum(axis=1)
        a = tau_of(M[:, sl].sum(axis=1))
        r = tau_of(M[:, sl].sum(axis=1) / np.clip(tot, 1e-30, None))
        if np.isfinite(a):
            A.append(a)
        if np.isfinite(r):
            R.append(r)
    if not A or not R:
        continue
    a, r = float(np.median(A)), float(np.median(R))
    rows.append(dict(band=lab, absolute=a, fraction=r, diff=a - r,
                     units=min(len(A), len(R))))
    print(f"{lab:<16}{min(len(A),len(R)):>6}{a:>11.3f}{r:>11.3f}{a-r:>13.3f}")
print("-" * 57)

d = [r["diff"] for r in rows if r["band"] != "full band"]
print(f"\nnormalising by total power brings the earliest usable age forward by")
print(f"{min(d):.3f} to {max(d):.3f} of life, median {np.median(d):.3f}.\n")
fb = [r for r in rows if r["band"] == "full band"]
if fb:
    print(f"the full band is the control: its fraction is 1 by construction, so")
    print(f"absolute {fb[0]['absolute']:.3f} and fraction {fb[0]['fraction']:.3f} "
          f"should agree, and the difference {fb[0]['diff']:+.3f} is the")
    print("measurement's own noise on this comparison.")
print()
print("Early bearing damage redistributes vibration energy toward higher")
print("frequencies before it raises the overall level. An indicator that")
print("tracks the redistribution sees it; one that tracks the level does not.")
json.dump(rows, open(os.path.join(HERE, "shape_vs_level.json"), "w"),
          indent=2, default=float)
