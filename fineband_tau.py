"""
The frequency question again, at four times the resolution.

Seven bands across 12.8 kHz could not settle whether a structural resonance
makes the high-frequency channels usable early: the two rigs disagreed on the
best band and band energy growth correlated with the earliest usable age in
opposite directions. At 1.8 kHz per band a resonance and its surroundings fall
in the same bin, so the test had little chance.

Thirty-two bands of 400 Hz, computed from the raw acquisitions, give a real
chance. Three things are asked of the result:

  shape        -- does tau_min fall smoothly with frequency, or is there a
                  localised minimum? A resonance produces the second.
  consistency  -- is the minimum in the same place for every bearing, or does
                  each bearing have its own? A structural resonance belongs to
                  the rig and should be shared.
  condition    -- does it move between the three operating conditions? A
                  resonance should not; a damage-frequency effect should.

Bearing1_4 failed extraction and is absent; sixteen of seventeen remain.
"""
import os
import json
import numpy as np
from nonparam import weighted_density, estimate_floor, quantile

HERE = os.path.dirname(os.path.abspath(__file__))
Q = 0.35
SEED = 13579


def tau_of(x, rng):
    dens, res = weighted_density(np.asarray(x, float))
    if dens is None:
        return np.nan, np.nan
    d = np.clip(dens - estimate_floor(dens, res, "surrogate", rng, reps=6),
                0.0, None)
    if d.sum() <= 0:
        return np.nan, np.nan
    F = np.cumsum(d) / d.sum()
    t = quantile(F, Q)
    k = max(1, min(int(round(t * len(d))) - 1, len(d) - 1))
    rk = float(dens[:k + 1].sum())
    return t, (float(d[:k + 1].sum()) / rk if rk > 0 else np.nan)


z = np.load(os.path.join(HERE, "fineband.npz"))
centres = z["centres"] / 1000.0                    # kHz
BK = sorted(k for k in z.files if k != "centres")
print(f"{len(BK)} bearings, {len(centres)} bands "
      f"({centres[1]-centres[0]:.1f} kHz wide)\n")

rng = np.random.default_rng(SEED)
per_bearing = {}
for b in BK:
    M = z[b]
    row = []
    for j in range(M.shape[1]):
        t, s = tau_of(M[:, j], rng)
        row.append((t, s))
    per_bearing[b] = row
    best = int(np.nanargmin([r[0] for r in row]))
    print(f"  {b:<14} n={M.shape[0]:>5}  earliest band "
          f"{centres[best]:>5.1f} kHz  tau {row[best][0]:.3f}", flush=True)

CONDS = {"1": [b for b in BK if b.startswith("Bearing1_")],
         "2": [b for b in BK if b.startswith("Bearing2_")],
         "3": [b for b in BK if b.startswith("Bearing3_")]}

print("\nearliest usable age against frequency, by operating condition\n")
prof = {}
for c, keys in CONDS.items():
    if not keys:
        continue
    med = []
    for j in range(len(centres)):
        v = [per_bearing[b][j][0] for b in keys
             if np.isfinite(per_bearing[b][j][0])]
        med.append(float(np.median(v)) if len(v) >= 2 else np.nan)
    prof[c] = med

hdr = "".join(f"{c:>10}" for c in prof)
print(f"{'kHz':>6}{hdr}")
print("-" * (6 + 10 * len(prof)))
for j, f in enumerate(centres):
    cells = "".join(
        ("       n/a" if not np.isfinite(prof[c][j]) else f"{prof[c][j]:>10.3f}")
        for c in prof)
    print(f"{f:>6.1f}{cells}")
print("-" * (6 + 10 * len(prof)))

print("\nwhere the minimum sits\n")
for c in prof:
    v = np.array(prof[c], float)
    if not np.any(np.isfinite(v)):
        continue
    j = int(np.nanargmin(v))
    print(f"  condition {c}: {centres[j]:.1f} kHz  (tau {v[j]:.3f}), "
          f"band range {np.nanmin(v):.3f}-{np.nanmax(v):.3f}")

print("\nis the minimum shared across bearings within a condition?\n")
for c, keys in CONDS.items():
    if len(keys) < 3:
        continue
    js = []
    for b in keys:
        v = [r[0] for r in per_bearing[b]]
        if np.any(np.isfinite(v)):
            js.append(centres[int(np.nanargmin(v))])
    if js:
        print(f"  condition {c}: per-bearing minima at "
              + ", ".join(f"{x:.1f}" for x in sorted(js)) + " kHz"
              + f"   (spread {max(js)-min(js):.1f} kHz)")

fin = {c: np.array(v, float) for c, v in prof.items()}
ks = [c for c in fin if np.sum(np.isfinite(fin[c])) > 4]
if len(ks) >= 2:
    print("\nagreement between conditions on the frequency profile")
    for i in range(len(ks)):
        for j in range(i + 1, len(ks)):
            a, b = fin[ks[i]], fin[ks[j]]
            m = np.isfinite(a) & np.isfinite(b)
            if m.sum() > 4:
                ra = np.argsort(np.argsort(a[m])); rb = np.argsort(np.argsort(b[m]))
                print(f"  {ks[i]} vs {ks[j]}: {float(np.corrcoef(ra, rb)[0,1]):+.2f}"
                      f"  ({m.sum()} bands)")
json.dump(dict(centres=centres.tolist(),
               profiles={c: list(map(float, v)) for c, v in prof.items()},
               per_bearing={b: [float(r[0]) for r in v]
                            for b, v in per_bearing.items()}),
          open(os.path.join(HERE, "fineband_tau.json"), "w"), indent=2)
