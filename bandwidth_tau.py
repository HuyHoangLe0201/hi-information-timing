"""
It is bandwidth, not centre frequency.

Thirty-two bands of 400 Hz put almost every band at tau = 0.97, while the coarse
4-10 kHz channel -- which is three 2 kHz bands added together -- sits at 0.24.
Narrowing the band made things worse, uniformly, and the per-bearing minima
scattered over the whole spectrum with no shared location. So there is no narrow
resonance carrying the early information.

The obvious alternative is that the information is spread across a wide range of
frequencies and integrating over more of it is what makes an indicator usable
early: a wider band collects more signal against noise that partly averages out.

That is directly testable now the narrow bands exist. Adjacent bands are summed
into progressively wider ones at a fixed centre, and tau_min re-measured. If it
falls with width and does not depend much on where the centre is, the useful
statement for an instrumentation engineer is about bandwidth, not about hunting
for a resonance -- which is the opposite of the advice the resonance-demodulation
tradition would give.
"""
import os
import json
import numpy as np
from nonparam import weighted_density, estimate_floor, quantile

HERE = os.path.dirname(os.path.abspath(__file__))
Q = 0.35
SEED = 24680
WIDTHS = (1, 2, 4, 8, 16, 32)          # in units of 400 Hz


def tau_of(x, rng):
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
NB = len(centres)
rng = np.random.default_rng(SEED)

print(f"summing adjacent 400 Hz bands into wider ones\n")
print(f"{'width':>8}{'kHz':>7}{'bands':>7}{'placements':>12}"
      f"{'tau: median':>13}{'best':>8}{'worst':>8}")
print("-" * 63)
rows = []
for w in WIDTHS:
    starts = list(range(0, NB - w + 1, max(1, w // 2)))
    per_start = []
    for s in starts:
        v = []
        for b in BK:
            x = z[b][:, s:s + w].sum(axis=1)
            t = tau_of(x, rng)
            if np.isfinite(t):
                v.append(t)
        if len(v) >= 4:
            per_start.append((float(np.median(v)),
                              float(centres[s] + (w - 1) * 0.2)))
    if not per_start:
        continue
    taus = [p[0] for p in per_start]
    best = min(per_start, key=lambda p: p[0])
    rows.append(dict(width_bands=w, width_kHz=w * 0.4,
                     placements=len(per_start),
                     median=float(np.median(taus)),
                     best=best[0], best_centre=best[1], worst=max(taus)))
    r = rows[-1]
    print(f"{w*0.4:>7.1f}k{'':>1}{'':>6}{w:>7}{len(per_start):>12}"
          f"{r['median']:>13.3f}{r['best']:>8.3f}{r['worst']:>8.3f}")
print("-" * 63)
print("placements = how many positions of that width fit in the spectrum;")
print("the median is over placements, so it is not one lucky band.\n")

if len(rows) >= 3:
    lw = np.log([r["width_kHz"] for r in rows])
    lt = np.log([r["median"] for r in rows])
    slope = float(np.polyfit(lw, lt, 1)[0])
    rc = float(np.corrcoef(lw, lt)[0, 1])
    print(f"tau_min against bandwidth: slope {slope:+.3f}, r {rc:+.2f}")
    print(f"widest band ({rows[-1]['width_kHz']:.1f} kHz) reaches "
          f"{rows[-1]['median']:.3f}; narrowest "
          f"({rows[0]['width_kHz']:.1f} kHz) sits at {rows[0]['median']:.3f}")

    # does the centre matter once the width is fixed?
    print("\nspread across placements at each width "
          "(how much the centre matters)\n")
    print(f"{'width kHz':>10}{'best':>9}{'worst':>9}{'spread':>9}")
    print("-" * 37)
    for r in rows:
        print(f"{r['width_kHz']:>10.1f}{r['best']:>9.3f}{r['worst']:>9.3f}"
              f"{r['worst']-r['best']:>9.3f}")
    print("-" * 37)
    wide = [r for r in rows if r["width_kHz"] >= 3]
    if wide:
        sp = float(np.median([r["worst"] - r["best"] for r in wide]))
        print(f"\nat 3 kHz and wider the centre still moves tau by {sp:.3f} "
              f"of life,")
        print("so width is not the whole story -- but it is the larger effect")
        print(f"({rows[0]['median'] - rows[-1]['median']:.3f} from narrowest "
              f"to widest).")
json.dump(rows, open(os.path.join(HERE, "bandwidth_tau.json"), "w"), indent=2,
          default=float)
