"""
The family gap under a two-level bootstrap: units, then measures.

Definition 2 makes the family gap a median of medians, and the interval the
paper quotes resamples units only, so it covers the variation between bearings
of one rig and not the identity of the measures that make up each family.  The
leave-one-out control drops one measure at a time; it bounds a single
favourable member but is not an interval.

Here both levels are resampled.  Each replicate draws bearings with
replacement, then draws the six distributional and the four amount measures
with replacement within their own family, rebuilds each drawn measure's age as
a median over the drawn bearings, and forms the gap.  The per-unit ages are the
cached ones every other interval in the paper is built from.

    python hier_bootstrap.py   ->  hier_bootstrap.json
"""
import os
import json
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
SEED = 20260924
BOOT = 4000
PER = json.load(open(os.path.join(HERE, "tau_percell_v3.json")))
DIST = ["spectral entropy", "spectral centroid", "spectral spread",
        "spectral Gini", "top-8-band share", "high/low band ratio"]
AMT = ["total power", "peak band power", "high-band power", "log total power"]


def gap(rows, units, dist, amt):
    sub = [rows[i] for i in units]

    def med(nm):
        v = [r[nm] for r in sub if r.get(nm) is not None]
        return float(np.median(v)) if v else np.nan

    d = [v for v in (med(n) for n in dist) if np.isfinite(v)]
    a = [v for v in (med(n) for n in amt) if np.isfinite(v)]
    if not d or not a:
        return np.nan
    return float(np.median(a) - np.median(d))


rng = np.random.default_rng(SEED)
out = {}
print(f"{'rig':<11}{'units':>6}{'gap':>8}{'units only':>22}{'units and measures':>24}")
print("-" * 71)
for rig, rows in PER.items():
    n = len(rows)
    g = gap(rows, range(n), DIST, AMT)
    unit_only, both = [], []
    for _ in range(BOOT):
        u = rng.integers(0, n, n)
        unit_only.append(gap(rows, u, DIST, AMT))
        d = [DIST[i] for i in rng.integers(0, len(DIST), len(DIST))]
        a = [AMT[i] for i in rng.integers(0, len(AMT), len(AMT))]
        both.append(gap(rows, u, d, a))
    unit_only = np.array([x for x in unit_only if np.isfinite(x)])
    both = np.array([x for x in both if np.isfinite(x)])
    lo1, hi1 = np.percentile(unit_only, [2.5, 97.5])
    lo2, hi2 = np.percentile(both, [2.5, 97.5])
    out[rig] = dict(units=n, gap=g, unit_lo=float(lo1), unit_hi=float(hi1),
                    hier_lo=float(lo2), hier_hi=float(hi2),
                    hier_positive_share=float((both > 0).mean()))
    print(f"{rig:<11}{n:>6}{g:>8.3f}   [{lo1:+.3f}, {hi1:+.3f}]"
          f"      [{lo2:+.3f}, {hi2:+.3f}]")
print()
for rig, v in out.items():
    print(f"{rig}: the two-level interval {'excludes' if v['hier_lo'] > 0 else 'does NOT exclude'}"
          f" zero; {100 * v['hier_positive_share']:.1f}% of replicates positive")
json.dump(dict(boot=BOOT, seed=SEED, rigs=out),
          open(os.path.join(HERE, "hier_bootstrap.json"), "w"), indent=2)
