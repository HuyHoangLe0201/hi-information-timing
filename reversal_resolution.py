r"""Are the four reversals real, or inside the measurement's own resolution?

ordering_census.py finds four records of thirty on which the amount family
reaches the demand before the distributional one, with margins of 0.009, 0.025,
0.205 and 0.237 of a life.  All four are shorter than three hundred snapshots.
Two readings fit that: short records are where the tendency genuinely fails, or
short records are where the margin cannot be measured.  They differ in what the
paper should say, so they are separated here rather than argued about.

The separation uses decimation again, and on the margin itself.  A long bearing
decimated by k gives k subsamples of one length differing only in phase; the
spread of the margin across them is its sampling error at that length, with the
bearing held fixed.  If that spread at a hundred and fifty samples is larger than
0.237, the two large reversals are inside the noise and the census counts
resolution rather than physics.  If it is much smaller, they are real and the
paper has four counterexamples to report as such.

The confound of the earlier study does not arise here: the margin is a difference
of two ages read off the same record with the same pipeline, and neither age
depends on the serial correlation that decimation thins.
"""
import os
import json
import numpy as np

from nonparam import weighted_density, estimate_floor

HERE = os.path.dirname(os.path.abspath(__file__))
rng = np.random.default_rng(13579)
OUT = {}
Q = 0.35
THETA = Q / (1.0 - Q)
AMOUNT = ("rms", "peak")
DISTRIB = ("b4_6", "b6_8", "b8_10")
KS = (1, 2, 4, 8, 16)


def age_at_odds(F, x):
    p = x / (1.0 + x)
    if p >= 1.0:
        return np.inf
    i = int(np.searchsorted(F, p))
    return np.inf if i >= len(F) else (i + 1) / len(F)


def age_of(x):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    if len(x) < 60:
        return None
    dens, res = weighted_density(x)
    if dens is None:
        return None
    d = np.clip(dens - estimate_floor(dens, res, "surrogate", rng, reps=6),
                0.0, None)
    if d.sum() <= 0:
        return None
    a = age_at_odds(np.cumsum(d) / d.sum(), THETA)
    return float(a) if np.isfinite(a) else None


def margin_of(block, FN):
    got = {}
    for grp, names in (("d", DISTRIB), ("a", AMOUNT)):
        cand = [age_of(block[:, FN.index(nm)]) for nm in names if nm in FN]
        cand = [c for c in cand if c is not None]
        if cand:
            got[grp] = min(cand)
    if "d" not in got or "a" not in got:
        return None
    return got["d"] - got["a"]


z = np.load(os.path.join(HERE, "feat_raw.npz"), allow_pickle=True)
FN = [str(s) for s in z["featnames"]]
LONG = [k for k in z.files
        if k != "featnames" and z[k].shape[0] >= 2200]

print("=" * 92)
print("The sampling spread of the family margin, against record length")
print("=" * 92)
print("""
Each of the long bearings is decimated by k and the margin recomputed for every
phase.  The spread is across phases, so the bearing, the channels and the
pipeline are all held fixed and only the length changes.
""")
print("  %-8s %9s %9s %12s %12s %8s"
      % ("k", "length", "records", "spread", "worst |dev|", "phases"))
print("  " + "-" * 62)
rows = []
base = {}
for u in LONG:
    m = margin_of(z[u], FN)
    if m is not None:
        base[u] = m
for k in KS:
    spreads, devs, ph = [], [], 0
    for u in LONG:
        got = []
        for off in range(k):
            m = margin_of(z[u][off::k], FN)
            if m is not None:
                got.append(m)
        ph += len(got)
        if k == 1 or len(got) < 2:
            continue
        spreads.append(float(np.std(got, ddof=1)))
        if u in base:
            devs.append(float(np.max(np.abs(np.array(got) - base[u]))))
    ln = int(np.median([z[u].shape[0] for u in LONG]) / k)
    r = dict(k=k, length=ln, records=len(LONG),
             spread=float(np.median(spreads)) if spreads else 0.0,
             worst_dev=float(np.median(devs)) if devs else 0.0, phases=ph)
    rows.append(r)
    print("  %-8d %9d %9d %12.4f %12.4f %8d"
          % (k, ln, len(LONG), r["spread"], r["worst_dev"], ph))
OUT["decimation"] = rows
OUT["long_records"] = len(LONG)

# --- put the four reversals beside the resolution at their own lengths ------
cen = json.load(open(os.path.join(HERE, "ordering_census.json")))
rev = cen.get("reversals", [])
print()
print("  %-24s %7s %10s %14s %10s"
      % ("reversed record", "n", "margin", "spread at n", "margin/sd"))
print("  " + "-" * 70)


def spread_at(n):
    """Interpolate the measured spread to a record length, on a log scale."""
    xs = np.array([r["length"] for r in rows if r["spread"] > 0], float)
    ys = np.array([r["spread"] for r in rows if r["spread"] > 0], float)
    if len(xs) < 2:
        return np.nan
    o = np.argsort(xs)
    return float(np.interp(np.log(n), np.log(xs[o]), ys[o]))


out_rev = []
for r in rev:
    s = spread_at(r["n"])
    out_rev.append(dict(unit=r["unit"], n=r["n"], margin=r["margin"],
                        spread=s, z=r["margin"] / s if s > 0 else np.nan))
    print("  %-24s %7d %10.4f %14.4f %10.2f"
          % (r["unit"][:24], r["n"], r["margin"], s, out_rev[-1]["z"]))
OUT["reversals"] = out_rev
OUT["max_z"] = float(max((r["z"] for r in out_rev), default=np.nan))
OUT["n_above_two_sd"] = int(sum(1 for r in out_rev if r["z"] > 2.0))
OUT["spread_at_150"] = spread_at(150)
OUT["spread_at_300"] = spread_at(300)

print("""
  The margin's sampling spread is %.3f at a length of one hundred and fifty and
  %.3f at three hundred, measured on bearings that are not themselves reversed.
  Against that, %d of the %d reversals exceed two standard deviations and the
  largest reaches %.2f.

  The reading follows from those numbers rather than from a preference, and it is
  what the manuscript should say about the four records.
""" % (OUT["spread_at_150"], OUT["spread_at_300"], OUT["n_above_two_sd"],
       len(out_rev), OUT["max_z"]))

json.dump(OUT, open(os.path.join(HERE, "reversal_resolution.json"), "w"),
          indent=2, default=float)
print("written to reversal_resolution.json")
