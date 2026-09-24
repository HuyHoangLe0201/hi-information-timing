"""
Which records entered each analysis, and why the others did not.

The paper quotes seventeen PRONOSTIA bearings in one section and sixteen in
another, and never says why.  The pipeline has several places that can silently
drop a record -- a length threshold, a degenerate scale, the rank guard, a curve
that fails to build -- and none of them is accounted for in the manuscript.

That matters more than tidiness.  A bug in the mechanism replication of Section
9.4 kept nine of a hundred turbofan units and discarded ninety-one without any
outward sign: the summary statistic barely moved, so nothing looked wrong, and
the evidence was eleven times thinner than it appeared.  Silent attrition is
invisible precisely when it is worst.  This script makes it visible for every
headline analysis by counting what each source offers and what each entry
condition removes.

Nothing here recomputes an analysis.  It applies the same entry conditions the
analyses apply and counts, so it is cheap and can be rerun whenever a threshold
changes.
"""
import os
import json
import numpy as np

from pipeline import robust_scale, _win
from nonparam import weighted_density

HERE = os.path.dirname(os.path.abspath(__file__))
EPS = 1e-30


def offered(fn, kind):
    """(label, series-provider) for every record a source file holds."""
    p = os.path.join(HERE, fn)
    if not os.path.exists(p):
        return None
    z = np.load(p, allow_pickle=True)
    if kind == "feat":
        names = [str(s) for s in z["featnames"]]
        i = names.index("rms")
        return [(u, np.asarray(z[u][:, i], float))
                for u in sorted(k for k in z.files if k != "featnames")]
    if kind == "band":
        return [(b, z[b].astype(float).sum(axis=1))
                for b in sorted(k for k in z.files if k != "centres")]
    if kind == "cmapss":
        # The probe must be the series the analyses actually build, not a raw
        # column: several C-MAPSS sensors are constant, and taking column zero
        # made every FD001 unit look scale-free when none of them is.
        out = []
        for u in sorted({k.split("__")[0] for k in z.files}):
            k = f"{u}__sensors"
            if k not in z.files:
                continue
            S = np.asarray(z[k], float)
            n = S.shape[0]
            h = S[:max(5, int(0.20 * n))]
            mu = np.median(h, axis=0)
            sd = np.median(np.abs(h - mu), axis=0) * 1.4826
            keep = np.isfinite(sd) & (sd > 0)
            if keep.sum() < 8:
                out.append((u, np.zeros(n)))       # will fail the scale test
                continue
            out.append((u, np.abs((S[:, keep] - mu[keep]) / sd[keep]).sum(axis=1)))
        return out
    return [(u, np.asarray(z[u], float).ravel()) for u in sorted(z.files)]


SOURCES = [
    ("PRONOSTIA, ten channels", "feat_raw.npz", "feat", 60),
    ("PRONOSTIA, 32-band spectra", "fineband.npz", "band", 60),
    ("XJTU-SY, 32-band spectra", "fineband_xjtu.npz", "band", 60),
    ("Turbofan FD001", "cmapss_FD001.npz", "cmapss", 100),
    ("Turbofan FD004", "cmapss_FD004.npz", "cmapss", 100),
]

print("what each source offers, and what the entry conditions remove\n")
print(f"{'source':<28}{'offered':>9}{'too short':>11}{'no scale':>10}"
      f"{'no curve':>10}{'entering':>10}")
print("-" * 78)
rows = []
for label, fn, kind, min_n in SOURCES:
    recs = offered(fn, kind)
    if recs is None:
        print(f"{label:<28}{'file not present':>50}")
        continue
    short = flat = nocurve = 0
    ok = []
    for u, x in recs:
        x = np.asarray(x, float)
        x = x[np.isfinite(x)]
        if len(x) < min_n:
            short += 1
            continue
        s = robust_scale(x)
        if not np.isfinite(s) or s <= 0:
            flat += 1
            continue
        d, _ = weighted_density(x)
        if d is None or not np.isfinite(d).all() or d.sum() <= 0:
            nocurve += 1
            continue
        ok.append(u)
    rows.append(dict(source=label, offered=len(recs), short=short,
                     no_scale=flat, no_curve=nocurve, entering=len(ok),
                     min_n=min_n))
    print(f"{label:<28}{len(recs):>9}{short:>11}{flat:>10}{nocurve:>10}"
          f"{len(ok):>10}")
print("-" * 78)
print(f"Length thresholds are those the analyses use: {SOURCES[0][3]} samples for")
print("the bearing work and 100 for the turbofan, the latter being the limit")
print("Section 5 finds on estimating a budget at all.\n")

# --- the discrepancy the manuscript never explains --------------------------
a = next((r for r in rows if r["source"].startswith("PRONOSTIA, ten")), None)
b = next((r for r in rows if r["source"].startswith("PRONOSTIA, 32")), None)
if a and b:
    print("the seventeen-against-sixteen question\n")
    print(f"  the ten-channel feature set holds {a['offered']} bearings, of which "
          f"{a['entering']} enter")
    print(f"  the 32-band spectra hold {b['offered']}, of which {b['entering']} "
          f"enter")
    if a["offered"] != b["offered"]:
        print(f"  so the difference is in the SOURCES, not in any exclusion: the")
        print(f"  spectral extraction produced {b['offered']} records where the")
        print(f"  scalar-feature extraction produced {a['offered']}.  Analyses")
        print(f"  built on each therefore quote different counts, and the paper")
        print(f"  should say so rather than leave a reader to notice it.")
    else:
        print("  the two sources hold the same number, so the difference lies in")
        print("  an exclusion downstream and is reported per analysis.")
print()

# --- the distortion block, whose screen this script did not know about ------
# This file was written to make silent attrition visible and then missed some:
# the results of Section 2.4 require three hundred samples rather than sixty and
# also need a record on which the two families are ordered, and neither condition
# appears above.  Both are counted here, from the census that classifies every
# record at the documented screen, so the reversed records are visible instead of
# being absorbed into a length count.
_cen = os.path.join(HERE, "ordering_census.json")
dist_rows = []
if os.path.exists(_cen):
    cen = json.load(open(_cen))
    print("the distortion results of Section 2.4, whose screen is 300\n")
    print(f"{'rig':<28}{'offered':>9}{'too short':>11}{'no pair':>10}"
          f"{'reversed':>10}{'entering':>10}")
    print("-" * 78)
    for rig in ("PRONOSTIA", "XJTU-SY"):
        recs = [r for r in cen["records"] if r["rig"] == rig]
        if not recs:
            continue
        short = sum(1 for r in recs if r["n"] < 300)
        long_ = [r for r in recs if r["n"] >= 300]
        nopair = sum(1 for r in long_ if r["cls"] == "unusable")
        rev = sum(1 for r in long_ if r["cls"] == "reversed")
        ent = sum(1 for r in long_ if r["cls"] == "ordered")
        dist_rows.append(dict(rig=rig, offered=len(recs), short=short,
                              no_pair=nopair, reversed=rev, entering=ent,
                              reversed_below_screen=sum(
                                  1 for r in recs
                                  if r["n"] < 300 and r["cls"] == "reversed")))
        print(f"{rig:<28}{len(recs):>9}{short:>11}{nopair:>10}{rev:>10}"
              f"{ent:>10}")
    print("-" * 78)
    _rb = sum(r["reversed_below_screen"] for r in dist_rows)
    _rt = sum(1 for r in cen["records"] if r["cls"] == "reversed")
    print(f"Reversed records above the screen: {sum(r['reversed'] for r in dist_rows)}."
          f"  Reversed records below it: {_rb} of the {_rt} that exist.")
    print("The screen removes every record on which the ordering fails, which is")
    print("a property of the screen and has to be stated with the certificate.\n")

# --- attrition already recorded by the analyses themselves ------------------
print("attrition the analyses record for themselves\n")
extra = []
for fn, key, note in (
        ("turbofan_distribution.json", "skipped", "turbofan, too short"),
        ("mechanism_turbofan.json", None, "mechanism replication"),
        ("linear_vs_shape.json", "dropped_short", "linear span, too short")):
    p = os.path.join(HERE, fn)
    if not os.path.exists(p):
        continue
    d = json.load(open(p))
    if fn == "mechanism_turbofan.json":
        for r in d["fleets"]:
            extra.append(dict(analysis=f"{note}, {r['fleet']}",
                              offered=r["offered"], dropped=r["dropped"]))
            print(f"  {note}, {r['fleet']:<7} offered {r['offered']:>4}, "
                  f"dropped {r['dropped']}")
    elif key in d:
        v = d[key]
        if isinstance(v, dict):
            for k2, n in v.items():
                extra.append(dict(analysis=f"{note} ({k2})", dropped=n))
                print(f"  {note} ({k2}): dropped {n}")
        else:
            extra.append(dict(analysis=note, dropped=v))
            print(f"  {note}: dropped {v}")
print()
tot = sum(e.get("dropped", 0) for e in extra)
print(f"total recorded attrition across those analyses: {tot}")
print("A count of zero is the useful case: it says the analysis used every")
print("record its source offered, which is what the mechanism replication now")
print("does after a bug that had it using nine of a hundred.")

json.dump(dict(sources=rows, recorded=extra, distortion=dist_rows),
          open(os.path.join(HERE, "attrition.json"), "w"), indent=2,
          default=float)
