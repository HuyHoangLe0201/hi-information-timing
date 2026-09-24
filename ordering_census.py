r"""Every record, classified: ordered, reversed, or unusable.

Section 2.4 says the pair "is ordered on all sixteen records that carry both
families".  Seventeen records carry both families.  The sixteen are the ones the
distortion results admit, at a length screen of three hundred that Section 5 does
not document, and the seventeenth is not merely absent from the count: on it the
ordering is REVERSED.  A screen that happens to remove the only counterexample is
the worst kind of screen, whatever the reason it was set for.

So the census is taken here, on both rigs, at the length screen the manuscript
documents rather than the one these results use.  Each record is put in one of
three classes and none is dropped silently:

  unusable  -- one family gives no finite age at the demand, so no pair exists;
  ordered   -- the earliest distributional channel precedes the earliest amount;
  reversed  -- it does not.

The margin is reported with each reversal, in fractions of a life and in samples,
because a reversal by one sample on a short record is a different fact from a
reversal by a tenth of a life.
"""
import os
import json
import numpy as np

from pipeline import robust_scale
from nonparam import weighted_density, estimate_floor

HERE = os.path.dirname(os.path.abspath(__file__))
rng = np.random.default_rng(90210)
OUT = {}
Q = 0.35
THETA = Q / (1.0 - Q)
MIN_N = 60                     # the screen Section 5 documents
AMOUNT = ("rms", "peak")
DISTRIB = ("b4_6", "b6_8", "b8_10")
DATASETS = (("PRONOSTIA", "feat_raw.npz"), ("XJTU-SY", "xjtu_feat.npz"))


def age_at_odds(F, x):
    p = x / (1.0 + x)
    if p >= 1.0:
        return np.inf
    i = int(np.searchsorted(F, p))
    return np.inf if i >= len(F) else (i + 1) / len(F)


def age_of(x):
    """The age at which this channel's odds reach the demand, or None."""
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    if len(x) < MIN_N:
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


print("=" * 92)
print("Every record on both rigs, at the documented screen of %d" % MIN_N)
print("=" * 92)
print("""
The earliest channel of each family is taken, as the ordering does.  A record is
reversed when the amount family reaches the demand no later than the
distributional one.  The margin is distributional minus amount, so a positive
margin is a reversal and its size is what the reversal is worth.
""")
print("  %-10s %-24s %7s %9s %9s %10s %10s"
      % ("rig", "record", "n", "distrib", "amount", "margin", "class"))
print("  " + "-" * 84)
rows = []
for dn, fn in DATASETS:
    z = np.load(os.path.join(HERE, fn), allow_pickle=True)
    FN = [str(s) for s in z["featnames"]]
    for u in [k for k in z.files if k != "featnames"]:
        n = int(z[u].shape[0])
        got = {}
        for grp, names in (("d", DISTRIB), ("a", AMOUNT)):
            cand = [age_of(z[u][:, FN.index(nm)]) for nm in names if nm in FN]
            cand = [c for c in cand if c is not None]
            if cand:
                got[grp] = min(cand)
        if "d" not in got or "a" not in got:
            cls, margin = "unusable", None
        else:
            margin = got["d"] - got["a"]
            cls = "ordered" if got["d"] < got["a"] else "reversed"
        rows.append(dict(rig=dn, unit=u, n=n, d=got.get("d"), a=got.get("a"),
                         margin=margin, cls=cls))
        print("  %-10s %-24s %7d %9s %9s %10s %10s"
              % (dn, u[:24], n,
                 "%.4f" % got["d"] if "d" in got else "-",
                 "%.4f" % got["a"] if "a" in got else "-",
                 "%+.4f" % margin if margin is not None else "-", cls))
OUT["records"] = rows

for dn, _ in DATASETS:
    sub = [r for r in rows if r["rig"] == dn]
    for cls in ("ordered", "reversed", "unusable"):
        OUT["%s_%s" % (dn.replace("-", "_"), cls)] = sum(
            1 for r in sub if r["cls"] == cls)
    OUT["%s_total" % dn.replace("-", "_")] = len(sub)

rev = [r for r in rows if r["cls"] == "reversed"]
OUT["reversed_total"] = len(rev)
OUT["usable_total"] = sum(1 for r in rows if r["cls"] != "unusable")
if rev:
    OUT["reversals"] = [dict(rig=r["rig"], unit=r["unit"], n=r["n"],
                             margin=r["margin"],
                             samples=r["margin"] * r["n"]) for r in rev]
    _w = max(rev, key=lambda r: r["margin"])
    OUT["worst_reversal_margin"] = float(_w["margin"])
    OUT["worst_reversal_unit"] = _w["unit"]
    OUT["worst_reversal_samples"] = float(_w["margin"] * _w["n"])

print("""
  %d of the %d records that give a usable pair are reversed, %d on the first rig
  and %d on the second.  The largest reversal is %s at %+.4f of a life, which is
  %.1f samples of its %d.

  Every reversal is a record the distortion results of Section 2.4 do not see:
  their screen of three hundred admits none of them.  That is a fact about the
  screen and it has to be stated with the certificate, because a guarantee
  computed on the records that agree with a claim is not a guarantee about the
  claim.
""" % (OUT["reversed_total"], OUT["usable_total"],
       sum(1 for r in rev if r["rig"] == "PRONOSTIA"),
       sum(1 for r in rev if r["rig"] == "XJTU-SY"),
       OUT.get("worst_reversal_unit", "-"),
       OUT.get("worst_reversal_margin", 0.0),
       OUT.get("worst_reversal_samples", 0.0),
       next((r["n"] for r in rev
             if r["unit"] == OUT.get("worst_reversal_unit")), 0)))

json.dump(OUT, open(os.path.join(HERE, "ordering_census.json"), "w"), indent=2,
          default=float)
print("written to ordering_census.json")
