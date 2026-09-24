r"""The deployable comparison, against the comparator the sentence names.

Section 10 says: "Among those [quantities fixed before the record is seen], a
distributional statistic is earlier than the best TRANSFERABLE linear combination
on 15 of 16 bearings, by a paired median 0.804 of a lifetime", and the
introduction repeats the count and the margin as the deployable result.

linear_vs_shape.py computes that pair from lin_own, the direction fitted on the
very record it is scored on.  Its own printout says so -- "paired against the
linear side at its most optimistic (own fit)" -- and Figure 8's caption says so
too.  Only the two sentences that carry the claim say "transferable", and a
direction fitted on the record it is scored on is precisely not deployable, which
is the property the paragraph is about.

So the paired comparison is redone here against lin_tr, the direction averaged
over the other bearings, which is what "fixed before the record is seen" means.
Two records need care rather than dropping: on them the transferable direction
never reaches the demand at all, so its age is not missing but censored above,
and discarding them would discard two records on which the shape statistic wins
outright.  They are counted as wins with the gap taken at its lower bound.

Nothing is recomputed; every age is read from linear_vs_shape.json.
"""
import os
import json
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = {}
LS = json.load(open(os.path.join(HERE, "linear_vs_shape.json")))
R = LS["per_unit"]

print("=" * 92)
print("Shape against the two linear comparators, bearing by bearing")
print("=" * 92)
print("""
"own" is the leading direction of the record it is scored on; "transferable" is
the mean of the other bearings' directions, normalised.  A positive gap is a win
for the shape statistic.  Where the transferable direction never reaches the
demand its age is censored above at one, and the gap is a lower bound.
""")
print("  %-14s %8s %9s %9s %9s %10s %10s"
      % ("bearing", "samples", "shape", "own", "transf", "gap own",
         "gap transf"))
print("  " + "-" * 76)
rows = []
for r in sorted(R, key=lambda r: r["unit"]):
    sh, own, tr = r["best_shape"], r["lin_own"], r["lin_tr"]
    cens = not np.isfinite(tr)
    g_own = own - sh if np.isfinite(own) and np.isfinite(sh) else np.nan
    g_tr = (1.0 - sh) if cens else (tr - sh)
    rows.append(dict(unit=r["unit"], m=r["m"], shape=sh, own=own,
                     tr=(None if cens else tr), censored=cens,
                     gap_own=float(g_own), gap_tr=float(g_tr)))
    print("  %-14s %8d %9.4f %9.4f %9s %+10.4f %10s"
          % (r["unit"], r["m"], sh, own,
             "censored" if cens else "%.4f" % tr, g_own,
             ("> %+.4f" % g_tr) if cens else "%+.4f" % g_tr))
OUT["per_unit"] = rows

go = np.array([r["gap_own"] for r in rows])
gt = np.array([r["gap_tr"] for r in rows])
fin = np.array([not r["censored"] for r in rows])
OUT["n"] = len(rows)
OUT["own_wins"] = int((go > 0).sum())
OUT["own_gap"] = float(np.median(go))
OUT["transfer_wins"] = int((gt > 0).sum())
OUT["transfer_gap"] = float(np.median(gt))
OUT["censored"] = int((~fin).sum())
OUT["transfer_wins_finite"] = int((gt[fin] > 0).sum())
OUT["transfer_finite"] = int(fin.sum())
OUT["transfer_gap_finite"] = float(np.median(gt[fin]))

# which records the two comparators disagree about, and by how much
dis = [r for r in rows if (r["gap_own"] > 0) != (r["gap_tr"] > 0)]
OUT["disagreeing"] = [dict(unit=r["unit"], m=r["m"], gap_own=r["gap_own"],
                           gap_tr=r["gap_tr"]) for r in dis]
inv = [r for r in rows if not r["censored"] and r["tr"] < r["own"] - 1e-9]
OUT["transfer_earlier_than_own"] = len(inv)
OUT["transfer_earlier_units"] = [r["unit"] for r in inv]

print("""
  Against the own fit the shape statistic wins on %d of %d with a paired median of
  %+.4f, which are the figures the manuscript quotes.  Against the transferable
  direction, the comparison the sentence names and the argument needs, it wins on
  %d of %d with a paired median of %+.4f; %d of those wins are censored, the
  transferable direction never reaching the demand, so the median is a lower
  bound.  Over the %d records where both are finite the median is %+.4f.
""" % (OUT["own_wins"], OUT["n"], OUT["own_gap"], OUT["transfer_wins"],
       OUT["n"], OUT["transfer_gap"], OUT["censored"], OUT["transfer_finite"],
       OUT["transfer_gap_finite"]))

if dis:
    d = dis[0]
    print("""  The two counts differ on %d record%s.  On %s the shape statistic is earlier
  than the own fit by %.4f of a life and later than the transferable direction by
  %.4f, both of which are far inside the sampling spread of such a margin: the
  decimation study of reversal_resolution.py puts that spread at 0.013 even on a
  record of twelve hundred samples, and this one runs to %d.  The difference
  between fifteen and %d is therefore a difference between two point estimates
  that the data does not separate.
""" % (len(dis), "" if len(dis) == 1 else "s", d["unit"],
       abs(d["gap_own"]), abs(d["gap_tr"]), d["m"], OUT["transfer_wins"]))

print("""  One more thing follows from the table and is worth stating, because the
  manuscript calls the own fit "generous".  It is generous in what it has seen,
  but the direction is chosen to maximise INFORMATION and is scored on when it
  reaches a demand, and those are not the same objective.  On %d of the %d records
  where both are finite the transferable direction is in fact the earlier of the
  two, so fitting on the record scored does not guarantee the earlier age.
""" % (OUT["transfer_earlier_than_own"], OUT["transfer_finite"]))

json.dump(OUT, open(os.path.join(HERE, "linear_transfer_paired.json"), "w"),
          indent=2, default=float)
print("written to linear_transfer_paired.json")
