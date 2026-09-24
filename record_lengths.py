r"""Two record properties the estimator comparisons turn on.

Section 8.4 and Supplementary Section S5 read the causal result against the
length of the records it is estimated from: the causal window is
max(7, 0.08n) samples, so the two rigs give one-sided quadratics of very
different widths and the collapse of the XJTU-SY gap cannot be separated from
that.  Those lengths were quoted from a one-off calculation.  They are a number
the paper depends on, so they come from a script and a stored file like every
other number in it.

The lengths are of the fine-band spectra the ten family measures are computed
from, which is what the causal estimator actually runs on -- not of the raw
acquisitions, which are counted separately in Section 3.  The window is forced
odd, as pipeline._win forces it, so the number stored here is the one the
estimator uses rather than 0.08n rounded down.

The second is the flat stretches in the battery channels.  clock_scale.py can
read the term a clock-indexed noise scale would add on half the battery
channels only, because on the other half the rolling scale collapses; this
counts the runs of identical consecutive values that cause it, so the sentence
reporting that in Section 9.6 has a source.

    python record_lengths.py   ->  record_lengths.json
"""
import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
RIGS = (("PRONOSTIA", "fineband.npz"), ("XJTU", "fineband_xjtu.npz"))
SMOOTH_FRAC = 0.08
MIN_WIN = 7

out = {}
print("%-12s%10s%9s%9s%9s%9s%14s"
      % ("rig", "bearings", "median", "mean", "min", "max", "causal window"))
print("-" * 76)
for rig, fn in RIGS:
    path = os.path.join(HERE, fn)
    if not os.path.exists(path):
        print("%-12s  %s not found" % (rig, fn))
        continue
    z = np.load(path)
    L = np.array([z[k].shape[0] for k in z.files if k != "centres"], float)
    med = float(np.median(L))
    win = max(MIN_WIN, int(SMOOTH_FRAC * med))
    win += (win % 2 == 0)          # the filter window is forced odd, as in _win
    out[rig] = dict(bearings=int(L.size), median_n=med, mean_n=float(L.mean()),
                    min_n=float(L.min()), max_n=float(L.max()),
                    causal_window=int(win))
    print("%-12s%10d%9d%9d%9d%9d%14d"
          % (rig, L.size, med, L.mean(), L.min(), L.max(), win))

print("-" * 76)
if len(out) == 2:
    a, b = out["PRONOSTIA"], out["XJTU"]
    print("the one-sided quadratic is %d samples wide on PRONOSTIA against %d on"
          " XJTU-SY" % (a["causal_window"], b["causal_window"]))
    out["ratio_median_n"] = a["median_n"] / b["median_n"]
    print("a factor of %.1f in median record length"
          % out["ratio_median_n"])


# ------------------------------------------- flat stretches in the battery --
# A rolling robust scale is zero wherever more than half its window holds one
# value, and sigma'/sigma is then undefined.  Six of the twelve battery
# channels carry such a run; the other six do not, which is why Section 9.6
# reports the clock-scale term on half of them.
bat = os.path.join(HERE, "battery_raw.npz")
if os.path.exists(bat):
    z = np.load(bat)
    flat, rows = 0, []
    for cell in sorted({k.split("__")[0] for k in z.files}):
        for ind in ("cap", "Re", "Rct"):
            key = "%s__%s" % (cell, ind)
            if key not in z.files:
                continue
            x = np.asarray(z[key], float)
            x = x[np.isfinite(x)]
            if len(x) < 10:
                continue
            run, cur = 1, 1
            for i in range(1, len(x)):
                cur = cur + 1 if x[i] == x[i - 1] else 1
                run = max(run, cur)
            rows.append(("%s/%s" % (cell, ind), len(x), int(run)))
            flat += run > 1
    out["battery"] = dict(channels=len(rows), with_flat_run=int(flat),
                          longest_run=int(max(r[2] for r in rows)),
                          per_channel={r[0]: r[2] for r in rows})
    print()
    print("%-14s%8s%22s" % ("battery channel", "n", "longest identical run"))
    print("-" * 46)
    for nm, n, run in rows:
        print("%-14s%8d%22d" % (nm, n, run))
    print("-" * 46)
    print("%d of %d channels carry a run, the longest being %d samples"
          % (flat, len(rows), max(r[2] for r in rows)))

json.dump(out, open(os.path.join(HERE, "record_lengths.json"), "w"), indent=1)
print("written to record_lengths.json")
