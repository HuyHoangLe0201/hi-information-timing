r"""How many records does the purged scheme lose, and does that explain the gap?

Part 2 of oof_leak.py drops any record the purged scheme cannot fit, from BOTH
arms, so that the comparison is paired.  That keeps the comparison honest but
changes the population: the uncorrected gap it reports is not the paper's,
because it is computed on a different set of records.  On XJTU-SY, whose
shortest record is 42 samples, a widened window can fail often, and any change
in the gap has to be separated from that.
"""
import os
import io
import json
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
src = io.open(os.path.join(HERE, "oof_leak.py"), encoding="utf-8").read()
head = src[:src.index("# " + "=" * 77)]
mid = src[src.index("from scipy.signal import savgol_filter as _sg"):
          src.index('print("  %-12s %-13s %10s %10s %10s"')]
ns = {"__file__": os.path.join(HERE, "oof_leak.py")}
exec(head + mid, ns)
density_pair, MEASURES, RIGS = ns["density_pair"], ns["MEASURES"], ns["RIGS"]

ROWS = []
print("%-12s %8s %8s %8s %10s %10s"
      % ("rig", "bearings", "kept", "total", "kept %", "shortest"))
print("-" * 62)
for rig, fn in RIGS.items():
    p = os.path.join(HERE, fn)
    if not os.path.exists(p):
        continue
    z = np.load(p)
    f = z["centres"] / 1000.0
    BK = sorted(k for k in z.files if k != "centres")
    lens = [len(z[b]) for b in BK]
    kept = total = 0
    lost_len = []
    for lab, kind, fun in MEASURES:
        vals = fun(z[BK[0]].astype(float), f)  # touch once to fail fast
        for b in BK:
            total += 1
            if density_pair(fun(z[b].astype(float), f)) is not None:
                kept += 1
            else:
                lost_len.append(len(z[b]))
    ROWS.append(dict(rig=rig, bearings=len(BK), kept=kept, total=total,
                     kept_frac=kept / total, shortest=int(min(lens)),
                     lost_min=int(min(lost_len)) if lost_len else None,
                     lost_max=int(max(lost_len)) if lost_len else None))
    print("%-12s %8d %8d %8d %9.0f%% %10d"
          % (rig, len(BK), kept, total, 100.0 * kept / total, min(lens)))
    if lost_len:
        print("             lost cells come from records of %d to %d samples"
              % (min(lost_len), max(lost_len)))


json.dump(ROWS, open(os.path.join(HERE, "oof_attrition.json"), "w"), indent=2)
print("written to oof_attrition.json")
