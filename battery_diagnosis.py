"""
How much of the cell-dataset failure is record length, and how much is tails?

Two controls exist, each isolating one property:

  length -- turbofan records decimated to progressively fewer samples, keeping
            the physical span, which is what a lower-rate sensor would give;
  tails  -- the same cell records analysed with a non-robust scale estimator,
            which stops discounting the heavy residual tails.

The question is quantitative: does the decimation curve, read at the cells' own
record length, land on the cells' measured value? If it does for the slope, the
slope is a length effect and nothing about cells is implicated.
"""
import os
import json
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))

# from diag_series_length.py: turbofan records decimated, keeping the span
DECIM = [(195, 0.863, 1.048), (98, 0.620, 1.155),
         (65, 0.252, 1.358), (49, 0.042, 1.480)]
CRB = json.load(open(os.path.join(HERE, "crb_full.json")))
TAILS = json.load(open(os.path.join(HERE, "battery_tails.json")))

n_cell = 89
ns = np.array([d[0] for d in DECIM][::-1], float)
sl = np.array([d[1] for d in DECIM][::-1], float)
md = np.array([d[2] for d in DECIM][::-1], float)

pred_slope = float(np.interp(n_cell, ns, sl))
pred_med = float(np.interp(n_cell, ns, md))
obs_slope = CRB["battery"]["slope"]
obs_med = CRB["battery"]["median"]

print("turbofan records decimated to the cells' own length (n = 89):")
print(f"{'quantity':<26}{'length predicts':>17}{'cells show':>13}{'gap':>8}")
print("-" * 64)
print(f"{'regression slope':<26}{pred_slope:>17.3f}{obs_slope:>13.3f}"
      f"{obs_slope - pred_slope:>8.3f}")
print(f"{'median ratio':<26}{pred_med:>17.3f}{obs_med:>13.3f}"
      f"{obs_med - pred_med:>8.3f}")
print()
print("the residue, against the tail control:")
rob = TAILS["robust scale (as before)"]
sd = TAILS["standard deviation"]
print(f"  robust scale (kurtosis {rob['kurt']:.0f}): median {rob['median']:.3f}")
print(f"  standard deviation           : median {sd['median']:.3f}")
print(f"  length alone predicts        : median {pred_med:.3f}")
print()
share = (obs_med - sd["median"]) / (obs_med - 1.0)
print(f"Of the cells' excess over 1, the tail treatment accounts for "
      f"{100*share:.0f}%;")
print(f"record length accounts for {100*(pred_med-1)/(obs_med-1):.0f}%. "
      f"They are not exclusive -- a short record")
print("makes a robust scale estimate noisier as well as more conservative.")

json.dump(dict(n_cell=n_cell, pred_slope=pred_slope, pred_median=pred_med,
               obs_slope=obs_slope, obs_median=obs_med,
               decim=[dict(n=d[0], slope=d[1], median=d[2]) for d in DECIM],
               tail_median_robust=rob["median"],
               tail_median_sd=sd["median"], tail_kurtosis=rob["kurt"]),
          open(os.path.join(HERE, "battery_diagnosis.json"), "w"), indent=2)
print("\nwrote battery_diagnosis.json")
