"""
Does the battery answer depend on where end of life is placed?

Capacity fade is roughly linear before the knee and steepens after it, so an
EOL threshold that truncates before the knee will report a near-uniform
information profile whether or not the cell has one. The threshold is swept
here from 90% to 65% of initial capacity, plus the full record, so the claim
is stated against the choice rather than on top of it.
"""
import os
import numpy as np
import scipy.io as sio
from pipeline import prepare, tau_min, beta_eff

# MSSP_DATA names the folder holding the raw datasets; by default it is the
# data/ folder three levels above this one.
DATA = os.environ.get("MSSP_DATA", os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "data"))
ROOT = os.path.join(DATA, "NASABattery")
CELLS = ["B0005", "B0006", "B0007", "B0018"]
QSTAR = 0.35


def cell_capacity(cell):
    m = sio.loadmat(os.path.join(ROOT, cell + ".mat"), simplify_cells=True)
    cyc = m[cell]["cycle"]
    idx, val = [], []
    for i, c in enumerate(cyc):
        if c["type"] == "discharge" and "Capacity" in c["data"]:
            idx.append(i)
            val.append(float(np.atleast_1d(c["data"]["Capacity"]).flat[0]))
    return np.array(idx), np.array(val)


print(f"{'EOL':<12}{'cells':>6}{'med n':>7}{'beta_eff':>10}{'tau_min':>9}"
      f"{'info<0.5':>10}   note")
print("-" * 70)
for label, frac in [("90% of C0", 0.90), ("85% of C0", 0.85), ("80% of C0", 0.80),
                    ("75% of C0", 0.75), ("70% of C0", 0.70), ("full record", None)]:
    bs, ts, i5, ns = [], [], [], []
    for cell in CELLS:
        idx, val = cell_capacity(cell)
        C0 = float(np.median(val[:3]))
        if frac is None:
            k = len(val) - 1
        else:
            below = np.where(val <= frac * C0)[0]
            if not len(below):
                continue
            k = int(below[0])
        if k < 25:
            continue
        u, why = prepare(-val[:k + 1], cell)
        if u is None:
            continue
        bs.append(beta_eff(u)); ts.append(tau_min(u, QSTAR))
        i5.append(float(u["F"][int(0.5 * u["n"]) - 1])); ns.append(u["n"])
    if not bs:
        print(f"{label:<12}{'--':>6}   no cell reaches this threshold")
        continue
    note = "" if len(bs) == len(CELLS) else f"only {len(bs)}/{len(CELLS)} cells reach it"
    print(f"{label:<12}{len(bs):>6}{int(np.median(ns)):>7}{np.median(bs):>10.2f}"
          f"{np.median(ts):>9.3f}{np.median(i5):>10.3f}   {note}")

print("-" * 70)
print("beta_eff = 1 is a uniform information profile; the nominal battery class")
print("of [1] (stretched-exponential, b = 0.7) puts almost all information early.")
