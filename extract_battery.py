"""
Battery domain: NASA PCoE cells B0005/6/7/18.

Three health indicators are built from the SAME cell record, mirroring the
several vibration indicators built from one accelerometer:
    cap  -- capacity fade, the standard prognostic indicator
    Re   -- electrolyte resistance, from the interleaved impedance sweeps
    Rct  -- charge-transfer resistance, likewise
End of life is the first cycle at which capacity falls to 80% of that cell's
own initial value, which every cell here reaches. Impedance sweeps interleave
with discharge cycles, so all three indicators are placed on the common
cycle-index axis and read on the discharge grid.
"""
import os
import json
import numpy as np
import scipy.io as sio

# MSSP_DATA names the folder holding the raw datasets; by default it is the
# data/ folder three levels above this one.
DATA = os.environ.get("MSSP_DATA", os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "data"))
ROOT = os.path.join(DATA, "NASABattery")
CELLS = ["B0005", "B0006", "B0007", "B0018"]
EOL_FRAC = 0.80
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "battery_raw.npz")


def load(cell):
    m = sio.loadmat(os.path.join(ROOT, cell + ".mat"), simplify_cells=True)
    return m[cell]["cycle"]


store, meta = {}, {}
print(f"{'cell':<8}{'EOL cyc':>9}{'n_cap':>7}{'n_Re':>7}{'C0':>8}{'C_eol':>8}")
print("-" * 46)
for cell in CELLS:
    cyc = load(cell)
    cap_idx, cap_val, re_idx, re_val, rct_idx, rct_val = [], [], [], [], [], []
    for i, c in enumerate(cyc):
        d = c["data"]
        if c["type"] == "discharge" and "Capacity" in d:
            cap_idx.append(i); cap_val.append(float(np.atleast_1d(d["Capacity"]).flat[0]))
        elif c["type"] == "impedance":
            re_idx.append(i); re_val.append(float(np.real(np.atleast_1d(d["Re"]).flat[0])))
            rct_idx.append(i); rct_val.append(float(np.real(np.atleast_1d(d["Rct"]).flat[0])))
    cap_idx, cap_val = np.array(cap_idx), np.array(cap_val)

    # end of life: first discharge at or below 80% of this cell's initial capacity
    C0 = float(np.median(cap_val[:3]))
    thr = EOL_FRAC * C0
    below = np.where(cap_val <= thr)[0]
    if not len(below):
        print(f"{cell:<8}  never reaches {EOL_FRAC:.0%} of C0 -- skipped")
        continue
    k_eol = int(below[0])
    T = cap_idx[k_eol]                       # EOL position on the cycle-index axis

    keep = cap_idx <= T
    grid = cap_idx[keep].astype(float)       # read every indicator on this grid
    tau = grid / T

    # capacity falls, so invert it into a degradation that rises
    cap = -cap_val[keep]
    # impedance rises already; interpolate onto the discharge grid
    reA = np.interp(grid, np.array(re_idx, float), np.array(re_val))
    rctA = np.interp(grid, np.array(rct_idx, float), np.array(rct_val))

    for tag, series in [("cap", cap), ("Re", reA), ("Rct", rctA)]:
        store[f"{cell}__{tag}"] = series
    store[f"{cell}__tau"] = tau
    meta[cell] = dict(T_index=int(T), n=int(keep.sum()), C0=C0,
                      C_eol=float(cap_val[k_eol]), eol_cycle=int(k_eol))
    print(f"{cell:<8}{k_eol:>9}{keep.sum():>7}{len(re_idx):>7}"
          f"{C0:>8.3f}{cap_val[k_eol]:>8.3f}")

np.savez_compressed(OUT, **store)
json.dump(meta, open(OUT.replace(".npz", "_meta.json"), "w"), indent=2)
print(f"\nwrote {OUT}  ({len(meta)} cells x 3 indicators)")
