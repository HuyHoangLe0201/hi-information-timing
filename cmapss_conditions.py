"""
FD004 carries six operating regimes; a raw sensor trace is then dominated by
regime switching rather than by degradation, which is why 93% of unit-channels
were screened out as having no net trend. Standardising each sensor within its
regime -- the usual preprocessing for the multi-condition subsets -- is what
makes the channel a candidate health indicator at all, and is itself a
feature-extraction choice.
"""
import os
import numpy as np

# MSSP_DATA names the folder holding the raw datasets; by default it is the
# data/ folder three levels above this one.
DATA = os.environ.get("MSSP_DATA", os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "data"))
ROOT = os.path.join(DATA, "CMAPSSData")
HERE = os.path.dirname(os.path.abspath(__file__))


def regimes(settings, k=6):
    """Six discrete C-MAPSS regimes; rounding the settings separates them."""
    key = np.round(settings[:, :2], 0)
    key = np.column_stack([key, np.round(settings[:, 2], 0)])
    uniq, inv = np.unique(key, axis=0, return_inverse=True)
    return inv, len(uniq)


for sub in ["FD004"]:
    a = np.loadtxt(os.path.join(ROOT, f"train_{sub}.txt"))
    sett, sens = a[:, 2:5], a[:, 5:26]
    reg, nreg = regimes(sett)
    print(f"{sub}: {nreg} operating regimes detected, sizes "
          f"{np.bincount(reg).tolist()}")

    # z-score each sensor within its regime, using fleet-wide regime statistics
    Z = np.empty_like(sens)
    for r in range(nreg):
        m = reg == r
        mu, sd = sens[m].mean(0), sens[m].std(0)
        sd = np.where(sd < 1e-9, 1.0, sd)
        Z[m] = (sens[m] - mu) / sd

    units = np.unique(a[:, 0]).astype(int)
    store, lives = {}, []
    for u in units:
        m = a[:, 0] == u
        idx = np.argsort(a[m, 1])
        T = int(m.sum())
        if T < 60:
            continue
        store[f"u{u}__sensors"] = Z[m][idx].astype(np.float32)
        lives.append(T)
    np.savez_compressed(os.path.join(HERE, f"cmapss_{sub}_norm.npz"), **store)
    print(f"  wrote {len(lives)} units, mean life {np.mean(lives):.0f} cycles")
