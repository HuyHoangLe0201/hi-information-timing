"""
Turbofan domain: C-MAPSS.

The third domain, and the one the window-length sweeps in the literature come
from. Each engine runs to failure, so tau = cycle / T_unit directly. Every one
of the 21 sensor channels is treated as a candidate health indicator, which is
the same experiment as the vibration bands: many indicators, one machine.
"""
import os
import numpy as np

# MSSP_DATA names the folder holding the raw datasets; by default it is the
# data/ folder three levels above this one.
DATA = os.environ.get("MSSP_DATA", os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "data"))
ROOT = os.path.join(DATA, "CMAPSSData")
HERE = os.path.dirname(os.path.abspath(__file__))
SUBSETS = ["FD001", "FD004"]        # single- and multi-condition
MIN_LIFE = 60                        # cycles; shorter units cannot support the smoother


def load(sub):
    a = np.loadtxt(os.path.join(ROOT, f"train_{sub}.txt"))
    return a


for sub in SUBSETS:
    a = load(sub)
    units = np.unique(a[:, 0]).astype(int)
    store = {}
    lives, kept = [], 0
    for u in units:
        m = a[a[:, 0] == u]
        m = m[np.argsort(m[:, 1])]
        T = len(m)
        if T < MIN_LIFE:
            continue
        kept += 1
        lives.append(T)
        # columns: 0 unit, 1 cycle, 2-4 settings, 5-25 sensors s1..s21
        store[f"u{u}__sensors"] = m[:, 5:26].astype(np.float32)
        store[f"u{u}__T"] = np.array([T])
    np.savez_compressed(os.path.join(HERE, f"cmapss_{sub}.npz"), **store)
    print(f"{sub}: {kept}/{len(units)} units kept (life >= {MIN_LIFE}), "
          f"life {min(lives)}-{max(lives)}, mean {np.mean(lives):.0f} cycles")

# which channels carry anything at all?
a = load("FD001")
print("\nFD001 channel variability (std over all rows, normalized by mean |value|):")
s = a[:, 5:26]
rel = s.std(0) / np.maximum(np.abs(s.mean(0)), 1e-9)
live = [i + 1 for i in range(21) if rel[i] > 1e-6]
dead = [i + 1 for i in range(21) if rel[i] <= 1e-6]
print(f"  varying : {live}")
print(f"  constant: {dead}")
