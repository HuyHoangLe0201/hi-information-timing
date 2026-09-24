"""
The same thirty-two bands, from the XJTU-SY raw acquisitions.

The distribution-against-amount result was built on indicators constructed here
from PRONOSTIA spectra, which removed any dependence on feature definitions
inherited from elsewhere. It has support from XJTU only through the stored
feature set, where the level-shape gap came out at 0.199 against PRONOSTIA's
0.221 -- the same direction, but measured on columns whose definitions are not
all to hand.

XJTU records the same two accelerometers at the same 25.6 kHz, so the identical
band structure applies and the two rigs can be compared on indicators built the
same way. Acquisitions here are 32768 samples against PRONOSTIA's 2560, which
only sharpens the spectrum; the bands are the same 400 Hz.
"""
import os
import sys
import time
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(HERE))),
                    "data", "XJTU-SY bearing dataset", "Data", "extracted",
                    "XJTU-SY_Bearing_Datasets")
NBAND = 32
FS = 25600.0
BW = (FS / 2) / NBAND


def band_powers(path):
    try:
        a = pd.read_csv(path, usecols=[0], dtype=np.float64,
                        engine="c").to_numpy().ravel()
    except Exception:
        return None
    n = len(a)
    if n < 1024:
        return None
    a = a - a.mean()
    P = np.abs(np.fft.rfft(a)) ** 2
    P = P[1:]                                  # drop DC
    per = len(P) // NBAND
    if per < 1:
        return None
    return P[:per * NBAND].reshape(NBAND, per).sum(axis=1)


def numeric_key(name):
    stem = os.path.splitext(name)[0]
    return int(stem) if stem.isdigit() else 10 ** 9


if not os.path.isdir(ROOT):
    sys.exit(f"not found: {ROOT}")

jobs = []
for cond in sorted(os.listdir(ROOT)):
    cp = os.path.join(ROOT, cond)
    if not os.path.isdir(cp):
        continue
    for b in sorted(os.listdir(cp)):
        bp = os.path.join(cp, b)
        if os.path.isdir(bp):
            jobs.append((f"{cond}__{b}", bp))

print(f"{len(jobs)} bearings under {ROOT}")
print(f"{NBAND} bands of {BW:.0f} Hz\n")
out, t0 = {}, time.perf_counter()
for i, (tag, path) in enumerate(jobs, 1):
    files = sorted((f for f in os.listdir(path) if f.lower().endswith(".csv")),
                   key=numeric_key)
    rows = []
    for f in files:
        v = band_powers(os.path.join(path, f))
        if v is not None:
            rows.append(v)
    if rows:
        out[tag] = np.asarray(rows, dtype=np.float32)
        print(f"  [{i:>2}/{len(jobs)}] {tag:<24} {len(rows):>5} acquisitions"
              f"   {(time.perf_counter()-t0)/60:>5.1f} min", flush=True)

if out:
    np.savez_compressed(os.path.join(HERE, "fineband_xjtu.npz"),
                        centres=(np.arange(NBAND) + 0.5) * BW, **out)
    print(f"\nwrote fineband_xjtu.npz: {len(out)} bearings, "
          f"{sum(len(v) for v in out.values())} acquisitions")
else:
    print("nothing extracted", file=sys.stderr)
