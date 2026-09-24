"""
Fine-grained frequency bands from the raw PRONOSTIA vibration.

The resonance question could not be answered from the seven pre-computed bands:
the two rigs disagreed on which was best, and band energy growth correlated with
the earliest usable age in opposite directions on the two. Seven bands spanning
12.8 kHz is 1.8 kHz of resolution, which is coarse enough that a resonance and
its neighbourhood fall in the same bin.

The raw records are on disk -- 24878 acquisitions across all seventeen bearings,
2560 samples each at 25.6 kHz -- so the bands can be made as narrow as the FFT
allows. Thirty-two bands of 400 Hz give four times the resolution and let the
question be asked properly: is there a narrow frequency range whose information
arrives early, and is it where a structural resonance would be?

Only the horizontal accelerometer is used, which is the channel the published
feature sets are built from.
"""
import os
import sys
import time
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(HERE))),
                    "data", "phm-ieee-2012-data-challenge-dataset-master")
FS = 25600.0
NSAMP = 2560
NBAND = 32
BW = (FS / 2) / NBAND          # 400 Hz per band
BINS = NSAMP // 2              # positive frequencies


def bearing_dirs():
    out = {}
    for sub in ("Learning_set", "Full_Test_Set"):
        d = os.path.join(ROOT, sub)
        if not os.path.isdir(d):
            continue
        for b in sorted(os.listdir(d)):
            p = os.path.join(d, b)
            if os.path.isdir(p):
                out[b] = p
    return out


def band_powers(path):
    """Power in each band for one acquisition, horizontal channel."""
    try:
        a = pd.read_csv(path, header=None, usecols=[4], dtype=np.float64,
                        engine="c").to_numpy().ravel()
    except Exception:
        return None
    if len(a) < NSAMP:
        return None
    a = a[:NSAMP] - a[:NSAMP].mean()
    P = np.abs(np.fft.rfft(a)) ** 2
    P = P[1:BINS + 1]                       # drop DC
    per = len(P) // NBAND
    return P[:per * NBAND].reshape(NBAND, per).sum(axis=1)


dirs = bearing_dirs()
print(f"{len(dirs)} bearings under {ROOT}")
print(f"{NBAND} bands of {BW:.0f} Hz, 0 to {FS/2/1000:.1f} kHz\n")
out, t0 = {}, time.perf_counter()
for i, (b, p) in enumerate(sorted(dirs.items()), 1):
    files = sorted(f for f in os.listdir(p) if f.lower().startswith("acc"))
    rows = []
    for f in files:
        v = band_powers(os.path.join(p, f))
        if v is not None:
            rows.append(v)
    if rows:
        out[b] = np.asarray(rows, dtype=np.float32)
        el = time.perf_counter() - t0
        print(f"  [{i:>2}/{len(dirs)}] {b:<14} {len(rows):>5} acquisitions"
              f"   {el/60:>5.1f} min elapsed", flush=True)

if out:
    centres = (np.arange(NBAND) + 0.5) * BW
    np.savez_compressed(os.path.join(HERE, "fineband.npz"),
                        centres=centres, **out)
    print(f"\nwrote fineband.npz: {len(out)} bearings, "
          f"{sum(len(v) for v in out.values())} acquisitions")
else:
    print("nothing extracted", file=sys.stderr)
