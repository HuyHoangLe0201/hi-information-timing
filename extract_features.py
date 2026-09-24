"""
Stage 1b -- richer per-snapshot feature set.

Broadband RMS turns out to be a step-like indicator on PRONOSTIA (flat for most
of life, then a terminal spike), which no smooth degradation class describes.
Bearing-fault energy normally shows up first in the high-frequency resonance
bands, so we extract band energies alongside the usual scalar features and let
the data say which indicator is best described by a smooth degradation law.

Snapshot = 2560 samples at 25.6 kHz (Nyquist 12.8 kHz).
"""
import os
import sys
import numpy as np
import pandas as pd
from concurrent.futures import ProcessPoolExecutor

# MSSP_DATA names the folder holding the raw datasets; by default it is the
# data/ folder three levels above this one.
DATA = os.environ.get("MSSP_DATA", os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "data"))
ROOT = os.path.join(DATA, "phm-ieee-2012-data-challenge-dataset-master")
HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "feat_raw.npz")

SETS = ["Learning_set", "Full_Test_Set"]
FS = 25600.0
NS = 2560
BANDS_KHZ = [(0, 1), (1, 2), (2, 4), (4, 6), (6, 8), (8, 10), (10, 12.8)]
FEATS = ["rms", "kurt", "peak"] + [f"b{lo}_{hi}" for lo, hi in BANDS_KHZ]


def _sniff_sep(path):
    with open(path, "r", errors="replace") as fh:
        return ";" if ";" in fh.readline() else ","


def _snapshot(path, sep):
    try:
        df = pd.read_csv(path, sep=sep, header=None, usecols=[4, 5], engine="c",
                         na_values=["", " "], on_bad_lines="skip")
    except Exception:
        return None
    a = df.to_numpy(dtype=np.float64)
    a = a[np.isfinite(a).all(axis=1)]
    if a.shape[0] < NS // 2:
        return None
    x = a[:, 0]                      # horizontal channel
    x = x - x.mean()
    n = len(x)
    out = [float(np.sqrt(np.mean(x ** 2))),
           float(np.mean(x ** 4) / max(np.mean(x ** 2) ** 2, 1e-18)),
           float(np.max(np.abs(x)))]
    # one-sided power spectrum, energy per band
    P = np.abs(np.fft.rfft(x)) ** 2
    f = np.fft.rfftfreq(n, d=1.0 / FS) / 1000.0     # kHz
    for lo, hi in BANDS_KHZ:
        m = (f >= lo) & (f < hi)
        out.append(float(P[m].sum() / max(P.sum(), 1e-18)) if m.any() else np.nan)
    return out


def process_bearing(args):
    setname, bearing = args
    folder = os.path.join(ROOT, setname, bearing)
    files = sorted(f for f in os.listdir(folder)
                   if f.startswith("acc_") and f.endswith(".csv"))
    sep = _sniff_sep(os.path.join(folder, files[0]))
    M = np.full((len(files), len(FEATS)), np.nan)
    for i, fn in enumerate(files):
        v = _snapshot(os.path.join(folder, fn), sep)
        if v is not None:
            M[i] = v
    for j in range(M.shape[1]):
        col = M[:, j]
        bad = ~np.isfinite(col)
        if bad.any() and (~bad).sum() > 1:
            idx = np.arange(len(col))
            col[bad] = np.interp(idx[bad], idx[~bad], col[~bad])
    print(f"  {bearing}: {len(files)} snapshots", flush=True)
    return bearing, M


def main():
    jobs = [(s, b) for s in SETS
            for b in sorted(os.listdir(os.path.join(ROOT, s)))
            if os.path.isdir(os.path.join(ROOT, s, b))]
    print(f"Extracting {len(FEATS)} features x {len(jobs)} bearings ...", flush=True)
    out = {}
    with ProcessPoolExecutor(max_workers=min(os.cpu_count() or 4, len(jobs))) as ex:
        for bearing, M in ex.map(process_bearing, jobs):
            out[bearing] = M
    np.savez_compressed(OUT, featnames=np.array(FEATS), **out)
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    sys.exit(main())
