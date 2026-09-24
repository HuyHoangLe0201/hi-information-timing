"""
Independent bearing set: XJTU-SY.

The 40x spread across vibration indicators rests on PRONOSTIA alone, and that
rig stops each test at a 20 g threshold -- which could manufacture the terminal
RMS spike that drives the extreme exponent. XJTU-SY is a different rig, three
load/speed conditions, its own stopping rule. The identical feature set is
extracted so the two are directly comparable.

Each file holds 32768 samples at 25.6 kHz (1.28 s), one file per minute.
"""
import os, sys
import numpy as np
import pandas as pd
from concurrent.futures import ProcessPoolExecutor

# MSSP_DATA names the folder holding the raw datasets; by default it is the
# data/ folder three levels above this one.
DATA = os.environ.get("MSSP_DATA", os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "data"))
ROOT = os.path.join(DATA, "XJTU-SY bearing dataset", "Data",
                    "extracted", "XJTU-SY_Bearing_Datasets")
HERE = os.path.dirname(os.path.abspath(__file__))
FS = 25600.0
BANDS = [(0, 1), (1, 2), (2, 4), (4, 6), (6, 8), (8, 10), (10, 12.8)]
FEATS = ["rms", "kurt", "peak"] + [f"b{lo}_{hi}" for lo, hi in BANDS]


def snapshot(path):
    try:
        x = pd.read_csv(path, usecols=[0]).to_numpy(dtype=np.float64).ravel()
    except Exception:
        return None
    x = x[np.isfinite(x)]
    if x.size < 4096:
        return None
    x = x - x.mean()
    out = [float(np.sqrt(np.mean(x ** 2))),
           float(np.mean(x ** 4) / max(np.mean(x ** 2) ** 2, 1e-18)),
           float(np.max(np.abs(x)))]
    P = np.abs(np.fft.rfft(x)) ** 2
    f = np.fft.rfftfreq(len(x), 1.0 / FS) / 1000.0
    tot = max(P.sum(), 1e-18)
    for lo, hi in BANDS:
        m = (f >= lo) & (f < hi)
        out.append(float(P[m].sum() / tot) if m.any() else np.nan)
    return out


def do_bearing(args):
    cond, name = args
    d = os.path.join(ROOT, cond, name)
    files = sorted((f for f in os.listdir(d) if f.endswith(".csv")),
                   key=lambda s: int(os.path.splitext(s)[0]))
    M = np.full((len(files), len(FEATS)), np.nan)
    for i, f in enumerate(files):
        v = snapshot(os.path.join(d, f))
        if v is not None:
            M[i] = v
    for j in range(M.shape[1]):
        col = M[:, j]
        bad = ~np.isfinite(col)
        if bad.any() and (~bad).sum() > 1:
            idx = np.arange(len(col))
            col[bad] = np.interp(idx[bad], idx[~bad], col[~bad])
    print(f"  {cond}/{name}: {len(files)} snapshots", flush=True)
    return f"{cond}__{name}", M


def main():
    jobs = [(c, b) for c in sorted(os.listdir(ROOT))
            if os.path.isdir(os.path.join(ROOT, c))
            for b in sorted(os.listdir(os.path.join(ROOT, c)))
            if os.path.isdir(os.path.join(ROOT, c, b))]
    print(f"extracting {len(jobs)} bearings ...", flush=True)
    out = {}
    with ProcessPoolExecutor(max_workers=min(os.cpu_count() or 4, len(jobs))) as ex:
        for k, M in ex.map(do_bearing, jobs):
            out[k] = M
    np.savez_compressed(os.path.join(HERE, "xjtu_feat.npz"),
                        featnames=np.array(FEATS), **out)
    print(f"\nwrote xjtu_feat.npz ({len(out)} bearings)")


if __name__ == "__main__":
    sys.exit(main())
