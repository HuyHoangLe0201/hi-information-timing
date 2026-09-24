"""
Stage 1 -- Health-indicator extraction from the PRONOSTIA / FEMTO bearing set.

Each acc_*.csv snapshot holds 2560 samples of horizontal (col 4) and vertical
(col 5) acceleration recorded at 25.6 kHz; snapshots are spaced 10 s apart.
We reduce every snapshot to its RMS, giving one health-indicator sample per
10 s of bearing life.

Two quirks of the public release are handled here:
  * Bearing1_4 is semicolon-delimited while every other bearing is comma-
    delimited, so the separator is sniffed per file.
  * A handful of snapshots carry short/ragged rows; those rows are dropped
    rather than the whole snapshot.

Output: hi_raw.npz with one (n_snapshots,) array per bearing for each channel.
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
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "hi_raw.npz")

SETS = ["Learning_set", "Full_Test_Set"]
SNAPSHOT_PERIOD_S = 10.0      # spacing between consecutive acc_*.csv files
SAMPLES_PER_SNAPSHOT = 2560
ACQ_RATE_HZ = 25600.0


def _sniff_sep(path):
    with open(path, "r", errors="replace") as fh:
        first = fh.readline()
    return ";" if ";" in first else ","


def _snapshot_rms(path, sep):
    """Return (rms_horizontal, rms_vertical) for one snapshot."""
    try:
        df = pd.read_csv(path, sep=sep, header=None, usecols=[4, 5],
                         engine="c", na_values=["", " "], on_bad_lines="skip")
    except Exception:
        return np.nan, np.nan
    arr = df.to_numpy(dtype=np.float64)
    arr = arr[np.isfinite(arr).all(axis=1)]
    if arr.shape[0] < SAMPLES_PER_SNAPSHOT // 2:
        return np.nan, np.nan
    return float(np.sqrt(np.mean(arr[:, 0] ** 2))), float(np.sqrt(np.mean(arr[:, 1] ** 2)))


def process_bearing(args):
    setname, bearing = args
    folder = os.path.join(ROOT, setname, bearing)
    files = sorted(f for f in os.listdir(folder) if f.startswith("acc_") and f.endswith(".csv"))
    if not files:
        return bearing, None
    sep = _sniff_sep(os.path.join(folder, files[0]))
    h = np.empty(len(files))
    v = np.empty(len(files))
    for i, f in enumerate(files):
        h[i], v[i] = _snapshot_rms(os.path.join(folder, f), sep)
    # A snapshot that failed to parse is filled by linear interpolation so the
    # series stays uniformly sampled in time.
    for arr in (h, v):
        bad = ~np.isfinite(arr)
        if bad.any():
            idx = np.arange(len(arr))
            arr[bad] = np.interp(idx[bad], idx[~bad], arr[~bad])
    print(f"  {setname}/{bearing}: {len(files)} snapshots, "
          f"life = {len(files)*SNAPSHOT_PERIOD_S/60:.1f} min", flush=True)
    return bearing, (h, v, len(files))


def main():
    jobs = []
    for s in SETS:
        for b in sorted(os.listdir(os.path.join(ROOT, s))):
            if os.path.isdir(os.path.join(ROOT, s, b)):
                jobs.append((s, b))
    print(f"Extracting {len(jobs)} run-to-failure bearings ...", flush=True)

    out = {}
    n_workers = min(os.cpu_count() or 4, len(jobs))
    with ProcessPoolExecutor(max_workers=n_workers) as ex:
        for bearing, res in ex.map(process_bearing, jobs):
            if res is None:
                continue
            h, v, n = res
            out[f"{bearing}__h"] = h
            out[f"{bearing}__v"] = v

    meta = dict(snapshot_period_s=SNAPSHOT_PERIOD_S,
                acq_rate_hz=ACQ_RATE_HZ,
                samples_per_snapshot=SAMPLES_PER_SNAPSHOT)
    np.savez_compressed(OUT, **out, **{f"meta__{k}": v for k, v in meta.items()})
    print(f"\nwrote {OUT}  ({len(out)//2} bearings)")


if __name__ == "__main__":
    sys.exit(main())
