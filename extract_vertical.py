"""
The vertical accelerometer, for both rigs.

The distribution-against-amount test used spectral measures only, because both
extractions took the horizontal channel alone. That left one strand untested:
the ratio of the two accelerometers was early on PRONOSTIA (0.501) and carries
no spectral information at all, so it says the effect is about distribution over
DIRECTION as well as over frequency -- but it has never been checked on the
second rig.

Both datasets record a vertical channel alongside the horizontal one, in the
same file. Reading dominates the cost and the files are read either way, so the
vertical bands are extracted on the same thirty-two-band grid, which also allows
spatial measures finer than a single ratio.

PRONOSTIA stores six unlabelled columns per acquisition, horizontal fifth and
vertical sixth. XJTU stores two named columns.
"""
import os
import sys
import time
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(HERE))),
                    "data")
NBAND = 32
FS = 25600.0


def bands(a):
    a = np.asarray(a, float)
    if len(a) < 1024:
        return None
    a = a - a.mean()
    P = np.abs(np.fft.rfft(a)) ** 2
    P = P[1:]
    per = len(P) // NBAND
    if per < 1:
        return None
    return P[:per * NBAND].reshape(NBAND, per).sum(axis=1)


def pronostia_jobs():
    root = os.path.join(DATA, "phm-ieee-2012-data-challenge-dataset-master")
    out = []
    for sub in ("Learning_set", "Full_Test_Set"):
        d = os.path.join(root, sub)
        if not os.path.isdir(d):
            continue
        for b in sorted(os.listdir(d)):
            p = os.path.join(d, b)
            if os.path.isdir(p):
                out.append((b, p, "pronostia"))
    return out


def xjtu_jobs():
    root = os.path.join(DATA, "XJTU-SY bearing dataset", "Data", "extracted",
                        "XJTU-SY_Bearing_Datasets")
    out = []
    if not os.path.isdir(root):
        return out
    for cond in sorted(os.listdir(root)):
        cp = os.path.join(root, cond)
        if not os.path.isdir(cp):
            continue
        for b in sorted(os.listdir(cp)):
            bp = os.path.join(cp, b)
            if os.path.isdir(bp):
                out.append((f"{cond}__{b}", bp, "xjtu"))
    return out


def read_vertical(path, kind):
    try:
        if kind == "pronostia":
            a = pd.read_csv(path, header=None, usecols=[5], dtype=np.float64,
                            engine="c").to_numpy().ravel()
        else:
            a = pd.read_csv(path, usecols=[1], dtype=np.float64,
                            engine="c").to_numpy().ravel()
    except Exception:
        return None
    return a


def numeric_key(name):
    stem = os.path.splitext(name)[0]
    return int(stem) if stem.isdigit() else 10 ** 9


for tag, jobs, outfile in (("PRONOSTIA", pronostia_jobs(), "vertical.npz"),
                           ("XJTU", xjtu_jobs(), "vertical_xjtu.npz")):
    if not jobs:
        print(f"{tag}: no data found")
        continue
    print(f"\n{tag}: {len(jobs)} bearings")
    out, t0 = {}, time.perf_counter()
    for i, (name, path, kind) in enumerate(jobs, 1):
        files = sorted((f for f in os.listdir(path)
                        if f.lower().endswith(".csv")
                        and (kind == "xjtu" or f.lower().startswith("acc"))),
                       key=numeric_key if kind == "xjtu" else str)
        rows = []
        for f in files:
            a = read_vertical(os.path.join(path, f), kind)
            if a is None:
                continue
            v = bands(a)
            if v is not None:
                rows.append(v)
        if rows:
            out[name] = np.asarray(rows, dtype=np.float32)
            print(f"  [{i:>2}/{len(jobs)}] {name:<24} {len(rows):>5}"
                  f"   {(time.perf_counter()-t0)/60:>5.1f} min", flush=True)
    if out:
        np.savez_compressed(os.path.join(HERE, outfile),
                            centres=(np.arange(NBAND) + 0.5) * (FS / 2 / NBAND),
                            **out)
        print(f"  wrote {outfile}: {len(out)} bearings, "
              f"{sum(len(v) for v in out.values())} acquisitions")
