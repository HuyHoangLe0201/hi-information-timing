"""Build 32-band vibration spectra from the raw PRONOSTIA or XJTU-SY files.

Every acquisition of the horizontal accelerometer is Fourier-transformed, the
DC bin is dropped, and the power is summed into 32 bands of 400 Hz from 0 to
12.8 kHz. The result is one array of shape (n_acquisitions, 32) per bearing,
saved with the band centres in an .npz file that family_gap.py and causal.py
read.

    python prepare_spectra.py pronostia <path to PRONOSTIA>   -> spectra_pronostia.npz
    python prepare_spectra.py xjtu <path to XJTU-SY_Bearing_Datasets> -> spectra_xjtu.npz

PRONOSTIA (FEMTO-ST, IEEE PHM 2012 challenge): the folder that contains
Learning_set/ and Full_Test_Set/. XJTU-SY: the folder that contains the three
operating-condition folders (35Hz12kN, 37.5Hz11kN, 40Hz10kN).

Files that cannot be read as comma-separated values are skipped; in the
PRONOSTIA release this drops Bearing1_4, whose files use another separator,
which is why the paper reports sixteen PRONOSTIA bearings.
"""
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
FS = 25600.0
NBAND = 32
BW = (FS / 2) / NBAND           # 400 Hz


def _bands(a, nsamp=None):
    if nsamp is not None:
        if len(a) < nsamp:
            return None
        a = a[:nsamp]
    elif len(a) < 1024:
        return None
    a = a - a.mean()
    P = np.abs(np.fft.rfft(a)) ** 2
    P = P[1:len(a) // 2 + 1] if nsamp is not None else P[1:]
    per = len(P) // NBAND
    if per < 1:
        return None
    return P[:per * NBAND].reshape(NBAND, per).sum(axis=1)


def pronostia(root):
    """Acquisitions of 2560 samples; column 5 is the horizontal accelerometer."""
    out = {}
    for sub in ("Learning_set", "Full_Test_Set"):
        d = os.path.join(root, sub)
        if not os.path.isdir(d):
            continue
        for b in sorted(os.listdir(d)):
            p = os.path.join(d, b)
            if not os.path.isdir(p):
                continue
            rows = []
            for f in sorted(x for x in os.listdir(p) if x.lower().startswith("acc")):
                try:
                    a = pd.read_csv(os.path.join(p, f), header=None, usecols=[4],
                                    dtype=np.float64, engine="c").to_numpy().ravel()
                except Exception:
                    continue
                v = _bands(a, nsamp=2560)
                if v is not None:
                    rows.append(v)
            if rows:
                out[b] = np.asarray(rows, dtype=np.float32)
                print("  %-14s %5d acquisitions" % (b, len(rows)), flush=True)
    return out


def xjtu(root):
    """Acquisitions of 32768 samples; column 1 is the horizontal accelerometer."""
    def key(name):
        stem = os.path.splitext(name)[0]
        return int(stem) if stem.isdigit() else 10 ** 9

    out = {}
    for cond in sorted(os.listdir(root)):
        cp = os.path.join(root, cond)
        if not os.path.isdir(cp):
            continue
        for b in sorted(os.listdir(cp)):
            bp = os.path.join(cp, b)
            if not os.path.isdir(bp):
                continue
            rows = []
            for f in sorted((x for x in os.listdir(bp) if x.lower().endswith(".csv")),
                            key=key):
                try:
                    a = pd.read_csv(os.path.join(bp, f), usecols=[0], dtype=np.float64,
                                    engine="c").to_numpy().ravel()
                except Exception:
                    continue
                v = _bands(a)
                if v is not None:
                    rows.append(v)
            if rows:
                tag = "%s__%s" % (cond, b)
                out[tag] = np.asarray(rows, dtype=np.float32)
                print("  %-24s %5d acquisitions" % (tag, len(rows)), flush=True)
    return out


def main():
    if len(sys.argv) != 3 or sys.argv[1] not in ("pronostia", "xjtu"):
        sys.exit(__doc__)
    which, root = sys.argv[1], sys.argv[2]
    if not os.path.isdir(root):
        sys.exit("not a folder: %s" % root)
    out = pronostia(root) if which == "pronostia" else xjtu(root)
    if not out:
        sys.exit("nothing extracted from %s" % root)
    path = os.path.join(HERE, "spectra_%s.npz" % which)
    np.savez_compressed(path, centres=(np.arange(NBAND) + 0.5) * BW, **out)
    print("wrote %s: %d bearings" % (path, len(out)))


if __name__ == "__main__":
    main()
