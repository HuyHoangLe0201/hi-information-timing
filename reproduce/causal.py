"""Reproduce the causal comparison of the paper (Table 8).

For every indicator and bearing, the information age is computed twice: with
the two-sided estimator used throughout the paper and with the strictly causal
one. The family gap is formed under each.

This is the slowest script: the causal noise floor is re-estimated at 40
points along every record. It runs one process per (rig, indicator).

    python causal.py            (about 30 minutes on 10 cores)
"""
import os
from concurrent.futures import ProcessPoolExecutor

import numpy as np

import hitiming as ht

HERE = os.path.dirname(os.path.abspath(__file__))
FOLDER = os.environ.get("HITIMING_SPECTRA", HERE)
RIGS = {"PRONOSTIA": "spectra_pronostia.npz", "XJTU-SY": "spectra_xjtu.npz"}
Q = 0.35
SEED = 8675309          # a fresh generator with this seed for every record
PAPER = {"PRONOSTIA": (0.641, 0.563), "XJTU-SY": (0.357, 0.029)}


def one(job):
    rig, fn, name = job
    z = np.load(os.path.join(FOLDER, fn))
    f = z["centres"] / 1000.0
    func = ht.spectral.INDICATORS[name][1]
    centred, causal = [], []
    for u in sorted(k for k in z.files if k != "centres"):
        x = func(z[u].astype(float), f)
        x = x[np.isfinite(x)]
        if len(x) < 60:
            continue
        centred.append(ht.information_age(x, Q, rng=np.random.default_rng(SEED)))
        causal.append(ht.causal_information_age(x, Q, rng=np.random.default_rng(SEED)))
    med = lambda v: float(np.nanmedian(v)) if np.isfinite(v).any() else np.nan
    return rig, name, med(centred), med(causal)


def main():
    jobs = [(rig, fn, name) for rig, fn in RIGS.items()
            if os.path.exists(os.path.join(FOLDER, fn))
            for name in ht.spectral.INDICATORS]
    if not jobs:
        print("no spectra found (run the prepare_*.py scripts first)")
        return
    with ProcessPoolExecutor() as pool:
        rows = list(pool.map(one, jobs))
    fam = ht.spectral.families()
    for rig in RIGS:
        r = {name: (c, u) for rg, name, c, u in rows if rg == rig}
        if not r:
            continue
        print("\n%s" % rig)
        print("%-22s %9s %9s" % ("indicator", "centred", "causal"))
        for name, (c, u) in r.items():
            print("%-22s %9.3f %9.3f" % (name, c, u))
        for k, label in ((0, "centred"), (1, "causal")):
            gap = (np.median([r[n][k] for n in fam["amount"]])
                   - np.median([r[n][k] for n in fam["distribution"]]))
            print("family gap, %-8s %.3f (paper %.3f)" % (label, gap, PAPER[rig][k]))


if __name__ == "__main__":
    main()
