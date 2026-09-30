"""Reproduce the applied result of the paper (Table 5 and its intervals).

Ten indicators are computed from the 32-band vibration spectra of the
PRONOSTIA and XJTU-SY bearings; each indicator's information age tau_0.35 is
its median over the bearings; the family gap is the median age of the amount
family minus that of the distributional family.

The bootstrap intervals are computed, as in the paper, from per-bearing ages
obtained in a second pass (bearings in the outer loop, one generator seeded
20260828), resampling the bearings that have at least one usable age.

Input: the band-power files written by prepare_spectra.py (spectra_pronostia.npz,
spectra_xjtu.npz) in this folder, or in the folder given by HITIMING_SPECTRA.

    python family_gap.py        (about ten minutes)
"""
import os

import numpy as np

import hitiming as ht

HERE = os.path.dirname(os.path.abspath(__file__))
FOLDER = os.environ.get("HITIMING_SPECTRA", HERE)
RIGS = {"PRONOSTIA": "spectra_pronostia.npz", "XJTU-SY": "spectra_xjtu.npz"}
Q = 0.35
THRESHOLD = 0.90        # surviving fraction below which an age is flagged

# the values printed in the paper, for comparison
PAPER = {"PRONOSTIA": {"spectral entropy": 0.347, "spectral centroid": 0.531,
                       "spectral spread": 0.151, "spectral Gini": 0.288,
                       "top-8-band share": 0.360, "high/low band ratio": 0.312,
                       "total power": 0.971, "peak band power": 0.972,
                       "high-band power": 0.963, "log total power": 0.969,
                       "gap": 0.640, "units": (0.215, 0.830),
                       "both": (0.182, 0.841)},
         "XJTU-SY": {"spectral entropy": 0.661, "spectral centroid": 0.509,
                     "spectral spread": 0.463, "spectral Gini": 0.561,
                     "top-8-band share": 0.491, "high/low band ratio": 0.602,
                     "total power": 0.907, "peak band power": 0.900,
                     "high-band power": 0.614, "log total power": 0.886,
                     "gap": 0.358, "units": (0.057, 0.439),
                     "both": (-0.105, 0.501)}}


def load_rig(path):
    z = np.load(path)
    f = z["centres"] / 1000.0
    units = sorted(k for k in z.files if k != "centres")
    per_unit = [ht.spectral.spectral_indicators(z[u].astype(float), f) for u in units]
    return per_unit, units


def main():
    fam = ht.spectral.families()
    rng_table = np.random.default_rng(8675309)      # Table 5: indicators, then bearings
    rng_units = np.random.default_rng(20260828)     # intervals: bearings, then indicators
    rng_boot = np.random.default_rng(20260924)      # bootstrap: one generator, both rigs
    for rig, fn in RIGS.items():
        path = os.path.join(FOLDER, fn)
        if not os.path.exists(path):
            print("%s: %s not found (run prepare_spectra.py first)" % (rig, fn))
            continue
        per_unit, units = load_rig(path)

        records = {name: [ind[name] for ind in per_unit] for name in ht.spectral.INDICATORS}
        table = ht.indicator_ages(records, q=Q, rng=rng_table)
        print("\n%s, %d bearings" % (rig, len(units)))
        print("%-22s %-13s %8s %8s %10s" % ("indicator", "family", "age", "paper",
                                            "surviving"))
        for name, r in table.items():
            print("%-22s %-13s %8.3f %8.3f %10.3f" % (
                name, ht.spectral.INDICATORS[name][0], r["median_age"],
                PAPER[rig][name], r["median_surviving_fraction"]))
        reliable = [n for n, r in table.items()
                    if r["median_surviving_fraction"] >= THRESHOLD]
        med = lambda names: float(np.median([table[n]["median_age"] for n in names
                                             if n in reliable]))
        gap = med(fam["amount"]) - med(fam["distribution"])
        print("family gap %.3f (paper %.3f)" % (gap, PAPER[rig]["gap"]))

        ages = {name: [] for name in ht.spectral.INDICATORS}
        for ind in per_unit:
            for name in ht.spectral.INDICATORS:
                ages[name].append(ht.information_age(ind[name], Q, rng=rng_units))
        boot = ht.bootstrap_family_gap(ages, fam["distribution"], fam["amount"],
                                       n_boot=4000, seed=rng_boot)
        print("  95%% interval, bearings resampled:             [%+.3f, %+.3f]"
              "   paper [%+.3f, %+.3f]" % (boot["units"] + PAPER[rig]["units"]))
        print("  95%% interval, bearings and indicators resampled: [%+.3f, %+.3f]"
              "   paper [%+.3f, %+.3f]; %.1f%% of replicates positive"
              % (boot["units_and_indicators"] + PAPER[rig]["both"]
                 + (100 * boot["share_positive"],)))


if __name__ == "__main__":
    main()
