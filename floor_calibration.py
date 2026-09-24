"""
Is the noise floor calibrated?

Every curve in this paper has a noise floor subtracted, and the signal fraction
that survives that subtraction is used as the paper's trust diagnostic.  The
estimator was chosen by comparison with an alternative -- a flat lookup by record
length, which was shown not to transfer -- and never against ground truth.  A
comparison between two estimators says which is better, not whether either is
right.

Two calibrations are available because the truth can be constructed.

On a record with NO degradation, all of the density is noise, so the
floor-subtracted density should vanish and the signal fraction should read zero.
Whatever it reads instead is the diagnostic's false-positive level, and it is the
number that decides how much a signal fraction of, say, 0.9 is worth.

On a record with a known trend, the true signal fraction can be computed from the
noiseless density, and the estimate can be compared with it directly across
signal-to-noise ratios.

Both are run across the noise profiles the paper cares about -- constant, rising,
falling, and tracking the signal -- because the floor's whole justification was
that the weighted density has a flat floor whatever the noise does, and that
claim has been tested only for the TILT of the floor, never for its LEVEL.
"""
import os
import json
import numpy as np

from nonparam import weighted_density, estimate_floor

HERE = os.path.dirname(os.path.abspath(__file__))
SEED = 20260828
REPS = 40
NS = (200, 500, 1200, 2500)
PROFILES = ("constant", "rising", "falling", "tracking")
SNRS = (3.0, 10.0, 30.0, 100.0)


def noise_profile(kind, n, tau):
    if kind == "constant":
        return np.ones(n)
    if kind == "rising":
        return 0.5 + 2.0 * tau
    if kind == "falling":
        return 2.5 - 2.0 * tau
    return 0.5 + 2.0 * tau ** 3          # tracking a steep signal


def signal_fraction(x, rng):
    dens, res = weighted_density(np.asarray(x, float))
    if dens is None:
        return np.nan
    fl = estimate_floor(dens, res, "surrogate", rng)
    d = np.clip(dens - fl, 0.0, None)
    tot = float(dens.sum())
    return float(d.sum() / tot) if tot > 0 else np.nan


rng = np.random.default_rng(SEED)

# --- 1. no degradation: the fraction should read zero -----------------------
print("records with NO degradation; the signal fraction should be zero\n")
print(f"{'n':>7}" + "".join(f"{p:>13}" for p in PROFILES))
print("-" * (7 + 13 * len(PROFILES)))
null_rows = []
for n in NS:
    tau = (np.arange(n) + 1.0) / n
    line = []
    for kind in PROFILES:
        s = noise_profile(kind, n, tau)
        v = []
        for _ in range(REPS):
            x = s * rng.normal(size=n)
            f = signal_fraction(x, rng)
            if np.isfinite(f):
                v.append(f)
        m = float(np.median(v)) if v else np.nan
        p90 = float(np.percentile(v, 90)) if v else np.nan
        null_rows.append(dict(n=n, profile=kind, median=m, p90=p90,
                              reps=len(v)))
        line.append(m)
    print(f"{n:>7}" + "".join(f"{m:>13.3f}" for m in line))
print("-" * (7 + 13 * len(PROFILES)))
allm = np.array([r["median"] for r in null_rows], float)
allp = np.array([r["p90"] for r in null_rows], float)
print(f"pooled median {np.median(allm):.3f}, worst cell {np.nanmax(allm):.3f}, "
      f"worst 90th percentile {np.nanmax(allp):.3f}")
print()
print("This is the diagnostic's false-positive level: how much apparent signal")
print("the estimator credits to a record that has none.  It is what a reported")
print("signal fraction has to be read against.\n")

# --- 2. known trend: does the estimate track the truth? ---------------------
# The truth cannot be taken as the density of the noiseless trend: the weighted
# density divides by the local scale of the residual, which is essentially zero
# for a perfectly smooth curve, so that quantity diverges and the comparison is
# meaningless.  The well-defined truth is the share of the observed density that
# is NOT attributable to noise, obtained by running the same pipeline on a
# noise-only record with the same scale profile:
#
#     true fraction = 1 - E[ density of noise alone ] / density observed.
print("records with a known trend; estimated against true signal fraction\n")
print("the truth is 1 - (density of noise alone) / (density observed), both")
print("through the same pipeline\n")
print(f"{'SNR':>7}{'profile':>12}{'true':>9}{'estimated':>12}{'ratio':>9}")
print("-" * 49)
cal_rows = []
n = 1200
tau = (np.arange(n) + 1.0) / n
trend = tau ** 3                          # a steep, realistic shape
CAL_REPS = 15
for snr in SNRS:
    for kind in PROFILES:
        s = noise_profile(kind, n, tau)
        s = s / np.mean(s) * (np.std(trend) / snr)
        est, tru = [], []
        for _ in range(CAL_REPS):
            e = s * rng.normal(size=n)
            x = trend + e
            d_noisy, _ = weighted_density(x)
            d_noise, _ = weighted_density(e)
            if d_noisy is None or d_noise is None:
                continue
            tot = float(d_noisy.sum())
            if tot <= 0:
                continue
            t = 1.0 - float(d_noise.sum()) / tot
            f = signal_fraction(x, rng)
            if np.isfinite(f) and np.isfinite(t):
                est.append(f)
                tru.append(max(min(t, 1.0), 0.0))
        if not est:
            continue
        me, mt = float(np.median(est)), float(np.median(tru))
        cal_rows.append(dict(snr=snr, profile=kind, true=mt, est=me,
                             excess=me - mt))
        print(f"{snr:>7.0f}{kind:>12}{mt:>9.3f}{me:>12.3f}{me - mt:>+9.3f}")
    print()
print("-" * 49)
rr = np.array([r["excess"] for r in cal_rows], float)
print(f"estimated minus true: median {np.median(rr):+.3f}, "
      f"range {rr.min():+.3f} to {rr.max():+.3f} over {len(rr)} cells")
print()
print("The two halves say different things and both matter.")
print()
print(f"On a record with no signal the diagnostic reads {np.median(allm):.2f}, not zero.")
print("That is a property of the estimator rather than a bug: the floor is a")
print("MEDIAN level subtracted from a density that is a squared quantity and")
print("therefore heavy-tailed, so about half the samples survive the")
print("subtraction and they carry most of the mass.  The consequence is that a")
print("signal fraction cannot be read as the share of the density that is real.")
print()
if abs(np.median(rr)) < 0.10:
    print(f"Where there IS signal the estimate tracks the truth to "
          f"{np.median(rr):+.3f}, so the")
    print("quantity is informative above its floor even though it is not a")
    print("fraction.  The paper uses it as a threshold at 0.90, which sits well")
    print(f"above the {np.median(allm):.2f} null, so the diagnostic as USED is unaffected;")
    print("what has to change is the description of what it measures.")
else:
    print(f"Where there is signal the estimate departs from the truth by "
          f"{np.median(rr):+.3f},")
    print("so the quantity is neither a fraction nor a calibrated measure of")
    print("one, and the paper must present it purely as an ordering.")

json.dump(dict(reps=REPS, null=null_rows, calibration=cal_rows,
               null_median=float(np.median(allm)),
               null_worst=float(np.nanmax(allm)),
               null_worst_p90=float(np.nanmax(allp)),
               cal_excess_median=float(np.median(rr)),
               cal_excess_min=float(rr.min()),
               cal_excess_max=float(rr.max())),
          open(os.path.join(HERE, "floor_calibration.json"), "w"), indent=2,
          default=float)
