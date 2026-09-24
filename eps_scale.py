"""
What precision target puts the earliest usable age somewhere interesting?

The absolute form gave tau_min of 0.001 to 0.010 at eps = 0.02, meaning the
target is met within the first percent of life. In that regime G accumulates
roughly linearly, so tau_min is inversely proportional to the acquisition rate
by arithmetic rather than by anything about degradation, and the fitted slope of
-1 carries no information.

This finds the eps that places tau_min in the middle of the record, and then
asks whether that eps is credible -- because the same absolute G was already
shown to imply implausible precision once before, for a reason that is now
understood: the bound assumes the degradation trend is known, and a fleet prior
leaves only a quarter to a half of the nominal information.
"""
import os
import json
import numpy as np
from nonparam import weighted_density, estimate_floor

HERE = os.path.dirname(os.path.abspath(__file__))
rng = np.random.default_rng(2468)
z = np.load(os.path.join(HERE, "feat_raw.npz"), allow_pickle=True)
FN = [str(s) for s in z["featnames"]]
I = {k: i for i, k in enumerate(FN)}
BK = sorted(k for k in z.files if k != "featnames")
CH = {"rms": ["rms"], "0--2 kHz": ["b0_1", "b1_2"],
      "4--10 kHz": ["b4_6", "b6_8", "b8_10"]}

print(f"{'channel':<12}{'G(0.5)':>11}{'G(1.0)':>11}"
      f"{'eps for tau=0.5':>17}{'with nuisance':>15}")
print("-" * 66)
for lab, parts in CH.items():
    Gh, Gt = [], []
    for b in BK:
        x = np.asarray(sum(z[b][:, I[p]] for p in parts), float)
        dens, res = weighted_density(x)
        if dens is None:
            continue
        d = np.clip(dens - estimate_floor(dens, res, "surrogate", rng), 0.0, None)
        c = np.cumsum(d)
        if c[-1] <= 0:
            continue
        Gh.append(float(c[len(c) // 2])); Gt.append(float(c[-1]))
    if not Gh:
        continue
    gh, gt = float(np.median(Gh)), float(np.median(Gt))
    eps = 1.0 / np.sqrt(gh)
    # only a quarter to a half of nominal information survives once the trend
    # must be inferred (fleet_prior.json), and a practical model is far from
    # efficient again on top of that
    eps_real = eps / np.sqrt(0.35)
    print(f"{lab:<12}{gh:>11.3g}{gt:>11.3g}{eps:>17.5f}{eps_real:>15.5f}")
print("-" * 66)
print("eps is in units of normalised life: 0.001 means resolving the damage")
print("clock to a tenth of a percent of a lifetime, which no one measures and")
print("no fleet would support. The absolute calibration is not usable, so the")
print("rate rule can only be stated as a shape: how tau_min moves, not from where.")
