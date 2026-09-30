"""Example 3: spectral indicators from vibration band powers.

If your data are vibration spectra (band powers per acquisition), the package
computes ten indicators from them: six that normalise each spectrum to a
distribution before summarising it (entropy, centroid, spread, Gini, top-band
share, high/low ratio) and four that use its absolute amplitude (total, peak,
high-band and log total power). The paper finds that the first group reaches a
given share of its information earlier on two bearing test rigs.

Your data: a dict unit -> array of shape (n_acquisitions, n_bands), in time
order to failure, and the band centre frequencies `f`. Here a synthetic fleet
is generated in which damage first shifts energy towards high bands and only
later raises the overall level.

    python 03_vibration_spectra.py
"""
import numpy as np

import hitiming as ht

rng = np.random.default_rng(2)
n_bands = 32
f = (np.arange(n_bands) + 0.5) * 0.4              # band centres, kHz
spectra = {}
for u in range(10):
    n = int(rng.integers(400, 800))
    tau = np.arange(1, n + 1) / n
    shape = np.exp(-f / 4.0)[None, :] * (1 + 3 * tau[:, None] ** 1.2 * (f[None, :] > 6))
    level = 1 + 20 * np.exp(10 * (tau - 1))[:, None]
    spectra["unit_%02d" % u] = shape * level * rng.lognormal(0, 0.15, (n, n_bands))

records = {name: [] for name in ht.spectral.INDICATORS}
for P in spectra.values():
    for name, series in ht.spectral.spectral_indicators(P, f).items():
        records[name].append(series)

table = ht.indicator_ages(records, q=0.35, rng=0)
ht.print_summary(table)

fam = ht.spectral.families()
res = ht.bootstrap_family_gap(table, early=fam["distribution"], late=fam["amount"],
                              n_boot=2000, seed=0)
print("\nfamily gap (amount - distribution): %.3f of life" % res["gap"])
print("95%% interval, units resampled:            [%.3f, %.3f]" % res["units"])
print("95%% interval, units and indicators too:   [%.3f, %.3f]" % res["units_and_indicators"])
