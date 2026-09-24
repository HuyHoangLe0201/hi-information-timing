"""
Is it the spectral distribution, or merely being a ratio?

Spectral shape indicators become usable about 0.2 of a lifetime earlier than
level indicators, on both rigs. The explanation offered was physical: early
damage redistributes vibration energy upward before it raises the level, so a
measure of the distribution sees it first.

There is a duller explanation to rule out. Every shape indicator is a ratio and
therefore bounded, while every level indicator grows without bound. If dividing
one quantity by another is what makes an indicator early -- for statistical
reasons having nothing to do with where the energy sits -- then the
recommendation should be about ratios, not about spectra, and the physical story
is wrong.

The control builds ratios that carry no spectral information at all:

  rms / constant           a pure rescaling; must not move tau_min, since the
                           density is invariant to it, and serves as a check
                           that the pipeline behaves;
  rms / slow(rms)          divided by its own long-window average -- bounded,
                           ratio-shaped, and containing nothing the level did
                           not already contain;
  rms / p2p                one level statistic over another, from the same
                           waveform;
  rms_h / rms_v            the same statistic across the two accelerometers.

If these stay late with the level indicators, the effect belongs to the spectral
distribution. If they become early, it belongs to the arithmetic.
"""
import os
import glob
import json
import numpy as np
import pandas as pd
from scipy.signal import savgol_filter
from nonparam import weighted_density, estimate_floor, quantile

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))
FEAT = os.path.join(ROOT, "GBS_RUL", "results", "features")
Q = 0.35
THRESH = 0.90
SEED = 90909


def profile(series, rng):
    T, S = [], []
    for x in series:
        x = np.asarray(x, float)
        x = x[np.isfinite(x)]
        if len(x) < 80:
            continue
        dens, res = weighted_density(x)
        if dens is None:
            continue
        d = np.clip(dens - estimate_floor(dens, res, "surrogate", rng), 0.0, None)
        if d.sum() <= 0:
            continue
        F = np.cumsum(d) / d.sum()
        t = quantile(F, Q)
        k = max(1, min(int(round(t * len(d))) - 1, len(d) - 1))
        rk = float(dens[:k + 1].sum())
        if rk <= 0:
            continue
        T.append(t)
        S.append(float(d[:k + 1].sum()) / rk)
    if len(T) < 4:
        return None
    return float(np.median(T)), float(np.median(S)), len(T)


def slow(x, frac=0.30):
    """Long-window smooth of the indicator itself: a denominator carrying
    nothing the numerator does not already carry."""
    n = len(x)
    w = max(11, int(frac * n))
    w += (w % 2 == 0)
    w = min(w, n - 1 if (n - 1) % 2 == 1 else n - 2)
    return savgol_filter(x, w, 1)


files = sorted(glob.glob(os.path.join(FEAT, "Bearing*.csv")))
tabs = [pd.read_csv(f) for f in files]
print(f"{len(tabs)} bearings\n")

EPS = 1e-30
CASES = [
    ("h_rms (level, reference)", lambda t: t["h_rms"].to_numpy(), "level"),
    ("h_rms / constant", lambda t: t["h_rms"].to_numpy() / 7.3, "control"),
    ("h_rms / slow(h_rms)",
     lambda t: t["h_rms"].to_numpy()
     / np.clip(slow(t["h_rms"].to_numpy()), EPS, None), "ratio, no spectrum"),
    ("h_rms / h_p2p",
     lambda t: t["h_rms"].to_numpy()
     / np.clip(t["h_p2p"].to_numpy(), EPS, None), "ratio, no spectrum"),
    ("h_rms / v_rms",
     lambda t: t["h_rms"].to_numpy()
     / np.clip(t["v_rms"].to_numpy(), EPS, None), "ratio, no spectrum"),
    ("h_peak / h_rms (crest)",
     lambda t: t["h_peak"].to_numpy()
     / np.clip(t["h_rms"].to_numpy(), EPS, None), "ratio, no spectrum"),
    ("h_spec_entropy", lambda t: t["h_spec_entropy"].to_numpy(),
     "spectral shape"),
    ("v_spec_entropy", lambda t: t["v_spec_entropy"].to_numpy(),
     "spectral shape"),
    ("h_spec_spread", lambda t: t["h_spec_spread"].to_numpy(),
     "spectral shape"),
]

rng = np.random.default_rng(SEED)
print(f"{'indicator':<26}{'kind':<20}{'units':>6}{'signal':>9}{'tau':>9}")
print("-" * 70)
rows = []
for lab, fn, kind in CASES:
    try:
        series = [fn(t) for t in tabs]
    except KeyError:
        continue
    p = profile(series, rng)
    if not p:
        print(f"{lab:<26}{kind:<20}   not measurable")
        continue
    t, s, n = p
    rows.append(dict(indicator=lab, kind=kind, tau=t, signal=s, units=n))
    flag = "" if s >= THRESH else "   (below signal threshold)"
    print(f"{lab:<26}{kind:<20}{n:>6}{100*s:>8.0f}%{t:>9.3f}{flag}")
print("-" * 70)

ref = next((r["tau"] for r in rows if r["kind"] == "level"), np.nan)
ctl = next((r["tau"] for r in rows if r["kind"] == "control"), np.nan)
rat = [r["tau"] for r in rows if r["kind"] == "ratio, no spectrum"
       and r["signal"] >= THRESH]
spc = [r["tau"] for r in rows if r["kind"] == "spectral shape"
       and r["signal"] >= THRESH]

print(f"\nlevel reference           : {ref:.3f}")
if np.isfinite(ctl):
    print(f"rescaled by a constant    : {ctl:.3f}   "
          f"(moves {abs(ctl-ref):.3f} -- the invariance check)")
if rat:
    print(f"ratios without spectrum   : {min(rat):.3f} to {max(rat):.3f}, "
          f"median {np.median(rat):.3f}")
if spc:
    print(f"spectral shape            : {min(spc):.3f} to {max(spc):.3f}, "
          f"median {np.median(spc):.3f}")
print()
# The verdict cannot rest on the median of the ratio group: only some of its
# members clear the signal threshold, and they differ in kind. The two that
# matter are named individually.
self_norm = next((r["tau"] for r in rows
                  if r["indicator"].startswith("h_rms / slow")), np.nan)
cross_ch = next((r["tau"] for r in rows
                 if r["indicator"].startswith("h_rms / v_rms")), np.nan)
print("the two informative controls:")
print(f"  self-normalised (h_rms / slow(h_rms)) : {self_norm:.3f}")
print(f"  cross-channel   (h_rms / v_rms)       : {cross_ch:.3f}")
print()
if np.isfinite(self_norm) and self_norm > ref - 0.05:
    print("Being a bounded ratio does nothing on its own: dividing the level by")
    print("its own slow average leaves tau_min exactly where it was. So the")
    print("effect is not the arithmetic.")
if np.isfinite(cross_ch) and cross_ch < ref - 0.2:
    print("But the ratio of the two accelerometers, which carries no spectral")
    print("information at all, is early too. It measures how the vibration is")
    print("distributed across DIRECTION rather than across frequency.")
    print()
    print("So the distinction is not spectral against non-spectral, nor ratio")
    print("against absolute. It is between measuring how vibration is")
    print("DISTRIBUTED -- over frequency or over direction -- and measuring how")
    print("MUCH of it there is. Damage changes the distribution first.")
json.dump(rows, open(os.path.join(HERE, "ratio_control.json"), "w"), indent=2,
          default=float)
