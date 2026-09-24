"""
Is tau_min invariant to how the indicator is scaled?

The framework's density is D'^2 / sigma^2. Under a smooth monotone transform
g of the indicator, the delta method sends D' to g'(D) D' and sigma to
g'(D) sigma, so the ratio is unchanged and every quantile of F with it. That
invariance is what lets indicators in different physical units be compared.

It is now in doubt. A band computed here as FFT POWER over the whole spectrum
gives tau = 0.971, while the pre-computed 4-10 kHz channel -- nominally the same
information -- gives 0.241. Power is the square of amplitude, so if squaring an
indicator moves tau_min by three quarters of a lifetime, the invariance fails on
real data and every comparison in this study rests on an arbitrary choice of
units.

Four transforms are applied to the same channel: identity, square root, square,
and logarithm. If tau_min is invariant they agree.
"""
import os
import json
import numpy as np
from nonparam import weighted_density, estimate_floor, quantile

HERE = os.path.dirname(os.path.abspath(__file__))
Q = 0.35
rng = np.random.default_rng(112233)


def tau_of(x):
    dens, res = weighted_density(np.asarray(x, float))
    if dens is None:
        return np.nan
    d = np.clip(dens - estimate_floor(dens, res, "surrogate", rng, reps=6),
                0.0, None)
    if d.sum() <= 0:
        return np.nan
    return quantile(np.cumsum(d) / d.sum(), Q)


TR = [("identity", lambda v: v),
      ("sqrt (amplitude)", lambda v: np.sqrt(np.clip(v, 0, None))),
      ("square", lambda v: v ** 2),
      ("log", lambda v: np.log(np.clip(v, 1e-30, None)))]

z = np.load(os.path.join(HERE, "fineband.npz"))
BK = sorted(k for k in z.files if k != "centres")
print("the same channel under four transforms\n")
print(f"{'channel':<22}" + "".join(f"{n:>18}" for n, _ in TR))
print("-" * (22 + 18 * len(TR)))
rows = []
for lab, sl in (("fine 4-10 kHz (power)", slice(10, 25)),
                ("fine full band (power)", slice(0, 32))):
    cells = []
    for nm, f in TR:
        v = []
        for b in BK:
            t = tau_of(f(z[b][:, sl].sum(axis=1)))
            if np.isfinite(t):
                v.append(t)
        cells.append(float(np.median(v)) if v else np.nan)
    rows.append(dict(channel=lab, **{n: c for (n, _), c in zip(TR, cells)}))
    print(f"{lab:<22}" + "".join(f"{c:>18.3f}" for c in cells))

zp = np.load(os.path.join(HERE, "feat_raw.npz"), allow_pickle=True)
FN = [str(s) for s in zp["featnames"]]
I = {k: i for i, k in enumerate(FN)}
BP = sorted(k for k in zp.files if k != "featnames")
cells = []
for nm, f in TR:
    v = []
    for b in BP:
        x = zp[b][:, I["b4_6"]] + zp[b][:, I["b6_8"]] + zp[b][:, I["b8_10"]]
        t = tau_of(f(np.asarray(x, float)))
        if np.isfinite(t):
            v.append(t)
    cells.append(float(np.median(v)) if v else np.nan)
rows.append(dict(channel="stored 4-10 kHz",
                 **{n: c for (n, _), c in zip(TR, cells)}))
print(f"{'stored 4-10 kHz':<22}" + "".join(f"{c:>18.3f}" for c in cells))
print("-" * (22 + 18 * len(TR)))

sp = []
for r in rows:
    v = [r[n] for n, _ in TR if np.isfinite(r[n])]
    if len(v) > 1:
        sp.append(max(v) - min(v))
print(f"\nlargest movement under transform: {max(sp):.3f} of life")
if max(sp) < 0.05:
    print("tau_min is invariant to how the indicator is scaled, as the delta")
    print("method says it should be. The gap between the stored channel and the")
    print("one recomputed here is therefore NOT the power-versus-amplitude")
    print("choice, and something else about the two extractions differs.")
else:
    print("tau_min is NOT invariant: scaling the indicator moves it by more")
    print("than the differences this study reports between indicators, so the")
    print("comparisons depend on an arbitrary choice of units.")
json.dump(rows, open(os.path.join(HERE, "invariance_check.json"), "w"),
          indent=2, default=float)
