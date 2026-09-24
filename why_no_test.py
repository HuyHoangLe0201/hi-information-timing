"""
Why the end-to-end test cannot be run on these datasets.

The noise sweep was meant to settle whether the information curve predicts model
error, by moving G over orders of magnitude with everything else fixed. It did
not work, and the way it failed is informative.

On turbofan records G did not move at all: thirty-two times the noise left it at
1.2e4, while the model's error rose from 0.15 to 0.23. On bearing records G fell
by a factor of twenty-seven, exactly as it should -- but there the model has no
skill to correlate against.

The suspected cause of the turbofan behaviour is that G there is almost entirely
floor. Adding white noise raises the Savitzky-Golay derivative and the residual
scale together, so the floor is scale-free and does not move; only the signal
part of the density falls. If the signal part is small to begin with, G is
pinned to the floor and stops measuring information.

That is checked here directly, along with the other half of the problem: whether
a bearing fleet of this size can support RUL regression at all, or whether the
no-skill baseline is simply very hard to beat when lifetimes span twelve-fold.
"""
import os
import json
import numpy as np
from nonparam import weighted_density, estimate_floor

HERE = os.path.dirname(os.path.abspath(__file__))
rng = np.random.default_rng(77)


def split(series, t0=0.60):
    """Total, floor and signal parts of G, accumulated to t0."""
    T, F = [], []
    for raw in series:
        dens, res = weighted_density(np.asarray(raw, float))
        if dens is None:
            continue
        fl = estimate_floor(dens, res, "surrogate", rng, reps=8)
        k0 = int(round(t0 * len(dens))) - 1
        if k0 < 2:
            continue
        T.append(float(dens[:k0 + 1].sum()))
        F.append(float(fl[:k0 + 1].sum()))
    if not T:
        return None
    t, f = float(np.median(T)), float(np.median(F))
    return dict(total=t, floor=f, signal=max(t - f, 0.0),
                signal_frac=max(t - f, 0.0) / t if t > 0 else np.nan)


print("how much of G is floor?\n")
print(f"{'indicator':<20}{'n':>6}{'G total':>12}{'floor':>12}"
      f"{'signal':>12}{'signal %':>10}")
print("-" * 72)
rows = []
zt = np.load(os.path.join(HERE, "cmapss_FD001.npz"))
un = sorted({k.split("__")[0] for k in zt.files})[:40]
n_t = int(np.median([len(zt[f"{u}__sensors"]) for u in un]))
for c in (4, 11, 15, 20):
    s = split([zt[f"{u}__sensors"][:, c - 1].astype(float) for u in un])
    if s:
        rows.append(dict(indicator=f"turbofan s{c}", n=n_t, **s))
z = np.load(os.path.join(HERE, "feat_raw.npz"), allow_pickle=True)
FN = [str(x) for x in z["featnames"]]
I = {k: i for i, k in enumerate(FN)}
BK = sorted(k for k in z.files if k != "featnames")
n_b = int(np.median([len(z[b]) for b in BK]))
for lab, parts in (("bearing rms", ["rms"]),
                   ("bearing 0--2k", ["b0_1", "b1_2"]),
                   ("bearing 4--10k", ["b4_6", "b6_8", "b8_10"])):
    s = split([sum(z[b][:, I[p]] for p in parts) for b in BK])
    if s:
        rows.append(dict(indicator=lab, n=n_b, **s))
for r in rows:
    print(f"{r['indicator']:<20}{r['n']:>6}{r['total']:>12.3g}"
          f"{r['floor']:>12.3g}{r['signal']:>12.3g}"
          f"{100*r['signal_frac']:>9.0f}%")
print("-" * 72)
tf = [r for r in rows if r["indicator"].startswith("turbofan")]
bg = [r for r in rows if r["indicator"].startswith("bearing")]
print(f"turbofan (n={n_t}): signal is "
      f"{100*min(r['signal_frac'] for r in tf):.0f}-"
      f"{100*max(r['signal_frac'] for r in tf):.0f}% of G")
print(f"bearing  (n={n_b}): signal is "
      f"{100*min(r['signal_frac'] for r in bg):.0f}-"
      f"{100*max(r['signal_frac'] for r in bg):.0f}% of G")
print("\nA record whose G is mostly floor cannot report a change in")
print("information, because the floor is scale-free: adding noise raises the")
print("derivative and the residual scale together and leaves it where it was.\n")

# --- the other half: can a bearing fleet of this size support the regression?
print("is the bearing fleet's own life spread the obstacle?\n")
for nm, lens in (("PRONOSTIA bearings", [len(z[b]) for b in BK]),
                 ("C-MAPSS FD001", [len(zt[f"{u}__sensors"]) for u in un])):
    L = np.array(lens, float)
    cv = L.std(ddof=1) / L.mean()
    print(f"  {nm:<22} {len(L):>3} units, life {int(L.min())}-{int(L.max())}"
          f" ({L.max()/L.min():.1f}x), CV {cv:.2f}")
print()
print("The no-skill baseline's error is the fleet's own spread of remaining")
print("life. A fleet with a wide spread sets a high baseline, but it also")
print("gives the model more to learn from; a fleet of seventeen units with a")
print("twelve-fold spread gives it very little of either.")

json.dump(dict(rows=rows), open(os.path.join(HERE, "why_no_test.json"), "w"),
          indent=2, default=float)
