"""
The minimum window: validated, and its monotonicity settled.

Proposition 1 names three design quantities.  The earliest usable age is checked
throughout the paper and the censoring efficiency is checked in Section 10, but
the minimum window w*(tau_0) = tau_0 - G^{-1}(G(tau_0) - c) appears in the
abstract, the contributions and the conclusion without ever being measured.  It
is the quantity a buffer is actually sized from, so leaving it asserted is the
weakest point in the theory section.  Two things are done about it.

FIRST, its shape.  Differentiating the defining identity G(tau_0) -
G(tau_0 - w) = c implicitly, with g = G' the information density,

    g(tau_0) - g(tau_0 - w)(1 - w') = 0   =>   w' = 1 - g(tau_0)/g(tau_0 - w),

so w*' <= 0 exactly when g(tau_0) >= g(tau_0 - w): the required window shrinks
with age if and only if the information density is larger at the later end of it.
For a degrading system, whose density rises, the buffer requirement therefore
falls monotonically, and the worst case is the earliest usable age.  That is a
design rule -- provision at tau_min and never revisit it -- and it is checked on
the records rather than assumed.

SECOND, its value.  The window is simulated: each record is driven by its own
smoothed trend and its own resampled residuals, the windowed weighted
maximum-likelihood estimator of Proposition 1 is run over [tau_0 - w, tau_0] for
w a multiple of w*, and the achieved standard deviation is compared with the
target the window was sized for.  The prediction is sharp: the achieved error
should cross the target at w = w* and nowhere else.

The heavy-tail factor found in Section 6.6 is present here too, since the same
robust scale enters the same floor.  It is measured and divided out exactly as
there, and the raw figures are kept.
"""
import os
import json
import numpy as np
from scipy.signal import savgol_filter

from pipeline import robust_scale, oof_trend, local_scale, _win, LOCAL_BW

HERE = os.path.dirname(os.path.abspath(__file__))
SEED = 20260828
QSTAR = 0.35
REPS = 300
MULT = (0.4, 0.7, 1.0, 1.5, 2.5)
TAU0 = (0.55, 0.70, 0.85, 0.95)


def prep(x):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    n = len(x)
    if n < 200:
        return None
    s = robust_scale(x)
    if not np.isfinite(s) or s <= 0:
        return None
    D = (x - np.median(x)) / s
    w = _win(n)
    if w < 7:
        return None
    T = savgol_filter(D, w, 2)
    d1 = savgol_filter(D, w, 2, deriv=1, delta=1.0 / n)
    res = D - oof_trend(D, w // 2)
    sig = np.clip(local_scale(res, max(5, int(LOCAL_BW * n))), 1e-12, None)
    return dict(T=T, d1=d1, sig=sig, res=res, n=n,
                dens=(d1 / sig) ** 2)


def w_star(dens, tau0, c):
    """tau_0 - G^{-1}(G(tau_0) - c), read off the measured density."""
    n = len(dens)
    k0 = min(n - 1, max(1, int(round(tau0 * n)) - 1))
    G = np.concatenate([[0.0], np.cumsum(dens)])
    target = G[k0 + 1] - c
    if target < 0:
        return np.nan                       # infeasible at this age
    j = int(np.searchsorted(G, target))
    return (k0 + 1 - j) / n


z = np.load(os.path.join(HERE, "feat_raw.npz"), allow_pickle=True)
FN = [str(s) for s in z["featnames"]]
I = {k: i for i, k in enumerate(FN)}
units = sorted(k for k in z.files if k != "featnames")

recs = []
for u in units:
    p = prep(np.asarray(z[u][:, I["rms"]], float))
    if p is not None:
        recs.append((u, p))
print(f"{len(recs)} bearings, root-mean-square amplitude, demand at "
      f"q = {QSTAR}\n")

# --- 1. is the window monotone in age, and does the sign law hold? ----------
# A fixed grid of ages is the wrong place to look: with a demand of 0.35 of the
# whole budget, root-mean-square amplitude has accumulated nothing like that
# before tau = 0.97, so w* is undefined at every age below it and a fixed grid
# returns almost nothing.  The ages are therefore placed by each record's OWN
# information: tau_0 = G^{-1}(f G(1)) for f above the demand, where the window
# is defined by construction.
#
# The proposition is also a LOCAL statement -- w' = 1 - g(tau_0)/g(tau_0 - w) --
# so it is tested pair by pair: between consecutive ages the window should fall
# exactly when the density is larger at the late end of the window than at the
# early end.  Testing a global "the density rises" instead tests something the
# proposition never claims.
FRACS = (0.50, 0.65, 0.80, 0.95)
SMOOTH_G = 25          # the sign law compares densities, which need smoothing


def age_at(dens, frac):
    G = np.cumsum(dens)
    k = int(np.searchsorted(G, frac * G[-1]))
    return (k + 1.0) / len(dens) if k < len(dens) else np.nan


def dens_at(dens, tau, k=SMOOTH_G):
    n = len(dens)
    i = min(n - 1, max(0, int(round(tau * n)) - 1))
    lo, hi = max(0, i - k), min(n, i + k + 1)
    return float(np.median(dens[lo:hi]))


print("shape of the minimum window, at ages placed by each record's own "
      "information\n")
print(f"{'bearing':<13}" + "".join(f"{f:>10.2f}" for f in FRACS)
      + f"{'falls':>8}{'sign law':>11}")
print("-" * 72)
shape, mono_ok, law_ok, pairs = [], 0, 0, 0
for u, p in recs:
    c = QSTAR * float(p["dens"].sum())
    t0s = [age_at(p["dens"], f) for f in FRACS]
    ws = [w_star(p["dens"], t, c) if np.isfinite(t) else np.nan for t in t0s]
    fin = [(t, w) for t, w in zip(t0s, ws) if np.isfinite(t) and np.isfinite(w)]
    falls = len(fin) >= 2 and all(b[1] <= a[1] + 1e-9
                                  for a, b in zip(fin, fin[1:]))
    # the sign law, pair by pair
    agree = True
    for (ta, wa), (tb, wb) in zip(fin, fin[1:]):
        pairs += 1
        predicted_fall = dens_at(p["dens"], tb) >= dens_at(p["dens"], tb - wb)
        observed_fall = wb <= wa + 1e-9
        if predicted_fall == observed_fall:
            law_ok += 1
        else:
            agree = False
    mono_ok += int(falls)
    shape.append(dict(unit=u, ages=[None if not np.isfinite(t) else float(t)
                                    for t in t0s],
                      ws=[None if not np.isfinite(w) else float(w) for w in ws],
                      falls=bool(falls), law=bool(agree)))
    print(f"{u:<13}"
          + "".join(f"{w:>10.3f}" if np.isfinite(w) else f"{'--':>10}"
                    for w in ws)
          + f"{('yes' if falls else 'no'):>8}{('yes' if agree else 'NO'):>11}")
print("-" * 72)
print(f"the window falls with age on {mono_ok} of {len(recs)} bearings")
print(f"the sign law w' = 1 - g(tau_0)/g(tau_0-w) predicts the direction "
      f"correctly")
print(f"in {law_ok} of {pairs} consecutive pairs")
disagree = [s for s in shape if not s["law"]]
print(f"bearings with at least one pair against the law: {len(disagree)}"
      f"{'' if not disagree else '  ' + ', '.join(s['unit'] for s in disagree)}")
print()

# --- 2. does the window deliver the accuracy it was sized for? --------------
print("achieved error against the target the window was sized for\n")
print("A multiple of one is the window the proposition prescribes; the achieved")
print("error should cross the target there.\n")
rng = np.random.default_rng(SEED)
print(f"{'w / w*':>8}{'cells':>8}{'raw':>10}{'heavy tail':>12}"
      f"{'corrected':>11}{'verdict':>26}")
print("-" * 75)
rows = []
for mult in MULT:
    got, hv = [], []
    for u, p in recs:
        n = p["n"]
        c = QSTAR * float(p["dens"].sum())
        std = p["res"] / p["sig"]
        std = std[np.isfinite(std)]
        h = float(np.std(std))
        tau = (np.arange(n) + 1.0) / n
        for f in FRACS:
            t0 = age_at(p["dens"], f)
            if not np.isfinite(t0):
                continue
            w0 = w_star(p["dens"], t0, c)
            if not np.isfinite(w0) or w0 <= 0:
                continue
            w = mult * w0
            sel = (tau > t0 - w) & (tau <= t0)
            if sel.sum() < 8:
                continue
            d1, sg = p["d1"][sel], p["sig"][sel]
            Gw = float((d1 ** 2 / sg ** 2).sum())
            if Gw <= 0:
                continue
            wgt = (d1 / sg ** 2) / Gw
            base = p["T"][sel]
            est = np.empty(REPS)
            for r in range(REPS):
                y = base + sg * rng.choice(std, size=int(sel.sum()),
                                           replace=True)
                est[r] = float(wgt @ (y - base))
            got.append(float(np.std(est, ddof=1)) * np.sqrt(c))
            hv.append(h)
    if not got:
        continue
    raw = float(np.median(got))
    hm = float(np.median(hv))
    corr = float(np.median(np.array(got) / np.array(hv)))
    verdict = ("meets the target" if corr <= 1.02 else
               "misses the target")
    rows.append(dict(mult=mult, cells=len(got), raw=raw, heavy=hm,
                     corrected=corr, meets=bool(corr <= 1.02)))
    print(f"{mult:>8.1f}{len(got):>8}{raw:>10.2f}{hm:>12.2f}{corr:>11.2f}"
          f"{verdict:>26}")
print("-" * 75)
print("The figure is the achieved standard deviation divided by the target, so")
print("one is exactly the demand and below one is a surplus.\n")

at1 = next((r for r in rows if abs(r["mult"] - 1.0) < 1e-9), None)
under = [r for r in rows if r["mult"] < 1.0]
over = [r for r in rows if r["mult"] > 1.0]
if at1:
    print(f"At the prescribed window the achieved error is {at1['corrected']:.2f} "
          f"times the target.")
if under and all(not r["meets"] for r in under) and at1 and at1["meets"]:
    print("Every narrower window misses it and the prescribed one does not, so")
    print("w* is the smallest window that works and not merely a sufficient one.")
elif under and any(r["meets"] for r in under):
    print("A narrower window than w* already meets the target, so the")
    print("proposition is conservative here rather than tight; the smallest")
    print(f"multiple that suffices is {min(r['mult'] for r in under if r['meets']):.1f}.")
if over:
    print(f"Widening to {max(r['mult'] for r in over):.1f} times w* buys a further factor "
          f"{at1['corrected'] / min(r['corrected'] for r in over):.2f},")
    print("which is the surplus a longer buffer purchases.")

json.dump(dict(units=len(recs), qstar=QSTAR, fracs=list(FRACS),
               shape=shape, falls_count=mono_ok,
               sign_law_ok=law_ok, sign_law_pairs=pairs,
               disagree=[s["unit"] for s in disagree],
               window=rows),
          open(os.path.join(HERE, "window_validate.json"), "w"), indent=2,
          default=float)
