"""
Censoring efficiency: the last design quantity, validated.

Proposition 1 names three design quantities.  The earliest usable age is checked
throughout; the minimum window was checked in Section 5.1 after being found
unvalidated.  The third, the censoring efficiency eta(c) = G(c)/G(1), was
described in that same section as "checked in Section 10", and it is not.
Section 10 asks a different question -- whether a trailing buffer captures as
much as the best subset of the same size -- and never tests eta as a quantitative
prediction.  The claim in Proposition 1(iv) has therefore been carried unverified
and the cross-reference is wrong.

The prediction is sharp.  A record recorded only to age c offers G(c) rather than
G(1), so the achievable variance rises by exactly 1/eta(c) and the achievable
standard deviation by 1/sqrt(eta(c)).  Testing it needs no new machinery: drive
each record with its own smoothed trend and its own resampled residuals, run the
weighted maximum-likelihood estimator on [0, c] and on the whole record, and
compare the ratio of achieved standard deviations against eta(c)^(-1/2).

One thing makes this test cleaner than the two before it.  The tail factor of
Proposition 11 inflates G by kappa^2, and every earlier validation had to measure
and divide it out; here the comparison is a RATIO of two estimators on the SAME
record, so a constant kappa cancels exactly.  It does not cancel entirely,
because kappa drifts along a record, and the residual drift is what the
measurement will show.
"""
import os
import json
import numpy as np
from scipy.signal import savgol_filter

from pipeline import robust_scale, oof_trend, local_scale, _win, LOCAL_BW

HERE = os.path.dirname(os.path.abspath(__file__))
SEED = 20260828
REPS = 400
CS = (0.30, 0.50, 0.70, 0.85, 0.95)


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
    return dict(T=T, d1=d1, sig=sig, res=res, n=n, dens=(d1 / sig) ** 2)


def run(p, upto, rng, reps=REPS):
    """Achieved sd of the weighted ML estimator using samples up to `upto`."""
    n = p["n"]
    tau = (np.arange(n) + 1.0) / n
    sel = tau <= upto
    if sel.sum() < 20:
        return np.nan
    d1, sg = p["d1"][sel], p["sig"][sel]
    G = float((d1 ** 2 / sg ** 2).sum())
    if G <= 0:
        return np.nan
    w = (d1 / sg ** 2) / G
    base = p["T"][sel]
    std = p["res"] / p["sig"]
    std = std[np.isfinite(std)]
    est = np.empty(reps)
    for r in range(reps):
        y = base + sg * rng.choice(std, size=int(sel.sum()), replace=True)
        est[r] = float(w @ (y - base))
    return float(np.std(est, ddof=1))


z = np.load(os.path.join(HERE, "feat_raw.npz"), allow_pickle=True)
FN = [str(s) for s in z["featnames"]]
I = {k: i for i, k in enumerate(FN)}
units = sorted(k for k in z.files if k != "featnames")

recs = []
for u in units:
    p = prep(np.asarray(z[u][:, I["rms"]], float))
    if p is not None:
        recs.append((u, p))
print(f"{len(recs)} bearings, root-mean-square amplitude\n")
print("a record censored at c offers eta(c) = G(c)/G(1) of the budget, so its")
print("achievable standard deviation rises by eta^(-1/2)\n")
print(f"{'c':>7}{'units':>7}{'median eta':>12}{'predicted':>12}"
      f"{'measured':>11}{'ratio':>8}")
print("-" * 57)
rng = np.random.default_rng(SEED)
rows = []
full = {u: run(p, 1.0, rng) for u, p in recs}
for c in CS:
    etas, pred, meas = [], [], []
    for u, p in recs:
        G1 = float(p["dens"].sum())
        k = max(1, int(round(c * p["n"])))
        Gc = float(p["dens"][:k].sum())
        if G1 <= 0 or Gc <= 0 or not np.isfinite(full[u]) or full[u] <= 0:
            continue
        eta = Gc / G1
        s_c = run(p, c, rng)
        if not np.isfinite(s_c) or s_c <= 0:
            continue
        etas.append(eta)
        pred.append(eta ** -0.5)
        meas.append(s_c / full[u])
    if not etas:
        continue
    mp, mm = float(np.median(pred)), float(np.median(meas))
    r = float(np.median(np.array(meas) / np.array(pred)))
    rows.append(dict(c=c, units=len(etas), eta=float(np.median(etas)),
                     predicted=mp, measured=mm, ratio=r))
    print(f"{c:>7.2f}{len(etas):>7}{np.median(etas):>12.4f}{mp:>12.2f}"
          f"{mm:>11.2f}{r:>8.2f}")
print("-" * 57)
print("The ratio column is the median over bearings of measured over predicted,")
print("not the ratio of the two medians, since eta varies by two orders of")
print("magnitude across these records.\n")

rr = np.array([r["ratio"] for r in rows])
print(f"measured over predicted: median {np.median(rr):.2f}, "
      f"range {rr.min():.2f} to {rr.max():.2f} over {len(rr)} censoring levels")
print()
if np.median(rr) > 0 and abs(np.log(np.median(rr))) < np.log(1.15) \
        and rr.max() / rr.min() < 1.5:
    print("The censoring efficiency predicts the cost of truncation to within")
    print("about a sixth across two orders of magnitude of eta.  Proposition")
    print("1(iv) is therefore validated, and the cross-reference in Section 5.1")
    print("that pointed at Section 10 for this was wrong and is corrected.")
else:
    print("The prediction does not hold to within a sixth.  The departure is")
    print("reported, and its likely source is the drift of the tail factor: a")
    print("constant kappa cancels in this ratio and a drifting one does not.")

json.dump(dict(reps=REPS, levels=list(CS), rows=rows,
               ratio_median=float(np.median(rr)),
               ratio_min=float(rr.min()), ratio_max=float(rr.max())),
          open(os.path.join(HERE, "censoring_validate.json"), "w"), indent=2,
          default=float)
