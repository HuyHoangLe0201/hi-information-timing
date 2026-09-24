"""
Is the multivariate floor attained?

Section 5 validates the single-channel floor over a hundred thousand cells.  The
multivariate form d^T Sigma^{-1} d is never validated at all, and it carries
more of this paper than the scalar one does: monotonicity under channel addition,
the fusion result, the span bound and the leverage bound are all statements about
it.  Whether an estimator actually achieves it has been assumed.

The prediction is the same shape as before.  With K channels, sensitivity rows
d_k and noise covariance Sigma, the joint maximum-likelihood estimator

    delta_hat = ( sum_k d_k^T Sigma^{-1} (y_k - D_k) ) / G,
    G = sum_k d_k^T Sigma^{-1} d_k,

has standard deviation G^{-1/2}.

One design point decides whether the test means anything.  The whole content of
Sigma^{-1} is the CROSS-CHANNEL correlation, so the simulated noise must carry
it: residuals are resampled by ROW, keeping each snapshot's channels together.
Resampling each channel independently would destroy the correlation the estimator
is exploiting and would make the floor look unattainable for a reason that has
nothing to do with the bound.

Three things are then measured.  Whether the floor is attained with the paper's
own machinery.  Whether the Wishart correction m/(m-K-1), which the paper applies
throughout, is doing the work it is supposed to -- checked by turning it off.
And whether the tail factor of Proposition 11 appears here too, which it must,
since the same robust scale enters the same floor.
"""
import os
import json
import numpy as np
from scipy.signal import savgol_filter

from pipeline import robust_scale, oof_trend, local_scale, _win, LOCAL_BW

HERE = os.path.dirname(os.path.abspath(__file__))
SEED = 20260828
REPS = 500
COND_MAX = 1e3
RIDGE = 1e-6
SETS = {
    "RMS + peak": ["rms", "peak"],
    "RMS + peak + kurt": ["rms", "peak", "kurt"],
    "three bands": ["b4_6", "b6_8", "b8_10"],
    "six bands": ["b0_1", "b1_2", "b2_4", "b4_6", "b6_8", "b8_10"],
}


def prep(x):
    x = np.asarray(x, float)
    n = len(x)
    if n < 200 or not np.all(np.isfinite(x)):
        return None
    s = robust_scale(x)
    if not np.isfinite(s) or s <= 0:
        s = float(np.std(x))
    if not np.isfinite(s) or s <= 0:
        return None
    D = (x - np.median(x)) / s
    w = _win(n)
    if w < 7:
        return None
    return dict(T=savgol_filter(D, w, 2),
                d1=savgol_filter(D, w, 2, deriv=1, delta=1.0 / n),
                res=D - oof_trend(D, w // 2), n=n)


def cond_of(R):
    return float(np.linalg.cond(np.nan_to_num(np.corrcoef(R, rowvar=False),
                                              nan=0.0)))


def rank_safe(R, names):
    keep = list(range(len(names)))
    while len(keep) > 1 and cond_of(R[:, keep]) > COND_MAX:
        keep.remove(min(keep,
                        key=lambda j: cond_of(R[:, [k for k in keep
                                                    if k != j]])))
    return keep


def sigma_of(R):
    K = R.shape[1]
    s = np.array([robust_scale(R[:, j]) for j in range(K)], float)
    if not np.isfinite(s).all() or (s <= 0).any():
        return None
    C = np.nan_to_num(np.corrcoef(R, rowvar=False), nan=0.0) if K > 1 \
        else np.ones((1, 1))
    C = (1 - RIDGE) * C + RIDGE * np.eye(K)
    return np.outer(s, s) * C


z = np.load(os.path.join(HERE, "feat_raw.npz"), allow_pickle=True)
FN = [str(s) for s in z["featnames"]]
I = {k: i for i, k in enumerate(FN)}
units = sorted(k for k in z.files if k != "featnames")

prepared = []
for u in units:
    ch = {nm: prep(np.asarray(z[u][:, I[nm]], float)) for nm in FN}
    if all(v is not None for v in ch.values()) and \
            len({v["n"] for v in ch.values()}) == 1:
        prepared.append((u, ch))
print(f"{len(prepared)} bearings carrying all {len(FN)} channels\n")
print("residuals are resampled BY ROW, so the cross-channel correlation the")
print(f"estimator exploits is preserved; {REPS} repetitions per cell\n")

rng = np.random.default_rng(SEED)
print(f"{'set':<20}{'K':>3}{'cells':>7}{'raw':>9}{'tail':>8}"
      f"{'scalar fix':>12}{'exact Cov':>11}")
print("-" * 70)
rows = []
for lab, names in SETS.items():
    raw, tails, nowis, exacts = [], [], [], []
    for u, ch in prepared:
        R = np.column_stack([ch[nm]["res"] for nm in names])
        keep = rank_safe(R, names)
        nm2 = [names[j] for j in keep]
        R = R[:, keep]
        S = sigma_of(R)
        if S is None:
            continue
        D = np.column_stack([ch[nm]["d1"] for nm in nm2])
        T = np.column_stack([ch[nm]["T"] for nm in nm2])
        m, K = D.shape
        if m <= K + 2:
            continue
        Si = np.linalg.inv(S)
        G = float(np.einsum("kj,ji,ki->", D, Si, D))
        if G <= 0:
            continue
        infl = (m - K - 1) / m                    # the Wishart correction
        Gc = G * infl
        # the tail factor, multivariate: the residual whitened by Sigma
        L = np.linalg.cholesky(S)
        Z = np.linalg.solve(L, R.T).T
        kap = float(np.median([np.std(Z[:, j], ddof=1)
                               / max(robust_scale(Z[:, j]), 1e-12)
                               for j in range(K)]))
        w = (D @ Si) / Gc
        est = np.empty(REPS)
        for r in range(REPS):
            idx = rng.integers(0, m, size=m)       # whole snapshots, not channels
            Y = T + R[idx]
            est[r] = float((w * (Y - T)).sum())
        sd = float(np.std(est, ddof=1))
        # The exact prediction, when the noise has second-moment covariance C
        # rather than the robust Sigma the floor was built from:
        #   Var = sum_k w_k^T C w_k  (rows are resampled independently)
        C = np.cov(R, rowvar=False)
        C = np.atleast_2d(C)
        exact = float(np.sqrt(np.einsum("kj,ji,ki->", w, C, w)))
        raw.append(sd * np.sqrt(Gc))
        tails.append(kap)
        nowis.append(sd * np.sqrt(G))
        exacts.append(sd / max(exact, 1e-300))
    if not raw:
        continue
    mr, mt = float(np.median(raw)), float(np.median(tails))
    corr = float(np.median(np.array(raw) / np.array(tails)))
    mn = float(np.median(nowis))
    me = float(np.median(exacts))
    rows.append(dict(set=lab, K=len(names), cells=len(raw), raw=mr, tail=mt,
                     corrected=corr, no_wishart=mn, exact=me))
    print(f"{lab:<20}{len(names):>3}{len(raw):>7}{mr:>9.2f}{mt:>8.2f}"
          f"{corr:>12.2f}{me:>11.2f}")
print("-" * 70)
print("Columns are the achieved standard deviation divided by the predicted")
print("floor, so one is exact attainment.  'corrected' divides out the")
print("multivariate tail factor of Proposition 11, measured on the residual")
print("whitened by Sigma.\n")

ee = np.array([r["exact"] for r in rows])
cc = np.array([r["corrected"] for r in rows])
print(f"against the floor as built, tail divided out : median "
      f"{np.median(cc):.2f}, range {cc.min():.2f} to {cc.max():.2f}")
print(f"against the exact covariance of the noise    : median "
      f"{np.median(ee):.2f}, range {ee.min():.2f} to {ee.max():.2f}")
print()
if abs(np.log(np.median(ee))) < np.log(1.08) and ee.max() / ee.min() < 1.2:
    print("The estimator attains the floor the noise it actually sees implies.")
    print("The excess against the floor AS BUILT is therefore not a failure of")
    print("the multivariate bound but the price of building it from a robust")
    print("covariance: the scalar tail factor of Proposition 11 corrects the")
    print("SCALE of that mismatch and cannot correct its SHAPE, because a robust")
    print("correlation matrix and a product-moment one differ in more than a")
    print("common multiplier.  With K channels that shape error is what remains,")
    print(f"and it costs a median {np.median(cc):.2f} where the scalar case costs 1.35.")
else:
    print("The estimator does not attain even the floor implied by the exact")
    print("noise covariance, so the departure is not explained by the robust")
    print("covariance and is reported unexplained.")
print()
wr = np.array([r["no_wishart"] for r in rows])
rr = np.array([r["raw"] for r in rows])
print(f"turning the Wishart correction off moves attainment by "
      f"{100 * abs(np.median(wr) / np.median(rr) - 1):.1f} per cent")
print("at these channel counts, because m is large against K.  It is short")
print("records with many channels where it would matter, and the rank guard")
print("removes those.")

json.dump(dict(reps=REPS, sets=rows,
               vs_built_median=float(np.median(cc)),
               vs_built_min=float(cc.min()),
               vs_built_max=float(cc.max()),
               vs_exact_median=float(np.median(ee)),
               vs_exact_min=float(ee.min()),
               vs_exact_max=float(ee.max())),
          open(os.path.join(HERE, "multivariate_attained.json"), "w"), indent=2,
          default=float)
