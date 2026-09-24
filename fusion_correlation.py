"""
Additivity assumed independent noise. Real indicators do not have it.

Gamma_total = sum_j Gamma_j holds only if the indicators' observation noises are
independent. Bearing indicators are all computed from the same vibration
snapshot, so they cannot be. With a residual covariance Sigma across indicators
the correct information is the multivariate form

    Gamma_W = sum_k  d(tau_k)^T Sigma^{-1} d(tau_k),
    d(tau)  = ( D_1'(tau), ..., D_K'(tau) )^T,

which is NOT generally the sum of the diagonal terms. It can be far larger --
correlated noise with differently-shaped signals is common-mode that the
inverse covariance cancels -- or smaller, when the signals are aligned with the
noise. Which happens is an empirical question, and it decides whether the
additive fusion result is usable on real data or only on paper.

Everything is computed in the normalised indicator units, in which Gamma_j is
invariant to affine rescaling of the indicator.
"""
import os
import json
import numpy as np
from pipeline import prepare

HERE = os.path.dirname(os.path.abspath(__file__))
RIDGE = 1e-9          # guards Sigma against exact rank deficiency


def gammas(chans):
    """Per-indicator information, the independent-sum total, and the true one."""
    D = np.column_stack([u["dtrend"] for u in chans])     # n x K derivative
    R = np.column_stack([u["res"] for u in chans])        # n x K residual
    n, K = D.shape
    S = np.cov(R, rowvar=False)
    if K == 1:
        S = np.array([[float(S)]])
    S = S + RIDGE * np.trace(S) / K * np.eye(K)
    Si = np.linalg.inv(S)
    g_ind = np.array([float(np.sum(D[:, j] ** 2) / S[j, j]) for j in range(K)])
    g_corr = float(np.einsum("kj,ji,ki->", D, Si, D))
    sd = np.sqrt(np.diag(S))
    C = S / np.outer(sd, sd)
    off = C[~np.eye(K, dtype=bool)]
    return g_ind, g_corr, float(np.median(np.abs(off))), float(np.max(np.abs(off)))


def load_bearings(names):
    z = np.load(os.path.join(HERE, "feat_raw.npz"), allow_pickle=True)
    FN = [str(s) for s in z["featnames"]]
    I = {k: i for i, k in enumerate(FN)}
    out = []
    for b in sorted(k for k in z.files if k != "featnames"):
        M = z[b]
        cols = []
        for nm in names:
            x = sum(M[:, I[p]] for p in nm.split("+"))
            u, _ = prepare(np.asarray(x, float), f"{b}/{nm}")
            if u is None:
                cols = None
                break
            cols.append(u)
        if cols and len({u["n"] for u in cols}) == 1:
            out.append((b, cols))
    return out


SETS = {
    "RMS + 4--10 kHz": ["rms", "b4_6+b6_8+b8_10"],
    "RMS + 0--2 kHz": ["rms", "b0_1+b1_2"],
    "0--2 + 4--10 kHz": ["b0_1+b1_2", "b4_6+b6_8+b8_10"],
    "all six bands": ["b0_1", "b1_2", "b2_4", "b4_6", "b6_8", "b8_10"],
    "RMS + peak + kurtosis": ["rms", "peak", "kurt"],
}

print(f"{'indicator set':<24}{'units':>6}{'|corr| med':>12}{'max':>7}"
      f"{'sum of Gamma':>14}{'true Gamma':>12}{'ratio':>8}")
print("-" * 83)
rows = []
for lab, names in SETS.items():
    data = load_bearings(names)
    if not data:
        print(f"{lab:<24}  no unit has every channel")
        continue
    R = []
    for b, cols in data:
        gi, gc, cmed, cmax = gammas(cols)
        R.append((gi.sum(), gc, cmed, cmax, gi.max()))
    s = np.array([r[0] for r in R]); c = np.array([r[1] for r in R])
    rows.append(dict(set=lab, units=len(R),
                     corr_med=float(np.median([r[2] for r in R])),
                     corr_max=float(np.max([r[3] for r in R])),
                     sum_gamma=float(np.median(s)),
                     true_gamma=float(np.median(c)),
                     ratio=float(np.median(c / s)),
                     best_single=float(np.median([r[4] for r in R]))))
    r = rows[-1]
    print(f"{lab:<24}{r['units']:>6}{r['corr_med']:>12.3f}{r['corr_max']:>7.3f}"
          f"{r['sum_gamma']:>14.3g}{r['true_gamma']:>12.3g}{r['ratio']:>8.2f}")

print("-" * 83)
print("ratio = true information / what additivity would predict.")
print("  > 1 : the correlation is common-mode and cancels; fusion beats the sum")
print("  < 1 : the indicators repeat each other; fusion is worse than the sum")
print()
for r in rows:
    gain = r["true_gamma"] / r["best_single"]
    print(f"  {r['set']:<24} fused carries {gain:>6.1f}x the best single "
          f"indicator")
json.dump(rows, open(os.path.join(HERE, "fusion_correlation.json"), "w"),
          indent=2)
