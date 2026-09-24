"""
Preparation and joint information for the multi-indicator case.

Three things have to be right before any fusion result can be believed.

1. Normalisation must not screen channels. pipeline.prepare divides by the net
   rise over life and rejects channels whose rise is small against their noise.
   That is right for one indicator and wrong here: a channel that carries little
   alone still contributes. The joint information

       d(tau)^T Sigma^{-1} d(tau),   d = (D_1', ..., D_K')^T

   is invariant under per-channel affine rescaling (d -> A d, Sigma -> A Sigma
   A^T with A diagonal), so the normalisation is free and is chosen here for
   conditioning only. verify_invariance() checks that numerically.

2. The single-channel and joint expressions must use the same scale. A robust
   scale on one side and a sample covariance on the other disagree by an order
   of magnitude on heavy-tailed residuals, which showed up as a one-channel set
   reporting a tenth of its own information. Sigma is therefore built as
   diag(s) C diag(s) with s the per-channel robust scale and C the residual
   correlation matrix, so K = 1 reduces exactly to d^2 / s^2.

3. Inverting an estimated covariance inflates the quadratic form, and the
   inflation grows with the number of channels: for Sigma estimated from m
   samples, E[d^T S^{-1} d] = m/(m-K-1) times the truth. Left uncorrected, a
   larger indicator set wins automatically, whatever the data say. The
   correction factor (m-K-1)/m is applied throughout.
"""
import numpy as np
from scipy.signal import savgol_filter
from pipeline import robust_scale, oof_trend, local_scale, _win, LOCAL_BW

RIDGE = 1e-6      # relative, on the correlation matrix


def prepare_fusion(raw, name=""):
    """Raw channel -> derivative and out-of-fold residual, in robust units."""
    x = np.asarray(raw, float)
    n = len(x)
    if n < 20 or not np.all(np.isfinite(x)):
        return None, "short or non-finite"
    s = robust_scale(x)
    if not np.isfinite(s) or s <= 0:
        s = float(np.std(x))
    if not np.isfinite(s) or s <= 0:
        return None, "constant channel"
    D = (x - np.median(x)) / s
    w = _win(n)
    if w < 7:
        return None, "series too short"
    dt = savgol_filter(D, w, 2, deriv=1, delta=1.0 / n)
    res = D - oof_trend(D, w // 2)
    sig = robust_scale(res)
    if not np.isfinite(sig) or sig <= 0:
        return None, "degenerate residual"
    return dict(name=name, n=n, D=D, dtrend=dt, res=res, sig=sig,
                sig_loc=local_scale(res, max(5, int(LOCAL_BW * n)))), None


def _sigma(chans):
    """Robust covariance: robust scales on the diagonal, correlation off it."""
    R = np.column_stack([u["res"] for u in chans])
    s = np.array([u["sig"] for u in chans], float)
    K = R.shape[1]
    if K == 1:
        return np.array([[s[0] ** 2]])
    C = np.corrcoef(R, rowvar=False)
    C = np.nan_to_num(C, nan=0.0)
    C = (1 - RIDGE) * C + RIDGE * np.eye(K)
    return np.outer(s, s) * C


def joint_info(chans, debias=True):
    """Per-sample joint Fisher information, correlation and estimation bias
    both accounted for."""
    D = np.column_stack([u["dtrend"] for u in chans])
    m, K = D.shape
    S = _sigma(chans)
    q = np.einsum("kj,ji,ki->k", D, np.linalg.inv(S), D)
    if debias and K > 1:
        if m - K - 1 <= 0:
            return np.full(m, np.nan)
        q = q * (m - K - 1) / m          # inverse-covariance inflation
    return q


def single_info(u):
    return u["dtrend"] ** 2 / u["sig"] ** 2


def tau_min_from(per_sample, target):
    c = np.cumsum(per_sample)
    if not np.all(np.isfinite(c)) or c[-1] < target:
        return np.nan
    return (int(np.searchsorted(c, target)) + 1) / len(c)


def verify_invariance(seed=0):
    rng = np.random.default_rng(seed)
    n = 600
    tau = np.arange(1, n + 1) / n
    raws = [b + rng.normal(0, 0.05, n)
            for b in (tau ** 3, 1 - np.exp(-3 * tau), tau ** 0.7)]
    p1 = [prepare_fusion(x)[0] for x in raws]
    p2 = [prepare_fusion(a * x + b)[0]
          for x, a, b in zip(raws, (1.0, 1e4, 1e-3), (0.0, -7.0, 250.0))]
    g1, g2 = joint_info(p1).sum(), joint_info(p2).sum()
    return float(g1), float(g2), float(abs(g1 - g2) / g1)


def verify_single(seed=1):
    """K = 1 must reproduce the single-channel expression exactly."""
    rng = np.random.default_rng(seed)
    n = 800
    tau = np.arange(1, n + 1) / n
    # heavy-tailed noise, the case that exposed the mismatch
    e = rng.standard_t(2.5, n) * 0.05
    u = prepare_fusion(tau ** 2.5 + e)[0]
    a, b = joint_info([u]).sum(), single_info(u).sum()
    return float(a), float(b), float(abs(a - b) / b)


def verify_debias(seed=2, reps=60):
    """With pure noise the true information is nil; what remains is the bias."""
    rng = np.random.default_rng(seed)
    n = 300
    out = []
    for K in (2, 5, 10, 20):
        raw, cor = [], []
        for _ in range(reps):
            X = rng.normal(0, 1, (n, K))
            ch = [prepare_fusion(X[:, j])[0] for j in range(K)]
            if any(c is None for c in ch):
                continue
            raw.append(joint_info(ch, debias=False).sum())
            cor.append(joint_info(ch, debias=True).sum())
        out.append((K, float(np.median(raw)), float(np.median(cor))))
    return out


if __name__ == "__main__":
    g1, g2, rel = verify_invariance()
    print("1. invariance under per-channel affine rescaling")
    print(f"   {g1:.6e} vs {g2:.6e}   relative difference {rel:.1e}")

    a, b, rel = verify_single()
    print("\n2. K = 1 reduces to the single-channel expression")
    print(f"   joint {a:.6e}   single {b:.6e}   relative difference {rel:.1e}")

    print("\n3. inverse-covariance inflation, on pure noise (n = 300)")
    print(f"   {'K':>4}{'uncorrected':>14}{'corrected':>12}{'inflation':>12}")
    for K, r, c in verify_debias():
        print(f"   {K:>4}{r:>14.1f}{c:>12.1f}{r/c:>12.3f}")
    print("   the uncorrected column grows with K on data containing no")
    print("   degradation at all: that growth is what would masquerade as a")
    print("   benefit of fusing more channels.")
