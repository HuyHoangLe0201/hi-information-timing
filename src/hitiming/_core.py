"""Numerical building blocks shared by the profile and the causal estimator.

These functions reproduce, operation for operation, the ones used to compute
the results reported in the paper, so that the package returns the same
numbers from the same records.
"""
import numpy as np
from scipy.signal import savgol_filter

K_FOLDS = 10            # interleaved folds for the out-of-fold trend
MIN_FIT = 6             # points needed for a local quadratic fit


def robust_scale(x):
    """Normalised median absolute deviation, 1.4826 * MAD."""
    x = np.asarray(x, float)
    return float(1.4826 * np.median(np.abs(x - np.median(x))))


def window_length(n, bandwidth):
    """Odd Savitzky-Golay window: `bandwidth` of the record, at least 7 samples."""
    w = max(7, int(bandwidth * n))
    w += (w % 2 == 0)
    return min(w, n - 1 if (n - 1) % 2 == 1 else n - 2)


def oof_trend_interleaved(D, half):
    """Local-quadratic trend whose value at sample i never uses sample i.

    The record is split into ten interleaved folds; the trend at each sample is
    fitted on the samples of the other folds within `half` of it.
    """
    n = len(D)
    idx = np.arange(n)
    pred = np.empty(n)
    for j in range(K_FOLDS):
        keep = idx % K_FOLDS != j
        for i in idx[idx % K_FOLDS == j]:
            lo, hi = max(0, i - half), min(n, i + half + 1)
            m = keep[lo:hi]
            xs = idx[lo:hi][m].astype(float) - i
            pred[i] = D[i] if len(xs) < MIN_FIT else np.polyfit(xs, D[lo:hi][m], 2)[-1]
    return pred


def oof_trend_purged(D, half, purge=3):
    """h-block trend: the fit at i excludes i and its `purge` neighbours on each
    side, and the window is widened by `purge` so the fit keeps its point count.
    Useful as a check when residuals are strongly serially correlated.
    """
    n = len(D)
    idx = np.arange(n)
    pred = np.empty(n)
    h = half + purge
    for i in range(n):
        lo, hi = max(0, i - h), min(n, i + h + 1)
        xs = idx[lo:hi] - i
        keep = np.abs(xs) > purge
        if keep.sum() < MIN_FIT:
            pred[i] = D[i]
        else:
            pred[i] = np.polyfit(xs[keep].astype(float), D[lo:hi][keep], 2)[-1]
    return pred


def local_scale(r, half):
    """Robust scale of `r` over a centred window of half-width `half`."""
    n = len(r)
    out = np.empty(n)
    if n > 2 * half:
        W = np.lib.stride_tricks.sliding_window_view(r, 2 * half + 1)
        med = np.median(W, axis=1)
        out[half:n - half] = 1.4826 * np.median(np.abs(W - med[:, None]), axis=1)
        edges = list(range(half)) + list(range(n - half, n))
    else:
        edges = range(n)
    for i in edges:
        lo, hi = max(0, i - half), min(n, i + half + 1)
        out[i] = robust_scale(r[lo:hi])
    return np.maximum(out, 1e-9)


def standardise(x):
    """Centre on the median and divide by the robust scale (std as fallback)."""
    x = np.asarray(x, float)
    s = robust_scale(x)
    if not np.isfinite(s) or s <= 0:
        s = float(np.std(x))
    if not np.isfinite(s) or s <= 0:
        return None
    return (x - np.median(x)) / s


def derivative(D, w):
    """Savitzky-Golay (quadratic) derivative per unit of normalised age."""
    n = len(D)
    return savgol_filter(D, w, 2, deriv=1, delta=1.0 / n)


def smooth(D, w):
    return savgol_filter(D, w, 2)
