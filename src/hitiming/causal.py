"""A strictly causal version of the profile, for use while a unit is in service.

At each age the derivative, the trend, the noise scale and the noise floor are
estimated from the samples observed up to that age only:

* derivative: a quadratic fitted on the trailing window, differentiated at its
  right edge;
* trend: a quadratic fitted on the window ending one sample earlier and
  extrapolated one step (a one-step-ahead prediction);
* noise scale: robust scale of the trailing residuals;
* noise floor: re-estimated at `checkpoints` points from the residuals available
  there, and held until the next checkpoint.

Two quantities still refer to the completed record: the age axis t/T and the
window widths, which are fractions of the completed life. The causal age is
therefore a retrospective measure of online estimability, and it is sensitive
to record length: short records leave the trailing window with few samples.
"""
import numpy as np

from . import _core
from .profile import (DEFAULT_BANDWIDTH, DEFAULT_FLOOR_REPS, MIN_LENGTH,
                      InformationProfile, _as_rng, floor_level)


def _edge_filters(w):
    """Least-squares quadratic on t = -(w-1)..0: value and slope at t = 0."""
    t = np.arange(-(w - 1), 1, dtype=float)
    X = np.vstack([np.ones_like(t), t, t ** 2]).T
    P = np.linalg.pinv(X)
    return P[0], P[1]


def _causal_density(x, bandwidth, scale_bandwidth):
    D = _core.standardise(x)
    if D is None:
        return None, None
    n = len(D)
    w = _core.window_length(n, bandwidth)
    if w < 7 or n < w + 2:
        return None, None
    _, c_der = _edge_filters(w)
    dt = np.zeros(n)
    pred = np.full(n, np.nan)
    for i in range(w - 1, n):
        dt[i] = float(c_der @ D[i - w + 1:i + 1]) * n
    t = np.arange(-w, 0, dtype=float)
    for i in range(w, n):
        pred[i] = np.polyval(np.polyfit(t, D[i - w:i], 2), 0.0)
    res = D - pred
    h = max(5, int(scale_bandwidth * n))
    sl = np.full(n, np.nan)
    for i in range(n):
        seg = res[max(0, i - h + 1):i + 1]
        seg = seg[np.isfinite(seg)]
        if len(seg) >= 5:
            sl[i] = _core.robust_scale(seg)
    sl = np.where(np.isfinite(sl) & (sl > 0), sl, np.nan)
    dens = np.zeros(n)
    ok = np.isfinite(sl) & (np.arange(n) >= w - 1)
    dens[ok] = (dt[ok] / sl[ok]) ** 2
    if dens.sum() <= 0:
        return None, None
    return dens, res, sl


def causal_profile(x, bandwidth=DEFAULT_BANDWIDTH, scale_bandwidth=None,
                   floor_reps=DEFAULT_FLOOR_REPS, checkpoints=40, min_pool=10,
                   rng=None, min_length=MIN_LENGTH):
    """Causal counterpart of `information_profile`.

    Returns an InformationProfile whose `floor` is an array (one level per
    sample; infinite before `min_pool` residuals exist, so no information is
    counted there), or None if the record cannot be estimated.
    """
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    if len(x) < min_length:
        return None
    scale_bandwidth = bandwidth if scale_bandwidth is None else scale_bandwidth
    out = _causal_density(x, bandwidth, scale_bandwidth)
    if out[0] is None:
        return None
    dens, res, sl = out
    rng = _as_rng(rng)
    n = len(dens)
    fin = np.isfinite(res)
    idx = np.flatnonzero(fin)
    if len(idx) < min_pool:
        return None
    i0 = int(idx[min_pool - 1])
    step = max(1, (n - i0) // checkpoints)
    floor = np.full(n, np.inf)
    for c in range(i0, n, step):
        pool = res[:c + 1][fin[:c + 1]]
        floor[c:c + step] = floor_level(pool, n, rng, floor_reps, bandwidth,
                                        scale_bandwidth)
    corrected = np.clip(dens - floor, 0.0, None)
    if corrected.sum() <= 0:
        return None
    return InformationProfile(density=dens, floor=floor, corrected=corrected,
                              residual=res, scale=sl)


def causal_information_age(x, q=0.35, **kwargs):
    """Causal information age tau_q^c of one record (NaN if not computable)."""
    p = causal_profile(x, **kwargs)
    return p.age(q) if p is not None else float("nan")
