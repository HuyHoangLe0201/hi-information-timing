"""
The information curve without a functional form.

The power-law summary is dropped. What replaces it is not another summary but
the observation that the design quantities ARE the quantile function of the
information distribution:

    tau_min(q)      = F^{-1}(q)                       earliest usable age
    w*(tau_0, q)    = tau_0 - F^{-1}( F(tau_0) - q )  minimum window
    eta(c)          = F(c)                            censoring efficiency

so reporting F at a few quantiles reports the whole design curve, exactly and
without assuming a shape. An exponent was only ever a way to interpolate
between those points, and it interpolates badly on the indicators that matter.

One correction survives the change, and it is not optional. Differentiating
noise produces density that no degradation put there, and it does not cancel:
it inflates the tails of F and pulls every quantile toward the middle. The floor
is estimated and subtracted before F is formed.

estimate_floor offers two routes. 'surrogate' resamples the record's own
out-of-fold residual, destroying the trend while keeping the noise, and measures
what the pipeline then reports on it; it assumes nothing but costs a bootstrap.
'flat' reads a level off record length alone. The floor IS flat in shape -- its
tilt is 0.04 to 0.07 whatever the noise profile does, against 0.03 to 3.43 for
the unweighted density -- but its LEVEL is not transferable: on real records it
runs 1.2x to 14x above what Gaussian noise of the same length gives, because
real residuals are heavy-tailed. So 'flat' is for synthetic work and
'surrogate' for measurement.
"""
import os
import numpy as np
from scipy.signal import savgol_filter
from pipeline import robust_scale, oof_trend, local_scale, _win, LOCAL_BW

HERE = os.path.dirname(os.path.abspath(__file__))
FLOOR_REPS = 8


def weighted_density(x):
    """d(tau)^2 / sigma(tau)^2, the density the Fisher information is built on."""
    x = np.asarray(x, float)
    n = len(x)
    if n < 60 or not np.all(np.isfinite(x)):
        return None, None
    s = robust_scale(x)
    if not np.isfinite(s) or s <= 0:
        s = float(np.std(x))
    if not np.isfinite(s) or s <= 0:
        return None, None
    D = (x - np.median(x)) / s
    w = _win(n)
    if w < 7:
        return None, None
    dt = savgol_filter(D, w, 2, deriv=1, delta=1.0 / n)
    res = D - oof_trend(D, w // 2)
    sl = np.clip(local_scale(res, max(5, int(LOCAL_BW * n))), 1e-12, None)
    return (dt / sl) ** 2, res


def estimate_floor(dens, res, method="surrogate", rng=None, reps=FLOOR_REPS):
    """Density the pipeline reports when there is no degradation to report."""
    if method == "none":
        return np.zeros_like(dens)
    if method == "surrogate":
        # The surrogate has no trend by construction, so the out-of-fold trend
        # -- which exists to stop a smoother absorbing real degradation, and
        # costs 238x a plain Savitzky-Golay fit -- is not needed here. Using
        # the plain residual instead agrees to 1.07x and halves the cost.
        rng = rng or np.random.default_rng(0)
        n = len(res)
        w = _win(n)
        acc = []
        for _ in range(reps):
            s = res[rng.integers(0, n, n)]
            sc = robust_scale(s)
            if not np.isfinite(sc) or sc <= 0:
                continue
            Dz = (s - np.median(s)) / sc
            dt = savgol_filter(Dz, w, 2, deriv=1, delta=1.0 / n)
            r2 = Dz - savgol_filter(Dz, w, 2)
            sl = np.clip(local_scale(r2, max(5, int(LOCAL_BW * n))), 1e-12, None)
            acc.append((dt / sl) ** 2)
        if not acc:
            return np.zeros_like(dens)
        return np.full_like(dens, float(np.median(np.concatenate(acc))))
    if method == "flat":
        return np.full_like(dens, flat_level(len(dens)))
    raise ValueError(method)


# Floor level of the weighted density on trend-free records, measured in
# floor_shape.py. Two properties make the lookup valid: the floor is flat
# along the record whatever the noise profile does (tilt 0.04-0.07 against
# 0.03-3.43 for the unweighted density), and its level does not depend on the
# noise scale -- 32.4, 32.2, 32.3 at n = 300 for sigma of 0.01, 0.05, 0.25.
# Only the record length enters.
_FLOOR_N = np.array([300.0, 600.0, 1200.0, 2400.0])
_FLOOR_V = np.array([32.3, 16.7, 9.53, 4.87])


def flat_level(n):
    """Expected weighted density where there is no degradation, at length n."""
    ln, lv = np.log(_FLOOR_N), np.log(_FLOOR_V)
    return float(np.exp(np.interp(np.log(float(n)), ln, lv)))


def info_curve(x, floor="surrogate", rng=None):
    """Normalised cumulative information F, with the noise floor removed."""
    dens, res = weighted_density(x)
    if dens is None:
        return None
    fl = estimate_floor(dens, res, floor, rng)
    d = np.clip(dens - fl, 0.0, None)
    tot = d.sum()
    if tot <= 0:
        return None
    return np.cumsum(d) / tot


def quantile(F, q):
    """tau at which the fraction q of the budget has arrived."""
    if F is None:
        return np.nan
    k = int(np.searchsorted(F, q))
    return (k + 1) / len(F) if k < len(F) else 1.0


def tau_min(F, q):
    """Earliest age at which a target costing q of the budget is reachable."""
    return quantile(F, q)


def w_star(F, tau0, q):
    """Shortest window ending at tau_0 that collects q of the budget."""
    if F is None:
        return np.nan
    n = len(F)
    k0 = int(round(tau0 * n)) - 1
    if k0 < 0 or k0 >= n:
        return np.nan
    target = F[k0] - q
    if target < 0:
        return np.nan                      # infeasible at this age
    return tau0 - quantile(F, target)


def summarise(F, qs=(0.10, 0.25, 0.50, 0.75, 0.90)):
    return {q: quantile(F, q) for q in qs}


def spread(F):
    """Fraction of life over which the middle 80% of the budget arrives."""
    return quantile(F, 0.90) - quantile(F, 0.10)
