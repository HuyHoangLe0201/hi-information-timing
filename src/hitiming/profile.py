"""The cumulative information profile of one run-to-failure record."""
from dataclasses import dataclass

import numpy as np

from . import _core

DEFAULT_BANDWIDTH = 0.08        # smoother and local-scale window, fraction of life
DEFAULT_FLOOR_REPS = 8          # surrogate records used to estimate the noise floor
MIN_LENGTH = 60                 # shortest record the estimator accepts


def _as_rng(rng):
    if rng is None:
        return np.random.default_rng(0)
    if isinstance(rng, np.random.Generator):
        return rng
    return np.random.default_rng(rng)


def floor_level(pool, n, rng, reps=DEFAULT_FLOOR_REPS, bandwidth=DEFAULT_BANDWIDTH,
                scale_bandwidth=DEFAULT_BANDWIDTH):
    """Density the estimator reports on a record that contains no degradation.

    The residuals in `pool` are resampled into `reps` trend-free records of
    length `n`, each is run through the same derivative and local scale as the
    real record, and the median of the resulting density is returned.
    """
    w = _core.window_length(n, bandwidth)
    acc = []
    for _ in range(reps):
        s = pool[rng.integers(0, len(pool), n)]
        sc = _core.robust_scale(s)
        if not np.isfinite(sc) or sc <= 0:
            continue
        Dz = (s - np.median(s)) / sc
        dt = _core.derivative(Dz, w)
        r2 = Dz - _core.smooth(Dz, w)
        sl = np.clip(_core.local_scale(r2, max(5, int(scale_bandwidth * n))), 1e-12, None)
        acc.append((dt / sl) ** 2)
    return float(np.median(np.concatenate(acc))) if acc else 0.0


@dataclass
class InformationProfile:
    """Fisher-information profile of one record, sampled at ages k/n, k = 1..n.

    Attributes
    ----------
    density : weighted density (D'/sigma)^2 at each sample, before the floor.
    floor : noise-floor level subtracted from the density (scalar or array).
    corrected : max(density - floor, 0), the information attributed to degradation.
    residual : out-of-fold residual of the standardised record.
    scale : local robust noise scale used in the weighting.
    """
    density: np.ndarray
    floor: object
    corrected: np.ndarray
    residual: np.ndarray
    scale: np.ndarray

    @property
    def n(self):
        return len(self.corrected)

    @property
    def tau(self):
        """Normalised age of each sample, t/T."""
        return np.arange(1, self.n + 1) / self.n

    @property
    def budget(self):
        """Total floor-corrected information of the record.

        Its absolute value depends on how the trend was estimated; compare
        budgets only between indicators computed on the same records.
        """
        return float(self.corrected.sum())

    @property
    def curve(self):
        """Normalised cumulative information F(tau), rising from 0 to 1."""
        return np.cumsum(self.corrected) / self.budget

    def age(self, q=0.35):
        """Information age tau_q: the fraction of life by which the share q of
        the record's information has arrived. Smaller means earlier."""
        F = self.curve
        k = int(np.searchsorted(F, q))
        return (k + 1) / self.n if k < self.n else 1.0

    def surviving_fraction(self, q=0.35):
        """Share of the density up to tau_q that remains after the noise floor.

        A value near 1 means the age is read from degradation; values below
        about 0.9 mean it is read largely from noise and should be flagged.
        """
        t = self.age(q)
        k = max(1, min(int(round(t * self.n)) - 1, self.n - 1))
        raw = float(self.density[:k + 1].sum())
        return float(self.corrected[:k + 1].sum()) / raw if raw > 0 else float("nan")

    def is_reliable(self, q=0.35, threshold=0.90):
        return self.surviving_fraction(q) >= threshold

    def window(self, tau0, q):
        """Shortest window ending at age tau0 that collects the share q of the
        information; NaN if the record holds less than q before tau0."""
        F = self.curve
        k0 = int(round(tau0 * self.n)) - 1
        if k0 < 0 or k0 >= self.n or F[k0] - q < 0:
            return float("nan")
        k = int(np.searchsorted(F, F[k0] - q))
        start = (k + 1) / self.n if k < self.n else 1.0
        return tau0 - start

    def censoring_efficiency(self, c):
        """Share of the information available if the record stops at age c."""
        k = int(round(c * self.n)) - 1
        if k < 0:
            return 0.0
        return float(self.curve[min(k, self.n - 1)])


def information_profile(x, bandwidth=DEFAULT_BANDWIDTH, scale_bandwidth=None,
                        floor="surrogate", floor_reps=DEFAULT_FLOOR_REPS,
                        oof="interleaved", purge=3, rng=None, min_length=MIN_LENGTH):
    """Compute the cumulative information profile of one run-to-failure record.

    Parameters
    ----------
    x : 1-D array of health-indicator values, one per inspection, in time order,
        from the start of monitoring to failure. Inspections are assumed to be
        equally spaced in time; non-finite values are dropped.
    bandwidth : Savitzky-Golay window as a fraction of the record (default 0.08).
    scale_bandwidth : window of the local noise scale (defaults to `bandwidth`).
    floor : "surrogate" (default) estimates the noise floor from the record's
        own residuals; "none" subtracts nothing; a number is used as the level.
    floor_reps : number of surrogate records for the floor.
    oof : "interleaved" (default, as in the paper) or "purged" out-of-fold trend.
    purge : neighbours excluded on each side when oof="purged".
    rng : seed or numpy Generator used by the surrogate floor.
    min_length : records shorter than this are rejected.

    Returns
    -------
    InformationProfile, or None if the record is too short, constant, or holds
    no information above the noise floor.
    """
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    n = len(x)
    if n < min_length:
        return None
    scale_bandwidth = bandwidth if scale_bandwidth is None else scale_bandwidth
    D = _core.standardise(x)
    if D is None:
        return None
    w = _core.window_length(n, bandwidth)
    if w < 7:
        return None
    dt = _core.derivative(D, w)
    if oof == "interleaved":
        trend = _core.oof_trend_interleaved(D, w // 2)
    elif oof == "purged":
        trend = _core.oof_trend_purged(D, w // 2, purge)
    else:
        raise ValueError("oof must be 'interleaved' or 'purged'")
    res = D - trend
    sl = np.clip(_core.local_scale(res, max(5, int(scale_bandwidth * n))), 1e-12, None)
    dens = (dt / sl) ** 2

    if isinstance(floor, str) and floor == "surrogate":
        level = floor_level(res, n, _as_rng(rng), floor_reps, bandwidth, scale_bandwidth)
    elif isinstance(floor, str) and floor == "none":
        level = 0.0
    elif isinstance(floor, (int, float)):
        level = float(floor)
    else:
        raise ValueError("floor must be 'surrogate', 'none' or a number")
    corrected = np.clip(dens - level, 0.0, None)
    if corrected.sum() <= 0:
        return None
    return InformationProfile(density=dens, floor=level, corrected=corrected,
                              residual=res, scale=sl)


def information_age(x, q=0.35, **kwargs):
    """Shortcut: information age tau_q of one record (NaN if not computable)."""
    p = information_profile(x, **kwargs)
    return p.age(q) if p is not None else float("nan")
