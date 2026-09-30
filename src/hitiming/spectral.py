"""Health indicators computed from band-power spectra.

Each function takes P, an array of shape (n_inspections, n_bands) of band
powers, and f, the band centre frequencies, and returns one value per
inspection. Distributional indicators normalise every spectrum by its total
power before summarising it; amount indicators use the absolute power.
"""
import numpy as np

EPS = 1e-30


def _norm(P):
    return P / np.clip(P.sum(axis=1, keepdims=True), EPS, None)


def spectral_entropy(P, f=None):
    p = _norm(P)
    return -(p * np.log(np.clip(p, EPS, None))).sum(axis=1)


def spectral_centroid(P, f):
    return (_norm(P) * f).sum(axis=1)


def spectral_spread(P, f):
    p = _norm(P)
    c = (p * f).sum(axis=1, keepdims=True)
    return np.sqrt((p * (f - c) ** 2).sum(axis=1))


def spectral_gini(P, f=None):
    p = np.sort(_norm(P), axis=1)
    m = p.shape[1]
    return (2 * (p * np.arange(1, m + 1)).sum(axis=1)) \
        / np.clip(p.sum(axis=1), EPS, None) / m - (m + 1) / m


def top_band_share(P, f=None, k=8):
    return np.sort(_norm(P), axis=1)[:, -k:].sum(axis=1)


def high_low_ratio(P, f=None, k=8):
    return P[:, -k:].sum(axis=1) / np.clip(P[:, :k].sum(axis=1), EPS, None)


def total_power(P, f=None):
    return P.sum(axis=1)


def peak_band_power(P, f=None):
    return P.max(axis=1)


def high_band_power(P, f=None, k=8):
    return P[:, -k:].sum(axis=1)


def log_total_power(P, f=None):
    return np.log(np.clip(P.sum(axis=1), EPS, None))


# name -> (family, function); the ten indicators of the paper's Table 5
INDICATORS = {
    "spectral entropy": ("distribution", spectral_entropy),
    "spectral centroid": ("distribution", spectral_centroid),
    "spectral spread": ("distribution", spectral_spread),
    "spectral Gini": ("distribution", spectral_gini),
    "top-8-band share": ("distribution", top_band_share),
    "high/low band ratio": ("distribution", high_low_ratio),
    "total power": ("amount", total_power),
    "peak band power": ("amount", peak_band_power),
    "high-band power": ("amount", high_band_power),
    "log total power": ("amount", log_total_power),
}


def spectral_indicators(P, f):
    """All ten indicators for one record, as a dict name -> 1-D array."""
    P = np.asarray(P, float)
    f = np.asarray(f, float)
    return {name: fn(P, f) for name, (_, fn) in INDICATORS.items()}


def families():
    """Indicator names grouped as {'distribution': [...], 'amount': [...]}."""
    out = {}
    for name, (fam, _) in INDICATORS.items():
        out.setdefault(fam, []).append(name)
    return out
