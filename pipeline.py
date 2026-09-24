"""
Shared analysis pipeline.

Both domains -- vibration and battery -- must run through byte-identical
machinery, otherwise a cross-domain claim about the exponent is really a claim
about two different processing chains. Everything domain-specific stops at the
point where a raw record becomes a normalized indicator series; from there on
this module is the only code that touches it.
"""
import os
import numpy as np
from scipy.signal import savgol_filter
from scipy.optimize import minimize_scalar

SMOOTH_FRAC = 0.08      # Savitzky-Golay window, fraction of life
LOCAL_BW = 0.08         # bandwidth of the rolling noise scale
K_FOLDS = 10            # interleaved folds for the out-of-fold residual
OOF_PURGE = 3           # neighbours purged either side, in the purged control
OOF_SCHEME = os.environ.get("OOF_SCHEME", "interleaved")
FALLBACKS = [0, 0]      # purged samples left unfitted, samples seen
SIGMA_MAX = 0.5         # screen: noise must be small vs the [0,1] range


def robust_scale(x):
    x = np.asarray(x, float)
    return float(1.4826 * np.median(np.abs(x - np.median(x))))


def normalize(x, baseline_frac=0.05):
    """Map a raw indicator onto the D(0)=0, D(1)=1 convention."""
    x = np.asarray(x, float)
    n = len(x)
    base = np.median(x[:max(3, int(baseline_frac * n))])
    end = np.median(x[-3:])
    if not np.isfinite(base) or not np.isfinite(end) or abs(end - base) < 1e-12:
        return None
    return (x - base) / (end - base)


def _win(n):
    w = max(7, int(SMOOTH_FRAC * n))
    w += (w % 2 == 0)
    return min(w, n - 1 if (n - 1) % 2 == 1 else n - 2)


def oof_trend_interleaved(D, half):
    """Interleaved K-fold local-quadratic trend; the fit at i never uses i.

    It holds i out but keeps i-1 and i+1, which on serially correlated
    residuals partly sees the held-out sample: the scale is understated by up to
    4% at 240 samples and phi = 0.45 (oof_leak.py), while a purge overstates it
    by more on the same records.  The purged scheme is the control.
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
            pred[i] = D[i] if len(xs) < 6 else np.polyfit(xs, D[lo:hi][m], 2)[-1]
    return pred


def oof_trend_purged(D, half, m=None):
    """h-block local-quadratic trend: the fit at i drops i and its m neighbours
    either side.  The window is widened by m on each side, so the fit keeps the
    point count of the unpurged window; without that, a short record's window
    (half-width 4 at 120 samples) has nothing left to fit once 7 samples go.
    """
    m = OOF_PURGE if m is None else m
    n = len(D)
    idx = np.arange(n)
    pred = np.empty(n)
    h = half + m
    for i in range(n):
        lo, hi = max(0, i - h), min(n, i + h + 1)
        xs = idx[lo:hi] - i
        keep = np.abs(xs) > m
        if keep.sum() < 6:
            pred[i] = D[i]
            FALLBACKS[0] += 1
        else:
            pred[i] = np.polyfit(xs[keep].astype(float), D[lo:hi][keep], 2)[-1]
        FALLBACKS[1] += 1
    return pred


def oof_trend(D, half):
    """The out-of-fold trend the pipeline uses: interleaved unless OOF_SCHEME says."""
    if OOF_SCHEME == "purged":
        return oof_trend_purged(D, half)
    return oof_trend_interleaved(D, half)


def local_scale(res, half):
    n = len(res)
    out = np.empty(n)
    for i in range(n):
        lo, hi = max(0, i - half), min(n, i + half + 1)
        out[i] = robust_scale(res[lo:hi])
    return np.maximum(out, 1e-9)


def prepare(raw, name=""):
    """Raw indicator -> everything the analysis needs, or (None, reason)."""
    D = normalize(raw)
    if D is None:
        return None, "degenerate normalization"
    n = len(D)
    w = _win(n)
    if w < 7:
        return None, "series too short"
    dt = savgol_filter(D, w, 2, deriv=1, delta=1.0 / n)
    sig_in = robust_scale(D - savgol_filter(D, w, 2))
    if not np.isfinite(sig_in) or sig_in > SIGMA_MAX:
        return None, f"no net degradation (sigma={sig_in:.2f})"
    tr = oof_trend(D, w // 2)
    res = D - tr
    dens = dt ** 2
    if dens.sum() <= 0:
        return None, "no information"
    return dict(name=name, n=n, D=D, trend=tr, dtrend=dt, res=res,
                sig_in=sig_in, sig_oof=robust_scale(res),
                sig_loc=local_scale(res, max(5, int(LOCAL_BW * n))),
                dens=dens / dens.sum(), F=np.cumsum(dens) / dens.sum()), None


def tau_min(u, qstar):
    """Finv(q*) read off the empirical information distribution."""
    k = int(np.searchsorted(u["F"], qstar))
    return (k + 1) / u["n"] if k < u["n"] else 1.0


def beta_eff(u):
    """Power-law exponent whose cumulative profile best matches the observed F."""
    tau = np.arange(1, u["n"] + 1) / u["n"]
    f = lambda b: float(np.mean((u["F"] - tau ** (2 * b - 1)) ** 2))
    return float(minimize_scalar(f, bounds=(0.51, 30.0), method="bounded",
                                 options={"xatol": 1e-3}).x)


def profile(raw, name=""):
    """Light path: information profile only, no out-of-fold trend.

    beta_eff / tau_min depend on Dhat' alone, so the K-fold refit -- the
    expensive part -- is skipped here and run only where residuals matter.
    """
    D = normalize(raw)
    if D is None:
        return None, "degenerate normalization"
    n = len(D)
    w = _win(n)
    if w < 7:
        return None, "series too short"
    dt = savgol_filter(D, w, 2, deriv=1, delta=1.0 / n)
    sig = robust_scale(D - savgol_filter(D, w, 2))
    if not np.isfinite(sig) or sig > SIGMA_MAX:
        return None, f"no net degradation (sigma={sig:.2f})"
    dens = dt ** 2
    if dens.sum() <= 0:
        return None, "no information"
    return dict(name=name, n=n, D=D, dtrend=dt, sig_in=sig,
                dens=dens / dens.sum(), F=np.cumsum(dens) / dens.sum()), None
