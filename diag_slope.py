"""
Why is the empirical window-scaling shallower than the CRB predicts?

Two candidate explanations, separated here:

  (H1) Linearization radius.  Proposition 1 linearizes D about the believed age.
       The neglected term is (1/2) D''(tau) delta^2, which does NOT shrink as the
       window widens -- so once the CRB drops below that bias, the empirical error
       stops following it.  Prediction: shrinking the prior error DELTA_MAX should
       restore slope -> 1.

  (H2) Correlated residuals.  The CRB counts f_s*w independent samples per window.
       If the residual has correlation length L samples, the true information is
       ~L times smaller.  A constant L shifts the intercept but NOT the slope;
       an L that grows with w would flatten it.

We sweep DELTA_MAX and measure the residual autocorrelation length to tell them
apart.
"""
import os
import json
import numpy as np
from scipy.signal import savgol_filter

HERE = os.path.dirname(os.path.abspath(__file__))
fz = np.load(os.path.join(HERE, "feat_raw.npz"), allow_pickle=True)
FEATNAMES = [str(s) for s in fz["featnames"]]
BEARINGS = sorted(k for k in fz.files if k != "featnames")

SMOOTH_FRAC = 0.08
TAU0_GRID = np.round(np.arange(0.30, 0.96, 0.05), 3)
W_GRID = np.array([0.05, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50, 0.60])
N_DRAWS = 40


def indicator(M):
    i = {n: k for k, n in enumerate(FEATNAMES)}
    return M[:, i["b4_6"]] + M[:, i["b6_8"]] + M[:, i["b8_10"]]


def normalize(x):
    n = len(x)
    base = np.median(x[:max(3, int(0.05 * n))])
    end = np.median(x[-3:])
    if not np.isfinite(base) or not np.isfinite(end) or abs(end - base) < 1e-12:
        return None
    return (x - base) / (end - base)


def prep(D):
    n = len(D)
    win = max(7, int(SMOOTH_FRAC * n))
    win = win + 1 if win % 2 == 0 else win
    win = min(win, n - 1 if (n - 1) % 2 == 1 else n - 2)
    if win < 7:
        return None
    trend = savgol_filter(D, win, 2)
    dtrend = savgol_filter(D, win, 2, deriv=1, delta=1.0 / n)
    resid = D - trend
    sigma = 1.4826 * np.median(np.abs(resid - np.median(resid)))
    return trend, dtrend, float(max(sigma, 1e-9)), resid


def acorr_length(r):
    """Lag at which the residual autocorrelation first drops below 1/e."""
    r = r - r.mean()
    n = len(r)
    ac = np.correlate(r, r, mode="full")[n - 1:]
    ac /= ac[0]
    below = np.where(ac < np.exp(-1.0))[0]
    return int(below[0]) if len(below) else n


# ---------------------------------------------------------------- H2 first
print("=== residual autocorrelation (H2) ===")
print(f"{'bearing':<13}{'N':>7}{'L (samples)':>14}{'L/N':>9}{'sigma_m':>10}")
print("-" * 53)
Ls = []
for b in BEARINGS:
    D = normalize(indicator(fz[b]))
    if D is None:
        continue
    p = prep(D)
    if p is None:
        continue
    trend, dtrend, sigma, resid = p
    L = acorr_length(resid)
    Ls.append(L)
    print(f"{b:<13}{len(D):>7}{L:>14}{L/len(D):>9.4f}{sigma:>10.4f}")
print("-" * 53)
print(f"median correlation length: {np.median(Ls):.0f} samples")
print("A constant L rescales Gamma_w by ~1/L: it moves the intercept, not the slope.\n")


# ---------------------------------------------------------------- H1 sweep
def run(delta_max, rng):
    lp, lr = [], []
    for b in BEARINGS:
        D = normalize(indicator(fz[b]))
        if D is None:
            continue
        p = prep(D)
        if p is None:
            continue
        trend, dtrend, sigma, _ = p
        n = len(D)
        dens = dtrend ** 2
        for tau0 in TAU0_GRID:
            k0 = int(round(tau0 * n)) - 1
            if k0 < 5 or k0 >= n:
                continue
            for w in W_GRID:
                if w > tau0:
                    continue
                klo = max(0, int(round((tau0 - w) * n)) - 1)
                if k0 - klo < 5:
                    continue
                info = float(np.sum(dens[klo:k0 + 1]))
                if info <= 0:
                    continue
                crb = sigma / np.sqrt(info)
                errs = []
                for _ in range(N_DRAWS):
                    dt = rng.uniform(-delta_max, delta_max)
                    sh = int(round(dt * n))
                    jlo, jhi = klo - sh, k0 - sh
                    if jlo < 0 or jhi >= n or jhi - jlo < 5:
                        continue
                    g = dtrend[jlo:jhi + 1]
                    den = float(np.sum(g ** 2))
                    if den <= 0:
                        continue
                    y = D[klo:k0 + 1] - trend[jlo:jhi + 1]
                    errs.append(float(np.sum(g * y) / den) - dt)
                if len(errs) < N_DRAWS // 2:
                    continue
                lp.append(np.log(crb))
                lr.append(np.log(np.sqrt(np.mean(np.asarray(errs) ** 2))))
    lp, lr = np.asarray(lp), np.asarray(lr)
    slope = float(np.polyfit(lp, lr, 1)[0])
    corr = float(np.corrcoef(lp, lr)[0, 1])
    ratio = np.exp(lr - lp)
    return slope, corr, float(np.median(ratio)), len(lp)


print("=== linearization radius sweep (H1) ===")
print(f"{'delta_max':>11}{'slope':>9}{'r':>8}{'median ratio':>15}{'cells':>8}")
print("-" * 51)
out = {}
for dm in [0.005, 0.01, 0.02, 0.03, 0.05, 0.08]:
    rng = np.random.default_rng(20260809)
    s, c, m, k = run(dm, rng)
    out[dm] = dict(slope=s, corr=c, median_ratio=m, cells=k)
    print(f"{dm:>11.3f}{s:>9.3f}{c:>8.3f}{m:>15.2f}{k:>8}")
print("-" * 51)
print("Proposition 1 predicts slope 1. If the slope climbs toward 1 as the prior")
print("error shrinks, the shortfall is the linearization bias, not the bound.")

with open(os.path.join(HERE, "diag_slope.json"), "w") as fh:
    json.dump({"acorr_median": float(np.median(Ls)), "sweep": out}, fh, indent=2)
