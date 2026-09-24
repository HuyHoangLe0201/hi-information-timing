"""
Stage 4 -- final validation, in two clean parts.

The earlier pooled design conflated two things: whether the Cramer-Rao floor is
attained, and how tight the prior happened to be. They are separated here.

PART A -- is the floor attained under real sensor noise?
    Build the observation exactly as Proposition 1 models it, but drive it with
    the REAL residual sequence measured on each bearing:
        Y_k = Dhat'(tau_k) * delta_true + eps_k^real
    and apply the ML estimator. There is no linearization error by construction,
    so any departure from Var(delta_hat) = Gamma_w^-1 is attributable to the real
    noise failing to be iid Gaussian (heavy tails, heteroscedasticity). This is
    the actual test of the bound.

PART B -- how far does the linearization reach?
    Re-run with the true nonlinear degradation profile and a genuine prior offset
    delta_true, and locate the offset at which the neglected (1/2)D'' delta^2 term
    overtakes the floor. That offset is the validity radius of the bound and is
    the practical statement: re-anchor the prior more often than this.

PART C -- the infeasible band, non-parametrically.
    rho_max(tau0) = fraction of total Fisher information available in [0, tau0].
    For target q*, ages with rho_max(tau0) < q* are unreachable by ANY window.

Bearings whose indicator shows no net degradation are screened out and reported.
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
SNAPSHOT_PERIOD_S = 10.0
SIGMA_MAX = 0.5          # screen: noise must be small vs the [0,1] indicator range
TAU0_GRID = np.round(np.arange(0.30, 0.96, 0.05), 3)
W_GRID = np.array([0.05, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50, 0.60])
N_MC = 400
RNG = np.random.default_rng(20260809)


def indicator(M):
    i = {n: k for k, n in enumerate(FEATNAMES)}
    return M[:, i["b4_6"]] + M[:, i["b6_8"]] + M[:, i["b8_10"]]


def prep(b):
    x = indicator(fz[b])
    n = len(x)
    base = np.median(x[:max(3, int(0.05 * n))])
    end = np.median(x[-3:])
    if not np.isfinite(base) or not np.isfinite(end) or abs(end - base) < 1e-12:
        return None, "degenerate normalization"
    D = (x - base) / (end - base)
    win = max(7, int(SMOOTH_FRAC * n))
    win = win + 1 if win % 2 == 0 else win
    win = min(win, n - 1 if (n - 1) % 2 == 1 else n - 2)
    if win < 7:
        return None, "series too short"
    trend = savgol_filter(D, win, 2)
    dtrend = savgol_filter(D, win, 2, deriv=1, delta=1.0 / n)
    resid = D - trend
    sigma = float(1.4826 * np.median(np.abs(resid - np.median(resid))))
    if not np.isfinite(sigma) or sigma > SIGMA_MAX:
        return None, f"no net degradation (sigma_m={sigma:.2f} vs unit range)"
    return dict(D=D, trend=trend, dtrend=dtrend, resid=resid,
                sigma=max(sigma, 1e-9), n=n, life_s=n * SNAPSHOT_PERIOD_S), None


# ------------------------------------------------------------------ screening
units, dropped = {}, {}
for b in BEARINGS:
    u, why = prep(b)
    if u is None:
        dropped[b] = why
    else:
        units[b] = u

print(f"usable bearings: {len(units)} of {len(BEARINGS)}")
for b, why in dropped.items():
    print(f"  dropped {b}: {why}")

# excess kurtosis of the real residual: how far from Gaussian is the noise?
kurts = []
for b, u in units.items():
    r = u["resid"] / u["sigma"]
    kurts.append(float(np.mean(r ** 4) - 3.0))
print(f"\nresidual excess kurtosis: median {np.median(kurts):.1f} "
      f"(Gaussian = 0), range [{min(kurts):.1f}, {max(kurts):.1f}]")


# ------------------------------------------------------------------- PART A
print("\n=== PART A: is the floor attained under real sensor noise? ===")
rowsA = []
for b, u in units.items():
    n, dt, sig, res = u["n"], u["dtrend"], u["sigma"], u["resid"]
    for tau0 in TAU0_GRID:
        k0 = int(round(tau0 * n)) - 1
        if k0 < 5 or k0 >= n:
            continue
        for w in W_GRID:
            if w > tau0:
                continue
            klo = max(0, int(round((tau0 - w) * n)) - 1)
            if k0 - klo < 10:
                continue
            g = dt[klo:k0 + 1]
            den = float(np.sum(g ** 2))
            if den <= 0:
                continue
            crb_sd = sig / np.sqrt(den)
            m = len(g)
            errs = np.empty(N_MC)
            for t in range(N_MC):
                # real residual block, randomly relocated within the trajectory
                s = RNG.integers(0, n - m)
                eps = res[s:s + m]
                errs[t] = float(np.sum(g * eps) / den)   # delta_hat - delta_true
            rowsA.append(dict(bearing=b, tau0=float(tau0), w=float(w),
                              crb_sd=float(crb_sd),
                              emp_sd=float(np.std(errs, ddof=1))))

ratA = np.array([r["emp_sd"] / r["crb_sd"] for r in rowsA])
lp = np.log([r["crb_sd"] for r in rowsA])
lr = np.log([r["emp_sd"] for r in rowsA])
slopeA = float(np.polyfit(lp, lr, 1)[0])
corrA = float(np.corrcoef(lp, lr)[0, 1])
print(f"  cells: {len(rowsA)}")
print(f"  empirical sd / CRB sd: median {np.median(ratA):.3f}, "
      f"IQR [{np.percentile(ratA,25):.3f}, {np.percentile(ratA,75):.3f}]")
print(f"  log-log slope {slopeA:.3f} (predicted 1), r = {corrA:.4f}")


# ------------------------------------------------------------------- PART B
print("\n=== PART B: validity radius of the linearization ===")
DELTAS = [0.005, 0.01, 0.02, 0.03, 0.05, 0.08, 0.12]
rowsB = []
print(f"{'|delta|':>9}{'median emp/CRB':>17}{'slope':>9}{'r':>8}")
print("-" * 43)
for dm in DELTAS:
    lps, lrs = [], []
    for b, u in units.items():
        n, D, tr, dt, sig = u["n"], u["D"], u["trend"], u["dtrend"], u["sigma"]
        for tau0 in TAU0_GRID:
            k0 = int(round(tau0 * n)) - 1
            if k0 < 5 or k0 >= n:
                continue
            for w in W_GRID:
                if w > tau0:
                    continue
                klo = max(0, int(round((tau0 - w) * n)) - 1)
                if k0 - klo < 10:
                    continue
                den0 = float(np.sum(dt[klo:k0 + 1] ** 2))
                if den0 <= 0:
                    continue
                crb_sd = sig / np.sqrt(den0)
                errs = []
                for sgn in (-1, 1):
                    sh = int(round(sgn * dm * n))
                    jlo, jhi = klo - sh, k0 - sh
                    if jlo < 0 or jhi >= n or jhi - jlo != k0 - klo:
                        continue
                    g = dt[jlo:jhi + 1]
                    den = float(np.sum(g ** 2))
                    if den <= 0:
                        continue
                    y = D[klo:k0 + 1] - tr[jlo:jhi + 1]
                    errs.append(float(np.sum(g * y) / den) - sgn * dm)
                if len(errs) < 2:
                    continue
                lps.append(np.log(crb_sd))
                lrs.append(np.log(np.sqrt(np.mean(np.asarray(errs) ** 2))))
    lps, lrs = np.asarray(lps), np.asarray(lrs)
    sl = float(np.polyfit(lps, lrs, 1)[0])
    co = float(np.corrcoef(lps, lrs)[0, 1])
    md = float(np.median(np.exp(lrs - lps)))
    rowsB.append(dict(delta=dm, ratio=md, slope=sl, corr=co))
    print(f"{dm:>9.3f}{md:>17.2f}{sl:>9.3f}{co:>8.3f}")


# ------------------------------------------------------------------- PART C
print("\n=== PART C: infeasible band (non-parametric) ===")
qstars = [0.10, 0.20, 0.35, 0.50]
print(f"{'q*':>7}{'median tau_min':>17}{'IQR':>22}{'% of life unreachable':>24}")
print("-" * 70)
bandC = {}
for q in qstars:
    tmins = []
    for b, u in units.items():
        dens = u["dtrend"] ** 2
        cum = np.cumsum(dens) / np.sum(dens)
        idx = np.searchsorted(cum, q)
        tmins.append((idx + 1) / u["n"] if idx < u["n"] else 1.0)
    tmins = np.asarray(tmins)
    bandC[q] = dict(median=float(np.median(tmins)),
                    q25=float(np.percentile(tmins, 25)),
                    q75=float(np.percentile(tmins, 75)),
                    values={b: float(t) for b, t in zip(units, tmins)})
    print(f"{q:>7.2f}{np.median(tmins):>17.3f}"
          f"{'[%.3f, %.3f]' % (np.percentile(tmins,25), np.percentile(tmins,75)):>22}"
          f"{100*np.median(tmins):>23.1f}%")

# where does the information actually sit?
print("\ninformation concentration (fraction of total Fisher info before tau):")
print(f"{'tau':>7}" + "".join(f"{t:>9.1f}" for t in [0.5, 0.7, 0.8, 0.9, 0.95]))
fracs = {t: [] for t in [0.5, 0.7, 0.8, 0.9, 0.95]}
for b, u in units.items():
    dens = u["dtrend"] ** 2
    tot = np.sum(dens)
    for t in fracs:
        k = int(round(t * u["n"]))
        fracs[t].append(float(np.sum(dens[:k]) / tot))
print(f"{'median':>7}" + "".join(f"{np.median(fracs[t]):>9.3f}" for t in [0.5, 0.7, 0.8, 0.9, 0.95]))

life = np.array([u["life_s"] for u in units.values()])
out = dict(
    n_units=len(units), dropped=dropped,
    resid_excess_kurtosis_median=float(np.median(kurts)),
    partA=dict(median_ratio=float(np.median(ratA)),
               iqr=[float(np.percentile(ratA, 25)), float(np.percentile(ratA, 75))],
               slope=slopeA, corr=corrA, cells=len(rowsA)),
    partB=rowsB,
    partC={str(k): v for k, v in bandC.items()},
    info_fraction_median={str(t): float(np.median(fracs[t])) for t in fracs},
    Tbar_s=float(life.mean()),
    fs_median=float(np.median([u["n"] for u in units.values()])),
    sigma_m_median=float(np.median([u["sigma"] for u in units.values()])),
)
with open(os.path.join(HERE, "validate.json"), "w") as fh:
    json.dump(out, fh, indent=2)
print(f"\nTbar = {life.mean():.0f} s ({life.mean()/60:.1f} min); wrote validate.json")
