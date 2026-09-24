"""
Stage 2 -- Calibrate the power-law degradation class on PRONOSTIA.

For each run-to-failure bearing we
  1. normalize the RMS health indicator so D(0)=0, D(1)=1;
  2. fit the power-law shape D(tau) = tau^beta by least squares;
  3. estimate the measurement-noise scale sigma_m as the residual scale about
     that fit -- the same quantity the sampled observation model of the Letter
     calls sigma_m, i.e. everything the power-law trend does not explain;
  4. form Gamma = (f_s/sigma_m^2) * int_0^1 D'^2 dtau = (f_s/sigma_m^2)*beta^2/(2beta-1),
     with f_s = number of health-indicator samples per lifetime.

Output: calib.npz + a printed table.
"""
import os
import json
import numpy as np
from scipy.optimize import minimize_scalar

HERE = os.path.dirname(os.path.abspath(__file__))
RAW = os.path.join(HERE, "hi_raw.npz")
OUT = os.path.join(HERE, "calib.npz")

SNAPSHOT_PERIOD_S = 10.0
BASELINE_FRAC = 0.05      # first 5% of life defines the healthy baseline
CHANNEL = "h"             # horizontal accelerometer (primary PRONOSTIA channel)


def normalize_hi(r):
    """Map a raw RMS series onto the D(0)=0, D(1)=1 convention."""
    n = len(r)
    base = np.median(r[:max(3, int(BASELINE_FRAC * n))])
    end = np.median(r[-3:])
    if end <= base:
        return None
    return (r - base) / (end - base)


def fit_beta(tau, D):
    """Least-squares fit of D(tau)=tau^beta; returns beta and residual scale."""
    def sse(b):
        return float(np.sum((D - tau ** b) ** 2))
    res = minimize_scalar(sse, bounds=(1.01, 40.0), method="bounded",
                          options={"xatol": 1e-4})
    beta = float(res.x)
    resid = D - tau ** beta
    # robust scale: MAD, so the end-of-life spike does not inflate the noise term
    sigma_m = float(1.4826 * np.median(np.abs(resid - np.median(resid))))
    return beta, sigma_m, resid


def main():
    z = np.load(RAW)
    bearings = sorted({k.split("__")[0] for k in z.files if not k.startswith("meta__")})

    rows = []
    store = {}
    for b in bearings:
        r = z[f"{b}__{CHANNEL}"]
        D = normalize_hi(r)
        if D is None:
            print(f"  skip {b}: no net degradation in the RMS channel")
            continue
        n = len(D)
        # tau_k = k/n for k=1..n so that tau in (0,1] and tau_n = 1
        tau = np.arange(1, n + 1) / n
        beta, sigma_m, resid = fit_beta(tau, D)
        f_s = float(n)                      # samples per lifetime
        life_s = n * SNAPSHOT_PERIOD_S
        # Gamma = (f_s/sigma_m^2) * int_0^1 (beta tau^(beta-1))^2 dtau
        gamma = (f_s / sigma_m ** 2) * beta ** 2 / (2 * beta - 1)
        rows.append(dict(bearing=b, n=n, life_s=life_s, beta=beta,
                         sigma_m=sigma_m, f_s=f_s, gamma=gamma))
        store[f"{b}__D"] = D
        store[f"{b}__tau"] = tau

    betas = np.array([r["beta"] for r in rows])
    lives = np.array([r["life_s"] for r in rows])
    gammas = np.array([r["gamma"] for r in rows])
    sigmas = np.array([r["sigma_m"] for r in rows])

    print(f"\n{'bearing':<14}{'N':>6}{'life(s)':>10}{'beta':>8}{'sigma_m':>10}{'Gamma':>12}")
    print("-" * 60)
    for r in rows:
        print(f"{r['bearing']:<14}{r['n']:>6}{r['life_s']:>10.0f}"
              f"{r['beta']:>8.2f}{r['sigma_m']:>10.4f}{r['gamma']:>12.3e}")
    print("-" * 60)
    print(f"{'fleet mean':<14}{np.mean([r['n'] for r in rows]):>6.0f}"
          f"{lives.mean():>10.0f}{betas.mean():>8.2f}{sigmas.mean():>10.4f}{gammas.mean():>12.3e}")
    print(f"{'fleet median':<14}{'':>6}{np.median(lives):>10.0f}"
          f"{np.median(betas):>8.2f}{np.median(sigmas):>10.4f}{np.median(gammas):>12.3e}")
    print(f"\nbeta: mean {betas.mean():.2f}, sd {betas.std(ddof=1):.2f}, "
          f"range [{betas.min():.2f}, {betas.max():.2f}]")
    print(f"Tbar (mean life): {lives.mean():.0f} s = {lives.mean()/60:.1f} min")

    summary = dict(
        n_bearings=len(rows),
        beta_mean=float(betas.mean()), beta_sd=float(betas.std(ddof=1)),
        beta_median=float(np.median(betas)),
        beta_min=float(betas.min()), beta_max=float(betas.max()),
        Tbar_s=float(lives.mean()),
        gamma_mean=float(gammas.mean()), gamma_median=float(np.median(gammas)),
        sigma_m_median=float(np.median(sigmas)),
        fs_median=float(np.median([r["f_s"] for r in rows])),
        rows=rows,
    )
    np.savez_compressed(OUT, **store,
                        summary=np.array([json.dumps(summary)], dtype=object))
    with open(os.path.join(HERE, "calib_summary.json"), "w") as fh:
        json.dump(summary, fh, indent=2)
    print(f"\nwrote {OUT} and calib_summary.json")


if __name__ == "__main__":
    main()
