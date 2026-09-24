"""
Stage 3 -- Does the windowed Cramer-Rao floor of Proposition 1 actually govern
sliding-window age estimation on real accelerometer data?

The test is deliberately *non-parametric in the degradation class*. Instead of
assuming exponential / power-law / stretched-exponential, we take the empirical
degradation profile of each bearing and read its Fisher-information density
straight off the data:

    Dhat(tau)   -- Savitzky-Golay trend of the normalized indicator
    Dhat'(tau)  -- its analytic derivative from the same filter
    sigma_m     -- robust scale of the residual X - Dhat
    Gamma_w(tau0) = sigma_m^-2 * sum_{tau_k in [tau0-w, tau0]} Dhat'(tau_k)^2
    rho_w(tau0)   = Gamma_w(tau0) / Gamma_1(1)

This makes three predictions testable without any class assumption:

  T1  CRB attainment.  The windowed ML estimator's age-error variance should
      track Gamma_w(tau0)^-1.
  T2  Window scaling.  At fixed tau0, RMSE(w) should fall as rho_w(tau0)^-1/2.
  T3  Infeasible band.  For a target accuracy eps, ages with
      rho_max(tau0) := rho_{w=tau0}(tau0) < q* are unreachable by ANY window.

Protocol per trial: a unit is truly at normalized age tau0; the predictor holds
a prior belief tau0_tilde = tau0 - delta_true with delta_true drawn uniformly on
[-DELTA_MAX, DELTA_MAX]; it observes the real indicator samples in the trailing
window and applies the linear-Gaussian ML correction of Proposition 1. The
reported error is the residual age error after that correction.
"""
import os
import json
import numpy as np
from scipy.signal import savgol_filter

HERE = os.path.dirname(os.path.abspath(__file__))
FEAT = os.path.join(HERE, "feat_raw.npz")
RAW = os.path.join(HERE, "hi_raw.npz")

SNAPSHOT_PERIOD_S = 10.0
SMOOTH_FRAC = 0.08          # Savitzky-Golay window as a fraction of life
DELTA_MAX = 0.05            # prior age error, +-5% of life
N_DRAWS = 24                # prior draws per (bearing, tau0, w) cell
RNG = np.random.default_rng(20260809)

TAU0_GRID = np.round(np.arange(0.30, 0.96, 0.05), 3)
W_GRID = np.array([0.05, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50, 0.60])


# ---------------------------------------------------------------- indicators
def build_indicator(M, featnames, kind):
    """Return the scalar indicator series for one bearing."""
    idx = {n: i for i, n in enumerate(featnames)}
    if kind == "rms":
        return M[:, idx["rms"]]
    if kind == "hfband":
        # high-frequency resonance energy: bearing defects excite these first
        return M[:, idx["b4_6"]] + M[:, idx["b6_8"]] + M[:, idx["b8_10"]]
    if kind == "hfratio":
        hf = M[:, idx["b4_6"]] + M[:, idx["b6_8"]] + M[:, idx["b8_10"]]
        lf = M[:, idx["b0_1"]] + M[:, idx["b1_2"]]
        return hf / np.maximum(lf, 1e-12)
    if kind == "rms_hf":
        hf = M[:, idx["b4_6"]] + M[:, idx["b6_8"]] + M[:, idx["b8_10"]]
        return M[:, idx["rms"]] * np.sqrt(np.maximum(hf, 0.0))
    raise ValueError(kind)


def normalize(x):
    n = len(x)
    base = np.median(x[:max(3, int(0.05 * n))])
    end = np.median(x[-3:])
    if not np.isfinite(base) or not np.isfinite(end) or abs(end - base) < 1e-12:
        return None
    return (x - base) / (end - base)


def trend_and_noise(D):
    """Savitzky-Golay trend, its derivative in normalized time, and noise scale."""
    n = len(D)
    win = max(7, int(SMOOTH_FRAC * n))
    win = win + 1 if win % 2 == 0 else win
    win = min(win, n - 1 if (n - 1) % 2 == 1 else n - 2)
    if win < 7:
        return None
    trend = savgol_filter(D, win, 2)
    # derivative w.r.t. normalized time: delta_tau between samples is 1/n
    dtrend = savgol_filter(D, win, 2, deriv=1, delta=1.0 / n)
    resid = D - trend
    sigma_m = 1.4826 * np.median(np.abs(resid - np.median(resid)))
    return trend, dtrend, float(max(sigma_m, 1e-9))


# ---------------------------------------------------------------- experiment
def run_bearing(D, trend, dtrend, sigma_m):
    """Yield per-cell results for one bearing."""
    n = len(D)
    tau = np.arange(1, n + 1) / n
    fisher_density = dtrend ** 2              # up to the common f_s/sigma_m^2
    total_info = float(np.sum(fisher_density))
    if total_info <= 0:
        return []

    out = []
    for tau0 in TAU0_GRID:
        k0 = int(round(tau0 * n)) - 1
        if k0 < 5 or k0 >= n:
            continue
        # rho_max: fraction of total Fisher information available in [0, tau0]
        rho_max = float(np.sum(fisher_density[:k0 + 1]) / total_info)
        for w in W_GRID:
            if w > tau0:
                continue
            klo = max(0, int(round((tau0 - w) * n)) - 1)
            if k0 - klo < 5:
                continue
            sl = slice(klo, k0 + 1)
            info_w = float(np.sum(fisher_density[sl]))
            if info_w <= 0:
                continue
            rho_w = info_w / total_info
            gamma_w = info_w / sigma_m ** 2            # = Fisher info for delta
            crb_sd = 1.0 / np.sqrt(gamma_w)            # predicted sd of age error

            # --- run the actual windowed ML estimator on the real samples ---
            errs = []
            for _ in range(N_DRAWS):
                d_true = RNG.uniform(-DELTA_MAX, DELTA_MAX)
                shift = int(round(d_true * n))
                jlo, jhi = klo - shift, k0 - shift     # indices of the believed positions
                if jlo < 0 or jhi >= n or jhi - jlo < 5:
                    continue
                g = dtrend[jlo:jhi + 1]                # regressor at believed positions
                denom = float(np.sum(g ** 2))
                if denom <= 0:
                    continue
                y = D[sl] - trend[jlo:jhi + 1]         # observed minus believed trend
                d_hat = float(np.sum(g * y) / denom)
                errs.append(d_hat - d_true)
            if len(errs) < N_DRAWS // 2:
                continue
            errs = np.asarray(errs)
            out.append(dict(tau0=float(tau0), w=float(w), rho_w=rho_w,
                            rho_max=rho_max, crb_sd=float(crb_sd),
                            emp_rmse=float(np.sqrt(np.mean(errs ** 2))),
                            emp_sd=float(np.std(errs, ddof=1)),
                            n_eff=len(errs)))
    return out


def main():
    fz = np.load(FEAT, allow_pickle=True)
    featnames = [str(s) for s in fz["featnames"]]
    bearings = sorted(k for k in fz.files if k != "featnames")

    results = {}
    for kind in ["rms", "hfband", "hfratio", "rms_hf"]:
        rows, per_bearing = [], {}
        for b in bearings:
            D = normalize(build_indicator(fz[b], featnames, kind))
            if D is None:
                continue
            tn = trend_and_noise(D)
            if tn is None:
                continue
            trend, dtrend, sigma_m = tn
            r = run_bearing(D, trend, dtrend, sigma_m)
            for d in r:
                d["bearing"] = b
            rows.extend(r)
            per_bearing[b] = dict(sigma_m=sigma_m,
                                  n=len(D),
                                  info_total=float(np.sum(dtrend ** 2)))
        results[kind] = dict(rows=rows, per_bearing=per_bearing)

        # ---- headline numbers for this indicator ----
        if not rows:
            print(f"{kind}: no usable cells")
            continue
        ratio = np.array([d["emp_rmse"] / d["crb_sd"] for d in rows])
        lr = np.log(np.array([d["emp_rmse"] for d in rows]))
        lp = np.log(np.array([d["crb_sd"] for d in rows]))
        slope = float(np.polyfit(lp, lr, 1)[0])
        corr = float(np.corrcoef(lp, lr)[0, 1])
        print(f"\n=== indicator: {kind} ===")
        print(f"  cells: {len(rows)} over {len(per_bearing)} bearings")
        print(f"  empirical RMSE / predicted CRB sd:  "
              f"median {np.median(ratio):.2f}, IQR [{np.percentile(ratio,25):.2f}, "
              f"{np.percentile(ratio,75):.2f}]")
        print(f"  log-log fit of empirical vs predicted: slope {slope:.3f}, r = {corr:.3f}")
        print(f"    (Proposition 1 predicts slope 1; a constant offset is the")
        print(f"     inefficiency of the real, non-Gaussian residual)")

    with open(os.path.join(HERE, "crb_results.json"), "w") as fh:
        json.dump({k: dict(rows=v["rows"], per_bearing=v["per_bearing"])
                   for k, v in results.items()}, fh)
    print(f"\nwrote crb_results.json")


if __name__ == "__main__":
    main()
