"""
Stage 4b -- Part A redone with out-of-fold residuals, and a diagnosis of why
the in-sample version sat below the nominal floor.

The first pass used the Savitzky-Golay residual D - trend, with the trend fitted
on the same points. Two things could push the measured spread below the floor:

  (C1) In-sample collinearity. The smoother removes the residual component
       explained by local quadratics, and the regressor Dhat' is smooth, so
       sum(g*eps) is systematically shrunk. Fix: out-of-fold residuals.

  (C2) Heteroscedasticity. sigma_m was a single global robust scale, but the
       accelerometer residual is far larger near end of life -- exactly where
       Dhat' is largest. A window sitting in a quiet stretch then beats a floor
       computed from the global scale. Fix: a local sigma_m(tau), which is also
       the honest reading of (3): Gamma_w = sum_k Dhat'(tau_k)^2 / sigma_m(tau_k)^2.

Three configurations are compared so the mechanism is identifiable:
   A0  in-sample residual, global sigma      (the original)
   A1  out-of-fold residual, global sigma    (isolates C1)
   A2  out-of-fold residual, local sigma     (adds C2 -- the principled version)

Monte Carlo draws iid residuals, which is exactly what model (1) asserts and
what the measured autocorrelation length of one sample supports.
"""
import os
import json
import numpy as np
from scipy.signal import savgol_filter

HERE = os.path.dirname(os.path.abspath(__file__))
fz = np.load(os.path.join(HERE, "feat_raw.npz"), allow_pickle=True)
FN = [str(s) for s in fz["featnames"]]
I = {n: k for k, n in enumerate(FN)}
BEARINGS = sorted(k for k in fz.files if k != "featnames")

SMOOTH_FRAC = 0.08          # smoothing bandwidth, fraction of life
LOCAL_BW = 0.08             # bandwidth for the local noise scale
SIGMA_MAX = 0.5
K_FOLDS = 10
TAU0_GRID = np.round(np.arange(0.30, 0.96, 0.05), 3)
W_GRID = np.array([0.05, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50, 0.60])
N_MC = 400
RNG = np.random.default_rng(20260809)


def robust_scale(x):
    return float(1.4826 * np.median(np.abs(x - np.median(x))))


def oof_trend(D, half):
    """Interleaved K-fold local-quadratic trend: the fit at index i never uses i.

    Folds are interleaved (i mod K) rather than contiguous because the noise is
    white and the trend is smooth -- neighbours are informative about the trend
    but carry independent noise, which is the property cross-validation needs.
    """
    n = len(D)
    idx = np.arange(n)
    pred = np.empty(n)
    for j in range(K_FOLDS):
        test = idx[idx % K_FOLDS == j]
        keep = idx % K_FOLDS != j
        for i in test:
            lo, hi = max(0, i - half), min(n, i + half + 1)
            m = keep[lo:hi]
            xs = idx[lo:hi][m].astype(float) - i
            ys = D[lo:hi][m]
            if len(xs) < 6:
                pred[i] = D[i]
                continue
            c = np.polyfit(xs, ys, 2)
            pred[i] = c[-1]                    # value at the held-out point
    return pred


def local_scale(resid, half):
    """Rolling robust noise scale, so sigma_m may vary over life."""
    n = len(resid)
    out = np.empty(n)
    for i in range(n):
        lo, hi = max(0, i - half), min(n, i + half + 1)
        out[i] = robust_scale(resid[lo:hi])
    return np.maximum(out, 1e-9)


def prep(b):
    M = fz[b]
    x = M[:, I["b4_6"]] + M[:, I["b6_8"]] + M[:, I["b8_10"]]
    n = len(x)
    base = np.median(x[:max(3, int(0.05 * n))]); end = np.median(x[-3:])
    if not np.isfinite(base) or not np.isfinite(end) or abs(end - base) < 1e-12:
        return None
    D = (x - base) / (end - base)
    win = max(7, int(SMOOTH_FRAC * n)); win += (win % 2 == 0)
    win = min(win, n - 1 if (n - 1) % 2 == 1 else n - 2)
    if win < 7:
        return None
    tr_in = savgol_filter(D, win, 2)
    dtrend = savgol_filter(D, win, 2, deriv=1, delta=1.0 / n)
    res_in = D - tr_in
    if not np.isfinite(robust_scale(res_in)) or robust_scale(res_in) > SIGMA_MAX:
        return None
    half = win // 2
    tr_oof = oof_trend(D, half)
    res_oof = D - tr_oof
    return dict(n=n, dt=dtrend, D=D, tr_oof=tr_oof,
                res_in=res_in, res_oof=res_oof,
                sig_in=robust_scale(res_in), sig_oof=robust_scale(res_oof),
                sig_loc=local_scale(res_oof, max(5, int(LOCAL_BW * n))))


print("preparing bearings (out-of-fold smoothing) ...", flush=True)
units = {}
for b in BEARINGS:
    u = prep(b)
    if u is not None:
        units[b] = u
        print(f"  {b}: sigma_in {u['sig_in']:.4f} -> sigma_oof {u['sig_oof']:.4f} "
              f"(x{u['sig_oof']/u['sig_in']:.2f}); local scale spans "
              f"[{u['sig_loc'].min():.4f}, {u['sig_loc'].max():.4f}] "
              f"({u['sig_loc'].max()/u['sig_loc'].min():.0f}x)", flush=True)

infl = np.array([u["sig_oof"] / u["sig_in"] for u in units.values()])
span = np.array([u["sig_loc"].max() / u["sig_loc"].min() for u in units.values()])
print(f"\nout-of-fold inflates sigma_m by a median {np.median(infl):.3f}x")
print(f"local noise scale varies by a median {np.median(span):.0f}x over a lifetime")


# ------------------------------------------------------------------ sweeps
def sweep(cfg, collect=None):
    """cfg in {'A0','A1','A2'} -- see module docstring."""
    lp, lr, cells = [], [], 0
    wcol, tcol = [], []
    for b, u in units.items():
        n, dt = u["n"], u["dt"]
        res = u["res_in"] if cfg == "A0" else u["res_oof"]
        sig_g = u["sig_in"] if cfg == "A0" else u["sig_oof"]
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
                m = len(g)
                if cfg == "A2":
                    s = u["sig_loc"][klo:k0 + 1]
                    fisher = float(np.sum((g / s) ** 2))
                    if fisher <= 0:
                        continue
                    crb = 1.0 / np.sqrt(fisher)
                    # GLS estimator: weight each sample by its own noise scale
                    wt = g / s ** 2
                    den = float(np.sum(wt * g))
                    pool = res[klo:k0 + 1]        # noise drawn from this window
                    draws = RNG.choice(pool, size=(N_MC, m), replace=True)
                    err = draws @ wt / den
                else:
                    den = float(np.sum(g ** 2))
                    if den <= 0:
                        continue
                    crb = sig_g / np.sqrt(den)
                    draws = RNG.choice(res, size=(N_MC, m), replace=True)
                    err = draws @ g / den
                lp.append(np.log(crb)); lr.append(np.log(float(np.std(err, ddof=1))))
                wcol.append(w); tcol.append(tau0)
                cells += 1
    lp, lr = np.asarray(lp), np.asarray(lr)
    slope = float(np.polyfit(lp, lr, 1)[0])
    corr = float(np.corrcoef(lp, lr)[0, 1])
    ratio = np.exp(lr - lp)
    if collect is not None:
        np.savez_compressed(collect, crb=np.exp(lp), emp=np.exp(lr),
                            w=np.asarray(wcol), tau0=np.asarray(tcol))
    return dict(cells=cells, slope=slope, corr=corr,
                median=float(np.median(ratio)),
                q25=float(np.percentile(ratio, 25)),
                q75=float(np.percentile(ratio, 75)))


LABEL = {"A0": "in-sample resid, global sigma",
         "A1": "out-of-fold resid, global sigma",
         "A2": "out-of-fold resid, local sigma"}
print(f"\n{'cfg':<5}{'configuration':<34}{'median':>9}{'IQR':>18}{'slope':>8}{'r':>7}")
print("-" * 81)
out = {}
for cfg in ["A0", "A1", "A2"]:
    r = sweep(cfg, collect=os.path.join(HERE, f"cells_{cfg}.npz"))
    out[cfg] = r
    print(f"{cfg:<5}{LABEL[cfg]:<34}{r['median']:>9.3f}"
          f"{'[%.2f, %.2f]' % (r['q25'], r['q75']):>18}{r['slope']:>8.3f}{r['corr']:>7.3f}")
print("-" * 81)
print("Proposition 1 predicts median 1 and slope 1.")

# ------------------------------------------------------- Part B, recomputed
# The linearization radius was previously measured against the global-sigma
# floor; with that floor now corrected, the breakpoint has to be re-read.
print("\n=== Part B recomputed against the local-sigma floor ===")
print(f"{'|delta|':>9}{'median ratio':>15}{'slope':>9}{'r':>8}")
print("-" * 41)
JITTER = [-24, -16, -8, 0, 8, 16, 24]   # sample offsets, independent noise
partB = []
for dm in [0.005, 0.01, 0.02, 0.03, 0.05, 0.08, 0.12]:
    lps, lrs = [], []
    for b, u in units.items():
        n, dt, sl = u["n"], u["dt"], u["sig_loc"]
        D, tr = u["D"], u["tr_oof"]
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
                g0, s0 = dt[klo:k0 + 1], sl[klo:k0 + 1]
                fisher = float(np.sum((g0 / s0) ** 2))
                if fisher <= 0:
                    continue
                crb = 1.0 / np.sqrt(fisher)
                # err = (deterministic linearization bias) + (noise term).
                # The bias is fixed by the trend and the shift; the noise is
                # resampled, so the small-|delta| limit reproduces A2 exactly
                # instead of being read off a couple of overlapping draws.
                errs = []
                for sgn in (-1, 1):
                    sh = int(round(sgn * dm * n))
                    jlo, jhi = klo - sh, k0 - sh
                    if jlo < 0 or jhi >= n:
                        continue
                    g, s = dt[jlo:jhi + 1], sl[jlo:jhi + 1]
                    if len(g) != k0 - klo + 1:
                        continue
                    wt = g / s ** 2
                    den = float(np.sum(wt * g))
                    if den <= 0:
                        continue
                    bias = float(np.sum(wt * (tr[klo:k0 + 1] - tr[jlo:jhi + 1]))
                                 / den) - sgn * dm
                    pool = u["res_oof"][klo:k0 + 1]
                    draws = RNG.choice(pool, size=(N_MC, len(g)), replace=True)
                    errs.append(np.sqrt(np.mean((bias + draws @ wt / den) ** 2)))
                if len(errs) < 2:
                    continue
                lps.append(np.log(crb))
                lrs.append(np.log(float(np.mean(errs))))
    lps, lrs = np.asarray(lps), np.asarray(lrs)
    sl_ = float(np.polyfit(lps, lrs, 1)[0])
    co = float(np.corrcoef(lps, lrs)[0, 1])
    md = float(np.median(np.exp(lrs - lps)))
    partB.append(dict(delta=dm, ratio=md, slope=sl_, corr=co))
    print(f"{dm:>9.3f}{md:>15.2f}{sl_:>9.3f}{co:>8.3f}")

with open(os.path.join(HERE, "oof_results.json"), "w") as fh:
    json.dump(dict(sigma_inflation_median=float(np.median(infl)),
                   local_scale_span_median=float(np.median(span)),
                   n_units=len(units), configs=out, partB=partB), fh, indent=2)
print("\nwrote oof_results.json")
