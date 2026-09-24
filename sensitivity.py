"""
Stage 6 -- do the Section V conclusions survive the analyst's free choices?

Three decisions were made before the results were seen, and each is a place a
reviewer can reasonably push:

  (S1) the exclusion rule  -- bearings with sigma_m > SIGMA_MAX are dropped as
       showing no net degradation (2 of 17 at the default 0.5);
  (S2) the smoothing bandwidth SMOOTH_FRAC, which sets both Dhat' and the
       residual;
  (S3) whether any single bearing carries the result.

S1 and S2 are swept jointly; S3 is a leave-one-bearing-out jackknife at the
default setting. Reported for each cell: floor attainment (median ratio, slope,
r), the linearization radius, and the per-band exponents that Fig. 2(c) quotes.

Kept deliberately blunt: no configuration is dropped from the report, including
the one with no screening at all.
"""
import os
import json
import numpy as np
from scipy.signal import savgol_filter
from scipy.optimize import minimize_scalar

HERE = os.path.dirname(os.path.abspath(__file__))
fz = np.load(os.path.join(HERE, "feat_raw.npz"), allow_pickle=True)
FN = [str(s) for s in fz["featnames"]]
I = {n: k for k, n in enumerate(FN)}
BEARINGS = sorted(k for k in fz.files if k != "featnames")

K_FOLDS, LOCAL_BW = 10, 0.08
TAU0_GRID = np.round(np.arange(0.30, 0.96, 0.05), 3)
W_GRID = np.array([0.05, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50, 0.60])
N_MC = 200
QSTAR = 0.35
DELTAS = [0.01, 0.02, 0.03, 0.05, 0.08]
BREAK_AT = 2.5           # ratio marking the end of the plateau

SIGMA_GRID = [0.3, 0.5, 1.0, np.inf]
SMOOTH_GRID = [0.05, 0.08, 0.12]

BANDS = {
    "rms": lambda M: M[:, I["rms"]],
    "lf 0-2kHz": lambda M: M[:, I["b0_1"]] + M[:, I["b1_2"]],
    "hf 4-10kHz": lambda M: M[:, I["b4_6"]] + M[:, I["b6_8"]] + M[:, I["b8_10"]],
}


def rscale(x):
    return float(1.4826 * np.median(np.abs(x - np.median(x))))


def normalize(x):
    n = len(x)
    base = np.median(x[:max(3, int(0.05 * n))]); end = np.median(x[-3:])
    if not np.isfinite(base) or not np.isfinite(end) or abs(end - base) < 1e-12:
        return None
    return (x - base) / (end - base)


def oof_trend(D, half):
    n = len(D); idx = np.arange(n); pred = np.empty(n)
    for j in range(K_FOLDS):
        keep = idx % K_FOLDS != j
        for i in idx[idx % K_FOLDS == j]:
            lo, hi = max(0, i - half), min(n, i + half + 1)
            m = keep[lo:hi]
            xs = idx[lo:hi][m].astype(float) - i
            if len(xs) < 6:
                pred[i] = D[i]; continue
            pred[i] = np.polyfit(xs, D[lo:hi][m], 2)[-1]
    return pred


def local_scale(res, half):
    n = len(res); out = np.empty(n)
    for i in range(n):
        lo, hi = max(0, i - half), min(n, i + half + 1)
        out[i] = rscale(res[lo:hi])
    return np.maximum(out, 1e-9)


def build(b, smooth_frac, band="hf 4-10kHz"):
    """All per-bearing quantities at a given bandwidth. No screening here."""
    D = normalize(BANDS[band](fz[b]))
    if D is None:
        return None
    n = len(D)
    win = max(7, int(smooth_frac * n)); win += (win % 2 == 0)
    win = min(win, n - 1 if (n - 1) % 2 == 1 else n - 2)
    if win < 7:
        return None
    dt = savgol_filter(D, win, 2, deriv=1, delta=1.0 / n)
    sig_in = rscale(D - savgol_filter(D, win, 2))
    tr = oof_trend(D, win // 2)
    res = D - tr
    return dict(n=n, D=D, tr=tr, dt=dt, res=res, sig_in=sig_in,
                sig_loc=local_scale(res, max(5, int(LOCAL_BW * n))))


def cells(units, rng):
    """A2 configuration: out-of-fold residual, local sigma."""
    lp, lr = [], []
    for u in units.values():
        n, dt, sl, res = u["n"], u["dt"], u["sig_loc"], u["res"]
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
                g, s = dt[klo:k0 + 1], sl[klo:k0 + 1]
                fisher = float(np.sum((g / s) ** 2))
                if fisher <= 0:
                    continue
                wt = g / s ** 2
                den = float(np.sum(wt * g))
                if den <= 0:
                    continue
                draws = rng.choice(res[klo:k0 + 1], size=(N_MC, len(g)), replace=True)
                lp.append(np.log(1.0 / np.sqrt(fisher)))
                lr.append(np.log(float(np.std(draws @ wt / den, ddof=1))))
    if len(lp) < 20:
        return None
    lp, lr = np.asarray(lp), np.asarray(lr)
    return dict(n_cells=len(lp),
                median=float(np.median(np.exp(lr - lp))),
                slope=float(np.polyfit(lp, lr, 1)[0]),
                corr=float(np.corrcoef(lp, lr)[0, 1]))


def radius(units, rng):
    """Largest |delta| whose median ratio is still below BREAK_AT."""
    best = 0.0
    for dm in DELTAS:
        lps, lrs = [], []
        for u in units.values():
            n, D, tr, dt, sl, res = (u["n"], u["D"], u["tr"], u["dt"],
                                     u["sig_loc"], u["res"])
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
                    f0 = float(np.sum((dt[klo:k0 + 1] / sl[klo:k0 + 1]) ** 2))
                    if f0 <= 0:
                        continue
                    errs = []
                    for sgn in (-1, 1):
                        sh = int(round(sgn * dm * n))
                        jlo, jhi = klo - sh, k0 - sh
                        if jlo < 0 or jhi >= n or jhi - jlo != k0 - klo:
                            continue
                        g, s = dt[jlo:jhi + 1], sl[jlo:jhi + 1]
                        wt = g / s ** 2
                        den = float(np.sum(wt * g))
                        if den <= 0:
                            continue
                        bias = float(np.sum(wt * (tr[klo:k0 + 1] - tr[jlo:jhi + 1]))
                                     / den) - sgn * dm
                        dr = rng.choice(res[klo:k0 + 1], size=(N_MC, len(g)),
                                        replace=True)
                        errs.append(np.sqrt(np.mean((bias + dr @ wt / den) ** 2)))
                    if len(errs) < 2:
                        continue
                    lps.append(np.log(1.0 / np.sqrt(f0)))
                    lrs.append(np.log(float(np.mean(errs))))
        if not lps:
            continue
        if float(np.median(np.exp(np.asarray(lrs) - np.asarray(lps)))) < BREAK_AT:
            best = dm
    return best


def band_exponents(smooth_frac, sigma_max):
    """beta_eff and tau_min per band, under the same screening rule."""
    out = {}
    for name in BANDS:
        betas, tmins, kept = [], [], 0
        for b in BEARINGS:
            u = build(b, smooth_frac, band=name)
            if u is None or u["sig_in"] > sigma_max:
                continue
            kept += 1
            dens = u["dt"] ** 2
            if dens.sum() <= 0:
                continue
            cum = np.cumsum(dens) / dens.sum()
            n = u["n"]
            tau = np.arange(1, n + 1) / n
            f = lambda bb: float(np.mean((cum - tau ** (2 * bb - 1)) ** 2))
            betas.append(float(minimize_scalar(f, bounds=(0.51, 30.0),
                                               method="bounded",
                                               options={"xatol": 1e-3}).x))
            k = int(np.searchsorted(cum, QSTAR))
            tmins.append((k + 1) / n if k < n else 1.0)
        out[name] = dict(units=kept,
                         beta=float(np.median(betas)) if betas else None,
                         tau_min=float(np.median(tmins)) if tmins else None)
    return out


# ------------------------------------------------------------------ S1 x S2
print("=== S1 x S2: exclusion threshold x smoothing bandwidth ===")
print(f"{'smooth':>7}{'sigma_max':>11}{'units':>7}{'median':>9}{'slope':>8}"
      f"{'r':>7}{'radius':>9}{'b_rms':>8}{'b_lf':>7}{'b_hf':>7}{'tmin_hf':>9}")
print("-" * 90)
grid = {}
for sf in SMOOTH_GRID:
    built = {b: build(b, sf) for b in BEARINGS}
    built = {b: u for b, u in built.items() if u is not None}
    for sm in SIGMA_GRID:
        units = {b: u for b, u in built.items() if u["sig_in"] <= sm}
        if len(units) < 3:
            continue
        rng = np.random.default_rng(20260809)
        c = cells(units, rng)
        r = radius(units, np.random.default_rng(20260809))
        be = band_exponents(sf, sm)
        key = f"sf{sf}_sm{sm}"
        grid[key] = dict(smooth=sf, sigma_max=None if np.isinf(sm) else sm,
                         units=len(units), cells=c, radius=r, bands=be)
        lab = "none" if np.isinf(sm) else f"{sm:.2f}"
        print(f"{sf:>7.2f}{lab:>11}{len(units):>7}{c['median']:>9.3f}"
              f"{c['slope']:>8.3f}{c['corr']:>7.3f}{r:>9.3f}"
              f"{be['rms']['beta']:>8.1f}{be['lf 0-2kHz']['beta']:>7.2f}"
              f"{be['hf 4-10kHz']['beta']:>7.2f}{be['hf 4-10kHz']['tau_min']:>9.3f}")
print("-" * 90)
print("'radius' = largest |delta| whose median ratio is still under "
      f"{BREAK_AT}; 'none' = no bearing excluded.")

# ---------------------------------------------------------------------- S3
print("\n=== S3: leave-one-bearing-out (default 0.08 / 0.5) ===")
built = {b: u for b in BEARINGS if (u := build(b, 0.08)) is not None}
base_units = {b: u for b, u in built.items() if u["sig_in"] <= 0.5}
full = cells(base_units, np.random.default_rng(20260809))
print(f"all {len(base_units)} units: median {full['median']:.3f}, "
      f"slope {full['slope']:.3f}, r {full['corr']:.3f}")
jack = {}
for b in list(base_units):
    sub = {k: v for k, v in base_units.items() if k != b}
    c = cells(sub, np.random.default_rng(20260809))
    jack[b] = c
med = np.array([c["median"] for c in jack.values()])
slo = np.array([c["slope"] for c in jack.values()])
worst = max(jack, key=lambda b: abs(jack[b]["median"] - full["median"]))
print(f"jackknife median ratio: [{med.min():.3f}, {med.max():.3f}]")
print(f"jackknife slope:        [{slo.min():.3f}, {slo.max():.3f}]")
print(f"largest single-unit influence: dropping {worst} moves the median to "
      f"{jack[worst]['median']:.3f} (from {full['median']:.3f})")

with open(os.path.join(HERE, "sensitivity.json"), "w") as fh:
    json.dump(dict(grid=grid, full=full,
                   jackknife={b: c for b, c in jack.items()},
                   jack_median_range=[float(med.min()), float(med.max())],
                   jack_slope_range=[float(slo.min()), float(slo.max())]),
              fh, indent=2)
print("\nwrote sensitivity.json")
