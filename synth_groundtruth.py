"""
Ground truth: does the pipeline recover what it is supposed to recover?

Every empirical number so far assumes the estimation chain -- Savitzky-Golay
trend, out-of-fold residual, rolling robust scale, windowed ML estimator -- is
unbiased. That has never been checked against data whose answer is known.

Here trajectories are generated from a chosen class with known beta and known
noise, run through the IDENTICAL pipeline, and three things are asked:
    does the slope of measured-vs-predicted come out at 1?
    does beta_eff recover the beta that generated the data?
    at what record length and noise level do those stop being true?

Anything the pipeline cannot recover on synthetic data, it cannot be trusted to
have measured on real data.
"""
import os, json
import numpy as np
from pipeline import prepare, beta_eff, LOCAL_BW

HERE = os.path.dirname(os.path.abspath(__file__))
TAU0 = np.round(np.arange(0.30, 0.96, 0.05), 3)
WGRID = np.array([0.10, 0.15, 0.20, 0.30, 0.40, 0.50, 0.60])
N_MC = 300


def make(n, beta, sigma, reps, rng, tail=None):
    """Power-law degradation sampled at n points with additive noise."""
    tau = np.arange(1, n + 1) / n
    D = tau ** beta
    out = []
    for _ in range(reps):
        if tail is None:
            e = rng.normal(0, sigma, n)
        else:                       # Student-t: same scale, heavier tails
            e = rng.standard_t(tail, n)
            e *= sigma / np.sqrt(tail / (tail - 2))
        out.append(D + e)
    return out


def crb_fit(units, rng):
    lp, lr = [], []
    for u in units:
        n, dt, sl, res = u["n"], u["dtrend"], u["sig_loc"], u["res"]
        for t0 in TAU0:
            k0 = int(round(t0 * n)) - 1
            if k0 < 5 or k0 >= n:
                continue
            for w in WGRID:
                if w > t0:
                    continue
                klo = max(0, int(round((t0 - w) * n)) - 1)
                if k0 - klo < 10:
                    continue
                g, s = dt[klo:k0 + 1], sl[klo:k0 + 1]
                f = float(np.sum((g / s) ** 2))
                if f <= 0:
                    continue
                wt = g / s ** 2
                den = float(np.sum(wt * g))
                if den <= 0:
                    continue
                dr = rng.choice(res[klo:k0 + 1], size=(N_MC, len(g)), replace=True)
                lp.append(np.log(1.0 / np.sqrt(f)))
                lr.append(np.log(float(np.std(dr @ wt / den, ddof=1))))
    if len(lp) < 30:
        return None
    lp, lr = np.asarray(lp), np.asarray(lr)
    return (float(np.polyfit(lp, lr, 1)[0]), float(np.corrcoef(lp, lr)[0, 1]),
            float(np.median(np.exp(lr - lp))), len(lp))


print("=== A. record length, with the theory's own assumptions satisfied ===")
print("    true beta = 3, Gaussian noise, sigma = 0.03")
print(f"{'n':>7}{'series':>8}{'cells':>7}{'slope':>8}{'r':>7}{'median':>8}"
      f"{'beta_eff':>10}{'bias':>8}")
print("-" * 63)
rng = np.random.default_rng(7)
lenrows = []
for n in [50, 90, 150, 300, 600, 1400]:
    raws = make(n, 3.0, 0.03, 40, rng)
    us = [u for r in raws for u, _ in [prepare(r)] if u]
    if not us:
        print(f"{n:>7}   all series screened out"); continue
    res = crb_fit(us, rng)
    be = float(np.median([beta_eff(u) for u in us]))
    if res is None:
        print(f"{n:>7}{len(us):>8}   too few cells"); continue
    sl, r_, md, nc = res
    lenrows.append(dict(n=n, slope=sl, corr=r_, median=md, beta_eff=be))
    print(f"{n:>7}{len(us):>8}{nc:>7}{sl:>8.3f}{r_:>7.3f}{md:>8.3f}"
          f"{be:>10.2f}{be-3.0:>8.2f}")

print("\n=== B. noise level, at a length where the test works (n = 600) ===")
print(f"{'sigma':>8}{'slope':>8}{'r':>7}{'median':>8}{'beta_eff':>10}{'bias':>8}")
print("-" * 49)
noiserows = []
for s in [0.005, 0.015, 0.03, 0.06, 0.12]:
    raws = make(600, 3.0, s, 40, rng)
    us = [u for r in raws for u, _ in [prepare(r)] if u]
    if not us:
        print(f"{s:>8.3f}   screened out"); continue
    res = crb_fit(us, rng)
    be = float(np.median([beta_eff(u) for u in us]))
    if res is None:
        print(f"{s:>8.3f}   too few cells"); continue
    sl, r_, md, _ = res
    noiserows.append(dict(sigma=s, slope=sl, corr=r_, median=md, beta_eff=be))
    print(f"{s:>8.3f}{sl:>8.3f}{r_:>7.3f}{md:>8.3f}{be:>10.2f}{be-3.0:>8.2f}")

print("\n=== C. heavy tails, as in the battery record (n = 600, sigma = 0.03) ===")
print(f"{'noise':>12}{'kurt':>8}{'slope':>8}{'r':>7}{'median':>8}")
print("-" * 43)
tailrows = []
for label, df in [("Gaussian", None), ("t, df=5", 5), ("t, df=3", 3), ("t, df=2.5", 2.5)]:
    raws = make(600, 3.0, 0.03, 40, rng, tail=df)
    us = [u for r in raws for u, _ in [prepare(r)] if u]
    if not us:
        continue
    k = float(np.median([np.mean((u["res"]/u["sig_oof"])**4) - 3 for u in us]))
    res = crb_fit(us, rng)
    if res is None:
        continue
    sl, r_, md, _ = res
    tailrows.append(dict(noise=label, kurt=k, slope=sl, corr=r_, median=md))
    print(f"{label:>12}{k:>8.1f}{sl:>8.3f}{r_:>7.3f}{md:>8.3f}")

json.dump(dict(length=lenrows, noise=noiserows, tails=tailrows),
          open(os.path.join(HERE, "synth_groundtruth.json"), "w"), indent=2)
print("\nwrote synth_groundtruth.json")
