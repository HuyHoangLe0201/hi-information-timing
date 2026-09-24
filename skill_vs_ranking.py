"""
Does the ranking predict accuracy?  A leave-one-bearing-out test.

The paper states as an open weakness that it demonstrates no accuracy
improvement: Section 8.2 could not settle whether the distribution finding buys
anything, because a remaining-life regression on seventeen bearings has no skill
to speak of.  That was the wrong question to ask of seventeen units.  The right
one is narrower and answerable: given an indicator reading, how well can a unit's
age be estimated, and does the framework's ordering of indicators predict the
ordering of that accuracy?

The task is age estimation from a single indicator, leave-one-bearing-out.  Three
choices keep it honest.

The held-out bearing is standardised on its own first twenty per cent only --
the healthy period a deployed monitor calibrates on -- never on statistics of the
whole record, which would leak the future into the present.  Ages below that
baseline are not scored.

The reference is not zero skill but the Bayesian floor of Proposition 2 realised:
an estimator using the fleet prior and nothing else.  With ages scored on
[0.2, 1] that estimator returns the prior mean and its error is the prior
standard deviation.  Beating it is the whole question, and an indicator that
cannot is not carrying usable information whatever its curve says.

And the fleet trend is built from the other sixteen bearings only, so no
information about the held-out unit enters through the model either.
"""
import os
import json
import numpy as np
from scipy.signal import medfilt

from nonparam import info_curve, quantile

HERE = os.path.dirname(os.path.abspath(__file__))
SEED = 20260828
QSTAR = 0.35
BASE = 0.20          # healthy fraction used for calibration, and the score floor
GRID = np.linspace(BASE, 1.0, 200)
SMOOTH = 9           # trailing samples a deployed monitor would smooth over


def calibrated(x, base=BASE):
    """Standardise on the healthy head of the record, as a monitor would."""
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    n = len(x)
    if n < 120:
        return None
    h = x[:max(10, int(base * n))]
    mu = float(np.median(h))
    sd = float(np.median(np.abs(h - mu)) * 1.4826)
    if not np.isfinite(sd) or sd <= 0:
        sd = float(np.std(h))
    if not np.isfinite(sd) or sd <= 0:
        return None
    z = (x - mu) / sd
    k = SMOOTH if SMOOTH % 2 else SMOOTH + 1
    return medfilt(z, kernel_size=min(k, n - (n + 1) % 2))


def on_grid(z):
    tau = (np.arange(len(z)) + 1.0) / len(z)
    return np.interp(GRID, tau, z)


z = np.load(os.path.join(HERE, "feat_raw.npz"), allow_pickle=True)
FN = [str(s) for s in z["featnames"]]
I = {k: i for i, k in enumerate(FN)}
units = sorted(k for k in z.files if k != "featnames")

series = {}
for nm in FN:
    d = {}
    for u in units:
        c = calibrated(np.asarray(z[u][:, I[nm]], float))
        if c is not None:
            d[u] = c
    if len(d) >= 8:
        series[nm] = d
print(f"{len(series)} channels, {len(units)} bearings, "
      f"calibrated on the first {int(100 * BASE)}% of each record\n")

# the prior-only reference: ages are scored on [BASE, 1], so the prior mean is
# its midpoint and the prior standard deviation is that of a uniform on it
PRIOR_SD = (1.0 - BASE) / np.sqrt(12.0)
PRIOR_MEAN = 0.5 * (1.0 + BASE)
print(f"prior-only reference: predict {PRIOR_MEAN:.2f} always, "
      f"error {PRIOR_SD:.4f} of a lifetime\n")

rng = np.random.default_rng(SEED)
print(f"{'channel':<10}{'units':>7}{'tau_min':>10}{'RMSE':>9}{'skill':>9}"
      f"{'beats prior':>13}")
print("-" * 58)
rows = []
for nm, d in series.items():
    us = sorted(d)
    curves = {u: on_grid(d[u]) for u in us}
    errs, wins = [], 0
    for held in us:
        train = [curves[u] for u in us if u != held]
        m = np.median(np.column_stack(train), axis=1)
        if np.ptp(m) <= 0:
            continue
        y = curves[held]
        # nearest point of the fleet trend: the maximum-likelihood age under a
        # constant-noise model, with no use of the held-out record's own scale
        est = GRID[np.argmin(np.abs(m[None, :] - y[:, None]), axis=1)]
        e = float(np.sqrt(np.mean((est - GRID) ** 2)))
        errs.append(e)
        wins += int(e < PRIOR_SD)
    if not errs:
        continue
    rmse = float(np.median(errs))
    # the framework's own ranking of this channel, on the same bearings
    taus = []
    for u in us:
        x = np.asarray(z[u][:, I[nm]], float)
        x = x[np.isfinite(x)]
        F = info_curve(x, rng=rng)
        if F is not None:
            taus.append(quantile(F, QSTAR))
    tmin = float(np.median(taus)) if taus else np.nan
    rows.append(dict(channel=nm, units=len(errs), tau_min=tmin, rmse=rmse,
                     skill=1.0 - rmse / PRIOR_SD, wins=wins))
    print(f"{nm:<10}{len(errs):>7}{tmin:>10.3f}{rmse:>9.4f}"
          f"{1.0 - rmse / PRIOR_SD:>9.3f}{wins:>9}/{len(errs):<3}")
print("-" * 58)

best = max(rows, key=lambda r: r["skill"])
any_beat = [r for r in rows if r["skill"] > 0]
print(f"\n{len(any_beat)} of {len(rows)} channels beat the prior-only reference; "
      f"the best is")
print(f"{best['channel']}, error {best['rmse']:.4f} against {PRIOR_SD:.4f}, "
      f"a skill of {best['skill']:.3f}.")

# --- the question the paper actually needs answering -------------------------
t = np.array([r["tau_min"] for r in rows], float)
s = np.array([r["skill"] for r in rows], float)
m = np.isfinite(t) & np.isfinite(s)
print()
if m.sum() >= 5:
    rk = lambda v: np.argsort(np.argsort(v))
    rho = float(np.corrcoef(rk(t[m]), rk(s[m]))[0, 1])
    print(f"Spearman correlation between the framework's earliest usable age and")
    print(f"the measured skill: {rho:+.2f} over {int(m.sum())} channels.")
    print("The framework says a smaller tau_min is a better indicator, so a")
    print("NEGATIVE correlation is the prediction.")
    print()
    if rho <= -0.5:
        print("The ordering the framework produces before any model is trained")
        print("predicts the ordering of accuracy after one is, which is the")
        print("claim Section 8.2 could not previously settle.")
    elif rho >= 0.5:
        print("The correlation runs the WRONG WAY: indicators the framework")
        print("ranks as usable earlier are the less accurate ones here.  That is")
        print("reported as it stands.")
    else:
        print("The correlation is weak, so on this task the ordering does not")
        print("predict accuracy either way, and the paper cannot claim it does.")

json.dump(dict(base=BASE, prior_sd=PRIOR_SD, prior_mean=PRIOR_MEAN,
               channels=rows,
               beat_prior=len(any_beat), total=len(rows),
               best=best,
               spearman=(float(np.corrcoef(np.argsort(np.argsort(t[m])),
                                           np.argsort(np.argsort(s[m])))[0, 1])
                         if m.sum() >= 5 else None)),
          open(os.path.join(HERE, "skill_vs_ranking.json"), "w"), indent=2,
          default=float)
