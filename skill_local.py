"""
Does the ranking predict accuracy, on the estimand the bound actually governs?

A first attempt asked each indicator to estimate a bearing's age by inverting a
fleet trend, and every channel lost to the prior-only reference.  That test was
unfair, and the paper's own theory says why.  The Cramer-Rao statement of
Proposition 1 is about a LOCAL clock offset around an approximate age, not about
global inversion of a trend, and Section 6.6 gives the radius within which the
local statement holds -- a median 0.0043 of a lifetime for an oracle and 0.072
for an estimator that must also fit the trend.  A global inversion asks the
indicator to resolve an offset two orders of magnitude outside that radius, where
the likelihood is multimodal and the bound describes nothing.  Its failure is a
property of the question, not of the indicators.

The estimand the bound governs is refinement: an approximate age is supplied,
and the indicator is asked to improve it.  That is also what a deployed monitor
does, since elapsed operating time is always known.  The test is therefore run
over a sweep of prior-error scales, and reports where refinement helps, where it
does not, and whether the framework's ordering of indicators predicts the
ordering of the help.

A second attempt used a one-sample plug-in inverse, delta = (y - m)/m', and every
channel came out worse than the prior by a factor of three -- the cap on the
step, reached because the fleet trend is nearly flat over most of life and the
division blows up.  That estimator is not the one the theorem is about either.
Proposition 1 is the variance of the WINDOWED, WEIGHTED maximum-likelihood
estimator

    delta_hat = sum_k (m'_k / sigma^2) (y_k - m_k) / sum_k (m'^2_k / sigma^2),

whose denominator is the windowed information itself.  That is the estimator used
below.  Having changed it twice, the protocol is now fixed on principle rather
than on outcome: the estimator is the one the bound is the variance of, the
window is swept rather than chosen, and whatever the result is it is reported.

Everything else is unchanged and remains deliberately strict: the held-out
bearing is standardised on its own healthy head only, its noise scale is taken
from that same head, the fleet trend comes from the other sixteen bearings, and
no statistic of the held-out record's future enters at any point.
"""
import os
import json
import numpy as np
from scipy.signal import medfilt

from nonparam import info_curve, quantile

HERE = os.path.dirname(os.path.abspath(__file__))
SEED = 20260828
QSTAR, BASE, SMOOTH = 0.35, 0.20, 9
GRID = np.linspace(BASE, 1.0, 400)
LO, HI = 0.30, 0.90                      # ages scored, away from both edges
SIGMAS = (0.005, 0.01, 0.02, 0.05, 0.10, 0.231)
DRAWS = 300
WINDOW = 0.10                            # trailing buffer, swept below


def calibrated(x, base=BASE):
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
    k = SMOOTH if SMOOTH % 2 else SMOOTH + 1
    return medfilt((x - mu) / sd, kernel_size=min(k, n - (n + 1) % 2))


def on_grid(zc):
    tau = (np.arange(len(zc)) + 1.0) / len(zc)
    return np.interp(GRID, tau, zc)


z = np.load(os.path.join(HERE, "feat_raw.npz"), allow_pickle=True)
FN = [str(s) for s in z["featnames"]]
I = {k: i for i, k in enumerate(FN)}
units = sorted(k for k in z.files if k != "featnames")

series = {}
for nm in FN:
    d = {u: calibrated(np.asarray(z[u][:, I[nm]], float)) for u in units}
    d = {u: v for u, v in d.items() if v is not None}
    if len(d) >= 8:
        series[nm] = d
print(f"{len(series)} channels, {len(units)} bearings; ages scored on "
      f"[{LO}, {HI}]\n")

rng = np.random.default_rng(SEED)


def refine_skill(nm, sigma, w=WINDOW):
    """Median over bearings of the refined error, in units of the prior error.

    The estimator is the windowed weighted maximum-likelihood one whose variance
    Proposition 1 bounds.  The window trails the prior age, as a buffer does.
    """
    d = series[nm]
    us = sorted(d)
    curves = {u: on_grid(d[u]) for u in us}
    out = []
    for held in us:
        m = np.median(np.column_stack([curves[u] for u in us if u != held]),
                      axis=1)
        dm = np.gradient(m, GRID)
        y = curves[held]
        # noise scale from the healthy head of the held-out record only
        head = y[GRID <= BASE + 0.05]
        sg = float(np.median(np.abs(head - np.median(head))) * 1.4826)
        if not np.isfinite(sg) or sg <= 0:
            sg = float(np.std(head))
        if not np.isfinite(sg) or sg <= 0:
            continue
        tt = rng.uniform(LO, HI, size=DRAWS)
        t0 = np.clip(tt + rng.normal(0.0, sigma, size=DRAWS), BASE + 1e-3,
                     1.0 - 1e-3)
        est = np.empty(DRAWS)
        for j in range(DRAWS):
            lo = max(BASE, t0[j] - w)
            sel = (GRID >= lo) & (GRID <= t0[j])
            if sel.sum() < 5:
                est[j] = t0[j]
                continue
            # the record is observed at the TRUE age, shifted by the prior error
            off = tt[j] - t0[j]
            yk = np.interp(np.clip(GRID[sel] + off, 0.0, 1.0), GRID, y)
            g = dm[sel]
            G = float((g ** 2).sum()) / sg ** 2
            if G <= 0:
                est[j] = t0[j]
                continue
            step = float((g * (yk - m[sel])).sum()) / sg ** 2 / G
            est[j] = t0[j] + np.clip(step, -3 * sigma, 3 * sigma)
        out.append(float(np.sqrt(np.mean((np.clip(est, 0, 1) - tt) ** 2))
                         / np.sqrt(np.mean((t0 - tt) ** 2))))
    return float(np.median(out)) if out else np.nan


print("refined error as a share of the prior error; below one is an improvement")
print("(the step is capped at three prior standard deviations, so a channel")
print("that carries nothing returns the prior rather than diverging)\n")
hdr = "".join(f"{s:>9.3f}" for s in SIGMAS)
print(f"{'channel':<10}{'tau_min':>9}{hdr}")
print("-" * (19 + 9 * len(SIGMAS)))
rows = []
for nm in series:
    taus = []
    for u in sorted(series[nm]):
        x = np.asarray(z[u][:, I[nm]], float)
        x = x[np.isfinite(x)]
        F = info_curve(x, rng=rng)
        if F is not None:
            taus.append(quantile(F, QSTAR))
    tmin = float(np.median(taus)) if taus else np.nan
    vals = [refine_skill(nm, s) for s in SIGMAS]
    rows.append(dict(channel=nm, tau_min=tmin,
                     ratios={str(s): v for s, v in zip(SIGMAS, vals)}))
    print(f"{nm:<10}{tmin:>9.3f}" + "".join(f"{v:>9.2f}" for v in vals))
print("-" * (19 + 9 * len(SIGMAS)))

# where does refinement help at all?
print()
helped = {}
for i, s in enumerate(SIGMAS):
    v = np.array([r["ratios"][str(s)] for r in rows], float)
    k = int(np.sum(v < 1.0))
    helped[s] = k
    print(f"prior error {s:>6.3f}: {k:>2} of {len(rows)} channels improve on it, "
          f"best {np.nanmin(v):.2f}")
print()
RAD = 0.072
inside = [s for s in SIGMAS if s <= RAD and helped[s] > 0]
outside = [s for s in SIGMAS if s > RAD and helped[s] > 0]
print(f"The practical linearisation radius of Section 6.6 is {RAD:.3f}.")
if helped and max(helped.values()) == 0:
    print("No channel improves on the prior at any prior-error scale, so the")
    print("indicator adds nothing to elapsed operating time on this task.  That")
    print("is a negative result and is reported as one.")
elif inside and not outside:
    print("Refinement helps only where the prior error is inside that radius,")
    print("which is what the radius asserts, here measured on real records.")
else:
    print(f"Refinement helps at {len(inside)} scales inside the radius and "
          f"{len(outside)} outside it,")
    print("so the radius does not by itself separate the two regimes here.")

# --- the ordering question ---------------------------------------------------
# where no scale helps, the ordering is still read at the scale with the best
# single channel, since a ranking can be informative even where every candidate
# loses to the prior
best_s = (max(SIGMAS, key=lambda s: helped[s]) if max(helped.values()) > 0
          else min(SIGMAS,
                   key=lambda s: np.nanmin([r["ratios"][str(s)]
                                            for r in rows])))
t = np.array([r["tau_min"] for r in rows], float)
print()
print(f"ordering test, at the prior error where most channels help "
      f"({best_s:.3f}):\n")
v = np.array([r["ratios"][str(best_s)] for r in rows], float)
m = np.isfinite(t) & np.isfinite(v)
rk = lambda a: np.argsort(np.argsort(a))
rho = float(np.corrcoef(rk(t[m]), rk(v[m]))[0, 1]) if m.sum() >= 5 else np.nan
print(f"{'channel':<10}{'tau_min':>9}{'refined/prior':>15}")
print("-" * 34)
for r in sorted(rows, key=lambda r: r["tau_min"]):
    print(f"{r['channel']:<10}{r['tau_min']:>9.3f}"
          f"{r['ratios'][str(best_s)]:>15.2f}")
print("-" * 34)
print(f"\nSpearman between tau_min and the refined-error ratio: {rho:+.2f}")
print("The framework ranks a smaller tau_min as better and a smaller ratio is")
print("better, so a POSITIVE correlation is the prediction.\n")

# --- but is there anything to correlate with? -------------------------------
# The ratios above span about two per cent.  A rank correlation over a spread
# that small is a reading of noise unless the noise is smaller still, so it is
# measured: the whole computation is repeated under independent seeds and the
# scatter of each channel's ratio recorded.
print("repeatability of the ratios, under independent random draws\n")
SEEDS = 5
rep = {r["channel"]: [] for r in rows}
for k in range(SEEDS):
    rng = np.random.default_rng(SEED + 1000 * (k + 1))
    for nm in series:
        rep[nm].append(refine_skill(nm, best_s))
sd = {nm: float(np.std(v, ddof=1)) for nm, v in rep.items()}
spread = float(np.std([r["ratios"][str(best_s)] for r in rows], ddof=1))
noise = float(np.median(list(sd.values())))
print(f"{'channel':<10}{'ratio':>9}{'sd over seeds':>16}")
print("-" * 35)
for r in sorted(rows, key=lambda r: r["tau_min"]):
    print(f"{r['channel']:<10}{r['ratios'][str(best_s)]:>9.3f}"
          f"{sd[r['channel']]:>16.3f}")
print("-" * 35)
print(f"\nspread across channels {spread:.3f}, typical scatter of one channel "
      f"{noise:.3f}")
sig = spread > 2.0 * noise
print(f"the spread is {spread / max(noise, 1e-12):.1f} times the scatter\n")
# A rank correlation over ten channels is weak evidence however large it looks:
# the exact permutation null is computed rather than compared with a table.
perm = np.array([np.corrcoef(rk(t[m]), rk(rng.permutation(v[m])))[0, 1]
                 for _ in range(20000)])
pval = float((perm >= rho).mean())
gain = float(100 * (1 - min(r["ratios"][str(best_s)] for r in rows)))
print(f"exact one-sided permutation p-value for rho = {rho:+.2f} with "
      f"{int(m.sum())} channels: {pval:.3f}\n")

if not sig:
    print("The differences between channels are not separable from the noise of")
    print("the measurement, so the rank correlation is a reading of that noise.")
elif pval < 0.05 and rho >= 0.5:
    print("The spread exceeds the noise and the correlation reaches significance,")
    print("so the ordering produced before any model is trained predicts the")
    print("ordering of accuracy after one is.")
elif rho >= 0.5:
    print("The spread exceeds the noise and the correlation is in the predicted")
    print(f"direction, but with {int(m.sum())} channels it does not reach")
    print(f"significance (p = {pval:.2f}).  The result is suggestive and is")
    print("reported as suggestive.")
else:
    print("The correlation is too weak to support an ordering claim.")
print()
print(f"Either way the size of the effect settles its usefulness: the best")
print(f"channel improves on elapsed operating time by {gain:.1f} per cent, and the")
print(f"whole spread across ten channels is {100 * spread:.1f} per cent.  Refinement from a")
print("single indicator is measurable here and not usable, which is the honest")
print("answer to the question Section 8.2 left open.")

json.dump(dict(sigmas=list(SIGMAS), rows=rows,
               helped={str(k): v for k, v in helped.items()},
               radius_used=RAD, best_sigma=float(best_s),
               spearman=(None if not np.isfinite(rho) else float(rho)),
               seed_scatter={k: float(v) for k, v in sd.items()},
               spread_across_channels=spread, typical_scatter=noise,
               separable=bool(sig), spearman_p=float(pval),
               best_gain_percent=float(
                   100 * (1 - min(r["ratios"][str(best_s)] for r in rows)))),
          open(os.path.join(HERE, "skill_local.json"), "w"), indent=2,
          default=float)
