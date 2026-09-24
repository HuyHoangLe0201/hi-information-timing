r"""Which clock variations survive the smoother: the estimand made precise.

Propositions 27 to 30 say what a nonparametric trend costs.  All of them treat a
CONSTANT offset, and all of them return essentially zero.  The paper does not
claim a constant offset: Section 12 says its estimand is local, and nothing has
said what "local" picks out.

The structure says.  A clock that varies with age perturbs the trend by
delta(tau) D'(tau), and the smoother absorbs whatever looks like a trend on its
own scale.  A constant delta produces a perturbation as smooth as D' itself and
is absorbed, which is Proposition 27.  A delta that varies faster than the
smoother's window produces something the smoother cannot follow, and that should
survive.  So the estimand is not a point but a BAND, with a threshold set by the
window, and the threshold is measurable.

The experiment is a transfer function.  delta(tau) = A sin(2 pi f tau) is applied
for a range of f, the offset amplitude is estimated against the pipeline's
out-of-fold trend, and the realised information is read as sensitivity squared
over variance, the measure Proposition 27 had to adopt.  A threshold near
f = 1/h, with h the window as a fraction of life, would say the estimand is
exactly the clock variation faster than the smoother.
"""
import os
import json
import numpy as np

from pipeline import oof_trend

HERE = os.path.dirname(os.path.abspath(__file__))
rng = np.random.default_rng(31415)
OUT = {}
N, BETA, SIG = 600, 3.0, 0.02
REPS = 120
FRAC = 0.08
W = max(7, int(FRAC * N))
W += (W % 2 == 0)
t = (np.arange(N) + 1.0) / N
Dp = BETA * np.maximum(t, 1e-12) ** (BETA - 1)


def record(amp, f, rg):
    """A record whose clock offset varies as amp * sin(2 pi f tau)."""
    dl = amp * np.sin(2 * np.pi * f * t)
    return np.maximum(t + dl, 1e-12) ** BETA + rg.normal(0, SIG, N)


def estimate(y, trend, patt):
    """Amplitude of the assumed pattern, by least squares on the residual."""
    r = y - trend
    g = patt * Dp                      # the perturbation the pattern produces
    den = float(np.sum(g * g))
    return float(np.sum(g * r) / den) if den > 0 else np.nan


print("=" * 92)
print("1.  The transfer function of the clock through the smoother")
print("=" * 92)
print("""
Each row applies a clock varying at f cycles per lifetime and estimates its
amplitude against the out-of-fold trend.  The floor is the same estimate against
the true trend.  Efficiency is the realised information over that floor, and the
window is %.2f of a lifetime, so the smoother's own scale is f = %.1f.
""" % (FRAC, 1.0 / FRAC))
print("  %-10s %12s %14s %14s %12s"
      % ("f", "slope", "sd", "efficiency", "verdict"))
print("  " + "-" * 66)
rows = []
AMPS = (-0.01, 0.01)
for f in (0.0, 1.0, 3.0, 6.0, 12.0, 25.0, 50.0):
    patt = np.sin(2 * np.pi * f * t) if f > 0 else np.ones(N)
    got = {"true": {}, "oof": {}}
    for a in AMPS:
        acc = {"true": [], "oof": []}
        for _ in range(REPS):
            y = record(a, f, rng)
            acc["true"].append(estimate(y, t ** BETA, patt))
            acc["oof"].append(estimate(y, oof_trend(y, W // 2), patt))
        for k in acc:
            got[k][a] = (float(np.mean(acc[k])), float(np.std(acc[k])))
    res = {}
    for k in got:
        sl = (got[k][AMPS[1]][0] - got[k][AMPS[0]][0]) / (AMPS[1] - AMPS[0])
        sd = float(np.mean([got[k][a][1] for a in AMPS]))
        res[k] = (sl, sd, (sl / sd) ** 2 if sd > 0 else np.nan)
    eff = res["oof"][2] / res["true"][2] if res["true"][2] > 0 else np.nan
    rows.append(dict(f=f, slope=res["oof"][0], sd=res["oof"][1],
                     eff=float(eff)))
    print("  %-10.1f %12.4f %14.6f %14.4f %12s"
          % (f, res["oof"][0], res["oof"][1], eff,
             "absorbed" if eff < 0.1 else ("partial" if eff < 0.7 else "kept")))
OUT["transfer"] = rows

if rows:
    OUT["eff_at_zero"] = float(rows[0]["eff"])
    OUT["eff_at_top"] = float(rows[-1]["eff"])
    # The crossing must be interpolated, not read off the grid.  Taking the
    # first grid point above one half gives 12, which happens to sit beside the
    # window scale of 12.5 and would overstate the agreement by a coincidence of
    # spacing; between f = 6 and f = 12 the half point is nearer 10.
    _cross = None
    for i in range(len(rows) - 1):
        a, b = rows[i], rows[i + 1]
        if a["eff"] < 0.5 <= b["eff"] and b["eff"] > a["eff"]:
            _cross = a["f"] + (b["f"] - a["f"]) * (0.5 - a["eff"]) / (
                b["eff"] - a["eff"])
            break
    OUT["crossing"] = None if _cross is None else float(_cross)
    OUT["window_scale"] = 1.0 / FRAC
    OUT["crossing_over_scale"] = (None if _cross is None
                                  else float(_cross / OUT["window_scale"]))
    print("""
  A constant clock is absorbed, efficiency %.4f, which is Proposition 27.  By
  f = %.0f the efficiency is %.3f and by %.0f it is %.3f: the clock is recovered
  in full once it varies fast enough.

  The half-efficiency point, interpolated between the grid points that bracket
  it, is f = %.1f cycles per lifetime against the smoother's own scale of %.1f, a
  ratio of %.2f.  The estimand is therefore a BAND and the threshold is the
  window: clock variation faster than the smoother is identifiable and slower is
  not.  That is what "local" has meant throughout, stated as a number.
""" % (OUT["eff_at_zero"], rows[-2]["f"], rows[-2]["eff"],
       rows[-1]["f"], rows[-1]["eff"],
       _cross if _cross else float("nan"), OUT["window_scale"],
       OUT["crossing_over_scale"] if _cross else float("nan")))

json.dump(OUT, open(os.path.join(HERE, "clock_band.json"), "w"), indent=2,
          default=float)
print("written to clock_band.json")
