r"""What a fleet recovers that one record cannot, and what it never recovers.

Proposition 27 is a negative: with the trend learned from the same record, a
clock offset is unidentifiable.  The setting the paper is written for is not one
record but a fleet, and there the trend can be pooled, so the negative should
have a resolution.  It has one, and the resolution is partial in a way that can
be stated exactly.

With N units sharing a trend shape and carrying their own clocks,

    y_ij = D(tau_j + delta_i) + eps_ij,

the likelihood is invariant under

    D(.) -> D(. + c),      delta_i -> delta_i - c

for any c.  That is one direction of the parameter space and no data can move
along it.  So of the N offsets exactly N - 1 combinations are identifiable, the
DIFFERENCES, and one is not, their common level.  A fleet resolves relative
clocks and never resolves the absolute one, however large it is.

Two things follow that matter for the paper.  Section 8's fleet prior is not a
convenience: it supplies exactly the one direction the data cannot, which is why
Proposition 4 buys what it buys.  And the information about a difference should
grow with the fleet, since the shared trend is better determined, so there is a
number to measure: how many units before a relative clock is worth estimating.
"""
import os
import json
import numpy as np
from scipy.signal import savgol_filter

HERE = os.path.dirname(os.path.abspath(__file__))
rng = np.random.default_rng(606060)
OUT = {}


def fleet(N, n, beta, sig, deltas, rg):
    """N units on a common grid, each with its own clock offset."""
    t = (np.arange(n) + 1.0) / n
    Y = np.empty((N, n))
    for i in range(N):
        Y[i] = np.maximum(t + deltas[i], 1e-12) ** beta + rg.normal(0, sig, n)
    return t, Y


def pooled_estimate(t, Y, beta, w):
    """Estimate offsets against a trend pooled across the fleet.

    The pooled trend is the smoothed mean record, which is the natural
    nonparametric estimate when the shape is shared.  Offsets are then read by
    the same linear rule as elsewhere.
    """
    n = Y.shape[1]
    Dp = beta * np.maximum(t, 1e-12) ** (beta - 1)
    trend = savgol_filter(Y.mean(axis=0), w, 2)
    den = float(np.sum(Dp * Dp))
    return np.array([float(np.sum(Dp * (Y[i] - trend)) / den)
                     for i in range(Y.shape[0])])


print("=" * 92)
print("1.  The common level is unidentifiable; the differences are not")
print("=" * 92)
print("""
Two fleets are simulated from the same offsets shifted by a constant.  If the
common level were identifiable the estimates would move with it; if only
differences are, the estimates move by the same constant and their differences do
not move at all.
""")
N, n, beta, sig = 8, 500, 3.0, 0.02
w = max(7, int(0.08 * n))
w += (w % 2 == 0)
base = rng.normal(0, 0.02, N)
print("  %-24s %14s %14s" % ("quantity", "shift c = 0", "shift c = 0.03"))
print("  " + "-" * 56)
est = {}
for c in (0.0, 0.03):
    acc = []
    for _ in range(200):
        t, Y = fleet(N, n, beta, sig, base + c, rng)
        acc.append(pooled_estimate(t, Y, beta, w))
    est[c] = np.mean(np.array(acc), axis=0)
lvl0, lvl1 = float(np.mean(est[0.0])), float(np.mean(est[0.03]))
d0 = est[0.0] - np.mean(est[0.0])
d1 = est[0.03] - np.mean(est[0.03])
OUT["level_shift"] = float(lvl1 - lvl0)
OUT["difference_shift"] = float(np.max(np.abs(d1 - d0)))
print("  %-24s %14.5f %14.5f" % ("mean of the estimates", lvl0, lvl1))
print("  %-24s %14.5f %14.5f"
      % ("largest difference", float(np.max(np.abs(d0))),
         float(np.max(np.abs(d1)))))
print("""
  Shifting every clock by %.3f moves the mean estimate by %.5f, that is by
  nothing: the common level is invisible.  The differences move by at most
  %.5f, so they are what the data sees.  The invariance is exact and no amount of
  data touches it.
""" % (0.03, OUT["level_shift"], OUT["difference_shift"]))

# =============================================================================
print("=" * 92)
print("2.  How the information about a difference grows with the fleet")
print("=" * 92)
print("""
The trend is pooled, so a larger fleet determines it better and leaves more of
each unit's residual to carry its own clock.  The floor is the case of a known
trend, where the fleet buys nothing.
""")
print("  %-8s %14s %14s %12s"
      % ("N", "sd of a diff", "floor", "efficiency"))
print("  " + "-" * 54)
rows = []
Dp = None
for Nf in (1, 2, 4, 8, 16, 32):
    t = (np.arange(n) + 1.0) / n
    Dp = beta * np.maximum(t, 1e-12) ** (beta - 1)
    # With one unit there is no difference to take, so the quantity is the
    # offset itself against the one-unit floor.  That is exactly Proposition 27's
    # setting and belongs in the same column as the row that follows it.
    one = (Nf == 1)
    floor = sig / np.sqrt(float(np.sum(Dp * Dp))) * (1.0 if one else np.sqrt(2.0))
    diffs = []
    for _ in range(300):
        dl = rng.normal(0, 0.02, Nf)
        t, Y = fleet(Nf, n, beta, sig, dl, rng)
        e = pooled_estimate(t, Y, beta, w)
        diffs.append((e[0] - dl[0]) if one
                     else (e[0] - e[1]) - (dl[0] - dl[1]))
    sd = float(np.std(diffs))
    rows.append(dict(N=Nf, sd=sd, floor=float(floor),
                     efficiency=float((floor / sd) ** 2) if sd > 0 else np.nan))
    print("  %-8d %14.6f %14.6f %12.4f"
          % (Nf, sd, floor, rows[-1]["efficiency"]))
OUT["growth"] = rows
if rows:
    _by = {r["N"]: r["efficiency"] for r in rows}
    OUT["eff_at_1"] = _by.get(1)
    OUT["eff_at_2"] = _by.get(2)
    OUT["eff_at_32"] = _by.get(32)
    _multi = [r["efficiency"] for r in rows if r["N"] >= 2]
    OUT["eff_spread_beyond_two"] = float(max(_multi) / max(min(_multi), 1e-300))
    print("""
  The fleet does not buy what was expected.  One unit gives %.4f, which is
  Proposition 27 again: with the trend taken from the record itself the clock is
  gone.  Two units give %.4f.  Beyond two, nothing: the efficiency across four,
  eight, sixteen and thirty-two units spans a factor of only %.2f and does not
  trend.

  So the resolution is a step and not a curve, and the reason is visible in what
  each part of the loss depends on.  Pooling removes the invariance, which needs
  a second unit and only a second.  What remains is the smoother absorbing each
  unit's own derivative INSIDE its own record, which Proposition 28 measures and
  which no number of other units can touch.  A fleet buys identifiability; it does
  not buy precision.
""" % (OUT["eff_at_1"], OUT["eff_at_2"], OUT["eff_spread_beyond_two"]))

json.dump(OUT, open(os.path.join(HERE, "fleet_identifiability.json"), "w"),
          indent=2, default=float)
print("written to fleet_identifiability.json")
