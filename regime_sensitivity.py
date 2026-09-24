"""
How much operational variation does the curve tolerate?

The turbofan result showed that regime switching corrupts the information curve
and that normalising within regime repairs about half of it.  That is one point
of a curve, measured on one fleet, with the regimes recovered by clustering.  The
general statement behind it has not been made: the framework assumes the noise is
the ONLY non-degradation variation in a record, so any systematic operational
variation is charged to degradation, and the size of that error is a function of
how large the operational steps are.

Here the whole function is measured, on synthetic records where the truth is
known exactly.  A record with a known trend is given regime steps of controlled
amplitude and rate; the earliest usable age is read; and the error against the
same record without steps is reported.  Sweeping the amplitude gives the
tolerance -- how large an operational step a record can carry before the age it
reports is wrong by more than a stated amount.

Two things are checked alongside.  Whether the corruption depends on how OFTEN
the regime changes as well as how far, since a step every few samples and a step
every hundred are different signals even at the same amplitude.  And whether
knowing the regimes repairs it completely, which on the turbofan could not be
established because there the regimes had to be recovered rather than given.
"""
import os
import json
import numpy as np

from nonparam import info_curve, quantile

HERE = os.path.dirname(os.path.abspath(__file__))
SEED = 20260828
QSTAR = 0.35
N = 1200
REPS = 30
AMPS = (0.0, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0)      # in units of the noise scale
RATES = (0.02, 0.10, 0.30)                        # switch probability per sample


def make(rng, amp, rate, sigma=0.02):
    """A known trend, additive noise, and piecewise-constant regime offsets."""
    tau = (np.arange(N) + 1.0) / N
    trend = tau ** 3
    noise = sigma * rng.normal(size=N)
    # a regime label that changes with probability `rate`, three regimes
    lab = np.empty(N, int)
    cur = 0
    for i in range(N):
        if rng.random() < rate:
            cur = int(rng.integers(0, 3))
        lab[i] = cur
    offs = rng.normal(size=3) * amp * sigma
    return trend + noise + offs[lab], trend + noise, lab, offs


def age(x, rng):
    F = info_curve(np.asarray(x, float), rng=rng)
    if F is None:
        return np.nan
    v = quantile(F, QSTAR)
    return float(v) if np.isfinite(v) else np.nan


rng = np.random.default_rng(SEED)
print("how far a regime step moves the earliest usable age\n")
print(f"trend tau^3, noise 0.02 of the trend's rise, {REPS} repetitions\n")
print(f"{'step / noise':>13}" + "".join(f"{f'rate {r}':>14}" for r in RATES))
print("-" * (13 + 14 * len(RATES)))
rows = []
for amp in AMPS:
    line = []
    for rate in RATES:
        errs = []
        for _ in range(REPS):
            x, clean, lab, offs = make(rng, amp, rate)
            a_dirty = age(x, rng)
            a_clean = age(clean, rng)
            if np.isfinite(a_dirty) and np.isfinite(a_clean):
                errs.append(a_dirty - a_clean)
        if not errs:
            line.append(np.nan)
            continue
        m = float(np.median(errs))
        ab = float(np.median(np.abs(errs)))
        rows.append(dict(amp=amp, rate=rate, bias=m, abs_err=ab,
                         reps=len(errs)))
        line.append(ab)
    print(f"{amp:>13.2f}" + "".join(f"{v:>14.3f}" for v in line))
print("-" * (13 + 14 * len(RATES)))
print("Entries are the median ABSOLUTE error in the age, in lifetimes,")
print("against the same record without steps.\n")

# --- the tolerance ----------------------------------------------------------
TOL = 0.05
print(f"the largest step a record tolerates before the age is wrong by "
      f"{TOL}\n")
print(f"{'rate':>8}{'tolerated step':>17}")
print("-" * 26)
tol_rows = []
for rate in RATES:
    r = sorted([x for x in rows if x["rate"] == rate], key=lambda x: x["amp"])
    # the largest amplitude that still PASSES, not the first that fails: a
    # first version printed the latter under this heading and overstated the
    # tolerance by one grid step
    ok = [x["amp"] for x in r if x["abs_err"] <= TOL]
    lim = max(ok) if ok else 0.0
    first_bad = min([x["amp"] for x in r if x["abs_err"] > TOL], default=None)
    tol_rows.append(dict(rate=rate, tolerated=lim, first_failing=first_bad))
    print(f"{rate:>8.2f}{f'{lim:.2f} x noise':>17}")
print("-" * 26)
print()

# --- does knowing the regimes repair it? ------------------------------------
print("with the regimes GIVEN, does normalising restore the age?\n")
print(f"{'step / noise':>13}{'uncorrected':>14}{'corrected':>12}"
      f"{'ratio':>9}")
print("-" * 50)
rep_rows = []
for amp in (1.0, 2.0, 4.0, 8.0):
    raw, fix = [], []
    for _ in range(REPS):
        x, clean, lab, offs = make(rng, amp, 0.10)
        # Normalising within regime, with the labels known exactly, and with
        # the baseline taken from the HEALTHY HEAD of each regime -- the same
        # correction applied to the turbofan.  Taking each regime's baseline
        # from the whole record instead removes part of the trend along with
        # the offset, and a first version of this test did exactly that: it
        # produced a constant error near 0.09 regardless of the step it was
        # correcting, and made the smallest steps worse than leaving them.
        head = slice(0, max(10, int(0.20 * N)))
        xc = x.copy()
        gbase = float(np.median(x[head]))
        for r in range(3):
            m = lab == r
            mh = m.copy()
            mh[head.stop:] = False
            if mh.sum() >= 5:
                xc[m] = x[m] - float(np.median(x[mh])) + gbase
        a_raw, a_fix, a_cl = age(x, rng), age(xc, rng), age(clean, rng)
        if np.isfinite(a_cl):
            if np.isfinite(a_raw):
                raw.append(abs(a_raw - a_cl))
            if np.isfinite(a_fix):
                fix.append(abs(a_fix - a_cl))
    if not raw or not fix:
        continue
    mr, mf = float(np.median(raw)), float(np.median(fix))
    rep_rows.append(dict(amp=amp, raw=mr, fixed=mf))
    print(f"{amp:>13.2f}{mr:>14.3f}{mf:>12.3f}{mf / max(mr, 1e-12):>9.2f}")
print("-" * 50)
print()

worst_fix = max((r["fixed"] for r in rep_rows), default=np.nan)
worst_raw = max((r["raw"] for r in rep_rows), default=np.nan)
if np.isfinite(worst_fix) and worst_fix < 0.02:
    print("With the regimes known the correction is essentially complete: the")
    print(f"residual error stays under {worst_fix:.3f} of a lifetime where the")
    print(f"uncorrected error reaches {worst_raw:.3f}.  On the turbofan the repair")
    print("was only partial, and the difference is that there the regimes had to")
    print("be recovered by clustering rather than given -- which locates the")
    print("remaining half of that shortfall in the regime assignment, not in the")
    print("method.")
elif np.isfinite(worst_fix):
    print(f"Even with the regimes known the residual reaches {worst_fix:.3f} of a")
    print(f"lifetime against {worst_raw:.3f} uncorrected, so normalising within")
    print("regime helps but does not restore the curve, and the turbofan's")
    print("partial repair is not explained by imperfect regime recovery alone.")

json.dump(dict(n=N, reps=REPS, amps=list(AMPS), rates=list(RATES),
               grid=rows, tolerance=TOL, limits=tol_rows, repair=rep_rows),
          open(os.path.join(HERE, "regime_sensitivity.json"), "w"), indent=2,
          default=float)
