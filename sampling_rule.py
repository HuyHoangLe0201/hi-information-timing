"""
How much faster must a sensor sample to detect a given amount earlier?

The budget carries the acquisition rate explicitly, G = (f_s / sigma^2) times the
integrated squared derivative, so doubling the rate doubles the information and
moves the earliest usable age earlier by an amount the curve determines. That is
the one question an instrumentation engineer actually asks, and the framework
answers it without a model.

The rule is measured rather than derived, because deriving it needs the
power-law form that was dropped. Records are decimated -- the physical span
kept, the sampling thinned, which is exactly what a slower sensor would give --
and tau_min re-read at each rate.

One confound is built in. Decimation lowers the rate AND shortens the record,
and a short record cannot separate the density from the noise floor: below about
90% signal fraction the floor is optimistic by two to six times. The fraction is
tracked at every rate and the rule is fitted only where it holds, so the answer
is not contaminated by the measurement failing.
"""
import os
import json
import numpy as np
from nonparam import weighted_density, estimate_floor, quantile

HERE = os.path.dirname(os.path.abspath(__file__))
Q = 0.35
THRESH = 0.90
STEPS = (1, 2, 3, 4, 6, 8)
SEED = 70707


def profile(series, rng):
    T, S, N = [], [], []
    for raw in series:
        x = np.asarray(raw, float)
        if len(x) < 60:
            continue
        dens, res = weighted_density(x)
        if dens is None:
            continue
        d = np.clip(dens - estimate_floor(dens, res, "surrogate", rng), 0.0, None)
        tot = float(d.sum())
        if tot <= 0:
            continue
        F = np.cumsum(d) / tot
        t = quantile(F, Q)
        k = max(1, min(int(round(t * len(d))) - 1, len(d) - 1))
        rk = float(dens[:k + 1].sum())
        if rk <= 0:
            continue
        T.append(t); S.append(float(d[:k + 1].sum()) / rk); N.append(len(d))
    if len(T) < 3:
        return None
    return (float(np.median(T)), float(np.median(S)), int(np.median(N)),
            len(T))


rng = np.random.default_rng(SEED)
z = np.load(os.path.join(HERE, "feat_raw.npz"), allow_pickle=True)
FN = [str(s) for s in z["featnames"]]
I = {k: i for i, k in enumerate(FN)}
BK = sorted(k for k in z.files if k != "featnames")
CH = {"rms": ["rms"], "0--2 kHz": ["b0_1", "b1_2"],
      "2--4 kHz": ["b2_4"], "4--10 kHz": ["b4_6", "b6_8", "b8_10"]}

print("earliest usable age against acquisition rate\n")
print(f"{'channel':<12}{'rate':>7}{'n':>7}{'signal':>9}{'tau(0.35)':>11}"
      f"{'usable':>9}")
print("-" * 55)
rows = []
for lab, parts in CH.items():
    base = [sum(z[b][:, I[p]] for p in parts) for b in BK]
    for st in STEPS:
        thin = [np.asarray(x, float)[::st] for x in base]
        p = profile(thin, rng)
        if not p:
            continue
        t, s, n, u = p
        ok = s >= THRESH
        rows.append(dict(channel=lab, step=st, rate=1.0 / st, n=n,
                         signal=s, tau=t, usable=bool(ok), units=u))
        print(f"{lab:<12}{1.0/st:>7.2f}{n:>7}{100*s:>8.0f}%{t:>11.3f}"
              f"{'yes' if ok else 'no':>9}")
    print()
print("-" * 55)
print("rate is relative to the original acquisition; 0.50 means every second")
print("acquisition kept, which is what half the sampling rate would give.\n")

print("the design rule, fitted where the measurement holds\n")
print(f"{'channel':<12}{'points':>8}{'slope':>9}{'to gain 0.10 of life':>22}")
print("-" * 51)
out = []
for lab in CH:
    v = [r for r in rows if r["channel"] == lab and r["usable"]]
    if len(v) < 3:
        print(f"{lab:<12}{len(v):>8}   too few rates with a usable measurement")
        continue
    lr = np.log([r["rate"] for r in v])
    lt = np.log([r["tau"] for r in v])
    slope = float(np.polyfit(lr, lt, 1)[0])
    # tau ~ rate^slope, so gaining d in log-tau needs rate^(d/slope)
    base_t = [r["tau"] for r in v if r["step"] == 1]
    if not base_t or slope >= -1e-6:
        factor = np.inf
    else:
        target = base_t[0] - 0.10
        factor = (np.exp(np.log(max(target, 1e-6) / base_t[0]) / slope)
                  if target > 0 else np.inf)
    out.append(dict(channel=lab, slope=slope, factor=float(factor),
                    n_points=len(v)))
    fs = "unreachable" if not np.isfinite(factor) else f"{factor:>6.1f}x faster"
    print(f"{lab:<12}{len(v):>8}{slope:>9.3f}{fs:>22}")
print("-" * 51)
print("slope is d(log tau_min) / d(log rate): more negative means sampling")
print("faster buys more. A slope near zero means it buys nothing, and no")
print("achievable rate brings that indicator forward.\n")
steep = [r for r in out if r["slope"] > -0.05]
if steep:
    for r in steep:
        print(f"  {r['channel']}: slope {r['slope']:+.3f} -- sampling faster")
        print(f"    does not move this indicator; its information is not")
        print(f"    there to be collected, at any rate.")
json.dump(dict(measurements=rows, rule=out),
          open(os.path.join(HERE, "sampling_rule.json"), "w"), indent=2,
          default=float)
