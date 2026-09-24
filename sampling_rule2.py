"""
How much faster must a sensor sample to detect a given amount earlier? (corrected)

A first attempt decimated records and re-read tau at the 0.35 quantile of the
normalised information curve. It found no dependence on rate at all, and the
reason is instructive: a quantile of a NORMALISED distribution cannot depend on
the total information. Decimation lowers Gamma but leaves the shape of F nearly
alone, so the quantile does not move. The experiment was measuring shape and
calling it rate.

That also sharpens what the numbers reported throughout this study mean.
"tau at q = 0.35" is the age by which 35% of the budget has accumulated -- a
property of the shape. It coincides with the earliest age at which a precision
target is reachable only when q* = 1/(eps^2 Gamma) happens to equal 0.35. To ask
about acquisition rate the absolute form is needed:

    tau_min(eps) = the earliest tau with  G(tau) >= 1 / eps^2

with G unnormalised. Gamma is proportional to the sampling rate, so halving the
rate halves G everywhere and pushes tau_min later by an amount the curve fixes.

The precision target is stated in units of normalised life, so eps = 0.05 means
resolving the damage clock to five percent of a lifetime.
"""
import os
import json
import numpy as np
from nonparam import weighted_density, estimate_floor

HERE = os.path.dirname(os.path.abspath(__file__))
EPS = (0.02, 0.05)
THRESH = 0.90
STEPS = (1, 2, 3, 4, 6, 8)
SEED = 515151


def curves(series, rng):
    """Unnormalised G per unit, plus the signal fraction over the record."""
    out = []
    for raw in series:
        x = np.asarray(raw, float)
        if len(x) < 60:
            continue
        dens, res = weighted_density(x)
        if dens is None:
            continue
        fl = estimate_floor(dens, res, "surrogate", rng)
        d = np.clip(dens - fl, 0.0, None)
        if d.sum() <= 0:
            continue
        raw_tot = float(dens.sum())
        out.append((np.cumsum(d), float(d.sum()) / raw_tot if raw_tot else np.nan))
    return out


def tau_min(G, target):
    """Earliest tau at which the accumulated information reaches the target."""
    if G[-1] < target:
        return np.nan                      # unreachable on this record
    return (int(np.searchsorted(G, target)) + 1) / len(G)


rng = np.random.default_rng(SEED)
z = np.load(os.path.join(HERE, "feat_raw.npz"), allow_pickle=True)
FN = [str(s) for s in z["featnames"]]
I = {k: i for i, k in enumerate(FN)}
BK = sorted(k for k in z.files if k != "featnames")
CH = {"rms": ["rms"], "0--2 kHz": ["b0_1", "b1_2"],
      "2--4 kHz": ["b2_4"], "4--10 kHz": ["b4_6", "b6_8", "b8_10"]}

rows = []
for lab, parts in CH.items():
    base = [sum(z[b][:, I[p]] for p in parts) for b in BK]
    for st in STEPS:
        thin = [np.asarray(x, float)[::st] for x in base]
        cs = curves(thin, rng)
        if len(cs) < 3:
            continue
        sf = float(np.median([c[1] for c in cs]))
        for e in EPS:
            tgt = 1.0 / e ** 2
            v = [tau_min(c[0], tgt) for c in cs]
            ok = [q for q in v if np.isfinite(q)]
            rows.append(dict(channel=lab, step=st, rate=1.0 / st, eps=e,
                             signal=sf, units=len(cs), reached=len(ok),
                             tau=float(np.median(ok)) if ok else np.nan))

for e in EPS:
    print(f"\nprecision target eps = {e:.2f} of life  "
          f"(information needed: {1/e**2:.0f})\n")
    print(f"{'channel':<12}{'rate':>7}{'signal':>9}{'reached':>10}"
          f"{'tau_min':>10}")
    print("-" * 48)
    for lab in CH:
        for r in [x for x in rows if x["channel"] == lab and x["eps"] == e]:
            ts = "  never" if not np.isfinite(r["tau"]) else f"{r['tau']:>8.3f}"
            print(f"{lab:<12}{r['rate']:>7.2f}{100*r['signal']:>8.0f}%"
                  f"{r['reached']:>6}/{r['units']:<3}{ts}")
        print()
print("-" * 48)

print("\nthe design rule\n")
print(f"{'channel':<12}{'eps':>6}{'points':>8}{'slope':>9}"
      f"{'rate for 0.10 earlier':>23}")
print("-" * 58)
rule = []
for lab in CH:
    for e in EPS:
        v = [r for r in rows if r["channel"] == lab and r["eps"] == e
             and np.isfinite(r["tau"]) and r["signal"] >= THRESH]
        if len(v) < 3:
            continue
        lr = np.log([r["rate"] for r in v])
        lt = np.log([r["tau"] for r in v])
        slope = float(np.polyfit(lr, lt, 1)[0])
        base = [r["tau"] for r in v if r["step"] == 1]
        if base and slope < -1e-3 and base[0] > 0.10:
            factor = float(np.exp(np.log((base[0] - 0.10) / base[0]) / slope))
            fs = f"{factor:.1f}x"
        else:
            factor, fs = np.inf, "unreachable"
        rule.append(dict(channel=lab, eps=e, slope=slope, factor=factor,
                         points=len(v)))
        print(f"{lab:<12}{e:>6.2f}{len(v):>8}{slope:>9.3f}{fs:>23}")
print("-" * 58)
print("slope = d(log tau_min) / d(log rate). The Cramer-Rao form predicts a")
print("negative slope of magnitude set by how steeply information accumulates:")
print("an indicator whose budget arrives abruptly gains little from a faster")
print("sensor, because the information is not yet there to be collected.")
json.dump(dict(measurements=rows, rule=rule),
          open(os.path.join(HERE, "sampling_rule2.json"), "w"), indent=2,
          default=float)
