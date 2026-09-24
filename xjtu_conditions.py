"""
Is the earliest usable age a property of the failure mode, or of the load?

XJTU-SY runs three operating conditions -- 35 Hz / 12 kN, 37.5 Hz / 11 kN and
40 Hz / 10 kN -- five bearings each. If tau_min moves between them, the numbers
reported for bearings are condition-specific and must be qualified.

The comparison has a confound that has to be removed first. Record length varies
by a factor of sixty across these bearings, and not at random: the 35 Hz group
runs 52 to 161 acquisitions while the 40 Hz group runs 114 to 2538. Record
length is already known to affect what the measurement can see -- below roughly
300 samples the noise floor stops being separable from the density -- so a
difference between conditions could be length rather than load.

The control is decimation. The long condition is thinned to the length of the
short one, keeping its physical span, and re-measured. If its tau_min then moves
to meet the short condition's, the difference was length; if it stays put, the
difference is real and belongs to the load.
"""
import os
import json
import numpy as np
from nonparam import weighted_density, estimate_floor, quantile

HERE = os.path.dirname(os.path.abspath(__file__))
Q = 0.35
SEED = 33344


def profile(series, rng):
    T, S, N = [], [], []
    for raw in series:
        x = np.asarray(raw, float)
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
    return dict(units=len(T), n=int(np.median(N)),
                tau=float(np.median(T)), signal=float(np.median(S)))


z = np.load(os.path.join(HERE, "xjtu_feat.npz"), allow_pickle=True)
FN = [str(s) for s in z["featnames"]]
I = {k: i for i, k in enumerate(FN)}
BK = sorted(k for k in z.files if k != "featnames")
CONDS = sorted({k.split("__")[0] for k in BK})
CH = {"rms": ["rms"], "kurt": ["kurt"],
      "0--2 kHz": ["b0_1", "b1_2"], "4--10 kHz": ["b4_6", "b6_8", "b8_10"]}

rng = np.random.default_rng(SEED)
print("earliest usable age by operating condition\n")
print(f"{'condition':<14}{'channel':<12}{'units':>6}{'n':>7}{'signal':>9}"
      f"{'tau(0.35)':>11}")
print("-" * 59)
rows = []
for c in CONDS:
    keys = [k for k in BK if k.startswith(c + "__")]
    for lab, parts in CH.items():
        s = [sum(z[k][:, I[p]] for p in parts) for k in keys]
        p = profile(s, rng)
        if not p:
            continue
        rows.append(dict(cond=c, channel=lab, **p))
        print(f"{c:<14}{lab:<12}{p['units']:>6}{p['n']:>7}"
              f"{100*p['signal']:>8.0f}%{p['tau']:>11.3f}")
    print()
print("-" * 59)

print("\nspread across conditions, per channel\n")
print(f"{'channel':<12}{'lowest':>9}{'highest':>9}{'spread':>9}   conditions")
print("-" * 60)
for lab in CH:
    v = [r for r in rows if r["channel"] == lab]
    if len(v) < 2:
        continue
    lo = min(v, key=lambda r: r["tau"]); hi = max(v, key=lambda r: r["tau"])
    print(f"{lab:<12}{lo['tau']:>9.3f}{hi['tau']:>9.3f}"
          f"{hi['tau']-lo['tau']:>9.3f}   {lo['cond']} -> {hi['cond']}")
print("-" * 60)

# --- the control: is it load, or record length? -----------------------------
short = min(CONDS, key=lambda c: np.median(
    [len(z[k]) for k in BK if k.startswith(c + "__")]))
long_ = max(CONDS, key=lambda c: np.median(
    [len(z[k]) for k in BK if k.startswith(c + "__")]))
n_short = int(np.median([len(z[k]) for k in BK if k.startswith(short + "__")]))
print(f"\ncontrol: thinning {long_} to the length of {short} "
      f"(n ~ {n_short})\n")
print(f"{'channel':<12}{'short cond':>12}{'long cond':>11}"
      f"{'long, thinned':>15}")
print("-" * 50)
ctrl = []
for lab, parts in CH.items():
    ks_s = [k for k in BK if k.startswith(short + "__")]
    ks_l = [k for k in BK if k.startswith(long_ + "__")]
    a = profile([sum(z[k][:, I[p]] for p in parts) for k in ks_s], rng)
    b = profile([sum(z[k][:, I[p]] for p in parts) for k in ks_l], rng)
    thin = []
    for k in ks_l:
        x = sum(z[k][:, I[p]] for p in parts)
        step = max(1, int(round(len(x) / n_short)))
        thin.append(np.asarray(x, float)[::step])
    c = profile(thin, rng)
    if not (a and b and c):
        continue
    ctrl.append(dict(channel=lab, short=a["tau"], long=b["tau"],
                     thinned=c["tau"]))
    print(f"{lab:<12}{a['tau']:>12.3f}{b['tau']:>11.3f}{c['tau']:>15.3f}")
print("-" * 50)
if ctrl:
    moved = float(np.median([abs(r["thinned"] - r["long"]) for r in ctrl]))
    gap = float(np.median([abs(r["long"] - r["short"]) for r in ctrl]))
    closed = float(np.median([abs(r["thinned"] - r["short"]) for r in ctrl]))
    print(f"gap between conditions before thinning : {gap:.3f} of life")
    print(f"gap after thinning the long condition  : {closed:.3f}")
    print(f"how far thinning moved it              : {moved:.3f}")
    print()
    if closed < 0.5 * gap:
        print("Thinning closes most of the gap, so the difference between")
        print("conditions is record length and not load: tau_min is a property")
        print("of the failure mode, measured better on longer records.")
    else:
        print("Thinning does not close the gap, so the difference survives the")
        print("length confound and belongs to the operating condition. The")
        print("bearing numbers must then be quoted per condition.")
json.dump(dict(by_condition=rows, control=ctrl),
          open(os.path.join(HERE, "xjtu_conditions.json"), "w"), indent=2,
          default=float)
