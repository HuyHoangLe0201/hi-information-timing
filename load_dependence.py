"""
Does the earliest usable age depend on how hard the bearing is worked?

XJTU alone gave three operating conditions and a spread of up to 0.35 of life,
which was too few points to say anything. PRONOSTIA also has three conditions --
its bearings are numbered by them and the split had simply never been made -- so
six conditions and thirty-two bearings are available, spanning 1500 to 2400 rpm
and 4 to 12 kN.

One limitation has to be stated rather than fitted around. In both rigs the
heavier load is paired with the lower speed, so load and speed move together and
cannot be separated: any relationship found is with the combination.

What can be separated is the consequence. Working a bearing harder shortens its
life, and life is measured directly, so the testable question is whether
tau_min tracks the mean life of the condition. If short-lived conditions become
usable later in relative terms, damage in them is concentrated closer to failure
-- a statement about how the damage progresses, not about the sensor.

PRONOSTIA conditions: 1800 rpm / 4000 N, 1650 / 4200, 1500 / 5000.
XJTU conditions: 2100 rpm / 12 kN, 2250 / 11 kN, 2400 / 10 kN.
"""
import os
import json
import numpy as np
from nonparam import weighted_density, estimate_floor, quantile

HERE = os.path.dirname(os.path.abspath(__file__))
Q = 0.35
SEED = 987654

COND = {
    ("PRONOSTIA", "1"): dict(rpm=1800, load_N=4000),
    ("PRONOSTIA", "2"): dict(rpm=1650, load_N=4200),
    ("PRONOSTIA", "3"): dict(rpm=1500, load_N=5000),
    ("XJTU", "35Hz12kN"): dict(rpm=2100, load_N=12000),
    ("XJTU", "37.5Hz11kN"): dict(rpm=2250, load_N=11000),
    ("XJTU", "40Hz10kN"): dict(rpm=2400, load_N=10000),
}


def profile(series, rng):
    T, S = [], []
    for raw in series:
        x = np.asarray(raw, float)
        dens, res = weighted_density(x)
        if dens is None:
            continue
        d = np.clip(dens - estimate_floor(dens, res, "surrogate", rng), 0.0, None)
        if d.sum() <= 0:
            continue
        F = np.cumsum(d) / d.sum()
        t = quantile(F, Q)
        k = max(1, min(int(round(t * len(d))) - 1, len(d) - 1))
        rk = float(dens[:k + 1].sum())
        if rk <= 0:
            continue
        T.append(t); S.append(float(d[:k + 1].sum()) / rk)
    if len(T) < 3:
        return None
    return float(np.median(T)), float(np.median(S))


rng = np.random.default_rng(SEED)
CH = {"rms": ["rms"], "0--2 kHz": ["b0_1", "b1_2"],
      "4--10 kHz": ["b4_6", "b6_8", "b8_10"]}
rows = []

zp = np.load(os.path.join(HERE, "feat_raw.npz"), allow_pickle=True)
FP = [str(s) for s in zp["featnames"]]
IP = {k: i for i, k in enumerate(FP)}
BP = sorted(k for k in zp.files if k != "featnames")
for c in ("1", "2", "3"):
    keys = [b for b in BP if b.startswith(f"Bearing{c}_")]
    lives = [len(zp[b]) for b in keys]
    for lab, parts in CH.items():
        p = profile([sum(zp[b][:, IP[q]] for q in parts) for b in keys], rng)
        if p:
            rows.append(dict(rig="PRONOSTIA", cond=c, channel=lab,
                             units=len(keys), mean_life=float(np.mean(lives)),
                             tau=p[0], signal=p[1], **COND[("PRONOSTIA", c)]))

zx = np.load(os.path.join(HERE, "xjtu_feat.npz"), allow_pickle=True)
FX = [str(s) for s in zx["featnames"]]
IX = {k: i for i, k in enumerate(FX)}
BX = sorted(k for k in zx.files if k != "featnames")
for c in sorted({k.split("__")[0] for k in BX}):
    keys = [b for b in BX if b.startswith(c + "__")]
    lives = [len(zx[b]) for b in keys]
    for lab, parts in CH.items():
        if not all(q in IX for q in parts):
            continue
        p = profile([sum(zx[b][:, IX[q]] for q in parts) for b in keys], rng)
        if p:
            rows.append(dict(rig="XJTU", cond=c, channel=lab, units=len(keys),
                             mean_life=float(np.mean(lives)), tau=p[0],
                             signal=p[1], **COND[("XJTU", c)]))

print("six operating conditions across two rigs\n")
print(f"{'rig':<11}{'condition':<12}{'rpm':>6}{'load N':>8}{'units':>6}"
      f"{'mean life':>11}{'channel':<12}{'tau':>8}")
print("-" * 76)
for r in sorted(rows, key=lambda r: (r["rig"], r["cond"], r["channel"])):
    print(f"{r['rig']:<11}{r['cond']:<12}{r['rpm']:>6}{r['load_N']:>8}"
          f"{r['units']:>6}{r['mean_life']:>11.0f}  {r['channel']:<12}"
          f"{r['tau']:>8.3f}")
print("-" * 76)


def rank_corr(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    if len(a) < 3:
        return np.nan
    return float(np.corrcoef(np.argsort(np.argsort(a)),
                             np.argsort(np.argsort(b)))[0, 1])


print("\ncorrelation of the earliest usable age with the condition\n")
print(f"{'channel':<12}{'scope':<14}{'vs load':>9}{'vs rpm':>9}"
      f"{'vs mean life':>14}{'n':>4}")
print("-" * 62)
for lab in CH:
    for scope in ("PRONOSTIA", "XJTU", "both"):
        v = [r for r in rows if r["channel"] == lab
             and (scope == "both" or r["rig"] == scope)]
        if len(v) < 3:
            continue
        print(f"{lab:<12}{scope:<14}"
              f"{rank_corr([r['load_N'] for r in v], [r['tau'] for r in v]):>9.2f}"
              f"{rank_corr([r['rpm'] for r in v], [r['tau'] for r in v]):>9.2f}"
              f"{rank_corr([r['mean_life'] for r in v], [r['tau'] for r in v]):>14.2f}"
              f"{len(v):>4}")
    print()
print("-" * 62)
print("With three conditions per rig, a rank correlation is +1, 0 or -1 and")
print("carries almost no evidence on its own; the six-point 'both' rows are")
print("the ones worth reading, and they mix two rigs whose absolute loads")
print("differ by a factor of three.")

both = {}
for lab in CH:
    v = [r for r in rows if r["channel"] == lab]
    if len(v) >= 6:
        both[lab] = dict(
            load=rank_corr([r["load_N"] for r in v], [r["tau"] for r in v]),
            life=rank_corr([r["mean_life"] for r in v], [r["tau"] for r in v]))
if both:
    print("\nacross all six conditions:")
    for lab, d in both.items():
        print(f"  {lab:<12} vs load {d['load']:+.2f}   vs mean life {d['life']:+.2f}")
    cons = all(abs(d["load"]) > 0.6 for d in both.values())
    same = len({np.sign(d["load"]) for d in both.values()}) == 1
    if cons and same:
        print("\nAll channels agree in sign and strength, so the effect is a")
        print("property of the condition rather than of a particular channel.")
    else:
        print("\nThe channels disagree in sign or strength, so no consistent")
        print("dependence on the operating condition is established.")
json.dump(rows, open(os.path.join(HERE, "load_dependence.json"), "w"),
          indent=2, default=float)
