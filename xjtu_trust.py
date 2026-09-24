"""
The second bearing rig: does the picture hold on XJTU-SY?

Everything reported for bearings so far comes from PRONOSTIA. That is one test
rig, one loading arrangement and one sampling regime, so a finding that
root-mean-square amplitude is unusable until the last few percent of life while
the high-frequency bands are usable early could be a property of that rig rather
than of rolling-element degradation.

XJTU-SY is a separate rig with fifteen bearings under three operating
conditions. The same measurement is applied unchanged -- weighted density, noise
floor subtracted, quantiles read off the curve -- and the same trust threshold on
the signal fraction. Agreement would make the ordering a property of the failure
mode; disagreement would confine every bearing number in this study to
PRONOSTIA.
"""
import os
import json
import numpy as np
from nonparam import weighted_density, estimate_floor, quantile

HERE = os.path.dirname(os.path.abspath(__file__))
QS = (0.10, 0.35)
THRESH = 0.90
SEED = 20260828


def profile(series, rng):
    Q = {q: [] for q in QS}
    SF, NS = [], []
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
        for q in QS:
            Q[q].append(quantile(F, q))
        k = int(round(quantile(F, 0.35) * len(d))) - 1
        k = max(1, min(k, len(d) - 1))
        rk = float(dens[:k + 1].sum())
        if rk > 0:
            SF.append(float(d[:k + 1].sum()) / rk)
        NS.append(len(d))
    if not SF or not NS:
        return None
    return dict(units=len(NS), n=int(np.median(NS)),
                signal=float(np.median(SF)),
                **{f"tau{q:.2f}": float(np.median(Q[q])) for q in QS})


z = np.load(os.path.join(HERE, "xjtu_feat.npz"), allow_pickle=True)
FN = [str(s) for s in z["featnames"]]
I = {k: i for i, k in enumerate(FN)}
BK = sorted(k for k in z.files if k != "featnames")
print(f"XJTU-SY: {len(BK)} bearings, channels available: {', '.join(FN)}\n")

# match the PRONOSTIA channel set as closely as the feature names allow
WANT = [("rms", ["rms"]), ("peak", ["peak"]), ("kurt", ["kurt"])]
bands = [f for f in FN if f.startswith("b")]
if bands:
    lo = [f for f in bands if f in ("b0_1", "b1_2")]
    hi = [f for f in bands if f in ("b4_6", "b6_8", "b8_10")]
    if lo:
        WANT.append(("0--2 kHz", lo))
    if hi:
        WANT.append(("4--10 kHz", hi))
    for f in bands:
        WANT.append((f, [f]))

rng = np.random.default_rng(SEED)
rows = []
for lab, parts in WANT:
    if not all(p in I for p in parts):
        continue
    series = [sum(z[b][:, I[p]] for p in parts) for b in BK]
    p = profile(series, rng)
    if p:
        rows.append(dict(indicator=lab, **p))

if not rows:
    raise SystemExit("no XJTU channel could be profiled")

rows.sort(key=lambda r: r["tau0.35"])
print(f"{'indicator':<14}{'units':>6}{'n':>6}{'signal':>9}{'tau(0.10)':>11}"
      f"{'tau(0.35)':>11}   verdict")
print("-" * 72)
for r in rows:
    ok = r["signal"] >= THRESH
    print(f"{r['indicator']:<14}{r['units']:>6}{r['n']:>6}{100*r['signal']:>8.0f}%"
          f"{r['tau0.10']:>11.3f}{r['tau0.35']:>11.3f}   "
          f"{'trustworthy' if ok else 'floor optimistic'}")
print("-" * 72)

PRON = {r["indicator"].replace("bearing ", ""): r
        for r in json.load(open(os.path.join(HERE, "trust_table.json")))
        if r["indicator"].startswith("bearing")}
common = [r for r in rows if r["indicator"] in PRON]
if common:
    print("\nagainst PRONOSTIA, on the channels both rigs carry\n")
    print(f"{'indicator':<14}{'XJTU tau':>10}{'PRONOSTIA':>11}{'difference':>12}")
    print("-" * 47)
    for r in common:
        p = PRON[r["indicator"]]["tau_0.35"]
        print(f"{r['indicator']:<14}{r['tau0.35']:>10.3f}{p:>11.3f}"
              f"{r['tau0.35']-p:>+12.3f}")
    print("-" * 47)
    a = [r["tau0.35"] for r in common]
    b = [PRON[r["indicator"]]["tau_0.35"] for r in common]
    if len(a) >= 3:
        ra = np.argsort(np.argsort(a)); rb = np.argsort(np.argsort(b))
        print(f"rank agreement across rigs: "
              f"{float(np.corrcoef(ra, rb)[0, 1]):+.2f}")
    print(f"median absolute difference: "
          f"{float(np.median(np.abs(np.array(a) - np.array(b)))):.3f} of life")
    print("\nAgreement in ordering would make the finding a property of how")
    print("rolling-element bearings fail; disagreement would confine it to one rig.")
json.dump(rows, open(os.path.join(HERE, "xjtu_trust.json"), "w"), indent=2,
          default=float)
