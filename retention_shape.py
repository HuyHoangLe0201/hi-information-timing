"""
Why does a trailing window fail on bearings but not on turbofans?

A power-law density rises monotonically, so its top-w level set is one interval
sitting at the end -- exactly what FIFO keeps. If bearings really had beta ~ 3,
FIFO would be near-optimal there too. It is not, so the density must not be
rising smoothly. This measures the shape directly: how many disjoint pieces the
optimal set breaks into, and how far back the information actually sits.
"""
import os, json
import numpy as np
from pipeline import profile

HERE = os.path.dirname(os.path.abspath(__file__))
TAU0, W = 0.7, 0.20


def shape(dens, k0, kw):
    d = dens[:k0 + 1]
    idx = np.sort(np.argsort(d)[::-1][:kw])
    runs = 1 + int(np.sum(np.diff(idx) > 1))
    frac = idx / (k0 + 1)
    return dict(runs=runs, med=float(np.median(frac)), earliest=float(frac.min()),
                gini=float(1 - (d.sum() ** 2) / (len(d) * np.sum(d ** 2))))


def collect(series, label):
    out = []
    for raw in series:
        u, _ = profile(raw)
        if not u:
            continue
        n = u["n"]; k0 = int(round(TAU0 * n)) - 1; kw = max(3, int(round(W * n)))
        if k0 < kw + 3:
            continue
        out.append(shape(u["dens"], k0, kw))
    if not out:
        return None
    agg = {k: float(np.median([o[k] for o in out])) for k in out[0]}
    agg["label"] = label; agg["units"] = len(out)
    return agg


targets = []
fz = np.load(os.path.join(HERE, "feat_raw.npz"), allow_pickle=True)
FN = [str(s) for s in fz["featnames"]]; I = {k: i for i, k in enumerate(FN)}
bk = sorted(k for k in fz.files if k != "featnames")
targets += [("bearing RMS", [fz[b][:, I["rms"]] for b in bk]),
            ("bearing 4-10 kHz",
             [fz[b][:, I["b4_6"]] + fz[b][:, I["b6_8"]] + fz[b][:, I["b8_10"]] for b in bk])]
z1 = np.load(os.path.join(HERE, "cmapss_FD001.npz"))
un = sorted({k.split("__")[0] for k in z1.files})
targets.append(("turbofan s11", [z1[f"{u}__sensors"][:, 10].astype(float) for u in un]))
zb = np.load(os.path.join(HERE, "battery_raw.npz"))
bc = sorted({k.split("__")[0] for k in zb.files})
targets.append(("battery cap", [zb[f"{c}__cap"] for c in bc]))

rng = np.random.default_rng(63)
for lab, b in [("synth power-law b=3", 3.0), ("synth exponential", None)]:
    S = []
    for _ in range(15):
        tau = np.arange(1, 1201) / 1200
        D = tau ** 3.0 if b else (1 - np.exp(-2 * tau)) / (1 - np.exp(-2.0))
        S.append(D + rng.normal(0, 0.02, 1200))
    targets.append((lab, S))

print(f"{'indicator':<22}{'units':>6}{'pieces':>8}{'median pos':>12}"
      f"{'earliest':>10}{'concentration':>15}")
print("-" * 73)
rows = []
for name, series in targets:
    a = collect(series, name)
    if not a:
        continue
    rows.append(a)
    print(f"{name:<22}{a['units']:>6}{a['runs']:>8.0f}{a['med']:>12.2f}"
          f"{a['earliest']:>10.2f}{a['gini']:>15.3f}")

print("-" * 73)
print("pieces     = disjoint intervals in the optimal retention set (1 = FIFO-shaped)")
print("median pos = where that information sits, as a fraction of elapsed life")
print("earliest   = the oldest sample the optimum wants to keep")
json.dump(rows, open(os.path.join(HERE, "retention_shape.json"), "w"), indent=2)
