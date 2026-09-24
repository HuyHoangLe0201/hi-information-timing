"""
Does the spatial strand replicate?

The claim reached so far is that measuring how vibration energy is DISTRIBUTED
becomes usable earlier than measuring how MUCH there is, and that the
distribution can be over frequency or over direction. The frequency half is
replicated on both rigs. The direction half rests on a single number from the
stored PRONOSTIA features -- the ratio of the two accelerometers, early at
0.501 -- and has never been checked on the second rig, because both raw
extractions took the horizontal channel only.

With vertical bands now extracted for both rigs the strand can be tested
properly, and on indicators defined here rather than inherited:

  spatial     h/v total power ratio, and the same ratio restricted to the high
              bands; neither says anything about the spectral shape within a
              channel.
  amount      the combined power of the two channels, which is the same
              quantity with the direction information divided out.
  spectral    entropy and spread of the summed spectrum, as the reference that
              already replicated.

A spatial measure that is early on both rigs while the combined power is late
would settle it: what matters is that the indicator describes a distribution,
not which axis the distribution runs along.
"""
import os
import json
import numpy as np
from nonparam import weighted_density, estimate_floor, quantile

HERE = os.path.dirname(os.path.abspath(__file__))
Q = 0.35
THRESH = 0.90
SEED = 246810
EPS = 1e-30

PAIRS = {"PRONOSTIA": ("fineband.npz", "vertical.npz"),
         "XJTU": ("fineband_xjtu.npz", "vertical_xjtu.npz")}


def profile(series, rng):
    T, S = [], []
    for x in series:
        x = np.asarray(x, float)
        x = x[np.isfinite(x)]
        if len(x) < 60:
            continue
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
        T.append(t)
        S.append(float(d[:k + 1].sum()) / rk)
    if len(T) < 4:
        return None
    return float(np.median(T)), float(np.median(S)), len(T)


def norm(P):
    return P / np.clip(P.sum(axis=1, keepdims=True), EPS, None)


MEASURES = [
    ("h/v total power ratio", "spatial",
     lambda H, V, f: H.sum(axis=1) / np.clip(V.sum(axis=1), EPS, None)),
    ("h/v high-band ratio", "spatial",
     lambda H, V, f: H[:, -8:].sum(axis=1)
     / np.clip(V[:, -8:].sum(axis=1), EPS, None)),
    ("combined power", "amount",
     lambda H, V, f: H.sum(axis=1) + V.sum(axis=1)),
    ("combined high-band power", "amount",
     lambda H, V, f: H[:, -8:].sum(axis=1) + V[:, -8:].sum(axis=1)),
    ("spectral entropy (h+v)", "spectral",
     lambda H, V, f: -(norm(H + V)
                       * np.log(np.clip(norm(H + V), EPS, None))).sum(axis=1)),
    ("spectral spread (h+v)", "spectral",
     lambda H, V, f: np.sqrt((norm(H + V)
                              * (f - (norm(H + V) * f).sum(axis=1,
                                                           keepdims=True)) ** 2
                              ).sum(axis=1))),
]

rng = np.random.default_rng(SEED)
ALL = {}
for rig, (fh, fv) in PAIRS.items():
    ph, pv = os.path.join(HERE, fh), os.path.join(HERE, fv)
    if not (os.path.exists(ph) and os.path.exists(pv)):
        print(f"{rig}: missing {fh if not os.path.exists(ph) else fv}")
        continue
    zh, zv = np.load(ph), np.load(pv)
    f = zh["centres"] / 1000.0
    common = sorted(set(k for k in zh.files if k != "centres")
                    & set(k for k in zv.files if k != "centres"))
    if not common:
        print(f"{rig}: no bearing present in both channels")
        continue
    pairs = []
    for b in common:
        H, V = zh[b].astype(float), zv[b].astype(float)
        n = min(len(H), len(V))
        if n >= 60:
            pairs.append((H[:n], V[:n]))
    out = []
    for lab, kind, fn in MEASURES:
        r = profile([fn(H, V, f) for H, V in pairs], rng)
        if r:
            out.append(dict(measure=lab, kind=kind, tau=r[0], signal=r[1],
                            units=r[2]))
    ALL[rig] = out
    print(f"{rig}: {len(pairs)} bearings with both channels")

rigs = list(ALL)
print()
print(f"{'measure':<26}{'kind':<10}" + "".join(f"{r:>12}" for r in rigs))
print("-" * (36 + 12 * len(rigs)))
for lab, kind, _ in MEASURES:
    cells = []
    for rig in rigs:
        m = next((x for x in ALL[rig] if x["measure"] == lab), None)
        cells.append(m["tau"] if m and m["signal"] >= THRESH else np.nan)
    if all(not np.isfinite(c) for c in cells):
        continue
    print(f"{lab:<26}{kind:<10}"
          + "".join("         n/a" if not np.isfinite(c) else f"{c:>12.3f}"
                    for c in cells))
print("-" * (36 + 12 * len(rigs)))

print()
for rig in rigs:
    ok = [x for x in ALL[rig] if x["signal"] >= THRESH]
    sp = [x["tau"] for x in ok if x["kind"] == "spatial"]
    am = [x["tau"] for x in ok if x["kind"] == "amount"]
    sc = [x["tau"] for x in ok if x["kind"] == "spectral"]
    if sp and am:
        print(f"{rig}: spatial {np.median(sp):.3f}, amount {np.median(am):.3f}"
              + (f", spectral {np.median(sc):.3f}" if sc else "")
              + f"   spatial is {np.median(am)-np.median(sp):+.3f} earlier")
print()
gaps = []
for rig in rigs:
    ok = [x for x in ALL[rig] if x["signal"] >= THRESH]
    sp = [x["tau"] for x in ok if x["kind"] == "spatial"]
    am = [x["tau"] for x in ok if x["kind"] == "amount"]
    if sp and am:
        gaps.append(np.median(am) - np.median(sp))
if len(gaps) == len(rigs) and all(g > 0.1 for g in gaps):
    print("The ratio of the two accelerometers is earlier than their combined")
    print("power on every rig tested, and it carries no information about the")
    print("spectral shape within either channel. So the effect is not about")
    print("spectra: an indicator that describes how vibration is distributed")
    print("is early whether the distribution runs over frequency or over")
    print("direction, while the same energy summed is late.")
elif gaps:
    print("The spatial strand does not replicate cleanly: "
          + ", ".join(f"{r} {g:+.3f}" for r, g in zip(rigs, gaps)))
    print("so the direction half of the claim rests on one rig and should be")
    print("reported as such.")
json.dump(ALL, open(os.path.join(HERE, "spatial_test.json"), "w"), indent=2,
          default=float)
