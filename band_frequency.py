"""
Why are the high-frequency bands usable earlier? Is there a mechanism?

Every bearing result so far says the same thing without explaining it: the
high-frequency bands become usable early and broadband amplitude does not. That
is an observation about seven channels, and it stays an observation until it is
tied to something physical.

Rolling-element bearing damage has a well-known signature. Early spalling is
impulsive, and an impulse rings whatever structural resonances the bearing and
its housing possess, which sit in the kilohertz range. The characteristic defect
frequencies themselves are low -- tens to hundreds of hertz -- and carry little
energy until the damage is advanced. That is the basis of resonance demodulation
as the standard early-detection technique, and it predicts a specific shape:
tau_min should fall as band centre frequency rises, because the resonant band
responds to damage that the low bands cannot yet see.

The prediction is testable on both rigs, which share a band structure. A
monotone fall would support the mechanism. A flat relation would mean the
earlier finding is about those particular channels and not about frequency. An
interior minimum would be more useful still: it would name the band to
instrument.
"""
import os
import json
import numpy as np
from nonparam import weighted_density, estimate_floor, quantile

HERE = os.path.dirname(os.path.abspath(__file__))
Q = 0.35
SEED = 606060

# band edges in kHz, from the feature names
EDGES = {"b0_1": (0, 1), "b1_2": (1, 2), "b2_4": (2, 4), "b4_6": (4, 6),
         "b6_8": (6, 8), "b8_10": (8, 10), "b10_12.8": (10, 12.8)}


def profile(series, rng):
    T, S = [], []
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
        T.append(t); S.append(float(d[:k + 1].sum()) / rk)
    if len(T) < 3:
        return None
    return float(np.median(T)), float(np.median(S)), len(T)


def run(tag, path, rng):
    z = np.load(os.path.join(HERE, path), allow_pickle=True)
    FN = [str(s) for s in z["featnames"]]
    I = {k: i for i, k in enumerate(FN)}
    BK = sorted(k for k in z.files if k != "featnames")
    out = []
    for nm, (lo, hi) in EDGES.items():
        if nm not in I:
            continue
        p = profile([z[b][:, I[nm]] for b in BK], rng)
        if not p:
            continue
        t, s, n = p
        out.append(dict(rig=tag, band=nm, centre=(lo + hi) / 2,
                        width=hi - lo, tau=t, signal=s, units=n))
    return out


rng = np.random.default_rng(SEED)
rows = run("PRONOSTIA", "feat_raw.npz", rng) + run("XJTU", "xjtu_feat.npz", rng)

print("earliest usable age against band centre frequency\n")
for rig in ("PRONOSTIA", "XJTU"):
    sub = [r for r in rows if r["rig"] == rig]
    if not sub:
        continue
    print(f"  {rig}")
    print(f"  {'band':<12}{'centre kHz':>11}{'units':>7}{'signal':>9}"
          f"{'tau(0.35)':>11}")
    print("  " + "-" * 48)
    for r in sorted(sub, key=lambda r: r["centre"]):
        print(f"  {r['band']:<12}{r['centre']:>11.1f}{r['units']:>7}"
              f"{100*r['signal']:>8.0f}%{r['tau']:>11.3f}")
    print()


def rank_corr(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    return float(np.corrcoef(np.argsort(np.argsort(a)),
                             np.argsort(np.argsort(b)))[0, 1])


print("-" * 62)
print("correlation of the earliest usable age with centre frequency\n")
for rig in ("PRONOSTIA", "XJTU"):
    sub = [r for r in rows if r["rig"] == rig]
    if len(sub) < 4:
        continue
    c = rank_corr([r["centre"] for r in sub], [r["tau"] for r in sub])
    best = min(sub, key=lambda r: r["tau"])
    print(f"  {rig:<12} rank correlation {c:+.2f}   "
          f"earliest band: {best['band']} at {best['centre']:.1f} kHz "
          f"(tau {best['tau']:.3f})")

both = {}
for r in rows:
    both.setdefault(r["band"], {})[r["rig"]] = r["tau"]
shared = {k: v for k, v in both.items() if len(v) == 2}
if len(shared) >= 4:
    a = [v["PRONOSTIA"] for v in shared.values()]
    b = [v["XJTU"] for v in shared.values()]
    print(f"\n  agreement between rigs on the band ordering: "
          f"{rank_corr(a, b):+.2f}  ({len(shared)} bands)")

print()
mono = all(rank_corr([r["centre"] for r in rows if r["rig"] == g],
                     [r["tau"] for r in rows if r["rig"] == g]) < -0.5
           for g in ("PRONOSTIA", "XJTU")
           if len([r for r in rows if r["rig"] == g]) >= 4)
interior = {}
for rig in ("PRONOSTIA", "XJTU"):
    sub = sorted([r for r in rows if r["rig"] == rig], key=lambda r: r["centre"])
    if len(sub) >= 4:
        i = int(np.argmin([r["tau"] for r in sub]))
        interior[rig] = (i not in (0, len(sub) - 1), sub[i]["band"])
if mono:
    print("The earliest usable age falls monotonically with centre frequency on")
    print("both rigs, which is what impulsive damage ringing a structural")
    print("resonance predicts, and is the mechanism behind resonance")
    print("demodulation as an early-detection technique.")
elif any(v[0] for v in interior.values()):
    print("The minimum sits at an interior band on "
          + ", ".join(k for k, v in interior.items() if v[0]) + ": "
          + ", ".join(f"{k} at {v[1]}" for k, v in interior.items() if v[0]))
    print("An interior optimum names the band worth instrumenting, which is")
    print("more useful than a monotone trend and is what a resonance -- rather")
    print("than simply 'higher is better' -- would produce.")
else:
    print("Centre frequency does not order the bands consistently, so the")
    print("earlier finding is about those particular channels and not about")
    print("frequency; no mechanism is established.")
json.dump(rows, open(os.path.join(HERE, "band_frequency.json"), "w"), indent=2,
          default=float)
