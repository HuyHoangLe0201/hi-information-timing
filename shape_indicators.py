"""
If relative beats absolute, do pure shape measures beat both?

Normalising a band by total spectrum power brought the earliest usable age
forward by two thirds of a lifetime, because early damage redistributes
vibration energy upward before it raises the level. A band fraction is a crude
shape measure -- it asks what share of the energy sits in one interval. Measures
built to describe the whole distribution should do at least as well, and the
older extraction already computed several that were never used here: spectral
centroid, spread and entropy, a high-band ratio, a Hilbert envelope level and a
spectral-kurtosis maximum.

They are compared against the amplitude statistics on the same records, both
accelerometer channels and both rigs, so the question is settled on one footing:
given a vibration record, which construction gives the earliest usable
indicator, and does the answer transfer between test stands?

Two classifications are judgement calls and are flagged rather than hidden.
band_hi was produced by an extraction whose definition is not to hand, and
energy_op is a Teager operator, which is arguably neither level nor shape. The
conclusion does not rest on either.
"""
import os
import glob
import json
import numpy as np
import pandas as pd
from nonparam import weighted_density, estimate_floor, quantile

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))
RIGS = {"PRONOSTIA": os.path.join(ROOT, "GBS_RUL", "results", "features"),
        "XJTU": os.path.join(ROOT, "GBS_RUL", "results", "features_xjtu")}
Q = 0.35
THRESH = 0.90
SEED = 777

KIND = {
    "rms": "level", "peak": "level", "p2p": "level", "energy_op": "level",
    "hilbert_rms": "level",
    "kurtosis": "shape (time)", "skewness": "shape (time)",
    "crest": "shape (time)", "shape": "shape (time)",
    "impulse": "shape (time)", "clearance": "shape (time)",
    "spec_centroid": "shape (spectral)", "spec_spread": "shape (spectral)",
    "spec_entropy": "shape (spectral)", "band_hi": "shape (spectral)",
    "sk_max": "shape (spectral)",
}


def profile(series, rng):
    T, S = [], []
    for x in series:
        x = np.asarray(x, float)
        x = x[np.isfinite(x)]
        if len(x) < 80:
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


def survey(feat, rig, rng):
    files = sorted(glob.glob(os.path.join(feat, "Bearing*.csv")))
    if not files:
        print(f"no feature files under {feat}")
        return []
    tabs = {os.path.basename(f)[:-4]: pd.read_csv(f) for f in files}
    cols = [c for c in next(iter(tabs.values())).columns
            if c.startswith(("h_", "v_"))]
    print(f"{rig}: {len(tabs)} bearings, {len(cols)} indicator columns")
    out = []
    for c in cols:
        p = profile([t[c].to_numpy() for t in tabs.values() if c in t], rng)
        if not p:
            continue
        t, s, n = p
        out.append(dict(rig=rig, column=c, channel=c[0], base=c[2:],
                        kind=KIND.get(c[2:], "other"), tau=t, signal=s,
                        units=n))
    return out


rng = np.random.default_rng(SEED)
ALL = {rig: survey(feat, rig, rng) for rig, feat in RIGS.items()}
print()

usable = {rig: sorted([r for r in v if r["signal"] >= THRESH],
                      key=lambda r: r["tau"])
          for rig, v in ALL.items()}

for rig, v in usable.items():
    if not v:
        continue
    print(f"\n{rig}, earliest usable first\n")
    print(f"{'indicator':<16}{'ch':>4}  {'kind':<18}{'signal':>8}"
          f"{'tau(0.35)':>11}")
    print("-" * 60)
    for r in v[:10]:
        print(f"{r['base']:<16}{r['channel']:>4}  {r['kind']:<18}"
              f"{100*r['signal']:>7.0f}%{r['tau']:>11.3f}")
    if len(v) > 10:
        print(f"{'...':<16}{'':>4}  {'':<18}{'':>8}")
        for r in v[-3:]:
            print(f"{r['base']:<16}{r['channel']:>4}  {r['kind']:<18}"
                  f"{100*r['signal']:>7.0f}%{r['tau']:>11.3f}")
    print("-" * 60)
    dropped = sorted({r["base"] for r in ALL[rig] if r["signal"] < THRESH})
    if dropped:
        print(f"below the signal threshold, not ranked: {', '.join(dropped)}")

print("\n\nby construction, both rigs\n")
print(f"{'kind':<20}{'PRONOSTIA':>12}{'XJTU':>10}{'n':>5}")
print("-" * 47)
summ = []
for k in ("level", "shape (time)", "shape (spectral)"):
    a = [r["tau"] for r in usable.get("PRONOSTIA", []) if r["kind"] == k]
    b = [r["tau"] for r in usable.get("XJTU", []) if r["kind"] == k]
    if not a and not b:
        continue
    ma = float(np.median(a)) if a else np.nan
    mb = float(np.median(b)) if b else np.nan
    summ.append(dict(kind=k, pronostia=ma, xjtu=mb, n=len(a)))
    print(f"{k:<20}{ma:>12.3f}{mb:>10.3f}{len(a):>5}")
print("-" * 47)

pa = {r["column"]: r["tau"] for r in usable.get("PRONOSTIA", [])}
pb = {r["column"]: r["tau"] for r in usable.get("XJTU", [])}
common = sorted(set(pa) & set(pb))
if len(common) >= 6:
    a = [pa[c] for c in common]
    b = [pb[c] for c in common]
    rk = float(np.corrcoef(np.argsort(np.argsort(a)),
                           np.argsort(np.argsort(b)))[0, 1])
    print(f"\nrank agreement across rigs on {len(common)} shared indicators: "
          f"{rk:+.2f}")
    lv = [c for c in common if KIND.get(c[2:]) == "level"]
    sh = [c for c in common if str(KIND.get(c[2:], "")).startswith("shape")]
    if lv and sh:
        print(f"  level : PRONOSTIA {np.median([pa[c] for c in lv]):.3f}"
              f"   XJTU {np.median([pb[c] for c in lv]):.3f}")
        print(f"  shape : PRONOSTIA {np.median([pa[c] for c in sh]):.3f}"
              f"   XJTU {np.median([pb[c] for c in sh]):.3f}")
        gap_a = np.median([pa[c] for c in lv]) - np.median([pa[c] for c in sh])
        gap_b = np.median([pb[c] for c in lv]) - np.median([pb[c] for c in sh])
        print(f"\n  the level-shape gap is {gap_a:.3f} of life on PRONOSTIA "
              f"and {gap_b:.3f} on XJTU")

json.dump({rig: v for rig, v in ALL.items()} | {"by_kind": summ},
          open(os.path.join(HERE, "shape_indicators.json"), "w"), indent=2,
          default=float)
