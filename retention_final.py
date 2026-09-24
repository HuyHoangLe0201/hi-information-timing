"""
The retention result, with both corrections applied.

Two things inflated the first version of this number:
  1. the metric reads low when the density estimate is noisy or the record short
     (measured on a case whose true answer is 1), and
  2. the first tenth of a bearing record is run-in, which the degradation trend
     does not describe, so its large derivative was credited as information.

This applies both and reports what survives. It also settles the apparent
contradiction between a density that peaks at end of life and an optimal
retention set that sits early, by reporting where the mass actually is.
"""
import os, json
import numpy as np
from pipeline import profile

HERE = os.path.dirname(os.path.abspath(__file__))
TAU0 = 0.7
WS = (0.10, 0.20, 0.30)
REPS = 25
rng = np.random.default_rng(71)


def eff_pos(d, k0, kw):
    best = float(np.sort(d[:k0 + 1])[::-1][:kw].sum())
    trail = float(d[max(0, k0 - kw + 1):k0 + 1].sum())
    return (trail / best if best > 0 else np.nan)


def floor_at(n, sigma, w):
    tau = np.arange(1, n + 1) / n
    D = tau ** 3.0
    k0 = int(round(TAU0 * n)) - 1; kw = max(3, int(round(w * n)))
    v = []
    for _ in range(REPS):
        u, _ = profile(D + rng.normal(0, sigma, n))
        if u:
            v.append(eff_pos(u["dens"], k0, kw))
    return float(np.median(v)) if v else np.nan


def run(series, w, trim):
    E, S, N, M = [], [], [], []
    for raw in series:
        raw = raw[int(round(trim * len(raw))):]
        u, _ = profile(raw)
        if not u:
            continue
        n = u["n"]; k0 = int(round(TAU0 * n)) - 1; kw = max(3, int(round(w * n)))
        if k0 < kw + 3:
            continue
        d = u["dens"]
        E.append(eff_pos(d, k0, kw)); S.append(u["sig_in"]); N.append(n)
        c = np.cumsum(d) / d.sum()
        M.append(1 - np.interp(0.8, np.arange(1, n + 1) / n, c))
    if not E:
        return None
    return (float(np.median(E)), float(np.median(S)), int(np.median(N)),
            float(np.median(M)), len(E))


fz = np.load(os.path.join(HERE, "feat_raw.npz"), allow_pickle=True)
FN = [str(s) for s in fz["featnames"]]; I = {k: i for i, k in enumerate(FN)}
bk = sorted(k for k in fz.files if k != "featnames")
z1 = np.load(os.path.join(HERE, "cmapss_FD001.npz"))
un = sorted({k.split("__")[0] for k in z1.files})
zb = np.load(os.path.join(HERE, "battery_raw.npz"))
bc = sorted({k.split("__")[0] for k in zb.files})

TARGETS = [
    ("bearing RMS", [fz[b][:, I["rms"]] for b in bk], 0.10),
    ("bearing 4-10 kHz",
     [fz[b][:, I["b4_6"]] + fz[b][:, I["b6_8"]] + fz[b][:, I["b8_10"]] for b in bk], 0.10),
    ("turbofan s11", [z1[f"{u}__sensors"][:, 10].astype(float) for u in un], 0.0),
    ("battery cap", [zb[f"{c}__cap"] for c in bc], 0.0)]

print(f"{'indicator':<20}{'w':>6}{'raw':>8}{'-run-in':>9}{'-noise':>8}"
      f"{'penalty':>10}{'mass last 20%':>15}")
print("-" * 76)
rows = []
for name, series, trim in TARGETS:
    for w in WS:
        a = run(series, w, 0.0)
        b = run(series, w, trim) if trim else a
        if a is None or b is None:
            continue
        e1, _, _, _, _ = a
        e2, sg, n, mass, k = b
        fl = floor_at(n, sg, w)
        e3 = min(1.0, e2 / fl) if fl and np.isfinite(fl) and fl > 0 else np.nan
        rows.append(dict(name=name, w=w, units=k, raw=e1, detrimmed=e2,
                         floor=float(fl), corrected=float(e3),
                         mass_last20=mass, n=n, sigma=sg))
        print(f"{name:<20}{w:>6.2f}{e1:>8.3f}{e2:>9.3f}{e3:>8.3f}"
              f"{1/e3 if e3 > 0 else np.inf:>9.1f}x{mass:>15.2f}")
    print()

print("-" * 76)
b = [r for r in rows if r["name"].startswith("bearing")]
t = [r for r in rows if r["name"].startswith("turbofan")]
bt = [r for r in rows if r["name"].startswith("battery")]
print(f"bearings  : {min(r['corrected'] for r in b):.2f}-"
      f"{max(r['corrected'] for r in b):.2f}  (penalty stands)")
print(f"turbofan  : {min(r['corrected'] for r in t):.2f}-"
      f"{max(r['corrected'] for r in t):.2f}  (no penalty; earlier number was the floor)")
print(f"battery   : {min(r['corrected'] for r in bt):.2f}-"
      f"{max(r['corrected'] for r in bt):.2f}")
json.dump(rows, open(os.path.join(HERE, "retention_final.json"), "w"), indent=2)
