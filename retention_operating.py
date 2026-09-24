"""
Retention evaluated where it actually matters.

The previous table fixed tau_0 = 0.7 for every indicator. That is the wrong
place to ask the question: an indicator cannot support a decision before its own
tau_min, and for bearing RMS tau_min is 0.98 -- at 70% of life there is almost
nothing to retain, so the number described a regime no one operates in.

Here each indicator is evaluated at its own earliest actionable age, and at the
midpoint between that and end of life. The buffer inflation factor -- how much
larger a trailing window must be to match the best same-size set -- is the
quantity a designer actually needs.
"""
import os, json
import numpy as np
from pipeline import profile, tau_min

HERE = os.path.dirname(os.path.abspath(__file__))
QSTAR = 0.10
WS = (0.10, 0.20, 0.30)
REPS = 25
rng = np.random.default_rng(83)


def eff(d, k0, kw):
    best = float(np.sort(d[:k0 + 1])[::-1][:kw].sum())
    trail = float(d[max(0, k0 - kw + 1):k0 + 1].sum())
    return trail / best if best > 0 else np.nan


def inflation(d, k0, kw):
    target = float(np.sort(d[:k0 + 1])[::-1][:kw].sum())
    c = np.cumsum(d[:k0 + 1][::-1])
    j = int(np.searchsorted(c, target))
    return (j + 1) / kw if j < len(c) else np.inf


def floor_at(n, sigma, w, t0):
    tau = np.arange(1, n + 1) / n
    D = tau ** 3.0
    k0 = int(round(t0 * n)) - 1; kw = max(3, int(round(w * n)))
    if k0 < kw + 3:
        return np.nan
    v = [eff(u["dens"], k0, kw) for u, _ in
         (profile(D + rng.normal(0, sigma, n)) for _ in range(REPS)) if u]
    return float(np.median(v)) if v else np.nan


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

print(f"{'indicator':<20}{'tau_min':>9}{'tau_0':>7}{'w':>6}"
      f"{'efficiency':>12}{'buffer needed':>15}")
print("-" * 69)
rows = []
for name, series, trim in TARGETS:
    P = []
    for raw in series:
        raw = raw[int(round(trim * len(raw))):]
        u, _ = profile(raw)
        if u:
            P.append(u)
    if not P:
        continue
    tm = float(np.median([tau_min(u, QSTAR) for u in P
                          if np.isfinite(tau_min(u, QSTAR))]))
    for t0 in (tm, min(1.0, (tm + 1.0) / 2)):
        for w in WS:
            E, F, S, N = [], [], [], []
            for u in P:
                n = u["n"]; k0 = int(round(t0 * n)) - 1
                kw = max(3, int(round(w * n)))
                if k0 < kw + 3 or k0 >= n:
                    continue
                E.append(eff(u["dens"], k0, kw))
                F.append(inflation(u["dens"], k0, kw))
                S.append(u["sig_in"]); N.append(n)
            if not E:
                continue
            fl = floor_at(int(np.median(N)), float(np.median(S)), w, t0)
            e = float(np.median(E))
            ec = min(1.0, e / fl) if fl and np.isfinite(fl) and fl > 0 else np.nan
            fi = float(np.median([x for x in F if np.isfinite(x)])) if any(
                np.isfinite(F)) else np.inf
            rows.append(dict(name=name, tau_min=tm, tau0=float(t0), w=w,
                             corrected=float(ec), inflation=fi, units=len(E)))
            print(f"{name:<20}{tm:>9.2f}{t0:>7.2f}{w:>6.2f}{ec:>12.3f}"
                  f"{fi:>14.1f}x")
        print()

print("-" * 69)
ok = [r for r in rows if np.isfinite(r["corrected"])]
bear = [r for r in ok if r["name"].startswith("bearing")]
rest = [r for r in ok if not r["name"].startswith("bearing")]
print(f"bearings      efficiency {min(r['corrected'] for r in bear):.2f}-"
      f"{max(r['corrected'] for r in bear):.2f}, buffer up to "
      f"{max(r['inflation'] for r in bear if np.isfinite(r['inflation'])):.1f}x")
print(f"turbofan+batt efficiency {min(r['corrected'] for r in rest):.2f}-"
      f"{max(r['corrected'] for r in rest):.2f}")
json.dump(rows, open(os.path.join(HERE, "retention_operating.json"), "w"), indent=2)
