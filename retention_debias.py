"""
Per-indicator de-biasing of the FIFO efficiency.

The validation showed the metric reads low when the density estimate is noisy or
the record is short: on a power-law, where the trailing window is provably the
optimal retention set, the pipeline still reported 0.66-0.88 instead of 1.00.

The correction is the same device used for beta_eff. For each real indicator,
synthetic power-law data are generated at exactly that indicator's sigma and n --
a case whose true efficiency is 1 by construction -- and the value the pipeline
returns there is the metric's noise floor. Dividing by it removes the part of the
measured penalty that is measurement rather than physics.
"""
import os, json
import numpy as np
from pipeline import profile

HERE = os.path.dirname(os.path.abspath(__file__))
TAU0 = 0.7
WS = (0.10, 0.20, 0.30)
REPS = 25


def eff(dens, k0, kw):
    best = float(np.sort(dens[:k0 + 1])[::-1][:kw].sum())
    trail = float(dens[max(0, k0 - kw + 1):k0 + 1].sum())
    return trail / best if best > 0 else np.nan


def noise_floor(n, sigma, w, rng):
    """What the metric returns when the true efficiency is exactly 1."""
    tau = np.arange(1, n + 1) / n
    D = tau ** 3.0
    k0 = int(round(TAU0 * n)) - 1
    kw = max(3, int(round(w * n)))
    v = []
    for _ in range(REPS):
        u, _ = profile(D + rng.normal(0, sigma, n))
        if u:
            v.append(eff(u["dens"], k0, kw))
    return float(np.median(v)) if v else np.nan


def measure(series, w):
    e, sg, ns = [], [], []
    for raw in series:
        u, _ = profile(raw)
        if not u:
            continue
        n = u["n"]; k0 = int(round(TAU0 * n)) - 1; kw = max(3, int(round(w * n)))
        if k0 < kw + 3:
            continue
        e.append(eff(u["dens"], k0, kw)); sg.append(u["sig_in"]); ns.append(n)
    if not e:
        return None
    return float(np.median(e)), float(np.median(sg)), int(np.median(ns))


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

rng = np.random.default_rng(57)
print(f"{'indicator':<20}{'w':>6}{'n':>7}{'sigma':>8}{'measured':>10}"
      f"{'metric floor':>14}{'corrected':>11}")
print("-" * 76)
rows = []
for name, series in targets:
    for w in WS:
        m = measure(series, w)
        if m is None:
            continue
        meas, sg, n = m
        nf = noise_floor(n, sg, w, rng)
        corr = min(1.0, meas / nf) if nf and np.isfinite(nf) and nf > 0 else np.nan
        rows.append(dict(name=name, w=w, n=n, sigma=sg, measured=meas,
                         floor=float(nf), corrected=float(corr)))
        print(f"{name:<20}{w:>6.2f}{n:>7}{sg:>8.3f}{meas:>10.3f}"
              f"{nf:>14.3f}{corr:>11.3f}")
    print()

print("-" * 76)
print("corrected = measured / (what the metric returns when the answer is 1).")
print("Values still well below 1 are a real retention penalty; values near 1")
print("mean the trailing window was close to optimal all along.")
json.dump(rows, open(os.path.join(HERE, "retention_debias.json"), "w"), indent=2)
