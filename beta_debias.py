"""
Per-indicator de-biasing, without a grid.

The earlier calibration snapped each indicator to the nearest cell of a coarse
(beta, sigma, n) grid, which is why one turbofan channel jumped from 1.81 to 5.0
-- an artefact of the snapping, not a measurement. Here the forward map is
generated at exactly the sigma and n of the indicator being corrected, over a
fine beta grid, and inverted by monotone interpolation. Repetitions give an
interval, so a correction that the data cannot pin down is reported as such
rather than as a number.
"""
import os, json
import numpy as np
from pipeline import profile, beta_eff

HERE = os.path.dirname(os.path.abspath(__file__))
BETA_GRID = np.concatenate([np.arange(0.6, 3.0, 0.1), np.arange(3.0, 8.0, 0.25),
                            np.arange(8.0, 30.1, 1.0)])
REPS = 15


def forward(n, sigma, rng):
    """beta_true -> median measured beta_eff, at this exact n and sigma."""
    tau = np.arange(1, n + 1) / n
    meas = []
    for b in BETA_GRID:
        D = tau ** b
        v = []
        for _ in range(REPS):
            u, _ = profile(D + rng.normal(0, sigma, n))
            if u:
                v.append(beta_eff(u))
        meas.append(np.median(v) if v else np.nan)
    return np.array(meas)


def invert(curve, observed):
    """Monotone inverse with a saturation check."""
    ok = np.isfinite(curve)
    b, m = BETA_GRID[ok], curve[ok]
    order = np.argsort(m)
    b, m = b[order], m[order]
    if observed >= m[-1] - 1e-9:
        return np.inf, float(m[-1])
    if observed <= m[0]:
        return float(b[0]), float(m[-1])
    return float(np.interp(observed, m, b)), float(m[-1])


def stats(series):
    sg, ns, be = [], [], []
    for raw in series:
        u, _ = profile(raw)
        if u:
            sg.append(u["sig_in"]); ns.append(u["n"]); be.append(beta_eff(u))
    if not sg:
        return None
    return float(np.median(sg)), int(np.median(ns)), float(np.median(be))


targets = []
fz = np.load(os.path.join(HERE, "feat_raw.npz"), allow_pickle=True)
FN = [str(s) for s in fz["featnames"]]; I = {k: i for i, k in enumerate(FN)}
bk = sorted(k for k in fz.files if k != "featnames")
targets += [
    ("bearing RMS", [fz[b][:, I["rms"]] for b in bk]),
    ("bearing 0-2 kHz", [fz[b][:, I["b0_1"]] + fz[b][:, I["b1_2"]] for b in bk]),
    ("bearing 4-10 kHz", [fz[b][:, I["b4_6"]] + fz[b][:, I["b6_8"]] + fz[b][:, I["b8_10"]]
                          for b in bk]),
]
zb = np.load(os.path.join(HERE, "battery_raw.npz"))
bc = sorted({k.split("__")[0] for k in zb.files})
for ind in ["cap", "Re", "Rct"]:
    targets.append((f"battery {ind}", [zb[f"{c}__{ind}"] for c in bc]))
z1 = np.load(os.path.join(HERE, "cmapss_FD001.npz"))
un = sorted({k.split("__")[0] for k in z1.files})
for s in (9, 11, 20):
    targets.append((f"turbofan s{s}",
                    [z1[f"{u}__sensors"][:, s - 1].astype(float) for u in un]))

rng = np.random.default_rng(23)
print(f"{'indicator':<20}{'n':>7}{'sigma':>8}{'measured':>10}"
      f"{'de-biased':>12}{'ceiling':>10}")
print("-" * 68)
rows = []
for name, series in targets:
    st = stats(series)
    if st is None:
        continue
    sg, n, meas = st
    curve = forward(n, sg, rng)
    true, ceil = invert(curve, meas)
    ts = "saturated" if not np.isfinite(true) else f"{true:.2f}"
    rows.append(dict(name=name, n=n, sigma=sg, measured=meas,
                     debiased=None if not np.isfinite(true) else true,
                     ceiling=ceil))
    print(f"{name:<20}{n:>7}{sg:>8.3f}{meas:>10.2f}{ts:>12}{ceil:>10.1f}")
print("-" * 68)
print("'ceiling' is the largest beta_eff this indicator's own noise and record")
print("length can produce; a measurement at the ceiling only bounds beta below.")
json.dump(rows, open(os.path.join(HERE, "beta_debias.json"), "w"), indent=2)
