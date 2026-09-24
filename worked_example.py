"""Numbers quoted in Section V: the worked buffer-sizing example."""
import os
import json
import numpy as np
from scipy.signal import savgol_filter

HERE = os.path.dirname(os.path.abspath(__file__))
fz = np.load(os.path.join(HERE, "feat_raw.npz"), allow_pickle=True)
FN = [str(s) for s in fz["featnames"]]
I = {n: k for k, n in enumerate(FN)}
BEARINGS = sorted(k for k in fz.files if k != "featnames")
SMOOTH_FRAC, SIGMA_MAX = 0.08, 0.5
PERIOD = 10.0
TAU0, QSTAR = 0.7, 0.35

BANDS = {"rms": lambda M: M[:, I["rms"]],
         "hf 4-10kHz": lambda M: M[:, I["b4_6"]] + M[:, I["b6_8"]] + M[:, I["b8_10"]],
         "lf 0-2kHz": lambda M: M[:, I["b0_1"]] + M[:, I["b1_2"]]}


def dens_of(x):
    n = len(x)
    base = np.median(x[:max(3, int(0.05 * n))]); end = np.median(x[-3:])
    if not np.isfinite(base) or not np.isfinite(end) or abs(end - base) < 1e-12:
        return None
    D = (x - base) / (end - base)
    win = max(7, int(SMOOTH_FRAC * n)); win += (win % 2 == 0)
    win = min(win, n - 1 if (n - 1) % 2 == 1 else n - 2)
    if win < 7:
        return None
    tr = savgol_filter(D, win, 2)
    dt = savgol_filter(D, win, 2, deriv=1, delta=1.0 / n)
    res = D - tr
    sig = float(1.4826 * np.median(np.abs(res - np.median(res))))
    if not np.isfinite(sig) or sig > SIGMA_MAX:
        return None
    d = dt ** 2
    return (d / d.sum(), n) if d.sum() > 0 else None


out = {}
for name, fn in BANDS.items():
    ws, ns, fss, lives = [], [], [], []
    for b in BEARINGS:
        r = dens_of(fn(fz[b]))
        if r is None:
            continue
        dens, n = r
        k0 = int(round(TAU0 * n)) - 1
        cum = np.cumsum(dens)
        info_upto = cum[k0]
        if info_upto < QSTAR:
            ws.append(np.nan); ns.append(np.nan)
        else:
            # smallest w with cum[k0] - cum[klo] >= QSTAR
            target = info_upto - QSTAR
            klo = int(np.searchsorted(cum, target))
            w = (k0 - klo) / n
            ws.append(w); ns.append(k0 - klo)
        fss.append(n); lives.append(n * PERIOD)
    ws, ns = np.array(ws, float), np.array(ns, float)
    ok = np.isfinite(ws)
    out[name] = dict(
        units=len(ws), feasible=int(ok.sum()),
        w_med=float(np.median(ws[ok])) if ok.any() else None,
        N_med=float(np.median(ns[ok])) if ok.any() else None,
        fs_med=float(np.median(fss)), life_med=float(np.median(lives)))
    o = out[name]
    print(f"{name}:")
    print(f"   feasible at tau0={TAU0}: {o['feasible']}/{o['units']} units")
    if o["w_med"] is not None:
        print(f"   median w*        = {o['w_med']:.3f} of life")
        print(f"   median N*        = {o['N_med']:.0f} snapshots "
              f"= {o['N_med']*PERIOD/60:.0f} min of retained history")
    print(f"   median f_s       = {o['fs_med']:.0f} samples/life")
    print(f"   median life      = {o['life_med']:.0f} s = {o['life_med']/3600:.1f} h")

with open(os.path.join(HERE, "worked_example.json"), "w") as fh:
    json.dump(dict(tau0=TAU0, qstar=QSTAR, bands=out), fh, indent=2)
