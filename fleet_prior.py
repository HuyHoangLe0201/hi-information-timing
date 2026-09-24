"""
Between "trend known" and "trend unknown" lies the fleet.

Two bounds have now been computed and neither is the operational one.

  I_dd            treats the degradation trend as known exactly. It is what the
                  validation confirmed, because that resampled each unit's own
                  residuals around its own fitted trend -- so the result stands
                  for what it tested, and only for that.
  Schur           treats the trend parameters as wholly unknown. For beta = 3 it
                  leaves 0.08% of I_dd, which is far too pessimistic: nobody
                  deploys a monitor knowing nothing about how these units fail.

The truth is a prior. A fleet supplies one, and its strength is measurable: fit
the shape parameters per unit, and their spread across the fleet IS the prior
covariance. The van Trees form then interpolates between the two limits,

    I_eff  =  I_dd  -  I_{d phi} ( I_{phi phi} + Lambda )^{-1} I_{phi d},

with Lambda the inverse of the fleet's covariance over the shape parameters.
Lambda -> infinity recovers I_dd, Lambda -> 0 recovers the Schur complement.

The retained fraction I_eff / I_dd is then a property of the indicator AND the
fleet it is deployed on -- which is the honest answer to how much a record is
worth.
"""
import os
import json
import numpy as np
from scipy.optimize import curve_fit
from pipeline import robust_scale, oof_trend, _win
from scipy.signal import savgol_filter

HERE = os.path.dirname(os.path.abspath(__file__))


def fit_shape(x):
    """Fit A tau^beta to one record; returns (log A, beta) or None."""
    n = len(x)
    s = robust_scale(x)
    if not np.isfinite(s) or s <= 0:
        return None
    w = _win(n)
    if w < 7:
        return None
    tr = savgol_filter((x - np.median(x[:max(3, n // 20)])) / s, w, 2)
    tau = np.arange(1, n + 1) / n
    lo = tr[:max(3, n // 20)].mean()
    y = tr - lo
    if y[-1] <= 0:
        return None
    y = y / y[-1]
    m = y > 1e-3
    if m.sum() < 30:
        return None
    try:
        p, _ = curve_fit(lambda t, A, b: A * t ** b, tau[m], y[m],
                         p0=[1.0, 2.0], bounds=([1e-3, 0.51], [1e3, 30.0]),
                         maxfev=20000)
    except Exception:
        return None
    return float(np.log(p[0])), float(p[1])


def grads(tau, beta, A):
    t = np.clip(tau, 1e-9, None)
    return {"delta": -A * beta * t ** (beta - 1.0),
            "logA": A * t ** beta,            # d/d(log A) = A tau^beta
            "beta": A * t ** beta * np.log(t)}


def retained(tau, beta, A, Lam):
    """I_eff / I_dd with prior information Lam on (log A, beta)."""
    g = grads(tau, beta, A)
    G = np.column_stack([g["delta"], g["logA"], g["beta"]])
    I = G.T @ G
    idd = I[0, 0]
    if idd <= 0:
        return np.nan
    M = I[1:, 1:] + Lam
    eff = idd - I[0, 1:] @ np.linalg.solve(M, I[1:, 0])
    return float(np.clip(eff, 0.0, idd) / idd)


def fleet(name, series):
    P = [fit_shape(np.asarray(x, float)) for x in series]
    P = [p for p in P if p is not None]
    if len(P) < 4:
        return None
    P = np.array(P)
    C = np.cov(P, rowvar=False)
    C = C + 1e-12 * np.eye(2)
    Lam = np.linalg.inv(C)
    beta = float(np.median(P[:, 1]))
    A = float(np.exp(np.median(P[:, 0])))
    tau = np.linspace(1e-4, 1.0, 2000)
    r_fleet = retained(tau, beta, A, Lam)
    r_none = retained(tau, beta, A, np.zeros((2, 2)))
    r_known = 1.0
    return dict(domain=name, units=len(P), beta=beta,
                sd_logA=float(np.sqrt(C[0, 0])), sd_beta=float(np.sqrt(C[1, 1])),
                retained_fleet=r_fleet, retained_none=r_none,
                retained_known=r_known)


print("how much of the nominal information survives, given what the fleet knows\n")
print(f"{'indicator':<22}{'units':>6}{'beta':>7}{'sd log A':>10}{'sd beta':>9}"
      f"{'trend known':>13}{'fleet prior':>13}{'no prior':>10}")
print("-" * 90)
rows = []

z = np.load(os.path.join(HERE, "feat_raw.npz"), allow_pickle=True)
FN = [str(s) for s in z["featnames"]]
I = {k: i for i, k in enumerate(FN)}
BK = sorted(k for k in z.files if k != "featnames")
for lab, parts in (("bearing RMS", ["rms"]),
                   ("bearing 0--2 kHz", ["b0_1", "b1_2"]),
                   ("bearing 4--10 kHz", ["b4_6", "b6_8", "b8_10"])):
    r = fleet(lab, [sum(z[b][:, I[p]] for p in parts) for b in BK])
    if r:
        rows.append(r)

zt = np.load(os.path.join(HERE, "cmapss_FD001.npz"))
un = sorted({k.split("__")[0] for k in zt.files})
for c, lab in ((4, "turbofan s4"), (11, "turbofan s11")):
    r = fleet(lab, [zt[f"{u}__sensors"][:, c - 1].astype(float) for u in un])
    if r:
        rows.append(r)

zb = np.load(os.path.join(HERE, "battery_raw.npz"))
cl = sorted({k.split("__")[0] for k in zb.files})
r = fleet("cell capacity", [zb[f"{c}__cap"] for c in cl])
if r:
    rows.append(r)

for r in rows:
    print(f"{r['domain']:<22}{r['units']:>6}{r['beta']:>7.2f}"
          f"{r['sd_logA']:>10.3f}{r['sd_beta']:>9.2f}"
          f"{1.0:>13.3f}{r['retained_fleet']:>13.5f}{r['retained_none']:>10.5f}")
print("-" * 90)
print("sd log A, sd beta = the fleet's own spread in the shape parameters;")
print("a tight fleet is a strong prior and pulls the retained fraction up.\n")
for r in rows:
    f = r["retained_fleet"]
    print(f"  {r['domain']:<22} the fleet prior is worth "
          f"{f / max(r['retained_none'], 1e-12):>8.0f}x the no-prior case, "
          f"and still costs {1/max(f,1e-12):.0f}x in information")
json.dump(rows, open(os.path.join(HERE, "fleet_prior.json"), "w"), indent=2)
