"""
Fit the battery indicators with the class the parent paper assigns them.

[1] places batteries in the stretched-exponential family (b = 0.7, alpha_b = 3
as its illustrative values). Fitting that family to the measured information
distribution asks whether the nominal class actually describes the data --
the same question the bearing analysis asked of the nominal beta = 3.
"""
import os, json
import numpy as np
from scipy.special import gammainc, gammaincinv
from scipy.optimize import minimize
from pipeline import prepare, tau_min, beta_eff

HERE = os.path.dirname(os.path.abspath(__file__))
z = np.load(os.path.join(HERE, "battery_raw.npz"))
CELLS = sorted({k.split("__")[0] for k in z.files})
QSTAR = 0.35
NOM_B, NOM_AB = 0.7, 3.0


def F_str(tau, ab, b):
    s = 2 - 1.0 / b
    return gammainc(s, 2 * ab * np.power(tau, b)) / gammainc(s, 2 * ab)


def Finv_str(u, ab, b):
    s = 2 - 1.0 / b
    return (gammaincinv(s, u * gammainc(s, 2 * ab)) / (2 * ab)) ** (1.0 / b)


def fit_str(Fobs, n):
    tau = np.arange(1, n + 1) / n
    def obj(p):
        b, ab = p
        if not (0.51 < b <= 1.0) or not (0.05 < ab < 30):
            return 1e6
        try:
            return float(np.mean((Fobs - F_str(tau, ab, b)) ** 2))
        except Exception:
            return 1e6
    best = None
    for b0 in (0.6, 0.75, 0.9):
        for a0 in (0.5, 3.0, 10.0):
            r = minimize(obj, [b0, a0], method="Nelder-Mead",
                         options={"xatol": 1e-4, "fatol": 1e-10, "maxiter": 2000})
            if best is None or r.fun < best.fun:
                best = r
    return float(best.x[0]), float(best.x[1]), float(best.fun)


nom_tau_min = float(Finv_str(QSTAR, NOM_AB, NOM_B))
print(f"nominal class of [1]: b={NOM_B}, alpha_b={NOM_AB}  ->  tau_min = {nom_tau_min:.4f}")
print(f"(at q* = {QSTAR}; the target would be reachable almost from the start)\n")

print(f"{'cell':<8}{'ind':<6}{'b_eff':>8}{'ab_eff':>9}{'rmse':>9}"
      f"{'tau_min obs':>13}{'nominal':>9}")
print("-" * 62)
rows = []
for cell in CELLS:
    for tag in ["cap", "Re", "Rct"]:
        u, why = prepare(z[f"{cell}__{tag}"], f"{cell}/{tag}")
        if u is None:
            continue
        b, ab, mse = fit_str(u["F"], u["n"])
        tm = tau_min(u, QSTAR)
        rows.append(dict(cell=cell, tag=tag, b_eff=b, ab_eff=ab,
                         rmse=float(np.sqrt(mse)), tau_min=tm,
                         beta_eff=beta_eff(u)))
        print(f"{cell:<8}{tag:<6}{b:>8.3f}{ab:>9.3f}{np.sqrt(mse):>9.4f}"
              f"{tm:>13.3f}{nom_tau_min:>9.3f}")

print("-" * 62)
for tag in ["cap", "Re", "Rct"]:
    r = [x for x in rows if x["tag"] == tag]
    if r:
        print(f"median {tag:<5} b_eff={np.median([x['b_eff'] for x in r]):.3f}  "
              f"alpha_b={np.median([x['ab_eff'] for x in r]):.2f}  "
              f"tau_min={np.median([x['tau_min'] for x in r]):.3f}")

cap = [x for x in rows if x["tag"] == "cap"]
print(f"\ncapacity fade, the standard indicator:")
print(f"  measured tau_min  {np.median([x['tau_min'] for x in cap]):.3f}")
print(f"  nominal  tau_min  {nom_tau_min:.3f}")
print(f"  ratio             {np.median([x['tau_min'] for x in cap])/nom_tau_min:.0f}x")

json.dump(dict(qstar=QSTAR, nominal=dict(b=NOM_B, alpha_b=NOM_AB,
                                         tau_min=nom_tau_min), rows=rows),
          open(os.path.join(HERE, "battery_class_fit.json"), "w"), indent=2)
print("\nwrote battery_class_fit.json")
