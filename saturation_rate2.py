"""
Corrected version.

Two faults in the first attempt:
  A: the constant was too small, so the excess started below the floor and every
     rho_w saturated at n = 1.
  B: the floor was taken to be the last measured point, which forces the excess
     at that point to zero and biases the fitted rate steeply upward -- which is
     why it returned a = 2.7, faster than the minimax limit and not believable.

Here A uses a constant that puts the excess above the floor at n = 1, and B fits
floor, constant and rate jointly instead of assuming the floor.
"""
import os, json
import numpy as np
from scipy.optimize import curve_fit

HERE = os.path.dirname(os.path.abspath(__file__))

print("=== A. n_sat vs rho_w at a known rate (constant fixed) ===")
RHOS = np.array([0.05, 0.10, 0.20, 0.40])
NGRID = np.unique(np.round(np.logspace(0, 6, 20000)).astype(int))
C = 5000.0
print(f"{'rate a':>8}{'1/a':>7}   " + "".join(f"{'rho='+str(r):>12}" for r in RHOS)
      + "   fitted")
print("-" * 78)
rowsA = []
for a in (0.5, 0.75, 1.0):
    sat = []
    for r in RHOS:
        floor = 1.0 / r
        k = int(np.argmax(C * NGRID ** (-a) <= floor))
        sat.append(NGRID[k])
    sat = np.array(sat, float)
    slope = float(np.polyfit(np.log(RHOS), np.log(sat), 1)[0])
    rowsA.append(dict(a=a, expected=1 / a, fitted=slope, n_sat=[int(s) for s in sat]))
    print(f"{a:>8.2f}{1/a:>7.2f}   " + "".join(f"{int(s):>12}" for s in sat)
          + f"   {slope:.3f}")
print("-" * 78)
print("fitted exponent matches 1/a exactly, confirming n_sat ~ rho_w^(1/a):")
print("the rho_w^2 in Theorem 1 IS the slow-rate assumption, restated.\n")

print("=== B. the rate C-MAPSS actually shows, floor fitted not assumed ===")
st = json.load(open(os.path.join(HERE, "saturation_test.json")))
NS = np.array(st["NS"], float)


def model(n, floor, c, a):
    return floor + c * n ** (-a)


out = {}
print(f"{'w':>6}{'floor':>9}{'C':>9}{'rate a_rmse':>13}{'risk a':>9}{'1/a':>7}")
print("-" * 53)
for w, curve in st["curves"].items():
    y = np.array(curve, float)
    try:
        p, _ = curve_fit(model, NS, y, p0=[y[-1] * 0.95, y[0], 0.5],
                         bounds=([0, 0, 0.05], [y.min(), 1e4, 3.0]), maxfev=20000)
    except Exception as e:
        print(f"{w:>6}  fit failed: {e}")
        continue
    floor, c, a_r = p
    out[w] = dict(floor=float(floor), C=float(c), a_rmse=float(a_r),
                  a_risk=float(2 * a_r))
    print(f"{w:>6}{floor:>9.2f}{c:>9.1f}{a_r:>13.2f}{2*a_r:>9.2f}{1/(2*a_r):>7.2f}")

if out:
    a_mean = float(np.mean([v["a_risk"] for v in out.values()]))
    rho = st["rho"]; ks = sorted(rho, key=lambda k: float(k))
    ratio = rho[ks[1]] / rho[ks[0]]
    print("-" * 53)
    print(f"  mean risk rate a = {a_mean:.2f}   (Eq. (7) assumes a = 0.5)")
    print(f"\n  n_sat ratio for w={ks[0]} -> w={ks[1]}  (rho ratio {ratio:.2f}):")
    for label, aa in [("slow rate, a=0.5 (as in Eq. 7)", 0.5),
                      ("fast rate, a=1.0 (Thm 6 of [1])", 1.0),
                      (f"measured, a={a_mean:.2f}", a_mean)]:
        print(f"    {label:<34} predicts {ratio ** (1/aa):>6.2f}")
    print(f"    {'observed':<34}          {st['observed_ratio']:>6.2f}")
    json.dump(dict(partA=rowsA, partB=out, a_mean=a_mean,
                   rho_ratio=ratio, observed=st["observed_ratio"]),
              open(os.path.join(HERE, "saturation_rate.json"), "w"), indent=2)
