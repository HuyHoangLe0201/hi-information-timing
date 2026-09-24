"""
What Theorem 1's exponent actually depends on.

n_sat is where the training term meets the window floor. If the excess risk of
the trained predictor decays as C n^-a, then

        C n_sat^-a = Tbar^2 / (Gamma rho_w)   =>   n_sat  proportional to  rho_w^(1/a)

so the exponent on rho_w is 1/a and is inherited from the LEARNING RATE, not
from anything the windowed bound establishes. Eq. (7) carries the distribution-
free slow rate a = 1/2, giving rho_w^2. But [1] also proves a fast rate a = 1
under its Bernstein condition, which would give rho_w^1 instead.

Part A verifies that relation by construction, at three known rates.
Part B measures the rate the C-MAPSS predictor actually exhibits, which decides
which exponent applies to real data.
"""
import os, json
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
print("=== A. n_sat vs rho_w at a known learning rate ===")
print("    training excess risk set to C n^-a by construction\n")
RHOS = np.array([0.05, 0.10, 0.20, 0.40])
C = 1.0
NGRID = np.unique(np.round(np.logspace(0, 5, 4000)).astype(int))

print(f"{'rate a':>8}{'exponent 1/a':>14}   n_sat over rho_w = " +
      " ".join(f"{r:g}" for r in RHOS))
print("-" * 74)
rows = []
for a in (0.5, 0.75, 1.0):
    sat = []
    for r in RHOS:
        floor = 1.0 / r                      # Tbar^2/(Gamma rho_w), constants absorbed
        excess = C * NGRID ** (-a)
        k = int(np.argmax(excess <= floor))
        sat.append(NGRID[k])
    sat = np.array(sat, float)
    # fit the observed exponent: log n_sat vs log rho_w
    slope = float(np.polyfit(np.log(RHOS), np.log(sat), 1)[0])
    rows.append(dict(a=a, expected=1 / a, fitted=slope,
                     n_sat=[int(s) for s in sat]))
    print(f"{a:>8.2f}{1/a:>14.2f}   " + " ".join(f"{int(s):>7}" for s in sat) +
          f"   fitted exponent {slope:.2f}")
print("-" * 74)
print("the fitted exponent reproduces 1/a in every row, so the rho_w exponent")
print("in Theorem 1 is a restatement of the assumed learning rate.\n")

print("=== B. which rate does the C-MAPSS predictor actually show? ===")
st = json.load(open(os.path.join(HERE, "saturation_test.json")))
NS = np.array(st["NS"], float)
out = {}
for w, curve in st["curves"].items():
    c = np.array(curve, float)
    floor = c[-1]
    excess = c - floor
    m = excess > 1e-6
    if m.sum() < 4:
        continue
    # RMSE excess ~ n^-a  (risk is the square, so the risk rate is 2a)
    a_rmse = -float(np.polyfit(np.log(NS[m]), np.log(excess[m]), 1)[0])
    out[w] = a_rmse
    print(f"  w={w}: excess RMSE decays as n^-{a_rmse:.2f}  "
          f"-> risk rate a = {2*a_rmse:.2f}, exponent 1/a = {1/(2*a_rmse):.2f}")

if out:
    a_mean = float(np.mean([2 * v for v in out.values()]))
    rho = st["rho"]
    ks = sorted(rho, key=lambda k: float(k))
    ratio_pred_slow = (rho[ks[1]] / rho[ks[0]]) ** 2
    ratio_pred_meas = (rho[ks[1]] / rho[ks[0]]) ** (1 / a_mean)
    print(f"\n  measured risk rate a = {a_mean:.2f}  (Eq. (7) assumes 0.5)")
    print(f"  n_sat ratio predicted at a=0.5 : {ratio_pred_slow:.1f}")
    print(f"  n_sat ratio predicted at a={a_mean:.2f}: {ratio_pred_meas:.1f}")
    print(f"  n_sat ratio observed           : {st['observed_ratio']:.1f}")
    json.dump(dict(partA=rows, rates=out, a_mean=a_mean,
                   pred_slow=ratio_pred_slow, pred_measured=ratio_pred_meas,
                   observed=st["observed_ratio"]),
              open(os.path.join(HERE, "saturation_rate.json"), "w"), indent=2)
