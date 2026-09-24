"""
Does Theorem 1 hold for its own model?

The C-MAPSS attempt had no power: both window widths sat on the fleet-variance
floor, so changing rho_w changed nothing observable. Here the experiment is put
squarely in the regime the theorem describes, by simulating the structure of
Eq. (7) directly:

    training  -- n run-to-failure trajectories are used to estimate the physics
                 parameter beta, giving beta_hat with error falling in n;
    deployment -- a fresh unit is observed on a trailing window and its age is
                 estimated by the ML rule of Proposition 1, using D(.;beta_hat).

The RUL error then splits exactly as the bound says: a window floor set by
sensor noise and independent of n, plus a training term that decays with n.
n_sat is where the second falls below the first, and the theorem's constant-free
prediction is

        n_sat(w2) / n_sat(w1) = ( rho_w1 / rho_w2 )^-2 = (rho_w2/rho_w1)^2 ... (*)

Wait: floor ~ 1/rho_w, so n_sat ~ (1/floor)^2 ~ rho_w^2.  A WIDER window has a
LOWER floor and therefore needs MORE data before the training term stops binding.
"""
import os, json
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
NSAMP = 800          # indicator samples per lifetime (f_s)
BETA_BAR = 3.0
SIGMA_FLEET = 0.02   # deliberately small: keep the fleet term out of the way
SIGMA_M = 0.05
TBAR = 1.0
TAU0 = 0.7
WS = [0.10, 0.20, 0.40]
NS = [2, 3, 5, 8, 13, 21, 34, 55, 89, 144, 233]
DEPLOY = 3000
RNG = np.random.default_rng(31)
tau = np.arange(1, NSAMP + 1) / NSAMP


def dprime(t, b):
    return b * np.power(t, b - 1.0)


def fit_beta(n, rng):
    """Estimate the fleet physics parameter from n fully observed trajectories."""
    num = []
    for _ in range(n):
        b_i = rng.normal(BETA_BAR, SIGMA_FLEET)
        X = tau ** b_i + rng.normal(0, SIGMA_M, NSAMP)
        # least squares in log space on the informative part of the curve
        m = (tau > 0.05) & (X > 1e-3)
        if m.sum() < 20:
            continue
        num.append(float(np.polyfit(np.log(tau[m]), np.log(X[m]), 1)[0]))
    return float(np.mean(num)) if num else BETA_BAR


def rho_w(w, b):
    """Window mass of the information measure, power-law class."""
    lo = max(TAU0 - w, 0.0)
    return (TAU0 ** (2 * b - 1) - lo ** (2 * b - 1))


def deploy_mse(bhat, w, rng, reps=DEPLOY):
    """RUL MSE of the Proposition 1 estimator using the trained beta_hat."""
    kw = max(6, int(round(w * NSAMP)))
    k0 = int(round(TAU0 * NSAMP)) - 1
    idx = np.arange(k0 - kw + 1, k0 + 1)
    t_win = tau[idx]
    g = dprime(t_win, bhat)                       # regressor the predictor believes
    den = float(np.sum(g * g))
    err = np.empty(reps)
    for r in range(reps):
        b_i = rng.normal(BETA_BAR, SIGMA_FLEET)
        delta = rng.uniform(-0.02, 0.02)          # true age offset from the prior
        # what the sensor actually produces at the true age
        Xw = (t_win + delta) ** b_i + rng.normal(0, SIGMA_M, kw)
        y = Xw - t_win ** bhat                    # observed minus believed trend
        dhat = float(np.sum(g * y) / den)
        err[r] = (TBAR * (dhat - delta)) ** 2
    return float(np.mean(err))


print(f"regime: sigma_m={SIGMA_M}, sigma_fleet={SIGMA_FLEET}, f_s={NSAMP}, tau0={TAU0}")
print(f"rho_w: " + "  ".join(f"w={w}: {rho_w(w, BETA_BAR):.4f}" for w in WS))
print(f"\npredicted floor (MSE) = Tbar^2/(Gamma rho_w), "
      f"Gamma = f_s/sigma_m^2 * beta^2/(2beta-1)")
GAMMA = NSAMP / SIGMA_M ** 2 * BETA_BAR ** 2 / (2 * BETA_BAR - 1)
for w in WS:
    print(f"   w={w}: floor = {TBAR**2/(GAMMA*rho_w(w, BETA_BAR)):.3e}")

print(f"\n{'n':>6}" + "".join(f"{'w='+str(w):>13}" for w in WS))
print("-" * (6 + 13 * len(WS)))
curves = {w: [] for w in WS}
for n in NS:
    line = f"{n:>6}"
    bh = np.median([fit_beta(n, RNG) for _ in range(9)])
    for w in WS:
        m = deploy_mse(bh, w, RNG)
        curves[w].append(m)
        line += f"{m:>13.3e}"
    print(line)

print("-" * (6 + 13 * len(WS)))
sat, floors = {}, {}
for w in WS:
    c = np.array(curves[w])
    fl = float(np.min(c[-3:]))
    floors[w] = fl
    idx = int(np.argmax(c <= fl * 1.25))
    sat[w] = NS[idx]
    print(f"w={w}: empirical floor {fl:.3e}, "
          f"theoretical {TBAR**2/(GAMMA*rho_w(w, BETA_BAR)):.3e}, "
          f"n_sat = {NS[idx]}")

print("\n--- the constant-free prediction ---")
base = WS[0]
for w in WS[1:]:
    pred = (rho_w(w, BETA_BAR) / rho_w(base, BETA_BAR)) ** 2
    obs = sat[w] / sat[base]
    ok = "consistent" if 0.35 <= obs / pred <= 3.0 else "NOT consistent"
    print(f"  w={base} -> w={w}: predicted n_sat ratio {pred:6.2f}, "
          f"observed {obs:6.2f}   {ok}")

json.dump(dict(NS=NS, curves={str(k): v for k, v in curves.items()},
               sat={str(k): v for k, v in sat.items()},
               floors={str(k): v for k, v in floors.items()},
               rho={str(w): rho_w(w, BETA_BAR) for w in WS}),
          open(os.path.join(HERE, "saturation_synth.json"), "w"), indent=2)
