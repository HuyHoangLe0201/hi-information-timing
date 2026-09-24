"""
Diagnostic -- is the power-law class actually a good description of the
PRONOSTIA health indicator, and how much does the choice of indicator move
the calibrated exponent?

Compares four standard indicator constructions on the same raw RMS series:
  raw      : RMS, normalized to D(0)=0, D(1)=1
  smooth   : centred moving average of RMS over 5% of life
  logrms   : log(RMS/baseline), normalized  (compresses the end-of-life spike)
  monotone : running maximum of the smoothed RMS (enforces monotone degradation)

For each we report the fitted beta and the coefficient of determination of the
power-law fit. A class that only fits at the bound of the search interval is
flagged.
"""
import os
import numpy as np
from scipy.optimize import minimize_scalar

HERE = os.path.dirname(os.path.abspath(__file__))
z = np.load(os.path.join(HERE, "hi_raw.npz"))
BEARINGS = sorted({k.split("__")[0] for k in z.files if not k.startswith("meta__")})

BMAX = 200.0


def norm01(x):
    n = len(x)
    base = np.median(x[:max(3, int(0.05 * n))])
    end = np.median(x[-3:])
    if end <= base:
        return None
    return (x - base) / (end - base)


def variants(r):
    n = len(r)
    win = max(3, int(0.05 * n) | 1)
    kern = np.ones(win) / win
    sm = np.convolve(r, kern, mode="same")
    sm[:win] = r[:win].mean()
    sm[-win:] = r[-win:].mean()
    base = np.median(r[:max(3, int(0.05 * n))])
    lg = np.log(np.maximum(r, 1e-9) / max(base, 1e-9))
    return {"raw": norm01(r), "smooth": norm01(sm),
            "logrms": norm01(lg), "monotone": norm01(np.maximum.accumulate(sm))}


def fit(tau, D):
    f = lambda b: float(np.sum((D - tau ** b) ** 2))
    res = minimize_scalar(f, bounds=(1.001, BMAX), method="bounded",
                          options={"xatol": 1e-4})
    beta = float(res.x)
    ss_res = f(beta)
    ss_tot = float(np.sum((D - D.mean()) ** 2))
    return beta, 1.0 - ss_res / ss_tot


print(f"{'bearing':<13}" + "".join(f"{v:>22}" for v in ["raw", "smooth", "logrms", "monotone"]))
print(f"{'':<13}" + "".join(f"{'beta':>12}{'R2':>10}" for _ in range(4)))
print("-" * 101)

acc = {v: [] for v in ["raw", "smooth", "logrms", "monotone"]}
for b in BEARINGS:
    r = z[f"{b}__h"]
    n = len(r)
    tau = np.arange(1, n + 1) / n
    line = f"{b:<13}"
    for v in ["raw", "smooth", "logrms", "monotone"]:
        D = variants(r)[v]
        if D is None:
            line += f"{'--':>12}{'--':>10}"
            continue
        beta, r2 = fit(tau, D)
        acc[v].append((beta, r2))
        flag = "*" if beta > BMAX * 0.99 else " "
        line += f"{beta:>11.1f}{flag}{r2:>10.3f}"
    print(line)

print("-" * 101)
line = f"{'median':<13}"
for v in ["raw", "smooth", "logrms", "monotone"]:
    a = np.array(acc[v])
    line += f"{np.median(a[:, 0]):>12.1f}{np.median(a[:, 1]):>10.3f}"
print(line)
print("\n* = fit pinned at the search bound (beta effectively unbounded: the")
print("  indicator is flat until a terminal step, which no finite power law fits)")
