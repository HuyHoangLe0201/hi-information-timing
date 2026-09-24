"""
The local scale rises where the indicator moves fastest. Why?

On real bearing records corr(log sigma_loc, log |D'|) runs 0.23 to 0.69, against
0.03 on a synthetic record with genuinely constant noise. Two readings:

  multiplicative -- the noise really does grow with the signal, in which case
                    dividing by sigma is the correct thing to do and the
                    weighted density is right;
  leakage        -- the trend estimator under-fits where the signal turns
                    fastest, so unmodelled trend lands in the residual and
                    inflates sigma exactly there. Dividing by it would then
                    suppress the informative part of the record.

The statistic alone cannot tell them apart, so each is generated deliberately
and the statistic measured on it. If leakage on its own cannot manufacture a
correlation as large as the real records show, the multiplicative reading is
the one the data support.
"""
import os
import json
import numpy as np
from scipy.signal import savgol_filter
from pipeline import robust_scale, oof_trend, local_scale, _win, LOCAL_BW, SMOOTH_FRAC

HERE = os.path.dirname(os.path.abspath(__file__))
REPS = 40
N = 1200


def leak_corr(x, smooth_frac=None):
    n = len(x)
    if smooth_frac is None:
        w = _win(n)
    else:
        w = max(7, int(smooth_frac * n)); w += (w % 2 == 0)
        w = min(w, n - 1 if (n - 1) % 2 == 1 else n - 2)
    dt = savgol_filter(x, w, 2, deriv=1, delta=1.0 / n)
    res = x - oof_trend(x, w // 2)
    sl = local_scale(res, max(5, int(LOCAL_BW * n)))
    m = np.isfinite(dt) & np.isfinite(sl) & (np.abs(dt) > 0) & (sl > 0)
    if m.sum() < 60:
        return np.nan
    return float(np.corrcoef(np.log(sl[m]), np.log(np.abs(dt[m])))[0, 1])


rng = np.random.default_rng(551)
tau = np.arange(1, N + 1) / N

CASES = [
    ("constant noise, beta=3", 3.0, lambda t: np.full_like(t, 0.04), None),
    ("constant noise, beta=8", 8.0, lambda t: np.full_like(t, 0.04), None),
    ("constant noise, beta=20", 20.0, lambda t: np.full_like(t, 0.04), None),
    ("constant, deliberate underfit", 8.0, lambda t: np.full_like(t, 0.04), 0.30),
    ("constant, severe underfit", 20.0, lambda t: np.full_like(t, 0.04), 0.45),
    ("multiplicative, sigma ~ D", 3.0, lambda t: 0.02 + 0.20 * t ** 3, None),
    ("multiplicative, sigma ~ sqrt D", 3.0,
     lambda t: 0.02 + 0.20 * t ** 1.5, None),
]

print("what the leakage statistic reads on records built to order\n")
print(f"{'construction':<34}{'corr':>8}")
print("-" * 42)
rows = []
for lab, beta, sfn, sf in CASES:
    v = []
    for _ in range(REPS):
        x = tau ** beta + rng.normal(0, 1, N) * sfn(tau)
        c = leak_corr(x, sf)
        if np.isfinite(c):
            v.append(c)
    m = float(np.median(v))
    rows.append(dict(case=lab, corr=m))
    print(f"{lab:<34}{m:>8.3f}")
print("-" * 42)

REAL = {"rms": 0.657, "kurt": 0.693, "0--2 kHz": 0.231, "4--10 kHz": 0.447}
print("\nreal bearing records, for comparison")
for k, v in REAL.items():
    print(f"  {k:<32}{v:>8.3f}")

const = [r["corr"] for r in rows if r["case"].startswith("constant")]
mult = [r["corr"] for r in rows if r["case"].startswith("multiplicative")]
print("-" * 42)
print(f"leakage alone, over every constant-noise construction: "
      f"{min(const):+.3f} to {max(const):+.3f}")
print(f"genuine multiplicative noise:                          "
      f"{min(mult):+.3f} to {max(mult):+.3f}")
print(f"real records:                                          "
      f"{min(REAL.values()):+.3f} to {max(REAL.values()):+.3f}")
print()
if max(const) < min(REAL.values()):
    print("Leakage cannot manufacture the correlation the real records show,")
    print("even when the trend is deliberately under-fitted. The scale really")
    print("does grow with the signal, so the weighted density is the correct")
    print("one and the exponents computed from D'^2 alone are misspecified.")
else:
    print("Leakage alone reproduces the observed correlation, so the statistic")
    print("does not establish multiplicative noise and the weighted density")
    print("cannot be preferred on this evidence.")

json.dump(dict(synthetic=rows, real=REAL),
          open(os.path.join(HERE, "leakage_control.json"), "w"), indent=2)
