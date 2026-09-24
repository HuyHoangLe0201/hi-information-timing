"""
Is the measured cost of a FIFO buffer real, or an artefact of smoothing?

The retention numbers come from a density estimated with a Savitzky-Golay
derivative. Smoothing flattens peaks, and a flatter density makes the level-set
optimum look less advantageous -- so the measured penalty could be biased.

Two synthetic cases pin the bias down, because their answers are known exactly:

  power-law, beta > 1 : the density rises monotonically, so the trailing window
                        IS the optimal retention set. True efficiency = 1.000.
                        Anything the pipeline reports below 1 is pure bias.
  exponential         : the density falls monotonically, so the optimum is the
                        earliest w. True efficiency computed in closed form.
"""
import os, json
import numpy as np
from pipeline import profile

HERE = os.path.dirname(os.path.abspath(__file__))
TAU0 = 0.7
WS = (0.10, 0.20, 0.30)
REPS = 30


def eff_measured(dens, k0, kw):
    best = float(np.sort(dens[:k0 + 1])[::-1][:kw].sum())
    trail = float(dens[max(0, k0 - kw + 1):k0 + 1].sum())
    return trail / best if best > 0 else np.nan


def infl_measured(dens, k0, kw):
    target = float(np.sort(dens[:k0 + 1])[::-1][:kw].sum())
    c = np.cumsum(dens[:k0 + 1][::-1])
    j = int(np.searchsorted(c, target))
    return (j + 1) / kw if j < len(c) else np.inf


def true_exponential(alpha, w):
    """Closed form: density ~ exp(-2 a t); optimum is [0,w], trailing is [t0-w,t0]."""
    f = lambda t: (1 - np.exp(-2 * alpha * t)) / (2 * alpha)
    trail = f(TAU0) - f(TAU0 - w)
    best = f(w) - f(0.0)
    return trail / best


print(f"{'case':<26}{'n':>6}{'sigma':>8}{'w':>6}{'true':>8}{'measured':>10}{'bias':>8}")
print("-" * 72)
rng = np.random.default_rng(41)
rows = []

for label, beta, alpha, sigma, n in [
        ("power-law b=3 (clean)", 3.0, None, 0.01, 1400),
        ("power-law b=3 (noisy)", 3.0, None, 0.10, 1400),
        ("power-law b=3 (short)", 3.0, None, 0.05, 150),
        ("exponential a=2 (clean)", None, 2.0, 0.01, 1400),
        ("exponential a=2 (noisy)", None, 2.0, 0.10, 1400)]:
    tau = np.arange(1, n + 1) / n
    D = tau ** beta if beta else (1 - np.exp(-alpha * tau)) / (1 - np.exp(-alpha))
    for w in WS:
        truth = 1.0 if beta else true_exponential(alpha, w)
        vals = []
        for _ in range(REPS):
            u, _ = profile(D + rng.normal(0, sigma, n))
            if not u:
                continue
            k0 = int(round(TAU0 * n)) - 1
            kw = max(3, int(round(w * n)))
            vals.append(eff_measured(u["dens"], k0, kw))
        if not vals:
            continue
        m = float(np.median(vals))
        rows.append(dict(case=label, w=w, true=truth, measured=m, bias=m - truth))
        print(f"{label:<26}{n:>6}{sigma:>8.2f}{w:>6.2f}{truth:>8.3f}"
              f"{m:>10.3f}{m-truth:>8.3f}")
    print()

print("-" * 72)
pl = [r for r in rows if r["case"].startswith("power-law")]
print(f"power-law cases: true efficiency is exactly 1.000 by construction;")
print(f"  measured {min(r['measured'] for r in pl):.3f} .. "
      f"{max(r['measured'] for r in pl):.3f}")
ex = [r for r in rows if r["case"].startswith("exponential")]
print(f"exponential cases: median |bias| = "
      f"{np.median([abs(r['bias']) for r in ex]):.3f}")
print("\nA measured efficiency BELOW the truth means the penalty is overstated;")
print("ABOVE means understated, and the reported FIFO cost is a lower bound.")
json.dump(rows, open(os.path.join(HERE, "retention_validate.json"), "w"), indent=2)
