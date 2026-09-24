"""
Gamma assumes the degradation trend is known. It never is.

Everything so far has computed the information about the damage-clock offset as

    Gamma = sum_k D'(tau_k)^2 / sigma(tau_k)^2,

which is the Fisher information about delta when D is known exactly. In practice
D is estimated -- its amplitude and its exponent come from the fleet -- and a
parameter that has to be estimated alongside delta takes information away from
it. With theta = (delta, phi) and phi the shape parameters, the information
actually available about delta is the Schur complement

    I_{delta|phi}  =  I_{dd}  -  I_{d phi} I_{phi phi}^{-1} I_{phi d}   <=  I_{dd},

with equality only if the delta direction is orthogonal to every shape
direction. For a power law D = A (tau - delta)^beta the derivatives are

    dD/ddelta = -A beta tau^{beta-1},   dD/dA = tau^beta,
    dD/dbeta  =  A tau^beta ln tau,

and the first two differ only by the factor beta/tau, which varies slowly over
most of the record: the clock offset is nearly confounded with the amplitude.
How nearly is a number, computed here.

This is not a correction to Gamma so much as a statement about what Gamma means.
The bound built on I_{dd} is attainable only by an estimator that already knows
the trend -- which is what the validation resampled, so that result stands. Any
absolute claim about achievable precision on a new unit needs I_{delta|phi}.
"""
import os
import json
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))


def info_matrix(tau, beta, A=1.0, sigma=1.0, params=("delta",)):
    """Fisher information matrix for the listed parameters of A(tau-delta)^beta."""
    t = np.clip(tau, 1e-9, None)
    g = {"delta": -A * beta * t ** (beta - 1.0),
         "A": t ** beta,
         "beta": A * t ** beta * np.log(t)}
    G = np.column_stack([g[p] for p in params])
    return (G.T @ G) / sigma ** 2


def retained(tau, beta, nuisance):
    """Fraction of the information about delta that survives the nuisances."""
    params = ("delta",) + tuple(nuisance)
    I = info_matrix(tau, beta, params=params)
    idd = I[0, 0]
    if len(nuisance) == 0:
        return 1.0
    Ipp = I[1:, 1:]
    Idp = I[0, 1:]
    eff = idd - Idp @ np.linalg.solve(Ipp, Idp)
    return float(max(eff, 0.0) / idd)


n = 2000
tau_full = np.arange(1, n + 1) / n

print("fraction of the information about the damage clock that survives when")
print("the trend must be estimated too (whole record, 0 to 1)\n")
print(f"{'beta':>6}{'amplitude':>12}{'exponent':>11}{'both':>9}")
print("-" * 38)
rows = []
for beta in (0.7, 1.0, 1.5, 2.0, 3.0, 5.0, 8.0, 15.0, 25.0):
    a = retained(tau_full, beta, ("A",))
    b = retained(tau_full, beta, ("beta",))
    ab = retained(tau_full, beta, ("A", "beta"))
    rows.append(dict(beta=beta, amp=a, exp=b, both=ab))
    print(f"{beta:>6.1f}{a:>12.4f}{b:>11.4f}{ab:>9.4f}")
print("-" * 38)
print("1.000 would mean the nuisance costs nothing.\n")

print("the same, over a trailing window ending at the age shown\n")
print(f"{'window':>8}" + "".join(f"{b:>9.1f}" for b in (1.0, 3.0, 8.0)))
print("-" * 35)
win_rows = []
for t0, w in ((1.0, 1.0), (1.0, 0.5), (1.0, 0.2), (1.0, 0.1), (0.5, 0.5)):
    lo = max(t0 - w, 1e-6)
    seg = np.linspace(lo, t0, 800)
    cells = []
    for beta in (1.0, 3.0, 8.0):
        cells.append(retained(seg, beta, ("A", "beta")))
    win_rows.append(dict(tau0=t0, w=w, values=cells))
    lab = f"[{lo:.1f},{t0:.1f}]"
    print(f"{lab:>8}" + "".join(f"{c:>9.4f}" for c in cells))
print("-" * 35)
print("A shorter window confounds the clock with the trend more severely,")
print("because over a short span every smooth trend looks like a shifted one.")
print()

worst = min(r["both"] for r in rows)
best = max(r["both"] for r in rows)
print(f"over the whole record the surviving fraction is {worst:.4f} to "
      f"{best:.4f},")
print(f"so the precision claimed by the known-trend bound is optimistic by")
print(f"a factor of {1/np.sqrt(best):.1f} to {1/np.sqrt(worst):.1f} in standard "
      f"deviation.")
json.dump(dict(whole_record=rows, windows=win_rows),
          open(os.path.join(HERE, "nuisance.json"), "w"), indent=2)
