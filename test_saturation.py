"""
The first empirical test of Theorem 1.

Theorem 1 says the training-data term stops binding once n exceeds
n_sat proportional to rho_w^2. The constants (B, p_D, Gamma) are awkward to pin
down, but the RATIO between two window widths is constant-free:

        n_sat(w2) / n_sat(w1)  =  ( rho_w2 / rho_w1 )^2

so a wider window -- which captures more information and therefore has a lower
floor -- must keep improving with data for longer before it saturates.

Test: train a windowed RUL predictor on C-MAPSS FD001 at two window widths, sweep
the number of training units, and locate where each error curve flattens.
Predictor is deliberately plain (ridge on window summaries) because the claim is
about the data requirement, not about architecture.
"""
import os, json
import numpy as np
from pipeline import profile

HERE = os.path.dirname(os.path.abspath(__file__))
CH = [2, 3, 4, 7, 8, 9, 11, 12, 13, 14, 15, 17, 20, 21]
RUL_CAP = 125.0
W1, W2 = 0.10, 0.40          # narrow and wide, as fractions of life
NS = [4, 6, 10, 15, 22, 32, 45, 60]
REPEATS = 12
RNG = np.random.default_rng(5)

z = np.load(os.path.join(HERE, "cmapss_FD001.npz"))
units = sorted({k.split("__")[0] for k in z.files})
S = {u: z[f"{u}__sensors"].astype(float) for u in units}
T = {u: len(S[u]) for u in units}

# standardise channels fleet-wide so ridge is well conditioned
allrows = np.vstack([S[u][:, [c - 1 for c in CH]] for u in units])
MU, SD = allrows.mean(0), allrows.std(0)
SD[SD < 1e-9] = 1.0


def design(u, w):
    """Window summaries (level and slope per channel) at every reachable age."""
    X = (S[u][:, [c - 1 for c in CH]] - MU) / SD
    n = T[u]
    kw = max(4, int(round(w * n)))
    rows, y = [], []
    for k0 in range(kw, n):
        win = X[k0 - kw + 1:k0 + 1]
        t = np.linspace(-1, 1, len(win))
        lvl = win.mean(0)
        slp = (win * t[:, None]).sum(0) / max((t ** 2).sum(), 1e-9)
        rows.append(np.concatenate([lvl, slp, [k0 / n]]))
        y.append(min(n - 1 - k0, RUL_CAP))
    return np.asarray(rows), np.asarray(y, float)


CACHE = {w: {u: design(u, w) for u in units} for w in (W1, W2)}


def ridge_rmse(train_units, test_units, w, lam=10.0):
    Xtr = np.vstack([CACHE[w][u][0] for u in train_units])
    ytr = np.concatenate([CACHE[w][u][1] for u in train_units])
    Xte = np.vstack([CACHE[w][u][0] for u in test_units])
    yte = np.concatenate([CACHE[w][u][1] for u in test_units])
    Xtr = np.column_stack([Xtr, np.ones(len(Xtr))])
    Xte = np.column_stack([Xte, np.ones(len(Xte))])
    A = Xtr.T @ Xtr + lam * np.eye(Xtr.shape[1])
    beta = np.linalg.solve(A, Xtr.T @ ytr)
    return float(np.sqrt(np.mean((Xte @ beta - yte) ** 2)))


test_units = units[:30]
pool = units[30:]

print(f"{'n units':>8}" + "".join(f"{'w='+str(w):>12}" for w in (W1, W2)))
print("-" * 32)
curves = {W1: [], W2: []}
for n in NS:
    line = f"{n:>8}"
    for w in (W1, W2):
        vals = []
        for _ in range(REPEATS):
            tr = list(RNG.choice(pool, size=min(n, len(pool)), replace=False))
            vals.append(ridge_rmse(tr, test_units, w))
        m = float(np.median(vals))
        curves[w].append(m)
        line += f"{m:>12.2f}"
    print(line)

# where does each curve flatten? first n within 5% of its own asymptote
print("-" * 32)
sat = {}
for w in (W1, W2):
    c = np.array(curves[w])
    floor = c[-1]
    thr = floor * 1.05
    idx = int(np.argmax(c <= thr))
    sat[w] = NS[idx]
    print(f"w={w}: floor RMSE {floor:.2f}, within 5% of it from n = {NS[idx]}")

# what does Theorem 1 predict for the ratio?
zc = np.load(os.path.join(HERE, "cmapss_FD001.npz"))
rho = {}
for w in (W1, W2):
    vals = []
    for u in units:
        p, _ = profile(zc[f"{u}__sensors"][:, 10].astype(float))
        if not p:
            continue
        n = p["n"]; k0 = int(round(0.7 * n)) - 1
        klo = max(0, k0 - int(round(w * n)))
        vals.append(float(p["F"][k0] - p["F"][klo]))
    rho[w] = float(np.median(vals))

pred = (rho[W2] / rho[W1]) ** 2
obs = sat[W2] / sat[W1]
print(f"\nrho_w: {rho[W1]:.3f} (w={W1})  {rho[W2]:.3f} (w={W2})")
print(f"Theorem 1 predicts n_sat ratio = (rho2/rho1)^2 = {pred:.1f}")
print(f"observed ratio                                = {obs:.1f}")
print(f"{'consistent' if 0.4 <= obs/pred <= 2.5 else 'NOT consistent'} "
      f"(ratio of ratios {obs/pred:.2f})")

json.dump(dict(NS=NS, curves={str(k): v for k, v in curves.items()},
               sat={str(k): v for k, v in sat.items()},
               rho={str(k): v for k, v in rho.items()},
               predicted_ratio=pred, observed_ratio=obs),
          open(os.path.join(HERE, "saturation_test.json"), "w"), indent=2)
