r"""How much of the fused information is real, at each degree of degeneracy.

rank_guard3.py measured the seven-channel quadratic form against the six-channel
one and called the ratio inflation.  That conflates two different things.  When
the seventh channel carries independent noise it genuinely adds information, and
the ratio should exceed one; when it does not, any excess is manufactured by
inverting a near-singular estimate.  A ratio of 21 at a sample condition number
of 419 therefore says nothing on its own about whether the guard is too loose.

The construction here has a covariance that is known in closed form, so the
comparison can be made against the truth instead.  With Y standard normal in
K-1 dimensions and the last channel equal to -sum(Y) + delta sqrt(K-1) eps,

    Cov(Y_i, Y_j)   = 1 if i = j else 0
    Cov(Y_i, last)  = -1
    Var(last)       = (K-1)(1 + delta^2) .

The ratio that matters is then the ESTIMATED quadratic form over the TRUE one at
the same sensitivity vector.  One means the estimate is sound however correlated
the channels are; large means the estimate is manufacturing information; and the
genuine contribution of the seventh channel has been divided out on both sides.
"""
import os
import json
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
rng = np.random.default_rng(577215)
OUT = {}
K, REPS = 7, 500


def true_corr(delta):
    S = np.eye(K)
    S[:K - 1, K - 1] = S[K - 1, :K - 1] = -1.0
    S[K - 1, K - 1] = (K - 1) * (1.0 + delta ** 2)
    d = np.sqrt(np.diag(S))
    return S / np.outer(d, d)


def sample(m, delta, rs):
    Y = rs.normal(size=(m, K - 1))
    last = -Y.sum(axis=1) + delta * rs.normal(size=m) * np.sqrt(K - 1)
    return np.column_stack([Y, last])


def quad(X, d, debias=True):
    m, Kc = X.shape
    C = np.corrcoef(X, rowvar=False)
    try:
        q = float(d @ np.linalg.solve(C, d))
    except np.linalg.LinAlgError:
        return np.nan
    return q * ((m - Kc - 1) / m if debias and Kc > 1 else 1.0)


def split(X, d):
    m = len(X)
    a, b = quad(X[: m // 2], d, False), quad(X[m // 2:], d, False)
    if not (np.isfinite(a) and np.isfinite(b)) or min(a, b) <= 0:
        return np.inf
    return max(a, b) / min(a, b)


print("=" * 96)
print("Estimated information over true information, by degree of degeneracy")
print("=" * 96)
print("""
Overstatement is the estimated quadratic form over the true one at the same
sensitivity vector, so one is correct and larger is information the estimate
invented.  The sample condition number is what the guard tests.
""")
for m in (300, 1500):
    print("  m = %d\n" % m)
    print("  %8s %13s %14s %11s %11s"
          % ("delta", "sample cond", "overstatement", "split", "negatives"))
    print("  " + "-" * 62)
    for delta in (1e-4, 1e-3, 1e-2, 0.03, 0.1, 0.3, 1.0, 3.0):
        St = true_corr(delta)
        Sti = np.linalg.inv(St)
        cn, ov, sp, neg = [], [], [], 0
        for _ in range(REPS):
            X = sample(m, delta, rng)
            d = rng.normal(size=K)
            cn.append(float(np.linalg.cond(np.corrcoef(X, rowvar=False))))
            qe = quad(X, d)
            qt = float(d @ Sti @ d)
            if np.isfinite(qe) and qe < 0:
                neg += 1
            if np.isfinite(qe) and qt > 0:
                ov.append(qe / qt)
            sp.append(split(X, d))
        row = dict(m=m, delta=delta, cond=float(np.median(cn)),
                   over=float(np.median(ov)) if ov else float("nan"),
                   split=float(np.median(sp)), negatives=neg / REPS,
                   true_cond=float(np.linalg.cond(St)))
        OUT.setdefault("sweep", []).append(row)
        print("  %8.4g %13.2e %14.3f %11.2f %11.1f%%"
              % (delta, row["cond"], row["over"], row["split"],
                 100 * row["negatives"]))
    print()

# =============================================================================
print("=" * 96)
print("The condition number at which a stated overstatement is reached")
print("=" * 96)
print("""
Reading each sweep for the smallest sample condition number whose overstatement
exceeds a tolerance gives the threshold a guard should use at that record
length.  Compare against the 10^3 the paper fixes.
""")
print("  %6s %16s %16s %16s"
      % ("m", "overstate 1.1x", "overstate 1.5x", "overstate 3x"))
print("  " + "-" * 58)
implied = []
for m in (300, 1500):
    sub = sorted([r for r in OUT["sweep"] if r["m"] == m],
                 key=lambda r: r["cond"])
    got = []
    for tol in (1.1, 1.5, 3.0):
        bad = [r for r in sub if r["over"] > tol]
        got.append(min(r["cond"] for r in bad) if bad else float("nan"))
    implied.append(dict(m=m, t11=got[0], t15=got[1], t30=got[2]))
    print("  %6d %16.1e %16.1e %16.1e" % (m, got[0], got[1], got[2]))
OUT["implied"] = implied
OUT["paper_threshold"] = 1e3

json.dump(OUT, open(os.path.join(HERE, "rank_guard4.json"), "w"), indent=2,
          default=float)
print("\nwritten to rank_guard4.json")
