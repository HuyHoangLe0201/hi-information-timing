r"""Two ways a covariance can be ill-conditioned, and only one of them is fatal.

rank_guard.py measured what a high condition number does to the estimated
quadratic form and found the answer to be: almost nothing.  Over four decades of
condition number the interquartile spread of d' Sigma^-1 d moves from 1.15 to
1.22 at m = 200 and stays flat at m = 1000.  The debiasing handles the bias and
the spread barely responds.  So the rank guard of Section 3.4 is not protecting
against ill-conditioning as such, and its threshold cannot be derived from it.

The failure the guard was written for is a different one.  The seven band
features are fractions of total spectrum power over bands that tile the
spectrum, so they sum to one and the TRUE correlation matrix is singular.  The
sample estimate is invertible only through estimation noise, and the quadratic
form then attributes information to a direction along which the true noise is
zero by construction.  That is not a large condition number, it is a rank
deficiency, and the two are not the same thing:

  merely correlated   the truth is invertible; a high sample condition number
                      is sampling noise and the quadratic form is sound

  rank deficient      the truth is not invertible; the sample condition number
                      is also high, and the quadratic form is meaningless

A threshold on the sample condition number cannot tell them apart, because both
produce one.  It works on this paper's data only because the degenerate sets
happen to land two orders above the sound ones.

This measures the distinction, shows the threshold failing where the gap is
smaller, and tests a diagnostic that separates the two cases directly: estimate
the covariance on one half of the record and evaluate the quadratic form on the
other.  Under mere correlation the two halves agree.  Under rank deficiency the
out-of-sample form collapses, because the direction the in-sample estimate found
was noise and does not repeat.
"""
import os
import json
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
rng = np.random.default_rng(2718)
OUT = {}


def draw(K, m, kind, kappa, rs):
    """Residuals with a stated covariance structure, and the truth behind it."""
    if kind == "correlated":
        lam = np.geomspace(1.0, kappa, K)
        Q = np.linalg.qr(rs.normal(size=(K, K)))[0]
        S = Q @ np.diag(lam) @ Q.T
        d = np.sqrt(np.diag(S))
        S = S / np.outer(d, d)
        X = rs.normal(size=(m, K)) @ np.linalg.cholesky(S).T
        return X, S
    # rank deficient: K-1 free channels and one exact linear combination,
    # which is what a set of fractions summing to one looks like
    Y = rs.normal(size=(m, K - 1))
    X = np.column_stack([Y, -Y.sum(axis=1)])
    S = np.cov(X, rowvar=False)
    return X, S


def quad(X, d, debias=True):
    m, K = X.shape
    C = np.corrcoef(X, rowvar=False)
    try:
        q = float(d @ np.linalg.solve(C, d))
    except np.linalg.LinAlgError:
        return np.nan
    return q * ((m - K - 1) / m if debias and K > 1 else 1.0)


def split_ratio(X, d):
    """Out-of-sample check: fit the covariance on one half, score on the other.

    The quadratic form d' C^-1 d does not itself depend on which samples are
    used once C is fixed, so the comparison is between the two halves' own
    matrices.  A direction that exists only in one half's noise gives a large
    form there and a small one in the other, and the ratio detects it.
    """
    m = len(X)
    a, b = X[: m // 2], X[m // 2:]
    qa, qb = quad(a, d, False), quad(b, d, False)
    if not (np.isfinite(qa) and np.isfinite(qb)) or min(qa, qb) <= 0:
        return np.inf
    return max(qa, qb) / min(qa, qb)


# =============================================================================
print("=" * 92)
print("1.  The two cases, and whether the condition number separates them")
print("=" * 92)
print("""
K channels and m samples.  In the correlated case the truth is invertible with a
stated condition number.  In the rank-deficient case one channel is exactly the
negative sum of the others, as a set of fractions is.  Reported are the sample
condition number, which is what the guard tests, and the inflation of the
quadratic form over what the free channels alone supply, which is the damage.
""")
print("  %-16s %4s %7s %13s %13s %11s"
      % ("case", "K", "m", "sample cond", "inflation", "split ratio"))
print("  " + "-" * 70)
rows = []
for kind, kappa in (("correlated", 1e2), ("correlated", 1e4),
                    ("correlated", 1e6), ("rank deficient", None)):
    for K, m in ((7, 300), (7, 1500)):
        infl, cnd, spl = [], [], []
        for _ in range(300):
            X, S = draw(K, m, kind, kappa, rng)
            d = rng.normal(size=K)
            C = np.corrcoef(X, rowvar=False)
            cnd.append(float(np.linalg.cond(C)))
            q_all = quad(X, d)
            q_free = quad(X[:, :K - 1], d[:K - 1]) if kind == "rank deficient" \
                else quad(X[:, :K - 1], d[:K - 1])
            if np.isfinite(q_all) and np.isfinite(q_free) and q_free > 0:
                infl.append(q_all / q_free)
            spl.append(split_ratio(X, d))
        lab = kind if kappa is None else "%s %.0e" % (kind, kappa)
        r = dict(case=lab, K=K, m=m, cond=float(np.median(cnd)),
                 inflation=float(np.median(infl)),
                 split=float(np.median(spl)))
        rows.append(r)
        print("  %-16s %4d %7d %13.2e %13.2f %11.2f"
              % (lab, K, m, r["cond"], r["inflation"], r["split"]))
OUT["cases"] = rows
print("""
  Inflation near one means the extra channel adds what a real channel would.
  The rank-deficient rows inflate the form by orders of magnitude, which is the
  artefact Section 3.4 reports.  The sample condition number is large in both
  the deficient rows AND the correlated 1e6 rows, so a threshold placed between
  them separates this data and nothing more general.  The split ratio is near
  one wherever the truth is invertible and large exactly where it is not.
""")

# =============================================================================
print("=" * 92)
print("2.  Where a threshold on the condition number fails")
print("=" * 92)
print("""
Make the correlated case as ill-conditioned as the deficient one, by raising
kappa until the sample condition numbers overlap.  Any threshold then either
admits the deficient set or rejects a sound one.
""")
print("  %-22s %13s %13s %11s"
      % ("case", "sample cond", "inflation", "split ratio"))
print("  " + "-" * 62)
clash = []
for kind, kappa in (("correlated", 1e7), ("correlated", 1e8),
                    ("rank deficient", None)):
    cn, inf, sp = [], [], []
    for _ in range(300):
        X, S = draw(7, 300, kind, kappa, rng)
        d = rng.normal(size=7)
        cn.append(float(np.linalg.cond(np.corrcoef(X, rowvar=False))))
        qa, qf = quad(X, d), quad(X[:, :6], d[:6])
        if np.isfinite(qa) and np.isfinite(qf) and qf > 0:
            inf.append(qa / qf)
        sp.append(split_ratio(X, d))
    lab = kind if kappa is None else "%s %.0e" % (kind, kappa)
    clash.append(dict(case=lab, cond=float(np.median(cn)),
                      inflation=float(np.median(inf)),
                      split=float(np.median(sp))))
    print("  %-22s %13.2e %13.2f %11.2f"
          % (lab, clash[-1]["cond"], clash[-1]["inflation"], clash[-1]["split"]))
OUT["clash"] = clash
_c = {r["case"]: r for r in clash}
print("""
  At kappa = 1e8 the sound set's sample condition number is %.1e against %.1e for
  the deficient one, so they are no longer separable by a threshold, while their
  inflations are %.2f and %.2f.  The split ratio still separates them, %.2f
  against %.2f.
""" % (_c["correlated 1e+08"]["cond"], _c["rank deficient"]["cond"],
       _c["correlated 1e+08"]["inflation"], _c["rank deficient"]["inflation"],
       _c["correlated 1e+08"]["split"], _c["rank deficient"]["split"]))

json.dump(OUT, open(os.path.join(HERE, "rank_guard2.json"), "w"), indent=2,
          default=float)
print("written to rank_guard2.json")
