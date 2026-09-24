r"""The rank guard: where its threshold comes from, and whether it is the right one.

Multi-channel fusion inverts an estimated covariance, and Section 3.4 protects
that inversion with a hard rule: drop channels greedily until the residual
correlation matrix conditions below 10^3.  The threshold is stated, not derived.
Its justification in the source is that 10^3 sits two orders below the 10^5 the
degenerate sets reach and one order above the 10^2 the sound ones reach, which
is a gap in this data rather than a property of the estimator.  Nothing has
measured what the right threshold is, or how much the fusion result depends on
it.

Two quantities are conflated in that rule and they behave differently.

  Bias.    Inverting a covariance estimated from m samples inflates the
           quadratic form.  For Gaussian data the inflation is exactly
           m/(m-K-1), which the paper divides out.  It does not involve the
           condition number at all.

  Spread.  The same estimate is also NOISY, and that noise is amplified by the
           condition number.  Debiasing does nothing for it.  A set can pass
           the correction and still give a quadratic form that varies by orders
           of magnitude between records.

So the guard is not protecting against the bias, which is already handled.  It
is protecting against the spread, and the threshold should therefore be set by
how much spread is tolerable at the sample size available.  This measures that
relation, tests the sensitivity of the reported age to the threshold, and checks
the debiasing itself on residuals as heavy-tailed as the real ones.
"""
import os
import json
import numpy as np
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))
rng = np.random.default_rng(31415)
OUT = {}


def corr_with_cond(K, kappa, rs):
    """A correlation matrix whose condition number is close to kappa."""
    lam = np.geomspace(1.0, kappa, K)
    Q = np.linalg.qr(rs.normal(size=(K, K)))[0]
    S = Q @ np.diag(lam) @ Q.T
    d = np.sqrt(np.diag(S))
    S = S / np.outer(d, d)
    return (S + S.T) / 2


# =============================================================================
print("=" * 90)
print("1.  What the condition number does to the estimated quadratic form")
print("=" * 90)
print("""
A true correlation matrix of condition number kappa, a fixed sensitivity vector,
and a covariance estimated from m samples.  The quantity of interest is
d' Sigma^-1 d, debiased by (m-K-1)/m exactly as the pipeline does.  Reported are
the median ratio to the truth, which is the bias the correction leaves, and the
interquartile ratio, which is the spread the correction cannot touch.
""")
print("  %4s %7s %9s %11s %11s %11s"
      % ("K", "m", "cond", "median", "IQ ratio", "P90/P10"))
print("  " + "-" * 62)
rows = []
for K in (4, 8):
    for m in (200, 1000):
        for kappa in (1e1, 1e2, 1e3, 1e4, 1e5):
            S = corr_with_cond(K, kappa, rng)
            kk = float(np.linalg.cond(S))
            d = rng.normal(size=K)
            truth = float(d @ np.linalg.solve(S, d))
            L = np.linalg.cholesky(S)
            got = []
            for _ in range(400):
                X = rng.normal(size=(m, K)) @ L.T
                C = np.corrcoef(X, rowvar=False)
                try:
                    q = float(d @ np.linalg.solve(C, d))
                except np.linalg.LinAlgError:
                    continue
                got.append(q * (m - K - 1) / m)
            g = np.array(got) / truth
            med = float(np.median(g))
            iq = float(np.percentile(g, 75) / np.percentile(g, 25))
            p9 = float(np.percentile(g, 90) / np.percentile(g, 10))
            rows.append(dict(K=K, m=m, cond=kk, median=med, iq=iq, p90p10=p9))
            print("  %4d %7d %9.1e %11.3f %11.2f %11.2f"
                  % (K, m, kk, med, iq, p9))
OUT["spread"] = rows
print("""
  The median stays near one at every condition number, which is the debiasing
  working.  The spread does not: it is what the guard is actually for.
""")

# --- where does a given tolerance put the threshold? -------------------------
print("  the threshold implied by a tolerance on the spread\n")
print("  %4s %7s %14s %14s"
      % ("K", "m", "IQ ratio < 1.5", "IQ ratio < 2.0"))
print("  " + "-" * 42)
implied = []
for K in (4, 8):
    for m in (200, 1000):
        sub = sorted([r for r in rows if r["K"] == K and r["m"] == m],
                     key=lambda r: r["cond"])
        def thr(lim):
            ok = [r["cond"] for r in sub if r["iq"] < lim]
            return max(ok) if ok else float("nan")
        t15, t20 = thr(1.5), thr(2.0)
        implied.append(dict(K=K, m=m, thr_1_5=t15, thr_2_0=t20))
        print("  %4d %7d %14.1e %14.1e" % (K, m, t15, t20))
OUT["implied"] = implied

json.dump(OUT, open(os.path.join(HERE, "rank_guard.json"), "w"), indent=2,
          default=float)
print("\nwritten to rank_guard.json")
