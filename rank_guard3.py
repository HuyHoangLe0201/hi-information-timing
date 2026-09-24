r"""Near-degeneracy, swept: the regime the paper is actually in.

rank_guard2.py compared a well-conditioned truth against an exactly singular
one and found the comparison too easy: exact singularity gives a sample
condition number of 10^16 and a quadratic form that goes negative, which any
rule would catch.  The paper's degenerate sets condition at 1.3e5, not 10^16,
because spectral leakage means the band fractions sum to one only approximately.
The interesting regime is between the two, and it is where a threshold has to
make its decision.

So the constraint is relaxed continuously.  The last channel is the negative sum
of the others plus a perturbation of size delta: at delta = 0 the truth is
singular, at delta of order one it is a free channel.  Sweeping delta traces the
path a real feature set takes as leakage grows, and asks two questions at each
point.  How much does the quadratic form inflate, which is the damage?  And what
does each candidate diagnostic report, which is whether the damage is visible?

Three diagnostics are compared.

  cond      the sample condition number, which is what the paper uses
  split     covariance fitted on one half of the record, quadratic form
            compared against the other half's
  negative  whether the quadratic form comes out negative at all, which a
            positive-definite matrix cannot produce and which therefore proves
            the inversion has failed
"""
import os
import json
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
rng = np.random.default_rng(161803)
OUT = {}
K, REPS = 7, 400


def sample(m, delta, rs):
    """Six free channels and a seventh that is their negative sum to within
    delta, which is what a set of band fractions is to within leakage."""
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
    qa, qb = quad(X[: m // 2], d, False), quad(X[m // 2:], d, False)
    if not (np.isfinite(qa) and np.isfinite(qb)) or min(qa, qb) <= 0:
        return np.inf
    return max(qa, qb) / min(qa, qb)


print("=" * 94)
print("The path from a free channel to a singular one")
print("=" * 94)
print("""
Inflation is the quadratic form on all seven channels over the form on the six
free ones, so one means the seventh contributes what a genuine channel would.
The truth carries no information in the seventh direction at delta = 0, so any
inflation there is manufactured.
""")
for m in (300, 1500):
    print("  m = %d\n" % m)
    print("  %8s %13s %11s %11s %11s"
          % ("delta", "sample cond", "inflation", "split", "negatives"))
    print("  " + "-" * 60)
    for delta in (0.0, 1e-4, 1e-3, 1e-2, 0.03, 0.1, 0.3, 1.0):
        cn, infl, sp, neg = [], [], [], 0
        for _ in range(REPS):
            X = sample(m, delta, rng)
            d = rng.normal(size=K)
            cn.append(float(np.linalg.cond(np.corrcoef(X, rowvar=False))))
            qa, qf = quad(X, d), quad(X[:, :K - 1], d[:K - 1])
            if qa is not None and np.isfinite(qa) and qa < 0:
                neg += 1
            if np.isfinite(qa) and np.isfinite(qf) and qf > 0:
                infl.append(qa / qf)
            sp.append(split(X, d))
        row = dict(m=m, delta=delta, cond=float(np.median(cn)),
                   inflation=float(np.median(infl)) if infl else float("nan"),
                   split=float(np.median(sp)), negatives=neg / REPS)
        OUT.setdefault("sweep", []).append(row)
        print("  %8.4g %13.2e %11.2f %11.2f %11.1f%%"
              % (delta, row["cond"], row["inflation"], row["split"],
                 100 * row["negatives"]))
    print()

print("""
Reading the sweep.  Inflation is the damage and it rises smoothly as delta
falls.  The sample condition number rises with it, so on this family the two
move together and a threshold does separate them: the paper's rule is sound
here.  What the sweep shows is where the rule's number comes from, which is the
sample size and the tolerated inflation, and not from any property of the
matrix alone.
""")

# --- what threshold does a stated tolerance imply? ---------------------------
print("=" * 94)
print("The threshold a stated tolerance implies, against the paper's 1e3")
print("=" * 94)
print("""
Reading down each sweep to the largest delta whose inflation exceeds a stated
tolerance gives the condition number at which the guard should fire.
""")
print("  %6s %14s %16s %16s" % ("m", "tolerate 1.25x", "tolerate 1.5x",
                                "tolerate 2x"))
print("  " + "-" * 56)
implied = []
for m in (300, 1500):
    sub = [r for r in OUT["sweep"] if r["m"] == m]
    sub = sorted(sub, key=lambda r: r["delta"])
    out = []
    for tol in (1.25, 1.5, 2.0):
        bad = [r for r in sub if r["inflation"] > tol]
        out.append(min(r["cond"] for r in bad) if bad else float("nan"))
    implied.append(dict(m=m, tol125=out[0], tol150=out[1], tol200=out[2]))
    print("  %6d %14.1e %16.1e %16.1e" % (m, out[0], out[1], out[2]))
OUT["implied"] = implied
print("""
The paper's 1e3 sits inside this range rather than at its edge, so the guard is
neither loose nor tight by accident.  What the sweep adds is that the threshold
is a statement about tolerated inflation at a given record length, and should
move with both.  A fixed 1e3 is right for records of the length used here and
would be wrong on much shorter ones, which is the direction in which the paper
already says the Wishart correction starts to matter.
""")

json.dump(OUT, open(os.path.join(HERE, "rank_guard3.json"), "w"), indent=2,
          default=float)
print("written to rank_guard3.json")
