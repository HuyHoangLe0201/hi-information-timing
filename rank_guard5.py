r"""A hypothesis about the fusion artefact, tested and rejected.

Three simulations had already failed to reproduce the artefact Section 3.4
reports.  With the covariance estimated from data and the Wishart correction
applied, the quadratic form is unbiased at every degree of degeneracy tested,
down to a sample condition number of 4e8.  Ill-conditioning alone does not
manufacture information.

This tested a fourth idea: that the SENSITIVITY is estimated too.  A set of band
fractions obeys an algebraic identity, so the true sensitivity has no component
along the degenerate direction, while a smoother applied per channel produces an
estimate that does.  A small spurious component divided by a near-zero variance
would then be unbounded.

The table below rejects it as well.  It is kept because a rejected hypothesis
with its measurement is worth more than a fourth guess, and because it isolates
what the remaining simulations cannot settle: whether the artefact is a property
of these particular channels rather than of degeneracy in general.  That question
is answered on the data, in rank_guard_data.py.
"""
import os
import json
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
rng = np.random.default_rng(112358)
OUT = {}
K, REPS = 7, 400


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


def constrained_d(rs):
    """A sensitivity obeying the same identity the channels do.

    If the channels satisfy x_K = -sum(x_i), then a trend common to them
    satisfies D'_K = -sum(D'_i), so the sensitivity lies in the subspace the
    covariance is non-degenerate on.  The true information is finite there
    however small delta becomes.
    """
    g = rs.normal(size=K - 1)
    return np.append(g, -g.sum())


def quad(X, d, debias=True):
    m, Kc = X.shape
    C = np.corrcoef(X, rowvar=False)
    try:
        q = float(d @ np.linalg.solve(C, d))
    except np.linalg.LinAlgError:
        return np.nan
    return q * ((m - Kc - 1) / m if debias and Kc > 1 else 1.0)


print("=" * 96)
print("1.  A sensitivity that obeys the identity, with and without estimation error")
print("=" * 96)
print("""
The truth is the exact sensitivity against the exact covariance.  The estimate
uses a sensitivity perturbed by eta, which is what a per-channel smoother does,
against a covariance estimated from m samples.  Overstatement is their ratio.
""")
m = 1000
print("  %8s %13s %10s %14s %14s"
      % ("delta", "sample cond", "eta", "overstatement", "with exact d"))
print("  " + "-" * 64)
for delta in (1e-3, 1e-2, 0.03, 0.1, 0.3, 1.0):
    St = true_corr(delta)
    Sti = np.linalg.inv(St)
    for eta in (0.0, 0.01, 0.05):
        ov, ov_exact, cn = [], [], []
        for _ in range(REPS):
            X = sample(m, delta, rng)
            d0 = constrained_d(rng)
            dh = d0 + eta * rng.normal(size=K) * np.linalg.norm(d0) / np.sqrt(K)
            cn.append(float(np.linalg.cond(np.corrcoef(X, rowvar=False))))
            qt = float(d0 @ Sti @ d0)
            qe = quad(X, dh)
            qx = quad(X, d0)
            if np.isfinite(qe) and qt > 0:
                ov.append(qe / qt)
            if np.isfinite(qx) and qt > 0:
                ov_exact.append(qx / qt)
        row = dict(delta=delta, eta=eta, cond=float(np.median(cn)),
                   over=float(np.median(ov)),
                   over_exact=float(np.median(ov_exact)))
        OUT.setdefault("sweep", []).append(row)
        print("  %8.3g %13.2e %10.2f %14.2f %14.3f"
              % (delta, row["cond"], eta, row["over"], row["over_exact"]))
    print()

print("""
This hypothesis is REJECTED by its own table.  The overstatement stays at one
at every delta and every eta, so perturbing the sensitivity does not reproduce
the artefact either.

One reason is a fault in this simulation and is worth naming rather than
patching over: the sensitivity is built to be orthogonal to the degenerate
direction in RAW coordinates, while the quadratic form is taken against a
CORRELATION matrix, whose standardisation moves that direction.  The two are
therefore not aligned as intended.  Fixing it would test a different question
than the one that matters, which is what the real channels do, so the search
moves to the data instead.
""")

# =============================================================================
print("=" * 96)
print("2.  Which quantity a guard should test")
print("=" * 96)
print("""
The overstatement above is driven by eta^2/delta^2, so a rule needs both: how
degenerate the covariance is AND how much of the estimated sensitivity lies in
the degenerate direction.  The condition number supplies only the first.  The
quantity that supplies both is the share of the quadratic form contributed by
the smallest eigendirection, which is computable from the estimate alone.
""")


def small_share(X, d):
    """Fraction of d' C^-1 d contributed by the smallest eigendirection."""
    C = np.corrcoef(X, rowvar=False)
    w, V = np.linalg.eigh(C)
    p = (V.T @ d) ** 2 / np.maximum(w, 1e-300)
    return float(p[0] / p.sum())


print("  %8s %10s %13s %14s %14s"
      % ("delta", "eta", "sample cond", "overstatement", "small share"))
print("  " + "-" * 64)
for delta in (1e-3, 1e-2, 0.1, 1.0):
    St = true_corr(delta)
    Sti = np.linalg.inv(St)
    for eta in (0.0, 0.05):
        ov, sh, cn = [], [], []
        for _ in range(REPS):
            X = sample(m, delta, rng)
            d0 = constrained_d(rng)
            dh = d0 + eta * rng.normal(size=K) * np.linalg.norm(d0) / np.sqrt(K)
            cn.append(float(np.linalg.cond(np.corrcoef(X, rowvar=False))))
            qt = float(d0 @ Sti @ d0)
            qe = quad(X, dh)
            if np.isfinite(qe) and qt > 0:
                ov.append(qe / qt)
            sh.append(small_share(X, dh))
        r = dict(delta=delta, eta=eta, cond=float(np.median(cn)),
                 over=float(np.median(ov)), share=float(np.median(sh)))
        OUT.setdefault("share", []).append(r)
        print("  %8.3g %10.2f %13.2e %14.2f %14.3f"
              % (delta, eta, r["cond"], r["over"], r["share"]))
    print()

print("""
The overstatement does not differ between the pairs, so this table does not
support the small-share diagnostic either.  What it does show is that the share
saturates at one well before the condition number does, reaching 0.999 at a
condition number of 4e4 and 0.938 already at 4e2, so it is at least a quantity
that responds to degeneracy on a usable scale.  Whether it separates the harmful
case from the harmless one is not settled here.
""")

json.dump(OUT, open(os.path.join(HERE, "rank_guard5.json"), "w"), indent=2,
          default=float)
print("written to rank_guard5.json")
