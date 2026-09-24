"""
Is the nuisance cost achievable, or only an information ratio?

Proposition 3 says that profiling out an unknown amplitude and exponent leaves
the fraction r = 1/(16 beta^4) of the information about the damage clock.  That
is a statement about Fisher information.  Whether an estimator that must ACTUALLY
fit the trend loses exactly that factor is a different question, and the paper
has not asked it: a bound on information is not a bound on what a fitting
procedure achieves, and the ladder from oracle to practice has so far been
measured only at its two ends (Section 6, a factor of thirty-four).

The ladder is built here rung by rung on synthetic records where every rung is
known:

  (i)   the oracle, handed A and beta, estimating only the clock offset.  Its
        standard deviation should be G^{-1/2}.
  (ii)  the same estimator required to fit A as well.  Proposition 3 predicts
        (r_A G)^{-1/2} with r_A = 1/(4 beta^2).
  (iii) required to fit A and beta.  Predicted (r G)^{-1/2}, r = 1/(16 beta^4).

If (ii) and (iii) land on their predictions, Proposition 3 is achievable and the
oracle-to-practice gap has a named, computable first term.  If they land above
them, the theorem bounds something the fitting cannot reach and the paper must
say so.

The offsets are kept well inside the linearisation radius of Section 6.6, since
outside it the quadratic term rather than the nuisance would dominate and the
test would measure the wrong thing.
"""
import os
import json
import numpy as np
from scipy.optimize import least_squares

HERE = os.path.dirname(os.path.abspath(__file__))
SEED = 20260828
N = 600
REPS = 600
BETAS = (1.5, 2.0, 3.0, 5.0)
SNRS = (300.0, 1000.0, 3000.0)


def model(tau, delta, A, beta):
    return A * np.clip(tau + delta, 1e-9, None) ** beta


def fisher_G(tau, beta, A, sigma):
    return float((((A * beta * tau ** (beta - 1.0)) / sigma) ** 2).sum())


def fit(tau, y, beta0, A0, free, sigma):
    """Least squares over the free parameters; delta is always free."""
    def resid(p):
        d = p[0]
        A = p[1] if "A" in free else A0
        b = p[2] if "A" in free and "b" in free else (
            p[1] if "b" in free and "A" not in free else beta0)
        return (y - model(tau, d, A, b)) / sigma

    p0 = [0.0]
    if "A" in free:
        p0.append(A0)
    if "b" in free:
        p0.append(beta0)
    try:
        s = least_squares(resid, p0, method="lm", max_nfev=4000)
        return float(s.x[0])
    except Exception:
        return np.nan


print("achievability of the nuisance cost, synthetic records\n")
print(f"n = {N} samples, {REPS} repetitions per cell, offsets at the truth\n")
print(f"{'beta':>6}{'G':>11}{'rung':>22}{'predicted sd':>15}"
      f"{'measured sd':>14}{'ratio':>8}")
print("-" * 76)
rng = np.random.default_rng(SEED)
rows = []
tau = (np.arange(N) + 1.0) / N
for beta in BETAS:
    for snr in SNRS:
        A = 1.0
        sigma = A / snr
        G = fisher_G(tau, beta, A, sigma)
        truth = model(tau, 0.0, A, beta)
        est = {"oracle": [], "A": [], "Ab": []}
        for _ in range(REPS):
            y = truth + rng.normal(0.0, sigma, size=N)
            est["oracle"].append(fit(tau, y, beta, A, (), sigma))
            est["A"].append(fit(tau, y, beta, A, ("A",), sigma))
            est["Ab"].append(fit(tau, y, beta, A, ("A", "b"), sigma))
        for rung, lab, r in (("oracle", "trend known", 1.0),
                             ("A", "amplitude fitted", 1.0 / (4 * beta ** 2)),
                             ("Ab", "amplitude and exponent",
                              1.0 / (16 * beta ** 4))):
            v = np.array(est[rung], float)
            v = v[np.isfinite(v)]
            if len(v) < REPS // 2:
                continue
            sd = float(np.std(v, ddof=1))
            pred = 1.0 / np.sqrt(r * G)
            rows.append(dict(beta=beta, snr=snr, G=G, rung=lab, r=r,
                             predicted=pred, measured=sd, ratio=sd / pred,
                             kept=len(v)))
            print(f"{beta:>6.1f}{G:>11.2e}{lab:>22}{pred:>15.4f}{sd:>14.4f}"
                  f"{sd / pred:>8.2f}")
    print()
print("-" * 76)

for lab in ("trend known", "amplitude fitted", "amplitude and exponent"):
    v = np.array([r["ratio"] for r in rows if r["rung"] == lab])
    if len(v):
        print(f"{lab:<24} measured over predicted: median {np.median(v):.2f}, "
              f"range {v.min():.2f} to {v.max():.2f}")
print()
ok = {lab: np.median([r["ratio"] for r in rows if r["rung"] == lab])
      for lab in ("trend known", "amplitude fitted", "amplitude and exponent")}
if all(0.8 <= v <= 1.25 for v in ok.values()):
    print("Every rung lands on its prediction, so Proposition 3 is achievable")
    print("and not merely an information ratio: an estimator that must fit the")
    print("trend loses exactly the factor the theorem names.")
else:
    print("At least one rung departs from its prediction by more than a quarter,")
    print("so the theorem bounds information the fitting does not convert; the")
    print("departure is reported rather than the theorem restated.")

# --- what the ladder implies for the measured gap ---------------------------
print()
try:
    EM = json.load(open(os.path.join(HERE, "exponent_measure.json")))
    OM = json.load(open(os.path.join(HERE, "oracle_vs_model.json")))
    beta = EM["pooled_beta"]
    r = 1.0 / (16 * beta ** 4)
    nuis = 1.0 / np.sqrt(r)
    total = OM[0]["ridge"] / OM[0]["oracle"]
    print(f"At the exponent these records show, {beta:.1f}, the nuisance rung "
          f"alone costs")
    print(f"a factor {nuis:.0f} in error.  Section 6 measures the whole "
          f"oracle-to-ridge gap")
    print(f"at {total:.0f}.  The nuisance term therefore does not merely "
          f"contribute to that")
    print(f"gap, it {'exceeds' if nuis > total else 'falls short of'} it, which "
          f"says the two are not")
    print(f"measured on the same quantity: Section 6 sweeps a fixed synthetic "
          f"trend")
    print(f"shape while beta = {beta:.1f} is read off real bearings.  The "
          f"honest statement")
    print(f"is that the nuisance cost is large enough to account for a gap of "
          f"that")
    print(f"size, not that it has been shown to be its cause.")
    implied = float(nuis)
except FileNotFoundError:
    implied = None
    print("(comparison skipped: run exponent_measure.py and oracle_vs_model.py)")

json.dump(dict(n=N, reps=REPS, betas=list(BETAS), snrs=list(SNRS),
               rungs=rows,
               summary={k: float(v) for k, v in ok.items()},
               nuisance_error_factor=implied),
          open(os.path.join(HERE, "nuisance_achievable.json"), "w"), indent=2,
          default=float)
