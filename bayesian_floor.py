"""
A floor that covers biased estimators, and what a fleet prior is worth.

Proposition 1 bounds unbiased estimators.  Every deployed remaining-life model is
regularised, hence biased, so a reviewer is entitled to ask what the bound has to
do with practice, and the paper has so far answered only that "the bound sets the
scale of what regularisation must supply".  That is not a statement one can check.

The van Trees inequality supplies the missing one.  If the clock offset is drawn
from a prior pi rather than fixed, then EVERY estimator -- biased, regularised,
shrunk, or trained -- satisfies

    E[ (delta_hat - delta)^2 ]  >=  1 / ( E_pi[G] + J_pi ),
    J_pi = integral pi'(delta)^2 / pi(delta) d delta,

with J_pi = 1/s^2 for a Gaussian prior of standard deviation s.  In the
linearised model G does not depend on delta, so the bound is simply

    MSE  >=  1 / ( G(tau_0) + s^{-2} ) .

Three consequences, all checkable.

The prior SUBTRACTS a constant from the demand.  A target epsilon is met once
G >= epsilon^{-2} - s^{-2}, so the earliest usable age is
G^{-1}(epsilon^{-2} - s^{-2}) and every design quantity of Proposition 1 keeps
its form with the demand reduced.

A demand looser than the prior needs no measurement at all.  If epsilon >= s the
right-hand side is non-positive and the target is met at age zero.  The
unbiased bound cannot say this: it requires G >= epsilon^{-2} however large the
fleet's own spread already is.

And the unbiased floor is beatable, by exactly the amount the prior supplies.
That is not a defect of Proposition 1 but its scope, and it is measured here
rather than asserted: a shrinkage estimator is run against both floors over a
sweep of information, and must sit above the Bayesian one everywhere while
dipping below the unbiased one where the prior dominates.

The prior itself is measured from the fleets rather than chosen.  Knowing the
elapsed operating time but not the unit's life, the normalised age tau = t/L
inherits the fleet's spread in L, so at age tau the prior standard deviation is
about tau times the coefficient of variation of life.
"""
import os
import json
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
SEED = 20260828
QSTAR = 0.35

# --- 1. the fleet prior, measured -------------------------------------------
print("fleet spread of life, and the prior it implies on the clock\n")


def lives_from(npz, skip=("featnames", "centres")):
    p = os.path.join(HERE, npz)
    if not os.path.exists(p):
        return None
    z = np.load(p, allow_pickle=True)
    out = []
    for k in z.files:
        if k in skip:
            continue
        a = np.asarray(z[k])
        out.append(int(a.shape[0]))
    return np.array(out, float) if out else None


# One entry per DISTINCT fleet.  fineband.npz is the same PRONOSTIA bearings as
# feat_raw.npz seen through a different feature set, so including both would
# weight that fleet twice in any pooled figure.
FLEETS = {}
for name, fn in (("PRONOSTIA", "feat_raw.npz"),
                 ("XJTU-SY", "fineband_xjtu.npz")):
    L = lives_from(fn)
    if L is not None and len(L) >= 5:
        FLEETS[name] = L

# the turbofan fleets, if the loader's cache is present
for name, fn in (("C-MAPSS FD001", "cmapss_FD001_lives.npy"),
                 ("C-MAPSS FD004", "cmapss_FD004_lives.npy")):
    p = os.path.join(HERE, fn)
    if os.path.exists(p):
        FLEETS[name] = np.load(p).astype(float)

print(f"{'fleet':<22}{'units':>7}{'median life':>13}{'spread':>10}"
      f"{'CV':>8}{'s at tau=0.5':>15}")
print("-" * 75)
rows = []
for name, Lv in FLEETS.items():
    cv = float(np.std(Lv, ddof=1) / np.mean(Lv))
    s_half = 0.5 * cv
    rows.append(dict(fleet=name, units=len(Lv), median=float(np.median(Lv)),
                     spread=float(Lv.max() / Lv.min()), cv=cv, s_half=s_half))
    print(f"{name:<22}{len(Lv):>7}{np.median(Lv):>13.0f}"
          f"{Lv.max() / Lv.min():>9.1f}x{cv:>8.2f}{s_half:>15.3f}")
print("-" * 75)
# The tightest fleet is the working value: it makes the prior least helpful, so
# every design statement below is the conservative one.  The looser fleet only
# strengthens it.
tight = min(rows, key=lambda r: r["cv"])
CV, S = tight["cv"], tight["s_half"]
print(f"The tightest fleet is {tight['fleet']}, coefficient of variation "
      f"{CV:.2f},")
print(f"so the prior standard deviation at mid-life is s = {S:.3f} of a")
print(f"lifetime.  It is used throughout because it makes the prior least")
print(f"helpful; the looser fleet ({max(rows, key=lambda r: r['cv'])['fleet']}, "
      f"s = {max(r['s_half'] for r in rows):.3f}) only strengthens what follows.\n")

# --- 2. what the prior is worth in design terms -----------------------------
print("the demand the record must still supply, after the prior\n")
print("A target epsilon needs G >= epsilon^-2 - s^-2 rather than "
      "G >= epsilon^-2.\n")
print(f"{'target eps':>12}{'unbiased demand':>18}{'after the prior':>18}"
      f"{'share removed':>16}")
print("-" * 66)
dem = []
for eps in (0.30, 0.20, 0.10, 0.05, 0.02, 0.01):
    a = eps ** -2
    b = a - S ** -2
    dem.append(dict(eps=eps, unbiased=a, bayes=max(b, 0.0),
                    removed=float(min(1.0, S ** -2 / a))))
    tag = "none needed" if b <= 0 else f"{b:.0f}"
    print(f"{eps:>12.2f}{a:>18.0f}{tag:>18}"
          f"{min(1.0, S ** -2 / a) * 100:>15.0f}%")
print("-" * 66)
cross = S
print(f"The crossover is at epsilon = s = {cross:.3f}: a demand looser than the")
print("fleet's own spread is met at age zero, which the unbiased bound cannot")
print("express since it asks for G >= epsilon^-2 regardless.\n")

# --- 3. is the Bayesian floor respected, and the unbiased one beaten? -------
print("a shrinkage estimator against both floors\n")
print("The estimator is delta_hat_MAP = G/(G + s^-2) * delta_hat_ML, the")
print("posterior mean under the Gaussian prior.  It must sit above the")
print("Bayesian floor everywhere; it may sit below the unbiased one.\n")
rng = np.random.default_rng(SEED)
REPS = 400000
# the Monte Carlo standard error of an RMSE estimated from n draws is about
# 1/sqrt(2n) in relative terms; quoted so a ratio near one is read correctly
MCSE = 1.0 / np.sqrt(2.0 * REPS)
print(f"{'G':>11}{'unbiased floor':>17}{'Bayes floor':>14}{'ML':>10}"
      f"{'MAP':>10}{'MAP/Bayes':>12}{'MAP/unbiased':>15}")
print("-" * 89)
sim = []
worst_viol = np.inf
for G in (1e0, 1e1, 4e1, 1e2, 4e2, 1e3, 1e4):
    d = rng.normal(0.0, S, size=REPS)                  # drawn from the prior
    ml = d + rng.normal(0.0, 1.0 / np.sqrt(G), size=REPS)
    shrink = G / (G + S ** -2)
    mp = shrink * ml
    e_ml = float(np.sqrt(np.mean((ml - d) ** 2)))
    e_mp = float(np.sqrt(np.mean((mp - d) ** 2)))
    f_un = 1.0 / np.sqrt(G)
    f_by = 1.0 / np.sqrt(G + S ** -2)
    worst_viol = min(worst_viol, e_mp / f_by)
    sim.append(dict(G=G, unbiased=f_un, bayes=f_by, ml=e_ml, map=e_mp,
                    r_bayes=e_mp / f_by, r_unbiased=e_mp / f_un))
    print(f"{G:>11.0e}{f_un:>17.4f}{f_by:>14.4f}{e_ml:>10.4f}{e_mp:>10.4f}"
          f"{e_mp / f_by:>12.3f}{e_mp / f_un:>15.3f}")
print("-" * 89)
rb = np.array([s["r_bayes"] for s in sim])
print(f"MAP against the Bayesian floor: median {np.median(rb):.4f}, "
      f"range {rb.min():.4f} to {rb.max():.4f}")
print(f"Monte Carlo standard error on each ratio {MCSE:.4f}, so the departures")
print(f"are {abs(np.median(rb) - 1) / MCSE:.1f} standard errors and the floor is not merely")
print("respected but ATTAINED: for a Gaussian prior and Gaussian noise the")
print("posterior mean achieves the van Trees bound exactly, so unlike the")
print("unbiased floor of Section 5 its tightness needs no measurement.")
below = [s for s in sim if s["r_unbiased"] < 1.0 - 3 * MCSE]
if below:
    print()
    print(f"the unbiased floor is beaten at {len(below)} of {len(sim)} information")
    print(f"levels, by up to {1 / min(s['r_unbiased'] for s in below):.1f} times, and only where the prior")
    print("carries a comparable share of the total information.")
print()
print("So Proposition 1 is not the floor a deployed model faces; it is the floor")
print("a model faces that declines to use what the fleet already implies.  The")
print("Bayesian floor is the one that covers every estimator, and it differs from")
print(f"the unbiased floor by the additive constant s^-2 = {S ** -2:.1f}, which is")
print("negligible against the information these records carry late in life and")
print("dominant early.")

# --- 4. what the prior does to censoring ------------------------------------
# Censoring efficiency is the one design quantity the prior changes in a
# direction the retention section cares about: it becomes
# (G(c) + s^-2) / (G(1) + s^-2), which is nearer one, so a truncated record
# costs less than the unbiased form says.
print("censoring efficiency, with and without the prior\n")
print(f"{'record carries G(1)':>21}{'G(c)/G(1)':>12}{'with prior':>13}"
      f"{'recovered':>12}")
print("-" * 58)
cens = []
for G1 in (1e1, 1e2, 1e3, 1e4):
    for frac in (0.5,):
        Gc = frac * G1
        a = Gc / G1
        b = (Gc + S ** -2) / (G1 + S ** -2)
        cens.append(dict(G1=G1, raw=a, with_prior=b, gain=b - a))
        print(f"{G1:>21.0e}{a:>12.2f}{b:>13.2f}{b - a:>+12.2f}")
print("-" * 58)
print("A record censored at half its information keeps half of it in the")
print("unbiased account and more once the fleet prior is included, the")
print("difference vanishing as the record's own information grows.  This is the")
print("direction that matters for Section 10: truncation is less costly than the")
print("unbiased form implies, which is consistent with the retention null found")
print("there rather than in tension with it.")

json.dump(dict(fleets=rows, cv_working=CV, prior_sd=S,
               tightest_fleet=tight["fleet"],
               demand=dem, crossover=float(cross),
               simulation=sim, mcse=float(MCSE),
               map_over_bayes_median=float(np.median(rb)),
               map_over_bayes_min=float(rb.min()),
               map_over_bayes_max=float(rb.max()),
               levels_beating_unbiased=len(below),
               censoring=cens),
          open(os.path.join(HERE, "bayesian_floor.json"), "w"), indent=2,
          default=float)
