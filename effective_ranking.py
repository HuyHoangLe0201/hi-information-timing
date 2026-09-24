"""
Does correcting for what the bound ignores change which indicator to pick?

Three separate overstatements have now been measured, and each is
indicator-specific rather than a common factor:

  nuisance   -- the damage clock is confounded with the trend's own parameters.
                What the fleet leaves is 26-44% depending on the indicator.
  misfit     -- the residual about the best member of the trend family is larger
                than the residual about a local smoother, by 0.9x to 2.5x, so
                the information built on the local scale is 1x to 6x optimistic.
  correlation-- that misfit residual is not white. Its autocorrelation length is
                1 sample on turbofan channels and 19 to 105 on bearing ones, and
                a correlation length of l means l samples are worth one.

Their product varies by two orders of magnitude across indicators, so it cannot
be absorbed into a constant and ignored. The question is whether it reorders the
indicators -- because a ranking is what the theory is for. If the ordering
survives, the comparisons stand and only the absolute numbers were wrong; if it
does not, the discount has to be carried explicitly.
"""
import os
import json
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
L = lambda f: json.load(open(os.path.join(HERE, f)))

ME = {r["indicator"]: r for r in L("model_error.json")}
FP = {r["domain"]: r for r in L("fleet_prior.json")}
PC = {r["domain"]: r for r in L("prior_crossover.json")}

NAME = {"bearing RMS": "bearing RMS",
        "bearing 0--2 kHz": "bearing 0--2 kHz",
        "bearing 4--10 kHz": "bearing 4--10 kHz",
        "turbofan s4": "turbofan s4",
        "turbofan s11": "turbofan s11"}

print("the nominal information, and what survives each correction\n")
print(f"{'indicator':<20}{'nominal':>11}{'nuisance':>10}{'misfit':>9}"
      f"{'corr len':>10}{'effective':>12}{'discount':>11}")
print("-" * 83)
rows = []
for key in NAME:
    if key not in ME or key not in FP or key not in PC:
        continue
    g0 = PC[key]["gamma_record"]
    keep_n = FP[key]["retained_fleet"]
    mis = ME[key]["ratio"] ** 2
    ell = max(ME[key]["acorr_family"], 1.0)
    geff = g0 * keep_n / (mis * ell)
    rows.append(dict(indicator=key, nominal=g0, keep_nuisance=keep_n,
                     misfit_factor=mis, corr_len=ell, effective=geff,
                     discount=g0 / geff))
    print(f"{key:<20}{g0:>11.3g}{keep_n:>10.3f}{mis:>9.1f}{ell:>10.0f}"
          f"{geff:>12.3g}{g0/geff:>11.0f}")
print("-" * 83)
print("discount = nominal / effective, and it is not a common factor:")
d = [r["discount"] for r in rows]
print(f"  it ranges from {min(d):.0f}x to {max(d):.0f}x, a spread of "
      f"{max(d)/min(d):.0f}x across indicators.\n")

# ---- does the ranking survive? ---------------------------------------------
by_nom = [r["indicator"] for r in sorted(rows, key=lambda r: -r["nominal"])]
by_eff = [r["indicator"] for r in sorted(rows, key=lambda r: -r["effective"])]
print("indicators ranked most-informative-first")
print(f"  by nominal information   : {', '.join(by_nom)}")
print(f"  by effective information : {', '.join(by_eff)}")
moved = sum(1 for a, b in zip(by_nom, by_eff) if a != b)
print(f"  positions changed        : {moved} of {len(by_nom)}")

# Kendall tau between the two orderings, computed directly
def tau(a, b):
    ia = {k: i for i, k in enumerate(a)}
    ib = {k: i for i, k in enumerate(b)}
    ks = list(ia)
    c = d_ = 0
    for i in range(len(ks)):
        for j in range(i + 1, len(ks)):
            x, y = ks[i], ks[j]
            s = np.sign(ia[x] - ia[y]) * np.sign(ib[x] - ib[y])
            if s > 0:
                c += 1
            elif s < 0:
                d_ += 1
    return (c - d_) / (c + d_) if (c + d_) else np.nan


print(f"  rank correlation         : {tau(by_nom, by_eff):+.2f}")
print()
print("The discount is dominated by the correlation length of the model")
print("misfit, which is a property of how well a power law describes that")
print("particular indicator -- not of how much degradation information it")
print("carries. An indicator whose shape the family fits badly is penalised")
print("for the family's failure, not its own.")
json.dump(dict(rows=rows, rank_nominal=by_nom, rank_effective=by_eff),
          open(os.path.join(HERE, "effective_ranking.json"), "w"), indent=2)
