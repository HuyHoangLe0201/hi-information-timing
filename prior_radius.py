r"""Where may the Bayesian floor be used?  It is not everywhere it is quoted.

Two of the paper's own results are in tension and nothing connects them.

Proposition 4 bounds every estimator by 1/(G + s^-2), and its proof invokes the
linearised model of Section 3, "J(delta) = G(tau_0) independent of delta in the
linearised model".  The prior it is applied with has standard deviation
s = 0.294 of a lifetime.

Section 7.3 measures how far that linearisation reaches: a median radius of
0.0043 of a lifetime for an estimator handed the trend, 0.072 for one that must
fit it, and it closes with the sentence "an age prior must be re-anchored within
a few per cent of a lifetime for the bound to describe the estimator built on
it".

A prior of 0.294 is not a few per cent.  It is four times the practical radius
and sixty-eight times the oracle one.  The paper states the requirement in one
section and violates it in another, a thousand lines apart.

This asks the quantitative question the tension raises: at what point does the
posterior become narrow enough that the linearisation the bound assumes is
actually valid?  That is a threshold on G, and it can be compared with the G
the records carry and with the G at each demand the paper quotes.
"""
import os
import json
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = {}

LR = json.load(open(os.path.join(HERE, "linearisation_radius.json")))
BF = json.load(open(os.path.join(HERE, "bayesian_floor.json")))

r_oracle = float(LR["radius_median"])
r_prac = float(LR["radius_practical"])
s = float(BF["prior_sd"])

print("=" * 88)
print("The prior's width against the radius within which the bound is valid")
print("=" * 88)
print("  %-46s %10s" % ("prior standard deviation used by Proposition 4", "%.4f" % s))
print("  %-46s %10s" % ("linearisation radius, estimator given the trend",
                        "%.4f" % r_oracle))
print("  %-46s %10s" % ("linearisation radius, estimator fitting the trend",
                        "%.4f" % r_prac))
print("  " + "-" * 58)
print("  %-46s %10.1f" % ("prior width in oracle radii", s / r_oracle))
print("  %-46s %10.1f" % ("prior width in practical radii", s / r_prac))
OUT["prior_sd"] = s
OUT["radius_oracle"] = r_oracle
OUT["radius_practical"] = r_prac
OUT["prior_in_oracle_radii"] = s / r_oracle
OUT["prior_in_practical_radii"] = s / r_prac

print("""
  The bound is stated with a prior that is wider than the region in which the
  information entering it was shown to be a constant.  That does not make the
  inequality false; it makes it unproven, because its proof assumes the
  linearisation over the range the prior spans.
""")

# --- when does the posterior come inside the radius? --------------------------
print("=" * 88)
print("The information at which the bound becomes self-consistent")
print("=" * 88)
print("""
The posterior standard deviation under the bound is (G + s^-2)^(-1/2).  It comes
inside a radius r when G exceeds 1/r^2 - s^-2.  Below that the bound is being
read at offsets its own linearisation does not cover.
""")
rows = []
for lab, r in (("given the trend", r_oracle), ("fitting the trend", r_prac)):
    g_need = 1.0 / r ** 2 - 1.0 / s ** 2
    rows.append(dict(regime=lab, radius=r, G_needed=g_need))
    print("  %-22s radius %.4f   needs G > %12.0f" % (lab, r, g_need))
OUT["threshold"] = rows

# --- how does that compare with what the paper quotes? ------------------------
print()
print("  the demands Proposition 4 is quoted at, and whether each is covered")
print("  %10s %14s %16s %s" % ("demand eps", "G required", "posterior sd", "verdict"))
print("  " + "-" * 62)
dem = []
for eps in (0.30, 0.20, 0.10, 0.05, 0.01):
    g = max(eps ** -2 - 1.0 / s ** 2, 0.0)
    post = (g + 1.0 / s ** 2) ** -0.5
    ok_o = post <= r_oracle
    ok_p = post <= r_prac
    dem.append(dict(eps=eps, G=g, post=post, inside_oracle=bool(ok_o),
                    inside_practical=bool(ok_p)))
    print("  %10.2f %14.1f %16.4f %s" % (
        eps, g, post,
        "inside both radii" if ok_o else
        ("inside the practical radius only" if ok_p else
         "outside both")))
OUT["demands"] = dem

_first_ok = next((d for d in dem if d["inside_practical"]), None)
_first_oo = next((d for d in dem if d["inside_oracle"]), None)
print("""
  The demand at which the bound first becomes self-consistent is %s for an
  estimator that fits the trend and %s for one handed it.  Every looser demand,
  including the %.2f the paper singles out as "met before the machine is
  switched on", sits outside the linearisation, and the claim there rests on the
  trivial fact that an estimator with no data cannot beat its prior rather than
  on Proposition 4.
""" % ("eps = %.2f" % _first_ok["eps"] if _first_ok else "no demand tested",
       "eps = %.2f" % _first_oo["eps"] if _first_oo else "no demand tested",
       dem[0]["eps"]))

# --- does any of this touch what the paper measures? --------------------------
print("=" * 88)
print("The information the records actually carry")
print("=" * 88)
from nonparam import weighted_density, estimate_floor
rng = np.random.default_rng(3)
z = np.load(os.path.join(HERE, "feat_raw.npz"), allow_pickle=True)
FN = [str(t) for t in z["featnames"]]
i = FN.index("rms")
Gb = []
for u in [k for k in z.files if k != "featnames"]:
    x = np.asarray(z[u][:, i], float)
    dens, res = weighted_density(x)
    if dens is None:
        continue
    dd = np.clip(dens - estimate_floor(dens, res, "surrogate", rng, reps=6),
                 0.0, None)
    Gb.append(float(dd.sum()))
g_bear = float(np.median(Gb))
g_turbo = float(json.load(open(os.path.join(HERE, "endtoend_real.json")))
                ["min_n_300"]["rows"][0]["G"])
OUT["G_bearings_median"] = g_bear
OUT["G_turbofan"] = g_turbo
worst = max(r["G_needed"] for r in rows)
OUT["margin_turbofan"] = g_turbo / worst
OUT["margin_bearings"] = g_bear / worst
print("  %-44s %14.3g" % ("bearings, RMS, whole record, median G", g_bear))
print("  %-44s %14.3g" % ("turbofan at 0.60 of life", g_turbo))
print("  %-44s %14.0f" % ("threshold for the oracle radius", worst))
print("  " + "-" * 60)
print("  %-44s %14.0f" % ("margin, turbofan", g_turbo / worst))
print("  %-44s %14.0f" % ("margin, bearings", g_bear / worst))
print("""
  Both are above the threshold, so nothing measured here is read outside the
  linearisation, but the margins are not equal and the smaller one is worth
  stating rather than rounding into a comfortable phrase.  The bearings clear it
  by %.0f.  The turbofan clears it by %.1f, which is a margin and not a gulf: on
  a fleet whose records were shorter, or read earlier than 0.60 of life, the
  oracle form of the bound would cease to be self-consistent before the record
  ran out.  The tension restricts where the proposition may be quoted; it does
  not reach what has been measured with it here.
""" % (g_bear / worst, g_turbo / worst))

json.dump(OUT, open(os.path.join(HERE, "prior_radius.json"), "w"), indent=2,
          default=float)
print("written to prior_radius.json")
