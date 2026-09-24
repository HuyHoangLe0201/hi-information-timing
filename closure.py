"""
Does the theory predict the measurement?

Section 2 gives a closed form for the earliest age at which a precision target
can be met, for a power-law degradation path:

    tau_min = q*^(1/(2 beta - 1)).

Section 3 measures two things independently: the exponent beta_eff (de-biased
for noise) and, separately, tau_min read straight off the estimated information
curve. Neither measurement uses the other. If the closed form evaluated at the
de-biased exponent lands on the directly measured tau_min, the two halves of the
paper are describing the same object; if it does not, one of them is wrong.
"""
import os, json
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
LS = os.path.join(os.path.dirname(os.path.dirname(HERE)),
                  "IEEE_sensor_letters", "work")

beta = {r["name"]: r for r in json.load(open(os.path.join(HERE, "beta_debias.json")))}
prof = json.load(open(os.path.join(LS, "indicator_profiles.json")))
P = prof["indicators"]

# q* is fixed by the precision target, not fitted here; recover the value the
# measurement used from the one profile whose exponent is known by construction.
qstar = prof["tau_min_pow3"] ** 5.0
print(f"q* implied by the nominal power-law profile: {qstar:.3f}\n")

PAIR = [("bearing RMS", "rms"), ("bearing 0-2 kHz", "lf 0-2kHz"),
        ("bearing 4-10 kHz", "hf 4-10kHz")]

print(f"{'indicator':<20}{'beta (de-biased)':>18}{'predicted':>12}"
      f"{'measured':>11}{'error':>9}")
print("-" * 70)
rows = []
for bname, pname in PAIR:
    b = beta[bname]
    be = b["debiased"] if b["debiased"] else b["ceiling"]
    lab = f"{be:.2f}" + ("" if b["debiased"] else "+")
    pred = qstar ** (1.0 / (2 * be - 1))
    meas = P[pname]["tau_min_obs"]
    rows.append(dict(name=bname, beta=float(be), predicted=float(pred),
                     measured=float(meas), saturated=b["debiased"] is None))
    print(f"{bname:<20}{lab:>18}{pred:>12.3f}{meas:>11.3f}{pred-meas:>9.3f}")

print("-" * 70)
d = [abs(r["predicted"] - r["measured"]) for r in rows if not r["saturated"]]
sat = [r for r in rows if r["saturated"]]
print(f"unsaturated indicators: max |error| = {max(d):.3f}")
for r in sat:
    print(f"note: {r['name']} exponent is a lower bound, so its prediction is one too")
print("\nThe exponent and the earliest usable age were estimated by separate")
print("procedures from the same records; agreement is a consistency check on both.")
json.dump(dict(qstar=float(qstar), rows=rows),
          open(os.path.join(HERE, "closure.json"), "w"), indent=2)
