"""
Single source of truth for every number the paper quotes.

Each experiment wrote its own JSON. The manuscript should not read those
directly -- it should read one file, so that a number in the text can always be
traced to the run that produced it, and so an audit script can check the text
against the same file the text was written from.
"""
import os, json
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
L = lambda f: json.load(open(os.path.join(HERE, f)))
R = {}

# --- Result 1: the bound is attained on real degradation records -------------
R["crb"] = dict(L("crb_full.json"))   # full run: every unit, three channels
R["crb"]["_note"] = "Proposition 1 predicts slope 1 and median ratio 1."

# --- Result 2: the exponent that sets the minimum window ---------------------
R["beta"] = {r["name"]: {k: r[k] for k in
                         ("n", "sigma", "measured", "debiased", "ceiling")}
             for r in L("beta_debias.json")}

# --- Result 3 (demoted): retention ------------------------------------------
R["retention"] = {
    "fixed_tau0_0.7": L("retention_final.json"),
    "at_operating_point": L("retention_operating.json"),
    "validation": L("retention_validate.json"),
    "_note": ("Measured efficiency reads low when the density estimate is noisy "
              "or the record short; corrected against a case whose true value "
              "is 1. Bearing records additionally require the run-in period to "
              "be excluded. At each indicator's own operating age the trailing "
              "window is near-optimal except for early-usable indicators.")}

# --- Supporting: saturation exponent ----------------------------------------
try:
    R["saturation"] = L("saturation_rate.json")
except Exception:
    pass

# --- Supporting: pipeline validated against synthetic ground truth ----------
R["groundtruth"] = L("synth_groundtruth.json")

out = os.path.join(HERE, "RESULTS.json")
json.dump(R, open(out, "w"), indent=2, default=float)

def peek(o, d=0, k=""):
    pad = "  " * d
    if isinstance(o, dict):
        if d >= 2:
            print(f"{pad}{k}: {{{len(o)} keys}}")
            return
        if k:
            print(f"{pad}{k}:")
        for kk, vv in list(o.items())[:12]:
            peek(vv, d + 1, kk)
    elif isinstance(o, list):
        print(f"{pad}{k}: [{len(o)} rows]")
    else:
        v = f"{o:.4g}" if isinstance(o, float) else str(o)
        print(f"{pad}{k}: {v[:70]}")

print(f"wrote {out}\n")
peek(R)
