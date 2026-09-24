"""Turbofan: does the exponent depend on which sensor channel you call the HI?"""
import os, json
import numpy as np
from pipeline import profile, tau_min, beta_eff

HERE = os.path.dirname(os.path.abspath(__file__))
QSTAR = 0.35
CHANNELS = [2, 3, 4, 7, 8, 9, 11, 12, 13, 14, 15, 17, 20, 21]   # the informative set

for sub in ["FD001", "FD004"]:
    z = np.load(os.path.join(HERE, f"cmapss_{sub}.npz"))
    units = sorted({k.split("__")[0] for k in z.files})
    per_ch = {c: {"beta": [], "tmin": [], "i80": []} for c in CHANNELS}
    dropped = 0
    for u in units:
        S = z[f"{u}__sensors"]
        for c in CHANNELS:
            u_, why = profile(S[:, c - 1].astype(float), f"{u}/s{c}")
            if u_ is None:
                dropped += 1
                continue
            per_ch[c]["beta"].append(beta_eff(u_))
            per_ch[c]["tmin"].append(tau_min(u_, QSTAR))
            per_ch[c]["i80"].append(float(u_["F"][int(0.8 * u_["n"]) - 1]))

    print(f"\n=== C-MAPSS {sub} ({len(units)} units, {dropped} unit-channels dropped) ===")
    print(f"{'sensor':<9}{'units':>6}{'beta_eff':>10}{'tau_min':>9}{'info<0.8':>10}")
    print("-" * 45)
    rows = []
    for c in CHANNELS:
        d = per_ch[c]
        if len(d["beta"]) < 10:
            continue
        b, t, i8 = (float(np.median(d["beta"])), float(np.median(d["tmin"])),
                    float(np.median(d["i80"])))
        rows.append(dict(sensor=c, units=len(d["beta"]), beta_eff=b, tau_min=t, info80=i8))
        print(f"s{c:<8}{len(d['beta']):>6}{b:>10.2f}{t:>9.3f}{i8:>10.3f}")
    if rows:
        bs = [r["beta_eff"] for r in rows]
        ts = [r["tau_min"] for r in rows]
        print("-" * 45)
        print(f"spread across channels of ONE engine fleet: "
              f"beta_eff {min(bs):.2f}..{max(bs):.2f} (factor {max(bs)/min(bs):.1f}), "
              f"tau_min {min(ts):.3f}..{max(ts):.3f}")
    json.dump(dict(qstar=QSTAR, subset=sub, rows=rows),
              open(os.path.join(HERE, f"cmapss_profiles_{sub}.json"), "w"), indent=2)
