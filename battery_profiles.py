"""Does 'the exponent belongs to the indicator' replicate in the battery domain?"""
import os, json
import numpy as np
from pipeline import prepare, tau_min, beta_eff

HERE = os.path.dirname(os.path.abspath(__file__))
z = np.load(os.path.join(HERE, "battery_raw.npz"))
CELLS = sorted({k.split("__")[0] for k in z.files})
TAGS = ["cap", "Re", "Rct"]
QSTAR = 0.35

print(f"{'cell':<8}{'indicator':<7}{'n':>5}{'sigma':>9}{'beta_eff':>10}"
      f"{'tau_min':>9}{'info<0.5':>10}{'info<0.8':>10}")
print("-" * 68)
rows, dropped = [], []
for cell in CELLS:
    for tag in TAGS:
        u, why = prepare(z[f"{cell}__{tag}"], f"{cell}/{tag}")
        if u is None:
            dropped.append((f"{cell}/{tag}", why))
            print(f"{cell:<8}{tag:<7}{'':>5}  dropped: {why}")
            continue
        be, tm = beta_eff(u), tau_min(u, QSTAR)
        i50 = float(u["F"][int(0.5 * u["n"]) - 1])
        i80 = float(u["F"][int(0.8 * u["n"]) - 1])
        rows.append(dict(cell=cell, tag=tag, n=u["n"], sigma=u["sig_oof"],
                         beta_eff=be, tau_min=tm, info50=i50, info80=i80))
        print(f"{cell:<8}{tag:<7}{u['n']:>5}{u['sig_oof']:>9.4f}{be:>10.2f}"
              f"{tm:>9.3f}{i50:>10.3f}{i80:>10.3f}")

print("-" * 68)
print(f"{'MEDIAN by indicator':<15}{'beta_eff':>10}{'tau_min':>10}{'info<0.8':>11}{'n':>5}")
summary = {}
for tag in TAGS:
    r = [x for x in rows if x["tag"] == tag]
    if not r:
        continue
    b = float(np.median([x["beta_eff"] for x in r]))
    t = float(np.median([x["tau_min"] for x in r]))
    i8 = float(np.median([x["info80"] for x in r]))
    summary[tag] = dict(beta_eff=b, tau_min=t, info80=i8, cells=len(r))
    print(f"{tag:<15}{b:>10.2f}{t:>10.3f}{i8:>11.3f}{len(r):>5}")

if summary:
    bs = [v["beta_eff"] for v in summary.values()]
    ts = [v["tau_min"] for v in summary.values()]
    print(f"\nspread across indicators on the SAME cells:")
    print(f"  beta_eff  {min(bs):.2f} .. {max(bs):.2f}   (factor {max(bs)/min(bs):.1f})")
    print(f"  tau_min   {min(ts):.3f} .. {max(ts):.3f}")

json.dump(dict(qstar=QSTAR, rows=rows, summary=summary,
               dropped=dropped), open(os.path.join(HERE, "battery_profiles.json"), "w"),
          indent=2)
print("\nwrote battery_profiles.json")
