"""
The values move with load. Does the ordering?

tau_min varies by up to 0.35 of life across XJTU's three operating conditions,
and a decimation control showed that is the load and not record length. So the
absolute numbers are condition-specific.

The ordering is a separate question and the more useful one: a practitioner
choosing between indicators needs to know which becomes usable first, not the
exact age. This measures whether that ranking survives across the three loads
and across the two rigs.
"""
import os
import json
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
d = json.load(open(os.path.join(HERE, "xjtu_conditions.json")))
rows = d["by_condition"]
conds = sorted({r["cond"] for r in rows})
chans = sorted({r["channel"] for r in rows})

tab = {}
for c in conds:
    tab[c] = {r["channel"]: r["tau"] for r in rows if r["cond"] == c}

print("ranking within each condition, earliest usable first\n")
for c in conds:
    order = sorted(tab[c], key=lambda k: tab[c][k])
    print(f"  {c:<12} " + " < ".join(f"{k} ({tab[c][k]:.2f})" for k in order))
print()

def rank_corr(a, b):
    ra, rb = np.argsort(np.argsort(a)), np.argsort(np.argsort(b))
    return float(np.corrcoef(ra, rb)[0, 1])

print("rank agreement between conditions\n")
pairs = []
for i in range(len(conds)):
    for j in range(i + 1, len(conds)):
        ci, cj = conds[i], conds[j]
        common = [k for k in chans if k in tab[ci] and k in tab[cj]]
        if len(common) < 3:
            continue
        rc = rank_corr([tab[ci][k] for k in common],
                       [tab[cj][k] for k in common])
        pairs.append(rc)
        print(f"  {ci:<12} vs {cj:<12} {rc:+.2f}")
print(f"\n  mean {np.mean(pairs):+.2f}\n")

print("position of each channel across the three conditions\n")
print(f"{'channel':<12}" + "".join(f"{c:>14}" for c in conds) + "   verdict")
print("-" * 68)
for k in chans:
    pos = []
    for c in conds:
        order = sorted(tab[c], key=lambda x: tab[c][x])
        pos.append(order.index(k) + 1 if k in tab[c] else None)
    stable = len(set(pos)) == 1
    near = max(pos) - min(pos) <= 1
    v = "fixed" if stable else ("moves by one" if near else "unstable")
    print(f"{k:<12}" + "".join(f"{p:>14}" for p in pos) + f"   {v}")
print("-" * 68)
last = [sorted(tab[c], key=lambda x: tab[c][x])[-1] for c in conds]
print(f"\nlast to become usable, by condition: {', '.join(last)}")
if len(set(last)) == 1:
    print(f"-> {last[0]} is last under every load tested.")
first = [sorted(tab[c], key=lambda x: tab[c][x])[0] for c in conds]
print(f"first to become usable: {', '.join(first)}")
json.dump(dict(table=tab, pair_rank=pairs), 
          open(os.path.join(HERE, "order_stability.json"), "w"), indent=2)
