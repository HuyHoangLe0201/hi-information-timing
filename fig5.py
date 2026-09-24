"""
Figure 5 -- validation, robustness, and the drift that explains a family split.

Five recent results were carried as numbers and tables only.  Three of them are
distinct enough in message to earn a panel each.

(a) the censoring efficiency, the tightest validation in the paper.  A record cut
    at age c offers eta(c) of its budget, so its achievable error rises by
    eta^-1/2; the measurement lands on that line over a threefold range of
    predicted penalty.  This is the check in which the tail factor cancels,
    being a ratio of two estimators on one record, which is why it is tighter
    than the others.
(b) the demand sweep.  Every age in the paper is read at q = 0.35, and the
    applied result would be worth little if it were a property of that choice.
    The gap and its bootstrap interval are shown across seven demands on both
    rigs.
(c) the tail factor along life, by family.  The residuals of the amount
    indicators become markedly more impulsive as a bearing degrades while the
    distributional ones barely change -- what spalling does to a level and not
    to a shape -- and it is the reason the scale correction of Section 3.2 does
    not shrink the gap.
"""
import os
import json
import numpy as np
import matplotlib.pyplot as plt

from figstyle import COL2, FS, C_POW, C_EXP, C_ACC, C_LIN, tidy

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(os.path.dirname(HERE), "paper", "fig5.pdf")
L = lambda f: json.load(open(os.path.join(HERE, f)))

CV = L("censoring_validate.json")
QS = L("qstar_sweep.json")
TD = L("tail_drift.json")

fig, axes = plt.subplots(1, 3, figsize=(COL2, 2.45))

# ---- (a) censoring: predicted against achieved -----------------------------
ax = axes[0]
pr = np.array([r["predicted"] for r in CV["rows"]])
me = np.array([r["measured"] for r in CV["rows"]])
cs = np.array([r["c"] for r in CV["rows"]])
lo, hi = 2.0, 9.2
ax.plot([lo, hi], [lo, hi], "-", lw=0.9, color=C_LIN, zorder=1)
ax.plot(pr, me, "o", ms=5.0, color=C_ACC, mew=0, zorder=3)
# the two steepest levels sit almost on top of each other, so their labels are
# placed on opposite sides rather than overlapping
side = {0.30: (6, 2), 0.50: (-6, -11), 0.70: (6, -8), 0.85: (6, -8),
        0.95: (6, -8)}
for p, m, c in zip(pr, me, cs):
    dx, dy = side.get(round(c, 2), (6, -8))
    ax.annotate(rf"${c:.2f}$", xy=(p, m), xytext=(dx, dy),
                textcoords="offset points", fontsize=FS["annot"],
                color="#555555",
                ha="right" if dx < 0 else "left")
ax.set_xlim(lo, hi)
ax.set_ylim(lo, hi)
ax.set_xticks([2, 4, 6, 8])
ax.set_yticks([2, 4, 6, 8])
ax.text(2.3, 8.5, "labels: censoring age $c$", fontsize=FS["annot"],
        color="#555555")
tidy(ax, r"predicted penalty $\eta^{-1/2}$", r"measured penalty",
     r"(a) the cost of truncation")

# ---- (b) the demand sweep --------------------------------------------------
ax = axes[1]
for rig, col, mk in (("PRONOSTIA", C_EXP, "o"), ("XJTU", C_POW, "s")):
    r = [x for x in QS["rows"] if x["rig"] == rig]
    if not r:
        continue
    q = np.array([x["q"] for x in r])
    g = np.array([x["gap"] for x in r])
    lo_ = np.array([x["lo"] for x in r])
    hi_ = np.array([x["hi"] for x in r])
    ax.fill_between(q, lo_, hi_, color=col, alpha=0.13, lw=0)
    ax.plot(q, g, mk + "-", ms=3.4, lw=1.2, color=col, mew=0, label=rig)
ax.axhline(0.0, lw=0.8, color="#666666", zorder=1)
ax.axvline(0.35, lw=0.7, ls=":", color="#666666", zorder=1)
ax.text(0.365, 0.06, r"working $q$", fontsize=FS["annot"],
        color="#666666")
ax.set_xlim(0.0, 0.85)
ax.set_ylim(-0.06, 1.0)
ax.legend(fontsize=FS["legend"], loc="upper right", borderaxespad=0.3,
          handletextpad=0.35)
tidy(ax, r"demand $q$", r"amount $-$ distribution, in lifetimes",
     r"(b) the result across demands")

# ---- (c) the tail factor along life ----------------------------------------
ax = axes[2]
sty = {("PRONOSTIA", "amount"): (C_POW, "-", "o"),
       ("XJTU", "amount"): (C_POW, "--", "s"),
       ("PRONOSTIA", "distribution"): (C_ACC, "-", "o"),
       ("XJTU", "distribution"): (C_ACC, "--", "s")}
for r in TD["by_family"]:
    key = (r["rig"], r["kind"])
    if key not in sty:
        continue
    col, ls, mk = sty[key]
    ax.plot([0, 1], [r["early"], r["late"]], ls, marker=mk, ms=3.6, lw=1.2,
            color=col, mew=0)
ax.axhline(1.0, lw=0.7, ls=":", color="#666666", zorder=1)
ax.text(1.58, 1.01, "Gaussian", fontsize=FS["annot"],
        color="#666666", ha="right", va="bottom")
ax.set_xlim(-0.18, 1.62)
ax.set_xticks([0, 1])
ax.set_xticklabels(["first sixth\nof life", "last sixth"],
                   fontsize=FS["tick"])
ax.set_ylim(0.95, 2.55)
for lab, col, y in (("amount", C_POW, 2.30), ("distribution", C_ACC, 1.30)):
    ax.text(1.06, y, lab, fontsize=FS["annot"], color=col, va="center")
ax.plot([], [], "-", color="#888888", lw=1.0, label="PRONOSTIA")
ax.plot([], [], "--", color="#888888", lw=1.0, label="XJTU")
ax.legend(fontsize=FS["legend"], loc="upper left", borderaxespad=0.3,
          handletextpad=0.4)
tidy(ax, None, r"tail factor $\kappa$", r"(c) tails heavier only for amount")

fig.tight_layout(pad=0.35, w_pad=1.2)
fig.savefig(OUT)
print(f"wrote {OUT}")
print(f"(a) {len(pr)} censoring levels, predicted {pr.min():.2f}-{pr.max():.2f}")
print(f"(b) {len(QS['rows'])} rig-demand points, all intervals above zero: "
      f"{all(x['positive'] for x in QS['rows'])}")
for r in TD["by_family"]:
    print(f"(c) {r['rig']:<10} {r['kind']:<13} {r['early']:.2f} -> "
          f"{r['late']:.2f}")
