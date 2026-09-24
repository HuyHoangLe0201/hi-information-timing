"""
Figure 3 -- what the bound governs, and what the curve says about construction.

(a) the scope result. Information is swept over two orders of magnitude with
    everything else fixed; an estimator handed the degradation trend follows the
    predicted -1/2 scaling, a practical ridge regression does not and its error
    barely moves. The gap is the estimator's inefficiency, not missing
    information, which is why the framework is a feasibility test.
(b) the diagnostic. How far the best achievable error sits above the floor,
    against the share of the density surviving the noise subtraction. The
    fraction needs no ground truth, so it can be computed for any real record.
(c) the applied result. Ten indicators built from the same spectra on two rigs,
    split by whether they describe how energy is DISTRIBUTED or how much of it
    there is.
"""
import os
import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import NullFormatter
from figstyle import COL2, FS, C_POW, C_EXP, C_ACC, C_LIN, tidy

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(os.path.dirname(HERE), "paper", "fig3.pdf")
L = lambda f: json.load(open(os.path.join(HERE, f)))

fig, axes = plt.subplots(1, 3, figsize=(COL2, 2.35))

# ---- (a) oracle against a practical model ---------------------------------
ax = axes[0]
OM = L("oracle_vs_model.json")
G = np.array([r["G"] for r in OM])
for key, col, mk, lab in (("oracle", C_ACC, "o", "trend known"),
                          ("ridge", C_POW, "s", "ridge on windows")):
    y = np.array([r[key] for r in OM])
    ax.plot(G, y, mk, ms=3.2, color=col, mew=0, label=lab)
    s, i0 = np.polyfit(np.log(G), np.log(y), 1)
    xx = np.array([G.min(), G.max()])
    ax.plot(xx, np.exp(i0) * xx ** s, "-", lw=1.0, color=col)
    ax.annotate(rf"slope ${s:+.2f}$", xy=(G.min(), np.exp(i0) * G.min() ** s),
                xytext=(4, -9 if key == "oracle" else 5),
                textcoords="offset points", fontsize=FS["annot"], color=col)
# the predicted slope, anchored at the oracle's best point
g0, y0 = G.max(), min(r["oracle"] for r in OM)
ax.plot([G.min(), G.max()],
        [y0 * (G.min() / g0) ** -0.5, y0], ":", lw=1.0, color="#666666")
ax.text(G.min() * 1.4, y0 * (G.min() / g0) ** -0.5 * 1.5,
        r"$-1/2$", fontsize=FS["annot"], color="#666666")
ax.set_xscale("log"); ax.set_yscale("log")
ax.legend(fontsize=FS["legend"], loc="lower left", borderaxespad=0.2,
          handletextpad=0.3)
tidy(ax, r"information $G$ available", r"RUL error",
     r"(a) what the bound governs")

# ---- (b) the diagnostic ----------------------------------------------------
ax = axes[1]
T = [r for r in L("tightness2.json") if not r["edge"]]
sf = np.array([100 * r["signal_frac"] for r in T])
ex = np.array([r["excess"] for r in T])
hi, lo = sf > 90, sf <= 90
ax.plot(sf[hi], ex[hi], "o", ms=3.4, color=C_ACC, mew=0, label=r"$>90\%$")
ax.plot(sf[lo], ex[lo], "s", ms=3.4, color=C_POW, mew=0, label=r"$\le 90\%$")
ax.axvline(90, color="#999999", lw=0.8, ls="--")
ax.axhline(1.0, color="#666666", lw=0.7, ls=":")
for m, c, x in ((np.median(ex[hi]), C_ACC, 96), (np.median(ex[lo]), C_POW, 76)):
    ax.plot([x - 4, x + 4], [m, m], "-", lw=1.6, color=c)
    ax.text(x, m * 1.18, f"{m:.2f}", fontsize=FS["annot"], color=c,
            ha="center")
ax.set_yscale("log")
ax.set_ylim(0.9, 8)
ax.set_yticks([1, 2, 4, 8]); ax.set_yticklabels(["1", "2", "4", "8"])
ax.yaxis.set_minor_formatter(NullFormatter())   # log axis adds its own
ax.text(89, 7.0, "floor tight", fontsize=FS["annot"], color="#666666",
        ha="right")
ax.legend(fontsize=FS["legend"], loc="upper right", borderaxespad=0.2,
          handletextpad=0.3, title=None)
tidy(ax, r"surviving fraction (\%)", r"achievable / floor",
     r"(b) when the floor is tight")

# ---- (c) distribution against amount --------------------------------------
ax = axes[2]
DB = L("distribution_both.json")
rigs = ["PRONOSTIA", "XJTU"]
xs = {"PRONOSTIA": 0, "XJTU": 1}
for rig in rigs:
    for r in DB["rigs"][rig]:
        if r["signal"] < 0.90:
            continue
        col = C_EXP if r["kind"] == "distribution" else C_POW
        mk = "o" if r["kind"] == "distribution" else "s"
        ax.plot(xs[rig] + (0.10 if r["kind"] == "amount" else -0.10),
                r["tau"], mk, ms=3.6, color=col, mew=0, alpha=0.85)
S = DB["summary"]
for i, rig in enumerate(rigs):
    for kind, off, col in (("distribution", -0.10, C_EXP),
                           ("amount", 0.10, C_POW)):
        m = S[kind]["median"][i]
        ax.plot([i + off - 0.075, i + off + 0.075], [m, m], "-", lw=1.8,
                color=col)
    ax.annotate("", xy=(i, S["amount"]["median"][i]),
                xytext=(i, S["distribution"]["median"][i]),
                arrowprops=dict(arrowstyle="<->", lw=0.7, color="#444444"))
    gap = S["amount"]["median"][i] - S["distribution"]["median"][i]
    ax.text(i + 0.015, (S["amount"]["median"][i]
                        + S["distribution"]["median"][i]) / 2,
            f"{gap:.2f}", fontsize=FS["annot"], color="#444444", va="center")
ax.plot([], [], "o", ms=3.6, color=C_EXP, mew=0, label="distribution")
ax.plot([], [], "s", ms=3.6, color=C_POW, mew=0, label="amount")
ax.set_xlim(-0.4, 1.4)
ax.set_xticks([0, 1]); ax.set_xticklabels(rigs, fontsize=FS["tick"])
ax.set_ylim(0, 1.05); ax.set_yticks([0, 0.5, 1])
ax.legend(fontsize=FS["legend"], loc="lower right", borderaxespad=0.2,
          handletextpad=0.3)
tidy(ax, None, r"relative information age",
     r"(c) distribution before amount")

fig.tight_layout(pad=0.34, w_pad=1.1)
fig.savefig(OUT)
print("wrote", OUT)
so = np.polyfit(np.log(G), np.log([r["oracle"] for r in OM]), 1)[0]
sr = np.polyfit(np.log(G), np.log([r["ridge"] for r in OM]), 1)[0]
print(f"  (a) oracle {so:+.3f}, ridge {sr:+.3f}")
print(f"  (b) medians {np.median(ex[hi]):.2f} above 90%, "
      f"{np.median(ex[lo]):.2f} below")
print(f"  (c) gaps " + ", ".join(
    f"{r} {S['amount']['median'][i] - S['distribution']['median'][i]:.3f}"
    for i, r in enumerate(rigs)))
