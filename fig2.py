"""
Figure 2 -- the bound is attained, except where the estimator runs out of data.

(a) realised against predicted dispersion, one point per evaluation cell. The
    prediction is Proposition 1; the diagonal is what exact attainment looks
    like. Points are thinned for file size, but every statistic quoted is
    computed on the full set.
(b) the distribution of the ratio between them. Three domains sit just above 1,
    which is the bound being respected but not quite reached; the cells sit
    far above it.
(c) why the cells miss. Turbofan records decimated to progressively fewer
    samples, keeping their physical span. Read at the cells' own record length
    the decimation curve lands on the cells' slope, so that slope is a property
    of record length and not of cells. The median ratio is a separate matter,
    handled by the tail control in the text.
"""
import os
import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import NullFormatter
from figstyle import COL2, FS, tidy

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(os.path.dirname(HERE), "paper", "fig2.pdf")

Z = np.load(os.path.join(HERE, "crb_cells.npz"))
C = json.load(open(os.path.join(HERE, "crb_full.json")))
TAILS = json.load(open(os.path.join(HERE, "battery_tails.json")))

DOM = [("bearing", "bearings", "#D55E00", "o"),
       ("turbofan_FD001", "turbofan FD001", "#0072B2", "s"),
       ("turbofan_FD004", "turbofan FD004", "#56B4E9", "^"),
       ("battery", "Li-ion cells", "#AA3377", "D")]
RNG = np.random.default_rng(11)

fig, axes = plt.subplots(1, 3, figsize=(COL2, 2.42))

# ---- (a) realised against predicted ---------------------------------------
ax = axes[0]
allv = np.concatenate([Z[f"{k}__lp"] for k, *_ in DOM] +
                      [Z[f"{k}__lr"] for k, *_ in DOM])
lo, hi = np.percentile(allv, [0.2, 99.8])
pad = 0.06 * (hi - lo)
ax.plot([lo - pad, hi + pad], [lo - pad, hi + pad], color="#666666", lw=0.8,
        ls="--", zorder=5)
for key, lab, col, mk in DOM:
    lp, lr = Z[f"{key}__lp"], Z[f"{key}__lr"]
    if len(lp) > 1500:
        idx = RNG.choice(len(lp), 1500, replace=False)
        lp, lr = lp[idx], lr[idx]
    ax.plot(lp, lr, mk, ms=1.1, alpha=0.28, color=col, mew=0, zorder=3,
            rasterized=True)
for key, lab, col, mk in DOM:            # legend proxies at readable size
    ax.plot([], [], mk, ms=2.6, color=col, mew=0, label=lab)
ax.set_xlim(lo - pad, hi + pad); ax.set_ylim(lo - pad, hi + pad)
ax.set_aspect("equal")
ax.legend(fontsize=FS["legend"], loc="upper left", borderaxespad=0.2,
          labelspacing=0.22, handletextpad=0.3)
tidy(ax, r"predicted $\log \Gamma_W^{-1/2}$", r"realised $\log$ s.d.",
     r"(a) cell by cell")

# ---- (b) the ratio, as a distribution -------------------------------------
ax = axes[1]
ax.axvline(1.0, color="#666666", lw=0.8, ls="--", zorder=1)
for key, lab, col, mk in DOM:
    r = np.exp(Z[f"{key}__lr"] - Z[f"{key}__lp"])
    xs = np.sort(r)
    ax.plot(xs, np.linspace(0, 1, len(xs)), color=col, lw=1.2, zorder=3,
            label=lab)
    med = float(np.median(r))
    ax.plot([med], [0.5], "o", ms=3.0, color=col, zorder=5)
ax.set_xscale("log")
ax.set_xlim(0.5, 8)
ax.set_xticks([0.5, 1, 2, 4, 8])
ax.set_xticklabels(["0.5", "1", "2", "4", "8"])
ax.set_yticks([0, 0.5, 1])
ax.text(1.0, 1.06, r"attained", fontsize=FS["annot"],
        color="#666666", va="bottom", ha="center")
tidy(ax, r"realised / predicted", r"cumulative fraction",
     r"(b) how close, and how often")

# ---- (c) the cells' slope is a record-length effect ------------------------
ax = axes[2]
D = json.load(open(os.path.join(HERE, "battery_diagnosis.json")))
ns = [d["n"] for d in D["decim"]]
sl = [d["slope"] for d in D["decim"]]
ax.axhline(1.0, color="#666666", lw=0.8, ls="--", zorder=1)
ax.plot(ns, sl, "o-", ms=3.0, lw=1.2, color="#0072B2", zorder=3,
        label="turbofan, decimated")
ax.plot([D["n_cell"]], [D["obs_slope"]], "D", ms=4.2, color="#AA3377",
        zorder=5, label="Li-ion cells")
ax.plot([D["n_cell"]], [D["pred_slope"]], "x", ms=4.5, mew=1.0,
        color="#333333", zorder=6)
ax.annotate(r"length alone", xy=(D["n_cell"], D["pred_slope"]),
            xytext=(6, -13), textcoords="offset points",
            fontsize=FS["annot"], color="#333333",
            arrowprops=dict(arrowstyle="-", lw=0.5, color="#333333"))
ax.text(0.97, 0.94, r"bound attained", fontsize=FS["annot"], color="#666666",
        transform=ax.transAxes, ha="right")
ax.set_xscale("log")
ax.set_xlim(42, 240)
ax.set_xticks([50, 100, 200]); ax.set_xticklabels(["50", "100", "200"])
ax.xaxis.set_minor_formatter(NullFormatter())   # log axis adds its own
ax.set_ylim(-0.05, 1.12)
ax.set_yticks([0, 0.5, 1])
ax.legend(fontsize=FS["legend"], loc="lower right", borderaxespad=0.2,
          labelspacing=0.25)
tidy(ax, r"samples per record", r"regression slope",
     r"(c) the failure is record length")

fig.tight_layout(pad=0.34, w_pad=1.1)
fig.savefig(OUT, dpi=600)
print("wrote", OUT)
for key, lab, *_ in DOM:
    r = np.exp(Z[f"{key}__lr"] - Z[f"{key}__lp"])
    print(f"  {lab:<16} cells={len(r):>6}  median={np.median(r):.3f}  "
          f"frac above 1 = {np.mean(r > 1):.2f}")
print(f"  (c) at n={D['n_cell']}: length predicts slope {D['pred_slope']:.3f}, "
      f"cells show {D['obs_slope']:.3f}")
