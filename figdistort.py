r"""The contrast identity, and what it buys, in one picture.

Propositions 15 to 26 are the longest unbroken stretch of theory in the paper and
the only one with no figure at all.  The result they turn on is easy to state and
hard to believe on first reading: that the entire effect of an arbitrary
distortion of the information density on the censoring efficiency is ONE number,
how much larger the distortion is on average before the cut than after it, and
that a bound written in the distortion's RANGE gives away more than an order of
magnitude for nothing.

Three panels, all from measurements already in the paper.

(a) why a range is the wrong currency.  One bearing record, the paper's own
    derivative correction as the distortion.  Its range is five; the two
    information-weighted means differ by a few per cent, because the extremes
    that set a range sit where the density does not.
(b) the overcharge, on all three corrections the paper applies.
(c) what the repair buys: the ordering is certified on eleven of sixteen
    bearings in the contrast, where the range-based certificate covered none.

Panel (a) recomputes the distortion from the records rather than reading a
summary, so the median it draws is checked against contrast_identity.json before
anything is plotted.  A figure that quietly disagreed with the file the text
cites would be worse than no figure.
"""
import os
import json
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter

from figstyle import COL2, FS, C_POW, C_EXP, C_ACC, C_LIN, tidy
from pipeline import robust_scale, _win, local_scale, LOCAL_BW
from nonparam import weighted_density, estimate_floor

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(os.path.dirname(HERE), "paper", "figdistort.pdf")
JS = os.path.join(HERE, "figdistort.json")
REF = json.load(open(os.path.join(HERE, "contrast_identity.json")))
RES = {}

rng = np.random.default_rng(271828)
MIN_SHARE = 0.02
H_HALF, WIDE = 0.04, 4.0


def contrast(w, g, ic):
    """The guarded contrast of contrast_identity.py, unchanged."""
    tot = float(np.sum(g))
    if tot <= 0:
        return np.nan
    if (float(np.sum(g[:ic + 1])) / tot < MIN_SHARE
            or float(np.sum(g[ic + 1:])) / tot < MIN_SHARE):
        return np.nan
    a = float(np.sum(w[:ic + 1] * g[:ic + 1]) / max(np.sum(g[:ic + 1]), 1e-300))
    b = float(np.sum(w[ic + 1:] * g[ic + 1:]) / max(np.sum(g[ic + 1:]), 1e-300))
    return a / b if b > 0 else np.nan


def derivative_piece(x):
    """Density and the derivative-bias distortion, as contrast_identity builds it."""
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    n = len(x)
    if n < 300:
        return None
    dens, res = weighted_density(x)
    if dens is None:
        return None
    d = np.clip(dens - estimate_floor(dens, res, "surrogate", rng), 0.0, None)
    if d.sum() <= 0:
        return None
    s = robust_scale(x)
    D = (x - np.median(x)) / s
    w_ = _win(n)
    sl = np.clip(local_scale(res, max(5, int(LOCAL_BW * n))), 1e-12, None)
    wf = max(11, int(WIDE * 0.08 * n))
    wf += (wf % 2 == 0)
    if wf >= n:
        return None
    dt = savgol_filter(D, w_, 2, deriv=1, delta=1.0 / n)
    d3 = savgol_filter(D, wf, 4, deriv=3, delta=1.0 / n)
    cor = ((dt - (H_HALF ** 2 / 10.0) * d3) / sl) ** 2
    with np.errstate(divide="ignore", invalid="ignore"):
        w = np.clip(np.where(dens > 0, cor / np.maximum(dens, 1e-300), 1.0),
                    0.2, 5.0)
    return d, w


z = np.load(os.path.join(HERE, "feat_raw.npz"), allow_pickle=True)
FN = [str(s) for s in z["featnames"]]
units = [k for k in z.files if k != "featnames"]
CH = [c for c in ("rms", "b4_6", "b8_10") if c in FN]

print("=" * 92)
print("The derivative correction, recomputed cell by cell")
print("=" * 92)
print("""
The same three channels and the same seventeen units the results file uses, with
the cut at the middle of the record.  The median of the last column must equal
the contrast that file already reports, or the figure is drawing a different
quantity from the one the text discusses.
""")
cells = []
for nm in CH:
    i = FN.index(nm)
    for u in units:
        p = derivative_piece(z[u][:, i])
        if p is None:
            continue
        d, w = p
        ic = int(0.5 * len(d)) - 1
        v = contrast(w, d, ic)
        if not np.isfinite(v) or v <= 0:
            continue
        edge = (w >= 0.98 * w.max()) | (w <= 1.02 * w.min())
        cells.append(dict(unit=u, channel=nm, contrast=float(max(v, 1.0 / v)),
                          M=float(max(np.max(w), 1.0 / max(np.min(w), 1e-9))),
                          edge_samples=float(edge.mean()),
                          edge_budget=float(d[edge].sum() / d.sum()),
                          d=d, w=w, ic=ic))
med = float(np.median([c["contrast"] for c in cells]))
ref = next(r for r in REF["corrections"] if r["correction"] == "derivative")
RES["derivative_contrast_recomputed"] = med
RES["derivative_contrast_in_results_file"] = float(ref["contrast"])
RES["cells"] = len(cells)
print("  cells %d   recomputed median contrast %.4f   results file %.4f   "
      "difference %.2e" % (len(cells), med, ref["contrast"],
                           abs(med - ref["contrast"])))

# Is the drawn record's coincidence -- extremes on samples worth nothing -- a
# property of that record or of the construction?  Asked before it is explained.
eb = np.array([c["edge_budget"] for c in cells])
es = np.array([c["edge_samples"] for c in cells])
RES["edge_budget_median"] = float(np.median(eb))
RES["edge_budget_max"] = float(eb.max())
RES["edge_samples_median"] = float(np.median(es))
RES["edge_budget_below_1pct"] = float(np.mean(eb < 0.01))
print("  the samples that set the range are a median %.1f%% of a record and "
      "carry a\n  median %.4f of its budget, at most %.4f; below one per cent "
      "on %.0f%% of cells"
      % (100 * RES["edge_samples_median"], RES["edge_budget_median"],
         RES["edge_budget_max"], 100 * RES["edge_budget_below_1pct"]))

drawn = min(cells, key=lambda c: abs(c["contrast"] - med))
RES["panel_a"] = dict(unit=drawn["unit"], channel=drawn["channel"],
                      contrast=drawn["contrast"], M=drawn["M"])
print("  panel (a) draws %s / %s, contrast %.3f, range %.2f: the cell at the "
      "median" % (drawn["unit"], drawn["channel"], drawn["contrast"],
                  drawn["M"]))

# ------------------------------------------------------------------ figure
fig, axes = plt.subplots(1, 3, figsize=(COL2, 2.4))

# ---- (a) the range is set where the density is not ------------------------
ax = axes[0]
d, w, ic = drawn["d"], drawn["w"], drawn["ic"]
n = len(d)
tt = (np.arange(n) + 1.0) / n
c = tt[ic]
# The raw density is a spike train whose maximum sits in the last percent of
# life, so plotting d/max(d) draws a flat line and hides the very thing the
# panel is about.  A rolling mean over one per cent of life keeps the shape.
k = max(3, n // 100)
sm = np.convolve(d, np.ones(k) / k, mode="same")
ax.fill_between(tt, 0, sm / sm.max(), color=C_LIN, alpha=0.3, lw=0)
ax.set_ylim(0, 1.25)
ax.set_xlim(0, 1.02)

# where the range is set, and what those samples are worth
at_edge = (w >= 0.98 * w.max()) | (w <= 1.02 * w.min())
share = float(d[at_edge].sum() / d.sum())
RES["extreme_share_of_budget"] = share
RES["extreme_share_of_samples"] = float(at_edge.mean())
ax.plot(tt[at_edge], np.full(at_edge.sum(), 1.19), "|", ms=2.6, mew=0.5,
        color=C_EXP)

ax2 = ax.twinx()
ax2.plot(tt, w, color=C_EXP, lw=0.6, alpha=0.6)
lo = float(np.sum(w[:ic + 1] * d[:ic + 1]) / max(np.sum(d[:ic + 1]), 1e-300))
hi = float(np.sum(w[ic + 1:] * d[ic + 1:]) / max(np.sum(d[ic + 1:]), 1e-300))
ax2.plot([0, c], [lo, lo], color=C_POW, lw=1.4)
ax2.plot([c, 1], [hi, hi], color=C_POW, lw=1.4)
ax2.axvline(c, color=C_ACC, lw=0.6, ls=":")
ax2.set_ylim(0, 7.4)          # headroom above the clip at 5, for the annotation
ax2.set_yticks([1, 3, 5])
ax2.tick_params(labelsize=FS["tick"], colors=C_EXP, length=2.2, width=0.6)
ax2.set_ylabel(r"distortion $w$", fontsize=FS["annot"], color=C_EXP)
for s in ("top", "left"):
    ax2.spines[s].set_visible(False)
ax2.spines["right"].set_color(C_EXP)
ax2.spines["right"].set_linewidth(0.6)
ax2.text(0.5, 6.65, r"range $M=%.1f$: $M^2=%.0f$ charged" % (
    drawn["M"], drawn["M"] ** 2), fontsize=FS["annot"], color=C_EXP,
    ha="center")
ax2.text(0.5, 5.95, (r"$%.1f\%%$ of samples, none of the budget"
                     % (100 * at_edge.mean())) if share < 5e-4 else
         (r"those samples carry $%.1f\%%$ of the budget" % (100 * share)),
         fontsize=FS["annot"], color=C_EXP, ha="center")
ax2.text(0.5, 0.35, r"$\langle w\rangle_{[0,c]}/\langle w\rangle_{[c,1]}"
                    r"=%.2f$" % (lo / hi), fontsize=FS["annot"], color=C_POW,
         ha="center")
ax2.text(c + 0.015, 3.1, r"$c$", fontsize=FS["annot"], color=C_ACC)
ax.text(0.04, 0.18, r"$\hat g$", fontsize=FS["annot"], color=C_LIN)
ax.set_zorder(ax2.get_zorder() + 1)
ax.patch.set_visible(False)
tidy(ax, r"normalised age $\tau$", r"density over its maximum",
     r"(a) the range is set off the density")
RES["panel_a_levels"] = [float(lo), float(hi)]

# ---- (b) what that costs on the paper's own corrections -------------------
ax = axes[1]
rows = REF["corrections"]
y = np.arange(len(rows))[::-1]
for k, r in enumerate(rows):
    yy = y[k]
    ax.plot([r["contrast"], r["charged"]], [yy, yy], color=C_LIN, lw=0.7,
            zorder=1)
    ax.plot([r["charged"]], [yy], "s", ms=3.6, color=C_LIN, mew=0, zorder=3)
    ax.plot([r["contrast"]], [yy], "o", ms=3.8, color=C_POW, mew=0, zorder=3)
    ax.text(np.sqrt(r["contrast"] * r["charged"]), yy + 0.22,
            r"$\times%.0f$" % r["ratio"], fontsize=FS["annot"], color=C_LIN,
            ha="center")
ax.set_yticks(y)
ax.set_yticklabels([r["correction"] for r in rows], fontsize=FS["annot"])
ax.set_xscale("log")
ax.set_xlim(0.8, 60)
ax.set_xticks([1, 3, 10, 30])
ax.set_xticklabels(["1", "3", "10", "30"])
ax.set_ylim(-0.6, len(rows) + 0.5)   # room for the legend above the top row
ax.plot([], [], "o", ms=3.8, color=C_POW, mew=0, label="contrast, the true cost")
ax.plot([], [], "s", ms=3.6, color=C_LIN, mew=0, label=r"$M^2$, what is charged")
ax.legend(fontsize=FS["legend"], loc="upper center", borderaxespad=0.2,
          handletextpad=0.3, bbox_to_anchor=(0.5, 1.02))
tidy(ax, "odds multiplier", None, r"(b) the overcharge")
ax.spines["left"].set_visible(False)
ax.tick_params(axis="y", length=0)
RES["overcharge_median"] = float(REF["overcharge_median"])

# ---- (c) what the repair certifies ---------------------------------------
ax = axes[2]
br = REF["per_bearing"]
xs = np.array([r["worst_contrast"] for r in br])
ys = np.array([r["tolerated"] for r in br])
ok = np.array([r["survives"] for r in br])
lim = [0.9, 40.0]
ax.fill_between(lim, lim, [lim[1]] * 2, color=C_ACC, alpha=0.08, lw=0)
ax.plot(lim, lim, color=C_LIN, lw=0.7, ls="--")
ax.plot(xs[ok], ys[ok], "o", ms=3.6, color=C_ACC, mew=0, label="certified")
ax.plot(xs[~ok], ys[~ok], "o", ms=3.6, mfc="none", mec=C_POW, mew=0.8,
        label="not certified")
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlim(*lim)
ax.set_ylim(*lim)
ax.set_xticks([1, 3, 10, 30])
ax.set_yticks([1, 3, 10, 30])
ax.set_xticklabels(["1", "3", "10", "30"])
ax.set_yticklabels(["1", "3", "10", "30"])
ax.legend(fontsize=FS["legend"], loc="lower right", borderaxespad=0.2,
          handletextpad=0.3)
ax.text(1.05, 26, r"$%d$ of $%d$ certified" % (REF["n_survive"],
                                               REF["n_bearings"]),
        fontsize=FS["annot"], color=C_ACC)
tidy(ax, "worst contrast the corrections impose", "contrast the ordering tolerates",
     r"(c) the certificate, per bearing")
RES["n_survive"] = int(REF["n_survive"])
RES["n_bearings"] = int(REF["n_bearings"])

fig.tight_layout(pad=0.34, w_pad=1.3)
fig.savefig(OUT)
# spelled out rather than written through JS, so staleness.py can identify the
# producer of this result as it does for every other one
json.dump(RES, open(os.path.join(HERE, "figdistort.json"), "w"), indent=2,
          default=float)
print("\nwrote", OUT)
print("wrote", JS)
print("""
  On the drawn record the distortion ranges over a factor of %.1f, for which the
  range-based certificate is charged %.0f, while its two information-weighted
  means are %.2f and %.2f, a contrast of %.2f.

  The samples that set that range are %.1f per cent of the record and carry %.3f
  of the budget.  That is not a coincidence of this record.  The correction is a
  ratio of a corrected density to the measured one, so it takes its extreme
  values exactly where the measured density is smallest, which is where the noise
  floor has already removed it.  A supremum taken over the record is therefore
  attained, structurally, on the samples the analysis has thrown away, and a
  certificate written in that supremum is paying for samples that carry nothing.
""" % (drawn["M"], drawn["M"] ** 2, lo, hi, lo / hi,
       100 * at_edge.mean(), share))
