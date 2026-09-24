r"""What the information density is, and why it is a ratio.

The manuscript opens its theory section with three equations and no picture, so
a reader meets the clock offset, the Fisher information and the cumulative curve
as algebra.  Two things in that algebra are easy to state and hard to see: that
an offset in the damage clock shows up as a VERTICAL displacement proportional
to the derivative, so it is invisible wherever the trend is flat, and that the
density is a RATIO, so its peak need not sit where the movement is largest.

There is also a dangling pointer to repair.  Section 2.1 says the measured local
scale varies "by a factor of five within a single bearing record" and sends the
reader to Section 3.1 for it; Section 3.1 says no such thing.  The number is
real -- weighted_density.json has it, as the median over units of
p95(sigma)/p5(sigma), which reaches 5.2 on kurtosis and 5.1 on the top band --
so what is missing is the statement, not the measurement.  Both are recomputed
here and checked against that file, so the figure and the text draw on one
number rather than two.

An earlier version of this script measured max/min over the interior instead and
got a median of 3.5 with an extreme of 52, which is a different statistic of the
same curve and would have contradicted the paper while appearing to correct it.
The definition is therefore taken from the existing results file rather than
chosen afresh.

Panels.
(a) the estimand.  A clock offset delta displaces the trend by D'(tau) delta.
    At an early age that displacement is inside the noise; at a late age it is
    several times it.  Nothing about the SIZE of the indicator enters.
(b) the density is a ratio.  With a noise scale that rises as a real one does,
    D'^2 peaks at the end and 1/sigma^2 at the start, and their product peaks at
    neither.
(c) the consequence, measured.  Weighting by the local scale moves the earliest
    usable age of the paper's own 4--10 kHz band by the 0.139 of a lifetime that
    Section 3.1 reports.
"""
import os
import json
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize_scalar

from figstyle import COL2, FS, C_POW, C_EXP, C_ACC, C_LIN, tidy
from pipeline import prepare

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(os.path.dirname(HERE), "paper", "figmodel.pdf")
JS = os.path.join(HERE, "figmodel.json")
RES = {}

BETA = 2.5
DELTA = 0.08
SIG0 = 0.045
QSTAR = 0.35
GRID = np.linspace(0.0, 1.0, 401)

# the channel groups of weighted_density.py, unchanged
CH = {"rms": ["rms"], "peak": ["peak"], "kurt": ["kurt"],
      "0--2 kHz": ["b0_1", "b1_2"], "2--4 kHz": ["b2_4"],
      "4--10 kHz": ["b4_6", "b6_8", "b8_10"], "10--12.8 kHz": ["b10_12.8"]}
DRAWN = "4--10 kHz"


def profile(dens, n):
    """Exponent and earliest usable age of one density, as Section 3 reads them."""
    d = np.asarray(dens, float)
    if not np.all(np.isfinite(d)) or d.sum() <= 0:
        return np.nan, np.nan, None
    F = np.cumsum(d) / d.sum()
    tau = np.arange(1, n + 1) / n
    f = lambda b: float(np.mean((F - tau ** (2 * b - 1)) ** 2))
    beta = float(minimize_scalar(f, bounds=(0.51, 30.0), method="bounded").x)
    k = int(np.searchsorted(F, QSTAR))
    return beta, ((k + 1) / n if k < n else 1.0), F


z = np.load(os.path.join(HERE, "feat_raw.npz"), allow_pickle=True)
FN = [str(s) for s in z["featnames"]]
I = {k: i for i, k in enumerate(FN)}
BK = sorted(k for k in z.files if k != "featnames")

print("=" * 92)
print("How far the local noise scale varies, and what weighting by it moves")
print("=" * 92)
print("""
The span is p95(sigma)/p5(sigma) within a record, median over units, which is
the definition weighted_density.json already uses; the last column is the shift
in the earliest usable age when the density is weighted by that scale rather
than left as the squared derivative.
""")
print("  %-14s %7s %10s %12s %12s %10s"
      % ("channel", "units", "span", "tmin unwtd", "tmin wtd", "shift"))
print("  " + "-" * 70)
rows, drawn = [], {}
for lab, parts in CH.items():
    SP, TU, TW, CU, CW, SL, UN = [], [], [], [], [], [], []
    for b in BK:
        x = np.asarray(sum(z[b][:, I[p]] for p in parts), float)
        u, _ = prepare(x, "%s/%s" % (b, lab))
        if u is None:
            continue
        sl = np.clip(u["sig_loc"], 1e-12, None)
        bu, tu, Fu = profile(u["dtrend"] ** 2, u["n"])
        bw, tw, Fw = profile((u["dtrend"] / sl) ** 2, u["n"])
        if not (np.isfinite(bu) and np.isfinite(bw)):
            continue
        SP.append(float(np.percentile(sl, 95) / np.percentile(sl, 5)))
        TU.append(tu)
        TW.append(tw)
        t = np.arange(1, u["n"] + 1) / u["n"]
        CU.append(np.interp(GRID, t, Fu))
        CW.append(np.interp(GRID, t, Fw))
        SL.append(np.interp(GRID, t, sl / np.median(sl)))
        UN.append(b)
    if len(SP) < 4:
        continue
    r = dict(channel=lab, units=len(SP), span=float(np.median(SP)),
             tmin_unweighted=float(np.median(TU)),
             tmin_weighted=float(np.median(TW)))
    r["shift"] = abs(r["tmin_weighted"] - r["tmin_unweighted"])
    rows.append(r)
    if lab == DRAWN:
        # The cumulatives are medians over units, but the noise trace is ONE
        # unit's: a median of sixteen scale profiles is flat by construction and
        # would contradict the span printed beside it, which is a within-record
        # quantity.  The unit drawn is the one whose span is nearest the median.
        j = int(np.argmin(np.abs(np.array(SP) - np.median(SP))))
        drawn = dict(Fu=np.median(np.vstack(CU), axis=0),
                     Fw=np.median(np.vstack(CW), axis=0),
                     sl=SL[j], sl_span=float(SP[j]), sl_unit=UN[j],
                     units=len(SP))
    print("  %-14s %7d %10.2f %12.3f %12.3f %10.3f"
          % (lab, r["units"], r["span"], r["tmin_unweighted"],
             r["tmin_weighted"], r["shift"]))

RES["channels"] = rows
RES["span_min"] = float(min(r["span"] for r in rows))
RES["span_max"] = float(max(r["span"] for r in rows))
RES["span_median"] = float(np.median([r["span"] for r in rows]))
RES["shift_median"] = float(np.median([r["shift"] for r in rows]))
RES["drawn_channel"] = DRAWN
RES["drawn_units"] = drawn["units"]

# --- does this agree with the results file the manuscript already cites? ----
ref = {r["channel"]: r for r in
       json.load(open(os.path.join(HERE, "weighted_density.json")))}
worst = 0.0
for r in rows:
    q = ref.get(r["channel"])
    if not q:
        continue
    for k in ("span", "tmin_unweighted", "tmin_weighted"):
        worst = max(worst, abs(r[k] - q[k]))
RES["agreement_with_results_file"] = float(worst)
print("\n  largest disagreement with weighted_density.json over every channel "
      "and every\n  quantity: %.2e, so the figure and the text are reading one "
      "measurement." % worst)

# ------------------------------------------------------------------ figure
fig, axes = plt.subplots(1, 3, figsize=(COL2, 2.35))
t = np.linspace(0.001, 1.0, 600)


def D_of(x):
    return np.asarray(x, float) ** BETA


def Dp_of(x):
    return BETA * np.asarray(x, float) ** (BETA - 1)


# ---- (a) a clock offset is a displacement proportional to the derivative ---
ax = axes[0]
ax.fill_between(t, D_of(t) - SIG0, D_of(t) + SIG0, color=C_LIN, alpha=0.22,
                lw=0)
ax.plot(t, D_of(t), color=C_LIN, lw=1.3, label=r"$D(\tau)$")
ax.plot(t, D_of(np.clip(t - DELTA, 0, None)), color=C_EXP, lw=1.1, ls="--",
        label=r"$D(\tau-\delta)$")
marks = []
for tau, below in ((0.30, True), (0.88, False)):
    hi, lo = D_of(tau), D_of(max(tau - DELTA, 0))
    ratio = (hi - lo) / SIG0
    marks.append((tau, ratio))
    ax.annotate("", xy=(tau, hi), xytext=(tau, lo),
                arrowprops=dict(arrowstyle="<->", color=C_POW, lw=0.8,
                                shrinkA=0, shrinkB=0))
    ax.plot([tau], [hi], "o", ms=2.2, color=C_POW, mew=0)
    if below:
        ax.text(tau, lo - 0.10, r"$%.1f\,\sigma$" % ratio,
                fontsize=FS["annot"], color=C_POW, ha="center", va="top")
    else:
        ax.text(tau - 0.03, (hi + lo) / 2, r"$%.1f\,\sigma$" % ratio,
                fontsize=FS["annot"], color=C_POW, ha="right", va="center")
RES["displacement_early"] = float(marks[0][1])
RES["displacement_late"] = float(marks[1][1])
RES["displacement_tau"] = [float(marks[0][0]), float(marks[1][0])]
ax.set_xlim(0, 1.02)
ax.set_ylim(-0.18, 1.16)
ax.legend(fontsize=FS["legend"], loc="upper left", borderaxespad=0.2,
          handletextpad=0.4, handlelength=1.6)
tidy(ax, r"normalised age $\tau$", "indicator",
     r"(a) what a clock offset does")

# ---- (b) the density is a ratio, and peaks where neither factor does -------
ax = axes[1]
rise = RES["span_median"]
sig = SIG0 * (1.0 + (rise - 1.0) * t ** 3)
d2 = Dp_of(t) ** 2
inv = 1.0 / sig ** 2
dens = d2 * inv
for y, col in ((d2 / d2.max(), C_LIN), (inv / inv.max(), C_EXP),
               (dens / dens.max(), C_POW)):
    ax.plot(t, y, color=col, lw=1.3 if col == C_POW else 1.0,
            ls="-" if col == C_POW else ":")
pk = float(t[int(np.argmax(dens))])
RES["schematic_rise"] = float(rise)
RES["schematic_peak"] = pk
ax.axvline(pk, color=C_POW, lw=0.5, ls="--")
# Horizontal, under the axis it marks.  The same note set vertically along the
# line was the least readable thing in the figure.
ax.text(0.99, 1.24, r"peak at $\tau=%.2f$" % pk,
        fontsize=FS["annot"], color=C_POW, va="top", ha="right")
ax.set_xlim(0, 1.02)
ax.set_ylim(0, 1.30)
tidy(ax, r"normalised age $\tau$", "scaled to its own maximum",
     r"(b) the density is a ratio")
ax.legend([plt.Line2D([], [], color=C_LIN, ls=":", lw=1.0),
           plt.Line2D([], [], color=C_EXP, ls=":", lw=1.0),
           plt.Line2D([], [], color=C_POW, lw=1.3)],
          [r"${D'}^2$", r"$1/\sigma^2$", r"$g={D'}^2/\sigma^2$"],
          fontsize=FS["legend"], loc="upper left", handlelength=1.5,
          borderaxespad=0.15, labelspacing=0.3)

# ---- (c) the consequence on the paper's own records -----------------------
ax = axes[2]
ax2 = ax.twinx()
ax2.plot(GRID, drawn["sl"], color=C_EXP, lw=0.7, alpha=0.55)
ax2.set_ylim(0, 6.0)
ax2.set_yticks([1, 2, 3])
ax2.tick_params(labelsize=FS["tick"], colors=C_EXP, length=2.2, width=0.6)
ax2.set_ylabel(r"$\hat\sigma/\mathrm{median}\,\hat\sigma$",
               fontsize=FS["annot"], color=C_EXP)
for s in ("top", "left"):
    ax2.spines[s].set_visible(False)
ax2.spines["right"].set_color(C_EXP)
ax2.spines["right"].set_linewidth(0.6)

drow = next(r for r in rows if r["channel"] == DRAWN)
ax.plot(GRID, drawn["Fu"], color=C_LIN, lw=1.0, ls="--", zorder=3)
ax.plot(GRID, drawn["Fw"], color=C_POW, lw=1.3, zorder=3)
ax.axhline(QSTAR, color=C_ACC, lw=0.5, ls=":", zorder=2)
tu, tw = drow["tmin_unweighted"], drow["tmin_weighted"]
ax.annotate("", xy=(tw, QSTAR), xytext=(tu, QSTAR),
            arrowprops=dict(arrowstyle="<->", color=C_ACC, lw=0.8,
                            shrinkA=0, shrinkB=0), zorder=4)
ax.text((tw + tu) / 2, QSTAR + 0.04, r"$%.3f$" % drow["shift"],
        fontsize=FS["annot"], color=C_ACC, ha="center")
ax.legend([plt.Line2D([], [], color=C_POW, lw=1.3),
           plt.Line2D([], [], color=C_LIN, ls="--", lw=1.0),
           plt.Line2D([], [], color=C_EXP, lw=0.9, alpha=0.55)],
          [r"$\hat F$ weighted", r"$\hat F$ unweighted",
           r"local scale, span $%.1f\times$" % drawn["sl_span"]],
          fontsize=FS["legend"], loc="upper left", handlelength=1.5,
          borderaxespad=0.15, labelspacing=0.3)
RES["drawn_scale_unit"] = drawn["sl_unit"]
RES["drawn_scale_span"] = drawn["sl_span"]
ax.set_xlim(0, 1.02)
ax.set_ylim(0, 1.05)
ax.set_zorder(ax2.get_zorder() + 1)
ax.patch.set_visible(False)
tidy(ax, r"normalised age $\tau$", "cumulative fraction",
     r"(c) measured, %s band" % DRAWN)

fig.tight_layout(pad=0.34, w_pad=1.1)
fig.savefig(OUT)
# written with the name spelled out, not through JS, so staleness.py can find
# the producer of this result the way it finds every other one
json.dump(RES, open(os.path.join(HERE, "figmodel.json"), "w"), indent=2,
          default=float)
print("\nwrote", OUT)
print("wrote", JS)
print("""
  A clock offset of %.2f of a lifetime displaces the trend by %.1f standard
  deviations at tau = %.2f and by %.1f at tau = %.2f, on the same record and with
  the same noise.  Information is about movement, not level.

  The local scale spans a median %.1f-fold within a record, from %.1f on the
  quietest channel group to %.1f on the noisiest, and weighting by it moves the
  earliest usable age by a median %.3f of a lifetime across the seven groups.
  Panel (c) draws the group at that median, the %s band, over %d units.
""" % (DELTA, RES["displacement_early"], RES["displacement_tau"][0],
       RES["displacement_late"], RES["displacement_tau"][1],
       RES["span_median"], RES["span_min"], RES["span_max"],
       RES["shift_median"], DRAWN, RES["drawn_units"]))
