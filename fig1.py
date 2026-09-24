"""
Figure 1 -- the information curve, how it is read, and why it is not fitted.

(a) measured curves for three bearing indicators. They are not variations on one
    shape: root-mean-square amplitude delivers everything in the last few percent
    of life, the high-frequency band delivers most of it in the first third, and
    no single-parameter family holds both.
(b) the reading. Every design quantity is a value, a quantile or an increment of
    the curve, so the construction is the whole method.
(c) why reading rather than fitting. Mean absolute quantile error against
    synthetic records with known curves: the direct reading is the same 0.004 of
    life whatever the shape, while a fitted power law matches it only where the
    truth is a power law.
"""
import os
import json
import numpy as np
import matplotlib.pyplot as plt
from figstyle import COL2, FS, C_POW, C_EXP, C_ACC, C_LIN, C_STR, tidy
from nonparam import weighted_density, estimate_floor

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(os.path.dirname(HERE), "paper", "fig1.pdf")
GRID = np.linspace(0, 1, 401)


def mean_curve(series, rng):
    C = []
    for x in series:
        dens, res = weighted_density(np.asarray(x, float))
        if dens is None:
            continue
        d = np.clip(dens - estimate_floor(dens, res, "surrogate", rng, reps=6),
                    0.0, None)
        if d.sum() <= 0:
            continue
        F = np.cumsum(d) / d.sum()
        C.append(np.interp(GRID, np.arange(1, len(F) + 1) / len(F), F))
    return np.median(np.vstack(C), axis=0) if C else None


rng = np.random.default_rng(24)
z = np.load(os.path.join(HERE, "feat_raw.npz"), allow_pickle=True)
FN = [str(s) for s in z["featnames"]]
I = {k: i for i, k in enumerate(FN)}
BK = sorted(k for k in z.files if k != "featnames")
CH = [("RMS", ["rms"], C_POW),
      ("0--2\\,kHz", ["b0_1", "b1_2"], C_STR),
      ("4--10\\,kHz", ["b4_6", "b6_8", "b8_10"], C_EXP)]

fig, axes = plt.subplots(1, 3, figsize=(COL2, 2.3))

# ---- (a) measured curves ---------------------------------------------------
ax = axes[0]
curves = {}
for lab, parts, col in CH:
    c = mean_curve([sum(z[b][:, I[p]] for p in parts) for b in BK], rng)
    if c is None:
        continue
    curves[lab] = c
    ax.plot(GRID, c, color=col, lw=1.3, label=lab)
# A legend rather than labels dropped on the curves: at three curves the text
# landed on the lines it named, and one reference label had to be set at 38
# degrees to fit at all.
ax.plot([0, 1], [0, 1], color=C_LIN, lw=0.8, ls="--", label="uniform")
ax.legend(fontsize=FS["legend"], loc="upper left", handlelength=1.5,
          borderaxespad=0.15, labelspacing=0.3)
ax.set_xlim(0, 1); ax.set_ylim(0, 1.02)
ax.set_xticks([0, 0.5, 1]); ax.set_yticks([0, 0.5, 1])
tidy(ax, r"normalised age $\tau$", r"fraction of budget accumulated",
     r"(a) measured curves")

# ---- (b) the reading -------------------------------------------------------
ax = axes[1]
c = curves.get("0--2\\,kHz", list(curves.values())[0])
ax.plot(GRID, c, color=C_STR, lw=1.3, zorder=3)
q = 0.15
tmin = float(GRID[int(np.searchsorted(c, q))])
ax.plot([0, tmin], [q, q], color=C_ACC, lw=0.8, ls=":", zorder=2)
ax.plot([tmin, tmin], [0, q], color=C_ACC, lw=0.8, ls=":", zorder=2)
ax.plot([tmin], [q], "o", ms=3.4, color=C_ACC, zorder=5)
ax.text(0.02, q + 0.04, r"target", fontsize=FS["annot"], color=C_ACC)
# to the right of its own dotted line and below the target level, which is the
# one corner of this panel that nothing else occupies
ax.text(tmin + 0.03, 0.02, r"$\tau_{\min}$", fontsize=FS["annot"],
        color=C_ACC, ha="left", va="bottom")
t0 = 0.90
c0 = float(np.interp(t0, GRID, c))
lo = float(GRID[int(np.searchsorted(c, max(c0 - q, 0.0)))])
ax.fill_between([lo, t0], c0 - q, c0, color="#444444", alpha=0.11, lw=0,
                zorder=1)
ax.plot([lo, t0], [c0 - q] * 2, color="#444444", lw=0.7, zorder=2)
ax.plot([t0, t0], [c0 - q, c0], color="#444444", lw=0.7, zorder=2)
ax.annotate("", xy=(lo, 0.10), xytext=(t0, 0.10),
            arrowprops=dict(arrowstyle="<->", lw=0.6, color="#444444"))
ax.text((lo + t0) / 2, 0.14, r"$w^{\ast}$", fontsize=FS["annot"], ha="center")
ax.text(t0, 0.04, r"$\tau_0$", fontsize=FS["annot"], ha="center",
        va="bottom")
ax.set_xlim(0, 1.04); ax.set_ylim(0, 1.02)
ax.set_xticks([0, 0.5, 1]); ax.set_yticks([0, 0.5, 1])
tidy(ax, r"normalised age $\tau$", None, r"(b) reading the design off it")

# ---- (c) reading against fitting -------------------------------------------
ax = axes[2]
V = json.load(open(os.path.join(HERE, "nonparam_validate.json")))
order = sorted(V, key=lambda r: r["err_beta"])
y = np.arange(len(order))
for i, r in enumerate(order):
    ax.plot([r["err_nonparam"], r["err_beta"]], [i, i], "-", lw=0.7,
            color="#bbbbbb", zorder=1)
ax.plot([r["err_nonparam"] for r in order], y, "o", ms=3.4, color=C_ACC,
        mew=0, label="read", zorder=3)
ax.plot([r["err_beta"] for r in order], y, "s", ms=3.4, color=C_POW, mew=0,
        label=r"fitted $\beta$", zorder=3)
ax.set_yticks(y)
ax.set_yticklabels([r["shape"].replace("power law b=", r"$\beta=$")
                    for r in order], fontsize=FS["annot"])
ax.set_xscale("log")
ax.set_xlim(0.0008, 0.4)
ax.set_xticks([0.001, 0.01, 0.1])
ax.set_xticklabels(["0.001", "0.01", "0.1"])
ax.legend(fontsize=FS["legend"], loc="lower right", borderaxespad=0.2,
          handletextpad=0.3)
tidy(ax, r"quantile error (fraction of life)", None,
     r"(c) reading against fitting")
ax.spines["left"].set_visible(False)
ax.tick_params(axis="y", length=0)

fig.tight_layout(pad=0.34, w_pad=1.1)
fig.subplots_adjust(bottom=0.22)
fig.savefig(OUT)
print("wrote", OUT)
print(f"  (b) target {q}, tau_min {tmin:.3f}, tau_0 {t0}, w* {t0-lo:.3f}")
for lab, c in curves.items():
    print(f"  (a) {lab:<14} reaches {q} at tau = "
          f"{GRID[int(np.searchsorted(c, q))]:.3f}")
