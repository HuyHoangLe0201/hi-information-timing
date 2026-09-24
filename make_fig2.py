"""
Figure 2 -- empirical validation on PRONOSTIA.

(a) Attainment of the windowed Cramer-Rao floor: measured age-error sd against
    the floor predicted from each bearing's own information profile.
(b) Validity radius of the linearization: the floor is attained while the prior
    age error stays below a few percent of life, and is left behind past that.
(c) The calibrated degradation exponent is a property of the indicator, not of
    the bearing: the same accelerometer stream yields beta_eff from 0.7 to 30
    depending on which band the health indicator is built from.
"""
import os
import sys
import json
import numpy as np
from scipy.signal import savgol_filter

HERE = os.path.dirname(os.path.abspath(__file__))
LETTER = os.path.join(os.path.dirname(HERE), "lsens_letter_v2")
sys.path.insert(0, LETTER)
from common import *          # noqa: F401,F403  (style + palette)

prof = json.load(open(os.path.join(HERE, "indicator_profiles.json")))

FIGW, FIGH = 7.10, 2.42
# Same type scale as fig1, so the two figures read as one set.
FS_LABEL, FS_TICK, FS_TITLE, FS_LEG, FS_ANNOT = (
    FS["label"], FS["tick"], FS["title"], FS["legend"], FS["annot"])


# ---------------------------------------------------------------- panel (a)
# Cells come from oof_validate.py configuration A2: out-of-fold residuals and a
# local noise scale, which is the form of (3) valid when the indicator's noise
# is not stationary. Regenerate with `python oof_validate.py`.
cz = np.load(os.path.join(HERE, "cells_A2.npz"))
crb, emp, wcol, tcol = cz["crb"], cz["emp"], cz["w"], cz["tau0"]
slope, icept = np.polyfit(np.log(crb), np.log(emp), 1)
rho = float(np.corrcoef(np.log(crb), np.log(emp))[0, 1])
oof = json.load(open(os.path.join(HERE, "oof_results.json")))
noise_only = oof["configs"]["A2"]["median"]

fig = plt.figure(figsize=(FIGW, FIGH))
gs = fig.add_gridspec(1, 3, wspace=0.52, left=0.075, right=0.985, top=0.862, bottom=0.205)
axA, axB, axC = (fig.add_subplot(gs[0, i]) for i in range(3))

import matplotlib.colors as mcolors
_base = plt.get_cmap(SEQ_CMAP)
SEQ = mcolors.LinearSegmentedColormap.from_list(
    "seq", _base(np.linspace(SEQ_LO, SEQ_HI, 256)))
# Coloured by age, not by window width: w runs monotonically along the diagonal,
# so colouring by it merely restates the x position. tau0 does not, so the colour
# answers the question the pooled literature sweeps cannot -- whether the bound
# holds at every age.
sc = axA.scatter(crb, emp, c=tcol, s=4.5, cmap=SEQ, alpha=0.6,
                 linewidths=0, zorder=3, rasterized=True)
lim = [min(crb.min(), emp.min()) * 0.7, max(crb.max(), emp.max()) * 1.4]
axA.plot(lim, lim, '-', color='0.25', lw=1.0, zorder=4, label='CRB (slope 1)')
xs = np.array(lim)
axA.plot(xs, np.exp(icept) * xs ** slope, '--', color=C_POW, lw=1.2, zorder=5,
         label=rf'fit: slope {slope:.2f}')
axA.set_xscale('log'); axA.set_yscale('log')
axA.set_xlim(lim); axA.set_ylim(lim)
axA.set_xlabel(r'Predicted floor $\Gamma_w(\tau_0)^{-1/2}$', fontsize=FS_LABEL)
axA.set_ylabel(r'Measured age-error sd', fontsize=FS_LABEL)
axA.set_title(r'(a) Floor attainment, $r=%.2f$' % rho, fontsize=FS_TITLE)
axA.tick_params(labelsize=FS_TICK)
axA.legend(loc='upper left', frameon=False, fontsize=FS_LEG, handlelength=1.8,
           borderaxespad=0.15, labelspacing=0.25)
cb = fig.colorbar(sc, ax=axA, pad=0.025, fraction=0.036, aspect=26)
cb.set_label(r'age $\tau_0$', fontsize=FS_ANNOT - 0.5, labelpad=1.5)
cb.ax.tick_params(labelsize=FS_TICK - 1.4, length=1.8, pad=1.2)
for s in ['top', 'right']:
    axA.spines[s].set_visible(False)

# ---------------------------------------------------------------- panel (b)
pb = oof["partB"]
d = np.array([r["delta"] for r in pb]); ratio = np.array([r["ratio"] for r in pb])
i_break = int(np.argmax(ratio > 2.5))
radius = d[i_break - 1]

# The validity region itself is shaded, so the reader sees the answer without
# following an arrow; the noise-only level ties this panel to panel (a).
axB.axvspan(d.min() * 0.7, radius, color=C_EXP, alpha=0.09, lw=0, zorder=0)
axB.axhline(1.0, color='0.3', lw=0.8, ls=(0, (4, 2)), zorder=2)
axB.axhline(noise_only, color=C_EXP, lw=0.8, ls=(0, (1, 1.4)), zorder=2)
axB.plot(d, ratio, 'o-', color=C_POW, ms=3.4, lw=1.3, zorder=3,
         markeredgecolor='white', markeredgewidth=0.6)
axB.text(radius * 0.93, 11.5, r'valid: $|\delta|\lesssim %.0f\%%$' % (100 * radius),
         fontsize=FS_ANNOT, color=C_EXP, ha='right', va='center')
axB.text(d.max() * 1.02, noise_only, 'noise only', fontsize=FS_ANNOT - 0.3,
         color=C_EXP, ha='right', va='bottom')
axB.text(d.max() * 1.02, 1.0, 'floor', fontsize=FS_ANNOT - 0.3, color='0.3',
         ha='right', va='top')
axB.set_xscale('log'); axB.set_yscale('log')
axB.set_xlabel(r'Prior age error $|\delta|$', fontsize=FS_LABEL)
axB.set_ylabel(r'Measured sd / floor', fontsize=FS_LABEL)
axB.set_title(r'(b) Linearization radius', fontsize=FS_TITLE)
axB.tick_params(labelsize=FS_TICK)
for s in ['top', 'right']:
    axB.spines[s].set_visible(False)

# ---------------------------------------------------------------- panel (c)
lbl = {"rms": "RMS", "peak": "peak", "kurtosis": "kurtosis",
       "lf 0-2kHz": r"0--2\,kHz", "mf 2-6kHz": r"2--6\,kHz",
       "hf 4-10kHz": r"4--10\,kHz", "vhf 10-12.8kHz": r"10--12.8\,kHz"}
# Sorted by the quantity plotted: unsorted bars made the panel read as if the
# order meant something. Descending, so the longest sits at the top.
order = sorted(lbl, key=lambda k: prof["indicators"][k]["tau_min_obs"],
               reverse=True)
beta = [prof["indicators"][k]["beta_eff"] for k in order]
tmin = [prof["indicators"][k]["tau_min_obs"] for k in order]
ypos = np.arange(len(order))[::-1]          # first entry drawn at the top
cols = [C_GRAY if k in ("rms", "peak", "kurtosis") else C_STR for k in order]

axC.barh(ypos, tmin, height=0.62, color=cols, alpha=0.75, zorder=2,
         edgecolor='white', linewidth=0.6)
axC.axvline(prof["tau_min_pow3"], color=C_POW, lw=1.2, ls=(0, (4, 2)), zorder=4)
axC.text(prof["tau_min_pow3"] - 0.03, len(order) - 0.35,
         r'nominal $\beta{=}3$', fontsize=FS_ANNOT, color=C_POW, ha='right', va='center')
# Labels anchored to the START of each bar, not its end: at the end they collide
# with the nominal-beta rule wherever a bar happens to terminate near it.
for y, bb, tt in zip(ypos, beta, tmin):
    inside = tt > 0.42
    axC.text(0.022 if inside else tt + 0.022, y,
             rf'$\beta_{{\mathrm{{eff}}}}{{=}}{bb:.1f}$',
             fontsize=FS_ANNOT - 0.8, va='center', ha='left',
             color='white' if inside else '0.25', zorder=5)
axC.set_yticks(ypos); axC.set_yticklabels([lbl[k] for k in order], fontsize=FS_TICK)
axC.set_xlim(0, 1.06)
axC.set_xlabel(r'Earliest reachable age $\tau_{\min}$', fontsize=FS_LABEL)
axC.set_title(r'(c) The class is set by the indicator', fontsize=FS_TITLE)
axC.tick_params(labelsize=FS_TICK)
for s in ['top', 'right']:
    axC.spines[s].set_visible(False)

fig.savefig(os.path.join(LETTER, "fig2.pdf"), dpi=300)
print(f"fig2.pdf written to {LETTER}")
print(f"panel (a): slope {slope:.3f}, r {rho:.3f}, cells {len(crb)}")
print(f"panel (b): validity radius |delta| <= {d[i_break-1]:.3f}")
print(f"panel (c): beta_eff {min(beta):.2f} .. {max(beta):.2f}")
