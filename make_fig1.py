"""
Figure 1 -- the information distribution F, and the design rule it inverts to.

Panel (a) draws F itself, which the text calls the central object but which had
no picture: each degradation class is a different way of distributing Fisher
information over a lifetime, and the whole of Section III is then geometry on
this one curve. Reading off panel (a):
  * tau_min is where F crosses q*  -- left of it the target is unreachable;
  * w* is the horizontal gap that spans a vertical drop of q* below F(tau0),
    which is exactly Proposition 2, drawn for the exponential class at tau0=0.7.
Panel (b) is that inversion evaluated over all ages.
"""
import os
import sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
LETTER = os.path.join(os.path.dirname(HERE), "lsens_letter_v2")
sys.path.insert(0, LETTER)
from common import *          # noqa: F401,F403

FIGW, FIGH = 3.45, 2.76
FS_LABEL, FS_TICK, FS_TITLE, FS_LEG, FS_ANNOT = (
    FS["label"], FS["tick"], FS["title"], FS["legend"], FS["annot"])

A, BETA, AB, BSTR = 2.0, 3.0, 3.0, 0.7
QSTAR, TAU0 = 0.35, 0.7

fig = plt.figure(figsize=(FIGW, FIGH))
axA = fig.add_axes([0.155, 0.630, 0.825, 0.320])
axB = fig.add_axes([0.155, 0.125, 0.825, 0.300])

# ---------------------------------------------------------------- (a) F(tau)
t = np.linspace(1e-4, 1.0, 600)
curves = [
    (t, ':', C_GRAY, 1.0, r'Linear'),
    (np.array([F_str(x, AB, BSTR) for x in t]), '-.', C_STR, 1.3, r'Str.-exp.'),
    (np.array([F_exp(x, A) for x in t]), '-', C_EXP, 1.3, r'Exponential'),
    (np.array([F_pow(x, BETA) for x in t]), '--', C_POW, 1.3, r'Power-law'),
]
for y, ls, c, lw, lab in curves:
    axA.plot(t, y, ls, color=c, lw=lw, label=lab, zorder=3)

axA.axhline(QSTAR, color='0.3', lw=0.8, ls=(0, (4, 2)), zorder=2)
axA.text(0.995, QSTAR + 0.03, r'$q^{\ast}$', fontsize=FS_ANNOT, color='0.3',
         ha='right', va='bottom')

# tau_min = F^-1(q*) for each class: where the curve crosses q*
for finv, c in [(Finv_exp(QSTAR, A), C_EXP),
                (Finv_str(QSTAR, AB, BSTR), C_STR),
                (Finv_pow(QSTAR, BETA), C_POW)]:
    axA.plot([finv], [QSTAR], 'o', ms=3.2, color=c, zorder=6,
             markeredgecolor='white', markeredgewidth=0.6)
axA.annotate(r'$\tau_{\min}=F^{-1}(q^{\ast})$', xy=(Finv_pow(QSTAR, BETA), QSTAR),
             xytext=(0.50, 0.215), fontsize=FS_ANNOT, color=C_POW, ha='center',
             arrowprops=dict(arrowstyle='-|>', color=C_POW, lw=0.7,
                             connectionstyle='arc3,rad=-0.25', shrinkA=1, shrinkB=3))

# the w* construction, exponential class at tau0
f0 = F_exp(TAU0, A)
lo = Finv_exp(f0 - QSTAR, A)
axA.plot([TAU0, TAU0], [0, f0], color=C_EXP, lw=0.7, ls=(0, (1, 1.3)), zorder=2)
axA.plot([lo, lo], [0, f0 - QSTAR], color=C_EXP, lw=0.7, ls=(0, (1, 1.3)), zorder=2)
axA.plot([lo, TAU0], [f0 - QSTAR, f0 - QSTAR], color=C_EXP, lw=0.7,
         ls=(0, (1, 1.3)), zorder=2)
axA.annotate('', xy=(TAU0, f0), xytext=(TAU0, f0 - QSTAR),
             arrowprops=dict(arrowstyle='<->', color=C_EXP, lw=0.9, shrinkA=0, shrinkB=0))
axA.text(TAU0 - 0.022, f0 - QSTAR / 2, r'$q^{\ast}$', fontsize=FS_ANNOT,
         color=C_EXP, ha='right', va='center')
axA.annotate('', xy=(lo, 0.055), xytext=(TAU0, 0.055),
             arrowprops=dict(arrowstyle='<->', color=C_EXP, lw=0.9, shrinkA=0, shrinkB=0))
axA.text((lo + TAU0) / 2, 0.085, r'$w^{\ast}$', fontsize=FS_ANNOT, color=C_EXP,
         ha='center', va='bottom')

axA.set_xlim(0, 1); axA.set_ylim(0, 1.04)
axA.set_xlabel(r'Normalized time $\tau$', fontsize=FS_LABEL, labelpad=1.5)
axA.set_ylabel(r'$F(\tau)$', fontsize=FS_LABEL)
axA.set_title(r'(a) Information distribution', fontsize=FS_TITLE, pad=3)
axA.tick_params(labelsize=FS_TICK)
for s in ['top', 'right']:
    axA.spines[s].set_visible(False)

# ---------------------------------------------------------------- (b) w*
tau0 = np.linspace(0.14, 1.0, 400)
w_exp = np.array([invert_w_exp(x, QSTAR, A) for x in tau0])
w_pow = np.array([invert_w_pow(x, QSTAR, BETA) for x in tau0])
w_str = np.array([invert_w_str(x, QSTAR, AB, BSTR) for x in tau0])
t_min = tau0[~np.isnan(w_pow)].min()

axB.axvspan(0.14, t_min, facecolor=C_INFEAS, alpha=0.09, zorder=0, lw=0)
axB.axvspan(0.14, t_min, facecolor='none', edgecolor=C_INFEAS, alpha=0.30,
            hatch='////', zorder=0, lw=0)
# Linear degradation: F(tau)=tau, so w* = tau0 - (tau0 - q*) = q* for every
# age -- a constant window, which is the reference the other classes depart from.
w_lin = np.where(tau0 >= QSTAR, QSTAR, np.nan)
axB.plot(tau0, w_lin, ':', color=C_GRAY, lw=1.1)
axB.plot(tau0, w_str, '-.', color=C_STR, lw=1.3)
axB.plot(tau0, w_exp, '-', color=C_EXP, lw=1.3)
axB.plot(tau0, w_pow, '--', color=C_POW, lw=1.3)
axB.axvline(t_min, color=C_POW, lw=0.8, ls=(0, (1, 1.4)))
axB.plot([TAU0], [TAU0 - lo], 'o', ms=3.4, color=C_EXP, zorder=6,
         markeredgecolor='white', markeredgewidth=0.6)
axB.text(0.56, 0.045, 'no window suffices\n(power-law)', fontsize=FS_ANNOT - 0.3,
         color=C_INFEAS, ha='center', va='bottom', style='italic')

ticks = sorted([x for x in [0.2, 0.4, 0.6, 0.8, 1.0] if abs(x - t_min) > 0.05] + [t_min])
axB.set_xticks(ticks)
axB.set_xticklabels([f'{x:.2f}' if abs(x - t_min) < 1e-9 else f'{x:.1f}' for x in ticks])
for lb, x in zip(axB.get_xticklabels(), ticks):
    if abs(x - t_min) < 1e-9:
        lb.set_color(C_POW); lb.set_fontweight('bold'); lb.set_fontsize(FS_TICK - 0.6)

axB.set_xlim(0.14, 1.0); axB.set_ylim(0, 1.0)
axB.set_xlabel(r'Elapsed life $\tau_0$', fontsize=FS_LABEL, labelpad=1.5)
axB.set_ylabel(r'$w^{\ast}(\tau_0)$', fontsize=FS_LABEL)
axB.set_title(r'(b) Minimum window', fontsize=FS_TITLE, pad=3)
axB.tick_params(labelsize=FS_TICK)
# The legend lives in panel (b): its upper left is empty hatched area, whereas
# in (a) the steep stretched-exponential curve runs through any legend box.
axB.legend(*axA.get_legend_handles_labels(), loc='upper left', ncol=2,
           frameon=True, facecolor='white', framealpha=0.9, edgecolor='none',
           fontsize=FS_LEG, handlelength=1.6, columnspacing=0.9,
           borderaxespad=0.25, labelspacing=0.2)
for s in ['top', 'right']:
    axB.spines[s].set_visible(False)

fig.savefig(os.path.join(LETTER, "fig1.pdf"))
print(f"fig1.pdf written: tau_min(power-law)={t_min:.3f}, "
      f"w*_exp(tau0={TAU0})={TAU0-lo:.4f}")
