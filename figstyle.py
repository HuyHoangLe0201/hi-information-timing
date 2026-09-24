"""
Shared figure style. The palette is carried over unchanged from the earlier
study, where it was chosen by search and validated all-pairs against normal,
protan and deuteran vision: worst normal-vision dE 17.0 against a gate of 15,
worst CVD dE 11.1 against a target of 8.

Elsevier single-column width is 90 mm and double-column 190 mm; sizes below are
in inches so figures are placed at their natural size and never rescaled, which
would change the effective font size.
"""
import numpy as np
import matplotlib
matplotlib.use("pgf")
import matplotlib.pyplot as plt
from scipy.special import gammainc, gammaincinv

PREAMBLE = "\n".join([
    r"\usepackage[T1]{fontenc}",
    r"\usepackage[cmintegrals]{newtxmath}",
    r"\renewcommand{\rmdefault}{ptm}",
    r"\usepackage{amsmath}",
])

plt.rcParams.update({
    "pgf.texsystem": "pdflatex",
    "text.usetex": True,
    "pgf.rcfonts": False,
    "pgf.preamble": PREAMBLE,
    "font.family": "serif",
    "axes.linewidth": 0.6,
    "lines.linewidth": 1.25,
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "xtick.major.size": 2.6,
    "ytick.major.size": 2.6,
    "xtick.minor.size": 1.4,
    "ytick.minor.size": 1.4,
    "legend.borderpad": 0.35,
    "legend.handletextpad": 0.5,
    "legend.frameon": False,
})

COL1, COL2 = 3.54, 7.48          # 90 mm and 190 mm, in inches
# Raised once, after the figures were read at print size: annotations at
# 6.4pt against a 10pt body are legible only under magnification, and the
# cramped inline labels they forced were the main reason the panels read as
# cluttered.
FS = dict(label=8.4, tick=7.2, title=8.6, legend=7.2, annot=7.4)

C_LIN = "#8C8C8C"                # linear reference
C_POW = "#D55E00"                # power law
C_EXP = "#0072B2"                # exponential
C_STR = "#AA3377"                # stretched exponential
C_ACC = "#009E73"                # accent for annotation
SEQ_CMAP, SEQ_LO, SEQ_HI = "Blues", 0.45, 1.0


def seq(n):
    cm = plt.get_cmap(SEQ_CMAP)
    return [cm(x) for x in np.linspace(SEQ_LO, SEQ_HI, n)]


def tidy(ax, xlabel=None, ylabel=None, title=None):
    ax.tick_params(labelsize=FS["tick"])
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=FS["label"])
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=FS["label"])
    if title:
        ax.set_title(title, fontsize=FS["title"])
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    return ax


# --- the closed forms of Table 1, so figures and text cannot disagree -------
def F_lin(t):
    return np.asarray(t, float)


def F_pow(t, beta):
    return np.asarray(t, float) ** (2 * beta - 1)


def Finv_pow(u, beta):
    return np.asarray(u, float) ** (1.0 / (2 * beta - 1))


def F_exp(t, a):
    return (1 - np.exp(-2 * a * np.asarray(t, float))) / (1 - np.exp(-2 * a))


def Finv_exp(u, a):
    return -np.log(1 - np.asarray(u, float) * (1 - np.exp(-2 * a))) / (2 * a)


def F_str(t, ab, b):
    s = 2 - 1.0 / b
    return gammainc(s, 2 * ab * np.asarray(t, float) ** b) / gammainc(s, 2 * ab)


def Finv_str(u, ab, b):
    s = 2 - 1.0 / b
    x = gammaincinv(s, np.asarray(u, float) * gammainc(s, 2 * ab))
    return (x / (2 * ab)) ** (1.0 / b)


def wstar(F, Finv, tau0, q):
    """Minimum window, Eq. (12): tau0 - F^{-1}(F(tau0) - q*)."""
    target = F(tau0) - q
    return np.nan if target < 0 else tau0 - Finv(target)
