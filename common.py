import numpy as np
import matplotlib
matplotlib.use("pgf")
import matplotlib.pyplot as plt
from scipy.special import gammainc, gammaincinv
from matplotlib.path import Path
from matplotlib.patches import PathPatch, FancyArrowPatch, Rectangle
from matplotlib import transforms

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
    "lines.linewidth": 1.3,
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "xtick.major.size": 2.6,
    "ytick.major.size": 2.6,
    "xtick.minor.size": 1.4,
    "ytick.minor.size": 1.4,
    "legend.borderpad": 0.35,
    "legend.handletextpad": 0.5,
})

# ---- shared typography, so both figures read as one set ----
FS = dict(label=7.8, tick=6.6, title=8.0, legend=6.1, annot=6.0)

# ---- palette --------------------------------------------------------------
# Categorical slots, fixed order, never cycled. Chosen by search over published
# CVD-safe qualitative sets and validated all-pairs against normal, protan and
# deuteran vision (work/validate_palette.py, work/search_palette.py):
# worst normal dE 17.0 (gate 15), worst CVD dE 11.1 (target 8).
# The previous steel-blue / teal pair failed the normal-vision gate at 14.6.
C_EXP   = "#0072B2"   # slot 1  Okabe-Ito blue
C_POW   = "#D55E00"   # slot 2  Okabe-Ito vermillion
C_STR   = "#AA3377"   # slot 3  Tol purple
C_GRAY  = "#8C8C8C"   # neutral reference, not an identity slot
C_INFEAS= "#D55E00"   # the infeasible band belongs to the power-law class
C_NARROW= "#D55E00"
C_WIDE  = "#0072B2"

# Sequential scale for window width: one hue, light->dark, truncated so the
# light end still clears 2:1 on white (Blues 0.45-1.0 -> L 0.76..0.32, 2.13:1).
SEQ_CMAP = "Blues"
SEQ_LO, SEQ_HI = 0.45, 1.0

# ---- information distribution F and its inverse (Table I of the Letter) ----
# F(tau) = int_0^tau D'^2 / int_0^1 D'^2 is the normalized cumulative Fisher
# information. Everything the theory needs is an increment or an inverse of F:
#     rho_w(tau0) = F(tau0) - F((tau0-w)_+)          <= F(tau0)
#     w*(tau0)    = tau0 - Finv(F(tau0) - q*)        feasible iff F(tau0) >= q*
# so the three degradation classes differ only in F.

def F_exp(tau, alpha):
    return (1 - np.exp(-2*alpha*tau)) / (1 - np.exp(-2*alpha))

def Finv_exp(u, alpha):
    return -np.log(1 - u*(1 - np.exp(-2*alpha))) / (2*alpha)

def F_pow(tau, beta):
    return tau**(2*beta - 1)

def Finv_pow(u, beta):
    return u**(1.0/(2*beta - 1))

def F_str(tau, alpha_b, b):
    s = 2 - 1.0/b
    return gammainc(s, 2*alpha_b*tau**b) / gammainc(s, 2*alpha_b)

def Finv_str(u, alpha_b, b):
    s = 2 - 1.0/b
    x = gammaincinv(s, u * gammainc(s, 2*alpha_b))
    return (x / (2*alpha_b))**(1.0/b)


# ---- the two derived quantities, class-independent ----
def _rho(F, tau0, w):
    return F(tau0) - F(max(tau0 - w, 0.0))

def _wstar(F, Finv, tau0, qstar):
    target = F(tau0) - qstar
    return np.nan if target < 0 else tau0 - Finv(target)


def rho_exp(tau0, w, alpha):
    return _rho(lambda t: F_exp(t, alpha), tau0, w)

def rho_pow(tau0, w, beta):
    return _rho(lambda t: F_pow(t, beta), tau0, w)

def rho_str(tau0, w, alpha_b, b):
    return _rho(lambda t: F_str(t, alpha_b, b), tau0, w)

def invert_w_exp(tau0, qstar, alpha):
    return _wstar(lambda t: F_exp(t, alpha), lambda u: Finv_exp(u, alpha),
                  tau0, qstar)

def invert_w_pow(tau0, qstar, beta):
    return _wstar(lambda t: F_pow(t, beta), lambda u: Finv_pow(u, beta),
                  tau0, qstar)

def invert_w_str(tau0, qstar, alpha_b, b):
    return _wstar(lambda t: F_str(t, alpha_b, b),
                  lambda u: Finv_str(u, alpha_b, b), tau0, qstar)

def gradient_fill_path(ax, xs, ys_top, ys_bot, cmap, alpha=0.85, zorder=1):
    """Vertical colour gradient clipped to the area between ys_bot and ys_top."""
    xs = np.asarray(xs); ys_top = np.asarray(ys_top); ys_bot = np.asarray(ys_bot)
    z = np.linspace(0, 1, 256).reshape(-1, 1)
    x0, x1 = xs.min(), xs.max()
    y0, y1 = ys_bot.min(), ys_top.max()
    im = ax.imshow(z, aspect="auto", extent=[x0, x1, y0, y1], origin="lower",
                    cmap=cmap, alpha=alpha, zorder=zorder)
    verts = np.column_stack([np.concatenate([xs, xs[::-1]]),
                              np.concatenate([ys_top, ys_bot[::-1]])])
    path = Path(verts)
    patch = PathPatch(path, facecolor="none", edgecolor="none")
    ax.add_patch(patch)
    im.set_clip_path(patch)
    return im
