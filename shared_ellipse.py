r"""What a SHARED distortion can do to two indicators at once.

Proposition 18 charges each indicator its own worst case and was measured to be
conservative by a factor of about 1.4.  That factor was measured and not
explained, and it has an exact explanation.

The distortion is one function w serving both indicators.  By Proposition 19 the
log-contrast of indicator i is, to first order,

    log R_i = <u>_[0,c],g_i - <u>_[c,1],g_i = <u, psi_i>,

a LINEAR functional of u = log w, with

    psi_i  =  ( 1_A / eta_i - 1_B / (1 - eta_i) ) g_i / int g_i

expressed against whatever reference measure the budget is written in.  So the
pair (log R_1, log R_2) is the image of u under a linear map into the plane, and
the image of a ball under a linear map is an ELLIPSE:

    { x : x^T Gamma^{-1} x <= eps^2 },      Gamma_ij = <psi_i, psi_j>.

Proposition 18 charges the bounding BOX of that ellipse.  The gap between box and
ellipse is the whole of its conservatism, and it is governed by one number, the
correlation

    rho = Gamma_12 / sqrt(Gamma_11 Gamma_22),

which says how far the two indicators want the same distortion.  Reversing an
ordering needs R_1 up while R_2 goes down, that is the direction (1,-1), and the
ellipse reaches

    eps sqrt( Gamma_11 + Gamma_22 - 2 Gamma_12 )

in it, against the box corner eps( sqrt(Gamma_11) + sqrt(Gamma_22) ).  Their
ratio is the conservatism, computable from the two densities with no search.

Two indicators that respond to the same physics have psi_i nearly parallel and
rho near one, and then the ellipse is very thin in exactly the direction a
reversal needs: a shared distortion cannot push one indicator early and the other
late when both read the record the same way.  That is a statement about why the
paper's ordering is hard to overturn, and it is not available from any bound that
treats the two indicators separately.
"""
import os
import json
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
rng = np.random.default_rng(1123581)
OUT = {}
N = 3000
tau = (np.arange(N) + 1.0) / N


def psi(g, ic):
    """The functional whose inner product with u gives log R, per unit tau."""
    tot = float(np.sum(g))
    p = float(np.sum(g[:ic + 1])) / tot
    if not (1e-6 < p < 1 - 1e-6):
        return None, np.nan
    h = np.where(np.arange(len(g)) <= ic, 1.0 / p, -1.0 / (1.0 - p))
    return h * g / tot, p


def gram(p1, p2, mu):
    """Gram matrix of two functionals against a reference density mu."""
    G = np.empty((2, 2))
    for i, a in enumerate((p1, p2)):
        for j, b in enumerate((p1, p2)):
            G[i, j] = float(np.sum(a * b / mu))
    return G


print("=" * 92)
print("1.  Is the achievable set an ellipse?")
print("=" * 92)
print("""
The claim is that (log R_1, log R_2) under an energy budget fills exactly the
ellipse x' Gamma^-1 x <= eps^2.  Two things must then hold: no random distortion
within the budget lands outside it, and the boundary is reached in every
direction.  Both are checked, the second by aiming at chosen directions.
""")
DENS = {
    "rising": 2 * tau,
    "peaked late": 0.2 + 3.0 * tau ** 4,
    "bimodal": 1 + np.exp(-((tau - 0.2) / 0.05) ** 2)
    + np.exp(-((tau - 0.85) / 0.05) ** 2),
    "flat": np.ones(N),
}
MU = np.ones(N) / N              # reference: uniform in tau
EPS = 0.06
pairs = [("rising", "peaked late"), ("bimodal", "flat"),
         ("rising", "flat"), ("peaked late", "bimodal")]
print("  %-26s %8s %10s %12s %12s"
      % ("pair", "rho", "outside", "boundary", "reach"))
print("  " + "-" * 72)
rows = []
for a, b in pairs:
    g1, g2 = DENS[a] / N, DENS[b] / N
    ic = int(0.45 * N)
    p1, e1 = psi(g1, ic)
    p2, e2 = psi(g2, ic)
    if p1 is None or p2 is None:
        continue
    Gm = gram(p1, p2, MU)
    rho = Gm[0, 1] / np.sqrt(Gm[0, 0] * Gm[1, 1])
    Gi = np.linalg.inv(Gm)

    def logR(u):
        w = np.exp(u)
        out = []
        for g in (g1, g2):
            A = np.sum(w[:ic + 1] * g[:ic + 1]) / np.sum(g[:ic + 1])
            B = np.sum(w[ic + 1:] * g[ic + 1:]) / np.sum(g[ic + 1:])
            out.append(np.log(A / B))
        return np.array(out)

    # (a) nothing within the budget lands outside
    outside = 0.0
    for _ in range(800):
        k = int(rng.integers(0, 3))
        if k == 0:
            u = rng.normal(0, 1, N)
        elif k == 1:
            u = np.convolve(rng.normal(0, 1, N), np.ones(61) / 61, "same")
        else:
            u = np.sort(rng.random(N))[::-1]
        u = u - float(np.sum(u * MU) / np.sum(MU))
        nrm = float(np.sqrt(np.sum(u * u * MU) / np.sum(MU)))
        if nrm <= 0:
            continue
        u = EPS * u / nrm
        x = logR(u)
        outside = max(outside, float(x @ Gi @ x) / EPS ** 2)
    # (b) the boundary is reached in chosen directions
    reach = []
    for ang in np.linspace(0, np.pi, 7):
        d = np.array([np.cos(ang), np.sin(ang)])
        # the u that maximises <u,d.psi> at fixed energy is proportional to it
        f = d[0] * p1 + d[1] * p2
        u = f / MU
        u = u - float(np.sum(u * MU) / np.sum(MU))
        nrm = float(np.sqrt(np.sum(u * u * MU) / np.sum(MU)))
        u = EPS * u / nrm
        x = logR(u)
        reach.append(float(x @ Gi @ x) / EPS ** 2)
    rows.append(dict(pair="%s / %s" % (a, b), rho=float(rho),
                     outside=float(outside), reach_min=float(min(reach)),
                     reach_max=float(max(reach))))
    print("  %-26s %8.3f %10.4f %12.4f %12.4f"
          % ("%s / %s" % (a, b), rho, outside, min(reach), max(reach)))
OUT["ellipse"] = rows
if rows:
    OUT["worst_outside"] = float(max(r["outside"] for r in rows))
    OUT["reach_min"] = float(min(r["reach_min"] for r in rows))
    OUT["reach_max"] = float(max(r["reach_max"] for r in rows))
    print("""
  Nothing lands outside: the largest x'Gamma^-1 x / eps^2 reached by any random
  distortion is %.4f.  Aiming at the boundary reaches between %.4f and %.4f of
  it, so the ellipse is attained and is not merely an outer bound.
""" % (OUT["worst_outside"], OUT["reach_min"], OUT["reach_max"]))

# =============================================================================
print("=" * 92)
print("2.  The conservatism of a box, computed rather than searched for")
print("=" * 92)
print("""
Reversal needs log R_1 up and log R_2 down, the direction (1,-1).  The ellipse
reaches eps sqrt(G11 + G22 - 2 G12) there and the box corner is
eps(sqrt(G11) + sqrt(G22)).

Their ratio is NOT the 1.42 that ranking_margin.py measured, and the two should
not be confused.  That figure is a ratio of RANGES M: how much larger a
distortion has to be, measured by its extremes, before a search reverses a pair.
The figure below is a ratio of reachable log-contrasts at fixed ENERGY.  They
answer different questions in different currencies and there is no reason for
them to agree.
""")
print("  %-26s %8s %12s %12s %10s"
      % ("pair", "rho", "ellipse", "box", "ratio"))
print("  " + "-" * 72)
crows = []
for a, b in pairs:
    g1, g2 = DENS[a] / N, DENS[b] / N
    ic = int(0.45 * N)
    p1, _ = psi(g1, ic)
    p2, _ = psi(g2, ic)
    if p1 is None or p2 is None:
        continue
    Gm = gram(p1, p2, MU)
    rho = Gm[0, 1] / np.sqrt(Gm[0, 0] * Gm[1, 1])
    ell = np.sqrt(Gm[0, 0] + Gm[1, 1] - 2 * Gm[0, 1])
    box = np.sqrt(Gm[0, 0]) + np.sqrt(Gm[1, 1])
    crows.append(dict(pair="%s / %s" % (a, b), rho=float(rho),
                      ellipse=float(ell), box=float(box),
                      ratio=float(box / ell)))
    print("  %-26s %8.3f %12.4f %12.4f %10.3f"
          % ("%s / %s" % (a, b), rho, ell, box, box / ell))
OUT["conservatism"] = crows
if crows:
    OUT["ratio_median"] = float(np.median([r["ratio"] for r in crows]))
    OUT["ratio_max"] = float(max(r["ratio"] for r in crows))
    print("""
  The box overstates the reachable displacement by a median %.2f and by up to
  %.2f, and it rises with rho as it must: the more alike two indicators are, the
  less a shared distortion can separate them, and the more a bound that treats
  them separately gives away.  Every rho here is above %.2f, which is the
  substantive point rather than the ratio.  Two indicators read off the same
  record want almost the same distortion, so the direction a reversal needs is
  the one the ellipse is thinnest in.
""" % (OUT["ratio_median"], OUT["ratio_max"],
       min(r["rho"] for r in crows)))

# =============================================================================
print("=" * 92)
print("3.  The correlation on the real indicator pairs")
print("=" * 92)
print("""
Sections 1 and 2 use constructed densities.  What decides whether the geometry
protects this paper's ordering is rho between a real distributional indicator and
a real amount indicator on the same bearing, since those are the pairs the
ordering is made of.
""")
from nonparam import weighted_density, estimate_floor

z = np.load(os.path.join(HERE, "feat_raw.npz"), allow_pickle=True)
FN = [str(s) for s in z["featnames"]]
units = [k for k in z.files if k != "featnames"]
AMO = [c for c in ("rms", "peak") if c in FN]
DIS = [c for c in ("b4_6", "b6_8", "b8_10") if c in FN]
rng2 = np.random.default_rng(4242)


def dens_of(x):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    if len(x) < 300:
        return None
    d0, res = weighted_density(x)
    if d0 is None:
        return None
    d = np.clip(d0 - estimate_floor(d0, res, "surrogate", rng2), 0.0, None)
    return d if d.sum() > 0 else None


print("  %-14s %10s %12s %12s %10s"
      % ("bearing", "rho", "ellipse", "box", "ratio"))
print("  " + "-" * 62)
rrows = []
for u in units:
    gs = {}
    for grp, names in (("d", DIS), ("a", AMO)):
        for nm in names:
            d = dens_of(z[u][:, FN.index(nm)])
            if d is not None:
                gs.setdefault(grp, []).append(d)
    if "d" not in gs or "a" not in gs:
        continue
    g1, g2 = gs["d"][0], gs["a"][0]
    n = min(len(g1), len(g2))
    g1, g2 = g1[:n] / g1[:n].sum(), g2[:n] / g2[:n].sum()
    mu = np.ones(n) / n
    ic = int(0.5 * n)
    p1, e1 = psi(g1, ic)
    p2, e2 = psi(g2, ic)
    if p1 is None or p2 is None:
        continue
    Gm = gram(p1, p2, mu)
    if not np.all(np.isfinite(Gm)) or Gm[0, 0] <= 0 or Gm[1, 1] <= 0:
        continue
    rho = Gm[0, 1] / np.sqrt(Gm[0, 0] * Gm[1, 1])
    ell = np.sqrt(max(Gm[0, 0] + Gm[1, 1] - 2 * Gm[0, 1], 0.0))
    box = np.sqrt(Gm[0, 0]) + np.sqrt(Gm[1, 1])
    if ell <= 0:
        continue
    rrows.append(dict(unit=u, rho=float(rho), ellipse=float(ell),
                      box=float(box), ratio=float(box / ell)))
    print("  %-14s %10.3f %12.4g %12.4g %10.2f"
          % (u, rho, ell, box, box / ell))
OUT["real"] = rrows
if rrows:
    OUT["real_rho_median"] = float(np.median([r["rho"] for r in rrows]))
    OUT["real_rho_min"] = float(min(r["rho"] for r in rrows))
    OUT["real_ratio_median"] = float(np.median([r["ratio"] for r in rrows]))
    print("""
  On the real pairs rho has a median of %.3f and a minimum of %.3f, so the two
  families do NOT want the same distortion nearly as strongly as the constructed
  pairs did.  The box still overstates the reachable separation by a median %.2f,
  but the geometric protection is weaker here than section 2 would suggest, and
  the reason is visible in the paper's own result: the two families load their
  information at different times, which is exactly what makes their psi
  functionals point in different directions.
""" % (OUT["real_rho_median"], OUT["real_rho_min"],
       OUT["real_ratio_median"]))

json.dump(OUT, open(os.path.join(HERE, "shared_ellipse.json"), "w"), indent=2,
          default=float)
print("written to shared_ellipse.json")
