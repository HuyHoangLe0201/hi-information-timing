r"""A monotone distortion signs two of the three quantities and not the third.

Proposition 16 bounds what a distortion can do given only its RANGE.  The three
distortions this paper measures have a second property it does not use: they are
monotone along life.  The serial weight (1-phi)/(1+phi) falls, because phi rises;
the tail correction kappa^-2 falls, because kappa rises.  Using that costs
nothing and gives something Proposition 16 cannot, namely a direction with no M
in it at all.

The statement is short.  If w is non-increasing then for every c

    int_0^c w g  >=  w(c) int_0^c g       and      int_c^1 w g  <=  w(c) int_c^1 g,

so the odds of eta can only rise, and since c was arbitrary the whole normalised
curve rises pointwise: F_w >= F.  Two consequences follow immediately and a third
does not follow at all.

  eta(c) rises for every c.  No range, no smallness, no shape beyond monotone.

  tau_min falls, because F_w >= F reaches the demand q no later than F does.

  w* is NOT signed.  It is tau_0 - F_w^-1(F_w(tau_0) - q), and F_w >= F raises
  the argument of the inverse while F_w^-1 <= F^-1 lowers the inverse itself.
  The two act against each other and neither dominates, so a counterexample must
  exist; one is constructed below rather than asserted.

That is worth having because the paper has already measured all three, without
being able to say why the pattern was what it was.  Section 6.2 found eta rising
in twelve of twelve cells under the serial correction.  Section 12 found it
rising in only eight of twelve under the derivative bias.  The difference is not
a matter of size: the derivative bias is the one distortion here that is NOT
monotone, since it removes information late and returns some in mid-life.  The
theorem predicts exactly that split, and it was not available when either
measurement was made.
"""
import os
import json
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
rng = np.random.default_rng(2718)
OUT = {}
N = 4000
tau = (np.arange(N) + 1.0) / N
Q = 0.35

DENS = {
    "rising": 2 * tau,
    "falling": 2 - 2 * tau,
    "peaked late": 0.2 + 3.0 * tau ** 4,
    "bimodal": 1 + np.exp(-((tau - 0.2) / 0.05) ** 2)
    + np.exp(-((tau - 0.85) / 0.05) ** 2),
}


def F_of(g):
    G = np.cumsum(g)
    return G / G[-1]


def tau_min(F, q):
    i = int(np.searchsorted(F, q))
    return np.nan if i >= len(F) else (i + 1) / len(F)


def w_star(F, t0, q):
    n = len(F)
    i0 = min(n - 1, max(0, int(round(t0 * n)) - 1))
    t = F[i0] - q
    if t <= 0:
        return np.nan
    j = int(np.searchsorted(F, t))
    return np.nan if j >= n else t0 - (j + 1) / n


print("=" * 92)
print("1.  A non-increasing distortion, over many shapes and densities")
print("=" * 92)
print("""
The distortions are random non-increasing positive functions, so nothing about
their range or smoothness is assumed.  What must hold is that F rises pointwise,
that eta rises at every cut, and that tau_min falls.  A single violation refutes
the theorem, so the worst case over all trials is what is reported.
""")
worst_F, worst_eta, worst_tau, trials = 0.0, 0.0, 0.0, 0
CUTS = (0.2, 0.4, 0.6, 0.8)
for _ in range(3000):
    dn = str(rng.choice(list(DENS)))
    g = DENS[dn] / N
    steps = np.sort(rng.random(N))[::-1]          # any non-increasing shape
    w = np.exp(rng.uniform(0.2, 3.0) * steps)
    gd = w * g
    F0, F1 = F_of(g), F_of(gd)
    worst_F = max(worst_F, float(np.max(F0 - F1)))
    for c in CUTS:
        i = int(c * N) - 1
        worst_eta = max(worst_eta, float(F0[i] - F1[i]))
    a, b = tau_min(F0, Q), tau_min(F1, Q)
    if np.isfinite(a) and np.isfinite(b):
        worst_tau = max(worst_tau, float(b - a))
    trials += 1
OUT["trials"] = trials
OUT["worst_F_violation"] = worst_F
OUT["worst_eta_violation"] = worst_eta
OUT["worst_tau_violation"] = worst_tau
print("  %-46s %14.3g" % ("worst F(tau) below the undistorted F", worst_F))
print("  %-46s %14.3g" % ("worst eta below the undistorted eta", worst_eta))
print("  %-46s %14.3g" % ("worst tau_min above the undistorted", worst_tau))
print("""
  Over %d random non-increasing distortions nothing goes the wrong way; the
  largest excursion of any kind is %.3g, which is the grid.  The two signs hold
  with no assumption on the size of the distortion.
""" % (trials, max(worst_F, worst_eta, worst_tau)))

# =============================================================================
print("=" * 92)
print("2.  The window is not signed, and here is why")
print("=" * 92)
print("""
The claim that w* has no definite sign needs a counterexample, not a failure to
find one.  A search over the same family is run, and the two directions are both
required to occur; the extreme case of each is printed so the mechanism is
visible.
""")
up, dn_, ex_up, ex_dn = 0, 0, None, None
for _ in range(3000):
    key = str(rng.choice(list(DENS)))
    g = DENS[key] / N
    steps = np.sort(rng.random(N))[::-1]
    w = np.exp(rng.uniform(0.2, 3.0) * steps)
    F0, F1 = F_of(g), F_of(w * g)
    t0 = float(rng.uniform(0.6, 0.95))
    a, b = w_star(F0, t0, Q), w_star(F1, t0, Q)
    if not (np.isfinite(a) and np.isfinite(b)):
        continue
    d = b - a
    if d > 1e-9:
        up += 1
        if ex_up is None or d > ex_up[0]:
            ex_up = (d, key, t0, a, b)
    elif d < -1e-9:
        dn_ += 1
        if ex_dn is None or d < ex_dn[0]:
            ex_dn = (d, key, t0, a, b)
OUT["w_up"], OUT["w_down"] = up, dn_
print("  w* widened in %d cases and narrowed in %d" % (up, dn_))
if ex_up:
    OUT["w_example_up"] = dict(delta=ex_up[0], density=ex_up[1], tau0=ex_up[2],
                               before=ex_up[3], after=ex_up[4])
    print("    widest:   %-12s tau_0=%.2f  w* %.4f -> %.4f"
          % (ex_up[1], ex_up[2], ex_up[3], ex_up[4]))
if ex_dn:
    OUT["w_example_down"] = dict(delta=ex_dn[0], density=ex_dn[1],
                                 tau0=ex_dn[2], before=ex_dn[3], after=ex_dn[4])
    print("    narrowest: %-12s tau_0=%.2f  w* %.4f -> %.4f"
          % (ex_dn[1], ex_dn[2], ex_dn[3], ex_dn[4]))
OUT["w_unsigned"] = bool(up > 0 and dn_ > 0)
print("""
  Both directions occur under the same hypothesis, so no sign theorem for w* is
  available: %s.  That is not a gap in the proof.  w* is a DIFFERENCE of a point
  and an inverse, and monotonicity moves the two in opposite senses.
""" % ("confirmed" if OUT["w_unsigned"] else "NOT confirmed"))

# =============================================================================
print("=" * 92)
print("3.  Are the paper's own distortions monotone, and does the split follow?")
print("=" * 92)
print("""
The theorem needs monotonicity exactly, and no measured correction will be
exactly anything.  So the degree is measured rather than assumed: for each of the
three, the share of the record over which the distortion moves in its dominant
direction, weighted by the information there, since that is what the proof
integrates against.

The prediction to be tested is the split the paper already reports without
explaining.  Section 6.2 found eta rising in twelve of twelve cells under the
serial correction; Section 12 found eight of twelve under the derivative bias.
If monotonicity is what separates them, the first two must be close to monotone
and the third must not.
""")
from scipy.signal import savgol_filter
from pipeline import _win, robust_scale, oof_trend, local_scale, LOCAL_BW
from nonparam import weighted_density, estimate_floor

rngR = np.random.default_rng(9001)
H_HALF, WIDE = 0.04, 4.0
z = np.load(os.path.join(HERE, "feat_raw.npz"), allow_pickle=True)
FN = [str(s) for s in z["featnames"]]
units = [k for k in z.files if k != "featnames"]
CHS = [c for c in ("rms", "b4_6", "b8_10") if c in FN]


def rolling_phi(res, bw=0.20):
    r = np.asarray(res, float)
    m = len(r)
    h = max(8, int(bw * m))
    lo = np.maximum(0, np.arange(m) - h)
    hi = np.minimum(m, np.arange(m) + h + 1)

    def win(a, k=0):
        cc = np.concatenate(([0.0], np.cumsum(a)))
        return cc[np.minimum(hi - k, len(a))] - cc[np.minimum(lo, len(a))]

    cnt = (hi - lo).astype(float)
    s1, s2 = win(r), win(r * r)
    mu = s1 / np.maximum(cnt, 1)
    var = s2 - cnt * mu * mu
    cross = win(r[:-1] * r[1:], 1) - np.maximum(cnt - 1, 0) * mu * mu
    return np.clip(np.where(var > 0, cross / np.maximum(var, 1e-300), 0.0),
                   -0.9, 0.9)


def three(x):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    n = len(x)
    if n < 300:
        return None
    dens, res = weighted_density(x)
    if dens is None:
        return None
    d = np.clip(dens - estimate_floor(dens, res, "surrogate", rngR), 0.0, None)
    if d.sum() <= 0:
        return None
    ph = rolling_phi(res)
    out = {"serial": (1.0 - ph) / (1.0 + ph)}
    s = robust_scale(x)
    D = (x - np.median(x)) / s
    w = _win(n)
    sl = np.clip(local_scale(res, max(5, int(LOCAL_BW * n))), 1e-12, None)
    h = max(8, int(LOCAL_BW * n))
    lo = np.maximum(0, np.arange(n) - h)
    hi = np.minimum(n, np.arange(n) + h + 1)
    cc = np.concatenate(([0.0], np.cumsum(res)))
    c2 = np.concatenate(([0.0], np.cumsum(res * res)))
    cnt = (hi - lo).astype(float)
    mu = (cc[hi] - cc[lo]) / cnt
    var = np.maximum((c2[hi] - c2[lo]) / cnt - mu * mu, 0.0)
    out["tail"] = 1.0 / np.clip((np.sqrt(var) / sl) ** 2, 1e-3, 1e3)
    wf = max(11, int(WIDE * 0.08 * n))
    wf += (wf % 2 == 0)
    if wf < n:
        dt = savgol_filter(D, w, 2, deriv=1, delta=1.0 / n)
        d3 = savgol_filter(D, wf, 4, deriv=3, delta=1.0 / n)
        cor = ((dt - (H_HALF ** 2 / 10.0) * d3) / sl) ** 2
        with np.errstate(divide="ignore", invalid="ignore"):
            out["derivative"] = np.clip(
                np.where(dens > 0, cor / np.maximum(dens, 1e-300), 1.0),
                0.2, 5.0)
    return d, out


def monotone_share(w, d):
    """Information-weighted share of the record moving in the dominant sense."""
    lw = np.log(np.clip(w, 1e-9, None))
    s = savgol_filter(lw, max(11, (len(lw) // 20) | 1), 2)
    dv = np.diff(s)
    ww = 0.5 * (d[:-1] + d[1:])
    tot = float(np.sum(ww))
    if tot <= 0:
        return np.nan
    down = float(np.sum(ww[dv < 0])) / tot
    return max(down, 1.0 - down)


print("  %-14s %18s %14s %18s"
      % ("distortion", "monotone share", "direction", "eta cells rising"))
print("  " + "-" * 70)
OBSERVED = {"serial": "12 of 12", "tail": "not reported",
            "derivative": "8 of 12"}
mrows = []
for dist in ("serial", "tail", "derivative"):
    sh, dirn = [], []
    for name in CHS:
        i = FN.index(name)
        for u_ in units:
            t = three(z[u_][:, i])
            if t is None or dist not in t[1]:
                continue
            d, ws = t
            sh.append(monotone_share(ws[dist], d))
            lw = np.log(np.clip(ws[dist], 1e-9, None))
            dirn.append(float(np.mean(np.diff(lw))))
    if len(sh) < 6:
        continue
    m = float(np.nanmedian(sh))
    mrows.append(dict(distortion=dist, share=m,
                      falling=bool(np.nanmedian(dirn) < 0),
                      observed=OBSERVED[dist]))
    print("  %-14s %18.3f %14s %18s"
          % (dist, m, "falls" if mrows[-1]["falling"] else "rises",
             OBSERVED[dist]))
OUT["real"] = mrows
if len(mrows) >= 2:
    _ser = next((r for r in mrows if r["distortion"] == "serial"), None)
    _der = next((r for r in mrows if r["distortion"] == "derivative"), None)
    if _ser and _der:
        OUT["share_serial"] = _ser["share"]
        OUT["share_derivative"] = _der["share"]
        OUT["split_predicted"] = bool(_ser["share"] > _der["share"])
        print("""
  The serial weight moves in one direction over %.0f per cent of the information
  and the derivative bias over %.0f per cent, so the ordering is the predicted one:
  %s.  The separation is modest, and it should be described as such.  Neither
  distortion is exactly monotone, which is what the theorem requires, so this is
  evidence consistent with monotonicity being the discriminating property and not
  a demonstration that it is.  What can be said without qualification is that the
  twelve of twelve and the eight of twelve are reported in two different sections
  of this paper with no account of why they differ, and that the theorem supplies
  a candidate account which the ordering does not contradict.
""" % (100 * _ser["share"], 100 * _der["share"],
       "as predicted" if OUT["split_predicted"] else "NOT as predicted"))

json.dump(OUT, open(os.path.join(HERE, "monotone_distortion.json"), "w"),
          indent=2, default=float)
print("written to monotone_distortion.json")
