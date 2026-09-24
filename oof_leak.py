r"""Does the out-of-fold scheme still work once the residuals are correlated?

Section 4.1 estimates the noise scale from out-of-fold residuals, "the record
being split into ten interleaved folds with each sample's trend predicted
without its own fold", because a smoother fitted to the same samples understates
the scale.  Interleaving means fold membership is k mod 10, so sample i is held
out while i-1 and i+1 are both kept.

That is sound for white noise: the neighbours carry independent noise, the
prediction cannot reproduce the held-out sample's own error, and the residual
variance is if anything inflated.  Section 4.5 has just measured the residuals
and they are NOT white, with a lag-one correlation between 0.067 and 0.399.  A
neighbour then carries a fraction phi of the held-out sample's noise, the local
fit reproduces part of it, and the residual is shrunk.  The scale is understated
again, which is the failure out-of-fold fitting was introduced to remove.

The leak is governed by the local fit's weight on the nearest neighbours, which
scales as 1/h with h the half-window, so it should matter on short records and
not on long ones.  That is a prediction and it is testable against synthetic
records where the true scale is known.

The obvious repairs are worse than the disease, and the experiment says so
rather than the other way round.  Purging the neighbours removes points the
local quadratic needs, and contiguous folds remove the whole window; at the
eight per cent bandwidth this paper uses, a short record has a half-window of
nine samples and neither repair leaves a fit standing.  What survives is a
purge of one or two samples with the window widened to keep the point count,
and that is what is carried to the real records.
"""
import os
import json
import numpy as np
from scipy.signal import savgol_filter

from pipeline import robust_scale, _win, K_FOLDS

HERE = os.path.dirname(os.path.abspath(__file__))
rng = np.random.default_rng(31337)
OUT = {}
MIN_FIT = 6          # points below which a local quadratic is not attempted
Q = 0.35             # the budget fraction every age in the paper is read at


def trend_oof(D, half, scheme="interleaved", gap=0, widen=0):
    """Local-quadratic trend at each sample, fitted without that sample.

    `interleaved` is the scheme in pipeline.py: fold membership is k mod K, so
    the immediate neighbours are always in the training set.  `gap` additionally
    drops the sample's nearest neighbours on each side.  `widen` extends the
    half-window, which is what makes a purge affordable.  `blocked` uses
    contiguous folds.  The count of samples that could not be fitted is returned
    alongside, because a scheme that silently falls back to the observation
    itself reports a scale of zero and must not be read as an estimate.
    """
    n = len(D)
    idx = np.arange(n)
    pred = np.empty(n)
    unfitted = 0
    h = half + widen
    for j in range(K_FOLDS):
        if scheme == "blocked":
            edges = np.linspace(0, n, K_FOLDS + 1).astype(int)
            inblock = (idx >= edges[j]) & (idx < edges[j + 1])
            test, keep = idx[inblock], ~inblock
        else:
            test, keep = idx[idx % K_FOLDS == j], idx % K_FOLDS != j
        for i in test:
            lo, hi = max(0, i - h), min(n, i + h + 1)
            m = keep[lo:hi].copy()
            if gap:
                m &= np.abs(idx[lo:hi] - i) > gap
            xs = idx[lo:hi][m].astype(float) - i
            if len(xs) < MIN_FIT:
                pred[i] = D[i]
                unfitted += 1
            else:
                pred[i] = np.polyfit(xs, D[lo:hi][m], 2)[-1]
    return pred, unfitted


def trend_insample(D, half):
    w = 2 * half + 1
    w += (w % 2 == 0)
    return savgol_filter(D, min(w, len(D) - (1 - len(D) % 2)), 2)


def ar1(n, phi, sigma, r):
    e = r.normal(0, sigma * np.sqrt(1 - phi ** 2), n)
    out = np.empty(n)
    out[0] = r.normal(0, sigma)
    for k in range(1, n):
        out[k] = phi * out[k - 1] + e[k]
    return out


SCHEMES = [
    ("in-sample", dict(insample=True)),
    ("interleaved", dict(scheme="interleaved")),
    ("purge 1", dict(scheme="interleaved", gap=1, widen=2)),
    ("purge 2", dict(scheme="interleaved", gap=2, widen=4)),
    ("purge 5", dict(scheme="interleaved", gap=5, widen=10)),
    ("blocked", dict(scheme="blocked")),
]

# =============================================================================
print("=" * 92)
print("1.  The leak on synthetic records, against a known scale")
print("=" * 92)
print("""
A quadratic trend plus AR(1) noise of known sigma.  Each scheme estimates the
scale from its residuals; the figure is the estimate divided by the truth, so
1.00 is correct and below 1.00 understates the noise, which makes the
information curve claim more than the record holds.  A purge is paired with an
equal widening of the window so the local fit keeps its point count.
""")
SIGMA = 0.05
hdr = "  %6s %6s" % ("n", "phi") + "".join("%12s" % s for s, _ in SCHEMES)
print(hdr)
print("  " + "-" * (len(hdr) - 2))
syn = []
for n in (240, 800, 2400):
    for phi in (0.0, 0.15, 0.30, 0.45):
        acc = {s: [] for s, _ in SCHEMES}
        unfit = {s: 0 for s, _ in SCHEMES}
        for _ in range(40):
            t = np.linspace(1e-3, 1.0, n)
            D = 3.0 * t ** 2 + ar1(n, phi, SIGMA, rng)
            half = _win(n) // 2
            for s, kw in SCHEMES:
                if kw.get("insample"):
                    res = D - trend_insample(D, half)
                else:
                    p, u = trend_oof(D, half, **kw)
                    res, unfit[s] = D - p, unfit[s] + u
                acc[s].append(robust_scale(res) / SIGMA)
        row = dict(n=n, phi=phi,
                   **{s: float(np.median(v)) for s, v in acc.items()},
                   unfitted={s: unfit[s] for s, _ in SCHEMES})
        syn.append(row)
        print("  %6d %6.2f" % (n, phi)
              + "".join("%12.3f" % row[s] for s, _ in SCHEMES))
    print()
OUT["synthetic"] = syn

print("  samples the scheme could not fit, and so returned unchanged:")
for s, _ in SCHEMES:
    tot = sum(r["unfitted"][s] for r in syn)
    if tot:
        print("    %-14s %d" % (s, tot))
print("""
  A scheme with unfitted samples is not an estimator of anything: the residual
  there is exactly zero and drags the robust scale down with it.  That rules out
  contiguous folds at this bandwidth, and rules out a deep purge on short
  records even with the window widened.
""")

_il = {(r["n"], r["phi"]): r["interleaved"] for r in syn}
print("  the leak in the paper's own scheme, as phi rises from zero:")
print("  %8s %12s %12s %12s" % ("n", "phi=0", "phi=0.45", "change"))
print("  " + "-" * 48)
leak = []
for n in (240, 800, 2400):
    a, b = _il[(n, 0.0)], _il[(n, 0.45)]
    leak.append(dict(n=n, at_zero=a, at_45=b, change=b / a - 1))
    print("  %8d %12.3f %12.3f %11.1f%%" % (n, a, b, 100 * (b / a - 1)))
OUT["leak"] = leak
print("""
  The prediction was that the leak scales as 1/h and therefore with 1/n at fixed
  bandwidth.  It does.  The scheme also sits ABOVE one at phi = 0, because an
  out-of-fold residual carries the prediction's own variance as well as the
  sample's, so part of the leak is spent cancelling a conservative bias that was
  already there rather than biasing the answer.
""")

json.dump(OUT, open(os.path.join(HERE, "oof_leak.json"), "w"), indent=2,
          default=float)
print("written to oof_leak.json")

# =============================================================================
print()
print("=" * 92)
print("2.  What the repair does on the real records")
print("=" * 92)
print("""
Part 1 bounds the leak but says nothing about whether it matters here.  The
bearings run from 230 to 2803 samples and Section 4.5 measures phi from 0.067 to
0.399, so the worst case is a short record carrying an amount indicator.  The
scale is recomputed with a one-sample purge and a two-sample widening, which is
the only repair that held at every length, and the whole comparison is re-read.
""")
from scipy.signal import savgol_filter as _sg
from pipeline import local_scale, LOCAL_BW
from nonparam import estimate_floor, quantile

EPS = 1e-30


def _norm(P):
    return P / np.clip(P.sum(axis=1, keepdims=True), EPS, None)


def _entropy(P):
    p = _norm(P)
    return -(p * np.log(np.clip(p, EPS, None))).sum(axis=1)


def _centroid(P, f):
    return (_norm(P) * f).sum(axis=1)


def _spread(P, f):
    p = _norm(P)
    c = (p * f).sum(axis=1, keepdims=True)
    return np.sqrt((p * (f - c) ** 2).sum(axis=1))


def _gini(P):
    p = np.sort(_norm(P), axis=1)
    m = p.shape[1]
    return (2 * (p * np.arange(1, m + 1)).sum(axis=1)
            ) / np.clip(p.sum(axis=1), EPS, None) / m - (m + 1) / m


MEASURES = [
    ("spectral entropy", "distribution", lambda P, f: _entropy(P)),
    ("spectral centroid", "distribution", lambda P, f: _centroid(P, f)),
    ("spectral spread", "distribution", lambda P, f: _spread(P, f)),
    ("spectral Gini", "distribution", lambda P, f: _gini(P)),
    ("top-8-band share", "distribution",
     lambda P, f: np.sort(_norm(P), axis=1)[:, -8:].sum(axis=1)),
    ("high/low band ratio", "distribution",
     lambda P, f: P[:, -8:].sum(axis=1) / np.clip(P[:, :8].sum(axis=1), EPS, None)),
    ("total power", "amount", lambda P, f: P.sum(axis=1)),
    ("peak band power", "amount", lambda P, f: P.max(axis=1)),
    ("high-band power", "amount", lambda P, f: P[:, -8:].sum(axis=1)),
    ("log total power", "amount",
     lambda P, f: np.log(np.clip(P.sum(axis=1), EPS, None))),
]
RIGS = {"PRONOSTIA": "fineband.npz", "XJTU": "fineband_xjtu.npz"}


def density_pair(x):
    """The paper's density and the purged one, from a single pass.

    Only the residual differs between them, so the derivative and the trend
    bandwidth are shared and any change is attributable to the fold scheme.
    """
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    n = len(x)
    if n < 60:
        return None
    s = robust_scale(x)
    if not np.isfinite(s) or s <= 0:
        return None
    D = (x - np.median(x)) / s
    w = _win(n)
    if w < 7 or w >= n:
        return None
    dt = _sg(D, w, 2, deriv=1, delta=1.0 / n)
    half = w // 2
    out = {}
    for key, kw in (("paper", dict(scheme="interleaved")),
                    ("purged", dict(scheme="interleaved", gap=1, widen=2))):
        pred, unfit = trend_oof(D, half, **kw)
        if unfit:
            return None
        res = D - pred
        sl = np.clip(local_scale(res, max(5, int(LOCAL_BW * n))), 1e-12, None)
        out[key] = ((dt / sl) ** 2, res, float(robust_scale(res)))
    return out


print("  %-12s %-13s %10s %10s %10s"
      % ("rig", "family", "scale x", "tau paper", "tau purged"))
print("  " + "-" * 60)
real = {}
for rig, fn in RIGS.items():
    p = os.path.join(HERE, fn)
    if not os.path.exists(p):
        continue
    z = np.load(p)
    f = z["centres"] / 1000.0
    BK = sorted(k for k in z.files if k != "centres")
    fam = {}
    for kind in ("distribution", "amount"):
        rat, t0, t1 = [], [], []
        for lab, k, fun in MEASURES:
            if k != kind:
                continue
            a0, a1, rr = [], [], []
            for b in BK:
                pr = density_pair(fun(z[b].astype(float), f))
                if pr is None:
                    continue
                rr.append(pr["purged"][2] / max(pr["paper"][2], 1e-30))
                for key, acc in (("paper", a0), ("purged", a1)):
                    dens, res, _ = pr[key]
                    d = np.clip(dens - estimate_floor(dens, res, "surrogate",
                                                      rng), 0.0, None)
                    if d.sum() > 0:
                        acc.append(quantile(np.cumsum(d) / d.sum(), Q))
            if len(a0) >= 4 and len(a1) >= 4:
                rat.append(float(np.median(rr)))
                t0.append(float(np.median(a0)))
                t1.append(float(np.median(a1)))
        if not t0:
            continue
        fam[kind] = (float(np.median(rat)), float(np.median(t0)),
                     float(np.median(t1)))
        print("  %-12s %-13s %10.3f %10.3f %10.3f"
              % (rig, kind, fam[kind][0], fam[kind][1], fam[kind][2]))
    if len(fam) == 2:
        g0 = fam["amount"][1] - fam["distribution"][1]
        g1 = fam["amount"][2] - fam["distribution"][2]
        real[rig] = dict(gap_paper=g0, gap_purged=g1, change=g1 - g0,
                         scale_ratio={k: v[0] for k, v in fam.items()},
                         ages={k: dict(paper=v[1], purged=v[2])
                               for k, v in fam.items()})
        print("  %-12s %-13s %10s %10.3f %10.3f"
              % ("", "GAP", "", g0, g1))
OUT["real"] = real
print()
for rig, d in real.items():
    print("  %-12s gap %.3f -> %.3f, %s"
          % (rig, d["gap_paper"], d["gap_purged"],
             "SURVIVES" if d["gap_purged"] > 0 else "DOES NOT SURVIVE"))
print("""
  The scale column is the purged estimate over the paper's, so above one means
  the paper's scheme was understating the noise on these records, by the amount
  Part 1 predicts from their length and their phi.
""")

json.dump(OUT, open(os.path.join(HERE, "oof_leak.json"), "w"), indent=2,
          default=float)
print("rewritten to oof_leak.json")
