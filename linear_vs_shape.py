"""
Is the distribution finding reachable by linear fusion?

Section 8 reports that indicators describing how vibration energy is
DISTRIBUTED become usable well before indicators describing how much there is.
optimal_indicator.py then shows that the best fixed LINEAR combination of the
standard channels buys 0.003 of a lifetime over root-mean-square amplitude, and
that anchoring the objective early trades feasibility for earliness rather than
improving on it.  Those two results are only consistent if the distribution
finding lies outside what linear recombination can reach.

That is testable, and the test has to be fair in one specific way: both sides
must be given the SAME raw material.  The distributional statistics of Section 8
are nonlinear functions of a 32-band spectrum, so the linear side is given the
same 32 bands and allowed the best fixed combination of them -- the leading
generalised eigenvector of (M, Sigma), by the same construction and the same
leave-one-bearing-out transfer.  If a nonlinear statistic of the bands is earlier
than every linear combination of those bands, the gain is a property of the
shape of the statistic and not of the information the bands contain.

Three protections against flattering the linear side, all of which would
otherwise favour it:

  the rank guard, because band fractions sum to one and the eigenproblem would
  otherwise hand the leading direction to a noiseless direction that carries no
  measurement;

  the Wishart inflation m/(m-K-1), which for K = 32 is severe on the shorter
  records, so bearings with fewer than 5K samples after guarding are dropped and
  counted;

  and the in-sample direction is reported alongside the transferred one, so the
  linear side is shown at its most optimistic as well as at its honest value.
"""
import os
import json
import numpy as np
from scipy.linalg import eigh
from scipy.signal import savgol_filter

from pipeline import robust_scale, oof_trend, local_scale, _win, LOCAL_BW
from fusion_prep import tau_min_from

HERE = os.path.dirname(os.path.abspath(__file__))
QSTAR = 0.35
COND_MAX = 1e3
RIDGE = 1e-6
EPS = 1e-30
MIN_SAMPLES_PER_CHANNEL = 5


def norm_rows(P):
    return P / np.clip(P.sum(axis=1, keepdims=True), EPS, None)


SHAPE = [
    ("spectral entropy",
     lambda P, f: -(norm_rows(P) * np.log(np.clip(norm_rows(P), EPS, None))
                    ).sum(axis=1)),
    ("spectral centroid", lambda P, f: (norm_rows(P) * f).sum(axis=1)),
    ("spectral spread",
     lambda P, f: np.sqrt((norm_rows(P)
                           * (f - (norm_rows(P) * f).sum(axis=1, keepdims=True))
                           ** 2).sum(axis=1))),
    ("high/low ratio",
     lambda P, f: P[:, -8:].sum(axis=1)
     / np.clip(P[:, :8].sum(axis=1), EPS, None)),
]
AMOUNT = [
    ("total power", lambda P, f: P.sum(axis=1)),
    ("peak band power", lambda P, f: P.max(axis=1)),
    ("high-band power", lambda P, f: P[:, -8:].sum(axis=1)),
]


def prep(x):
    """One channel -> derivative, residual, robust scale, as elsewhere."""
    x = np.asarray(x, float)
    n = len(x)
    if n < 60 or not np.all(np.isfinite(x)):
        return None
    s = robust_scale(x)
    if not np.isfinite(s) or s <= 0:
        s = float(np.std(x))
    if not np.isfinite(s) or s <= 0:
        return None
    D = (x - np.median(x)) / s
    w = _win(n)
    if w < 7:
        return None
    dt = savgol_filter(D, w, 2, deriv=1, delta=1.0 / n)
    res = D - oof_trend(D, w // 2)
    sig = robust_scale(res)
    if not np.isfinite(sig) or sig <= 0:
        return None
    return dict(dtrend=dt, res=res, sig=sig,
                sig_loc=np.clip(local_scale(res, max(5, int(LOCAL_BW * n))),
                                1e-12, None))


def single_curve(p):
    return (p["dtrend"] / p["sig_loc"]) ** 2


def cond_of(R):
    C = np.nan_to_num(np.corrcoef(R, rowvar=False), nan=0.0)
    return float(np.linalg.cond(C))


def guard(P):
    """Drop columns greedily until the residual correlation conditions."""
    keep = list(range(P.shape[1]))
    while len(keep) > 1 and cond_of(P[:, keep]) > COND_MAX:
        keep.remove(min(keep,
                        key=lambda j: cond_of(P[:, [k for k in keep if k != j]])))
    return keep


z = np.load(os.path.join(HERE, "fineband.npz"))
f = z["centres"] / 1000.0
BK = sorted(k for k in z.files if k != "centres")
print(f"{len(BK)} bearings, {len(f)} spectral bands\n")

units, dropped_short, keep_idx = [], 0, {}
for b in BK:
    P = z[b].astype(float)
    chans = [prep(P[:, j]) for j in range(P.shape[1])]
    ok = [j for j, c in enumerate(chans) if c is not None]
    if len(ok) < 4:
        continue
    R = np.column_stack([chans[j]["res"] for j in ok])
    keep_local = guard(R)
    idx = [ok[j] for j in keep_local]
    m = len(chans[idx[0]]["dtrend"])
    if m < MIN_SAMPLES_PER_CHANNEL * len(idx):
        dropped_short += 1
        continue
    keep_idx[b] = idx
    units.append((b, P, [chans[j] for j in idx], len(idx), m))
print(f"{len(units)} bearings usable; {dropped_short} dropped for having fewer "
      f"than {MIN_SAMPLES_PER_CHANNEL} samples per retained band\n")


def moments(cs):
    D = np.column_stack([c["dtrend"] for c in cs])
    R = np.column_stack([c["res"] for c in cs])
    s = np.array([c["sig"] for c in cs], float)
    K = len(cs)
    C = np.nan_to_num(np.corrcoef(R, rowvar=False), nan=0.0)
    C = (1 - RIDGE) * C + RIDGE * np.eye(K)
    return D.T @ D, np.outer(s, s) * C, D


def leading(M, S, D, m, K):
    w, V = eigh(M, S)
    a = np.asarray(V[:, -1], float)
    if float((D @ a).sum()) < 0:
        a = -a
    return a / max(float(np.linalg.norm(a)), 1e-300)


def lin_curve(a, D, S, m, K):
    infl = max((m - K - 1) / m, 1e-6)          # Wishart correction
    return infl * (D @ a) ** 2 / max(float(a @ S @ a), 1e-300)


# a common band subset so directions can be averaged across bearings
common = sorted(set.intersection(*[
    set(range(P.shape[1])) for _, P, _, _, _ in units]))
dirs = {}
for b, P, cs, K, m in units:
    M, S, D = moments(cs)
    dirs[b] = (leading(M, S, D, m, K), M, S, D, K, m)

# Section 8 quotes ages at each indicator's OWN budget fraction, not at a
# common absolute demand, because an absolute demand set from one indicator is
# usually unreachable by another and the comparison would measure the choice of
# reference rather than the indicators.  The same convention is used here, and
# the cleanest way to apply it is to treat the linear combination as what it is
# -- another indicator.  Its weights are formed from the standardised bands, the
# combined series is built, and it then goes through the identical pipeline as
# every other candidate, noise floor and all.
from nonparam import info_curve, quantile

SEED = 20260828
rng = np.random.default_rng(SEED)


def standardise(P, idx):
    Z = []
    for j in idx:
        x = np.asarray(P[:, j], float)
        s = robust_scale(x)
        if not np.isfinite(s) or s <= 0:
            s = float(np.std(x))
        Z.append((x - np.median(x)) / max(s, 1e-300))
    return np.column_stack(Z)


def age_of(series):
    try:
        F = info_curve(np.asarray(series, float), rng=rng)
    except Exception:
        return np.nan
    return quantile(F, QSTAR) if F is not None else np.nan


print("earliest usable age, each candidate at its own budget fraction "
      f"q = {QSTAR}\n")
print(f"{'bearing':<14}{'linear, own':>13}{'linear, transf':>16}"
      f"{'best shape':>13}{'best amount':>13}")
print("-" * 69)
res = []
for b, P, cs, K, m in units:
    a, M, S, D, K, m = dirs[b]
    idx = keep_idx[b]
    Z = standardise(P, idx)
    t_own = age_of(Z @ a)
    same = [dirs[v][0] for v, _, _, _, _ in units
            if v != b and len(dirs[v][0]) == len(a)]
    t_tr = np.nan
    if same:
        o = np.mean(same, axis=0)
        o = o / max(float(np.linalg.norm(o)), 1e-300)
        t_tr = age_of(Z @ o)
    sh = {nm: age_of(fn(P, f)) for nm, fn in SHAPE}
    am = {nm: age_of(fn(P, f)) for nm, fn in AMOUNT}
    fin = lambda d: [v for v in d.values() if np.isfinite(v)]
    bs = min(fin(sh)) if fin(sh) else np.nan
    ba = min(fin(am)) if fin(am) else np.nan
    res.append(dict(unit=b, K=K, m=m, lin_own=t_own, lin_tr=t_tr,
                    best_shape=bs, best_amount=ba,
                    shape={k: float(v) for k, v in sh.items()},
                    amount={k: float(v) for k, v in am.items()}))
    g = lambda x: f"{x:.3f}" if np.isfinite(x) else "--"
    print(f"{b:<14}{g(t_own):>13}{g(t_tr):>16}{g(bs):>13}{g(ba):>13}")
print("-" * 69)


def m_(key):
    v = np.array([r[key] for r in res], float)
    v = v[np.isfinite(v)]
    return float(np.median(v)) if len(v) else np.nan


print(f"\nmedian earliest usable age over {len(res)} bearings")
print(f"  best linear combination, fitted on the bearing itself : "
      f"{m_('lin_own'):.3f}")
print(f"  best linear combination, transferred                  : "
      f"{m_('lin_tr'):.3f}")
print(f"  best distributional statistic of the same bands       : "
      f"{m_('best_shape'):.3f}")
print(f"  best amount statistic of the same bands               : "
      f"{m_('best_amount'):.3f}")

# paired, which is what decides it
pair = [(r["best_shape"], r["lin_own"]) for r in res
        if np.isfinite(r["best_shape"]) and np.isfinite(r["lin_own"])]
win = sum(1 for s, l in pair if s < l - 1e-9)
gap = float(np.median([l - s for s, l in pair])) if pair else np.nan
print()
print(f"paired against the linear side at its most optimistic (own fit):")
print(f"  the best distributional statistic is earlier on {win} of {len(pair)}")
print(f"  bearings, by a median {gap:+.3f} of a lifetime")
print()
# --- fairness check: give the linear side the anchored objective too ---------
# The eigenvector above maximises information over the whole record, and
# optimal_indicator.py shows that objective is the wrong one for earliness.
# Losing to a shape statistic under the wrong objective would prove little, so
# the linear side is given the anchored objective as well, fitted in sample, and
# with the anchor chosen per bearing in hindsight -- every advantage the
# construction can be given.
print("\nfairness check: the linear side under its best objective\n")
ANCHORS = (0.10, 0.20, 0.30, 0.50)
best_anchored = {}
for b, P, cs, K, m in units:
    Z = standardise(P, keep_idx[b])
    D = np.column_stack([c["dtrend"] for c in cs])
    _, S, _ = moments(cs)
    ts = []
    for c in ANCHORS:
        mm = max(3, int(c * len(D)))
        w, V = eigh(D[:mm].T @ D[:mm], S)
        a = np.asarray(V[:, -1], float)
        if float((D[:mm] @ a).sum()) < 0:
            a = -a
        a = a / max(float(np.linalg.norm(a)), 1e-300)
        t = age_of(Z @ a)
        if np.isfinite(t):
            ts.append(t)
    best_anchored[b] = min(ts) if ts else np.nan
for r in res:
    r["lin_anchored"] = float(best_anchored.get(r["unit"], np.nan))

pair2 = [(r["best_shape"], r["lin_anchored"]) for r in res
         if np.isfinite(r["best_shape"]) and np.isfinite(r["lin_anchored"])]
win2 = sum(1 for s, l in pair2 if s < l - 1e-9)
gap2 = float(np.median([l - s for s, l in pair2])) if pair2 else np.nan
print(f"  best anchored linear combination, anchor chosen per bearing: "
      f"{np.nanmedian(list(best_anchored.values())):.3f}")
print(f"  best distributional statistic:                               "
      f"{m_('best_shape'):.3f}")
print(f"  shape earlier on {win2} of {len(pair2)} bearings, median "
      f"{gap2:+.3f} of a lifetime")
print()

if win >= 0.7 * len(pair) and gap > 0.05 and win2 >= 0.7 * len(pair2):
    print("A nonlinear statistic of the bands is earlier than the best linear")
    print("combination of those same bands under either objective, even when the")
    print("linear direction is fitted on the record it is scored on and its")
    print("anchor chosen per bearing in hindsight.  The distribution finding is")
    print("therefore not a fusion result in disguise: it is unreachable by")
    print("reweighting the bands, however the weights are chosen.")
else:
    print("The distributional statistics do not clearly beat the best linear")
    print("combination of the same bands, so the Section 8 gain cannot be")
    print("claimed to lie outside the linear span on this evidence.")

json.dump(dict(bearings=len(res), dropped_short=dropped_short,
               per_unit=res,
               median=dict(lin_own=m_("lin_own"), lin_tr=m_("lin_tr"),
                           lin_anchored=float(np.nanmedian(
                               list(best_anchored.values()))),
                           best_shape=m_("best_shape"),
                           best_amount=m_("best_amount")),
               paired=dict(n=len(pair), shape_wins=win, gap=gap),
               paired_anchored=dict(n=len(pair2), shape_wins=win2, gap=gap2)),
          open(os.path.join(HERE, "linear_vs_shape.json"), "w"), indent=2,
          default=float)
