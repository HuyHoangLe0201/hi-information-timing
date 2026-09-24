"""
An exact bound over the whole linear span, and a check on the claim it supports.

linear_vs_shape.py compares distributional statistics against the leading
generalised eigenvector of (M, Sigma).  That comparison has a hole.  The
eigenvector maximises TOTAL information, while what is scored is the age at which
the NORMALISED curve reaches a budget fraction -- a property of the curve's shape.
The two objectives are not merely different, they can be opposed: a direction
carrying enormous late information reaches its own fraction late.  Losing to a
nonlinear statistic under a mismatched objective would show only that the
objective was mismatched.

Worse for the claim, the spectral centroid sum_j p_j f_j is LINEAR in the bands.
If the centroid is among the winners, the claim that the gain lies outside the
linear span is simply false.

Both problems are settled by optimising the right objective, which turns out to
be a Rayleigh quotient as well.  For a fixed linear indicator h = a^T x, write
M(c) = sum_{tau_k <= c} d_k d_k^T.  Then

    G_a(c) / G_a(1)  =  a^T M(c) a / a^T M(1) a,

the noise covariance cancelling entirely, so the fraction of its own budget that a
linear indicator has accumulated by age c is a generalised Rayleigh quotient and
its maximum over EVERY linear combination is the largest generalised eigenvalue
lambda_max(M(c), M(1)).  Since M(1) - M(c) is positive semi-definite the
eigenvalues lie in [0, 1] and lambda_max is non-decreasing in c, so

    tau_lin(q) := min { c : lambda_max(M(c), M(1)) >= q }

is the earliest age at which ANY linear combination of the channels can be usable
at budget fraction q.  This is not a search: it is an exact characterisation of
the whole span, and no choice of weights can beat it.

The bound is computed here without noise-floor subtraction, which is the
conservative direction: leaving the floor in adds spurious density roughly evenly
along the record, flattens the normalised curve and moves the age EARLIER, so the
linear side is being flattered.  A statistic that beats a flattered bound is
outside the span for certain.  The shape statistics are scored both ways.
"""
import os
import json
import numpy as np
from scipy.linalg import eigh
from scipy.signal import savgol_filter

from pipeline import robust_scale, oof_trend, local_scale, _win, LOCAL_BW
from nonparam import info_curve, quantile

HERE = os.path.dirname(os.path.abspath(__file__))
QSTAR = 0.35
COND_MAX = 1e3
EPS = 1e-30
SEED = 20260828
GRID = np.round(np.arange(0.01, 1.001, 0.01), 3)


def norm_rows(P):
    return P / np.clip(P.sum(axis=1, keepdims=True), EPS, None)


SHAPE = [
    ("spectral entropy", False,
     lambda P, f: -(norm_rows(P) * np.log(np.clip(norm_rows(P), EPS, None))
                    ).sum(axis=1)),
    ("spectral centroid", True,          # linear in the bands, by construction
     lambda P, f: (norm_rows(P) * f).sum(axis=1)),
    ("spectral spread", False,
     lambda P, f: np.sqrt((norm_rows(P)
                           * (f - (norm_rows(P) * f).sum(axis=1, keepdims=True))
                           ** 2).sum(axis=1))),
    ("high/low ratio", False,
     lambda P, f: P[:, -8:].sum(axis=1)
     / np.clip(P[:, :8].sum(axis=1), EPS, None)),
]


def prep(x):
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
    sl = np.clip(local_scale(res, max(5, int(LOCAL_BW * n))), 1e-12, None)
    return dict(dtrend=dt, res=res, sig_loc=sl, n=n)


def unfloored_age(p, q=QSTAR):
    """Age at which the raw (floor-retained) normalised curve reaches q."""
    d = (p["dtrend"] / p["sig_loc"]) ** 2
    c = np.cumsum(d)
    if c[-1] <= 0:
        return np.nan
    k = int(np.searchsorted(c / c[-1], q))
    return (k + 1.0) / len(c) if k < len(c) else np.nan


def cond_of(R):
    C = np.nan_to_num(np.corrcoef(R, rowvar=False), nan=0.0)
    return float(np.linalg.cond(C))


def guard(R):
    keep = list(range(R.shape[1]))
    while len(keep) > 1 and cond_of(R[:, keep]) > COND_MAX:
        keep.remove(min(keep,
                        key=lambda j: cond_of(R[:, [k for k in keep if k != j]])))
    return keep


def tau_lin(D, q=QSTAR):
    """Earliest age at which SOME linear combination reaches fraction q.

    D holds the per-channel weighted sensitivities, so that M(c) = D[:m].T D[:m]
    and the generalised eigenproblem (M(c), M(1)) needs no covariance.
    """
    M1 = D.T @ D
    # a small ridge keeps M1 invertible when channels are near-dependent; it can
    # only lower lambda_max, so it never makes the bound look better than it is
    M1 = M1 + 1e-10 * np.trace(M1) / len(M1) * np.eye(len(M1))
    n = len(D)
    for c in GRID:
        m = max(2, int(round(c * n)))
        Mc = D[:m].T @ D[:m]
        lam = float(eigh(Mc, M1, eigvals_only=True)[-1])
        if lam >= q:
            return float(c), lam
    return np.nan, np.nan


z = np.load(os.path.join(HERE, "fineband.npz"))
f = z["centres"] / 1000.0
BK = sorted(k for k in z.files if k != "centres")
rng = np.random.default_rng(SEED)

print(f"{len(BK)} bearings, {len(f)} bands, budget fraction q = {QSTAR}\n")
print("tau_lin is the earliest age reachable by ANY linear combination of the")
print("bands; no weighting can beat it.  The shape statistics are scored with")
print("the paper's floor subtraction and without it.\n")
print(f"{'bearing':<13}{'K':>4}{'tau_lin':>9}{'best shape':>12}"
      f"{'which':>19}{'linear?':>9}{'beats':>7}")
print("-" * 73)
rows, beats, tested = [], 0, 0
for b in BK:
    P = z[b].astype(float)
    ch = [prep(P[:, j]) for j in range(P.shape[1])]
    ok = [j for j, c in enumerate(ch) if c is not None]
    if len(ok) < 4:
        continue
    R = np.column_stack([ch[j]["res"] for j in ok])
    idx = [ok[j] for j in guard(R)]
    D = np.column_stack([ch[j]["dtrend"] / ch[j]["sig_loc"] for j in idx])
    tl, lam = tau_lin(D)
    best, bname, blin = np.inf, "--", None
    per = {}
    for nm, is_lin, fn in SHAPE:
        p = prep(fn(P, f))
        if p is None:
            continue
        try:
            F = info_curve(fn(P, f), rng=rng)
            t_fl = quantile(F, QSTAR) if F is not None else np.nan
        except Exception:
            t_fl = np.nan
        t_raw = unfloored_age(p)
        per[nm] = dict(linear=is_lin, floored=float(t_fl), unfloored=float(t_raw))
        if np.isfinite(t_raw) and t_raw < best:
            best, bname, blin = t_raw, nm, is_lin
    if not np.isfinite(tl) or not np.isfinite(best):
        continue
    tested += 1
    win = best < tl - 1e-9
    beats += int(win)
    rows.append(dict(unit=b, K=len(idx), tau_lin=tl, lam=lam,
                     best_shape=float(best), which=bname, which_linear=blin,
                     beats=bool(win), per=per))
    print(f"{b:<13}{len(idx):>4}{tl:>9.3f}{best:>12.3f}{bname:>19}"
          f"{('yes' if blin else 'no'):>9}{('yes' if win else 'no'):>7}")
print("-" * 73)

tl_all = np.array([r["tau_lin"] for r in rows], float)
bs_all = np.array([r["best_shape"] for r in rows], float)
print(f"\nmedian tau_lin over the whole linear span : {np.median(tl_all):.3f}")
print(f"median best distributional statistic      : {np.median(bs_all):.3f}")
print(f"beaten on {beats} of {tested} bearings, by a median "
      f"{np.median(tl_all - bs_all):+.3f} of a lifetime")

lin_win = sum(1 for r in rows if r["beats"] and r["which_linear"])
print()
print(f"of the {beats} wins, {lin_win} were won by the spectral centroid, which is")
print("linear in the bands and therefore CANNOT beat the bound; any such case")
print("would be an error in the bound rather than a finding.")
print()
if beats and lin_win == 0:
    print("Every win is by a genuinely nonlinear statistic -- entropy, spread or")
    print("a band ratio.  Since the bound is exact over the linear span and was")
    print("computed in the direction that flatters it, those statistics lie")
    print("outside what any reweighting of the spectrum can reach.")
elif lin_win:
    print("A linear statistic appears to beat a bound that covers it, so the")
    print("bound is wrong; the claim cannot be made until that is resolved.")
else:
    print("No distributional statistic beats the bound, so the Section 9 claim")
    print("does not survive: the gain is reachable within the linear span and")
    print("the earlier comparison measured a mismatched objective instead.")

# where the centroid sits relative to the bound, as a consistency check
cent = [(r["per"]["spectral centroid"]["unfloored"], r["tau_lin"])
        for r in rows if "spectral centroid" in r["per"]
        and np.isfinite(r["per"]["spectral centroid"]["unfloored"])]
viol = sum(1 for c, t in cent if c < t - 1e-9)
print()
print(f"consistency: the centroid is linear, so it must satisfy centroid >= "
      f"tau_lin;")
print(f"violations {viol} of {len(cent)}   "
      f"(median centroid {np.median([c for c, _ in cent]):.3f} against bound "
      f"{np.median([t for _, t in cent]):.3f})")

json.dump(dict(bearings=tested, qstar=QSTAR, per_unit=rows,
               tau_lin_median=float(np.median(tl_all)),
               best_shape_median=float(np.median(bs_all)),
               beaten=beats, gap=float(np.median(tl_all - bs_all)),
               wins_by_linear_statistic=lin_win,
               centroid_violations=viol, centroid_n=len(cent)),
          open(os.path.join(HERE, "linear_bound.json"), "w"), indent=2,
          default=float)
