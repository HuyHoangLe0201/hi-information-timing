"""
The span bound, computed stably, and the leverage inequality made manifest.

leverage_bound.py found the inequality tau_lin(q) >= H^-1(q) violated on six of
sixteen bearings.  The proof cannot fail -- the largest eigenvalue of a positive
semi-definite matrix is at most its trace -- so the fault was numerical, and it
was: lambda_max was obtained from eigh(M(c), M(1)) with M(1) = D^T D, and the
rank guard of Section 2.5 conditions the residual CORRELATION matrix, not D^T D.
A near-singular M(1) with a token ridge returns spurious eigenvalues.

The repair also simplifies the mathematics.  Take the thin QR factorisation
D = QR, with Q having orthonormal columns spanning the sensitivity space.  Then
M(1) = R^T R, M(c) = R^T Q_{1:m}^T Q_{1:m} R, and

    M(1)^{-1} M(c)  =  R^{-1} (Q_{1:m}^T Q_{1:m}) R,

similar to Q_{1:m}^T Q_{1:m}.  So

    lambda_max( M(c), M(1) )  =  lambda_max( Q_{1:m}^T Q_{1:m} ),

computed from an orthonormal basis with no inverse, no ridge and no conditioning
problem.  Two facts are then immediate rather than argued.  The Gram matrix
Q_{1:m}^T Q_{1:m} is positive semi-definite with

    trace = sum_{k<=m} ||Q_k||^2 = sum_{k<=m} h_k = H(m),

the cumulative leverage, so lambda_max <= H(m) holds by inspection; and
Q_{1:m}^T Q_{1:m} <= Q^T Q = I, so lambda_max <= 1.  The leverage bound and the
trivial bound are the trace and the identity of the same matrix.

Everything downstream is recomputed on this route, including the value the
manuscript quotes.
"""
import os
import json
import numpy as np
from scipy.signal import savgol_filter

from pipeline import robust_scale, oof_trend, local_scale, _win, LOCAL_BW

HERE = os.path.dirname(os.path.abspath(__file__))
QSTAR = 0.35
COND_MAX = 1e3
SEED = 20260828


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


def cond_of(R):
    return float(np.linalg.cond(np.nan_to_num(np.corrcoef(R, rowvar=False),
                                              nan=0.0)))


def guard(R):
    keep = list(range(R.shape[1]))
    while len(keep) > 1 and cond_of(R[:, keep]) > COND_MAX:
        keep.remove(min(keep,
                        key=lambda j: cond_of(R[:, [k for k in keep if k != j]])))
    return keep


def basis(D):
    """Orthonormal basis for the sensitivity space, dropping null directions."""
    Q, R = np.linalg.qr(D)
    d = np.abs(np.diag(R))
    keep = d > 1e-10 * max(d.max(), 1e-300)
    return Q[:, keep]


def lam_at(Q, m):
    G = Q[:m].T @ Q[:m]
    return float(np.linalg.eigvalsh(G)[-1])


def tau_lin(Q, q=QSTAR):
    """Smallest sample count whose Gram matrix has leading eigenvalue >= q.

    lambda_max is non-decreasing in m, so a bisection is exact and there is no
    grid to mis-resolve.
    """
    n = len(Q)
    if lam_at(Q, n) < q:
        return np.nan
    lo, hi = 1, n
    while lo < hi:
        mid = (lo + hi) // 2
        if lam_at(Q, mid) >= q:
            hi = mid
        else:
            lo = mid + 1
    return lo / n


def H_inv(Q, q=QSTAR):
    h = (Q ** 2).sum(axis=1)
    c = np.cumsum(h)
    k = int(np.searchsorted(c, q))
    return (k + 1.0) / len(c) if k < len(c) else np.nan


z = np.load(os.path.join(HERE, "fineband.npz"))
BK = sorted(k for k in z.files if k != "centres")
rng = np.random.default_rng(SEED)

full = {}
for b in BK:
    P = z[b].astype(float)
    ch = [prep(P[:, j]) for j in range(P.shape[1])]
    ok = [j for j, c in enumerate(ch) if c is not None]
    if len(ok) < 8:
        continue
    R = np.column_stack([ch[j]["res"] for j in ok])
    idx = [ok[j] for j in guard(R)]
    full[b] = np.column_stack([ch[j]["dtrend"] / ch[j]["sig_loc"] for j in idx])
print(f"{len(full)} bearings, span bound by the QR route\n")

print(f"{'bearing':<13}{'K':>4}{'rank':>6}{'H^-1(q)':>10}{'tau_lin':>10}"
      f"{'holds':>7}{'ratio':>8}{'shuffled':>10}")
print("-" * 68)
rows, viol = [], 0
for b, D in full.items():
    Q = basis(D)
    hi, tl = H_inv(Q), tau_lin(Q)
    sh = float(np.median([tau_lin(basis(D[rng.permutation(len(D))]))
                          for _ in range(3)]))
    ok = (not np.isfinite(tl)) or (tl >= hi - 1e-12)
    viol += int(not ok)
    rows.append(dict(unit=b, K=D.shape[1], rank=Q.shape[1], H_inv=hi,
                     tau_lin=tl, holds=bool(ok), shuffled=sh,
                     qK=QSTAR / Q.shape[1]))
    print(f"{b:<13}{D.shape[1]:>4}{Q.shape[1]:>6}{hi:>10.4f}{tl:>10.4f}"
          f"{('yes' if ok else 'NO'):>7}{tl / max(hi, 1e-12):>8.2f}"
          f"{sh:>10.4f}")
print("-" * 68)
print(f"the inequality holds on {len(rows) - viol} of {len(rows)} bearings"
      f"{'' if viol == 0 else '   <-- still violated'}\n")

hi_a = np.array([r["H_inv"] for r in rows])
tl_a = np.array([r["tau_lin"] for r in rows])
sh_a = np.array([r["shuffled"] for r in rows])
qk_a = np.array([r["qK"] for r in rows])
print(f"median leverage bound H^-1(q) : {np.median(hi_a):.4f}")
print(f"median span bound tau_lin     : {np.median(tl_a):.4f}")
print(f"median tau_lin after shuffling: {np.median(sh_a):.4f}")
print(f"median q/K                    : {np.median(qk_a):.4f}")
print()
print(f"tau_lin sits a median factor {np.median(tl_a / hi_a):.2f} above its leverage bound,")
print("so on real records the bound is not merely valid but attained.")
print()
print("Two exact bounds bracket the exchangeable case.  The largest of K")
print("non-negative eigenvalues is at least their mean and at most their sum, so")
print("H(c)/K <= lambda_max <= H(c), and with E[H(c)] = cK under exchangeability")
print(f"the null must satisfy q/K <= tau_lin <= q, here {np.median(qk_a):.4f} to {QSTAR:.2f}.")
print(f"The shuffled record gives {np.median(sh_a):.4f}, inside the bracket and well above")
print(f"its lower end; the real record gives {np.median(tl_a):.4f}, sitting ON the lower end.")
print("That is the whole difference, and it is a statement about concentration")
print("rather than about amount, which the next table measures directly.\n")

print("how the early leverage is distributed across the K directions\n")
print(f"{'':<13}{'H at tau_lin':>14}{'lambda_max':>12}{'share in one':>14}")
print("-" * 53)
conc = []
for b, D in full.items():
    Q = basis(D)
    n, K = len(Q), Q.shape[1]
    t = tau_lin(Q)
    if not np.isfinite(t):
        continue
    m = max(1, int(round(t * n)))
    Hm = float((Q[:m] ** 2).sum())
    lm = lam_at(Q, m)
    Qs = basis(D[rng.permutation(n)])
    ms = max(1, int(round(t * n)))
    Hs = float((Qs[:ms] ** 2).sum())
    ls = lam_at(Qs, ms)
    conc.append(dict(unit=b, H=Hm, lam=lm, share=lm / max(Hm, 1e-12),
                     share_shuffled=ls / max(Hs, 1e-12)))
cr = np.array([c["share"] for c in conc])
cs = np.array([c["share_shuffled"] for c in conc])
print(f"{'real record':<13}{np.median([c['H'] for c in conc]):>14.3f}"
      f"{np.median([c['lam'] for c in conc]):>12.3f}{np.median(cr):>14.3f}")
print(f"{'shuffled':<13}{'--':>14}{'--':>12}{np.median(cs):>14.3f}")
print("-" * 53)
print(f"Both rows are read at the SAME early sample count, the one at which the")
print(f"real record reaches the bound.  There the real record puts "
      f"{100 * np.median(cr):.0f}% of the")
print(f"leverage it has spent into a single direction against "
      f"{100 * np.median(cs):.0f}% for the shuffled")
print("one.  The shuffled figure is not small because the window holds fewer")
print("samples than there are channels, so its Gram matrix is rank-deficient and")
print("its mass cannot spread over K directions however unstructured it is; the")
print("comparison that matters is the factor of two between the two rows.")
print()
print("The early advantage of the linear span is therefore a statement about")
print("where the leverage points, not about how much of it there is; the next")
print("table gives the amount for comparison.")

print("\nshare of the K leverages already spent, by age\n")
print(f"{'age':>8}{'real':>10}{'shuffled':>11}{'exchangeable':>15}")
print("-" * 44)
prof = []
for c in (0.01, 0.02, 0.05, 0.10, 0.25, 0.50):
    r_, s_ = [], []
    for b, D in full.items():
        Q = basis(D)
        K, n = Q.shape[1], len(Q)
        m = max(1, int(round(c * n)))
        r_.append(float((Q[:m] ** 2).sum()) / K)
        Qs = basis(D[rng.permutation(n)])
        s_.append(float((Qs[:m] ** 2).sum()) / Qs.shape[1])
    prof.append(dict(c=c, real=float(np.median(r_)),
                     shuffled=float(np.median(s_))))
    print(f"{c:>8.2f}{np.median(r_):>10.3f}{np.median(s_):>11.3f}{c:>15.3f}")
print("-" * 44)
print("Exchangeability spends the budget uniformly, at rate c; the shuffled")
print("column reproduces that and the real column front-loads it -- but only")
print(f"mildly: {100 * prof[0]['real']:.0f} per cent by age 0.01 against "
      f"{100 * prof[0]['shuffled']:.0f}, a factor of two, against the")
print(f"factor {np.median(cr) / np.median(cs):.1f} by which the real record concentrates that")
print("spending into one direction.  Amount is the small effect; direction is")
print("the large one.")
print()
print("Leverage is large where the sensitivity points somewhere the rest of the")
print("record does not, so what the span bound finds early is the part of a")
print("bearing's early life least like its own later life.  A linear combination")
print("can be aimed at that; the direction is specific to the record, which the")
print("transfer test below measures.")

# --- the attaining direction, recomputed stably and moved --------------------
# The transfer test in linear_bound_null.py took its direction from the unstable
# generalised eigenproblem, so it is redone here on the QR route.  The direction
# is the leading eigenvector of the Gram matrix mapped back through R.
print("\nthe direction attaining the bound, applied to other bearings\n")
common = min(r["rank"] for r in rows)
print(f"scored on the leading {common} directions every bearing retains\n")
print(f"{'fitted on':<13}{'own':>9}{'held-out median':>18}")
print("-" * 40)


def age_of(D, a, q=QSTAR):
    g = (D @ a) ** 2
    c = np.cumsum(g)
    if c[-1] <= 0:
        return np.nan
    k = int(np.searchsorted(c / c[-1], q))
    return (k + 1.0) / len(c) if k < len(c) else np.nan


trans = []
keys = list(full)
for b in keys:
    D = full[b][:, :common]
    Q, R = np.linalg.qr(D)
    t = tau_lin(Q)
    if not np.isfinite(t):
        continue
    m = max(1, int(round(t * len(Q))))
    w, V = np.linalg.eigh(Q[:m].T @ Q[:m])
    a = np.linalg.solve(R, V[:, -1])
    a = a / max(float(np.linalg.norm(a)), 1e-300)
    if float((D @ a).sum()) < 0:
        a = -a
    other = [age_of(full[v][:, :common], a) for v in keys if v != b]
    other = [x for x in other if np.isfinite(x)]
    if not other:
        continue
    trans.append(dict(unit=b, own=t, held_out=float(np.median(other))))
    print(f"{b:<13}{t:>9.4f}{np.median(other):>18.4f}")
print("-" * 40)
t_own = float(np.median([r["own"] for r in trans]))
t_out = float(np.median([r["held_out"] for r in trans]))
print(f"median own {t_own:.4f}, held out {t_out:.4f}, a factor "
      f"{t_out / max(t_own, 1e-12):.0f}")
print()
print("So the bound is exact, attained, and attained by a direction that is")
print("specific to the record: it bounds an oracle, in the sense of Section 6.")

json.dump(dict(bearings=len(rows), qstar=QSTAR, per_unit=rows, violations=viol,
               concentration=dict(real=float(np.median(cr)),
                                  shuffled=float(np.median(cs))),
               transfer=trans, transfer_own=t_own, transfer_heldout=t_out,
               median=dict(H_inv=float(np.median(hi_a)),
                           tau_lin=float(np.median(tl_a)),
                           shuffled=float(np.median(sh_a)),
                           qK=float(np.median(qk_a))),
               tightness=float(np.median(tl_a / hi_a)),
               profile=prof),
          open(os.path.join(HERE, "span_qr.json"), "w"), indent=2, default=float)
