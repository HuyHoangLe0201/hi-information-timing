"""
Figure 4 -- constructing an indicator, and the limit of linear methods.

(a) the claim of Section 9.3, paired. For each bearing, the best distributional
    statistic of a 32-band spectrum against the best fixed LINEAR combination of
    those same bands. The linear side is fitted on the record it is scored on,
    so the comparison is generous to it, and it still loses on 15 of 16.
    The panel title names the comparator rather than a limit: the span bound of
    Proposition 5 does reach earlier, on the record it is fitted to, and a title
    saying reweighting cannot reach the statistic contradicted Section 10.2.
(b) Proposition 6 made visual. The cumulative leverage H and the leading
    eigenvalue of the Gram matrix, against age, as medians over bearings. For a
    real record the two coincide over the early decade -- the bound is attained
    -- so the age at which lambda_max reaches q is the age at which H does. For
    the same record time-shuffled they separate, because the leverage is then
    spread over many directions instead of one.
(c) why the bound is an oracle. The direction attaining it on one bearing,
    applied to the other fifteen.

Panels (b) and (c) recompute their curves rather than reading summaries, since
the summaries hold only the crossing points.
"""
import os
import json
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter

from figstyle import COL2, FS, C_POW, C_EXP, C_ACC, C_LIN, tidy
from pipeline import robust_scale, oof_trend, local_scale, _win, LOCAL_BW

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(os.path.dirname(HERE), "paper", "fig4.pdf")
L = lambda f: json.load(open(os.path.join(HERE, f)))
QSTAR, COND_MAX, SEED = 0.35, 1e3, 20260828

LS = L("linear_vs_shape.json")
SQ = L("span_qr.json")

fig, axes = plt.subplots(1, 3, figsize=(COL2, 2.45))

# ---- (a) shape against the best linear combination -------------------------
ax = axes[0]
pairs = [(r["lin_own"], r["best_shape"]) for r in LS["per_unit"]
         if np.isfinite(r["lin_own"]) and np.isfinite(r["best_shape"])]
xl = np.array([p[0] for p in pairs])
ys = np.array([p[1] for p in pairs])
lo, hi = 5e-3, 1.4
# above the diagonal the linear side is the earlier one
ax.fill_between([lo, hi], [lo, hi], [hi, hi], color=C_POW, alpha=0.07, lw=0)
ax.plot([lo, hi], [lo, hi], "-", lw=0.8, color=C_LIN, zorder=1)
ax.plot(xl, ys, "o", ms=4.0, color=C_ACC, mew=0, zorder=3)
ax.set_xscale("log"); ax.set_yscale("log")
ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
won = int((ys < xl).sum())
ax.text(0.0075, 0.75, "linear\nearlier", fontsize=FS["annot"], color=C_POW,
        ha="left", va="top")
ax.text(0.85, 0.012, rf"shape earlier" "\n" rf"on ${won}$ of ${len(pairs)}$",
        fontsize=FS["annot"], color=C_ACC, ha="right", va="bottom")
tidy(ax, r"best linear combination, $\tau$",
     r"best distributional statistic, $\tau$",
     r"(a) against the best fitted combination")

# ---- (b) the leverage bound, attained --------------------------------------
ax = axes[1]


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
    return dict(dtrend=savgol_filter(D, w, 2, deriv=1, delta=1.0 / n),
                res=D - oof_trend(D, w // 2),
                sl=np.clip(local_scale(D - oof_trend(D, w // 2),
                                       max(5, int(LOCAL_BW * n))), 1e-12, None))


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
    Q, R = np.linalg.qr(D)
    d = np.abs(np.diag(R))
    return Q[:, d > 1e-10 * max(d.max(), 1e-300)]


z = np.load(os.path.join(HERE, "fineband.npz"))
rng = np.random.default_rng(SEED)
mats = {}
for b in sorted(k for k in z.files if k != "centres"):
    P = z[b].astype(float)
    ch = [prep(P[:, j]) for j in range(P.shape[1])]
    ok = [j for j, c in enumerate(ch) if c is not None]
    if len(ok) < 8:
        continue
    R = np.column_stack([ch[j]["res"] for j in ok])
    idx = [ok[j] for j in guard(R)]
    mats[b] = np.column_stack([ch[j]["dtrend"] / ch[j]["sl"] for j in idx])

AGES = np.geomspace(2e-3, 1.0, 34)


def curves(D):
    Q = basis(D)
    n = len(Q)
    h, lam = [], []
    for c in AGES:
        m = max(2, int(round(c * n)))
        h.append(float((Q[:m] ** 2).sum()))
        lam.append(float(np.linalg.eigvalsh(Q[:m].T @ Q[:m])[-1]))
    return np.array(h), np.array(lam)


real_h, real_l, shuf_h, shuf_l = [], [], [], []
for b, D in mats.items():
    a, c = curves(D)
    real_h.append(a); real_l.append(c)
    a, c = curves(D[rng.permutation(len(D))])
    shuf_h.append(a); shuf_l.append(c)
med = lambda A: np.median(np.array(A), axis=0)

ax.plot(AGES, med(real_h), "-", lw=1.3, color=C_EXP, label=r"$H$, record")
ax.plot(AGES, med(real_l), "--", lw=1.3, color=C_ACC,
        label=r"$\lambda_{\max}$, record")
ax.plot(AGES, med(shuf_h), "-", lw=1.0, color=C_LIN, label=r"$H$, shuffled")
ax.plot(AGES, med(shuf_l), ":", lw=1.3, color=C_POW,
        label=r"$\lambda_{\max}$, shuffled")
ax.axhline(QSTAR, lw=0.7, color="#666666", zorder=1)
ax.set_xscale("log"); ax.set_yscale("log")
ax.set_ylim(0.02, 4.5)
ax.set_xlim(AGES[0], 1.0)


def crossing(y):
    """Age at which a median curve first reaches q, on the log grid."""
    k = int(np.searchsorted(y, QSTAR))
    if k == 0 or k >= len(y):
        return np.nan
    lx = np.log(AGES)
    t = (np.log(QSTAR) - np.log(y[k - 1])) / (np.log(y[k]) - np.log(y[k - 1]))
    return float(np.exp(lx[k - 1] + t * (lx[k] - lx[k - 1])))


for y, col in ((med(real_l), C_ACC), (med(shuf_l), C_POW)):
    xc = crossing(y)
    if np.isfinite(xc):
        ax.plot([xc, xc], [0.02, QSTAR], "-", lw=0.7, color=col, alpha=0.8,
                zorder=1)
        ax.plot([xc], [QSTAR], "o", ms=3.0, color=col, mew=0, zorder=4)
xr, xs = crossing(med(real_l)), crossing(med(shuf_l))
ax.text(xr * 0.85, 0.024, r"$\tau_{\mathrm{lin}}$", fontsize=FS["annot"],
        color=C_ACC, ha="right", va="bottom")
ax.text(xs * 1.15, 0.024, "shuffled", fontsize=FS["annot"], color=C_POW,
        ha="left", va="bottom")
ax.text(0.95, QSTAR * 1.12, rf"$q={QSTAR}$", fontsize=FS["annot"],
        color="#666666", ha="right", va="bottom")
ax.legend(fontsize=FS["legend"], loc="upper left", borderaxespad=0.25,
          handletextpad=0.35, labelspacing=0.22)
tidy(ax, r"age $\tau$", r"information admitted by the span",
     r"(b) the bound is the leverage quantile")

# ---- (c) the attaining direction, moved ------------------------------------
ax = axes[2]
T = SQ["transfer"]
own = np.array([t["own"] for t in T])
out = np.array([t["held_out"] for t in T])
for a, b_ in zip(own, out):
    ax.plot([0, 1], [a, b_], "-", lw=0.7, color=C_LIN, alpha=0.7, zorder=1)
ax.plot(np.zeros_like(own), own, "o", ms=4.0, color=C_ACC, mew=0, zorder=3)
ax.plot(np.ones_like(out), out, "o", ms=4.0, color=C_POW, mew=0, zorder=3)
ax.set_yscale("log")
ax.set_xlim(-0.35, 1.35)
ax.set_xticks([0, 1])
ax.set_xticklabels(["fitted on\nthe record", "moved to\nanother"],
                   fontsize=FS["tick"])
ax.set_ylim(3e-3, 1.6)
f = SQ["transfer_heldout"] / SQ["transfer_own"]
ax.text(1.28, np.sqrt(SQ["transfer_own"] * SQ["transfer_heldout"]),
        rf"$\times {f:.0f}$", fontsize=FS["annot"] + 0.6, color="#666666",
        ha="right", va="center")
tidy(ax, None, r"relative information age $\tau_q$",
     r"(c) the direction is the record's own")

fig.tight_layout(pad=0.35, w_pad=1.1)
fig.savefig(OUT)
print(f"wrote {OUT}")
print(f"(a) shape earlier on {won} of {len(pairs)}")
print(f"(b) at tau=0.01 medians: H {np.interp(0.01, AGES, med(real_h)):.3f} "
      f"lam {np.interp(0.01, AGES, med(real_l)):.3f} | shuffled "
      f"H {np.interp(0.01, AGES, med(shuf_h)):.3f} "
      f"lam {np.interp(0.01, AGES, med(shuf_l)):.3f}")
print(f"(c) own {SQ['transfer_own']:.4f} -> held out "
      f"{SQ['transfer_heldout']:.4f}, factor {f:.0f}")
