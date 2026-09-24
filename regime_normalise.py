"""
Is FD004's weakness the operating regimes?  Fix it and find out.

The turbofan test found the applied result holding on both fleets but far more
weakly on FD004 -- a gap of 0.068 against FD001's 0.397 -- and offered an
explanation: FD004 carries six operating regimes, a unit's sensors step between
them, and those steps put derivative energy into every statistic.  The noise
floor removes noise, not systematic regime switching.

An explanation offered and not tested is a guess.  This one is testable by
removing the thing blamed.  If the regimes are the cause, normalising each sensor
WITHIN its regime should recover a gap of FD001's order; if the gap stays small,
the explanation is wrong and the paper should not offer it.

The regimes are not stored with these data, but they are recoverable: the regime
signature is shared across units and far larger than degradation drift, so
clustering pooled rows finds it where clustering one unit's rows would not.

The experiment carries its own control.  FD001 has a SINGLE operating regime, so
the same normalisation applied to it should change nothing.  If it moves FD001
appreciably, the procedure is doing something other than what it claims and the
FD004 result cannot be read.
"""
import os
import json
import time
import numpy as np
from scipy.cluster.vq import kmeans2

from nonparam import info_curve, quantile

HERE = os.path.dirname(os.path.abspath(__file__))
SEED = 20260828
QSTAR = 0.35
BASE = 0.20
MIN_N = 100
MIN_REGIME_HEAD = 5          # rows of a regime needed in the healthy head
EPS = 1e-30


def prof_norm(A):
    return A / np.clip(A.sum(axis=1, keepdims=True), EPS, None)


MEASURES = [
    ("deviation entropy", "distribution",
     lambda A: -(prof_norm(A) * np.log(np.clip(prof_norm(A), EPS, None))
                 ).sum(axis=1)),
    ("deviation Gini", "distribution",
     lambda A: (2.0 * (np.arange(1, A.shape[1] + 1)
                       * np.sort(prof_norm(A), axis=1)).sum(axis=1)
                - (A.shape[1] + 1.0)) / A.shape[1]),
    ("top-five share", "distribution",
     lambda A: np.sort(prof_norm(A), axis=1)[:, -5:].sum(axis=1)),
    ("participation ratio", "distribution",
     lambda A: 1.0 / np.clip((prof_norm(A) ** 2).sum(axis=1), EPS, None)),
    ("total deviation", "amount", lambda A: A.sum(axis=1)),
    ("max deviation", "amount", lambda A: A.max(axis=1)),
    ("mean deviation", "amount", lambda A: A.mean(axis=1)),
    ("log total deviation", "amount",
     lambda A: np.log(np.clip(A.sum(axis=1), EPS, None))),
]
DIST = [m[0] for m in MEASURES if m[1] == "distribution"]
AMT = [m[0] for m in MEASURES if m[1] == "amount"]


def load(fn):
    z = np.load(os.path.join(HERE, fn), allow_pickle=True)
    out = []
    for u in sorted({k.split("__")[0] for k in z.files}):
        k = f"{u}__sensors"
        if k in z.files:
            S = np.asarray(z[k], float)
            if S.shape[0] >= MIN_N:
                out.append((u, S))
    return out


def find_regimes(units, k, rng):
    """Cluster pooled rows; the regime signature is shared and dominant."""
    X = np.vstack([S for _, S in units])
    mu, sd = X.mean(0), np.clip(X.std(0), 1e-9, None)
    Z = (X - mu) / sd
    idx = rng.choice(len(Z), size=min(20000, len(Z)), replace=False)
    cent, _ = kmeans2(Z[idx], k, minit="++", seed=int(rng.integers(1 << 30)),
                      iter=40)
    return mu, sd, cent


def assign(S, mu, sd, cent):
    Z = (S - mu) / sd
    d = ((Z[:, None, :] - cent[None, :, :]) ** 2).sum(axis=2)
    return np.argmin(d, axis=1)


def deviation_profile(S, lab=None, nreg=1):
    """|z| against the healthy head, computed within regime when given one."""
    n, K = S.shape
    head = slice(0, max(5, int(BASE * n)))
    if lab is None:
        mu = np.median(S[head], axis=0)
        sd = np.median(np.abs(S[head] - mu), axis=0) * 1.4826
        keep = np.isfinite(sd) & (sd > 0)
        if keep.sum() < 8:
            return None
        return np.abs((S[:, keep] - mu[keep]) / sd[keep])
    Zs = np.zeros_like(S)
    gmu = np.median(S[head], axis=0)
    gsd = np.median(np.abs(S[head] - gmu), axis=0) * 1.4826
    for r in range(nreg):
        m = lab == r
        mh = m.copy()
        mh[head.stop:] = False
        if mh.sum() >= MIN_REGIME_HEAD:
            mu = np.median(S[mh], axis=0)
            sd = np.median(np.abs(S[mh] - mu), axis=0) * 1.4826
        else:
            mu, sd = gmu, gsd        # too thin: fall back, and it is counted
        sd = np.where(np.isfinite(sd) & (sd > 0), sd, np.nan)
        Zs[m] = (S[m] - mu) / sd
    keep = np.isfinite(Zs).all(axis=0)
    if keep.sum() < 8:
        return None
    return np.abs(Zs[:, keep])


def gap_for(profiles, rng, B=2000):
    per = []
    for A in profiles:
        row = {}
        for nm, kind, fun in MEASURES:
            x = np.asarray(fun(A), float)
            x = x[np.isfinite(x)]
            v = None
            if len(x) >= MIN_N:
                F = info_curve(x, rng=rng)
                if F is not None:
                    q = quantile(F, QSTAR)
                    if np.isfinite(q):
                        v = float(q)
            row[nm] = v
        per.append(row)

    def stat(rows):
        def med(nm):
            v = [r[nm] for r in rows if r.get(nm) is not None]
            return float(np.median(v)) if v else np.nan
        d = [v for v in (med(n) for n in DIST) if np.isfinite(v)]
        a = [v for v in (med(n) for n in AMT) if np.isfinite(v)]
        return (np.median(a) - np.median(d)) if d and a else np.nan

    g = stat(per)
    boot = [stat([per[i] for i in rng.integers(0, len(per), size=len(per))])
            for _ in range(B)]
    boot = np.array([b for b in boot if np.isfinite(b)], float)
    lo, hi = np.percentile(boot, [2.5, 97.5]) if len(boot) else (np.nan, np.nan)
    return dict(gap=float(g), lo=float(lo), hi=float(hi), units=len(per))


rng = np.random.default_rng(SEED)
print("regime normalisation, with FD001 as the control that must not move\n")
print(f"{'fleet':<8}{'regimes':>9}{'units':>7}{'gap, as before':>17}"
      f"{'gap, normalised':>18}{'change':>9}")
print("-" * 68)
out, t0 = [], time.time()
for fn, tag, k in (("cmapss_FD001.npz", "FD001", 1),
                   ("cmapss_FD004.npz", "FD004", 6)):
    p = os.path.join(HERE, fn)
    if not os.path.exists(p):
        continue
    units = load(fn)
    if not units:
        continue
    plain = [deviation_profile(S) for _, S in units]
    plain = [A for A in plain if A is not None]
    if k > 1:
        mu, sd, cent = find_regimes(units, k, rng)
        labs = [assign(S, mu, sd, cent) for _, S in units]
        sizes = np.bincount(np.concatenate(labs), minlength=k)
        norm = [deviation_profile(S, lab, k)
                for (_, S), lab in zip(units, labs)]
    else:
        sizes = np.array([sum(len(S) for _, S in units)])
        norm = [deviation_profile(S, np.zeros(len(S), int), 1)
                for _, S in units]
    norm = [A for A in norm if A is not None]
    a = gap_for(plain, np.random.default_rng(SEED))
    b = gap_for(norm, np.random.default_rng(SEED))
    out.append(dict(fleet=tag, regimes=k, units=a["units"],
                    before=a["gap"], before_lo=a["lo"], before_hi=a["hi"],
                    after=b["gap"], after_lo=b["lo"], after_hi=b["hi"],
                    regime_sizes=sizes.tolist()))
    print(f"{tag:<8}{k:>9}{a['units']:>7}{a['gap']:>17.3f}{b['gap']:>18.3f}"
          f"{b['gap'] - a['gap']:>+9.3f}")
    print(f"{'':<8}({time.time() - t0:.0f}s)  regime sizes "
          f"{sizes.tolist()}", flush=True)
print("-" * 68)
for r in out:
    print(f"  {r['fleet']}: before [{r['before_lo']:+.3f}, "
          f"{r['before_hi']:+.3f}], after [{r['after_lo']:+.3f}, "
          f"{r['after_hi']:+.3f}]")
print()

ctrl = next((r for r in out if r["fleet"] == "FD001"), None)
test = next((r for r in out if r["fleet"] == "FD004"), None)
if ctrl is None or test is None:
    print("one of the two fleets is missing; no conclusion")
else:
    moved_ctrl = abs(ctrl["after"] - ctrl["before"])
    print(f"control: FD001 has one regime and moves by {moved_ctrl:.3f}")
    if moved_ctrl > 0.05:
        print("The control moves, so the normalisation is doing something other")
        print("than removing regimes and the FD004 result below cannot be read.")
    elif test["after"] > test["before"] + 0.05:
        print(f"FD004 recovers from {test['before']:.3f} to {test['after']:.3f}.")
        print("The regimes were the cause, so the explanation offered for the")
        print("weak turbofan gap is confirmed rather than merely plausible, and")
        print("the remedy is a recipe: normalise within regime before reading")
        print("the curve.")
    else:
        print(f"FD004 moves only from {test['before']:.3f} to "
              f"{test['after']:.3f}, so removing")
        print("the regimes does not recover the gap.  The explanation offered")
        print("for the weak turbofan result is NOT supported and the paper")
        print("should withdraw it and report the weakness unexplained.")

json.dump(dict(qstar=QSTAR, min_n=MIN_N, fleets=out),
          open(os.path.join(HERE, "regime_normalise.json"), "w"), indent=2,
          default=float)
