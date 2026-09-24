"""
Does the transfer mechanism replicate in the turbofan domain?

Section 9.4 offers a mechanism for why a fixed nonlinear statistic transfers
between units while a fitted linear rule does not: weights that read the LEVEL of
a profile are shared between units, weights that read its DERIVATIVE are not.  It
was measured on sixteen PRONOSTIA bearings and nowhere else, which makes it the
newest and least replicated claim in the paper.

The turbofan fleets give a second domain, and one that shares none of the first's
structure: heterogeneous sensors instead of spectral bands, no frequency axis,
and a deviation profile built from healthy-period standardisation rather than
from power.  If the mechanism is about level against derivative it should hold
there too; if it was a property of spectra it should not.

The two weight vectors are the same functionals as before.  For a profile p over
sensors, entropy carries weights -(1 + log p), which depend on where the profile
sits.  A rule fitted to how the profile is moving carries p'/p.  Their transfer
is measured as the median absolute cosine between units, against the chance level
for the dimension, and no information curve is needed for either -- this is a
statement about the weights, not about the ages they produce.
"""
import os
import json
import numpy as np
from scipy.signal import savgol_filter

from pipeline import _win

HERE = os.path.dirname(os.path.abspath(__file__))
SEED = 20260828
BASE = 0.20
MIN_N = 100
EPS = 1e-30
REF_AGE = 0.60


def cosine(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if not np.isfinite(na) or not np.isfinite(nb) or na <= 0 or nb <= 0:
        return np.nan
    return float(abs(a @ b) / (na * nb))


def deviation_profile(S, base=BASE):
    S = np.asarray(S, float)
    n, K = S.shape
    h = S[:max(5, int(base * n))]
    mu = np.median(h, axis=0)
    sd = np.median(np.abs(h - mu), axis=0) * 1.4826
    keep = np.isfinite(sd) & (sd > 0)
    if keep.sum() < 8:
        return None, None
    return np.abs((S[:, keep] - mu[keep]) / sd[keep]), keep


def weights(A):
    """Level weights (entropy) and derivative weights (p'/p) at REF_AGE."""
    n, K = A.shape
    P = A / np.clip(A.sum(axis=1, keepdims=True), EPS, None)
    w = _win(n)
    if w < 7 or w >= n:
        return None, None
    prof = np.column_stack([savgol_filter(P[:, j], w, 2) for j in range(K)])
    dpro = np.column_stack([savgol_filter(P[:, j], w, 2, deriv=1,
                                          delta=1.0 / n) for j in range(K)])
    k = min(n - 1, max(0, int(REF_AGE * n) - 1))
    lvl = -(1.0 + np.log(np.clip(prof[k], 1e-12, None)))
    der = dpro[k] / np.clip(prof[k], 1e-9, None)
    if not (np.isfinite(lvl).all() and np.isfinite(der).all()):
        return None, None
    return lvl, der


rng = np.random.default_rng(SEED)
print("transfer of the two weight vectors, turbofan\n")
print(f"{'fleet':<9}{'units':>7}{'sensors':>9}{'level weights':>15}"
      f"{'derivative weights':>20}{'chance':>9}")
print("-" * 69)
out = []
for fn, tag in (("cmapss_FD001.npz", "FD001"), ("cmapss_FD004.npz", "FD004")):
    p = os.path.join(HERE, fn)
    if not os.path.exists(p):
        continue
    z = np.load(p, allow_pickle=True)
    # Which sensors survive standardisation differs by unit -- several C-MAPSS
    # sensors are constant over a unit's healthy period and have no scale.  The
    # common set is therefore the INTERSECTION over units.  Fixing it from the
    # first unit instead keeps whichever group that unit belongs to and discards
    # the rest: on FD001 that kept 9 units and threw away 91.
    keeps, raw = [], []
    for u in sorted({k.split("__")[0] for k in z.files}):
        key = f"{u}__sensors"
        if key not in z.files:
            continue
        S = np.asarray(z[key], float)
        if S.shape[0] < MIN_N:
            continue
        _, keep = deviation_profile(S)
        if keep is None:
            continue
        keeps.append(keep)
        raw.append(S)
    if not keeps:
        continue
    common = np.logical_and.reduce(keeps)
    if common.sum() < 8:
        continue
    L, D = [], []
    for S in raw:
        n = S.shape[0]
        h = S[:max(5, int(BASE * n))]
        mu = np.median(h, axis=0)
        sd = np.median(np.abs(h - mu), axis=0) * 1.4826
        A = np.abs((S[:, common] - mu[common]) / sd[common])
        lvl, der = weights(A)
        if lvl is None:
            continue
        L.append(lvl)
        D.append(der)
    K = int(common.sum())
    dropped = len(raw) - len(L)
    if len(L) < 5:
        continue
    cl = [cosine(L[i], L[j]) for i in range(len(L)) for j in range(i + 1, len(L))]
    cd = [cosine(D[i], D[j]) for i in range(len(D)) for j in range(i + 1, len(D))]
    cl = [c for c in cl if np.isfinite(c)]
    cd = [c for c in cd if np.isfinite(c)]
    chance = float(np.sqrt(2.0 / (np.pi * K)))
    out.append(dict(fleet=tag, units=len(L), sensors=K,
                    dropped=dropped, offered=len(raw),
                    level=float(np.median(cl)), derivative=float(np.median(cd)),
                    level_q1=float(np.percentile(cl, 25)),
                    deriv_q3=float(np.percentile(cd, 75)),
                    chance=chance, pairs=len(cl)))
    print(f"{tag:<9}{len(L):>7}{K:>9}{np.median(cl):>15.2f}"
          f"{np.median(cd):>20.2f}{chance:>9.2f}")
print("-" * 69)
print("Entries are median absolute cosines between units; chance is the")
print("expected value for two arbitrary directions in that many dimensions.\n")

if not out:
    print("no fleet had enough usable units")
else:
    ok = all(r["level"] > 3 * r["chance"] for r in out)
    weak = all(r["derivative"] < 3 * r["chance"] for r in out)
    for r in out:
        print(f"{r['fleet']}: level {r['level']:.2f} against chance "
              f"{r['chance']:.2f} ({r['level'] / r['chance']:.1f}x), "
              f"derivative {r['derivative']:.2f} ({r['derivative'] / r['chance']:.1f}x)")
    print()
    if ok and weak:
        print("The mechanism replicates.  Weights that read where the profile")
        print("sits are shared between units; weights that read how it is moving")
        print("are not, in a domain with no spectrum, no frequency axis and a")
        print("profile built from standardisation rather than power.  The")
        print("distinction is therefore about level against derivative and not")
        print("about spectra, which is what Section 9.4 claimed on one rig.")
    elif ok:
        print("Level weights transfer, but derivative weights transfer better")
        print("than chance here, so the contrast is weaker in this domain than")
        print("on the bearings and the mechanism replicates only in part.")
    else:
        print("Level weights do NOT transfer appreciably better than chance in")
        print("this domain, so the mechanism does not replicate and the claim")
        print("in Section 9.4 should be confined to the bearing rigs.")

json.dump(dict(ref_age=REF_AGE, fleets=out),
          open(os.path.join(HERE, "mechanism_turbofan.json"), "w"), indent=2,
          default=float)
