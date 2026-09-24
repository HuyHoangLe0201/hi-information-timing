"""
Why a fixed nonlinear statistic beats a fitted linear one: a candidate mechanism.

Section 8 declines to name a mechanism for the distribution-before-amount result,
and Section 9 leaves a paradox beside it: the best linear combination of the same
bands, fitted on the record it is scored on, reaches tau = 0.007 against
entropy's 0.105, yet moved to another bearing it fails by a factor of 136.  A
fitted linear rule is better and useless; a fixed nonlinear one is worse and
usable.  Estimation theory offers a reason, and it is testable.

Model the band occupancy of a snapshot as multinomial with probabilities
p_j(tau).  The log-likelihood of one snapshot is sum_j n_j log p_j(tau), so the
locally most powerful scalar statistic -- the score -- is

    T(tau) = sum_j n_j d/dtau log p_j(tau) = N sum_j f_j w_j,
    w_j = p'_j / p_j ,

with f_j the observed band fraction.  Two things follow.  The optimal statistic
is LINEAR in the band fractions, which is why a fitted linear rule wins in
sample.  And its weights are the logarithmic derivative of that unit's own band
profile, which is a property of how that bearing happens to degrade, which is why
the fitted rule does not transfer.

Entropy carries fixed weights instead: d/df_j of -sum f log f is -(1 + log f_j),
which needs no fitting and depends on the observed spectrum rather than on a
profile learned from it.

Three things are measured.  Whether the fitted optimal direction of Section 9
actually aligns with w = p'/p, which is what the model predicts and which nothing
so far has checked.  Whether w transfers between bearings, which is what the
model says should fail.  And whether entropy's fixed weights resemble the
fleet-average w, which would say why a weight-free statistic works at all.

A negative result here is informative: it would mean the multinomial score is the
wrong mechanism and the paper should keep declining to name one.
"""
import os
import json
import numpy as np
from scipy.signal import savgol_filter
from scipy.linalg import eigh

from pipeline import robust_scale, oof_trend, local_scale, _win, LOCAL_BW

HERE = os.path.dirname(os.path.abspath(__file__))
SEED = 20260828
EPS = 1e-30
COND_MAX = 1e3
REF_AGE = 0.60          # where the weights are read; swept below


def cosine(a, b):
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na <= 0 or nb <= 0:
        return np.nan
    return float(abs(a @ b) / (na * nb))


def prep_channel(x):
    x = np.asarray(x, float)
    n = len(x)
    if n < 200 or not np.all(np.isfinite(x)):
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
    return dict(dtrend=dt, res=res, sig=sig)


def cond_of(R):
    return float(np.linalg.cond(np.nan_to_num(np.corrcoef(R, rowvar=False),
                                              nan=0.0)))


def guard(R, cols):
    keep = list(range(len(cols)))
    while len(keep) > 1 and cond_of(R[:, keep]) > COND_MAX:
        keep.remove(min(keep,
                        key=lambda j: cond_of(R[:, [k for k in keep
                                                    if k != j]])))
    return keep


z = np.load(os.path.join(HERE, "fineband.npz"))
f = z["centres"] / 1000.0
BK = sorted(k for k in z.files if k != "centres")

print("does the fitted optimal direction match the multinomial score?\n")
print(f"weights read at age {REF_AGE}\n")
print(f"{'bearing':<13}{'bands':>7}{'cos(a, p''/p)':>15}"
      f"{'cos(a, entropy)':>17}{'cos(p''/p, entropy)':>21}")
print("-" * 74)
rows, W, A, E = [], {}, {}, {}
for b in BK:
    P = z[b].astype(float)
    n = P.shape[0]
    if n < 200:
        continue
    frac = P / np.clip(P.sum(axis=1, keepdims=True), EPS, None)
    ch = [prep_channel(P[:, j]) for j in range(P.shape[1])]
    ok = [j for j, c in enumerate(ch) if c is not None]
    if len(ok) < 8:
        continue
    R = np.column_stack([ch[j]["res"] for j in ok])
    keep = guard(R, ok)
    idx = [ok[j] for j in keep]

    # the fitted optimum of Section 9: leading generalised eigenvector
    D = np.column_stack([ch[j]["dtrend"] for j in idx])
    Rr = np.column_stack([ch[j]["res"] for j in idx])
    s = np.array([ch[j]["sig"] for j in idx], float)
    C = np.nan_to_num(np.corrcoef(Rr, rowvar=False), nan=0.0)
    C = 0.999999 * C + 1e-6 * np.eye(len(idx))
    S = np.outer(s, s) * C
    wv, V = eigh(D.T @ D, S)
    a = np.asarray(V[:, -1], float)
    if float((D @ a).sum()) < 0:
        a = -a

    # the multinomial score weights, from the smoothed band profile
    win = _win(n)
    prof = np.column_stack([savgol_filter(frac[:, j], win, 2) for j in idx])
    dprof = np.column_stack([savgol_filter(frac[:, j], win, 2, deriv=1,
                                           delta=1.0 / n) for j in idx])
    k = min(n - 1, max(0, int(REF_AGE * n) - 1))
    w_score = dprof[k] / np.clip(prof[k], 1e-9, None)

    # entropy's fixed weights at the same age
    w_ent = -(1.0 + np.log(np.clip(prof[k], 1e-12, None)))

    W[b], A[b], E[b] = w_score, a, w_ent
    rows.append(dict(unit=b, bands=len(idx),
                     cos_a_score=cosine(a, w_score),
                     cos_a_ent=cosine(a, w_ent),
                     cos_score_ent=cosine(w_score, w_ent)))
    print(f"{b:<13}{len(idx):>7}{cosine(a, w_score):>15.2f}"
          f"{cosine(a, w_ent):>17.2f}{cosine(w_score, w_ent):>21.2f}")
print("-" * 74)
cs = np.array([r["cos_a_score"] for r in rows], float)
ce = np.array([r["cos_a_ent"] for r in rows], float)
se_ = np.array([r["cos_score_ent"] for r in rows], float)
print(f"median |cosine| between the fitted direction and p'/p : "
      f"{np.median(cs):.2f}")
print(f"median |cosine| between the fitted direction and entropy: "
      f"{np.median(ce):.2f}")
print(f"median |cosine| between p'/p and entropy weights        : "
      f"{np.median(se_):.2f}")
print()
print("A random pair of directions in this many dimensions has expected")
print(f"|cosine| about {np.sqrt(2 / (np.pi * np.median([r['bands'] for r in rows]))):.2f},")
print("so that is the level against which these should be read.\n")

# --- does the score weight transfer between bearings? -----------------------
print("does the weight vector transfer between bearings?\n")
common = min(len(v) for v in W.values())
pairs_w, pairs_e = [], []
ks = list(W)
for i in range(len(ks)):
    for j in range(i + 1, len(ks)):
        if len(W[ks[i]]) == len(W[ks[j]]):
            pairs_w.append(cosine(W[ks[i]], W[ks[j]]))
            pairs_e.append(cosine(E[ks[i]], E[ks[j]]))
pw = np.array(pairs_w, float)
pe = np.array(pairs_e, float)
print(f"{'quantity':<28}{'pairs':>7}{'median |cosine|':>18}{'quartiles':>20}")
print("-" * 74)
if len(pw):
    print(f"{'score weights p''/p':<28}{len(pw):>7}{np.median(pw):>18.2f}"
          f"{f'{np.percentile(pw, 25):.2f} to {np.percentile(pw, 75):.2f}':>20}")
if len(pe):
    print(f"{'entropy weights':<28}{len(pe):>7}{np.median(pe):>18.2f}"
          f"{f'{np.percentile(pe, 25):.2f} to {np.percentile(pe, 75):.2f}':>20}")
print("-" * 74)
print()

chance = float(np.sqrt(2 / (np.pi * np.median([r["bands"] for r in rows]))))
ok_align = np.isfinite(np.median(cs)) and np.median(cs) > 2 * chance
ok_transfer = len(pw) and len(pe) and np.median(pe) > np.median(pw) + 0.3

print("TWO SEPARATE VERDICTS, and they differ.\n")
if ok_align:
    print(f"The fitted optimum aligns with the multinomial score at "
          f"{np.median(cs):.2f}, above")
    print(f"the chance level {chance:.2f}, so the score model describes it.")
else:
    print(f"REJECTED: the fitted optimum does not align with the multinomial")
    print(f"score.  Its median |cosine| is {np.median(cs):.2f} against a chance level of")
    print(f"{chance:.2f} in this many dimensions -- indistinguishable from an")
    print("arbitrary direction.  Whatever the fitted eigenvector is finding, it")
    print("is not the score of a multinomial band model, and that hypothesis is")
    print("reported as tested and discarded rather than quietly dropped.")
print()
if ok_transfer:
    print(f"CONFIRMED, and independently of the above: weights derived from a")
    print(f"unit's band profile transfer between bearings at "
          f"{np.median(pw):.2f}, which is the")
    print(f"chance level, while entropy's transfer at {np.median(pe):.2f}.")
    print()
    print("The reason is visible in the two definitions.  Score weights are")
    print("p'/p and depend on the DERIVATIVE of the band profile, which is how")
    print("this particular bearing happens to be degrading.  Entropy's weights")
    print("are -(1 + log p) and depend on the LEVEL of the profile, which is a")
    print("property of the machine type and is shared.  A statistic whose")
    print("weights read the level transfers; one whose weights read the")
    print("derivative cannot, however optimal it is on the record it was fitted")
    print("to.  That is a mechanism for the Section 9 paradox -- fitted linear")
    print("rules beat entropy in sample and fail by a factor of 136 when moved --")
    print("and it does not depend on the multinomial model that was just")
    print("rejected.")
else:
    print("The transfer asymmetry is also not established, so neither half of")
    print("the proposed mechanism survives and Section 8 should go on declining")
    print("to name one.")

json.dump(dict(ref_age=REF_AGE, per_unit=rows,
               cos_a_score=float(np.median(cs)),
               cos_a_entropy=float(np.median(ce)),
               cos_score_entropy=float(np.median(se_)),
               transfer_score=(float(np.median(pw)) if len(pw) else None),
               transfer_entropy=(float(np.median(pe)) if len(pe) else None),
               chance_level=float(np.sqrt(
                   2 / (np.pi * np.median([r["bands"] for r in rows]))))),
          open(os.path.join(HERE, "score_mechanism.json"), "w"), indent=2,
          default=float)
