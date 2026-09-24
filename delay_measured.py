"""
What a loss of information does to the earliest usable age, measured.

The closed forms give the loss exactly for the power-law class and predict that a
large loss produces only a modest delay, because information accumulates as a
steep power of age.  exponent_measure.py shows these records are not power laws
-- the local exponent varies by about a factor four within a record in 99 per
cent of cases -- so a delay computed from a fitted exponent would inherit a model
the data do not obey.

The prediction can be tested without any model.  The delay is a property of the
SHAPE of the measured curve: dividing the information by a factor is the same as
multiplying the demand, so the age moves from G^-1(t) to G^-1(t L), both read off
the curve the record actually produced.  Normalisation cancels in the ratio, so
the normalised curve serves and the base demand is a budget fraction q.

A first attempt used the paper's working fraction q = 0.35 and every row was
empty.  That is not a defect of the experiment but its main finding: a record
holds one unit of budget in total, so a demand of 0.35 cannot survive being
multiplied by ten at all.  Beyond a loss of 1/q the record does not become later,
it becomes infeasible.

A second attempt reported the delay as the ratio of the two ages, and returned
figures near a thousand.  Those are an artefact.  At a demand of 1e-4 the age is
pinned at the first sample, 1/n, so the ratio measures how far the later age sits
above the resolution of the record rather than how much later it is.  Two
corrections are applied: the delay is reported additively, as a fraction of a
lifetime, which is the unit the rest of the paper uses and which does not
degenerate when the base age is small; and curves are used only where the base
age clears the resolution floor by a factor of three, with the excluded count
shown.

The conclusion is the opposite of the one a power-law calculation alone would
suggest, and the paper reports it that way.
"""
import os
import json
import numpy as np

from nonparam import info_curve

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.join(HERE, "curves_cache.npz")
SEED = 20260828
BASES = (1e-4, 1e-3, 1e-2, 0.35)
LOSSES = (10.0, 1e2, 1e3, 1e4)


def load_curves():
    if os.path.exists(CACHE):
        d = np.load(CACHE, allow_pickle=True)
        return [d[k] for k in d.files]
    z = np.load(os.path.join(HERE, "feat_raw.npz"), allow_pickle=True)
    FN = [str(s) for s in z["featnames"]]
    I = {k: i for i, k in enumerate(FN)}
    rng = np.random.default_rng(SEED)
    out = []
    for u in sorted(k for k in z.files if k != "featnames"):
        for nm in FN:
            x = np.asarray(z[u][:, I[nm]], float)
            x = x[np.isfinite(x)]
            if len(x) < 60:
                continue
            try:
                F = info_curve(x, rng=rng)
            except Exception:
                continue
            if F is not None and np.isfinite(F).all() and F[-1] > 0:
                out.append(np.asarray(F, float))
    np.savez_compressed(CACHE, **{f"c{i}": c for i, c in enumerate(out)})
    return out


curves = load_curves()
print(f"{len(curves)} measured curves\n")


def age_at(F, q):
    """First age at which the normalised curve reaches the fraction q."""
    if q > 1.0:
        return np.inf
    k = int(np.searchsorted(F, q))
    return (k + 1.0) / len(F) if k < len(F) else np.inf


print("delay of the earliest usable age when the information is divided by a "
      "factor.\nThe delay is additive, in fractions of a lifetime; curves whose "
      "base age sits\nat the resolution floor are excluded and counted "
      "separately.\n")
print(f"{'base demand':>13}{'loss':>9}{'usable':>11}{'infeasible':>12}"
      f"{'delay: median':>16}{'quartiles':>16}")
print("-" * 78)
rows = []
for q in BASES:
    for L in LOSSES:
        d, gone, floor = [], 0, 0
        for F in curves:
            t0 = age_at(F, q)
            if not np.isfinite(t0):
                continue
            if t0 < 3.0 / len(F):          # base age at the resolution floor
                floor += 1
                continue
            t1 = age_at(F, q * L)
            if not np.isfinite(t1):
                gone += 1
                continue
            d.append(t1 - t0)
        n = len(d) + gone
        if n + floor == 0:
            continue
        rec = dict(base=q, loss=L, usable=len(d), infeasible=gone,
                   at_floor=floor)
        if not d:
            rec.update(median=None, q1=None, q3=None)
            rows.append(rec)
            print(f"{q:>13.0e}{L:>9.0e}{0:>11}{gone:>12}"
                  f"{'none remain feasible':>32}")
            continue
        d = np.array(d)
        q1, q3 = np.percentile(d, [25, 75])
        rec.update(median=float(np.median(d)), q1=float(q1), q3=float(q3))
        rows.append(rec)
        print(f"{q:>13.0e}{L:>9.0e}{len(d):>11}{gone:>12}"
              f"{np.median(d):>+16.3f}{q1:>+11.3f}{q3:>+5.3f}")
    print()
print("-" * 78)
print("A delay is only defined where the reduced demand still fits inside the")
print("record's budget.  The 'infeasible' column counts the curves for which it")
print("does not, and those are the cases that matter: they cannot be met at any")
print("age by any estimator, not merely met later.\n")

# --- the infeasibility threshold -------------------------------------------
print("\nthe loss at which a record stops being usable at all\n")
print(f"{'base demand':>13}{'largest loss it survives':>28}")
print("-" * 43)
thr = []
for q in BASES:
    L = 1.0 / q
    thr.append(dict(base=q, max_loss=L))
    print(f"{q:>13.0e}{L:>26.0f}x")
print("-" * 43)
print("This is exact and needs no data: a record carries one unit of budget, so")
print("a demand of q survives a loss of at most 1/q whatever the curve looks")
print("like.  At the working fraction of 0.35 used throughout the paper, an")
print("estimator losing more than about a threefold factor cannot meet the")
print("demand at any age.")

# --- comparison with the power-law calibration -----------------------------
try:
    em = json.load(open(os.path.join(HERE, "exponent_measure.json")))
    bmed = em["pooled_beta"]
    r = 1.0 / (16 * bmed ** 4)
    pred = (1.0 / r) ** (1.0 / (2 * bmed - 1))
    print(f"\nThe power-law calibration at the pooled measured exponent "
          f"{bmed:.2f} gives a")
    print(f"loss of {1 / r:.0f}-fold.  Read as a ratio of ages that is a delay of")
    print(f"{pred:.2f}x, which invites the reading that a steep curve is forgiving.")
    print("The measurement above shows that reading is wrong, for a reason the")
    print("ratio hides: a delay factor is only meaningful while the reduced")
    print("demand still fits in the budget, and at that loss it does not fit for")
    print(f"any demand above {1 / (1 / r):.0e}.  The oracle-to-practical gap on these")
    print("records is therefore not a delay at all.  It is the difference")
    print("between a demand that can be met and one that cannot.")
except FileNotFoundError:
    pred = None
    print("\nrun exponent_measure.py first for the power-law comparison")

json.dump(dict(curves=len(curves), rows=rows, thresholds=thr,
               powerlaw_pred=(None if pred is None else float(pred))),
          open(os.path.join(HERE, "delay_measured.json"), "w"), indent=2,
          default=float)
