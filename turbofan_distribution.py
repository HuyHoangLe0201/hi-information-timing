"""
Does distribution-before-amount survive a change of domain?

The applied result of this paper is measured on two bearing rigs carrying the
same kind of signal.  That leaves open whether it is a property of vibration
spectra or of how indicators are built, and the two have very different reach.
Turbofan units carry twenty-one heterogeneous sensors and answer the question
in a domain with nothing spectral in it.

The transplant has to be done carefully rather than literally.  The bearing test
normalised band powers into a distribution over frequency, and neither step
survives here: the sensors are temperatures, pressures, speeds and ratios in
different units, so they do not normalise into a probability distribution, and
their index carries no ordering, so a centroid or a spread over it would be an
artefact of how the columns happen to be arranged.  Two statistics from the
bearing set are therefore NOT transplanted, and saying which is part of the
result.

What does transplant is the question itself.  Standardise each sensor on the
unit's own healthy head, take absolute deviations as a non-negative profile
across sensors, and ask whether WHICH sensors deviate becomes usable before HOW
MUCH they deviate.  Only order-invariant functionals are admitted on the
distribution side -- entropy, Gini, top-five share, participation ratio -- and
the amount side is the total, maximum, mean and log-total deviation.

One caveat is inherited.  Section 5 finds the budget hard to estimate below
about a hundred samples per record, and these units run to a few hundred cycles,
so this domain sits nearer that limit than the bearings do.
"""
import os
import json
import time
import numpy as np

from nonparam import info_curve, quantile

HERE = os.path.dirname(os.path.abspath(__file__))
SEED = 20260828
QSTAR = 0.35
BASE = 0.20                 # healthy head used for standardisation
MIN_N = 100                 # Section 5's own limit on record length
EPS = 1e-30


def prof_norm(A):
    return A / np.clip(A.sum(axis=1, keepdims=True), EPS, None)


MEASURES = [
    # order-invariant only: the sensor index has no meaning
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


def deviation_profile(S, base=BASE):
    """|z| of each sensor against the unit's own healthy head."""
    S = np.asarray(S, float)
    n, K = S.shape
    h = S[:max(5, int(base * n))]
    mu = np.median(h, axis=0)
    sd = np.median(np.abs(h - mu), axis=0) * 1.4826
    keep = np.isfinite(sd) & (sd > 0)
    if keep.sum() < 8:
        return None
    return np.abs((S[:, keep] - mu[keep]) / sd[keep])


rng = np.random.default_rng(SEED)
RES = {}
for fn, tag in (("cmapss_FD001.npz", "FD001"), ("cmapss_FD004.npz", "FD004")):
    p = os.path.join(HERE, fn)
    if not os.path.exists(p):
        continue
    z = np.load(p, allow_pickle=True)
    units = sorted({k.split("__")[0] for k in z.files})
    per, skipped, t0 = [], 0, time.time()
    for i, u in enumerate(units):
        key = f"{u}__sensors"
        if key not in z.files:
            continue
        S = np.asarray(z[key], float)
        if S.shape[0] < MIN_N:
            skipped += 1
            continue
        A = deviation_profile(S)
        if A is None:
            skipped += 1
            continue
        row = {"unit": u, "n": int(S.shape[0])}
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
    RES[tag] = dict(per=per, skipped=skipped)
    print(f"{tag}: {len(per)} units used, {skipped} skipped for length "
          f"({time.time() - t0:.0f}s)", flush=True)
print()

print(f"{'fleet':<9}{'units':>7}{'median life':>13}{'distribution':>14}"
      f"{'amount':>9}{'gap':>8}")
print("-" * 60)
out = []
for tag, d in RES.items():
    rows = d["per"]
    if len(rows) < 5:
        continue

    def med(nm):
        v = [r[nm] for r in rows if r.get(nm) is not None]
        return float(np.median(v)) if v else np.nan

    dd = [v for v in (med(n) for n in DIST) if np.isfinite(v)]
    aa = [v for v in (med(n) for n in AMT) if np.isfinite(v)]
    if not dd or not aa:
        continue
    g = float(np.median(aa) - np.median(dd))
    # a bootstrap over units, as on the bearings
    B, boot = 2000, []
    for _ in range(B):
        idx = rng.integers(0, len(rows), size=len(rows))
        sub = [rows[i] for i in idx]

        def med2(nm):
            v = [r[nm] for r in sub if r.get(nm) is not None]
            return float(np.median(v)) if v else np.nan

        d2 = [v for v in (med2(n) for n in DIST) if np.isfinite(v)]
        a2 = [v for v in (med2(n) for n in AMT) if np.isfinite(v)]
        if d2 and a2:
            boot.append(np.median(a2) - np.median(d2))
    lo, hi = np.percentile(boot, [2.5, 97.5]) if boot else (np.nan, np.nan)
    out.append(dict(fleet=tag, units=len(rows),
                    life=float(np.median([r["n"] for r in rows])),
                    dist=float(np.median(dd)), amount=float(np.median(aa)),
                    gap=g, lo=float(lo), hi=float(hi),
                    positive=bool(lo > 0)))
    print(f"{tag:<9}{len(rows):>7}{np.median([r['n'] for r in rows]):>13.0f}"
          f"{np.median(dd):>14.3f}{np.median(aa):>9.3f}{g:>8.3f}")
print("-" * 60)
for r in out:
    print(f"  {r['fleet']}: 95% interval [{r['lo']:+.3f}, {r['hi']:+.3f}], "
          f"excludes zero: {'yes' if r['positive'] else 'NO'}")
print()

if out and all(r["gap"] > 0 for r in out) and all(r["positive"] for r in out):
    print("Distribution before amount holds in a domain with no spectrum in it,")
    print("on heterogeneous sensors, with only order-invariant functionals")
    print("admitted.  The principle is therefore about how an indicator is")
    print("built and not about vibration.")
elif out and all(r["gap"] > 0 for r in out):
    print("The gap is positive on both fleets but at least one interval")
    print("includes zero, so the direction transplants and the evidence for it")
    print("in this domain is weaker than on the bearings.")
elif out:
    bad = [r["fleet"] for r in out if r["gap"] <= 0]
    print(f"The gap is not positive on {', '.join(bad)}.  The principle does NOT")
    print("transplant to this domain, and that bounds the claim: it is a")
    print("statement about spectra, or about these bearings, and not a general")
    print("one about indicator construction.")
else:
    print("Too few usable units to decide.")

json.dump(dict(qstar=QSTAR, min_n=MIN_N, measures_dropped=
               ["spectral centroid", "spectral spread"],
               fleets=out,
               per_unit={k: v["per"] for k, v in RES.items()},
               skipped={k: v["skipped"] for k, v in RES.items()}),
          open(os.path.join(HERE, "turbofan_distribution.json"), "w"), indent=2,
          default=float)
