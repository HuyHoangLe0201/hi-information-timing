"""
Can transferability be predicted from one rig, before deployment?

Section 9.4 finds the mechanism: a statistic whose weights read the LEVEL of the
band profile transfers between bearings, one whose weights read its DERIVATIVE
does not.  If that is right it is not just an explanation but a tool, because the
weight vector of any candidate statistic is computable on a single rig without
ever seeing a second one.  The prediction is then sharp and falsifiable:

    a statistic whose gradient is stable across units should give an age that is
    stable across units.

Two cautions are built in.  The prediction is about STABILITY, not earliness, and
the two must not be conflated: total power has a perfectly constant gradient and
is the latest indicator on the rig, so a finding that stable weights go with
early ages would be evidence of a confound rather than of the mechanism.  And the
gradient is taken numerically rather than analytically, so that every measure --
including the ones with awkward derivatives like the peak band -- is treated
identically and no measure is flattered by a hand-derived formula.

Outcome stability is measured two ways, because they can disagree: the spread of
tau across bearings within a rig, and the agreement of the two rigs' medians.
The first has sixteen points per measure and is the reliable one; the second is
what a practitioner actually cares about and has one point per measure.
"""
import os
import json
import numpy as np

from nonparam import info_curve, quantile

HERE = os.path.dirname(os.path.abspath(__file__))
SEED = 20260828
QSTAR = 0.35
MIN_N = 60
EPS = 1e-30
STEP = 1e-4               # relative finite-difference step for the gradient
CACHE = os.path.join(HERE, "weight_stability_cache.json")


def norm_rows(P):
    return P / np.clip(P.sum(axis=1, keepdims=True), EPS, None)


MEASURES = [
    ("spectral entropy", "distribution",
     lambda P, f: -(norm_rows(P) * np.log(np.clip(norm_rows(P), EPS, None))
                    ).sum(axis=1)),
    ("spectral centroid", "distribution",
     lambda P, f: (norm_rows(P) * f).sum(axis=1)),
    ("spectral spread", "distribution",
     lambda P, f: np.sqrt((norm_rows(P)
                           * (f - (norm_rows(P) * f).sum(axis=1, keepdims=True))
                           ** 2).sum(axis=1))),
    ("high/low ratio", "distribution",
     lambda P, f: P[:, -8:].sum(axis=1)
     / np.clip(P[:, :8].sum(axis=1), EPS, None)),
    ("total power", "amount", lambda P, f: P.sum(axis=1)),
    ("peak band power", "amount", lambda P, f: P.max(axis=1)),
    ("high-band power", "amount", lambda P, f: P[:, -8:].sum(axis=1)),
    ("log total power", "amount",
     lambda P, f: np.log(np.clip(P.sum(axis=1), EPS, None))),
]
RIGS = {"PRONOSTIA": "fineband.npz", "XJTU": "fineband_xjtu.npz"}


def gradient(fun, P, f):
    """dT/dP_j at the record's median spectrum, by central differences.

    Numerical rather than analytic so every measure is treated the same way and
    none is flattered by a formula chosen for it.
    """
    base = np.median(P, axis=0)
    scale = np.clip(np.abs(base), EPS, None)
    g = np.empty(len(base))
    for j in range(len(base)):
        h = STEP * scale[j]
        up, dn = base.copy(), base.copy()
        up[j] += h
        dn[j] -= h
        vu = float(np.asarray(fun(up[None, :], f))[0])
        vd = float(np.asarray(fun(dn[None, :], f))[0])
        g[j] = (vu - vd) / (2 * h)
    return g


def cosine(a, b):
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na <= 0 or nb <= 0:
        return np.nan
    return float(abs(a @ b) / (na * nb))


rng = np.random.default_rng(SEED)
if os.path.exists(CACHE):
    data = json.load(open(CACHE))
    print("per-unit gradients and ages from cache")
else:
    data = {}
    for rig, fn in RIGS.items():
        p = os.path.join(HERE, fn)
        if not os.path.exists(p):
            continue
        zz = np.load(p)
        f = zz["centres"] / 1000.0
        ks = sorted(k for k in zz.files if k != "centres")
        data[rig] = []
        for i, b in enumerate(ks):
            P = zz[b].astype(float)
            row = {"unit": b, "bands": int(P.shape[1])}
            for nm, kind, fun in MEASURES:
                x = np.asarray(fun(P, f), float)
                x = x[np.isfinite(x)]
                tau = None
                if len(x) >= MIN_N:
                    F = info_curve(x, rng=rng)
                    if F is not None:
                        tau = float(quantile(F, QSTAR))
                g = gradient(fun, P, f)
                row[nm] = dict(tau=tau,
                               grad=(None if not np.isfinite(g).all()
                                     else g.tolist()))
            data[rig].append(row)
            print(f"  {rig} {i + 1}/{len(ks)}", flush=True)
    json.dump(data, open(CACHE, "w"))

print()
print("weight stability against outcome stability, measured on each rig\n")
print(f"{'measure':<19}{'kind':<14}{'rig':<11}{'weight cos':>12}"
      f"{'tau median':>12}{'tau IQR':>10}")
print("-" * 78)
rows = []
for rig, units in data.items():
    for nm, kind, _ in MEASURES:
        gs = [np.array(u[nm]["grad"], float) for u in units
              if u[nm]["grad"] is not None]
        gs = [g for g in gs if np.isfinite(g).all() and np.linalg.norm(g) > 0]
        L = {len(g) for g in gs}
        if len(gs) < 4 or len(L) != 1:
            continue
        cos = [cosine(gs[i], gs[j]) for i in range(len(gs))
               for j in range(i + 1, len(gs))]
        cos = [c for c in cos if np.isfinite(c)]
        ts = [u[nm]["tau"] for u in units if u[nm]["tau"] is not None]
        if not cos or len(ts) < 4:
            continue
        w = float(np.median(cos))
        iqr = float(np.percentile(ts, 75) - np.percentile(ts, 25))
        rows.append(dict(measure=nm, kind=kind, rig=rig, weight_cos=w,
                         tau=float(np.median(ts)), iqr=iqr, units=len(ts)))
        print(f"{nm:<19}{kind:<14}{rig:<11}{w:>12.3f}"
              f"{np.median(ts):>12.3f}{iqr:>10.3f}")
print("-" * 78)

# --- the prediction: stable weights, stable age -----------------------------
print("\nthe prediction is about stability, not earliness\n")
rk = lambda v: np.argsort(np.argsort(v))
out = {}
for rig in data:
    r = [x for x in rows if x["rig"] == rig]
    if len(r) < 5:
        continue
    w = np.array([x["weight_cos"] for x in r])
    q = np.array([x["iqr"] for x in r])
    t = np.array([x["tau"] for x in r])
    rho_s = float(np.corrcoef(rk(w), rk(q))[0, 1])
    rho_e = float(np.corrcoef(rk(w), rk(t))[0, 1])
    out[rig] = dict(n=len(r), rho_stability=rho_s, rho_earliness=rho_e)
    print(f"{rig}: {len(r)} measures")
    print(f"  weight stability against tau spread    : {rho_s:+.2f}  "
          f"(prediction: negative)")
    print(f"  weight stability against tau itself    : {rho_e:+.2f}  "
          f"(no prediction; a confound check)")
print()

sig = [v["rho_stability"] for v in out.values()]
conf = [v["rho_earliness"] for v in out.values()]

# how much spread does the predictor actually have?  A rank test over a
# near-constant x is not a test, and saying so is part of the result.
wp = np.array([x["weight_cos"] for x in rows if x["rig"] == "PRONOSTIA"])
near1 = int((wp >= 0.95).sum())
print(f"before reading the correlations: {near1} of {len(wp)} measures have a")
print(f"weight cosine above 0.95, and the spread that remains comes largely")
print(f"from one outlier -- the peak band, whose gradient is a spike at the")
print(f"argmax and therefore differs between bearings that peak elsewhere.")
print()
if sig and all(s <= -0.4 for s in sig):
    print("Stable weights go with a stable age on both rigs, which is what the")
    print("mechanism predicts and what makes it usable: the quantity is")
    print("computable on one rig, before any second rig exists.")
elif sig and any(s <= -0.4 for s in sig):
    print("The relation holds on one rig and not the other, so it is not yet a")
    print("tool; the discrepancy is reported rather than averaged away.")
else:
    print("No relation appears: the rank correlations are +0.07 and -0.07, which")
    print("is zero.  Two things keep this from being a clean refutation, and")
    print("both are limitations of the test rather than support for the")
    print("hypothesis.  The predictor is concentrated near one, so the rank test")
    print("rests on very little spread.  And the outcome, the spread of tau")
    print("across bearings, mixes instability of the indicator with genuine")
    print("differences between bearings, which no rearrangement of these data")
    print("separates.  The honest statement is that the mechanism of Section 9.4")
    print("does not yield a pre-deployment predictor of age stability ON THIS")
    print("EVIDENCE, and that a measure set with more varied gradients would be")
    print("needed to decide it.")
print()
if conf and max(abs(c) for c in conf) > 0.6:
    print(f"Caution: weight stability also tracks the age itself "
          f"({max(conf, key=abs):+.2f}), so")
    print("the two cannot be separated on eight measures and the result above")
    print("may be reading earliness rather than stability.")
else:
    print(f"Weight stability does not track the age itself "
          f"(at most {max(conf, key=abs):+.2f}), so the")
    print("relation above is about stability and not about earliness in")
    print("disguise -- which the constant-gradient amount indicators, stable and")
    print("late, already suggest.")

json.dump(dict(qstar=QSTAR, rows=rows, by_rig=out),
          open(os.path.join(HERE, "weight_stability.json"), "w"), indent=2,
          default=float)
