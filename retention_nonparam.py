"""
Retention, recomputed on the density the theory is actually written in.

The earlier retention numbers were built on the unweighted density D'^2 with no
floor subtracted. Both are now known to be wrong: the Fisher information is
D'^2/sigma^2, and differentiating noise contributes density that no degradation
put there. The second error is the more damaging one here, because the retention
question is decided by the SHAPE of the density -- which samples are worth
keeping -- and a floor spread evenly over the record makes every sample look
partly worth keeping.

That yields a prediction. The earlier analysis found the optimal retention set
broke into nine to eleven disjoint pieces, which was read as bearing damage
arriving in bursts. If those pieces were noise peaks standing above a floor,
removing the floor should join them up. If the fragmentation survives, it is
real structure.

Three quantities per indicator, all at the indicator's own operating age rather
than a fixed one, since a retention policy only matters where the indicator can
support a decision:

  efficiency  -- information in the trailing window over that in the best
                 same-sized subset available;
  pieces      -- how many disjoint intervals that best subset breaks into;
  inflation   -- how much longer a trailing window must be to match it.
"""
import os
import json
import numpy as np
from nonparam import weighted_density, estimate_floor, quantile

HERE = os.path.dirname(os.path.abspath(__file__))
WS = (0.10, 0.20, 0.30)
Q_OPERATING = 0.35          # the target the operating age is defined by
SEED = 2027


def curves(x, rng, subtract):
    dens, res = weighted_density(x)
    if dens is None:
        return None, None
    if subtract:
        fl = estimate_floor(dens, res, "surrogate", rng)
        dens = np.clip(dens - fl, 0.0, None)
    if dens.sum() <= 0:
        return None, None
    return dens, np.cumsum(dens) / dens.sum()


def measure(dens, k0, kw):
    d = dens[:k0 + 1]
    if d.sum() <= 0 or kw < 1:
        return None
    order = np.argsort(d)[::-1][:kw]
    best = float(d[order].sum())
    trail = float(d[max(0, k0 - kw + 1):k0 + 1].sum())
    idx = np.sort(order)
    pieces = 1 + int(np.sum(np.diff(idx) > 1))
    c = np.cumsum(d[::-1])
    j = int(np.searchsorted(c, best))
    infl = (j + 1) / kw if j < len(c) else np.inf
    return dict(eff=(trail / best if best > 0 else np.nan),
                pieces=pieces, infl=infl,
                pos=float(np.median(idx) / (k0 + 1)))


def analyse(label, series, rng):
    rows = {w: {"eff": [], "pieces": [], "infl": [], "pos": []} for w in WS}
    raw = {w: [] for w in WS}
    for x in series:
        x = np.asarray(x, float)
        d_sub, F_sub = curves(x, rng, True)
        d_raw, _ = curves(x, rng, False)
        if d_sub is None or d_raw is None:
            continue
        n = len(d_sub)
        t0 = quantile(F_sub, Q_OPERATING)
        k0 = int(round(t0 * n)) - 1
        for w in WS:
            kw = max(3, int(round(w * n)))
            if k0 < kw + 3 or k0 >= n:
                continue
            m = measure(d_sub, k0, kw)
            r = measure(d_raw, k0, kw)
            if m is None or r is None:
                continue
            for k in ("eff", "pieces", "infl", "pos"):
                rows[w][k].append(m[k])
            raw[w].append(r["eff"])
    out = []
    for w in WS:
        if not rows[w]["eff"]:
            continue
        fin = [v for v in rows[w]["infl"] if np.isfinite(v)]
        out.append(dict(indicator=label, w=w, units=len(rows[w]["eff"]),
                        eff=float(np.median(rows[w]["eff"])),
                        eff_unfloored=float(np.median(raw[w])),
                        pieces=float(np.median(rows[w]["pieces"])),
                        pos=float(np.median(rows[w]["pos"])),
                        infl=float(np.median(fin)) if fin else np.inf))
    return out


rng = np.random.default_rng(SEED)
z = np.load(os.path.join(HERE, "feat_raw.npz"), allow_pickle=True)
FN = [str(s) for s in z["featnames"]]
I = {k: i for i, k in enumerate(FN)}
BK = sorted(k for k in z.files if k != "featnames")
CH = {"rms": ["rms"], "kurt": ["kurt"], "0--2 kHz": ["b0_1", "b1_2"],
      "4--10 kHz": ["b4_6", "b6_8", "b8_10"]}

allrows = []
print("retention at each indicator's own operating age\n")
print(f"{'indicator':<14}{'w':>6}{'units':>6}{'efficiency':>12}"
      f"{'unfloored':>11}{'pieces':>8}{'position':>10}{'inflation':>11}")
print("-" * 78)
for lab, parts in CH.items():
    rs = analyse(lab, [sum(z[b][:, I[p]] for p in parts) for b in BK], rng)
    allrows += rs
    for r in rs:
        print(f"{r['indicator']:<14}{r['w']:>6.2f}{r['units']:>6}"
              f"{r['eff']:>12.3f}{r['eff_unfloored']:>11.3f}"
              f"{r['pieces']:>8.0f}{r['pos']:>10.2f}{r['infl']:>10.1f}x")
    print()

zt = np.load(os.path.join(HERE, "cmapss_FD001.npz"))
un = sorted({k.split("__")[0] for k in zt.files})[:50]
rs = analyse("turbofan s11",
             [zt[f"{u}__sensors"][:, 10].astype(float) for u in un], rng)
allrows += rs
for r in rs:
    print(f"{r['indicator']:<14}{r['w']:>6.2f}{r['units']:>6}"
          f"{r['eff']:>12.3f}{r['eff_unfloored']:>11.3f}"
          f"{r['pieces']:>8.0f}{r['pos']:>10.2f}{r['infl']:>10.1f}x")

print("-" * 78)
p_floor = [r["pieces"] for r in allrows]
print(f"pieces in the optimal set: {min(p_floor):.0f} to {max(p_floor):.0f}")
print("(the earlier analysis, on the unfloored unweighted density, found 9-11)")
print()
e = [r["eff"] for r in allrows]
u = [r["eff_unfloored"] for r in allrows]
print(f"efficiency with the floor removed : {min(e):.2f} to {max(e):.2f}")
print(f"without                           : {min(u):.2f} to {max(u):.2f}")
print()
print("efficiency = information a trailing window keeps, over the best")
print("same-sized subset; position = where that subset sits, as a fraction of")
print("elapsed life; inflation = how much longer the trailing window must be.")
json.dump(allrows, open(os.path.join(HERE, "retention_nonparam.json"), "w"),
          indent=2, default=float)
