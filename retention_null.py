"""
Is the fragmented retention set structure, or the estimator ranking noise?

The optimal retention set breaks into six to eighteen disjoint pieces on real
records, and removing the noise floor does not join them up. That was read
earlier as bearing damage arriving in bursts. It cannot be read that way without
a null, for a mechanical reason: the set is chosen by ranking a per-sample
density, the density is estimated from a differentiated noisy signal, and
ranking anything noisy produces a scattered top-k set even when the quantity
being ranked is perfectly smooth.

Subtracting a flat floor cannot fix this -- subtracting a constant leaves the
ranking unchanged, which is why the pieces count barely moved.

The null is built from each record itself: its own smooth fitted trend, plus its
own residual resampled. The true density is then smooth and monotone by
construction, so its optimal retention set is one contiguous interval and its
efficiency is whatever a trailing window genuinely gives on that shape. Anything
the estimator reports beyond that is measurement.
"""
import os
import json
import numpy as np
from scipy.signal import savgol_filter
from nonparam import weighted_density, estimate_floor, quantile
from pipeline import _win, robust_scale

HERE = os.path.dirname(os.path.abspath(__file__))
WS = (0.10, 0.20, 0.30)
Q_OPERATING = 0.35
NULL_REPS = 6
SEED = 5150


def stats(dens, k0, kw):
    d = dens[:k0 + 1]
    if d.sum() <= 0 or kw < 1 or k0 + 1 <= kw:
        return None
    order = np.argsort(d)[::-1][:kw]
    best = float(d[order].sum())
    trail = float(d[max(0, k0 - kw + 1):k0 + 1].sum())
    idx = np.sort(order)
    return (trail / best if best > 0 else np.nan,
            1 + int(np.sum(np.diff(idx) > 1)))


def prep(x, rng):
    dens, res = weighted_density(x)
    if dens is None:
        return None
    fl = estimate_floor(dens, res, "surrogate", rng)
    d = np.clip(dens - fl, 0.0, None)
    if d.sum() <= 0:
        return None
    return d, np.cumsum(d) / d.sum(), res


def null_for(x, res, rng, reps=NULL_REPS):
    """Same record, same noise, but a trend that is smooth by construction.

    res comes back from weighted_density in normalised units, where the record
    was divided by its own robust scale, so it has to be scaled back before it
    can be added to a trend in raw units. Permuting rather than resampling keeps
    the residual's marginal distribution exactly -- which matters, these tails
    are heavy -- while destroying its order.
    """
    x = np.asarray(x, float)
    n = len(x)
    s = robust_scale(x)
    if not np.isfinite(s) or s <= 0:
        return []
    smooth = savgol_filter(x, _win(n), 2)
    out = []
    for _ in range(reps):
        p = prep(smooth + rng.permutation(res) * s, rng)
        if p is not None:
            out.append(p)
    return out


def analyse(label, series, rng):
    obs = {w: {"eff": [], "pc": []} for w in WS}
    nul = {w: {"eff": [], "pc": []} for w in WS}
    for x in series:
        x = np.asarray(x, float)
        p = prep(x, rng)
        if p is None:
            continue
        d, F, res = p
        n = len(d)
        k0 = int(round(quantile(F, Q_OPERATING) * n)) - 1
        for w in WS:
            kw = max(3, int(round(w * n)))
            s = stats(d, k0, kw)
            if s is None:
                continue
            obs[w]["eff"].append(s[0]); obs[w]["pc"].append(s[1])
        for dn, Fn, _ in null_for(x, res, rng):
            kn = len(dn)
            k0n = int(round(quantile(Fn, Q_OPERATING) * kn)) - 1
            for w in WS:
                kw = max(3, int(round(w * kn)))
                s = stats(dn, k0n, kw)
                if s is None:
                    continue
                nul[w]["eff"].append(s[0]); nul[w]["pc"].append(s[1])
    out = []
    for w in WS:
        if not obs[w]["eff"] or not nul[w]["eff"]:
            continue
        oe = float(np.median(obs[w]["eff"])); ne = float(np.median(nul[w]["eff"]))
        op = float(np.median(obs[w]["pc"])); np_ = float(np.median(nul[w]["pc"]))
        out.append(dict(indicator=label, w=w, units=len(obs[w]["eff"]),
                        eff=oe, eff_null=ne,
                        eff_corrected=min(1.0, oe / ne) if ne > 0 else np.nan,
                        pieces=op, pieces_null=np_))
    return out


rng = np.random.default_rng(SEED)
z = np.load(os.path.join(HERE, "feat_raw.npz"), allow_pickle=True)
FN = [str(s) for s in z["featnames"]]
I = {k: i for i, k in enumerate(FN)}
BK = sorted(k for k in z.files if k != "featnames")
CH = {"rms": ["rms"], "kurt": ["kurt"], "0--2 kHz": ["b0_1", "b1_2"],
      "4--10 kHz": ["b4_6", "b6_8", "b8_10"]}

print("retention against a null with a smooth trend and the same noise\n")
print(f"{'indicator':<14}{'w':>6}{'pieces':>8}{'null':>7}"
      f"{'efficiency':>12}{'null':>8}{'corrected':>11}")
print("-" * 68)
rows = []
for lab, parts in CH.items():
    rs = analyse(lab, [sum(z[b][:, I[p]] for p in parts) for b in BK], rng)
    rows += rs
    for r in rs:
        print(f"{r['indicator']:<14}{r['w']:>6.2f}{r['pieces']:>8.0f}"
              f"{r['pieces_null']:>7.0f}{r['eff']:>12.3f}"
              f"{r['eff_null']:>8.3f}{r['eff_corrected']:>11.3f}")
    print()

zt = np.load(os.path.join(HERE, "cmapss_FD001.npz"))
un = sorted({k.split("__")[0] for k in zt.files})[:40]
rs = analyse("turbofan s11",
             [zt[f"{u}__sensors"][:, 10].astype(float) for u in un], rng)
rows += rs
for r in rs:
    print(f"{r['indicator']:<14}{r['w']:>6.2f}{r['pieces']:>8.0f}"
          f"{r['pieces_null']:>7.0f}{r['eff']:>12.3f}"
          f"{r['eff_null']:>8.3f}{r['eff_corrected']:>11.3f}")

print("-" * 68)
pr = [r["pieces"] / r["pieces_null"] for r in rows if r["pieces_null"] > 0]
print(f"pieces, observed over null: {min(pr):.2f} to {max(pr):.2f}")
print("A ratio near 1 means the fragmentation is what ranking a noisy density")
print("produces on a perfectly smooth trend, and carries no information about")
print("how the damage actually arrived.\n")
ce = [r["eff_corrected"] for r in rows if np.isfinite(r["eff_corrected"])]
print(f"corrected efficiency: {min(ce):.2f} to {max(ce):.2f}")
json.dump(rows, open(os.path.join(HERE, "retention_null.json"), "w"),
          indent=2, default=float)
