"""
When does measuring a unit beat simply knowing the fleet?

The Cramer-Rao bound constrains unbiased estimators. Every deployed RUL model
is biased, because it carries prior knowledge: trained on a fleet, it knows
roughly how long these things last before it sees any data from this unit. The
Bayesian (van Trees) form makes that explicit,

    Var(delta_hat)  >=  1 / ( Gamma_W + Gamma_prior ),

with Gamma_prior the Fisher information of the prior. For a Gaussian prior on
the damage-clock offset with standard deviation sigma_p, Gamma_prior = 1/sigma_p^2,
and sigma_p is not a modelling choice -- it is the fleet's own spread of life,
which the data give.

Two things follow that the unbiased theory cannot say.

  The earliest usable age becomes  G^{-1}( 1/eps^2 - 1/sigma_p^2 ), so a tight
  fleet can make a target reachable that no amount of measurement would reach,
  and an indicator can be worth nothing at all if the prior already suffices.

  There is an age at which the record overtakes the prior, G(tau) = 1/sigma_p^2.
  Before it a model is mostly restating the fleet average; after it the
  measurement is doing the work. That crossing is a property of the indicator
  and the fleet together, and is what a practitioner actually wants to know.

Both are computed here per domain, from the weighted density D'^2/sigma^2, which
is the one the information is written in.
"""
import os
import json
import numpy as np
from scipy.signal import savgol_filter
from pipeline import robust_scale, oof_trend, local_scale, _win, LOCAL_BW

HERE = os.path.dirname(os.path.abspath(__file__))


def accum(x):
    """Unnormalised cumulative information G(tau), weighted density."""
    n = len(x)
    w = _win(n)
    if w < 7:
        return None
    s = robust_scale(x)
    if not np.isfinite(s) or s <= 0:
        return None
    D = (x - np.median(x)) / s
    dt = savgol_filter(D, w, 2, deriv=1, delta=1.0 / n)
    res = D - oof_trend(D, w // 2)
    sl = np.clip(local_scale(res, max(5, int(LOCAL_BW * n))), 1e-12, None)
    return np.cumsum((dt / sl) ** 2)


def crossing(G, level):
    if G is None or G[-1] < level:
        return np.nan
    return (int(np.searchsorted(G, level)) + 1) / len(G)


def report(name, lives, series):
    T = np.asarray(lives, float)
    cv = float(np.std(T, ddof=1) / np.mean(T))     # fleet spread of life
    gp = 1.0 / cv ** 2                             # prior information
    Gs, tot = [], []
    for x in series:
        G = accum(np.asarray(x, float))
        if G is None or not np.isfinite(G[-1]) or G[-1] <= 0:
            continue
        Gs.append(G); tot.append(float(G[-1]))
    if not Gs:
        return None
    cross = [crossing(G, gp) for G in Gs]
    ok = [c for c in cross if np.isfinite(c)]
    r = dict(domain=name, units=len(Gs), life_cv=cv, gamma_prior=gp,
             gamma_record=float(np.median(tot)),
             ratio=float(np.median(tot)) / gp,
             crossing=float(np.median(ok)) if ok else np.nan,
             never=len(cross) - len(ok))
    cs = f"{r['crossing']:.3f}" if ok else "--"
    print(f"{name:<22}{len(Gs):>6}{cv:>9.2f}{gp:>12.1f}"
          f"{np.median(tot):>14.3g}{r['ratio']:>10.1f}{cs:>10}"
          f"{r['never']:>7}")
    return r


print("the record against the prior\n")
print(f"{'domain':<22}{'units':>6}{'life CV':>9}{'G_prior':>12}"
      f"{'G_record':>14}{'ratio':>10}{'crossing':>10}{'never':>7}")
print("-" * 90)
rows = []

z = np.load(os.path.join(HERE, "feat_raw.npz"), allow_pickle=True)
FN = [str(s) for s in z["featnames"]]
I = {k: i for i, k in enumerate(FN)}
BK = sorted(k for k in z.files if k != "featnames")
lives = [len(z[b]) for b in BK]
for lab, parts in (("bearing RMS", ["rms"]),
                   ("bearing 0--2 kHz", ["b0_1", "b1_2"]),
                   ("bearing 4--10 kHz", ["b4_6", "b6_8", "b8_10"])):
    s = [sum(z[b][:, I[p]] for p in parts) for b in BK]
    r = report(lab, lives, s)
    if r:
        rows.append(r)

zt = np.load(os.path.join(HERE, "cmapss_FD001.npz"))
un = sorted({k.split("__")[0] for k in zt.files})
tl = [len(zt[f"{u}__sensors"]) for u in un]
for c, lab in ((4, "turbofan s4"), (11, "turbofan s11"), (15, "turbofan s15")):
    s = [zt[f"{u}__sensors"][:, c - 1].astype(float) for u in un]
    r = report(lab, tl, s)
    if r:
        rows.append(r)

zb = np.load(os.path.join(HERE, "battery_raw.npz"))
cl = sorted({k.split("__")[0] for k in zb.files})
bl = [len(zb[f"{c}__cap"]) for c in cl]
r = report("cell capacity", bl, [zb[f"{c}__cap"] for c in cl])
if r:
    rows.append(r)

print("-" * 90)
print("life CV   = fleet standard deviation of life over its mean; the prior")
print("            is worth 1/CV^2 in the same units as the record's budget")
print("crossing  = age at which the record's accumulated information equals")
print("            the prior's; before it, the fleet average is the better")
print("            source, and a model is mostly restating it")
print("never     = units whose whole record never overtakes the prior")
print()
for r in sorted(rows, key=lambda r: r["ratio"]):
    if r["ratio"] < 1:
        print(f"  {r['domain']}: the entire record carries less than the fleet "
              f"prior ({r['ratio']:.2f}x)")
json.dump(rows, open(os.path.join(HERE, "prior_crossover.json"), "w"), indent=2)
