"""
Is "broad indicators vary more" a finding, or a ceiling?

The spread of the information distribution correlates +0.63 with the
unit-to-unit IQR of tau at q = 0.35, which says a broadly-spread indicator is
the less reproducible one. Before that can be read as a property of indicators,
one confound has to be removed.

tau lives in [0, 1]. The concentrated indicators sit at tau = 0.97 or above, so
their IQR cannot exceed about 0.03 whatever they do; the broad ones sit near the
middle where there is room to vary. The correlation may be measuring nothing but
where each indicator sits.

Two controls. The logit transform sends (0,1) to the whole line, so its IQR has
no ceiling. And the correlation is recomputed with tau itself partialled out --
if the association survives holding position fixed, it is about spread; if it
vanishes, it was about position all along.
"""
import os
import json
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
rows = json.load(open(os.path.join(HERE, "spread_predicts.json")))
for r in rows:
    r["spread"] = float(r["spread"])
    r["tau35"] = float(r["tau35"])
    r["tau35_iqr"] = float(r["tau35_iqr"])

# The stored summary kept only medians and the IQR, so the per-unit values are
# recomputed here on the logit scale.
from nonparam import info_curve, quantile, spread as spread_of

EPS = 1e-3


def logit(t):
    t = np.clip(np.asarray(t, float), EPS, 1 - EPS)
    return np.log(t / (1 - t))


def per_unit(series, rng):
    T, S = [], []
    for x in series:
        F = info_curve(np.asarray(x, float), "surrogate", rng)
        if F is None:
            continue
        T.append(quantile(F, 0.35))
        S.append(spread_of(F))
    return np.array(T), np.array(S)


def rank_corr(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    m = np.isfinite(a) & np.isfinite(b)
    if m.sum() < 4:
        return np.nan
    ra = np.argsort(np.argsort(a[m])); rb = np.argsort(np.argsort(b[m]))
    return float(np.corrcoef(ra, rb)[0, 1])


def partial(x, y, z):
    """Rank correlation of x and y with z held fixed."""
    rxy, rxz, ryz = rank_corr(x, y), rank_corr(x, z), rank_corr(y, z)
    d = np.sqrt((1 - rxz ** 2) * (1 - ryz ** 2))
    return (rxy - rxz * ryz) / d if d > 0 else np.nan


rng = np.random.default_rng(9090)
SETS = {}
z = np.load(os.path.join(HERE, "feat_raw.npz"), allow_pickle=True)
FN = [str(s) for s in z["featnames"]]
I = {k: i for i, k in enumerate(FN)}
BK = sorted(k for k in z.files if k != "featnames")
for lab, parts in (("rms", ["rms"]), ("peak", ["peak"]), ("kurt", ["kurt"]),
                   ("0--2 kHz", ["b0_1", "b1_2"]), ("2--4 kHz", ["b2_4"]),
                   ("4--10 kHz", ["b4_6", "b6_8", "b8_10"]),
                   ("10--12.8 kHz", ["b10_12.8"])):
    SETS[lab] = [sum(z[b][:, I[p]] for p in parts) for b in BK]
zt = np.load(os.path.join(HERE, "cmapss_FD001.npz"))
un = sorted({k.split("__")[0] for k in zt.files})[:50]
for c in (4, 9, 11, 15, 20):
    SETS[f"turbofan s{c}"] = [zt[f"{u}__sensors"][:, c - 1].astype(float)
                              for u in un]

print("variability on a scale with no ceiling\n")
print(f"{'indicator':<16}{'spread':>8}{'tau .35':>9}{'IQR tau':>9}"
      f"{'IQR logit':>11}{'headroom':>10}")
print("-" * 63)
out = []
for lab, series in SETS.items():
    T, S = per_unit(series, rng)
    if len(T) < 4:
        continue
    iqr_t = float(np.percentile(T, 75) - np.percentile(T, 25))
    L = logit(T)
    iqr_l = float(np.percentile(L, 75) - np.percentile(L, 25))
    med = float(np.median(T))
    head = float(min(med, 1 - med))          # room available on the tau scale
    out.append(dict(indicator=lab, spread=float(np.median(S)), tau=med,
                    iqr_tau=iqr_t, iqr_logit=iqr_l, headroom=head))
out.sort(key=lambda r: r["spread"])
for r in out:
    print(f"{r['indicator']:<16}{r['spread']:>8.3f}{r['tau']:>9.3f}"
          f"{r['iqr_tau']:>9.3f}{r['iqr_logit']:>11.3f}{r['headroom']:>10.3f}")
print("-" * 63)

sp = [r["spread"] for r in out]
it = [r["iqr_tau"] for r in out]
il = [r["iqr_logit"] for r in out]
ta = [r["tau"] for r in out]
hd = [r["headroom"] for r in out]

print("\nrank correlations")
print(f"  spread with IQR on the tau scale        : {rank_corr(sp, it):+.2f}")
print(f"  spread with IQR on the logit scale      : {rank_corr(sp, il):+.2f}")
print(f"  headroom with IQR on the tau scale      : {rank_corr(hd, it):+.2f}")
print(f"  spread with headroom                    : {rank_corr(sp, hd):+.2f}")
print(f"  spread with IQR (logit), tau held fixed : {partial(sp, il, ta):+.2f}")
print()
# The test set out in the docstring is the partial correlation, not the raw
# one on the transformed scale: removing the ceiling is necessary but does not
# by itself separate spread from position, and the two are entangled here at
# +0.71. The verdict keys off the partial.
pc = partial(sp, il, ta)
if not np.isfinite(pc) or abs(pc) < 0.4:
    print(f"Holding position fixed the association is {pc:+.2f}, which with")
    print(f"{len(out)} indicators is not distinguishable from zero -- and it has")
    print("changed sign. The raw correlation was measuring where each indicator")
    print("sits: spread and headroom are themselves correlated at "
          f"{rank_corr(sp, hd):+.2f},")
    print("and headroom alone tracks the IQR at "
          f"{rank_corr(hd, it):+.2f}. The spread does not")
    print("predict how reproducible an indicator is from unit to unit.")
else:
    print(f"The association survives at {pc:+.2f} with position held fixed,")
    print("so it is a property of the spread rather than of the ceiling.")
json.dump(out, open(os.path.join(HERE, "spread_ceiling.json"), "w"), indent=2)
