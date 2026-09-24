"""
Does the spread of the information distribution predict anything?

The quantile summary produced a quantity the exponent could not express: the
spread tau_0.90 - tau_0.10, the fraction of life over which the middle 80% of
the budget arrives. It runs from 0.04 for bearing RMS -- everything in the last
twentieth of life -- to 0.92 for the 2-4 kHz band. Whether that is a useful
number or merely a describable one depends on whether it predicts something not
already contained in its own definition.

Two things are tested, one weakly definitional and one not.

  censoring    -- how much of the budget survives if the record is cut short at
                  c. This is eta(c) = F(c), so it is a function of the same
                  curve the spread comes from; it is included because it is what
                  a practitioner actually needs, not as evidence.
  variability  -- how much the design quantities move from unit to unit. This is
                  NOT determined by the spread: an indicator can concentrate its
                  information tightly and do so at the same age in every unit, or
                  at a different age in each. If concentrated indicators turn out
                  to be the erratic ones, the spread is buying real information
                  about how much a fleet-level design can be trusted.
"""
import os
import json
import numpy as np
from nonparam import info_curve, quantile, spread

HERE = os.path.dirname(os.path.abspath(__file__))
CS = (0.50, 0.70, 0.90)
SEED = 8080


def analyse(label, series, rng):
    SP, TM, ETA = [], [], {c: [] for c in CS}
    for x in series:
        F = info_curve(np.asarray(x, float), "surrogate", rng)
        if F is None:
            continue
        SP.append(spread(F))
        TM.append(quantile(F, 0.35))
        n = len(F)
        for c in CS:
            k = min(n - 1, max(0, int(round(c * n)) - 1))
            ETA[c].append(float(F[k]))
    if len(SP) < 4:
        return None
    tm = np.array(TM)
    return dict(indicator=label, units=len(SP),
                spread=float(np.median(SP)),
                tau35=float(np.median(tm)),
                tau35_iqr=float(np.percentile(tm, 75) - np.percentile(tm, 25)),
                tau35_mad=float(np.median(np.abs(tm - np.median(tm)))),
                eta={c: float(np.median(ETA[c])) for c in CS})


rng = np.random.default_rng(SEED)
rows = []

z = np.load(os.path.join(HERE, "feat_raw.npz"), allow_pickle=True)
FN = [str(s) for s in z["featnames"]]
I = {k: i for i, k in enumerate(FN)}
BK = sorted(k for k in z.files if k != "featnames")
for lab, parts in (("rms", ["rms"]), ("peak", ["peak"]), ("kurt", ["kurt"]),
                   ("0--2 kHz", ["b0_1", "b1_2"]), ("2--4 kHz", ["b2_4"]),
                   ("4--10 kHz", ["b4_6", "b6_8", "b8_10"]),
                   ("10--12.8 kHz", ["b10_12.8"])):
    r = analyse(lab, [sum(z[b][:, I[p]] for p in parts) for b in BK], rng)
    if r:
        rows.append(r)

zt = np.load(os.path.join(HERE, "cmapss_FD001.npz"))
un = sorted({k.split("__")[0] for k in zt.files})[:50]
for c in (4, 9, 11, 15, 20):
    r = analyse(f"turbofan s{c}",
                [zt[f"{u}__sensors"][:, c - 1].astype(float) for u in un], rng)
    if r:
        rows.append(r)

zb = np.load(os.path.join(HERE, "battery_raw.npz"))
cl = sorted({k.split("__")[0] for k in zb.files})
for nm, lab in (("cap", "cell capacity"), ("Re", "cell R_e")):
    r = analyse(lab, [zb[f"{c}__{nm}"] for c in cl], rng)
    if r:
        rows.append(r)

rows.sort(key=lambda r: r["spread"])
print("spread against what it might predict\n")
print(f"{'indicator':<16}{'units':>6}{'spread':>8}"
      + "".join(f"{'eta('+str(c)+')':>10}" for c in CS)
      + f"{'tau .35':>9}{'its IQR':>9}")
print("-" * 78)
for r in rows:
    print(f"{r['indicator']:<16}{r['units']:>6}{r['spread']:>8.3f}"
          + "".join(f"{r['eta'][c]:>10.3f}" for c in CS)
          + f"{r['tau35']:>9.3f}{r['tau35_iqr']:>9.3f}")
print("-" * 78)


def corr(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    m = np.isfinite(a) & np.isfinite(b)
    if m.sum() < 4:
        return np.nan
    ra = np.argsort(np.argsort(a[m])); rb = np.argsort(np.argsort(b[m]))
    return float(np.corrcoef(ra, rb)[0, 1])


sp = [r["spread"] for r in rows]
print("\nrank correlation of spread with:")
for c in CS:
    print(f"  information surviving truncation at {c:.1f} : "
          f"{corr(sp, [r['eta'][c] for r in rows]):+.2f}")
print(f"  unit-to-unit IQR of tau at q=0.35        : "
      f"{corr(sp, [r['tau35_iqr'] for r in rows]):+.2f}")
print(f"  unit-to-unit MAD of tau at q=0.35        : "
      f"{corr(sp, [r['tau35_mad'] for r in rows]):+.2f}")
print()
tight = [r for r in rows if r["spread"] < 0.4]
broad = [r for r in rows if r["spread"] > 0.7]
if tight and broad:
    print(f"concentrated indicators (spread < 0.4, {len(tight)}): "
          f"median IQR {np.median([r['tau35_iqr'] for r in tight]):.3f}, "
          f"eta(0.9) {np.median([r['eta'][0.9] for r in tight]):.3f}")
    print(f"broad indicators (spread > 0.7, {len(broad)}): "
          f"median IQR {np.median([r['tau35_iqr'] for r in broad]):.3f}, "
          f"eta(0.9) {np.median([r['eta'][0.9] for r in broad]):.3f}")
print("\neta(c) is F(c) and so shares the spread's definition; the IQR does")
print("not, and is the test of whether the spread carries information about")
print("how far a fleet-level design can be trusted on an individual unit.")
json.dump(rows, open(os.path.join(HERE, "spread_predicts.json"), "w"),
          indent=2, default=str)
