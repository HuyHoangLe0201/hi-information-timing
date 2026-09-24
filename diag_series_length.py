"""
Control: is the battery slope a battery effect, or a series-length effect?

Battery records hold 61-125 points; turbofan ~200, bearings ~1400. If the bound
degrades simply because a short record gives few samples per window, then
decimating a domain that works down to battery length should reproduce the
battery slope. If it does not, something specific to batteries is responsible.

Decimation keeps the physical span and thins the sampling, which is what a
lower-rate sensor would do -- exactly the f_s in the model.
"""
import os, json
import numpy as np
from pipeline import prepare, LOCAL_BW

HERE = os.path.dirname(os.path.abspath(__file__))
TAU0 = np.round(np.arange(0.30, 0.96, 0.05), 3)
WGRID = np.array([0.10, 0.15, 0.20, 0.30, 0.40, 0.50, 0.60])
N_MC = 300


def run(units):
    rng = np.random.default_rng(20260809)
    lp, lr = [], []
    for u in units:
        n, dt, sl, res = u["n"], u["dtrend"], u["sig_loc"], u["res"]
        for t0 in TAU0:
            k0 = int(round(t0 * n)) - 1
            if k0 < 5 or k0 >= n:
                continue
            for w in WGRID:
                if w > t0:
                    continue
                klo = max(0, int(round((t0 - w) * n)) - 1)
                if k0 - klo < 10:
                    continue
                g, s = dt[klo:k0 + 1], sl[klo:k0 + 1]
                f = float(np.sum((g / s) ** 2))
                if f <= 0:
                    continue
                wt = g / s ** 2
                den = float(np.sum(wt * g))
                if den <= 0:
                    continue
                dr = rng.choice(res[klo:k0 + 1], size=(N_MC, len(g)), replace=True)
                lp.append(np.log(1.0 / np.sqrt(f)))
                lr.append(np.log(float(np.std(dr @ wt / den, ddof=1))))
    if len(lp) < 30:
        return None
    lp, lr = np.asarray(lp), np.asarray(lr)
    return dict(cells=len(lp), slope=float(np.polyfit(lp, lr, 1)[0]),
                corr=float(np.corrcoef(lp, lr)[0, 1]),
                median=float(np.median(np.exp(lr - lp))))


# turbofan FD001 s11, decimated to a range of record lengths
z = np.load(os.path.join(HERE, "cmapss_FD001.npz"))
names = sorted({k.split("__")[0] for k in z.files})[:60]
raws = [z[f"{n}__sensors"][:, 10].astype(float) for n in names]

print(f"{'series length':<20}{'series':>7}{'med n':>7}{'cells':>7}"
      f"{'slope':>8}{'r':>7}{'median':>8}")
print("-" * 64)
for label, step in [("full (~200)", 1), ("every 2nd (~100)", 2),
                    ("every 3rd (~66)", 3), ("every 4th (~50)", 4)]:
    us = []
    for r in raws:
        u, _ = prepare(r[::step])
        if u:
            us.append(u)
    res = run(us)
    if res is None:
        print(f"{label:<20} too few"); continue
    print(f"{label:<20}{len(us):>7}{int(np.median([u['n'] for u in us])):>7}"
          f"{res['cells']:>7}{res['slope']:>8.3f}{res['corr']:>7.3f}{res['median']:>8.3f}")

print("-" * 64)
bat = json.load(open(os.path.join(HERE, "crb_all_domains.json")))["battery"]
print(f"{'battery (n~89)':<20}{bat['series']:>7}{89:>7}{bat['cells']:>7}"
      f"{bat['slope']:>8.3f}{bat['corr']:>7.3f}{bat['median']:>8.3f}")
print("\nIf the decimated turbofan rows fall towards the battery row, record")
print("length is the cause and the bound is intact; if they hold near 0.9, it")
print("is not, and something about battery records is.")
