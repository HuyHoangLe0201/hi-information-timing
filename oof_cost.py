"""
How much does fitting the smoother to its own residuals cost?

Section 3.1 says the noise scale is estimated out of fold "because residuals of a
smoother fitted to the same samples understate the scale".  That is asserted and
never quantified, and it is the justification for a choice made in every curve in
the paper.  An assertion of that standing should carry a number, and if the
number is small the paper should say so rather than imply the choice is decisive.

Three quantities are measured on every record and channel.  How much the
in-sample residual scale understates the out-of-fold one, which is the effect
claimed.  What that does to the information budget, which scales as 1/sigma^2 and
therefore amplifies the difference.  And what it does to the earliest usable age,
which is the only thing the paper reports and which may be insensitive to both.

The third is the one that matters.  A choice that moves the budget appreciably
but the age not at all is a choice about presentation; a choice that moves the
age is a choice about results.
"""
import os
import json
import numpy as np
from scipy.signal import savgol_filter

from pipeline import robust_scale, oof_trend, local_scale, _win, LOCAL_BW

HERE = os.path.dirname(os.path.abspath(__file__))
QSTAR = 0.35


def both_scales(x):
    """Density and age with the residual taken in sample and out of fold."""
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    n = len(x)
    if n < 200:
        return None
    s = robust_scale(x)
    if not np.isfinite(s) or s <= 0:
        return None
    D = (x - np.median(x)) / s
    w = _win(n)
    if w < 7:
        return None
    d1 = savgol_filter(D, w, 2, deriv=1, delta=1.0 / n)
    r_in = D - savgol_filter(D, w, 2)          # fitted to the same samples
    r_oof = D - oof_trend(D, w // 2)           # each fold predicted without itself
    out = {}
    for tag, r in (("in", r_in), ("oof", r_oof)):
        sl = np.clip(local_scale(r, max(5, int(LOCAL_BW * n))), 1e-12, None)
        dens = (d1 / sl) ** 2
        tot = float(dens.sum())
        if tot <= 0:
            return None
        c = np.cumsum(dens)
        k = int(np.searchsorted(c, QSTAR * tot))
        out[tag] = dict(scale=float(np.median(sl)), G=tot,
                        tau=(k + 1.0) / n if k < n else np.nan)
    return out


z = np.load(os.path.join(HERE, "feat_raw.npz"), allow_pickle=True)
FN = [str(s) for s in z["featnames"]]
I = {k: i for i, k in enumerate(FN)}
units = sorted(k for k in z.files if k != "featnames")

rows = []
for u in units:
    for nm in FN:
        b = both_scales(np.asarray(z[u][:, I[nm]], float))
        if b is None:
            continue
        rows.append(dict(unit=u, channel=nm,
                         scale_ratio=b["oof"]["scale"] / b["in"]["scale"],
                         G_ratio=b["in"]["G"] / b["oof"]["G"],
                         tau_in=b["in"]["tau"], tau_oof=b["oof"]["tau"],
                         tau_shift=b["oof"]["tau"] - b["in"]["tau"]))
print(f"{len(rows)} record--channel cells\n")

sr = np.array([r["scale_ratio"] for r in rows], float)
gr = np.array([r["G_ratio"] for r in rows], float)
ts = np.array([r["tau_shift"] for r in rows], float)
ts = ts[np.isfinite(ts)]

print(f"{'quantity':<46}{'median':>10}{'quartiles':>20}")
print("-" * 76)
print(f"{'out-of-fold scale over in-sample scale':<46}{np.median(sr):>10.3f}"
      f"{f'{np.percentile(sr, 25):.3f} to {np.percentile(sr, 75):.3f}':>20}")
print(f"{'budget inflation from using the in-sample scale':<46}"
      f"{np.median(gr):>10.3f}"
      f"{f'{np.percentile(gr, 25):.3f} to {np.percentile(gr, 75):.3f}':>20}")
print(f"{'shift in the earliest usable age':<46}{np.median(ts):>+10.4f}"
      f"{f'{np.percentile(ts, 25):+.4f} to {np.percentile(ts, 75):+.4f}':>20}")
print("-" * 76)
big = int((np.abs(ts) > 0.01).sum())
print(f"\ncells whose age moves by more than 0.01 of a lifetime: {big} of "
      f"{len(ts)}")
print()
print("The scale ratio is the effect the paper asserts; the budget ratio is")
print("that effect squared, since G scales as 1/sigma^2; the age shift is the")
print("only one that reaches a reported number.\n")

worst = float(np.max(np.abs(ts)))
print(f"largest shift on any single cell: {worst:.3f} of a lifetime\n")
if np.median(sr) <= 1.005:
    print("The in-sample scale does NOT measurably understate on these records,")
    print("so the justification in Section 3.1 is not supported and should be")
    print("stated as a precaution rather than as a correction.")
else:
    print("The assertion is correct in direction and its magnitude is now on")
    print(f"record: the in-sample scale understates by {100 * (np.median(sr) - 1):.1f} per cent, which")
    print(f"inflates the budget by {100 * (np.median(gr) - 1):.1f} per cent, since G scales as 1/sigma^2.")
    print()
    if abs(np.median(ts)) < 0.005 and np.percentile(np.abs(ts), 75) < 0.01:
        print(f"On the quantity the paper reports the effect is negligible: the age")
        print(f"moves by a median {np.median(ts):+.4f} of a lifetime, with three quarters of")
        print(f"cells inside {np.percentile(np.abs(ts), 75):.4f}.  It is not inert -- {big} of {len(ts)} cells move")
        print(f"by more than 0.01 and the worst by {worst:.3f} -- so the choice is worth")
        print("making, but no reported median depends on it.  The paper should")
        print("present it as the correct precaution it is and not as a correction")
        print("the results rest on.")
    else:
        print("The reported age moves appreciably, so this is a choice about")
        print("results and not only about presentation.")

json.dump(dict(cells=len(rows),
               scale_ratio_median=float(np.median(sr)),
               scale_ratio_q1=float(np.percentile(sr, 25)),
               scale_ratio_q3=float(np.percentile(sr, 75)),
               G_ratio_median=float(np.median(gr)),
               tau_shift_median=float(np.median(ts)),
               tau_shift_q1=float(np.percentile(ts, 25)),
               tau_shift_q3=float(np.percentile(ts, 75)),
               cells_moved=big, cells_total=int(len(ts)),
               tau_shift_worst=float(np.max(np.abs(ts))),
               tau_shift_q75abs=float(np.percentile(np.abs(ts), 75))),
          open(os.path.join(HERE, "oof_cost.json"), "w"), indent=2, default=float)
