"""
Why a robust scale makes the floor optimistic, and by exactly how much.

Two facts sit in this paper without being connected.  Section 5 reports that the
battery cells' attainment ratio falls from 1.936 to 1.149 when a non-robust
scale is substituted, "accounting for 84 per cent of the excess".  Section 6.6
reports that on the bearings the achieved error exceeds the floor by a factor
1.35 at zero offset, and divides that out.  They are the same phenomenon and it
has an exact form.

G is built with a robust scale sigma_rob, taken as 1.4826 times the median
absolute deviation, while the variance of any estimator follows the second
moment sigma.  Writing the tail factor

    kappa = sigma / sigma_rob ,

which is 1 for Gaussian residuals and larger for heavy-tailed ones,

    G_hat = sum D'^2 / sigma_rob^2 = kappa^2 sum D'^2 / sigma^2 = kappa^2 G,

so the computed budget OVERSTATES the information by exactly kappa^2, the floor
1/sqrt(G_hat) understates the achievable error by exactly kappa, and an age read
at demand c is the age the true curve reaches at c/kappa^2.  Nothing here is
estimated: it is the definition of the two scales.

kappa is measurable from the record alone -- no ground truth, no fleet -- so it
joins the signal fraction as a diagnostic available before any model is trained.

The proposition makes a sharp prediction about numbers already in the paper.  If
the battery excess of 1.936 is the tail factor, then substituting the second
moment should divide it by kappa, and 1.936 / 1.149 = 1.68 should be the battery
cells' measured kappa.  That is checked here, along with the bearings' 1.35.
"""
import os
import json
import numpy as np
from scipy.signal import savgol_filter

from pipeline import robust_scale, oof_trend, local_scale, _win, LOCAL_BW

HERE = os.path.dirname(os.path.abspath(__file__))


def tail_factor(x):
    """The tail factor, defined with the scale the curve actually uses.

    G weights each sample by 1/sigma(tau_k)^2 with sigma a LOCAL robust scale,
    so the factor that matters is the one left after local standardisation.
    Taking a single global robust scale instead conflates the tails with the
    variation of the noise level over life -- which on these bearings is a
    factor of four -- and gives a number that answers no question the curve
    asks.  Both are computed so the difference is visible.
    """
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    n = len(x)
    if n < 60:
        return None
    s = robust_scale(x)
    if not np.isfinite(s) or s <= 0:
        return None
    D = (x - np.median(x)) / s
    w = _win(n)
    if w < 7 or w >= n:
        return None
    r = D - oof_trend(D, w // 2)
    ok = np.isfinite(r)
    r = r[ok]
    if len(r) < 40:
        return None
    rob = robust_scale(r)
    sd = float(np.std(r, ddof=1))
    if not np.isfinite(rob) or rob <= 0 or not np.isfinite(sd):
        return None
    sl = np.clip(local_scale(r, max(5, int(LOCAL_BW * len(r)))), 1e-12, None)
    z = r / sl
    z = z[np.isfinite(z)]
    if len(z) < 40:
        return None
    return dict(kappa=float(np.std(z, ddof=1)), kappa_global=sd / rob, n=n,
                kurtosis=float(np.mean(((r - r.mean()) / max(sd, 1e-12)) ** 4)))


def load_domain(name):
    """Yield (label, series) for each record of a domain, if its file exists."""
    if name == "bearings":
        p = os.path.join(HERE, "feat_raw.npz")
        if not os.path.exists(p):
            return
        z = np.load(p, allow_pickle=True)
        FN = [str(s) for s in z["featnames"]]
        i = FN.index("rms")
        for u in sorted(k for k in z.files if k != "featnames"):
            yield u, np.asarray(z[u][:, i], float)
    elif name == "battery":
        p = os.path.join(HERE, "battery_raw.npz")
        if not os.path.exists(p):
            return
        z = np.load(p, allow_pickle=True)
        for u in sorted(z.files):
            a = np.asarray(z[u], float)
            yield u, (a if a.ndim == 1 else a[:, 0])
    else:
        p = os.path.join(HERE, f"cmapss_{name}.npz")
        if not os.path.exists(p):
            return
        z = np.load(p, allow_pickle=True)
        ks = [k for k in z.files if k not in ("featnames", "sensors")]
        for u in sorted(ks)[:120]:
            a = np.asarray(z[u], float)
            if a.ndim == 1:
                yield u, a
            else:
                yield u, a[:, 0]


print("the tail factor kappa = sd / robust scale, on out-of-fold residuals\n")
print(f"{'domain':<16}{'records':>9}{'kappa, local':>14}{'quartiles':>18}"
      f"{'kappa, global':>15}{'kurtosis':>11}")
print("-" * 84)
out = {}
for dom in ("bearings", "battery", "FD001", "FD004"):
    vals = []
    for u, x in load_domain(dom):
        t = tail_factor(x)
        if t is not None and np.isfinite(t["kappa"]):
            vals.append(t)
    if len(vals) < 3:
        print(f"{dom:<16}{'--':>9}   (file not present or too few records)")
        continue
    k = np.array([v["kappa"] for v in vals])
    ku = np.array([v["kurtosis"] for v in vals])
    q1, q3 = np.percentile(k, [25, 75])
    kg = np.array([v['kappa_global'] for v in vals])
    out[dom] = dict(records=len(vals), kappa=float(np.median(k)),
                    q1=float(q1), q3=float(q3),
                    kappa_global=float(np.median(kg)),
                    kurtosis=float(np.median(ku)))
    print(f"{dom:<16}{len(vals):>9}{np.median(k):>14.3f}"
          f"{f'{q1:.3f} to {q3:.3f}':>18}{np.median(kg):>15.3f}"
          f"{np.median(ku):>11.1f}")
print("-" * 84)
print("A Gaussian residual gives kappa = 1 and kurtosis 3.  Everything above")
print("one is information the robust scale credits to the record and the")
print("estimator cannot collect.\n")

# --- the prediction the paper's own numbers make ----------------------------
print("checking the proposition against numbers already in the paper\n")
preds = []
if "battery" in out:
    want = 1.936 / 1.149
    got = out["battery"]["kappa"]
    preds.append(dict(domain="battery", predicted=float(want), measured=got,
                      ratio=got / want))
    print(f"battery : Section 5 gives 1.936 / 1.149 = {want:.2f} as the factor a")
    print(f"          non-robust scale removes; kappa measures {got:.2f} "
          f"(ratio {got / want:.2f})")
if "bearings" in out:
    want = 1.35
    got = out["bearings"]["kappa"]
    preds.append(dict(domain="bearings", predicted=float(want), measured=got,
                      ratio=got / want))
    print(f"bearings: Section 6.6 measures the excess at zero offset as "
          f"{want:.2f};")
    print(f"          kappa measures {got:.2f} (ratio {got / want:.2f})")
print()
print("The two checks are not of equal weight and should not be quoted as if")
print("they were.  The bearings agreement is DEFINITIONAL: Section 6.6 measured")
print("its factor as the standard deviation of the locally standardised")
print("residual, which is what kappa is, so the two must coincide and their")
print("agreement confirms only that no arithmetic slipped.  The battery is the")
print("independent test: its 1.68 comes from a separate experiment that")
print("substituted a non-robust scale and re-ran the attainment, and nothing in")
print("that experiment computed kappa.")
print()
bat = next((p for p in preds if p["domain"] == "battery"), None)
if bat is not None:
    off = abs(np.log(bat["ratio"]))
    if off < np.log(1.25):
        print(f"That independent prediction lands within "
              f"{100 * (np.exp(off) - 1):.0f} per cent, so the")
        print("two scattered corrections are one quantity, computable from the")
        print("record before any model is trained.")
    else:
        print(f"That independent prediction is off by "
              f"{100 * (np.exp(off) - 1):.0f} per cent, so the tail factor")
        print("accounts for much of the excess but not all of it, and the paper")
        print("should say that rather than merge the two.")
print()
print("A caveat the identity itself supplies: kappa must be formed with the")
print("scale the curve uses.  Taking a single global robust scale gives 3.20 on")
print("the bearings against 1.35 locally, because it charges the tails with the")
print("fourfold variation of the noise level over life as well.  The column")
print("above shows both so the difference cannot be lost.")

json.dump(dict(domains=out, predictions=preds),
          open(os.path.join(HERE, "tail_factor.json"), "w"), indent=2,
          default=float)
