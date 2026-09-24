"""
The theory assumes constant observation noise. Vibration indicators do not have it.

With a scale that varies along the record the information density is
D'(tau)^2 / sigma(tau)^2, not D'(tau)^2, and the closed forms of the power-law
class no longer follow. They do follow again if the scale is multiplicative,
sigma proportional to D^gamma, because then

    D'^2 / sigma^2  ~  tau^{2 beta - 2 - 2 beta gamma},

which is the power-law density of an indicator with exponent

    beta_eff = beta (1 - gamma).

If gamma is materially above zero on real records, every exponent measured with
the unweighted density is an underestimate of the trend exponent by that factor,
and the earliest usable age computed from it is wrong in a predictable
direction. This measures gamma, and then checks the prediction directly: the
weighted density should give an exponent larger than the unweighted one by
1/(1-gamma).

gamma is fitted against the raw indicator level rather than the normalised D,
because the normalisation puts D(0) = 0 and the logarithm of the trend is then
undefined over the early part of the record.
"""
import os
import json
import numpy as np
from scipy.signal import savgol_filter
from pipeline import robust_scale, oof_trend, local_scale, _win, LOCAL_BW
from fusion_prep import prepare_fusion


HERE = os.path.dirname(os.path.abspath(__file__))
MIN_PTS = 60


def gamma_of(raw):
    """Slope of log(local scale) against log(trend level)."""
    x = np.asarray(raw, float)
    n = len(x)
    w = _win(n)
    if w < 7 or n < MIN_PTS:
        return None
    tr = savgol_filter(x, w, 2)
    res = x - oof_trend(x, w // 2)
    s = local_scale(res, max(5, int(LOCAL_BW * n)))
    m = np.isfinite(tr) & np.isfinite(s) & (tr > 0) & (s > 0)
    if m.sum() < MIN_PTS:
        return None
    lt, ls = np.log(tr[m]), np.log(s[m])
    if np.std(lt) < 1e-9:
        return None
    g, c = np.polyfit(lt, ls, 1)
    r = float(np.corrcoef(lt, ls)[0, 1])
    return float(g), r, float(np.ptp(s[m]) / np.median(s[m]))


def beta_from_density(dens, n):
    """Exponent whose power-law cumulative best matches this density."""
    from scipy.optimize import minimize_scalar
    tau = np.arange(1, n + 1) / n
    F = np.cumsum(dens)
    if F[-1] <= 0:
        return np.nan
    F = F / F[-1]
    f = lambda b: float(np.mean((F - tau ** (2 * b - 1)) ** 2))
    r = minimize_scalar(f, bounds=(0.51, 30.0), method="bounded")
    return float(r.x)


def analyse(units, label):
    print(f"\n=== {label} ===\n")
    print(f"{'channel':<14}{'units':>6}{'gamma':>8}{'r':>7}{'scale span':>12}"
          f"{'beta unwt':>11}{'beta wtd':>10}{'predicted':>11}")
    print("-" * 79)
    rows = []
    for nm in units[0][1]:
        G, RR, SP, BU, BW = [], [], [], [], []
        for uname, chans in units:
            g = gamma_of(chans[nm])
            if g is None:
                continue
            p, _ = prepare_fusion(chans[nm], nm)
            if p is None:
                continue
            G.append(g[0]); RR.append(g[1]); SP.append(g[2])
            BU.append(beta_from_density(p["dtrend"] ** 2, p["n"]))
            sl = np.clip(p["sig_loc"], 1e-12, None)
            BW.append(beta_from_density((p["dtrend"] / sl) ** 2, p["n"]))
        if len(G) < 4:
            continue
        gm = float(np.median(G))
        bu, bw = float(np.median(BU)), float(np.median(BW))
        pred = bu / (1 - gm) if gm < 0.95 else np.nan
        rows.append(dict(channel=nm, units=len(G), gamma=gm,
                         r=float(np.median(RR)), span=float(np.median(SP)),
                         beta_unweighted=bu, beta_weighted=bw,
                         beta_predicted=float(pred)))
        ps = "     --  " if not np.isfinite(pred) else f"{pred:>11.2f}"
        print(f"{nm:<14}{len(G):>6}{gm:>8.3f}{np.median(RR):>7.2f}"
              f"{np.median(SP):>12.1f}{bu:>11.2f}{bw:>10.2f}{ps}")
    print("-" * 79)
    ok = [r for r in rows if np.isfinite(r["beta_predicted"])]
    if ok:
        err = [abs(r["beta_weighted"] - r["beta_predicted"]) /
               max(r["beta_predicted"], 1e-9) for r in ok]
        print(f"median |weighted - predicted| / predicted : "
              f"{np.median(err):.2f}")
    print("gamma > 0 means the noise grows with the signal, so the unweighted")
    print("exponent understates the trend exponent by the factor 1/(1-gamma).")
    return rows


out = {}
z = np.load(os.path.join(HERE, "feat_raw.npz"), allow_pickle=True)
FN = [str(s) for s in z["featnames"]]
I = {k: i for i, k in enumerate(FN)}
BU = [(b, {nm: z[b][:, I[nm]] for nm in FN})
      for b in sorted(k for k in z.files if k != "featnames")]
out["bearings"] = analyse(BU, "PRONOSTIA bearings")

zb = np.load(os.path.join(HERE, "battery_raw.npz"))
cells = sorted({k.split("__")[0] for k in zb.files})
CU = [(c, {nm: zb[f"{c}__{nm}"] for nm in ("cap", "Re", "Rct")}) for c in cells]
out["cells"] = analyse(CU, "NASA cells")

zt = np.load(os.path.join(HERE, "cmapss_FD001.npz"))
names = sorted({k.split("__")[0] for k in zt.files})[:60]
TU = [(nm, {f"s{c}": zt[f"{nm}__sensors"][:, c - 1].astype(float)
            for c in (4, 11, 15)}) for nm in names]
out["turbofan"] = analyse(TU, "C-MAPSS FD001")

json.dump(out, open(os.path.join(HERE, "hetero_exponent.json"), "w"), indent=2)
print("\nwrote hetero_exponent.json")
