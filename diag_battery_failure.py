"""
Why does the bound fail on batteries?

Two candidates, and they call for different conclusions:
  (H1) series too short -- 61-125 cycles, so each window holds few samples and
       the variance estimate is poor. A limitation of the data, not the theory.
  (H2) residuals not white -- capacity records show regeneration after rest,
       which is strongly autocorrelated. Model (1) assumes white measurement
       noise, so this would be the assumption failing, and it is checkable
       before anyone trusts the bound.
"""
import os
import numpy as np
from pipeline import prepare

HERE = os.path.dirname(os.path.abspath(__file__))


def acorr_len(r):
    r = r - r.mean()
    n = len(r)
    ac = np.correlate(r, r, "full")[n - 1:]
    ac /= ac[0]
    below = np.where(ac < np.exp(-1))[0]
    return int(below[0]) if len(below) else n


def report(label, series):
    Ls, ns, ks = [], [], []
    for nm, raw in series:
        u, why = prepare(raw, nm)
        if u is None:
            continue
        L = acorr_len(u["res"])
        Ls.append(L); ns.append(u["n"])
        r = u["res"] / u["sig_oof"]
        ks.append(float(np.mean(r ** 4) - 3.0))
    if not Ls:
        print(f"{label:<24} no usable series"); return
    print(f"{label:<24}{len(Ls):>7}{int(np.median(ns)):>7}"
          f"{np.median(Ls):>10.1f}{np.median(Ls)/np.median(ns)*100:>9.1f}%"
          f"{np.median(ks):>10.1f}")


print(f"{'domain / indicator':<24}{'series':>7}{'med n':>7}"
      f"{'acorr L':>10}{'L/n':>10}{'kurt':>10}")
print("-" * 68)

zb = np.load(os.path.join(HERE, "battery_raw.npz"))
for ind in ["cap", "Re", "Rct"]:
    report(f"battery {ind}",
           [(c, zb[f"{c}__{ind}"]) for c in sorted({k.split("__")[0] for k in zb.files})])

z1 = np.load(os.path.join(HERE, "cmapss_FD001.npz"))
names = sorted({k.split("__")[0] for k in z1.files})[:40]
report("turbofan FD001 s11",
       [(n, z1[f"{n}__sensors"][:, 10].astype(float)) for n in names])

fz = np.load(os.path.join(HERE, "feat_raw.npz"), allow_pickle=True)
FN = [str(s) for s in fz["featnames"]]; I = {n: k for k, n in enumerate(FN)}
report("bearing 4-10 kHz",
       [(b, fz[b][:, I["b4_6"]] + fz[b][:, I["b6_8"]] + fz[b][:, I["b8_10"]])
        for b in sorted(k for k in fz.files if k != "featnames")])

print("-" * 68)
print("Model (1) assumes white noise: acorr L = 1 sample. A long L means the")
print("residual carries structure the bound treats as independent, so it")
print("over-counts the information a window holds.")
