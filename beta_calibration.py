"""
Calibrate the beta_eff estimator against its own bias.

The synthetic sweep showed beta_eff is pulled towards 1 by noise: the smoother
flattens a noisy curve, and a flatter cumulative profile reads as a smaller
exponent. Every beta_eff reported so far is therefore a lower bound on the truth,
by an amount that depends on the indicator's own noise level and record length.

This builds the forward map (true beta, sigma, n) -> measured beta_eff on
synthetic data, then inverts it, so a measured value can be quoted with the
bias removed and with an honest ceiling where the map saturates.
"""
import os, json
import numpy as np
from pipeline import profile, beta_eff

HERE = os.path.dirname(os.path.abspath(__file__))
BETAS = [0.7, 1.0, 1.5, 2.0, 3.0, 5.0, 8.0, 15.0, 25.0]
SIGMAS = [0.01, 0.02, 0.04, 0.08, 0.16]
NS = [90, 200, 600, 1400]
REPS = 25
rng = np.random.default_rng(11)

table = {}
print("forward map: measured beta_eff for each (true beta, sigma, n)")
for n in NS:
    print(f"\n  n = {n}")
    print(f"{'true beta':>10}" + "".join(f"{'s='+str(s):>9}" for s in SIGMAS))
    print("  " + "-" * (10 + 9 * len(SIGMAS)))
    tau = np.arange(1, n + 1) / n
    for b in BETAS:
        D = tau ** b
        row = []
        for s in SIGMAS:
            vals = []
            for _ in range(REPS):
                u, _ = profile(D + rng.normal(0, s, n))
                if u:
                    vals.append(beta_eff(u))
            m = float(np.median(vals)) if vals else np.nan
            row.append(m)
            table[f"{n}|{b}|{s}"] = m
        print(f"{b:>10.1f}" + "".join(f"{v:>9.2f}" for v in row))

json.dump(table, open(os.path.join(HERE, "beta_calibration.json"), "w"), indent=2)


def invert(measured, sigma, n):
    """Nearest true beta whose synthetic measurement matches, at this sigma/n."""
    nn = min(NS, key=lambda x: abs(np.log(x) - np.log(n)))
    ss = min(SIGMAS, key=lambda x: abs(np.log(x) - np.log(max(sigma, 1e-3))))
    cand = [(abs(table[f"{nn}|{b}|{ss}"] - measured), b) for b in BETAS
            if np.isfinite(table[f"{nn}|{b}|{ss}"])]
    d, b = min(cand)
    top = max(table[f"{nn}|{bb}|{ss}"] for bb in BETAS)
    return (np.inf if measured > top - 1e-9 else b), nn, ss


print("\n\n=== de-biasing the reported measurements ===")
print(f"{'indicator':<22}{'n':>7}{'sigma':>8}{'measured':>10}{'corrected':>11}")
print("-" * 58)

prof = json.load(open(os.path.join(HERE, "indicator_profiles.json")))["indicators"]
sig_lookup = {}
fz = np.load(os.path.join(HERE, "feat_raw.npz"), allow_pickle=True)
FN = [str(s) for s in fz["featnames"]]; I = {k: i for i, k in enumerate(FN)}
BEAR = {"rms": lambda M: M[:, I["rms"]],
        "lf 0-2kHz": lambda M: M[:, I["b0_1"]] + M[:, I["b1_2"]],
        "hf 4-10kHz": lambda M: (M[:, I["b4_6"]] + M[:, I["b6_8"]] + M[:, I["b8_10"]])}
bkeys = sorted(k for k in fz.files if k != "featnames")
for nm, fn in BEAR.items():
    sg, ns = [], []
    for b in bkeys:
        u, _ = profile(fn(fz[b]))
        if u:
            sg.append(u["sig_in"]); ns.append(u["n"])
    if not sg:
        continue
    s, nmed, meas = float(np.median(sg)), int(np.median(ns)), prof[nm]["beta_eff"]
    corr, _, _ = invert(meas, s, nmed)
    cs = "> 25 (saturated)" if not np.isfinite(corr) else f"{corr:.1f}"
    print(f"{'bearing '+nm:<22}{nmed:>7}{s:>8.3f}{meas:>10.2f}{cs:>11}")

bat = json.load(open(os.path.join(HERE, "battery_profiles.json")))
zb = np.load(os.path.join(HERE, "battery_raw.npz"))
for ind in ["cap", "Re", "Rct"]:
    sg, ns = [], []
    for c in sorted({k.split("__")[0] for k in zb.files}):
        u, _ = profile(zb[f"{c}__{ind}"])
        if u:
            sg.append(u["sig_in"]); ns.append(u["n"])
    s, nmed = float(np.median(sg)), int(np.median(ns))
    meas = bat["summary"][ind]["beta_eff"]
    corr, _, _ = invert(meas, s, nmed)
    cs = "> 25 (saturated)" if not np.isfinite(corr) else f"{corr:.1f}"
    print(f"{'battery '+ind:<22}{nmed:>7}{s:>8.3f}{meas:>10.2f}{cs:>11}")

z1 = np.load(os.path.join(HERE, "cmapss_FD001.npz"))
names = sorted({k.split("__")[0] for k in z1.files})
cm = json.load(open(os.path.join(HERE, "cmapss_profiles_FD001.json")))["rows"]
for r in cm:
    if r["sensor"] not in (9, 11, 20):
        continue
    sg, ns = [], []
    for nm2 in names:
        u, _ = profile(z1[f"{nm2}__sensors"][:, r["sensor"] - 1].astype(float))
        if u:
            sg.append(u["sig_in"]); ns.append(u["n"])
    s, nmed = float(np.median(sg)), int(np.median(ns))
    corr, _, _ = invert(r["beta_eff"], s, nmed)
    cs = "> 25 (saturated)" if not np.isfinite(corr) else f"{corr:.1f}"
    print(f"{'turbofan s'+str(r['sensor']):<22}{nmed:>7}{s:>8.3f}{r['beta_eff']:>10.2f}{cs:>11}")
