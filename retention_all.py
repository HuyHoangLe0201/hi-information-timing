"""
What a sliding buffer costs, measured rather than assumed.

Proposition 1 makes the information of a retained sample set S proportional to
the mass of the information measure on S. For a retention budget |S| = w drawn
from the elapsed life [0, tau0], the optimum is therefore the super-level set of
the density (the bathtub solution), not necessarily the most recent w.

The earlier attempt asked which END interval wins, which needs the density to be
monotone; real ones are not. The level set needs no such assumption, so that is
what is computed here. Two figures are reported per indicator:

  efficiency  = mass(trailing w) / mass(best w)      -- what FIFO captures
  inflation   = w_FIFO / w_opt at equal information  -- how much bigger a FIFO
                                                       buffer must be to match
"""
import os, json
import numpy as np
from pipeline import profile

HERE = os.path.dirname(os.path.abspath(__file__))
TAU0, WS = 0.7, (0.10, 0.20, 0.30)


def best_mass(dens, k0, kw):
    """Mass of the kw highest-density samples within [0, k0]."""
    d = np.sort(dens[:k0 + 1])[::-1]
    return float(d[:kw].sum())


def trailing_mass(dens, k0, kw):
    return float(dens[max(0, k0 - kw + 1):k0 + 1].sum())


def inflation(dens, k0, kw):
    """How many trailing samples match the information of the best kw."""
    target = best_mass(dens, k0, kw)
    c = np.cumsum(dens[:k0 + 1][::-1])       # accumulate backwards from tau0
    j = int(np.searchsorted(c, target))
    return (j + 1) / kw if j < len(c) else np.inf


def analyse(units, label, rows):
    for w in WS:
        eff, infl = [], []
        for u in units:
            n, dens = u["n"], u["dens"]
            k0 = int(round(TAU0 * n)) - 1
            kw = max(3, int(round(w * n)))
            if k0 < kw + 3:
                continue
            b = best_mass(dens, k0, kw)
            if b <= 0:
                continue
            eff.append(trailing_mass(dens, k0, kw) / b)
            infl.append(inflation(dens, k0, kw))
        if not eff:
            continue
        fin = [x for x in infl if np.isfinite(x)]
        rows.append(dict(group=label, w=w, units=len(eff),
                         efficiency=float(np.median(eff)),
                         inflation=float(np.median(fin)) if fin else None,
                         unreachable=int(len(infl) - len(fin))))


rows = []

# --- vibration: the raw and the band-limited indicator ------------------------
fz = np.load(os.path.join(HERE, "feat_raw.npz"), allow_pickle=True)
FN = [str(s) for s in fz["featnames"]]
I = {n: k for k, n in enumerate(FN)}
BEAR = {"RMS": lambda M: M[:, I["rms"]],
        "0-2 kHz": lambda M: M[:, I["b0_1"]] + M[:, I["b1_2"]],
        "4-10 kHz": lambda M: (M[:, I["b4_6"]] + M[:, I["b6_8"]] + M[:, I["b8_10"]])}
for nm, fn in BEAR.items():
    us = [u for b in sorted(k for k in fz.files if k != "featnames")
          for u, _ in [profile(fn(fz[b]))] if u]
    analyse(us, f"bearing {nm}", rows)

# --- turbofan ---------------------------------------------------------------
for sub, f in [("FD001", "cmapss_FD001.npz"), ("FD004", "cmapss_FD004_norm.npz")]:
    z = np.load(os.path.join(HERE, f))
    names = sorted({k.split("__")[0] for k in z.files})
    us = [u for nm in names for u, _ in [profile(z[f"{nm}__sensors"][:, 10].astype(float))] if u]
    analyse(us, f"turbofan {sub} s11", rows)

# --- battery ----------------------------------------------------------------
zb = np.load(os.path.join(HERE, "battery_raw.npz"))
for ind in ["cap", "Re"]:
    us = [u for c in sorted({k.split("__")[0] for k in zb.files})
          for u, _ in [profile(zb[f"{c}__{ind}"])] if u]
    analyse(us, f"battery {ind}", rows)

print(f"{'indicator':<20}{'w':>6}{'units':>7}{'FIFO eff.':>11}{'inflation':>11}"
      f"{'unreachable':>13}")
print("-" * 70)
last = None
for r in rows:
    if last and r["group"] != last:
        print()
    last = r["group"]
    inf = f"{r['inflation']:.2f}x" if r["inflation"] else "--"
    print(f"{r['group']:<20}{r['w']:>6.2f}{r['units']:>7}{r['efficiency']:>11.3f}"
          f"{inf:>11}{r['unreachable']:>13}")
print("-" * 70)
print("efficiency 1.00 means the trailing window IS the optimal retention set;")
print("inflation is how many times more samples FIFO needs for equal information.")
print("'unreachable' counts units where no trailing window matches the optimum.")
json.dump(rows, open(os.path.join(HERE, "retention_all.json"), "w"), indent=2)
