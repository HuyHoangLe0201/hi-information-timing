"""
Is the early information peak real degradation, or the run-in transient?

The optimal retention set for bearings sits near 27% of life, not at the end,
even though the fitted trend is a late-rising power law. Two possibilities:

  (a) run-in / break-in.  The trend model does not describe the first part of
      life, so the derivative there is large for a reason that carries no
      information about the degradation parameters. F would be crediting it
      spuriously, and the retention penalty would be an artefact of
      misspecification rather than a property of the data.
  (b) a real, reproducible early stage that a trailing buffer discards.

They are distinguishable. Run-in is a fixed feature of the rig, so it should sit
at the same place in every unit; per-unit damage events should not. And if it is
run-in, excluding it should move the optimum to the end and make FIFO look fine.
"""
import os, json
import numpy as np
from pipeline import profile

HERE = os.path.dirname(os.path.abspath(__file__))
TAU0, W = 0.7, 0.20

fz = np.load(os.path.join(HERE, "feat_raw.npz"), allow_pickle=True)
FN = [str(s) for s in fz["featnames"]]; I = {k: i for i, k in enumerate(FN)}
bk = sorted(k for k in fz.files if k != "featnames")
SETS = {"bearing RMS": [(b, fz[b][:, I["rms"]]) for b in bk],
        "bearing 4-10 kHz":
            [(b, fz[b][:, I["b4_6"]] + fz[b][:, I["b6_8"]] + fz[b][:, I["b8_10"]])
             for b in bk]}

GRID = np.linspace(0, 1, 101)


def density_curve(raw):
    u, _ = profile(raw)
    if not u:
        return None
    d = u["dens"]; n = len(d)
    c = np.cumsum(d); c = c / c[-1]
    return np.interp(GRID, np.arange(1, n + 1) / n, c)


def eff_after_trim(raw, trim):
    """Efficiency when life before `trim` is excluded from the record."""
    n0 = len(raw)
    raw = raw[int(round(trim * n0)):]
    u, _ = profile(raw)
    if not u:
        return None
    n = u["n"]; k0 = int(round(TAU0 * n)) - 1; kw = max(3, int(round(W * n)))
    if k0 < kw + 3:
        return None
    d = u["dens"]
    best = float(np.sort(d[:k0 + 1])[::-1][:kw].sum())
    trail = float(d[max(0, k0 - kw + 1):k0 + 1].sum())
    pos = float(np.median(np.sort(np.argsort(d[:k0 + 1])[::-1][:kw])) / (k0 + 1))
    return (trail / best if best > 0 else np.nan), pos


print("(a) is the early peak in the same place in every unit?\n")
print(f"{'indicator':<20}{'units':>6}{'peak tau: median':>18}{'IQR':>18}"
      f"{'spread':>9}")
print("-" * 71)
peaks = {}
for name, series in SETS.items():
    P = []
    for _, raw in series:
        c = density_curve(raw)
        if c is None:
            continue
        rate = np.diff(c)
        P.append(GRID[1:][int(np.argmax(rate))])
    if not P:
        continue
    peaks[name] = P
    q1, q3 = np.percentile(P, [25, 75])
    print(f"{name:<20}{len(P):>6}{np.median(P):>18.2f}"
          f"{f'[{q1:.2f}, {q3:.2f}]':>18}{q3-q1:>9.2f}")

print("\n  A tight IQR means one shared rig feature; a wide one means the peak")
print("  is per-unit, i.e. not run-in.\n")

print("(b) does excluding early life restore a trailing window?\n")
print(f"{'indicator':<20}{'trim':>7}{'efficiency':>12}{'median pos':>12}")
print("-" * 51)
rows = []
for name, series in SETS.items():
    for trim in (0.0, 0.10, 0.20, 0.30):
        vals = [eff_after_trim(raw, trim) for _, raw in series]
        vals = [v for v in vals if v is not None]
        if not vals:
            continue
        e = float(np.median([v[0] for v in vals]))
        p = float(np.median([v[1] for v in vals]))
        rows.append(dict(name=name, trim=trim, eff=e, pos=p))
        print(f"{name:<20}{trim:>7.2f}{e:>12.3f}{p:>12.2f}")
    print()

print("-" * 51)
print("If efficiency climbs toward 1 as the early record is dropped, the penalty")
print("was misspecification. If it stays low, the information really is early.")
json.dump(dict(peaks={k: list(map(float, v)) for k, v in peaks.items()},
               trim=rows), open(os.path.join(HERE, "retention_runin.json"), "w"),
          indent=2)
