"""
The saturation-rate anomaly is an underpowered experiment, not a physical puzzle.

Theorem 1 gives n_sat proportional to rho_w^{1/a}. Part A confirmed that
identity exactly. Part B then measured a on C-MAPSS learning curves and got
a = 3.5 from the observed n_sat ratio -- far above the minimax rate of 1, which
would be remarkable if it were real.

Three properties of that measurement say it cannot be:

  resolution -- n_sat was located on the grid [4, 6, 10, 15, 22, 32, 45, 60],
                whose adjacent points differ by a factor of about 1.4. The
                observed ratio was 1.406. It is one grid step, which is the
                smallest non-zero answer the grid can return;
  separation -- the two learning curves converge to floors of 11.84 and 11.69,
                1.3% apart. If the floors are the same, rho_w does not change
                the asymptote and n_sat is not defined by it;
  noise      -- the w = 0.4 curve rises from 15.22 to 16.02 between two
                consecutive sample sizes, so its own scatter is about 5%, four
                times the floor separation it is being asked to resolve.

This quantifies all three, and then asks what a design able to resolve the
question would need.
"""
import os
import json
import numpy as np
from scipy.optimize import curve_fit

HERE = os.path.dirname(os.path.abspath(__file__))
st = json.load(open(os.path.join(HERE, "saturation_test.json")))
NS = np.array(st["NS"], float)
CURVES = {k: np.array(v, float) for k, v in st["curves"].items()}
RHO = {k: float(v) for k, v in st["rho"].items()}

print("1. what resolution does the sample grid have?\n")
steps = NS[1:] / NS[:-1]
print(f"   grid: {[int(v) for v in NS]}")
print(f"   adjacent ratios: {np.round(steps, 2).tolist()}")
print(f"   median step {np.median(steps):.3f}, observed n_sat ratio "
      f"{st['observed_ratio']:.3f}")
print(f"   -> the observed ratio is {st['observed_ratio']/np.median(steps):.2f} "
      f"grid steps\n")

print("2. are the two curves' floors distinguishable?\n")
floors = {k: float(v[-1]) for k, v in CURVES.items()}
print(f"   floors: " + ", ".join(f"w={k} -> {v:.2f}" for k, v in floors.items()))
sep = abs(floors[list(floors)[0]] - floors[list(floors)[1]])
rel = sep / np.mean(list(floors.values()))
print(f"   separation {sep:.2f} ({100*rel:.1f}% of the level)\n")

print("3. how noisy is each curve?\n")
for k, y in CURVES.items():
    d = np.diff(y)
    up = d[d > 0]
    rough = float(np.median(np.abs(d)) / np.mean(y))
    print(f"   w={k}: {len(up)} of {len(d)} steps go the wrong way; "
          f"typical step {100*rough:.1f}% of the level"
          + (f", worst rise {100*up.max()/np.mean(y):.1f}%" if len(up) else ""))
print()

print("4. what does the rate come out as, and how stable is it?\n")


def model(n, floor, c, a):
    return floor + c * n ** (-a)


print(f"{'w':>6}{'rate a':>9}{'95% interval':>22}")
print("-" * 39)
fits = {}
for k, y in CURVES.items():
    try:
        p, cov = curve_fit(model, NS, y, p0=[y[-1] * 0.95, y[0], 0.5],
                           bounds=([0, 0, 0.05], [y.min(), 1e4, 3.0]),
                           maxfev=40000)
        se = float(np.sqrt(np.diag(cov))[2])
        fits[k] = (float(p[2]), se)
        lo, hi = p[2] - 1.96 * se, p[2] + 1.96 * se
        print(f"{k:>6}{p[2]:>9.2f}{f'[{lo:.2f}, {hi:.2f}]':>22}")
    except Exception as e:
        print(f"{k:>6}   fit failed ({e})")
print("-" * 39)
if len(fits) == 2:
    a1, a2 = [v[0] for v in fits.values()]
    print(f"   the two curves disagree by {max(a1,a2)/min(a1,a2):.1f}x on a")
    print("   quantity they should share, which is itself the answer.\n")

print("5. what would be needed to resolve it?\n")
ks = sorted(RHO, key=float)
ratio = RHO[ks[1]] / RHO[ks[0]]
print(f"   rho_w ratio between the two windows: {ratio:.2f}")
for a in (0.5, 1.0, 2.0):
    print(f"   at a = {a:.1f} the n_sat ratio would be {ratio**(1/a):>5.2f}")
need = ratio ** (1 / 1.0)
print(f"\n   to separate a = 0.5 from a = 1.0 the grid must resolve "
      f"{ratio**2:.1f} from {ratio:.1f},")
print(f"   so its step must be well below {ratio:.2f}; the present step is "
      f"{np.median(steps):.2f}.")
gridpts = np.log(NS[-1] / NS[0]) / np.log(1.15)
print(f"   a step of 1.15 over the same range needs {gridpts:.0f} sample sizes, "
      f"against {len(NS)}.")
print(f"   and the floors must be separated by more than the {100*rel:.1f}% "
      f"they differ by now,")
print("   which needs windows whose rho_w differ by far more than "
      f"{ratio:.1f}x.\n")

print("The measured a = 3.5 is the grid's smallest resolvable answer, taken")
print("from two curves that end at the same place. It is not evidence about")
print("the learning rate, and the anomaly does not need a physical explanation.")
json.dump(dict(grid_step=float(np.median(steps)),
               observed_ratio=float(st["observed_ratio"]),
               floors=floors, floor_separation_rel=float(rel),
               fits={k: dict(a=v[0], se=v[1]) for k, v in fits.items()},
               rho_ratio=float(ratio)),
          open(os.path.join(HERE, "saturation_power.json"), "w"), indent=2)
