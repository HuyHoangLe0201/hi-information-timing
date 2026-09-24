"""Quick numeric look at the RMS trajectories, to see what shape the data
actually has before committing to a degradation class."""
import os
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
z = np.load(os.path.join(HERE, "hi_raw.npz"))
BEARINGS = sorted({k.split("__")[0] for k in z.files if not k.startswith("meta__")})

print(f"{'bearing':<13}{'N':>6}{'base':>8}{'last':>9}{'max':>9}{'argmax/N':>10}"
      f"{'r_last/base':>12}{'frac>2xbase':>12}")
print("-" * 79)
for b in BEARINGS:
    r = z[f"{b}__h"]
    n = len(r)
    base = np.median(r[:max(3, int(0.05 * n))])
    last = np.median(r[-3:])
    amax = int(np.argmax(r))
    frac = float(np.mean(r > 2 * base))
    print(f"{b:<13}{n:>6}{base:>8.3f}{last:>9.3f}{r.max():>9.3f}"
          f"{amax/n:>10.3f}{last/base:>12.2f}{frac:>12.3f}")

print("\n--- decile profile of normalized RMS r(tau)/base ---")
print(f"{'bearing':<13}" + "".join(f"{d/10:>7.1f}" for d in range(1, 11)))
print("-" * 83)
for b in BEARINGS:
    r = z[f"{b}__h"]
    n = len(r)
    base = np.median(r[:max(3, int(0.05 * n))])
    q = [np.median(r[int(d * n / 10):int((d + 1) * n / 10)]) / base for d in range(10)]
    print(f"{b:<13}" + "".join(f"{v:>7.2f}" for v in q))
