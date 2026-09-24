"""
Making the bootstrap floor affordable.

The lookup does not transfer: real residuals are heavy-tailed and their floor
runs 1.2x to 14x above what Gaussian noise of the same length gives. So the
floor has to be measured per record, and the only question is how cheaply.

Two things are checked. Where the time goes, since the surrogate re-runs the
whole pipeline including the out-of-fold trend; and how many repetitions the
floor estimate actually needs, given that each repetition already contributes n
samples to the median.
"""
import os
import time
import numpy as np
from scipy.signal import savgol_filter
from pipeline import robust_scale, oof_trend, local_scale, _win, LOCAL_BW
from nonparam import weighted_density, estimate_floor

HERE = os.path.dirname(os.path.abspath(__file__))
z = np.load(os.path.join(HERE, "feat_raw.npz"), allow_pickle=True)
FN = [str(s) for s in z["featnames"]]
I = {k: i for i, k in enumerate(FN)}
b = sorted(k for k in z.files if k != "featnames")[0]
x = np.asarray(z[b][:, I["rms"]], float)
n = len(x)
print(f"record length n = {n}\n")

D = (x - np.median(x)) / robust_scale(x)
w = _win(n)

t0 = time.perf_counter(); savgol_filter(D, w, 2, deriv=1, delta=1.0 / n)
t_sg = time.perf_counter() - t0
t0 = time.perf_counter(); res = D - oof_trend(D, w // 2)
t_oof = time.perf_counter() - t0
t0 = time.perf_counter(); local_scale(res, max(5, int(LOCAL_BW * n)))
t_ls = time.perf_counter() - t0
t0 = time.perf_counter(); D - savgol_filter(D, w, 2)
t_plain = time.perf_counter() - t0

print(f"{'step':<28}{'seconds':>10}")
print("-" * 38)
for lab, t in (("savgol derivative", t_sg), ("out-of-fold trend", t_oof),
               ("local scale", t_ls), ("plain savgol residual", t_plain)):
    print(f"{lab:<28}{t:>10.3f}")
print("-" * 38)
print(f"the out-of-fold trend is {t_oof/max(t_plain,1e-9):.0f}x the plain one\n")

# does the floor need the out-of-fold trend at all? the surrogate has no trend
def floor_fast(res, rng, reps):
    n = len(res)
    w = _win(n)
    acc = []
    for _ in range(reps):
        s = res[rng.integers(0, n, n)]
        sc = robust_scale(s)
        if not np.isfinite(sc) or sc <= 0:
            continue
        Dz = (s - np.median(s)) / sc
        dt = savgol_filter(Dz, w, 2, deriv=1, delta=1.0 / n)
        r2 = Dz - savgol_filter(Dz, w, 2)      # no trend to protect against
        sl = np.clip(local_scale(r2, max(5, int(LOCAL_BW * n))), 1e-12, None)
        acc.append((dt / sl) ** 2)
    return float(np.median(np.concatenate(acc))) if acc else np.nan


dens, res = weighted_density(x)
rng = np.random.default_rng(5)
t0 = time.perf_counter()
full = float(estimate_floor(dens, res, "surrogate", rng, reps=8)[0])
t_full = time.perf_counter() - t0
rng = np.random.default_rng(5)
t0 = time.perf_counter()
fast = floor_fast(res, rng, 8)
t_fast = time.perf_counter() - t0
print(f"floor via the full pipeline : {full:>8.2f}   ({t_full:.1f} s, 8 reps)")
print(f"floor via the fast surrogate: {fast:>8.2f}   ({t_fast:.1f} s, 8 reps)")
print(f"agreement {fast/full:.3f}x, speed-up {t_full/max(t_fast,1e-9):.0f}x\n")

print("stability of the fast floor against repetition count\n")
print(f"{'reps':>6}{'median':>10}{'spread over 5 seeds':>22}")
print("-" * 38)
for reps in (2, 4, 8, 16):
    v = [floor_fast(res, np.random.default_rng(s), reps) for s in range(5)]
    v = np.array(v)
    print(f"{reps:>6}{np.median(v):>10.2f}{v.max()/v.min():>21.3f}x")
print("-" * 38)
print("If the spread is already small at few repetitions, the extra ones are")
print("buying nothing: each repetition contributes n samples to the median.")
