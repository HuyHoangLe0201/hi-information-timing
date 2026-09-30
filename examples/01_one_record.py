"""Example 1: the information profile of a single run-to-failure record.

Replace `x` with your own indicator: one value per inspection, equally spaced
in time, from the start of monitoring to failure.

    python 01_one_record.py
"""
import numpy as np

import hitiming as ht

# a synthetic record: a gradual trend plus noise, 500 inspections to failure
rng = np.random.default_rng(0)
n = 500
tau = np.arange(1, n + 1) / n
x = tau ** 1.5 + 0.05 * rng.standard_normal(n)

p = ht.information_profile(x, rng=0)

print("information age tau_0.35      : %.3f of life" % p.age(0.35))
print("information age tau_0.50      : %.3f of life" % p.age(0.50))
print("surviving fraction at tau_0.35: %.3f (reliable if >= 0.90)" % p.surviving_fraction(0.35))
print("information budget            : %.4g" % p.budget)
print("share available if the record stops at 70%% of life: %.3f"
      % p.censoring_efficiency(0.70))
print("shortest window ending at 80%% of life that holds 20%% of the information: %.3f"
      % p.window(0.80, 0.20))
print("causal information age tau_0.35: %.3f of life" % ht.causal_information_age(x, rng=0))

# the whole cumulative curve F(tau), e.g. for plotting
tau_grid, F = p.tau, p.curve
try:
    import matplotlib
    matplotlib.use("Agg")               # write a file; no window needed
    import matplotlib.pyplot as plt
    plt.plot(tau_grid, F)
    plt.axhline(0.35, ls=":", c="grey")
    plt.axvline(p.age(0.35), ls=":", c="grey")
    plt.xlabel("fraction of life, t/T")
    plt.ylabel("cumulative information F")
    plt.title("information age tau_0.35 = %.2f" % p.age(0.35))
    plt.savefig("one_record.png", dpi=150, bbox_inches="tight")
    print("figure written to one_record.png")
except ImportError:
    pass
