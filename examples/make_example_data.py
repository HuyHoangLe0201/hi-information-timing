"""Write a small synthetic fleet in the CSV layout the package reads.

Twelve units are run to failure. Each unit has three health indicators:

  early_indicator   changes gradually from early in life (D = tau^1.5)
  late_indicator    stays flat and rises sharply near failure (D = exp(8(tau-1)))
  weak_indicator    a small trend buried in strong noise

One CSV file per unit, one column per indicator, one row per inspection from
the start of monitoring to failure. Record lengths differ between units, as in
real fleets.

    python make_example_data.py      -> data/fleet/unit_01.csv ... unit_12.csv
"""
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "data", "fleet")


def main(n_units=12, seed=1):
    rng = np.random.default_rng(seed)
    os.makedirs(OUT, exist_ok=True)
    for u in range(1, n_units + 1):
        n = int(rng.integers(300, 600))                 # inspections until failure
        tau = np.arange(1, n + 1) / n
        early = tau ** 1.5 + 0.05 * rng.standard_normal(n)
        late = 3.0 * np.exp(8.0 * (tau - 1.0)) + 0.05 * rng.standard_normal(n)
        weak = 0.1 * tau ** 2 + 0.4 * rng.standard_normal(n)
        path = os.path.join(OUT, "unit_%02d.csv" % u)
        with open(path, "w", encoding="utf-8", newline="\n") as fh:
            fh.write("early_indicator,late_indicator,weak_indicator\n")
            for row in zip(early, late, weak):
                fh.write(",".join("%.6f" % v for v in row) + "\n")
    print("wrote %d units to %s" % (n_units, OUT))


if __name__ == "__main__":
    main()
