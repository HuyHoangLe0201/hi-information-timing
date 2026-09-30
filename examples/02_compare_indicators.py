"""Example 2: compare candidate health indicators over a fleet, from CSV files.

Put one CSV file per unit in a folder, with a header row naming the indicators
and one row per inspection from the start of monitoring to failure (see
data/fleet/ for the layout; make_example_data.py writes it). Then run

    python 02_compare_indicators.py [folder]

The table lists each indicator's median information age over the fleet (the
fraction of life by which 35% of its information has arrived), its median
information budget, and its surviving fraction, which flags ages that are read
mostly from noise.
"""
import os
import sys

import hitiming as ht

HERE = os.path.dirname(os.path.abspath(__file__))
folder = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "data", "fleet")
if not os.path.isdir(folder):
    sys.exit("no data in %s; run make_example_data.py first" % folder)

records, units = ht.load_fleet_csv(folder)
print("%d units, indicators: %s\n" % (len(units), ", ".join(records)))

table = ht.indicator_ages(records, q=0.35, rng=0)
ht.print_summary(table)

# Is one group of indicators earlier than another? Here the groups are single
# indicators; with several indicators per group the median over each group is used.
res = ht.bootstrap_family_gap(table, early=["early_indicator"], late=["late_indicator"],
                              n_boot=2000, seed=0)
print("\nage(late_indicator) - age(early_indicator) = %.3f of life" % res["gap"])
print("95%% interval over units: [%.3f, %.3f]" % res["units"])
