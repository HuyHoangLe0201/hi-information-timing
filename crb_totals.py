r"""The corpus behind the attainment table, counted at all three levels.

Table 3 now carries a Units column and a totals row, so that 102 464 evaluation
cells cannot be read as 102 464 independent observations: they come from 1 068
unit-channel series, which come from 370 physical units, and the battery's 915
cells come from four cells and no more.

The totals are sums over stored results, and a sum is not itself a stored
result -- audit_orphans.py reported both as numbers the repository could not
account for the moment they were printed.  That layer may be satisfied by
accounting for a number and never by raising its ceiling, so the totals are
computed here and written where everything else is checked from.

    python crb_totals.py   ->  crb_totals.json
"""
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
L = lambda f: json.load(open(os.path.join(HERE, f)))

C = L("crb_full.json")
_ATU = {r["source"]: r["entering"] for r in L("attrition.json")["sources"]}

# Physical units per domain.  Three are in the census; the battery's four cells
# are a property of the NASA PCoE selection, and the check below is that they
# carry three channels each, which is what makes twelve series.
units = {"bearing": _ATU["PRONOSTIA, ten channels"],
         "turbofan_FD001": _ATU["Turbofan FD001"],
         "turbofan_FD004": _ATU["Turbofan FD004"],
         "battery": 4}

out = {
    "per_domain": {k: {"units": units[k],
                       "series": C[k]["series"],
                       "cells": C[k]["cells"]} for k in units},
    "total_units": sum(units.values()),
    "total_series": sum(C[k]["series"] for k in units),
    "total_cells": sum(C[k]["cells"] for k in units),
    "cells_per_series": sum(C[k]["cells"] for k in units)
                        / float(sum(C[k]["series"] for k in units)),
    "battery_channels_per_cell": C["battery"]["series"] / float(units["battery"]),
}
json.dump(out, open(os.path.join(HERE, "crb_totals.json"), "w"), indent=1)

print("%-26s %6s %7s %9s" % ("domain", "units", "series", "cells"))
print("-" * 52)
for k, v in out["per_domain"].items():
    print("%-26s %6d %7d %9d" % (k, v["units"], v["series"], v["cells"]))
print("-" * 52)
print("%-26s %6d %7d %9d" % ("total", out["total_units"], out["total_series"],
                             out["total_cells"]))
print("\n%.1f cells per series; the battery's %d cells carry %.0f channels each."
      % (out["cells_per_series"], units["battery"],
         out["battery_channels_per_cell"]))
print("written to crb_totals.json")
