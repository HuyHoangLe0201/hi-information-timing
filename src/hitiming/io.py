"""Reading a fleet of run-to-failure records from CSV files."""
import csv
import glob
import os

import numpy as np


def load_fleet_csv(folder, pattern="*.csv", time_column=None):
    """Read one CSV file per unit into the `records` layout used by the package.

    Each file holds one unit, with a header row naming the columns and one row
    per inspection from the start of monitoring to failure. Every column other
    than `time_column` is treated as a health indicator. If `time_column` is
    given, rows are sorted by it; otherwise the file order is used. Files are
    read in sorted order, so unit i is the i-th file for every indicator.

    Returns (records, units): records maps each indicator name to a list of
    1-D arrays (one per unit); units lists the file names.
    """
    files = sorted(glob.glob(os.path.join(folder, pattern)))
    if not files:
        raise FileNotFoundError("no files matching %s in %s" % (pattern, folder))
    records, units, names = {}, [], None
    for path in files:
        with open(path, newline="", encoding="utf-8") as fh:
            rows = list(csv.reader(fh))
        header, body = rows[0], [r for r in rows[1:] if r]
        data = np.array([[float(v) if v.strip() else np.nan for v in r] for r in body])
        if time_column is not None:
            t = header.index(time_column)
            data = data[np.argsort(data[:, t], kind="stable")]
        cols = [c for c in header if c != time_column]
        if names is None:
            names = cols
            records = {c: [] for c in cols}
        elif cols != names:
            raise ValueError("%s has columns %s, expected %s" % (path, cols, names))
        for c in cols:
            records[c].append(data[:, header.index(c)])
        units.append(os.path.basename(path))
    return records, units
