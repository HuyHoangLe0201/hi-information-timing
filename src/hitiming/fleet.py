"""Comparing health indicators over a fleet of run-to-failure units."""
import numpy as np

from .causal import causal_profile
from .profile import _as_rng, information_profile


def indicator_ages(records, q=0.35, causal=False, rng=None, min_units=4, **kwargs):
    """Information age of every indicator on every unit.

    Parameters
    ----------
    records : dict mapping an indicator name to a list of 1-D arrays, one array
        per unit (the same units, in the same order, for every indicator). Each
        array holds that unit's indicator values from start to failure.
    q : share of the information at which the age is read (default 0.35).
    causal : use the strictly causal estimator instead of the two-sided one.
    rng : seed or Generator for the noise floor. One generator is used for all
        records in order, so a fixed seed gives reproducible results.
    min_units : indicators with fewer usable units get no summary.
    **kwargs : passed to `information_profile` / `causal_profile`.

    Returns
    -------
    dict mapping each indicator to a dict with per-unit arrays `age`, `budget`
    and `surviving_fraction` (NaN where a unit could not be used) and the
    fleet summaries `median_age`, `median_budget`, `median_surviving_fraction`
    and `units` (number of usable units).
    """
    rng = _as_rng(rng)
    make = causal_profile if causal else information_profile
    table = {}
    for name, units in records.items():
        age, bud, surv = [], [], []
        for x in units:
            p = make(x, rng=rng, **kwargs)
            if p is None:
                age.append(np.nan)
                bud.append(np.nan)
                surv.append(np.nan)
                continue
            age.append(p.age(q))
            bud.append(p.budget)
            surv.append(p.surviving_fraction(q))
        age, bud, surv = map(np.asarray, (age, bud, surv))
        ok = np.isfinite(age)
        row = dict(age=age, budget=bud, surviving_fraction=surv, units=int(ok.sum()))
        if ok.sum() >= min_units:
            row.update(median_age=float(np.median(age[ok])),
                       median_budget=float(np.median(bud[ok])),
                       median_surviving_fraction=float(np.median(surv[ok])))
        else:
            row.update(median_age=np.nan, median_budget=np.nan,
                       median_surviving_fraction=np.nan)
        table[name] = row
    return table


def summary(table, threshold=0.90):
    """Rows (indicator, units, median age, median budget, surviving fraction,
    reliable) sorted from the earliest to the latest information age."""
    rows = []
    for name, r in table.items():
        rows.append((name, r["units"], r["median_age"], r["median_budget"],
                     r["median_surviving_fraction"],
                     bool(r["median_surviving_fraction"] >= threshold)))
    return sorted(rows, key=lambda t: (np.isnan(t[2]), t[2]))


def print_summary(table, threshold=0.90):
    print("%-24s %6s %9s %12s %10s  %s" % ("indicator", "units", "age", "budget",
                                           "surviving", "reliable"))
    print("-" * 78)
    for name, u, a, b, s, ok in summary(table, threshold):
        print("%-24s %6d %9.3f %12.4g %10.3f  %s" % (name, u, a, b, s,
                                                    "yes" if ok else "no"))


def _ages_matrix(table_or_ages):
    """Accept an `indicator_ages` table or a dict name -> per-unit ages."""
    out = {}
    for name, v in table_or_ages.items():
        out[name] = np.asarray(v["age"] if isinstance(v, dict) else v, float)
    return out


def _gap(ages, units, early, late):
    def med(name):
        v = ages[name][units]
        v = v[np.isfinite(v)]
        return float(np.median(v)) if len(v) else np.nan

    e = [m for m in (med(n) for n in early) if np.isfinite(m)]
    l_ = [m for m in (med(n) for n in late) if np.isfinite(m)]
    if not e or not l_:
        return np.nan
    return float(np.median(l_) - np.median(e))


def family_gap(ages, early, late):
    """Difference in information age between two groups of indicators.

    The age of each indicator is its median over units; the gap is the median
    over the `late` group minus the median over the `early` group. A positive
    gap means the `early` group reaches the share q of its information first.
    """
    A = _ages_matrix(ages)
    n = len(next(iter(A.values())))
    return _gap(A, np.arange(n), list(early), list(late))


def bootstrap_family_gap(ages, early, late, n_boot=4000, seed=None, level=0.95):
    """Bootstrap intervals for `family_gap`.

    Two intervals are returned. `units` resamples the units only, which covers
    the variation between units of one fleet. `units_and_indicators` also
    resamples the indicators within each group, which covers the choice of
    indicators that make up each group.

    Returns a dict with `gap`, `units` = (low, high), `units_and_indicators` =
    (low, high) and `share_positive` (share of two-level replicates above 0).
    """
    A = _ages_matrix(ages)
    early, late = list(early), list(late)
    # units with no usable age for any indicator carry no information: drop them
    keep = np.any([np.isfinite(v) for v in A.values()], axis=0)
    A = {k: v[keep] for k, v in A.items()}
    n = int(keep.sum())
    rng = _as_rng(seed)
    g = _gap(A, np.arange(n), early, late)
    unit_only, both = [], []
    for _ in range(n_boot):
        u = rng.integers(0, n, n)
        unit_only.append(_gap(A, u, early, late))
        e = [early[i] for i in rng.integers(0, len(early), len(early))]
        l_ = [late[i] for i in rng.integers(0, len(late), len(late))]
        both.append(_gap(A, u, e, l_))
    unit_only = np.array([x for x in unit_only if np.isfinite(x)])
    both = np.array([x for x in both if np.isfinite(x)])
    a = 100 * (1 - level) / 2
    return dict(gap=g,
                units=tuple(np.percentile(unit_only, [a, 100 - a])),
                units_and_indicators=tuple(np.percentile(both, [a, 100 - a])),
                share_positive=float((both > 0).mean()))
