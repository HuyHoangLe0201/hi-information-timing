"""Basic behaviour of the package on synthetic records (run with pytest)."""
import os

import numpy as np
import pytest

import hitiming as ht


def record(kind, n=400, noise=0.05, seed=0):
    rng = np.random.default_rng(seed)
    tau = np.arange(1, n + 1) / n
    if kind == "early":
        D = tau ** 1.5
    elif kind == "late":
        D = 3.0 * np.exp(8.0 * (tau - 1.0))
    else:
        D = np.zeros(n)
    return D + noise * rng.standard_normal(n)


def test_profile_curve_is_a_distribution():
    p = ht.information_profile(record("early"), rng=0)
    F = p.curve
    assert np.all(np.diff(F) >= -1e-12)
    assert F[-1] == pytest.approx(1.0)
    assert 0 < p.age(0.35) <= 1
    assert p.censoring_efficiency(1.0) == pytest.approx(1.0)


def test_late_trend_delivers_information_later():
    early = ht.information_age(record("early"), rng=0)
    late = ht.information_age(record("late"), rng=0)
    assert early < late


def test_ages_increase_with_the_share():
    p = ht.information_profile(record("early"), rng=0)
    assert p.age(0.1) <= p.age(0.35) <= p.age(0.9)


def test_noise_only_is_flagged_or_rejected():
    p = ht.information_profile(record("none", noise=0.2), rng=0)
    assert p is None or p.surviving_fraction(0.35) < 0.90


def test_short_record_is_rejected():
    assert ht.information_profile(record("early", n=40), rng=0) is None


def test_same_seed_same_result():
    x = record("early", seed=3)
    assert ht.information_age(x, rng=5) == ht.information_age(x, rng=5)


def test_window_is_positive_and_within_the_age():
    p = ht.information_profile(record("early"), rng=0)
    w = p.window(0.9, 0.2)
    assert 0 < w <= 0.9


def test_causal_age_is_computed():
    a = ht.causal_information_age(record("early"), rng=0)
    assert np.isfinite(a) and 0 < a <= 1


def test_fleet_and_family_gap():
    records = {"early": [record("early", seed=s) for s in range(6)],
               "late": [record("late", seed=10 + s) for s in range(6)]}
    table = ht.indicator_ages(records, rng=0)
    assert table["early"]["units"] == 6
    gap = ht.family_gap(table, early=["early"], late=["late"])
    assert gap > 0
    res = ht.bootstrap_family_gap(table, ["early"], ["late"], n_boot=200, seed=0)
    lo, hi = res["units"]
    assert lo <= res["gap"] <= hi


def test_spectral_indicators_have_one_value_per_acquisition():
    P = np.random.default_rng(0).lognormal(size=(50, 32))
    f = (np.arange(32) + 0.5) * 0.4
    out = ht.spectral.spectral_indicators(P, f)
    assert len(out) == 10 and all(len(v) == 50 for v in out.values())


def test_csv_round_trip(tmp_path):
    for u in range(3):
        x = record("early", seed=u)
        with open(os.path.join(tmp_path, "unit_%d.csv" % u), "w") as fh:
            fh.write("hi\n" + "\n".join("%.6f" % v for v in x) + "\n")
    records, units = ht.load_fleet_csv(str(tmp_path))
    assert list(records) == ["hi"] and len(records["hi"]) == 3 and len(units) == 3
