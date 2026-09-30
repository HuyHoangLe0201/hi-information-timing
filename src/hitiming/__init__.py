"""hitiming: when does a health indicator deliver its information?

Cumulative Fisher-information profiles of run-to-failure records, the
information age of an indicator, and comparisons of indicators over a fleet.

    import hitiming as ht
    p = ht.information_profile(x)        # x: one unit's indicator, start to failure
    p.age(0.35)                          # fraction of life by which 35% has arrived
"""
from .causal import causal_information_age, causal_profile
from .fleet import (bootstrap_family_gap, family_gap, indicator_ages,
                    print_summary, summary)
from .io import load_fleet_csv
from .profile import InformationProfile, information_age, information_profile
from . import spectral

__version__ = "1.0.0"

__all__ = [
    "InformationProfile", "information_profile", "information_age",
    "causal_profile", "causal_information_age",
    "indicator_ages", "summary", "print_summary",
    "family_gap", "bootstrap_family_gap",
    "load_fleet_csv", "spectral",
]
