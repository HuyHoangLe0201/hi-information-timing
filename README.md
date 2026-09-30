# hitiming

**When does a health indicator deliver its information?**

`hitiming` measures how the information that a health indicator carries about
a unit's degradation is distributed over the unit's life. From run-to-failure
records it computes a cumulative Fisher-information profile for each
indicator and summarises it by an *information age*: the fraction of life by
which a given share of the indicator's information has arrived. Indicators
can then be compared, and chosen, without first training a prognostic model
on each of them.

The method is described in

> H. H. Le, K.-A. Nguyen. *A cumulative Fisher-information framework for the
> timing of health-indicator information in prognostics.* Manuscript submitted
> to Mechanical Systems and Signal Processing, 2026.

## Installation

```bash
pip install git+https://github.com/HuyHoangLe0201/hi-information-timing
```

Python 3.9 or later with NumPy and SciPy. Matplotlib is optional (plots in
the examples), and pandas is needed only to rebuild the paper's spectra from
the raw datasets.

## Quick start

```python
import numpy as np
import hitiming as ht

# one unit's health indicator: one value per inspection, start of monitoring to failure
x = np.loadtxt("my_indicator.txt")

p = ht.information_profile(x)
p.age(0.35)                  # information age: fraction of life by which 35% has arrived
p.surviving_fraction(0.35)   # reliability flag: below about 0.9, the age is read from noise
p.budget                     # total information of the record (for comparing indicators)
p.curve                      # cumulative information F(tau) at tau = p.tau
p.censoring_efficiency(0.7)  # share of the information available if the record stops at 70% of life
p.window(0.8, 0.2)           # shortest window ending at 80% of life holding 20% of the information

ht.causal_information_age(x) # the same age, estimated from past samples only
```

## Using it on your own data

**What the data must be.** Each record is one unit's indicator, from the start
of monitoring to failure (or to the end-of-life criterion), sampled at equally
spaced inspections. The indicator can be on any scale: each record is
standardised internally. Records need at least 60 samples; ages from records
shorter than about 300 samples are imprecise, and the causal version needs
longer records still.

**A fleet from CSV files.** Put one CSV file per unit in a folder, with a
header row naming the indicators and one row per inspection:

```
rms,kurtosis,spectral_entropy
0.0213,3.02,4.118
0.0215,2.97,4.121
...
```

```python
records, units = ht.load_fleet_csv("my_fleet/")        # add time_column="t" to sort by time
table = ht.indicator_ages(records, q=0.35, rng=0)
ht.print_summary(table)
```

`print_summary` lists the indicators from the earliest to the latest median
information age, with their median budget and surviving fraction. On the
example fleet (`examples/02_compare_indicators.py`):

```
indicator                 units       age       budget  surviving  reliable
------------------------------------------------------------------------------
weak_indicator               12     0.413     2.79e+04      0.780  no
early_indicator              12     0.562    1.978e+05      0.911  yes
late_indicator               12     0.932     6.26e+06      0.995  yes
```

The weak indicator appears earliest, but its surviving fraction is 0.78: its
age is read mostly from noise and is flagged. Of the two reliable indicators,
the early one delivers its information about 0.37 of a life sooner, although
the late one carries about thirty times more information.

**Comparing groups of indicators.** To test whether one group of indicators
reaches its information earlier than another, use the family gap and its
bootstrap intervals (over units, and over units and indicators together):

```python
res = ht.bootstrap_family_gap(table, early=["spectral_entropy"], late=["rms"], seed=0)
res["gap"], res["units"], res["units_and_indicators"]
```

**Vibration spectra.** `ht.spectral.spectral_indicators(P, f)` computes the
ten indicators used in the paper from band powers `P` (acquisitions x bands):
six that normalise each spectrum to a distribution and four that use its
absolute amplitude.

### Reading the results

* **Information age** `tau_q` is relative to each indicator's own information:
  a small age means the indicator's information arrives early in life. It
  does *not* say that the indicator carries more information; compare
  `budget` for that. An indicator can deliver little information early, or a
  lot of information late.
* **Surviving fraction** is the share of the density up to `tau_q` that
  remains after the noise floor is removed. Ages with a surviving fraction
  below about 0.9 are read largely from noise and should not be compared.
* **Budget** is useful for comparing indicators computed on the same records.
  Its absolute value depends on how the trend is estimated and should not be
  read as a precision.
* **Causal age** uses only past samples, as an operator would. On short
  records it can differ strongly from the two-sided age.
* The default share `q = 0.35` and bandwidth `0.08` follow the paper. Report
  results for a few values of `q`; the ordering of indicators is the robust
  quantity, the size of the differences depends on `q`.

## Method in brief

For a record `x_1..x_n` with normalised age `tau_k = k/n`:

1. The record is standardised, its trend derivative `D'_k` is estimated with
   a quadratic Savitzky-Golay filter (window 8% of the record), and the noise
   scale `sigma_k` with a local robust scale of the out-of-fold residual.
2. The information density is `d_k = (D'_k / sigma_k)^2`. Under a known trend
   and Gaussian noise, `sum_k d_k` is the Fisher information that the record
   carries about the unit's position along its degradation path, and its
   inverse square root is the Cramer-Rao bound for any unbiased estimator of
   that position.
3. The density a trend-free record would show (the noise floor) is estimated
   from resampled residuals and subtracted.
4. The cumulative profile `F(tau) = sum_{k <= tau n} d_k / sum_k d_k` is read
   at a share `q`: `tau_q = F^{-1}(q)`.

The paper derives the framework, the corrections for heavy-tailed and serially
correlated residuals, the effect of an error in the estimated profile, and the
validation on 370 public run-to-failure records.

## API

| function | returns |
|---|---|
| `information_profile(x, bandwidth=0.08, floor="surrogate", rng=None, ...)` | `InformationProfile` or `None` |
| `InformationProfile.age(q)`, `.surviving_fraction(q)`, `.budget`, `.curve`, `.tau`, `.window(tau0, q)`, `.censoring_efficiency(c)` | the design quantities |
| `information_age(x, q)` | `tau_q` of one record |
| `causal_profile(x, ...)`, `causal_information_age(x, q)` | the causal counterparts |
| `indicator_ages(records, q, causal=False, rng=None)` | per-unit ages, budgets and surviving fractions, with fleet medians |
| `summary(table)`, `print_summary(table)` | the fleet table, sorted by age |
| `family_gap(ages, early, late)` | median age of `late` minus median age of `early` |
| `bootstrap_family_gap(ages, early, late, n_boot=4000, seed=None)` | gap with bootstrap intervals |
| `load_fleet_csv(folder, time_column=None)` | `records`, unit names |
| `spectral.spectral_indicators(P, f)`, `spectral.families()` | spectral indicators and their grouping |

## Examples

* `examples/01_one_record.py`: the profile of one record.
* `examples/02_compare_indicators.py`: a fleet read from CSV files
  (`examples/make_example_data.py` writes the example fleet).
* `examples/03_vibration_spectra.py`: spectral indicators from band powers.

## Reproducing the paper's applied result

`reproduce/` rebuilds the vibration spectra of the PRONOSTIA and XJTU-SY
bearing datasets and reproduces the comparison of distributional and
amplitude indicators (Table 5, with its intervals) and its causal version
(Table 8). See `reproduce/README.md`.

## Limitations

* The age axis is `t/T`, so the ages are defined on completed records. For a
  unit in service they describe where the information of similar units
  arrived, not the state of that unit.
* The bound in step 2 holds for a known trend and noise law. With both
  estimated from the record, the profile is used for its shape, and ages
  should be compared between indicators, not read as precisions.
* Systematic variation that is not degradation (changes of operating
  condition, for example) is counted as information. Normalise within
  operating regime before computing the profile.

## Licence

MIT. The datasets used in the paper are not part of this repository and remain
under the terms of their providers.
