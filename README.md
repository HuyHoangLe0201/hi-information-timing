# Analysis released with the manuscript

This is the code behind *A cumulative Fisher-information framework for the
timing of health-indicator information in prognostics*. It contains every
script that produces a number in the paper, every result file those numbers
are read from, and the audit layers that check the prose against those files.

## What is here, and what is not

| | |
|---|---|
| `*.py` | the analysis, the figure builders, and the audit layers |
| `*.json` | one result file per experiment; the audits compare the paper's sentences against these |
| `*.txt` | ceilings and allowances, each carrying the reason it was set |
| **not here** | `*.npz` feature caches (22 MB) and `.backup/` (superseded drafts) |

The `.npz` files are intermediate arrays derived from five public datasets, not
primary data. They are rebuilt from the datasets by the extraction scripts
below, so shipping them would add 22 MB of something reproducible.

## The datasets

None is redistributed here; each is public and cited in the paper.

- **PRONOSTIA / FEMTO** bearing run-to-failure — Nectoux et al., IEEE ICPHM 2012
- **XJTU-SY** bearing run-to-failure — Wang et al., IEEE Trans. Reliability 2020
- **C-MAPSS FD001, FD004** turbofan — Saxena et al., IEEE ICPHM 2008
- **NASA PCoE battery** — Saha and Goebel, NASA Ames Prognostics Data
  Repository, 2007

## Rebuilding

Put the datasets in one folder and point the environment variable `MSSP_DATA`
at it (the extraction scripts look for `NASABattery/`, `CMAPSSData/`,
`phm-ieee-2012-data-challenge-dataset-master/` and `XJTU-SY bearing dataset/`
inside it), then

```
python band_frequency.py        # 32-band spectra from the raw acquisitions
python attrition.py             # the census behind every count in the paper
python crb_all_domains.py       # the attainment measurement, all four domains
python retention_all.py         # censoring efficiency
```

Seven scripts answer questions a reviewer asked of the manuscript, and each
writes the file the corresponding claim is checked against:

```
python causal_check.py          # the applied result under a causal estimator
python clock_scale.py           # what a damage-clock-indexed noise scale adds
python frozen_split.py          # choices frozen on one set of units, read on another
python record_lengths.py        # the record lengths the causal comparison turns on
python timing_vs_amount.py      # whether tau_q ranks by timing or by amount
python oof_control.py           # the out-of-fold scale purged, against a known scale
python hier_bootstrap.py        # the family gap with units and measures resampled
```

`bandwidth_sweep.py` reads the ten measures of Table 5 and takes about half an hour;
`frozen_split.py` caches its grid in `frozen_grid_*.json`, which is an hour of
CPU; delete those files to recompute it.

Each writes the `.json` its results are read from. `make_fig1.py` and
`make_fig2.py` rebuild the plotted figures; the three schematics are drawn in
the manuscript itself with TikZ, so that a change of notation cannot leave a
figure behind.

## The audits

Thirty-four layers check the paper rather than the code. They are the reason
the numbers in the prose can be trusted: each reads the manuscript and the
supplement and compares what is written against what was computed. Two further
files, `audit_cover.py` and `audit_figures.py`, belong to the predecessor study
and say so in their first line; they fail on a missing file, which is correct.

```
python audit_paper.py           # every number in the prose, against its source
python audit_mathematics.py     # every proposition re-derived independently
python audit_refs.py            # the bibliography, its order, what is unverified
python audit_placement.py       # where each float lands in the typeset PDF
python audit_journal.py         # the journal's own limits: abstract, highlights
python audit_letter.py          # the cover letter, against the paper and the data
```

`audit_paper.py` is the largest: 780 numbers, each bound to the result file that
produced it. A layer that finds nothing prints its verdict and exits zero; one
that finds something exits non-zero and names it.

### Running them from this archive

The layers read the manuscript at `../paper/mssp.tex`, relative to their own
directory, so they need the project layout rather than the layout of this
archive. From the unzipped submission package:

```
mkdir paper && cp submission/source/* paper/
mv code work
cd work && python audit_paper.py
```

From a clone of the GitHub repository, which carries the analysis but not the
manuscript source while the paper is under review, clone into a folder named
`work` and put the manuscript source beside it in `paper/`.

Some layers additionally read the compiled PDF or the `.aux`, so build the
documents first (`pdflatex mssp` twice, `bibtex mssp`, `pdflatex mssp` twice,
then the same for `supplement`) if you want `audit_placement.py` and the
frontmatter checks to run.

The state everything is expected to be in: both documents compiling with 0
errors, 0 overfull boxes and 0 undefined references, and all thirty-four layers
exiting zero.

### What it needs

Python 3 with `numpy`, `scipy` and `pandas`; `PyMuPDF` for the layers that read the typeset PDF;
`matplotlib` for the figure builders. TeX Live with `elsarticle`, `tikz` and
`xr` for the documents.

## A note on the ceilings

Several layers hold a count against a ceiling in a `.txt` file rather than
requiring zero — `conservation_ceiling.txt`, `symbols_ceiling.txt` and others.
Each ceiling file carries, in prose, the reason every unit under it is there.
They are not thresholds tuned until the layer went quiet; raising one is meant
to be a decision taken against a printed list, and the files record those
decisions.
