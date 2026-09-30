# Reproducing the applied result of the paper

These scripts reproduce the paper's comparison of distributional and
amplitude vibration indicators with the `hitiming` package.

## 1. Get the datasets

* **PRONOSTIA** (FEMTO-ST, IEEE PHM 2012 Prognostic Challenge): the folder
  containing `Learning_set/` and `Full_Test_Set/`.
* **XJTU-SY** bearing datasets (Wang et al., IEEE Trans. Reliability, 2020):
  the folder containing `35Hz12kN/`, `37.5Hz11kN/` and `40Hz10kN/`.

## 2. Build the spectra

```bash
pip install pandas
python prepare_spectra.py pronostia /path/to/PRONOSTIA     # -> spectra_pronostia.npz
python prepare_spectra.py xjtu /path/to/XJTU-SY_Bearing_Datasets   # -> spectra_xjtu.npz
```

Each acquisition of the horizontal accelerometer becomes 32 band powers of
400 Hz. PRONOSTIA gives 16 bearings (Bearing1_4 uses a different file
separator and is skipped), XJTU-SY 15, of which 13 are long enough.

## 3. Run

| script | reproduces | runtime |
|---|---|---|
| `family_gap.py` | Table 5: information age of ten indicators on both rigs, the family gap (0.640 and 0.358) and its bootstrap intervals | ~10 min |
| `causal.py` | Table 8: the family gap under the two-sided and the strictly causal estimator (0.641 to 0.563; 0.357 to 0.029) | ~30 min on 10 cores |

Both scripts print the published values beside the recomputed ones, and they
agree to the last printed digit. The random draws (noise floor, bootstrap) use
the seeds of the paper's analysis.

Set `HITIMING_SPECTRA=/path/to/folder` if the `.npz` files are elsewhere.
