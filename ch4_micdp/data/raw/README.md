# Raw data (not included)

The raw data files are not redistributed here. Run `python fetch_data.py` from `ch4_micdp/` to download them into
this folder; the script checks each file against the SHA-256 pinned in `config.yaml` (section `data.files`).

| File | Dataset | Original source | SHA-256 |
|---|---|---|---|
| `adult.data`, `adult.test`, `adult.names` | Adult (Becker and Kohavi, 1996) | UCI Machine Learning Repository, id 2 | see `config.yaml` |
| `diabetes_binary_health_indicators_BRFSS2015.csv` | CDC Diabetes Health Indicators | UCI Machine Learning Repository, id 891 (CDC BRFSS 2015) | see `config.yaml` |
| `cal_housing.data` | California Housing (Pace and Barry, 1997) | StatLib, the file that `sklearn.datasets.fetch_california_housing` downloads | see `config.yaml` |
| `energydata_complete.csv` | Appliances Energy Prediction (Candanedo et al., 2017) | UCI id 374; the authors' repository LuisM78/Appliances-energy-prediction-data | see `config.yaml` |

The download URLs in `config.yaml` are GitHub mirrors of these files; the checksums identify the exact bytes used.
The synthetic datasets are generated from `config.yaml` and need no file.
