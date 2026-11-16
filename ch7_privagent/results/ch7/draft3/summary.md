# Chapter 7: misspecified simulated users (EXPLORATORY, Draft 3 follow-up)

Every result in this file and in results/ch7/draft3/ is exploratory. It is not part of the pre-registered run (commit 7bffabb), whose configuration, code and outputs (results/ch7/) are unchanged. Produced by `python -m src.misspec && python -m src.misspec_report`.

## Commits

- The archived snapshot of the pre-registered repository had no git history; it was restored as commit `dc16b0eb85c527a62b5d73f775e6d9f1e82fbdaf` (files as archived; results/ch7/summary.md records the pre-registration commit 7bffabb).
- Settings of this run, committed before the run: `4002e0e1be5fc1af284ec28b553c8dfb06821283` (config_draft3.yaml).
- Simulation code and tests, before the run: `61af0cdc216deb525ed02f5a25f564c7f0fe5fe0`; `2a112d5427862ad1e6c948addacca1750eff5afa` changes only how the worker function is passed to joblib (the first launch stopped before any replication ran).

## Design (config_draft3.yaml)

Uniform true weights (Dirichlet(1, 1, 1)), 1,000 users x 20 replications per population. Every population uses the streams of the stored uniform population for the true weights, prompt orders, answer draws, repair draws, slider noise and prior draws, so the populations share the same users and draws and differ only in the response rule; the Bayesian inference keeps the analyst temperature 0.1. Response rules: lapse, each answer replaced by a fair coin with probability p; privacy-always, exactly round(f x 1,000) users per replication choose the LP option in every LP-LA and LP-CR prompt and follow the logistic rule otherwise; side bias, +b on the log-odds of the option shown first; lexicographic, the option better on the highest-weighted criterion (in these prompts the third criterion is equal in both options, so the rule chooses A iff w_j > w_k, which is the tau -> 0 limit of the logistic rule).

Side assignment: the stored design (src/dialogue.py) has no screen position and does not alternate the privacy option. Option A always favors the first criterion of the pair (LP in LP-LA and LP-CR, LA in LA-CR). Two assignments were therefore run: fixed (A shown first in every prompt, the stored design read literally: the bias always favors LP, or LA in LA-CR) and alternating (A shown first at presented positions 1, 3, 5 and the repair prompt, second at 2, 4, 6, in each user's random order).

The expert default and the anchored slider do not use the dialogue and are the same in every population.

## Baseline check

```
PASS baseline: estimated weights equal users.parquet (uniform, main) for 120000 of 120000 user-method rows (max abs difference 1.8e-15)
PASS baseline: repair flags equal users.parquet
PASS baseline: decisions (adult, linear) equal decisions.parquet for 120000 of 120000 rows
PASS baseline: decisions (cdc, linear) equal decisions.parquet for 120000 of 120000 rows
INFO share of privacy-always users: privacy_0.1: 0.100, privacy_0.25: 0.250
INFO repair rate (Vote-6 dialogue): baseline: 0.0036, lapse_0.1: 0.0067, lapse_0.2: 0.0104, lapse_0.33: 0.0149, lexicographic: 0.0000, privacy_0.1: 0.0033, privacy_0.25: 0.0027, side_alt_0.5: 0.0044, side_alt_1.0: 0.0053, side_fixed_0.5: 0.0044, side_fixed_1.0: 0.0054
```

Pooled metrics of the baseline population against results/ch7/metrics.csv (uniform, main; largest absolute difference 1.1e-16):

| method | metric | stored | this run | abs difference |
|---|---|---|---|---|
| Vote-6 | share_error | 0.2118 | 0.2118 | 0.0e+00 |
| Vote-6 | cosine | 0.9299 | 0.9299 | 1.1e-16 |
| Vote-6 | agree_adult | 0.3257 | 0.3257 | 0.0e+00 |
| Vote-6 | loss_adult | 0.0835 | 0.0835 | 1.4e-17 |
| Vote-6 | agree_cdc | 0.3194 | 0.3194 | 0.0e+00 |
| Vote-6 | loss_cdc | 0.0866 | 0.0866 | 6.9e-17 |
| Bayes-6 | share_error | 0.1259 | 0.1259 | 2.8e-17 |
| Bayes-6 | cosine | 0.9615 | 0.9615 | 1.1e-16 |
| Bayes-6 | agree_adult | 0.7448 | 0.7448 | 0.0e+00 |
| Bayes-6 | loss_adult | 0.0233 | 0.0233 | 1.7e-17 |
| Bayes-6 | agree_cdc | 0.6956 | 0.6956 | 0.0e+00 |
| Bayes-6 | loss_cdc | 0.0226 | 0.0226 | 4.2e-17 |
| Vote-LPLA | share_error | 0.2279 | 0.2279 | 8.3e-17 |
| Vote-LPLA | cosine | 0.8499 | 0.8499 | 0.0e+00 |
| Vote-LPLA | agree_adult | 0.1755 | 0.1755 | 0.0e+00 |
| Vote-LPLA | loss_adult | 0.0880 | 0.0880 | 1.4e-17 |
| Vote-LPLA | agree_cdc | 0.1701 | 0.1701 | 0.0e+00 |
| Vote-LPLA | loss_cdc | 0.0938 | 0.0938 | 5.6e-17 |
| PrivAgent | share_error | 0.1134 | 0.1134 | 5.6e-17 |
| PrivAgent | cosine | 0.9082 | 0.9082 | 1.1e-16 |
| PrivAgent | agree_adult | 0.7722 | 0.7722 | 0.0e+00 |
| PrivAgent | loss_adult | 0.0175 | 0.0175 | 3.8e-17 |
| PrivAgent | agree_cdc | 0.7170 | 0.7170 | 0.0e+00 |
| PrivAgent | loss_cdc | 0.0176 | 0.0176 | 8.7e-17 |
| Expert default | share_error | 0.2512 | 0.2512 | 0.0e+00 |
| Expert default | cosine | 0.8030 | 0.8030 | 0.0e+00 |
| Expert default | agree_adult | 0.4105 | 0.4105 | 0.0e+00 |
| Expert default | loss_adult | 0.1212 | 0.1212 | 8.3e-17 |
| Expert default | agree_cdc | 0.3816 | 0.3816 | 0.0e+00 |
| Expert default | loss_cdc | 0.1032 | 0.1032 | 2.8e-17 |
| Anchored slider | share_error | 0.1158 | 0.1158 | 0.0e+00 |
| Anchored slider | cosine | 0.9719 | 0.9719 | 1.1e-16 |
| Anchored slider | agree_adult | 0.7728 | 0.7728 | 0.0e+00 |
| Anchored slider | loss_adult | 0.0163 | 0.0163 | 2.4e-17 |
| Anchored slider | agree_cdc | 0.7251 | 0.7251 | 0.0e+00 |
| Anchored slider | loss_cdc | 0.0159 | 0.0159 | 4.2e-17 |

## Results (exploratory)

Pooled means over all users of all replications with 95% cluster-bootstrap intervals over replications. Linear LP map. Loss = closeness loss C_true(eps*_true) - C_true(eps*_method).

### All metrics

| population | method | Share error | Cosine | Agreement, Adult | Loss, Adult | Agreement, CDC-Diabetes | Loss, CDC-Diabetes |
|---|---|---|---|---|---|---|---|
| Baseline (logistic, tau = 0.1) | Vote-6 | 0.212 [0.209, 0.215] | 0.930 [0.929, 0.931] | 0.326 [0.319, 0.332] | 0.083 [0.082, 0.085] | 0.319 [0.313, 0.326] | 0.087 [0.085, 0.088] |
| Baseline (logistic, tau = 0.1) | Bayes-6 | 0.126 [0.124, 0.127] | 0.961 [0.961, 0.962] | 0.745 [0.737, 0.753] | 0.023 [0.023, 0.024] | 0.696 [0.687, 0.704] | 0.023 [0.022, 0.023] |
| Baseline (logistic, tau = 0.1) | Vote-LPLA | 0.228 [0.225, 0.230] | 0.850 [0.848, 0.852] | 0.175 [0.169, 0.181] | 0.088 [0.087, 0.089] | 0.170 [0.163, 0.176] | 0.094 [0.092, 0.095] |
| Baseline (logistic, tau = 0.1) | PrivAgent | 0.113 [0.112, 0.115] | 0.908 [0.907, 0.909] | 0.772 [0.765, 0.779] | 0.018 [0.017, 0.018] | 0.717 [0.711, 0.723] | 0.018 [0.017, 0.018] |
| Baseline (logistic, tau = 0.1) | Expert default | 0.251 [0.249, 0.253] | 0.803 [0.801, 0.805] | 0.410 [0.404, 0.417] | 0.121 [0.119, 0.123] | 0.382 [0.375, 0.388] | 0.103 [0.101, 0.105] |
| Baseline (logistic, tau = 0.1) | Anchored slider | 0.116 [0.114, 0.117] | 0.972 [0.972, 0.972] | 0.773 [0.766, 0.779] | 0.016 [0.016, 0.017] | 0.725 [0.720, 0.731] | 0.016 [0.015, 0.017] |
| Lapse p = 0.1 | Vote-6 | 0.222 [0.219, 0.225] | 0.916 [0.915, 0.917] | 0.332 [0.324, 0.340] | 0.090 [0.088, 0.091] | 0.324 [0.315, 0.332] | 0.093 [0.091, 0.094] |
| Lapse p = 0.1 | Bayes-6 | 0.141 [0.139, 0.142] | 0.947 [0.947, 0.948] | 0.724 [0.716, 0.732] | 0.030 [0.029, 0.031] | 0.671 [0.663, 0.679] | 0.029 [0.028, 0.029] |
| Lapse p = 0.1 | Vote-LPLA | 0.231 [0.229, 0.234] | 0.848 [0.846, 0.850] | 0.189 [0.182, 0.195] | 0.091 [0.089, 0.092] | 0.182 [0.176, 0.189] | 0.096 [0.095, 0.098] |
| Lapse p = 0.1 | PrivAgent | 0.127 [0.126, 0.129] | 0.899 [0.898, 0.900] | 0.755 [0.748, 0.762] | 0.022 [0.021, 0.023] | 0.701 [0.694, 0.708] | 0.021 [0.020, 0.022] |
| Lapse p = 0.1 | Expert default | 0.251 [0.249, 0.253] | 0.803 [0.801, 0.805] | 0.410 [0.404, 0.417] | 0.121 [0.119, 0.123] | 0.382 [0.375, 0.388] | 0.103 [0.101, 0.105] |
| Lapse p = 0.1 | Anchored slider | 0.116 [0.115, 0.117] | 0.972 [0.972, 0.972] | 0.773 [0.766, 0.779] | 0.016 [0.016, 0.017] | 0.725 [0.720, 0.731] | 0.016 [0.015, 0.017] |
| Lapse p = 0.2 | Vote-6 | 0.233 [0.230, 0.236] | 0.899 [0.898, 0.901] | 0.337 [0.330, 0.345] | 0.097 [0.095, 0.099] | 0.328 [0.320, 0.336] | 0.100 [0.098, 0.102] |
| Lapse p = 0.2 | Bayes-6 | 0.155 [0.153, 0.157] | 0.932 [0.931, 0.933] | 0.699 [0.690, 0.708] | 0.038 [0.036, 0.039] | 0.644 [0.636, 0.653] | 0.036 [0.035, 0.037] |
| Lapse p = 0.2 | Vote-LPLA | 0.236 [0.234, 0.239] | 0.844 [0.842, 0.846] | 0.206 [0.199, 0.212] | 0.095 [0.093, 0.097] | 0.197 [0.190, 0.204] | 0.100 [0.098, 0.101] |
| Lapse p = 0.2 | PrivAgent | 0.141 [0.140, 0.143] | 0.890 [0.888, 0.891] | 0.735 [0.729, 0.742] | 0.027 [0.026, 0.028] | 0.681 [0.676, 0.687] | 0.026 [0.025, 0.027] |
| Lapse p = 0.2 | Expert default | 0.251 [0.249, 0.253] | 0.803 [0.801, 0.805] | 0.410 [0.404, 0.417] | 0.121 [0.119, 0.123] | 0.382 [0.375, 0.388] | 0.103 [0.101, 0.105] |
| Lapse p = 0.2 | Anchored slider | 0.116 [0.115, 0.117] | 0.972 [0.972, 0.972] | 0.773 [0.766, 0.779] | 0.016 [0.016, 0.017] | 0.725 [0.720, 0.731] | 0.016 [0.015, 0.017] |
| Lapse p = 0.33 | Vote-6 | 0.252 [0.249, 0.254] | 0.872 [0.871, 0.874] | 0.337 [0.330, 0.344] | 0.111 [0.109, 0.112] | 0.323 [0.316, 0.330] | 0.112 [0.110, 0.114] |
| Lapse p = 0.33 | Bayes-6 | 0.175 [0.173, 0.177] | 0.910 [0.909, 0.912] | 0.663 [0.654, 0.672] | 0.049 [0.048, 0.051] | 0.610 [0.601, 0.620] | 0.047 [0.046, 0.048] |
| Lapse p = 0.33 | Vote-LPLA | 0.248 [0.246, 0.251] | 0.834 [0.833, 0.836] | 0.222 [0.217, 0.228] | 0.105 [0.103, 0.107] | 0.211 [0.205, 0.215] | 0.109 [0.107, 0.111] |
| Lapse p = 0.33 | PrivAgent | 0.161 [0.160, 0.163] | 0.876 [0.875, 0.878] | 0.700 [0.693, 0.707] | 0.037 [0.036, 0.039] | 0.648 [0.642, 0.655] | 0.035 [0.034, 0.036] |
| Lapse p = 0.33 | Expert default | 0.251 [0.249, 0.253] | 0.803 [0.801, 0.805] | 0.410 [0.404, 0.417] | 0.121 [0.119, 0.123] | 0.382 [0.375, 0.388] | 0.103 [0.101, 0.105] |
| Lapse p = 0.33 | Anchored slider | 0.116 [0.115, 0.117] | 0.972 [0.972, 0.972] | 0.773 [0.766, 0.779] | 0.016 [0.016, 0.017] | 0.725 [0.720, 0.731] | 0.016 [0.015, 0.017] |
| Privacy-always f = 0.1 | Vote-6 | 0.230 [0.228, 0.233] | 0.901 [0.900, 0.903] | 0.310 [0.303, 0.316] | 0.097 [0.096, 0.099] | 0.302 [0.295, 0.308] | 0.100 [0.098, 0.102] |
| Privacy-always f = 0.1 | Bayes-6 | 0.146 [0.144, 0.147] | 0.938 [0.937, 0.939] | 0.712 [0.705, 0.718] | 0.033 [0.032, 0.034] | 0.654 [0.646, 0.663] | 0.034 [0.033, 0.035] |
| Privacy-always f = 0.1 | Vote-LPLA | 0.255 [0.252, 0.257] | 0.824 [0.822, 0.826] | 0.162 [0.156, 0.168] | 0.113 [0.111, 0.115] | 0.155 [0.148, 0.160] | 0.122 [0.120, 0.123] |
| Privacy-always f = 0.1 | PrivAgent | 0.138 [0.137, 0.139] | 0.887 [0.886, 0.888] | 0.737 [0.731, 0.744] | 0.027 [0.027, 0.028] | 0.684 [0.679, 0.689] | 0.026 [0.025, 0.027] |
| Privacy-always f = 0.1 | Expert default | 0.251 [0.249, 0.253] | 0.803 [0.801, 0.805] | 0.410 [0.404, 0.417] | 0.121 [0.119, 0.123] | 0.382 [0.375, 0.388] | 0.103 [0.101, 0.105] |
| Privacy-always f = 0.1 | Anchored slider | 0.116 [0.114, 0.117] | 0.972 [0.972, 0.972] | 0.773 [0.766, 0.779] | 0.016 [0.016, 0.017] | 0.725 [0.720, 0.731] | 0.016 [0.015, 0.017] |
| Privacy-always f = 0.25 | Vote-6 | 0.261 [0.258, 0.264] | 0.859 [0.858, 0.861] | 0.282 [0.277, 0.288] | 0.120 [0.118, 0.122] | 0.272 [0.266, 0.278] | 0.120 [0.118, 0.122] |
| Privacy-always f = 0.25 | Bayes-6 | 0.175 [0.173, 0.177] | 0.904 [0.903, 0.906] | 0.661 [0.654, 0.667] | 0.048 [0.046, 0.049] | 0.594 [0.587, 0.601] | 0.051 [0.050, 0.052] |
| Privacy-always f = 0.25 | Vote-LPLA | 0.296 [0.293, 0.299] | 0.784 [0.781, 0.787] | 0.145 [0.141, 0.150] | 0.151 [0.148, 0.153] | 0.133 [0.128, 0.138] | 0.164 [0.161, 0.166] |
| Privacy-always f = 0.25 | PrivAgent | 0.176 [0.174, 0.178] | 0.856 [0.855, 0.858] | 0.680 [0.675, 0.686] | 0.043 [0.042, 0.044] | 0.632 [0.627, 0.637] | 0.039 [0.038, 0.040] |
| Privacy-always f = 0.25 | Expert default | 0.251 [0.249, 0.253] | 0.803 [0.801, 0.805] | 0.410 [0.404, 0.417] | 0.121 [0.119, 0.123] | 0.382 [0.375, 0.388] | 0.103 [0.101, 0.105] |
| Privacy-always f = 0.25 | Anchored slider | 0.116 [0.115, 0.117] | 0.972 [0.972, 0.972] | 0.773 [0.766, 0.780] | 0.016 [0.016, 0.017] | 0.725 [0.720, 0.731] | 0.016 [0.015, 0.017] |
| Side bias b = 0.5 (fixed) | Vote-6 | 0.203 [0.200, 0.205] | 0.926 [0.925, 0.927] | 0.379 [0.373, 0.386] | 0.077 [0.075, 0.079] | 0.368 [0.361, 0.376] | 0.081 [0.079, 0.083] |
| Side bias b = 0.5 (fixed) | Bayes-6 | 0.130 [0.128, 0.131] | 0.958 [0.958, 0.959] | 0.743 [0.737, 0.750] | 0.023 [0.022, 0.024] | 0.691 [0.684, 0.699] | 0.022 [0.022, 0.023] |
| Side bias b = 0.5 (fixed) | Vote-LPLA | 0.235 [0.233, 0.237] | 0.845 [0.843, 0.847] | 0.168 [0.164, 0.171] | 0.093 [0.091, 0.094] | 0.160 [0.155, 0.165] | 0.103 [0.101, 0.104] |
| Side bias b = 0.5 (fixed) | PrivAgent | 0.121 [0.119, 0.122] | 0.905 [0.903, 0.906] | 0.755 [0.749, 0.762] | 0.020 [0.019, 0.021] | 0.700 [0.694, 0.707] | 0.020 [0.019, 0.021] |
| Side bias b = 0.5 (fixed) | Expert default | 0.251 [0.249, 0.253] | 0.803 [0.801, 0.805] | 0.410 [0.404, 0.417] | 0.121 [0.119, 0.123] | 0.382 [0.375, 0.388] | 0.103 [0.101, 0.105] |
| Side bias b = 0.5 (fixed) | Anchored slider | 0.116 [0.115, 0.117] | 0.972 [0.972, 0.972] | 0.773 [0.766, 0.780] | 0.016 [0.016, 0.017] | 0.725 [0.720, 0.731] | 0.016 [0.015, 0.017] |
| Side bias b = 1.0 (fixed) | Vote-6 | 0.200 [0.198, 0.202] | 0.915 [0.913, 0.916] | 0.424 [0.417, 0.430] | 0.074 [0.072, 0.076] | 0.409 [0.402, 0.416] | 0.079 [0.077, 0.080] |
| Side bias b = 1.0 (fixed) | Bayes-6 | 0.138 [0.137, 0.140] | 0.949 [0.949, 0.950] | 0.732 [0.727, 0.738] | 0.025 [0.024, 0.026] | 0.679 [0.674, 0.685] | 0.024 [0.023, 0.024] |
| Side bias b = 1.0 (fixed) | Vote-LPLA | 0.256 [0.253, 0.259] | 0.829 [0.826, 0.832] | 0.160 [0.155, 0.164] | 0.108 [0.106, 0.110] | 0.149 [0.146, 0.154] | 0.122 [0.119, 0.124] |
| Side bias b = 1.0 (fixed) | PrivAgent | 0.140 [0.138, 0.142] | 0.894 [0.892, 0.895] | 0.715 [0.709, 0.720] | 0.029 [0.028, 0.029] | 0.660 [0.655, 0.666] | 0.026 [0.026, 0.027] |
| Side bias b = 1.0 (fixed) | Expert default | 0.251 [0.249, 0.253] | 0.803 [0.801, 0.805] | 0.410 [0.404, 0.417] | 0.121 [0.119, 0.123] | 0.382 [0.375, 0.388] | 0.103 [0.101, 0.105] |
| Side bias b = 1.0 (fixed) | Anchored slider | 0.116 [0.115, 0.117] | 0.972 [0.972, 0.972] | 0.773 [0.766, 0.780] | 0.016 [0.016, 0.017] | 0.725 [0.720, 0.731] | 0.016 [0.015, 0.017] |
| Side bias b = 0.5 (alternating) | Vote-6 | 0.213 [0.210, 0.216] | 0.929 [0.927, 0.929] | 0.330 [0.324, 0.336] | 0.085 [0.083, 0.086] | 0.321 [0.313, 0.330] | 0.088 [0.086, 0.089] |
| Side bias b = 0.5 (alternating) | Bayes-6 | 0.128 [0.126, 0.130] | 0.960 [0.960, 0.961] | 0.741 [0.734, 0.748] | 0.024 [0.023, 0.025] | 0.692 [0.684, 0.700] | 0.023 [0.023, 0.024] |
| Side bias b = 0.5 (alternating) | Vote-LPLA | 0.227 [0.225, 0.229] | 0.851 [0.849, 0.853] | 0.179 [0.172, 0.185] | 0.088 [0.087, 0.089] | 0.173 [0.166, 0.179] | 0.093 [0.092, 0.095] |
| Side bias b = 0.5 (alternating) | PrivAgent | 0.115 [0.113, 0.116] | 0.909 [0.907, 0.910] | 0.769 [0.763, 0.776] | 0.018 [0.017, 0.019] | 0.714 [0.708, 0.720] | 0.018 [0.017, 0.019] |
| Side bias b = 0.5 (alternating) | Expert default | 0.251 [0.249, 0.253] | 0.803 [0.801, 0.805] | 0.410 [0.404, 0.417] | 0.121 [0.119, 0.123] | 0.382 [0.375, 0.388] | 0.103 [0.101, 0.105] |
| Side bias b = 0.5 (alternating) | Anchored slider | 0.116 [0.114, 0.117] | 0.972 [0.972, 0.972] | 0.773 [0.766, 0.780] | 0.016 [0.016, 0.017] | 0.725 [0.720, 0.731] | 0.016 [0.015, 0.017] |
| Side bias b = 1.0 (alternating) | Vote-6 | 0.218 [0.215, 0.221] | 0.926 [0.924, 0.927] | 0.332 [0.325, 0.340] | 0.089 [0.086, 0.091] | 0.324 [0.315, 0.333] | 0.092 [0.089, 0.094] |
| Side bias b = 1.0 (alternating) | Bayes-6 | 0.134 [0.132, 0.135] | 0.957 [0.957, 0.958] | 0.725 [0.718, 0.731] | 0.028 [0.027, 0.029] | 0.675 [0.669, 0.683] | 0.027 [0.026, 0.028] |
| Side bias b = 1.0 (alternating) | Vote-LPLA | 0.224 [0.222, 0.226] | 0.855 [0.853, 0.858] | 0.197 [0.191, 0.204] | 0.087 [0.085, 0.088] | 0.191 [0.184, 0.197] | 0.092 [0.090, 0.093] |
| Side bias b = 1.0 (alternating) | PrivAgent | 0.118 [0.116, 0.120] | 0.908 [0.907, 0.910] | 0.759 [0.751, 0.766] | 0.020 [0.019, 0.021] | 0.703 [0.696, 0.711] | 0.020 [0.019, 0.021] |
| Side bias b = 1.0 (alternating) | Expert default | 0.251 [0.249, 0.253] | 0.803 [0.801, 0.805] | 0.410 [0.404, 0.417] | 0.121 [0.119, 0.123] | 0.382 [0.375, 0.388] | 0.103 [0.101, 0.105] |
| Side bias b = 1.0 (alternating) | Anchored slider | 0.116 [0.115, 0.117] | 0.972 [0.972, 0.972] | 0.773 [0.766, 0.780] | 0.016 [0.016, 0.017] | 0.725 [0.720, 0.731] | 0.016 [0.015, 0.017] |
| Lexicographic | Vote-6 | 0.182 [0.180, 0.184] | 0.948 [0.947, 0.948] | 0.375 [0.369, 0.381] | 0.056 [0.055, 0.056] | 0.365 [0.357, 0.372] | 0.061 [0.060, 0.062] |
| Lexicographic | Bayes-6 | 0.109 [0.108, 0.110] | 0.968 [0.968, 0.968] | 0.848 [0.842, 0.854] | 0.005 [0.005, 0.006] | 0.784 [0.777, 0.792] | 0.007 [0.006, 0.007] |
| Lexicographic | Vote-LPLA | 0.249 [0.247, 0.251] | 0.808 [0.805, 0.811] | 0.067 [0.063, 0.070] | 0.098 [0.097, 0.099] | 0.063 [0.059, 0.066] | 0.107 [0.106, 0.108] |
| Lexicographic | PrivAgent | 0.140 [0.138, 0.141] | 0.869 [0.867, 0.872] | 0.848 [0.842, 0.854] | 0.005 [0.005, 0.006] | 0.792 [0.786, 0.798] | 0.007 [0.006, 0.007] |
| Lexicographic | Expert default | 0.251 [0.249, 0.253] | 0.803 [0.801, 0.805] | 0.410 [0.404, 0.417] | 0.121 [0.119, 0.123] | 0.382 [0.375, 0.388] | 0.103 [0.101, 0.105] |
| Lexicographic | Anchored slider | 0.116 [0.115, 0.117] | 0.972 [0.972, 0.972] | 0.773 [0.766, 0.780] | 0.016 [0.016, 0.017] | 0.725 [0.720, 0.731] | 0.016 [0.015, 0.017] |

### Privacy-always populations: affected and unaffected users

The metrics of the affected users compare the estimates with the weights drawn for them, which their answers to the LP prompts ignore.

| population | users | method | Share error | Cosine | Agreement, Adult | Loss, Adult | Agreement, CDC-Diabetes | Loss, CDC-Diabetes |
|---|---|---|---|---|---|---|---|---|
| Privacy-always f = 0.1 | affected | Vote-6 | 0.412 [0.403, 0.420] | 0.642 [0.633, 0.652] | 0.149 [0.139, 0.161] | 0.230 [0.221, 0.239] | 0.133 [0.123, 0.142] | 0.225 [0.215, 0.235] |
| Privacy-always f = 0.1 | affected | Bayes-6 | 0.326 [0.318, 0.334] | 0.728 [0.720, 0.736] | 0.414 [0.394, 0.435] | 0.119 [0.113, 0.124] | 0.283 [0.262, 0.308] | 0.138 [0.132, 0.144] |
| Privacy-always f = 0.1 | affected | Vote-LPLA | 0.503 [0.495, 0.511] | 0.582 [0.573, 0.592] | 0.051 [0.041, 0.061] | 0.340 [0.331, 0.350] | 0.022 [0.017, 0.029] | 0.375 [0.366, 0.385] |
| Privacy-always f = 0.1 | affected | PrivAgent | 0.366 [0.358, 0.374] | 0.697 [0.689, 0.706] | 0.414 [0.394, 0.436] | 0.119 [0.113, 0.124] | 0.386 [0.367, 0.405] | 0.101 [0.097, 0.106] |
| Privacy-always f = 0.1 | affected | Expert default | 0.245 [0.238, 0.252] | 0.801 [0.794, 0.807] | 0.414 [0.394, 0.436] | 0.119 [0.113, 0.124] | 0.386 [0.367, 0.405] | 0.101 [0.097, 0.106] |
| Privacy-always f = 0.1 | affected | Anchored slider | 0.114 [0.110, 0.118] | 0.972 [0.971, 0.974] | 0.769 [0.755, 0.785] | 0.017 [0.015, 0.019] | 0.722 [0.707, 0.738] | 0.016 [0.015, 0.018] |
| Privacy-always f = 0.1 | unaffected | Vote-6 | 0.210 [0.207, 0.213] | 0.930 [0.929, 0.931] | 0.327 [0.320, 0.334] | 0.083 [0.081, 0.084] | 0.321 [0.313, 0.328] | 0.086 [0.084, 0.087] |
| Privacy-always f = 0.1 | unaffected | Bayes-6 | 0.126 [0.124, 0.127] | 0.962 [0.961, 0.962] | 0.745 [0.736, 0.753] | 0.023 [0.022, 0.024] | 0.695 [0.687, 0.704] | 0.022 [0.022, 0.023] |
| Privacy-always f = 0.1 | unaffected | Vote-LPLA | 0.227 [0.225, 0.229] | 0.851 [0.848, 0.853] | 0.175 [0.168, 0.181] | 0.088 [0.086, 0.089] | 0.169 [0.163, 0.175] | 0.093 [0.092, 0.095] |
| Privacy-always f = 0.1 | unaffected | PrivAgent | 0.113 [0.111, 0.115] | 0.908 [0.907, 0.910] | 0.773 [0.766, 0.781] | 0.017 [0.016, 0.018] | 0.717 [0.711, 0.724] | 0.017 [0.017, 0.018] |
| Privacy-always f = 0.1 | unaffected | Expert default | 0.252 [0.250, 0.254] | 0.803 [0.801, 0.805] | 0.410 [0.403, 0.417] | 0.121 [0.119, 0.124] | 0.381 [0.374, 0.388] | 0.103 [0.102, 0.105] |
| Privacy-always f = 0.1 | unaffected | Anchored slider | 0.116 [0.115, 0.117] | 0.972 [0.971, 0.972] | 0.773 [0.766, 0.781] | 0.016 [0.015, 0.017] | 0.725 [0.719, 0.732] | 0.016 [0.015, 0.017] |
| Privacy-always f = 0.25 | affected | Vote-6 | 0.406 [0.398, 0.413] | 0.649 [0.643, 0.656] | 0.158 [0.150, 0.167] | 0.227 [0.220, 0.234] | 0.137 [0.130, 0.144] | 0.219 [0.213, 0.226] |
| Privacy-always f = 0.25 | affected | Bayes-6 | 0.323 [0.316, 0.330] | 0.733 [0.728, 0.739] | 0.409 [0.398, 0.420] | 0.120 [0.115, 0.124] | 0.294 [0.283, 0.305] | 0.136 [0.131, 0.140] |
| Privacy-always f = 0.25 | affected | Vote-LPLA | 0.499 [0.490, 0.507] | 0.588 [0.580, 0.595] | 0.055 [0.050, 0.061] | 0.340 [0.330, 0.349] | 0.022 [0.018, 0.026] | 0.374 [0.364, 0.382] |
| Privacy-always f = 0.25 | affected | PrivAgent | 0.363 [0.355, 0.370] | 0.701 [0.695, 0.707] | 0.409 [0.398, 0.420] | 0.120 [0.115, 0.124] | 0.385 [0.375, 0.396] | 0.102 [0.099, 0.106] |
| Privacy-always f = 0.25 | affected | Expert default | 0.251 [0.248, 0.255] | 0.804 [0.802, 0.807] | 0.409 [0.398, 0.420] | 0.120 [0.115, 0.124] | 0.385 [0.375, 0.396] | 0.102 [0.098, 0.106] |
| Privacy-always f = 0.25 | affected | Anchored slider | 0.115 [0.113, 0.117] | 0.972 [0.972, 0.972] | 0.770 [0.759, 0.780] | 0.016 [0.014, 0.017] | 0.729 [0.716, 0.741] | 0.016 [0.014, 0.017] |
| Privacy-always f = 0.25 | unaffected | Vote-6 | 0.213 [0.210, 0.215] | 0.930 [0.929, 0.931] | 0.324 [0.316, 0.332] | 0.084 [0.082, 0.086] | 0.317 [0.309, 0.326] | 0.087 [0.085, 0.089] |
| Privacy-always f = 0.25 | unaffected | Bayes-6 | 0.126 [0.125, 0.127] | 0.961 [0.961, 0.962] | 0.745 [0.736, 0.753] | 0.023 [0.023, 0.024] | 0.695 [0.686, 0.703] | 0.023 [0.022, 0.023] |
| Privacy-always f = 0.25 | unaffected | Vote-LPLA | 0.228 [0.226, 0.231] | 0.850 [0.847, 0.852] | 0.175 [0.169, 0.181] | 0.088 [0.087, 0.090] | 0.170 [0.164, 0.176] | 0.094 [0.092, 0.095] |
| Privacy-always f = 0.25 | unaffected | PrivAgent | 0.114 [0.112, 0.116] | 0.908 [0.906, 0.910] | 0.771 [0.763, 0.778] | 0.018 [0.017, 0.019] | 0.714 [0.707, 0.721] | 0.018 [0.017, 0.019] |
| Privacy-always f = 0.25 | unaffected | Expert default | 0.251 [0.249, 0.253] | 0.802 [0.800, 0.805] | 0.411 [0.402, 0.420] | 0.122 [0.119, 0.124] | 0.380 [0.372, 0.389] | 0.104 [0.102, 0.105] |
| Privacy-always f = 0.25 | unaffected | Anchored slider | 0.116 [0.114, 0.118] | 0.972 [0.971, 0.972] | 0.774 [0.766, 0.782] | 0.016 [0.016, 0.017] | 0.724 [0.717, 0.731] | 0.016 [0.015, 0.017] |

### Point estimates only (compact view)

| population | method | Share error | Cosine | Agreement, Adult | Agreement, CDC-Diabetes |
|---|---|---|---|---|---|
| Baseline (logistic, tau = 0.1) | Vote-6 | 0.212 | 0.930 | 0.326 | 0.319 |
| Baseline (logistic, tau = 0.1) | Bayes-6 | 0.126 | 0.961 | 0.745 | 0.696 |
| Baseline (logistic, tau = 0.1) | Vote-LPLA | 0.228 | 0.850 | 0.175 | 0.170 |
| Baseline (logistic, tau = 0.1) | PrivAgent | 0.113 | 0.908 | 0.772 | 0.717 |
| Baseline (logistic, tau = 0.1) | Expert default | 0.251 | 0.803 | 0.410 | 0.382 |
| Baseline (logistic, tau = 0.1) | Anchored slider | 0.116 | 0.972 | 0.773 | 0.725 |
| Lapse p = 0.1 | Vote-6 | 0.222 | 0.916 | 0.332 | 0.324 |
| Lapse p = 0.1 | Bayes-6 | 0.141 | 0.947 | 0.724 | 0.671 |
| Lapse p = 0.1 | Vote-LPLA | 0.231 | 0.848 | 0.189 | 0.182 |
| Lapse p = 0.1 | PrivAgent | 0.127 | 0.899 | 0.755 | 0.701 |
| Lapse p = 0.1 | Expert default | 0.251 | 0.803 | 0.410 | 0.382 |
| Lapse p = 0.1 | Anchored slider | 0.116 | 0.972 | 0.773 | 0.725 |
| Lapse p = 0.2 | Vote-6 | 0.233 | 0.899 | 0.337 | 0.328 |
| Lapse p = 0.2 | Bayes-6 | 0.155 | 0.932 | 0.699 | 0.644 |
| Lapse p = 0.2 | Vote-LPLA | 0.236 | 0.844 | 0.206 | 0.197 |
| Lapse p = 0.2 | PrivAgent | 0.141 | 0.890 | 0.735 | 0.681 |
| Lapse p = 0.2 | Expert default | 0.251 | 0.803 | 0.410 | 0.382 |
| Lapse p = 0.2 | Anchored slider | 0.116 | 0.972 | 0.773 | 0.725 |
| Lapse p = 0.33 | Vote-6 | 0.252 | 0.872 | 0.337 | 0.323 |
| Lapse p = 0.33 | Bayes-6 | 0.175 | 0.910 | 0.663 | 0.610 |
| Lapse p = 0.33 | Vote-LPLA | 0.248 | 0.834 | 0.222 | 0.211 |
| Lapse p = 0.33 | PrivAgent | 0.161 | 0.876 | 0.700 | 0.648 |
| Lapse p = 0.33 | Expert default | 0.251 | 0.803 | 0.410 | 0.382 |
| Lapse p = 0.33 | Anchored slider | 0.116 | 0.972 | 0.773 | 0.725 |
| Privacy-always f = 0.1 | Vote-6 | 0.230 | 0.901 | 0.310 | 0.302 |
| Privacy-always f = 0.1 | Bayes-6 | 0.146 | 0.938 | 0.712 | 0.654 |
| Privacy-always f = 0.1 | Vote-LPLA | 0.255 | 0.824 | 0.162 | 0.155 |
| Privacy-always f = 0.1 | PrivAgent | 0.138 | 0.887 | 0.737 | 0.684 |
| Privacy-always f = 0.1 | Expert default | 0.251 | 0.803 | 0.410 | 0.382 |
| Privacy-always f = 0.1 | Anchored slider | 0.116 | 0.972 | 0.773 | 0.725 |
| Privacy-always f = 0.25 | Vote-6 | 0.261 | 0.859 | 0.282 | 0.272 |
| Privacy-always f = 0.25 | Bayes-6 | 0.175 | 0.904 | 0.661 | 0.594 |
| Privacy-always f = 0.25 | Vote-LPLA | 0.296 | 0.784 | 0.145 | 0.133 |
| Privacy-always f = 0.25 | PrivAgent | 0.176 | 0.856 | 0.680 | 0.632 |
| Privacy-always f = 0.25 | Expert default | 0.251 | 0.803 | 0.410 | 0.382 |
| Privacy-always f = 0.25 | Anchored slider | 0.116 | 0.972 | 0.773 | 0.725 |
| Side bias b = 0.5 (fixed) | Vote-6 | 0.203 | 0.926 | 0.379 | 0.368 |
| Side bias b = 0.5 (fixed) | Bayes-6 | 0.130 | 0.958 | 0.743 | 0.691 |
| Side bias b = 0.5 (fixed) | Vote-LPLA | 0.235 | 0.845 | 0.168 | 0.160 |
| Side bias b = 0.5 (fixed) | PrivAgent | 0.121 | 0.905 | 0.755 | 0.700 |
| Side bias b = 0.5 (fixed) | Expert default | 0.251 | 0.803 | 0.410 | 0.382 |
| Side bias b = 0.5 (fixed) | Anchored slider | 0.116 | 0.972 | 0.773 | 0.725 |
| Side bias b = 1.0 (fixed) | Vote-6 | 0.200 | 0.915 | 0.424 | 0.409 |
| Side bias b = 1.0 (fixed) | Bayes-6 | 0.138 | 0.949 | 0.732 | 0.679 |
| Side bias b = 1.0 (fixed) | Vote-LPLA | 0.256 | 0.829 | 0.160 | 0.149 |
| Side bias b = 1.0 (fixed) | PrivAgent | 0.140 | 0.894 | 0.715 | 0.660 |
| Side bias b = 1.0 (fixed) | Expert default | 0.251 | 0.803 | 0.410 | 0.382 |
| Side bias b = 1.0 (fixed) | Anchored slider | 0.116 | 0.972 | 0.773 | 0.725 |
| Side bias b = 0.5 (alternating) | Vote-6 | 0.213 | 0.929 | 0.330 | 0.321 |
| Side bias b = 0.5 (alternating) | Bayes-6 | 0.128 | 0.960 | 0.741 | 0.692 |
| Side bias b = 0.5 (alternating) | Vote-LPLA | 0.227 | 0.851 | 0.179 | 0.173 |
| Side bias b = 0.5 (alternating) | PrivAgent | 0.115 | 0.909 | 0.769 | 0.714 |
| Side bias b = 0.5 (alternating) | Expert default | 0.251 | 0.803 | 0.410 | 0.382 |
| Side bias b = 0.5 (alternating) | Anchored slider | 0.116 | 0.972 | 0.773 | 0.725 |
| Side bias b = 1.0 (alternating) | Vote-6 | 0.218 | 0.926 | 0.332 | 0.324 |
| Side bias b = 1.0 (alternating) | Bayes-6 | 0.134 | 0.957 | 0.725 | 0.675 |
| Side bias b = 1.0 (alternating) | Vote-LPLA | 0.224 | 0.855 | 0.197 | 0.191 |
| Side bias b = 1.0 (alternating) | PrivAgent | 0.118 | 0.908 | 0.759 | 0.703 |
| Side bias b = 1.0 (alternating) | Expert default | 0.251 | 0.803 | 0.410 | 0.382 |
| Side bias b = 1.0 (alternating) | Anchored slider | 0.116 | 0.972 | 0.773 | 0.725 |
| Lexicographic | Vote-6 | 0.182 | 0.948 | 0.375 | 0.365 |
| Lexicographic | Bayes-6 | 0.109 | 0.968 | 0.848 | 0.784 |
| Lexicographic | Vote-LPLA | 0.249 | 0.808 | 0.067 | 0.063 |
| Lexicographic | PrivAgent | 0.140 | 0.869 | 0.848 | 0.792 |
| Lexicographic | Expert default | 0.251 | 0.803 | 0.410 | 0.382 |
| Lexicographic | Anchored slider | 0.116 | 0.972 | 0.773 | 0.725 |

Shares of undefined selections and repair rates per population and method: misspec_runs.csv.

## Regenerated thesis figures (stored data, current names)

budget_agreement_ch7.pdf (layout of the stored figure; the six methods named Vote-6, Bayes-6, Vote-LPLA, PrivAgent, Expert default, Anchored slider), rounds_ch7.pdf and tau_ch7.pdf, all from results/ch7/users.parquet and decisions.parquet of the pre-registered run, with its bootstrap seed scheme. Values plotted in tau_ch7:

| tau | method | cosine | agreement Adult |
|---|---|---|---|
| 0.05 | Vote-6 | 0.944 | 0.331 |
| 0.1 | Vote-6 | 0.930 | 0.326 |
| 0.2 | Vote-6 | 0.887 | 0.335 |
| 0.4 | Vote-6 | 0.814 | 0.315 |
| 0.05 | Bayes-6 | 0.972 | 0.798 |
| 0.1 | Bayes-6 | 0.961 | 0.745 |
| 0.2 | Bayes-6 | 0.925 | 0.667 |
| 0.4 | Bayes-6 | 0.871 | 0.576 |
| 0.05 | Vote-LPLA | 0.841 | 0.128 |
| 0.1 | Vote-LPLA | 0.850 | 0.175 |
| 0.2 | Vote-LPLA | 0.840 | 0.225 |
| 0.4 | Vote-LPLA | 0.802 | 0.247 |
| 0.05 | PrivAgent | 0.908 | 0.818 |
| 0.1 | PrivAgent | 0.908 | 0.772 |
| 0.2 | PrivAgent | 0.886 | 0.692 |
| 0.4 | PrivAgent | 0.853 | 0.597 |

rounds_ch7 (Vote-6, k = 1..6): cosine 0.764, 0.847, 0.883, 0.905, 0.921, 0.930; share error 0.323, 0.262, 0.248, 0.230, 0.218, 0.212; agreement Adult 0.227, 0.216, 0.276, 0.309, 0.336, 0.326; expert default 0.803, 0.251, 0.410.

## Runtime

python -m src.misspec: 100.4 s wall clock (20 replications x 11 populations x 1000 users); python -m src.misspec_report: 5.6 s.

## Deviations

D1. Side-biased users: the task asks for the privacy option to alternate sides "as in the stored design", but the stored design has no side assignment (option A always favors the first criterion of the pair). Both a fixed and an alternating assignment were run (config_draft3.yaml); neither is the stored design in the sense of an existing screen layout.

D2. Regenerated figures: the archive of the main run holds budget_agreement_ch7 and a two-panel rounds_ch7, but no tau_ch7. rounds_ch7 and tau_ch7 were drawn to the layouts described by the thesis captions (fig:ch7:rounds, three panels; fig:ch7:tau, two panels) from the stored data. The Uniform default (same eps* as the expert default for every user) is not among the six names asked for and is not in the regenerated budget_agreement_ch7. The stored figures in results/ch7/figures/ are not overwritten.

D3. The worker function is passed to joblib by module reference (commit 2a112d5) after the first launch stopped with a pickling error before any replication ran; the simulation is unchanged.

D4. Environment: Python 3.13 with the installed numpy, scipy, pandas and statsmodels instead of the recorded Python 3.10; the baseline check reproduces the stored estimates to 1.8e-15 and every stored decision.
