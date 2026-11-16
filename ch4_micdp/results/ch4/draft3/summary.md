# Chapter 4 (MIC-DP): Draft 3 follow-up runs

Produced by `python -m src.draft3 && python -m src.draft3_report`. Numbers are reported as they came out. Datasets: Adult and CDC-Diabetes; eps in {0.1, 0.3, 0.5, 1, 5, 10}; delta = 1e-5; 30 paired trials (seed 1000 + t); primary model HistGradientBoostingClassifier(max_iter=200, early_stopping=False, random_state=seed), as in the main run. No stored result, split, seed or hyperparameter was changed.

## Commits

- Configuration and notes of Parts A to D, before any part ran: `8b2e018da2a36f613d9e0ca541814428bc849a89` (2026-11-12 13:44:19 -0800).
- Code and tests, and the amendment of the Part C noise-aware prediction (after a one-unit smoke run, before the main run): `6209f42c05c63d32868277462be5e492108ecc68` (2026-11-12 13:53:11 -0800).
- The main run started after the second commit (`results/ch4/draft3/run.log`).

## Notes fixed before each part (draft3_notes.md, verbatim)

# Draft 3 follow-up runs: notes fixed before each part (committed before any part ran)

Scope: Adult and CDC-Diabetes; eps in {0.1, 0.3, 0.5, 1, 5, 10}; delta = 1e-5; 30 paired trials (seed 1000 + t);
primary model HistGradientBoostingClassifier(max_iter=200, early_stopping=False, random_state=seed). No stored
result, split, seed or hyperparameter is changed. All settings: config.yaml, section draft3.

### Part A. Analyses of stored results

A.1. For Uniform, MIC-DP and MIC-Public (15%) at eps in {0.1, 0.3, 0.5}: counts of trials with stored ROC-AUC
below 0.45, in [0.45, 0.55], and above 0.55, the minimum and the maximum; histogram auc_trials_ch4.pdf (one
panel per dataset and budget, methods overlaid). For the trials below 0.45: the top 3 features by total split
gain of the refit model (and their number of splits), and sign(AUC - 0.5) of the noise-aware score on the clean
test set, on the clean public split, and of the direct prediction on the privatized training rows.

A.2. Predictions are not stored (raw_trials.parquet holds ROC-AUC, accuracy and n_unique_pred only), so the
primary models are refit on the stored releases with the stored seeds and streams. Reported for Uniform, MIC-DP
and MIC-Public at every budget: balanced accuracy at threshold 0.5; accuracy and balanced accuracy at the
threshold t* that maximizes balanced accuracy of the noise-aware prediction on the clean public split.

A.3. 95% percentile bootstrap intervals (10,000 resamples) of the mean of MIC-DP - Uniform (primary model,
ROC-AUC) on CDC-Diabetes at eps = 5 and on Adult and CDC-Diabetes at eps = 0.5; printed with the stored F1
interval and the 90% interval from the same resamples.

A.4. The per-cell outcomes of F1 and F1_secondary (stats.csv) side by side; F1_secondary recomputed from the
secondary-model rows of raw_trials.parquet to state whether it was computed from those rows or copied.

### Part B. Noise-aware prediction, with and without

Same refits as A.2 (Uniform and MIC-DP, six budgets, 30 trials). Reported: ROC-AUC mean and std with direct and
with noise-aware prediction, the paired difference (noise-aware minus direct) with its 95% bootstrap interval,
the paired two-sided Wilcoxon p-value with Holm over the 24 tests, and the number of trials with a constant
prediction under each.

Claim to check, fixed before the run: "noise-aware prediction improves ROC-AUC for Uniform at every budget on
both datasets". It holds if and only if all 12 Uniform tests are significant (Holm-adjusted p < 0.05) with a
positive mean difference.

### Part C. Fine-tuning, repaired

Repair, chosen and recorded before any rerun: the raw score of the pretrained model on the privatized rows is
clipped to [-3, 3] before the continued boosting stage (it is the starting score of trees 101-200), and
l2_regularization = 1.0 for the continued trees (101-200). The pretrained trees are unchanged. The repaired
model predicts clip(F_pre(x), -3, 3) + F_cont(x); its noise-aware prediction is the expectation of that function
over the release noise (config.yaml, draft3.part_c). Rerun: Public-Pretrain + Uniform-FT and Public-Pretrain +
MIC-DP-FT, 15% and 5% public splits, six budgets, 30 trials, on Adult and on CDC-Diabetes (the latter as a check).
Reported: ROC-AUC, number of constant predictions, largest leaf value, and F7r: against Public-Only at the same
size and against the private parent (Uniform, MIC-DP), Holm within the 48 new tests of each dataset. The stored
(unrepaired) results and their F7 outcomes (Holm over 144) are printed next to the repaired ones.

### Part D. Attribute-sampling baseline

Random-k: for each record independently, a uniformly random subset of k of the d features is released with
noise s = sqrt(k) c(eps, delta) per released coordinate (features scaled to [0, 1]); the other coordinates are
replaced by the public-split mean of the feature. Random-k-fixed: one random subset of k features for all
records. k = d/4 and d/2 rounded half up (Adult 3 and 6, CDC-Diabetes 5 and 11). Noise-aware prediction is the
exact expected model output under the release distribution of a record (config.yaml, draft3.part_d). Reported:
ROC-AUC next to the stored Uniform, MIC-DP and MIC-Public, and the paired differences to Uniform and to MIC-DP,
family F9 (96 tests, Holm).

Why Random-k satisfies (eps, delta)-DP for the record. Neighboring datasets differ in one record x (replaced by
x'). The release of every other record does not depend on x. For the changed record, the released subset S is
drawn independently of the data, and the public means m_j and the subset are data independent. Conditional on
S, the release is x_S + N(0, s^2 I_k) together with constants: a Gaussian mechanism on k coordinates, each of
range one, so its L2 sensitivity is sqrt(k), and s = sqrt(k) c(eps, delta) gives (eps, delta)-DP by the analytic
Gaussian calibration (equivalently, sum over released coordinates of (1/s)^2 = 1/c^2, the invariant of the
main run). Since this holds for every S and the distribution of S is the same for both neighbors, the mixture
over S is (eps, delta)-DP. Random-k-fixed is the special case with one S for all records.

### Amendment before the main run (Part C prediction only)

Recorded after a smoke unit (Adult, trial 0, written to results/ch4/draft3_smoke/, not part of the results) and
before the main run. The noise-aware prediction of the repaired model fixed in commit 8b2e018 estimated
E[clip(F_pre(x + eta))] with a Monte-Carlo term of 256 draws per test record. On the smoke unit (Uniform release,
eps = 0.3, 15% split) the spread of the exact E[F_pre(x + eta)] across test records was a standard deviation of
0.012 in logit, far below the Monte-Carlo error per record, so the estimate was dominated by simulation noise
(ROC-AUC 0.503 with it, 0.833 with the deterministic form below). The noise-aware prediction is therefore
clip(E[F_pre(x + eta)], -3, 3) + E[F_cont(x + eta)] with both expectations exact. Nothing else changed: the
repair (clipped start, l2_regularization = 1.0 for the continued trees), the direct prediction
clip(F_pre(x), -3, 3) + F_cont(x), and every other part.


## Sanity checks

```
[PASS] all unit tests pass (pytest exit code 0)
[PASS] adult: X_train byte-identical to the split of the main run (SHA-256)
[PASS] adult: y_train byte-identical to the split of the main run (SHA-256)
[PASS] adult: X_test byte-identical to the split of the main run (SHA-256)
[PASS] adult: y_test byte-identical to the split of the main run (SHA-256)
[PASS] adult: idx_train byte-identical to the split of the main run (SHA-256)
[PASS] adult: idx_test byte-identical to the split of the main run (SHA-256)
[PASS] adult: X_public byte-identical to the split of the main run (SHA-256)
[PASS] adult: d = 12, k = [3, 6] (d/4 and d/2 rounded half up) as in config.yaml
[PASS] cdc_diabetes: X_train byte-identical to the split of the main run (SHA-256)
[PASS] cdc_diabetes: y_train byte-identical to the split of the main run (SHA-256)
[PASS] cdc_diabetes: X_test byte-identical to the split of the main run (SHA-256)
[PASS] cdc_diabetes: y_test byte-identical to the split of the main run (SHA-256)
[PASS] cdc_diabetes: idx_train byte-identical to the split of the main run (SHA-256)
[PASS] cdc_diabetes: idx_test byte-identical to the split of the main run (SHA-256)
[PASS] cdc_diabetes: X_public byte-identical to the split of the main run (SHA-256)
[PASS] cdc_diabetes: d = 21, k = [5, 11] (d/4 and d/2 rounded half up) as in config.yaml
== running 60 of 60 units (rest cached), n_jobs = 2
[PASS] ft_start_clipped: 1440 of 1440 checks pass
[PASS] pretrained_reproduces_public_only: 48 of 48 checks pass
[PASS] randomk_release: 1440 of 1440 checks pass
[PASS] refit_reproduces_stored: 1080 of 1080 checks pass
```

## Part A. Analyses of stored results

### A.1 Per-trial ROC-AUC at eps <= 0.5 (stored values; figure auc_trials_ch4.pdf)

| dataset | eps | method | n_below_0.45 | n_0.45_to_0.55 | n_above_0.55 | min | max |
|---|---|---|---|---|---|---|---|
| Adult | 0.100 | Uniform | 6 | 9 | 15 | 0.243 | 0.757 |
| Adult | 0.100 | MIC-DP | 7 | 8 | 15 | 0.272 | 0.784 |
| Adult | 0.100 | MIC-Public (15%) | 11 | 1 | 18 | 0.236 | 0.793 |
| Adult | 0.300 | Uniform | 1 | 8 | 21 | 0.342 | 0.799 |
| Adult | 0.300 | MIC-DP | 6 | 7 | 17 | 0.296 | 0.775 |
| Adult | 0.300 | MIC-Public (15%) | 1 | 4 | 25 | 0.308 | 0.850 |
| Adult | 0.500 | Uniform | 2 | 1 | 27 | 0.422 | 0.818 |
| Adult | 0.500 | MIC-DP | 5 | 2 | 23 | 0.262 | 0.755 |
| Adult | 0.500 | MIC-Public (15%) | 3 | 3 | 24 | 0.384 | 0.824 |
| CDC-Diabetes | 0.100 | Uniform | 8 | 9 | 13 | 0.242 | 0.749 |
| CDC-Diabetes | 0.100 | MIC-DP | 12 | 11 | 7 | 0.259 | 0.767 |
| CDC-Diabetes | 0.100 | MIC-Public (15%) | 8 | 5 | 17 | 0.210 | 0.788 |
| CDC-Diabetes | 0.300 | Uniform | 3 | 10 | 17 | 0.331 | 0.747 |
| CDC-Diabetes | 0.300 | MIC-DP | 4 | 3 | 23 | 0.333 | 0.757 |
| CDC-Diabetes | 0.300 | MIC-Public (15%) | 2 | 1 | 27 | 0.347 | 0.792 |
| CDC-Diabetes | 0.500 | Uniform | 3 | 3 | 24 | 0.349 | 0.747 |
| CDC-Diabetes | 0.500 | MIC-DP | 5 | 6 | 19 | 0.250 | 0.778 |
| CDC-Diabetes | 0.500 | MIC-Public (15%) | 2 | 0 | 28 | 0.356 | 0.799 |

Trials below 0.45 (refit models, which reproduce the stored ROC-AUC exactly): top 3 features by total split gain (gain share, number of splits), sign of the association of the noise-aware score with the label on the test set, ROC-AUC (sign) on the clean public split, and of the direct prediction on the privatized training rows:

| dataset | eps | method | trial | AUC_test | top features by gain (share, splits) | sign test | AUC public (sign) | AUC train release (sign) |
|---|---|---|---|---|---|---|---|---|
| Adult | 0.100 | MIC-DP | 2 | 0.375 | education-num (0.09, 529); age (0.09, 530); marital-status (0.09, 523) | - | 0.382 (-) | 0.932 (+) |
| Adult | 0.100 | MIC-DP | 5 | 0.404 | hours-per-week (0.09, 535); capital-gain (0.09, 531); relationship (0.09, 531) | - | 0.403 (-) | 0.934 (+) |
| Adult | 0.100 | MIC-DP | 12 | 0.284 | occupation (0.09, 524); race (0.09, 510); native-country (0.09, 538) | - | 0.272 (-) | 0.926 (+) |
| Adult | 0.100 | MIC-DP | 16 | 0.370 | native-country (0.09, 519); capital-loss (0.09, 514); workclass (0.09, 508) | - | 0.360 (-) | 0.926 (+) |
| Adult | 0.100 | MIC-DP | 18 | 0.272 | sex (0.09, 532); workclass (0.09, 513); native-country (0.09, 514) | - | 0.253 (-) | 0.927 (+) |
| Adult | 0.100 | MIC-DP | 19 | 0.372 | sex (0.09, 559); education-num (0.09, 546); relationship (0.09, 496) | - | 0.356 (-) | 0.933 (+) |
| Adult | 0.100 | MIC-DP | 22 | 0.342 | race (0.09, 519); relationship (0.09, 532); sex (0.09, 530) | - | 0.323 (-) | 0.931 (+) |
| Adult | 0.100 | MIC-Public (15%) | 0 | 0.362 | sex (0.13, 816); age (0.13, 768); education-num (0.13, 744) | - | 0.366 (-) | 0.902 (+) |
| Adult | 0.100 | MIC-Public (15%) | 2 | 0.269 | sex (0.14, 814); relationship (0.13, 767); education-num (0.13, 751) | - | 0.250 (-) | 0.898 (+) |
| Adult | 0.100 | MIC-Public (15%) | 5 | 0.392 | hours-per-week (0.13, 781); relationship (0.13, 793); occupation (0.13, 777) | - | 0.390 (-) | 0.905 (+) |
| Adult | 0.100 | MIC-Public (15%) | 9 | 0.313 | capital-gain (0.13, 763); sex (0.13, 763); relationship (0.13, 765) | - | 0.299 (-) | 0.906 (+) |
| Adult | 0.100 | MIC-Public (15%) | 10 | 0.374 | hours-per-week (0.13, 776); marital-status (0.13, 764); sex (0.13, 761) | - | 0.360 (-) | 0.896 (+) |
| Adult | 0.100 | MIC-Public (15%) | 14 | 0.236 | sex (0.15, 835); occupation (0.13, 782); age (0.13, 740) | - | 0.233 (-) | 0.896 (+) |
| Adult | 0.100 | MIC-Public (15%) | 17 | 0.378 | sex (0.14, 809); age (0.14, 797); marital-status (0.13, 760) | - | 0.353 (-) | 0.898 (+) |
| Adult | 0.100 | MIC-Public (15%) | 19 | 0.395 | education-num (0.13, 788); sex (0.13, 749); marital-status (0.13, 749) | - | 0.394 (-) | 0.903 (+) |
| Adult | 0.100 | MIC-Public (15%) | 25 | 0.375 | hours-per-week (0.14, 837); age (0.13, 769); relationship (0.12, 731) | - | 0.367 (-) | 0.908 (+) |
| Adult | 0.100 | MIC-Public (15%) | 26 | 0.319 | marital-status (0.13, 794); capital-gain (0.13, 769); relationship (0.13, 783) | - | 0.318 (-) | 0.903 (+) |
| Adult | 0.100 | MIC-Public (15%) | 29 | 0.326 | education-num (0.14, 834); capital-gain (0.13, 793); relationship (0.13, 766) | - | 0.320 (-) | 0.901 (+) |
| Adult | 0.100 | Uniform | 8 | 0.289 | hours-per-week (0.09, 553); native-country (0.09, 540); race (0.09, 514) | - | 0.288 (-) | 0.927 (+) |
| Adult | 0.100 | Uniform | 11 | 0.417 | hours-per-week (0.09, 534); sex (0.09, 522); occupation (0.09, 525) | - | 0.407 (-) | 0.922 (+) |
| Adult | 0.100 | Uniform | 14 | 0.419 | occupation (0.09, 526); marital-status (0.09, 520); race (0.09, 505) | - | 0.408 (-) | 0.924 (+) |
| Adult | 0.100 | Uniform | 18 | 0.431 | education-num (0.09, 530); age (0.09, 536); hours-per-week (0.09, 522) | - | 0.428 (-) | 0.927 (+) |
| Adult | 0.100 | Uniform | 19 | 0.243 | hours-per-week (0.09, 534); native-country (0.09, 535); sex (0.09, 528) | - | 0.231 (-) | 0.928 (+) |
| Adult | 0.100 | Uniform | 22 | 0.245 | capital-gain (0.09, 536); workclass (0.09, 536); marital-status (0.09, 527) | - | 0.237 (-) | 0.922 (+) |
| Adult | 0.300 | MIC-DP | 3 | 0.376 | race (0.09, 557); capital-gain (0.09, 533); education-num (0.09, 529) | - | 0.368 (-) | 0.933 (+) |
| Adult | 0.300 | MIC-DP | 17 | 0.432 | marital-status (0.09, 549); native-country (0.09, 527); occupation (0.09, 533) | - | 0.436 (-) | 0.929 (+) |
| Adult | 0.300 | MIC-DP | 22 | 0.413 | native-country (0.09, 537); relationship (0.09, 524); sex (0.09, 508) | - | 0.414 (-) | 0.928 (+) |
| Adult | 0.300 | MIC-DP | 23 | 0.296 | relationship (0.09, 545); marital-status (0.09, 537); race (0.09, 534) | - | 0.292 (-) | 0.928 (+) |
| Adult | 0.300 | MIC-DP | 28 | 0.409 | marital-status (0.09, 546); sex (0.09, 545); workclass (0.09, 515) | - | 0.428 (-) | 0.931 (+) |
| Adult | 0.300 | MIC-DP | 29 | 0.427 | age (0.09, 532); native-country (0.09, 539); workclass (0.09, 488) | - | 0.428 (-) | 0.928 (+) |
| Adult | 0.300 | MIC-Public (15%) | 12 | 0.308 | relationship (0.14, 808); sex (0.13, 785); education-num (0.13, 763) | - | 0.292 (-) | 0.903 (+) |
| Adult | 0.300 | Uniform | 11 | 0.342 | hours-per-week (0.09, 553); sex (0.09, 537); capital-gain (0.09, 526) | - | 0.318 (-) | 0.924 (+) |
| Adult | 0.500 | MIC-DP | 12 | 0.315 | workclass (0.09, 543); sex (0.09, 514); occupation (0.09, 511) | - | 0.312 (-) | 0.925 (+) |
| Adult | 0.500 | MIC-DP | 15 | 0.291 | native-country (0.10, 561); education-num (0.09, 530); marital-status (0.09, 517) | - | 0.293 (-) | 0.930 (+) |
| Adult | 0.500 | MIC-DP | 18 | 0.432 | native-country (0.09, 529); relationship (0.09, 534); race (0.09, 513) | - | 0.419 (-) | 0.931 (+) |
| Adult | 0.500 | MIC-DP | 19 | 0.442 | workclass (0.09, 548); capital-gain (0.09, 539); occupation (0.09, 545) | - | 0.457 (-) | 0.924 (+) |
| Adult | 0.500 | MIC-DP | 28 | 0.262 | race (0.09, 521); relationship (0.09, 521); workclass (0.09, 527) | - | 0.264 (-) | 0.926 (+) |
| Adult | 0.500 | MIC-Public (15%) | 3 | 0.437 | capital-gain (0.14, 807); occupation (0.13, 775); hours-per-week (0.13, 758) | - | 0.436 (-) | 0.901 (+) |
| Adult | 0.500 | MIC-Public (15%) | 7 | 0.384 | occupation (0.13, 786); age (0.13, 786); sex (0.13, 794) | - | 0.379 (-) | 0.905 (+) |
| Adult | 0.500 | MIC-Public (15%) | 21 | 0.432 | marital-status (0.13, 809); occupation (0.13, 765); capital-gain (0.13, 745) | - | 0.425 (-) | 0.901 (+) |
| Adult | 0.500 | Uniform | 4 | 0.428 | race (0.09, 546); marital-status (0.09, 542); capital-loss (0.09, 543) | - | 0.432 (-) | 0.929 (+) |
| Adult | 0.500 | Uniform | 9 | 0.422 | marital-status (0.09, 558); native-country (0.09, 522); age (0.09, 519) | - | 0.424 (-) | 0.922 (+) |
| CDC-Diabetes | 0.100 | MIC-DP | 1 | 0.383 | Fruits (0.06, 343); Veggies (0.05, 318); GenHlth (0.05, 302) | - | 0.387 (-) | 0.966 (+) |
| CDC-Diabetes | 0.100 | MIC-DP | 2 | 0.443 | Education (0.05, 333); PhysActivity (0.05, 304); HvyAlcoholConsump (0.05, 285) | - | 0.449 (-) | 0.962 (+) |
| CDC-Diabetes | 0.100 | MIC-DP | 10 | 0.432 | AnyHealthcare (0.05, 305); Education (0.05, 315); HeartDiseaseorAttack (0.05, 309) | - | 0.422 (-) | 0.953 (+) |
| CDC-Diabetes | 0.100 | MIC-DP | 11 | 0.357 | HighChol (0.05, 303); PhysActivity (0.05, 330); Income (0.05, 304) | - | 0.350 (-) | 0.963 (+) |
| CDC-Diabetes | 0.100 | MIC-DP | 13 | 0.390 | PhysHlth (0.05, 315); HeartDiseaseorAttack (0.05, 317); HvyAlcoholConsump (0.05, 298) | - | 0.396 (-) | 0.954 (+) |
| CDC-Diabetes | 0.100 | MIC-DP | 15 | 0.346 | CholCheck (0.05, 317); Smoker (0.05, 309); PhysActivity (0.05, 297) | - | 0.350 (-) | 0.961 (+) |
| CDC-Diabetes | 0.100 | MIC-DP | 17 | 0.379 | Smoker (0.05, 304); PhysActivity (0.05, 304); Income (0.05, 311) | - | 0.401 (-) | 0.959 (+) |
| CDC-Diabetes | 0.100 | MIC-DP | 18 | 0.259 | HeartDiseaseorAttack (0.05, 312); Fruits (0.05, 302); HighChol (0.05, 323) | - | 0.262 (-) | 0.957 (+) |
| CDC-Diabetes | 0.100 | MIC-DP | 19 | 0.345 | Smoker (0.06, 330); PhysHlth (0.05, 313); HighChol (0.05, 301) | - | 0.352 (-) | 0.958 (+) |
| CDC-Diabetes | 0.100 | MIC-DP | 22 | 0.442 | Age (0.06, 345); HighBP (0.05, 308); AnyHealthcare (0.05, 296) | - | 0.443 (-) | 0.962 (+) |
| CDC-Diabetes | 0.100 | MIC-DP | 23 | 0.398 | BMI (0.05, 339); AnyHealthcare (0.05, 310); HighChol (0.05, 318) | - | 0.418 (-) | 0.964 (+) |
| CDC-Diabetes | 0.100 | MIC-DP | 24 | 0.381 | HighBP (0.06, 349); HighChol (0.06, 329); Income (0.05, 327) | - | 0.390 (-) | 0.962 (+) |
| CDC-Diabetes | 0.100 | MIC-Public (15%) | 1 | 0.315 | Education (0.11, 640); PhysHlth (0.11, 637); HighBP (0.10, 622) | - | 0.333 (-) | 0.929 (+) |
| CDC-Diabetes | 0.100 | MIC-Public (15%) | 10 | 0.311 | Age (0.11, 630); Income (0.11, 624); HeartDiseaseorAttack (0.11, 651) | - | 0.333 (-) | 0.920 (+) |
| CDC-Diabetes | 0.100 | MIC-Public (15%) | 11 | 0.263 | BMI (0.11, 646); GenHlth (0.11, 645); HighChol (0.10, 595) | - | 0.265 (-) | 0.929 (+) |
| CDC-Diabetes | 0.100 | MIC-Public (15%) | 14 | 0.263 | HighBP (0.10, 631); BMI (0.10, 621); GenHlth (0.10, 593) | - | 0.274 (-) | 0.920 (+) |
| CDC-Diabetes | 0.100 | MIC-Public (15%) | 18 | 0.320 | BMI (0.11, 637); PhysHlth (0.11, 636); DiffWalk (0.11, 612) | - | 0.332 (-) | 0.928 (+) |
| CDC-Diabetes | 0.100 | MIC-Public (15%) | 21 | 0.308 | HighBP (0.11, 654); Income (0.11, 628); HighChol (0.10, 617) | - | 0.309 (-) | 0.926 (+) |
| CDC-Diabetes | 0.100 | MIC-Public (15%) | 24 | 0.346 | HeartDiseaseorAttack (0.11, 610); HighBP (0.10, 606); BMI (0.10, 645) | - | 0.358 (-) | 0.921 (+) |
| CDC-Diabetes | 0.100 | MIC-Public (15%) | 26 | 0.210 | Education (0.11, 659); Age (0.11, 629); Income (0.10, 616) | - | 0.213 (-) | 0.926 (+) |
| CDC-Diabetes | 0.100 | Uniform | 2 | 0.433 | BMI (0.06, 336); DiffWalk (0.05, 302); CholCheck (0.05, 325) | - | 0.425 (-) | 0.955 (+) |
| CDC-Diabetes | 0.100 | Uniform | 10 | 0.296 | PhysActivity (0.05, 308); AnyHealthcare (0.05, 299); Veggies (0.05, 315) | - | 0.314 (-) | 0.959 (+) |
| CDC-Diabetes | 0.100 | Uniform | 14 | 0.264 | NoDocbcCost (0.05, 333); GenHlth (0.05, 325); MentHlth (0.05, 315) | - | 0.268 (-) | 0.960 (+) |
| CDC-Diabetes | 0.100 | Uniform | 18 | 0.431 | Age (0.06, 319); HighChol (0.05, 299); Smoker (0.05, 298) | - | 0.442 (-) | 0.964 (+) |
| CDC-Diabetes | 0.100 | Uniform | 19 | 0.242 | Education (0.05, 328); PhysHlth (0.05, 321); GenHlth (0.05, 306) | - | 0.255 (-) | 0.967 (+) |
| CDC-Diabetes | 0.100 | Uniform | 20 | 0.365 | Education (0.05, 314); Sex (0.05, 306); HighBP (0.05, 288) | - | 0.387 (-) | 0.962 (+) |
| CDC-Diabetes | 0.100 | Uniform | 23 | 0.423 | PhysHlth (0.05, 330); Smoker (0.05, 307); NoDocbcCost (0.05, 301) | - | 0.421 (-) | 0.965 (+) |
| CDC-Diabetes | 0.100 | Uniform | 29 | 0.321 | Stroke (0.05, 314); HvyAlcoholConsump (0.05, 309); PhysActivity (0.05, 301) | - | 0.318 (-) | 0.963 (+) |
| CDC-Diabetes | 0.300 | MIC-DP | 5 | 0.388 | Education (0.05, 301); PhysHlth (0.05, 298); GenHlth (0.05, 304) | - | 0.385 (-) | 0.965 (+) |
| CDC-Diabetes | 0.300 | MIC-DP | 10 | 0.333 | GenHlth (0.05, 314); HeartDiseaseorAttack (0.05, 301); HighBP (0.05, 300) | - | 0.326 (-) | 0.968 (+) |
| CDC-Diabetes | 0.300 | MIC-DP | 11 | 0.434 | GenHlth (0.05, 289); Sex (0.05, 312); MentHlth (0.05, 290) | - | 0.444 (-) | 0.961 (+) |
| CDC-Diabetes | 0.300 | MIC-DP | 19 | 0.388 | DiffWalk (0.05, 320); Education (0.05, 310); NoDocbcCost (0.05, 294) | - | 0.398 (-) | 0.963 (+) |
| CDC-Diabetes | 0.300 | MIC-Public (15%) | 15 | 0.353 | BMI (0.11, 642); HeartDiseaseorAttack (0.11, 632); Income (0.11, 630) | - | 0.359 (-) | 0.924 (+) |
| CDC-Diabetes | 0.300 | MIC-Public (15%) | 20 | 0.347 | Age (0.11, 647); BMI (0.11, 642); GenHlth (0.10, 631) | - | 0.353 (-) | 0.932 (+) |
| CDC-Diabetes | 0.300 | Uniform | 5 | 0.331 | Veggies (0.06, 309); PhysHlth (0.05, 300); Fruits (0.05, 313) | - | 0.337 (-) | 0.959 (+) |
| CDC-Diabetes | 0.300 | Uniform | 9 | 0.420 | NoDocbcCost (0.05, 331); DiffWalk (0.05, 322); Sex (0.05, 293) | - | 0.422 (-) | 0.956 (+) |
| CDC-Diabetes | 0.300 | Uniform | 12 | 0.408 | Sex (0.05, 310); BMI (0.05, 300); Fruits (0.05, 319) | - | 0.412 (-) | 0.959 (+) |
| CDC-Diabetes | 0.500 | MIC-DP | 1 | 0.250 | MentHlth (0.05, 313); Stroke (0.05, 334); Income (0.05, 300) | - | 0.253 (-) | 0.964 (+) |
| CDC-Diabetes | 0.500 | MIC-DP | 13 | 0.383 | CholCheck (0.06, 334); GenHlth (0.05, 326); Income (0.05, 306) | - | 0.396 (-) | 0.956 (+) |
| CDC-Diabetes | 0.500 | MIC-DP | 15 | 0.404 | BMI (0.05, 332); HighBP (0.05, 300); Income (0.05, 319) | - | 0.398 (-) | 0.957 (+) |
| CDC-Diabetes | 0.500 | MIC-DP | 19 | 0.393 | Income (0.05, 322); GenHlth (0.05, 304); Smoker (0.05, 320) | - | 0.408 (-) | 0.955 (+) |
| CDC-Diabetes | 0.500 | MIC-DP | 28 | 0.424 | PhysActivity (0.05, 313); HvyAlcoholConsump (0.05, 304); Education (0.05, 300) | - | 0.426 (-) | 0.963 (+) |
| CDC-Diabetes | 0.500 | MIC-Public (15%) | 6 | 0.356 | Income (0.11, 631); HeartDiseaseorAttack (0.11, 637); Age (0.11, 632) | - | 0.367 (-) | 0.923 (+) |
| CDC-Diabetes | 0.500 | MIC-Public (15%) | 21 | 0.388 | Income (0.11, 624); HighBP (0.11, 640); HeartDiseaseorAttack (0.11, 631) | - | 0.396 (-) | 0.922 (+) |
| CDC-Diabetes | 0.500 | Uniform | 7 | 0.409 | Income (0.05, 325); GenHlth (0.05, 315); Age (0.05, 313) | - | 0.432 (-) | 0.957 (+) |
| CDC-Diabetes | 0.500 | Uniform | 16 | 0.349 | Sex (0.06, 343); CholCheck (0.06, 333); HighChol (0.05, 305) | - | 0.346 (-) | 0.956 (+) |
| CDC-Diabetes | 0.500 | Uniform | 18 | 0.428 | Stroke (0.05, 311); Education (0.05, 302); GenHlth (0.05, 304) | - | 0.429 (-) | 0.959 (+) |

Top-1 feature among the trials below 0.45:

| dataset | method | low trials | top-1 feature (count) | public AUC < 0.5 | train-release AUC < 0.5 |
|---|---|---|---|---|---|
| Adult | MIC-DP | 18 | native-country (4); race (3); workclass (2); marital-status (2); sex (2); education-num (1); hours-per-week (1); occupation (1); relationship (1); age (1) | 18 | 0 |
| Adult | MIC-Public (15%) | 15 | sex (4); hours-per-week (3); capital-gain (2); education-num (2); marital-status (2); occupation (1); relationship (1) | 15 | 0 |
| Adult | Uniform | 9 | hours-per-week (4); race (1); marital-status (1); occupation (1); education-num (1); capital-gain (1) | 9 | 0 |
| CDC-Diabetes | MIC-DP | 21 | Education (2); GenHlth (2); CholCheck (2); BMI (2); Smoker (2); Fruits (1); MentHlth (1); AnyHealthcare (1); HighChol (1); PhysHlth (1); HeartDiseaseorAttack (1); DiffWalk (1); Income (1); Age (1); HighBP (1); PhysActivity (1) | 21 | 0 |
| CDC-Diabetes | MIC-Public (15%) | 12 | BMI (3); Education (2); Income (2); Age (2); HighBP (2); HeartDiseaseorAttack (1) | 12 | 0 |
| CDC-Diabetes | Uniform | 14 | NoDocbcCost (2); Sex (2); Stroke (2); Education (2); BMI (1); Veggies (1); Income (1); PhysActivity (1); Age (1); PhysHlth (1) | 14 | 0 |

Same diagnostics over all 30 trials of each cell (for reference):

| dataset | eps | method | trials | test AUC < 0.45 | public AUC < 0.5 | train-release AUC < 0.5 | corr(test AUC, public AUC) |
|---|---|---|---|---|---|---|---|
| Adult | 0.100 | MIC-DP | 30.000 | 7.000 | 13.000 | 0.000 | 0.997 |
| Adult | 0.100 | MIC-Public (15%) | 30.000 | 11.000 | 11.000 | 0.000 | 0.998 |
| Adult | 0.100 | Uniform | 30.000 | 6.000 | 12.000 | 0.000 | 0.997 |
| Adult | 0.300 | MIC-DP | 30.000 | 6.000 | 10.000 | 0.000 | 0.997 |
| Adult | 0.300 | MIC-Public (15%) | 30.000 | 1.000 | 3.000 | 0.000 | 0.996 |
| Adult | 0.300 | Uniform | 30.000 | 1.000 | 3.000 | 0.000 | 0.993 |
| Adult | 0.500 | MIC-DP | 30.000 | 5.000 | 5.000 | 0.000 | 0.997 |
| Adult | 0.500 | MIC-Public (15%) | 30.000 | 3.000 | 3.000 | 0.000 | 0.998 |
| Adult | 0.500 | Uniform | 30.000 | 2.000 | 2.000 | 0.000 | 0.995 |
| CDC-Diabetes | 0.100 | MIC-DP | 30.000 | 12.000 | 15.000 | 0.000 | 0.997 |
| CDC-Diabetes | 0.100 | MIC-Public (15%) | 30.000 | 8.000 | 12.000 | 0.000 | 0.999 |
| CDC-Diabetes | 0.100 | Uniform | 30.000 | 8.000 | 10.000 | 0.000 | 0.997 |
| CDC-Diabetes | 0.300 | MIC-DP | 30.000 | 4.000 | 5.000 | 0.000 | 0.996 |
| CDC-Diabetes | 0.300 | MIC-Public (15%) | 30.000 | 2.000 | 2.000 | 0.000 | 0.999 |
| CDC-Diabetes | 0.300 | Uniform | 30.000 | 3.000 | 7.000 | 0.000 | 0.998 |
| CDC-Diabetes | 0.500 | MIC-DP | 30.000 | 5.000 | 6.000 | 0.000 | 0.997 |
| CDC-Diabetes | 0.500 | MIC-Public (15%) | 30.000 | 2.000 | 2.000 | 0.000 | 0.999 |
| CDC-Diabetes | 0.500 | Uniform | 30.000 | 3.000 | 3.000 | 0.000 | 0.997 |

### A.2 Accuracy at a sensible threshold

Predictions are not stored in raw_trials.parquet (it holds ROC-AUC, accuracy at 0.5 and the number of distinct predictions), so the primary models were refit on the stored releases with the stored seeds and streams; every refit reproduces the stored noise-aware ROC-AUC, accuracy and number of distinct predictions exactly (sanity check refit_reproduces_stored). acc@0.5 and BA@0.5: noise-aware prediction at threshold 0.5 on the clean test set; t*: the threshold that maximizes balanced accuracy of the noise-aware prediction on the clean 15% public split; acc@t* and BA@t*: on the test set.

| dataset | eps | method | acc@0.5 | BA@0.5 | t* (median) | acc@t* | BA@t* | BA public@t* |
|---|---|---|---|---|---|---|---|---|
| Adult | 0.1000 | Uniform | 0.761 ± 0.000 | 0.500 ± 0.000 | 0.231 | 0.447 ± 0.142 | 0.570 ± 0.058 | 0.576 ± 0.061 |
| Adult | 0.1000 | MIC-DP | 0.761 ± 0.000 | 0.500 ± 0.000 | 0.231 | 0.477 ± 0.176 | 0.582 ± 0.076 | 0.589 ± 0.078 |
| Adult | 0.1000 | MIC-Public (15%) | 0.761 ± 0.000 | 0.500 ± 0.000 | 0.232 | 0.485 ± 0.180 | 0.603 ± 0.083 | 0.608 ± 0.085 |
| Adult | 0.3000 | Uniform | 0.761 ± 0.000 | 0.500 ± 0.000 | 0.231 | 0.526 ± 0.117 | 0.608 ± 0.066 | 0.614 ± 0.065 |
| Adult | 0.3000 | MIC-DP | 0.761 ± 0.000 | 0.500 ± 0.000 | 0.231 | 0.504 ± 0.144 | 0.592 ± 0.069 | 0.600 ± 0.071 |
| Adult | 0.3000 | MIC-Public (15%) | 0.761 ± 0.000 | 0.500 ± 0.000 | 0.232 | 0.594 ± 0.137 | 0.667 ± 0.081 | 0.672 ± 0.082 |
| Adult | 0.5000 | Uniform | 0.761 ± 0.000 | 0.500 ± 0.000 | 0.231 | 0.611 ± 0.102 | 0.645 ± 0.061 | 0.656 ± 0.061 |
| Adult | 0.5000 | MIC-DP | 0.761 ± 0.000 | 0.500 ± 0.000 | 0.231 | 0.564 ± 0.129 | 0.622 ± 0.064 | 0.629 ± 0.065 |
| Adult | 0.5000 | MIC-Public (15%) | 0.761 ± 0.000 | 0.500 ± 0.000 | 0.232 | 0.610 ± 0.126 | 0.675 ± 0.075 | 0.680 ± 0.077 |
| Adult | 1.0000 | Uniform | 0.761 ± 0.000 | 0.500 ± 0.000 | 0.231 | 0.640 ± 0.095 | 0.669 ± 0.051 | 0.681 ± 0.051 |
| Adult | 1.0000 | MIC-DP | 0.761 ± 0.000 | 0.500 ± 0.000 | 0.231 | 0.656 ± 0.073 | 0.676 ± 0.064 | 0.683 ± 0.066 |
| Adult | 1.0000 | MIC-Public (15%) | 0.761 ± 0.000 | 0.500 ± 0.000 | 0.232 | 0.713 ± 0.048 | 0.746 ± 0.028 | 0.750 ± 0.027 |
| Adult | 5.0000 | Uniform | 0.761 ± 0.000 | 0.500 ± 0.000 | 0.232 | 0.747 ± 0.015 | 0.732 ± 0.010 | 0.752 ± 0.008 |
| Adult | 5.0000 | MIC-DP | 0.761 ± 0.000 | 0.500 ± 0.000 | 0.232 | 0.752 ± 0.017 | 0.769 ± 0.012 | 0.773 ± 0.012 |
| Adult | 5.0000 | MIC-Public (15%) | 0.761 ± 0.000 | 0.500 ± 0.000 | 0.233 | 0.757 ± 0.012 | 0.775 ± 0.008 | 0.779 ± 0.007 |
| Adult | 10.0000 | Uniform | 0.761 ± 0.000 | 0.500 ± 0.000 | 0.235 | 0.753 ± 0.012 | 0.735 ± 0.006 | 0.754 ± 0.006 |
| Adult | 10.0000 | MIC-DP | 0.761 ± 0.000 | 0.500 ± 0.000 | 0.234 | 0.759 ± 0.010 | 0.775 ± 0.005 | 0.778 ± 0.005 |
| Adult | 10.0000 | MIC-Public (15%) | 0.761 ± 0.000 | 0.500 ± 0.000 | 0.236 | 0.762 ± 0.008 | 0.779 ± 0.004 | 0.782 ± 0.003 |
| CDC-Diabetes | 0.1000 | Uniform | 0.861 ± 0.000 | 0.500 ± 0.000 | 0.128 | 0.613 ± 0.209 | 0.544 ± 0.046 | 0.548 ± 0.045 |
| CDC-Diabetes | 0.1000 | MIC-DP | 0.861 ± 0.000 | 0.500 ± 0.000 | 0.128 | 0.660 ± 0.207 | 0.542 ± 0.057 | 0.543 ± 0.056 |
| CDC-Diabetes | 0.1000 | MIC-Public (15%) | 0.861 ± 0.000 | 0.500 ± 0.000 | 0.130 | 0.616 ± 0.208 | 0.595 ± 0.083 | 0.594 ± 0.080 |
| CDC-Diabetes | 0.3000 | Uniform | 0.861 ± 0.000 | 0.500 ± 0.000 | 0.128 | 0.627 ± 0.153 | 0.579 ± 0.062 | 0.580 ± 0.061 |
| CDC-Diabetes | 0.3000 | MIC-DP | 0.861 ± 0.000 | 0.500 ± 0.000 | 0.128 | 0.624 ± 0.141 | 0.595 ± 0.061 | 0.597 ± 0.059 |
| CDC-Diabetes | 0.3000 | MIC-Public (15%) | 0.861 ± 0.000 | 0.500 ± 0.000 | 0.130 | 0.643 ± 0.114 | 0.669 ± 0.065 | 0.666 ± 0.063 |
| CDC-Diabetes | 0.5000 | Uniform | 0.861 ± 0.000 | 0.500 ± 0.000 | 0.128 | 0.638 ± 0.120 | 0.608 ± 0.063 | 0.607 ± 0.060 |
| CDC-Diabetes | 0.5000 | MIC-DP | 0.861 ± 0.000 | 0.500 ± 0.000 | 0.128 | 0.602 ± 0.182 | 0.585 ± 0.070 | 0.588 ± 0.068 |
| CDC-Diabetes | 0.5000 | MIC-Public (15%) | 0.861 ± 0.000 | 0.500 ± 0.000 | 0.130 | 0.664 ± 0.114 | 0.680 ± 0.062 | 0.676 ± 0.059 |
| CDC-Diabetes | 1.0000 | Uniform | 0.861 ± 0.000 | 0.500 ± 0.000 | 0.128 | 0.633 ± 0.058 | 0.660 ± 0.038 | 0.658 ± 0.037 |
| CDC-Diabetes | 1.0000 | MIC-DP | 0.861 ± 0.000 | 0.500 ± 0.000 | 0.128 | 0.640 ± 0.061 | 0.644 ± 0.041 | 0.644 ± 0.040 |
| CDC-Diabetes | 1.0000 | MIC-Public (15%) | 0.861 ± 0.000 | 0.500 ± 0.000 | 0.130 | 0.719 ± 0.031 | 0.722 ± 0.008 | 0.718 ± 0.009 |
| CDC-Diabetes | 5.0000 | Uniform | 0.861 ± 0.000 | 0.500 ± 0.000 | 0.128 | 0.686 ± 0.019 | 0.725 ± 0.005 | 0.720 ± 0.004 |
| CDC-Diabetes | 5.0000 | MIC-DP | 0.861 ± 0.000 | 0.500 ± 0.000 | 0.129 | 0.710 ± 0.031 | 0.721 ± 0.007 | 0.717 ± 0.006 |
| CDC-Diabetes | 5.0000 | MIC-Public (15%) | 0.861 ± 0.000 | 0.500 ± 0.000 | 0.133 | 0.749 ± 0.013 | 0.724 ± 0.002 | 0.721 ± 0.002 |
| CDC-Diabetes | 10.0000 | Uniform | 0.861 ± 0.000 | 0.500 ± 0.000 | 0.127 | 0.693 ± 0.026 | 0.728 ± 0.003 | 0.722 ± 0.003 |
| CDC-Diabetes | 10.0000 | MIC-DP | 0.861 ± 0.000 | 0.500 ± 0.000 | 0.131 | 0.714 ± 0.032 | 0.726 ± 0.005 | 0.723 ± 0.006 |
| CDC-Diabetes | 10.0000 | MIC-Public (15%) | 0.861 ± 0.000 | 0.500 ± 0.000 | 0.139 | 0.754 ± 0.009 | 0.723 ± 0.002 | 0.722 ± 0.002 |

### A.3 95% intervals of MIC-DP minus Uniform (ROC-AUC, primary model)

| dataset | eps | mean diff | 95% CI (new) | 90% CI (same resamples) | stored F1 95% CI (stats.csv) | Wilcoxon p (raw) | p Holm (F1) |
|---|---|---|---|---|---|---|---|
| CDC-Diabetes | 5.0000 | -0.0028 | [-0.0055, -0.0003] | [-0.0050, -0.0006] | [-0.0055, -0.0002] | 0.0879 | 1.0000 |
| Adult | 0.5000 | -0.0561 | [-0.1165, +0.0041] | [-0.1061, -0.0065] | [-0.1167, +0.0033] | 0.0841 | 1.0000 |
| CDC-Diabetes | 0.5000 | -0.0429 | [-0.0954, +0.0106] | [-0.0876, +0.0022] | [-0.0949, +0.0103] | 0.1004 | 1.0000 |

### A.4 Family F1 with the primary and with the linear model

F1: {'n.s.': 80, 'A better': 14, 'B better': 2}; F1_secondary: {'n.s.': 80, 'A better': 14, 'B better': 2}. Same 96 cells in the same order: True. Cells with the same outcome: 96 of 96. Cells with the same mean difference (to 1e-12): 0 of 96. F1_secondary recomputed from the secondary-model rows of raw_trials.parquet (src/stats.run_family, model = 'secondary') equals the stored F1_secondary in every mean difference, Holm p-value and outcome: True. Secondary-model rows of MIC-DP in raw_trials.parquet: 1440; model names: ['LogisticRegression', 'Ridge'].

| dataset | eps | baseline | diff primary | p_holm primary | outcome F1 | diff secondary | p_holm secondary | outcome F1_secondary |
|---|---|---|---|---|---|---|---|---|
| adult | 0.1000 | uniform | 0.0153 | 1.0000 | n.s. | 0.0210 | 1.0000 | n.s. |
| adult | 0.1000 | pearson_alloc | 0.0329 | 1.0000 | n.s. | 0.0317 | 1.0000 | n.s. |
| adult | 0.1000 | pearson_adj | 0.0323 | 1.0000 | n.s. | 0.0378 | 1.0000 | n.s. |
| adult | 0.1000 | partialcorr_adj | 0.0222 | 1.0000 | n.s. | 0.0212 | 1.0000 | n.s. |
| adult | 0.3000 | uniform | -0.0334 | 1.0000 | n.s. | -0.0145 | 1.0000 | n.s. |
| adult | 0.3000 | pearson_alloc | 0.0088 | 1.0000 | n.s. | 0.0194 | 1.0000 | n.s. |
| adult | 0.3000 | pearson_adj | 0.0248 | 1.0000 | n.s. | 0.0397 | 1.0000 | n.s. |
| adult | 0.3000 | partialcorr_adj | 0.0408 | 1.0000 | n.s. | 0.0469 | 1.0000 | n.s. |
| adult | 0.5000 | uniform | -0.0561 | 1.0000 | n.s. | -0.0686 | 1.0000 | n.s. |
| adult | 0.5000 | pearson_alloc | -0.0373 | 1.0000 | n.s. | -0.0410 | 1.0000 | n.s. |
| adult | 0.5000 | pearson_adj | 0.0300 | 1.0000 | n.s. | 0.0137 | 1.0000 | n.s. |
| adult | 0.5000 | partialcorr_adj | 0.0040 | 1.0000 | n.s. | 0.0027 | 1.0000 | n.s. |
| adult | 1.0000 | uniform | -0.0035 | 1.0000 | n.s. | 0.0074 | 1.0000 | n.s. |
| adult | 1.0000 | pearson_alloc | -0.0453 | 1.0000 | n.s. | -0.0412 | 1.0000 | n.s. |
| adult | 1.0000 | pearson_adj | 0.0348 | 1.0000 | n.s. | 0.0445 | 1.0000 | n.s. |
| adult | 1.0000 | partialcorr_adj | -0.0033 | 1.0000 | n.s. | 0.0073 | 1.0000 | n.s. |
| adult | 5.0000 | uniform | 0.0210 | 0.000656 | A better | 0.0200 | 0.000266 | A better |
| adult | 5.0000 | pearson_alloc | -0.0011 | 1.0000 | n.s. | -0.0015 | 1.0000 | n.s. |
| adult | 5.0000 | pearson_adj | 0.0384 | 3.04e-06 | A better | 0.0361 | 4.86e-07 | A better |
| adult | 5.0000 | partialcorr_adj | 0.0411 | 4.86e-07 | A better | 0.0394 | 8.01e-07 | A better |
| adult | 10.0000 | uniform | 0.0217 | 1.79e-07 | A better | 0.0195 | 1.79e-07 | A better |
| adult | 10.0000 | pearson_alloc | -0.000432 | 1.0000 | n.s. | -0.0027 | 1.0000 | n.s. |
| adult | 10.0000 | pearson_adj | 0.0341 | 1.79e-07 | A better | 0.0313 | 1.79e-07 | A better |
| adult | 10.0000 | partialcorr_adj | 0.0341 | 1.79e-07 | A better | 0.0320 | 1.79e-07 | A better |
| cdc_diabetes | 0.1000 | uniform | -0.0127 | 1.0000 | n.s. | 0.0083 | 1.0000 | n.s. |
| cdc_diabetes | 0.1000 | pearson_alloc | -0.0291 | 1.0000 | n.s. | -0.0235 | 1.0000 | n.s. |
| cdc_diabetes | 0.1000 | pearson_adj | -0.0312 | 1.0000 | n.s. | -0.0188 | 1.0000 | n.s. |
| cdc_diabetes | 0.1000 | partialcorr_adj | -0.0300 | 1.0000 | n.s. | -0.0152 | 1.0000 | n.s. |
| cdc_diabetes | 0.3000 | uniform | 0.0242 | 1.0000 | n.s. | 0.0165 | 1.0000 | n.s. |
| cdc_diabetes | 0.3000 | pearson_alloc | 0.0264 | 1.0000 | n.s. | 0.0268 | 1.0000 | n.s. |
| cdc_diabetes | 0.3000 | pearson_adj | 0.0546 | 1.0000 | n.s. | 0.0481 | 1.0000 | n.s. |
| cdc_diabetes | 0.3000 | partialcorr_adj | 0.0145 | 1.0000 | n.s. | 0.0114 | 1.0000 | n.s. |
| cdc_diabetes | 0.5000 | uniform | -0.0429 | 1.0000 | n.s. | -0.0520 | 1.0000 | n.s. |
| cdc_diabetes | 0.5000 | pearson_alloc | -0.0388 | 1.0000 | n.s. | -0.0215 | 1.0000 | n.s. |
| cdc_diabetes | 0.5000 | pearson_adj | -0.0307 | 1.0000 | n.s. | -0.0275 | 1.0000 | n.s. |
| cdc_diabetes | 0.5000 | partialcorr_adj | -0.0331 | 1.0000 | n.s. | -0.0314 | 1.0000 | n.s. |
| cdc_diabetes | 1.0000 | uniform | -0.0213 | 1.0000 | n.s. | -0.0263 | 1.0000 | n.s. |
| cdc_diabetes | 1.0000 | pearson_alloc | -0.0319 | 1.0000 | n.s. | -0.0321 | 1.0000 | n.s. |
| cdc_diabetes | 1.0000 | pearson_adj | 0.0123 | 1.0000 | n.s. | 0.0189 | 1.0000 | n.s. |
| cdc_diabetes | 1.0000 | partialcorr_adj | 0.0100 | 1.0000 | n.s. | 0.0024 | 1.0000 | n.s. |
| cdc_diabetes | 5.0000 | uniform | -0.0028 | 1.0000 | n.s. | -0.0023 | 1.0000 | n.s. |
| cdc_diabetes | 5.0000 | pearson_alloc | -0.0101 | 8.61e-06 | B better | -0.0099 | 3.01e-06 | B better |
| cdc_diabetes | 5.0000 | pearson_adj | 0.0053 | 0.9638 | n.s. | 0.0043 | 1.0000 | n.s. |
| cdc_diabetes | 5.0000 | partialcorr_adj | -0.000381 | 1.0000 | n.s. | -0.000708 | 1.0000 | n.s. |
| cdc_diabetes | 10.0000 | uniform | 0.00073 | 1.0000 | n.s. | 0.000752 | 1.0000 | n.s. |
| cdc_diabetes | 10.0000 | pearson_alloc | -0.0023 | 1.0000 | n.s. | -0.0025 | 0.4894 | n.s. |
| cdc_diabetes | 10.0000 | pearson_adj | 0.0054 | 0.000118 | A better | 0.0054 | 4.75e-05 | A better |
| cdc_diabetes | 10.0000 | partialcorr_adj | 0.0036 | 0.0449 | A better | 0.0039 | 0.0208 | A better |
| calhousing | 0.1000 | uniform | -0.000124 | 1.0000 | n.s. | -1.3e-05 | 1.0000 | n.s. |
| calhousing | 0.1000 | pearson_alloc | -0.00013 | 1.0000 | n.s. | 1.29e-07 | 1.0000 | n.s. |
| calhousing | 0.1000 | pearson_adj | -7.48e-05 | 1.0000 | n.s. | -8.3e-06 | 1.0000 | n.s. |
| calhousing | 0.1000 | partialcorr_adj | -0.000119 | 1.0000 | n.s. | -2.88e-06 | 1.0000 | n.s. |
| calhousing | 0.3000 | uniform | -0.00013 | 1.0000 | n.s. | -9.59e-06 | 1.0000 | n.s. |
| calhousing | 0.3000 | pearson_alloc | -0.000154 | 1.0000 | n.s. | 8.03e-06 | 1.0000 | n.s. |
| calhousing | 0.3000 | pearson_adj | -6.16e-05 | 1.0000 | n.s. | -5.83e-06 | 1.0000 | n.s. |
| calhousing | 0.3000 | partialcorr_adj | -2.77e-05 | 1.0000 | n.s. | -6.74e-06 | 1.0000 | n.s. |
| calhousing | 0.5000 | uniform | -5.99e-05 | 1.0000 | n.s. | -2.56e-05 | 1.0000 | n.s. |
| calhousing | 0.5000 | pearson_alloc | -8.56e-05 | 1.0000 | n.s. | -4.56e-06 | 1.0000 | n.s. |
| calhousing | 0.5000 | pearson_adj | -0.000213 | 1.0000 | n.s. | -1.5e-05 | 1.0000 | n.s. |
| calhousing | 0.5000 | partialcorr_adj | -0.000115 | 1.0000 | n.s. | -3.55e-06 | 1.0000 | n.s. |
| calhousing | 1.0000 | uniform | 6.09e-05 | 1.0000 | n.s. | -1.11e-05 | 1.0000 | n.s. |
| calhousing | 1.0000 | pearson_alloc | -0.000119 | 1.0000 | n.s. | 2.77e-05 | 1.0000 | n.s. |
| calhousing | 1.0000 | pearson_adj | 0.000115 | 1.0000 | n.s. | 4.81e-05 | 1.0000 | n.s. |
| calhousing | 1.0000 | partialcorr_adj | 7.81e-05 | 1.0000 | n.s. | 2.95e-05 | 1.0000 | n.s. |
| calhousing | 5.0000 | uniform | 0.0016 | 3.96e-06 | A better | 0.0014 | 5.16e-06 | A better |
| calhousing | 5.0000 | pearson_alloc | -0.000436 | 1.0000 | n.s. | -0.000576 | 0.1491 | n.s. |
| calhousing | 5.0000 | pearson_adj | 0.0025 | 1.79e-07 | A better | 0.0024 | 1.79e-07 | A better |
| calhousing | 5.0000 | partialcorr_adj | 0.0023 | 1.79e-07 | A better | 0.0020 | 3.32e-07 | A better |
| calhousing | 10.0000 | uniform | 0.0050 | 1.79e-07 | A better | 0.0048 | 1.79e-07 | A better |
| calhousing | 10.0000 | pearson_alloc | -0.0045 | 1.79e-07 | B better | -0.0043 | 3.32e-07 | B better |
| calhousing | 10.0000 | pearson_adj | 0.0079 | 1.79e-07 | A better | 0.0076 | 1.79e-07 | A better |
| calhousing | 10.0000 | partialcorr_adj | 0.0068 | 1.79e-07 | A better | 0.0065 | 1.79e-07 | A better |
| appliances | 0.1000 | uniform | 0.0177 | 1.0000 | n.s. | 0.0133 | 0.9397 | n.s. |
| appliances | 0.1000 | pearson_alloc | 0.0325 | 1.0000 | n.s. | 0.000231 | 1.0000 | n.s. |
| appliances | 0.1000 | pearson_adj | 0.0091 | 1.0000 | n.s. | 0.0023 | 1.0000 | n.s. |
| appliances | 0.1000 | partialcorr_adj | -0.0055 | 1.0000 | n.s. | -0.000531 | 1.0000 | n.s. |
| appliances | 0.3000 | uniform | -0.0183 | 1.0000 | n.s. | 0.0035 | 1.0000 | n.s. |
| appliances | 0.3000 | pearson_alloc | -0.0419 | 1.0000 | n.s. | 0.000259 | 1.0000 | n.s. |
| appliances | 0.3000 | pearson_adj | -0.0044 | 1.0000 | n.s. | 0.0048 | 1.0000 | n.s. |
| appliances | 0.3000 | partialcorr_adj | -0.0261 | 1.0000 | n.s. | -0.0016 | 1.0000 | n.s. |
| appliances | 0.5000 | uniform | 0.0512 | 1.0000 | n.s. | -0.0014 | 1.0000 | n.s. |
| appliances | 0.5000 | pearson_alloc | 0.0208 | 1.0000 | n.s. | -0.0029 | 1.0000 | n.s. |
| appliances | 0.5000 | pearson_adj | 0.0048 | 1.0000 | n.s. | -0.0070 | 1.0000 | n.s. |
| appliances | 0.5000 | partialcorr_adj | 0.0178 | 1.0000 | n.s. | -0.0040 | 1.0000 | n.s. |
| appliances | 1.0000 | uniform | -0.0347 | 1.0000 | n.s. | -0.0072 | 1.0000 | n.s. |
| appliances | 1.0000 | pearson_alloc | -0.0066 | 1.0000 | n.s. | -0.0031 | 1.0000 | n.s. |
| appliances | 1.0000 | pearson_adj | -0.0408 | 1.0000 | n.s. | -0.0084 | 1.0000 | n.s. |
| appliances | 1.0000 | partialcorr_adj | -0.0409 | 1.0000 | n.s. | 0.0040 | 1.0000 | n.s. |
| appliances | 5.0000 | uniform | -0.0131 | 1.0000 | n.s. | 0.000901 | 1.0000 | n.s. |
| appliances | 5.0000 | pearson_alloc | 0.0163 | 1.0000 | n.s. | 0.0070 | 1.0000 | n.s. |
| appliances | 5.0000 | pearson_adj | -0.0238 | 1.0000 | n.s. | 0.0064 | 1.0000 | n.s. |
| appliances | 5.0000 | partialcorr_adj | 0.0189 | 1.0000 | n.s. | 0.0154 | 1.0000 | n.s. |
| appliances | 10.0000 | uniform | -0.0201 | 1.0000 | n.s. | -0.0166 | 1.0000 | n.s. |
| appliances | 10.0000 | pearson_alloc | 0.0411 | 1.0000 | n.s. | 0.0175 | 1.0000 | n.s. |
| appliances | 10.0000 | pearson_adj | 0.0249 | 1.0000 | n.s. | 0.0203 | 1.0000 | n.s. |
| appliances | 10.0000 | partialcorr_adj | 0.0154 | 1.0000 | n.s. | 0.0140 | 1.0000 | n.s. |

## Part B. Noise-aware prediction, with and without

Refit models of the stored Uniform and MIC-DP releases; direct = predict_proba on the clean test inputs; difference = noise-aware minus direct per trial; 95% percentile bootstrap interval (10,000 resamples); paired two-sided Wilcoxon, Holm over the 24 tests. const = number of trials (of 30) with a constant test prediction.

| dataset | method | eps | AUC noise-aware | AUC direct | const na | const direct | mean diff (na - direct) | ci_lo | ci_hi | p_raw | p_holm | outcome |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Adult | Uniform | 0.1000 | 0.531 ± 0.129 | 0.503 ± 0.027 | 0 | 22 | 0.0283 | -0.0160 | 0.0707 | 0.1241 | 0.3844 | n.s. |
| Adult | Uniform | 0.3000 | 0.606 ± 0.104 | 0.510 ± 0.072 | 0 | 14 | 0.0958 | 0.0467 | 0.1443 | 0.000313 | 0.0025 | A better |
| Adult | Uniform | 0.5000 | 0.668 ± 0.094 | 0.474 ± 0.087 | 0 | 4 | 0.1943 | 0.1522 | 0.2374 | 1.86e-08 | 2.42e-07 | A better |
| Adult | Uniform | 1.0000 | 0.708 ± 0.077 | 0.495 ± 0.070 | 0 | 2 | 0.2130 | 0.1825 | 0.2433 | 1.86e-09 | 4.47e-08 | A better |
| Adult | Uniform | 5.0000 | 0.799 ± 0.010 | 0.545 ± 0.103 | 0 | 0 | 0.2534 | 0.2183 | 0.2870 | 1.86e-09 | 4.47e-08 | A better |
| Adult | Uniform | 10.0000 | 0.801 ± 0.005 | 0.652 ± 0.095 | 0 | 0 | 0.1490 | 0.1171 | 0.1837 | 1.86e-09 | 4.47e-08 | A better |
| Adult | MIC-DP | 0.1000 | 0.547 ± 0.148 | 0.493 ± 0.030 | 0 | 23 | 0.0539 | -0.000709 | 0.1111 | 0.0961 | 0.3844 | n.s. |
| Adult | MIC-DP | 0.3000 | 0.572 ± 0.128 | 0.492 ± 0.062 | 0 | 12 | 0.0802 | 0.0278 | 0.1316 | 0.0120 | 0.0602 | n.s. |
| Adult | MIC-DP | 0.5000 | 0.612 ± 0.137 | 0.523 ± 0.072 | 0 | 11 | 0.0884 | 0.0281 | 0.1434 | 0.0076 | 0.0457 | A better |
| Adult | MIC-DP | 1.0000 | 0.705 ± 0.097 | 0.467 ± 0.102 | 0 | 4 | 0.2375 | 0.1941 | 0.2798 | 9.31e-09 | 1.3e-07 | A better |
| Adult | MIC-DP | 5.0000 | 0.820 ± 0.017 | 0.548 ± 0.143 | 0 | 0 | 0.2720 | 0.2227 | 0.3235 | 1.86e-09 | 4.47e-08 | A better |
| Adult | MIC-DP | 10.0000 | 0.823 ± 0.008 | 0.565 ± 0.145 | 0 | 0 | 0.2578 | 0.2070 | 0.3089 | 1.86e-09 | 4.47e-08 | A better |
| CDC-Diabetes | Uniform | 0.1000 | 0.510 ± 0.120 | 0.502 ± 0.041 | 0 | 20 | 0.0083 | -0.0383 | 0.0507 | 0.2710 | 0.5420 | n.s. |
| CDC-Diabetes | Uniform | 0.3000 | 0.583 ± 0.106 | 0.498 ± 0.058 | 0 | 13 | 0.0848 | 0.0497 | 0.1198 | 2.37e-05 | 0.000237 | A better |
| CDC-Diabetes | Uniform | 0.5000 | 0.626 ± 0.107 | 0.471 ± 0.067 | 0 | 9 | 0.1546 | 0.1086 | 0.1950 | 9.98e-07 | 1.1e-05 | A better |
| CDC-Diabetes | Uniform | 1.0000 | 0.712 ± 0.051 | 0.517 ± 0.097 | 0 | 4 | 0.1952 | 0.1526 | 0.2358 | 3.54e-08 | 4.25e-07 | A better |
| CDC-Diabetes | Uniform | 5.0000 | 0.792 ± 0.005 | 0.644 ± 0.076 | 0 | 0 | 0.1487 | 0.1224 | 0.1769 | 1.86e-09 | 4.47e-08 | A better |
| CDC-Diabetes | Uniform | 10.0000 | 0.798 ± 0.003 | 0.727 ± 0.043 | 0 | 0 | 0.0703 | 0.0558 | 0.0854 | 1.86e-09 | 4.47e-08 | A better |
| CDC-Diabetes | MIC-DP | 0.1000 | 0.498 ± 0.125 | 0.511 ± 0.045 | 0 | 24 | -0.0132 | -0.0656 | 0.0389 | 0.5425 | 0.5425 | n.s. |
| CDC-Diabetes | MIC-DP | 0.3000 | 0.607 ± 0.113 | 0.513 ± 0.063 | 0 | 16 | 0.0935 | 0.0510 | 0.1333 | 0.000283 | 0.0025 | A better |
| CDC-Diabetes | MIC-DP | 0.5000 | 0.583 ± 0.128 | 0.499 ± 0.042 | 0 | 15 | 0.0843 | 0.0355 | 0.1307 | 0.0028 | 0.0194 | A better |
| CDC-Diabetes | MIC-DP | 1.0000 | 0.690 ± 0.056 | 0.499 ± 0.056 | 0 | 12 | 0.1917 | 0.1634 | 0.2216 | 1.86e-09 | 4.47e-08 | A better |
| CDC-Diabetes | MIC-DP | 5.0000 | 0.790 ± 0.008 | 0.658 ± 0.112 | 0 | 0 | 0.1318 | 0.0939 | 0.1733 | 1.86e-09 | 4.47e-08 | A better |
| CDC-Diabetes | MIC-DP | 10.0000 | 0.798 ± 0.004 | 0.735 ± 0.060 | 0 | 0 | 0.0629 | 0.0442 | 0.0870 | 1.86e-09 | 4.47e-08 | A better |

Claim "noise-aware prediction improves ROC-AUC for Uniform at every budget on both datasets": does not hold. Uniform cells with noise-aware significantly better: 10 of 12; significantly worse: 0; n.s.: 2. MIC-DP: 9 better, 0 worse, 3 n.s. of 12.

## Part C. Fine-tuning, repaired

Repaired = start of the continued stage clipped to [-3, 3] and l2_regularization = 1.0 for trees 101-200; noise-aware prediction clip(E[F_pre], -3, 3) + E[F_cont] (draft3_notes.md, amendment). Stored = Public-Pretrain + FT of the public-baselines run (raw_trials_public.parquet), with its F7 outcomes (Holm over 144 tests). F7r = the new tests of the repaired models (Holm within the 48 tests of each dataset); A = repaired model, B = Public-Only at the same size or the private parent (stored Uniform for Uniform-FT, stored MIC-DP for MIC-DP-FT). Max leaf = largest |leaf value| of the continued trees (101-200). frac start clipped = share of privatized training rows whose pretrained raw score lay outside [-3, 3] (mean over trials). AUC direct repaired: clip(F_pre(x), -3, 3) + F_cont(x) on the clean test inputs, for reference.

### Adult

| method | size | eps | AUC repaired | const repaired | max leaf (max over trials) | max leaf (median) | AUC stored | const stored | stored F7 vs Public-Only | stored F7 vs parent | AUC direct repaired | frac start clipped | repaired F7r vs Public-Only | repaired F7r vs parent |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Public-Pretrain + Uniform-FT | 15% | 0.100 | 0.834 ± 0.002 | 0 | 1.558 | 1.549 | 0.500 ± 0.000 | 30 | B better | n.s. | 0.902 ± 0.011 | 0.836 | B better | A better |
| Public-Pretrain + Uniform-FT | 15% | 0.300 | 0.834 ± 0.002 | 0 | 1.563 | 1.552 | 0.500 ± 0.000 | 30 | B better | B better | 0.892 ± 0.038 | 0.836 | B better | A better |
| Public-Pretrain + Uniform-FT | 15% | 0.500 | 0.835 ± 0.002 | 0 | 1.561 | 1.550 | 0.500 ± 0.000 | 30 | B better | B better | 0.880 ± 0.027 | 0.836 | B better | A better |
| Public-Pretrain + Uniform-FT | 15% | 1.000 | 0.836 ± 0.002 | 0 | 1.561 | 1.550 | 0.500 ± 0.000 | 30 | B better | B better | 0.860 ± 0.031 | 0.836 | B better | A better |
| Public-Pretrain + Uniform-FT | 15% | 5.000 | 0.841 ± 0.002 | 0 | 1.558 | 1.546 | 0.500 ± 0.000 | 30 | B better | B better | 0.825 ± 0.023 | 0.836 | B better | A better |
| Public-Pretrain + Uniform-FT | 15% | 10.000 | 0.844 ± 0.001 | 0 | 1.555 | 1.541 | 0.502 ± 0.009 | 25 | B better | B better | 0.834 ± 0.014 | 0.835 | B better | A better |
| Public-Pretrain + Uniform-FT | 5% | 0.100 | 0.858 ± 0.001 | 0 | 1.582 | 1.573 | 0.500 ± 0.000 | 30 | B better | n.s. | 0.882 ± 0.014 | 0.874 | B better | A better |
| Public-Pretrain + Uniform-FT | 5% | 0.300 | 0.858 ± 0.001 | 0 | 1.589 | 1.575 | 0.500 ± 0.000 | 30 | B better | B better | 0.876 ± 0.031 | 0.874 | B better | A better |
| Public-Pretrain + Uniform-FT | 5% | 0.500 | 0.859 ± 0.001 | 0 | 1.585 | 1.575 | 0.500 ± 0.000 | 30 | B better | B better | 0.855 ± 0.034 | 0.874 | B better | A better |
| Public-Pretrain + Uniform-FT | 5% | 1.000 | 0.859 ± 0.001 | 0 | 1.585 | 1.573 | 0.500 ± 0.000 | 30 | B better | B better | 0.842 ± 0.035 | 0.873 | B better | A better |
| Public-Pretrain + Uniform-FT | 5% | 5.000 | 0.860 ± 0.001 | 0 | 1.579 | 1.570 | 0.500 ± 0.000 | 30 | B better | B better | 0.804 ± 0.017 | 0.871 | B better | A better |
| Public-Pretrain + Uniform-FT | 5% | 10.000 | 0.859 ± 0.001 | 0 | 1.578 | 1.564 | 0.504 ± 0.020 | 26 | B better | B better | 0.808 ± 0.011 | 0.868 | B better | A better |
| Public-Pretrain + MIC-DP-FT | 15% | 0.100 | 0.835 ± 0.006 | 0 | 1.557 | 1.549 | 0.500 ± 0.000 | 30 | B better | n.s. | 0.900 ± 0.018 | 0.837 | B better | A better |
| Public-Pretrain + MIC-DP-FT | 15% | 0.300 | 0.835 ± 0.017 | 0 | 1.558 | 1.551 | 0.500 ± 0.000 | 30 | B better | n.s. | 0.891 ± 0.025 | 0.837 | B better | A better |
| Public-Pretrain + MIC-DP-FT | 15% | 0.500 | 0.829 ± 0.022 | 0 | 1.557 | 1.549 | 0.505 ± 0.028 | 29 | B better | B better | 0.884 ± 0.028 | 0.837 | B better | A better |
| Public-Pretrain + MIC-DP-FT | 15% | 1.000 | 0.810 ± 0.117 | 0 | 1.587 | 1.551 | 0.508 ± 0.044 | 29 | B better | B better | 0.866 ± 0.034 | 0.830 | B better | A better |
| Public-Pretrain + MIC-DP-FT | 15% | 5.000 | 0.837 ± 0.016 | 0 | 1.583 | 1.572 | 0.500 ± 0.000 | 30 | B better | B better | 0.823 ± 0.022 | 0.869 | B better | A better |
| Public-Pretrain + MIC-DP-FT | 15% | 10.000 | 0.847 ± 0.008 | 0 | 1.632 | 1.574 | 0.509 ± 0.046 | 28 | B better | B better | 0.816 ± 0.028 | 0.870 | B better | A better |
| Public-Pretrain + MIC-DP-FT | 5% | 0.100 | 0.858 ± 0.004 | 0 | 1.580 | 1.572 | 0.500 ± 0.000 | 30 | B better | n.s. | 0.877 ± 0.019 | 0.875 | B better | A better |
| Public-Pretrain + MIC-DP-FT | 5% | 0.300 | 0.856 ± 0.012 | 0 | 1.583 | 1.574 | 0.500 ± 0.000 | 30 | B better | n.s. | 0.872 ± 0.026 | 0.874 | B better | A better |
| Public-Pretrain + MIC-DP-FT | 5% | 0.500 | 0.850 ± 0.016 | 0 | 1.580 | 1.574 | 0.500 ± 0.000 | 30 | B better | B better | 0.861 ± 0.037 | 0.874 | B better | A better |
| Public-Pretrain + MIC-DP-FT | 5% | 1.000 | 0.828 ± 0.116 | 0 | 1.587 | 1.574 | 0.500 ± 0.000 | 30 | B better | B better | 0.846 ± 0.037 | 0.857 | B better | A better |
| Public-Pretrain + MIC-DP-FT | 5% | 5.000 | 0.849 ± 0.009 | 0 | 1.581 | 1.573 | 0.500 ± 0.000 | 30 | B better | B better | 0.801 ± 0.019 | 0.874 | B better | A better |
| Public-Pretrain + MIC-DP-FT | 5% | 10.000 | 0.851 ± 0.005 | 0 | 1.632 | 1.572 | 0.500 ± 0.000 | 30 | B better | B better | 0.794 ± 0.019 | 0.882 | B better | A better |

F7r tests (Adult):

| method | size | eps | vs | mean_a | mean_b | diff | ci_lo | ci_hi | n_a_wins | n_b_wins | p_raw | p_holm | outcome |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Public-Pretrain + Uniform-FT | 15% | 0.1000 | Public-Only | 0.8339 | 0.9109 | -0.0770 | -0.0778 | -0.0762 | 0 | 30 | 1.86e-09 | 8.94e-08 | B better |
| Public-Pretrain + Uniform-FT | 15% | 0.1000 | parent | 0.8339 | 0.5315 | 0.3025 | 0.2590 | 0.3495 | 30 | 0 | 1.86e-09 | 8.94e-08 | A better |
| Public-Pretrain + Uniform-FT | 15% | 0.3000 | Public-Only | 0.8342 | 0.9109 | -0.0767 | -0.0773 | -0.0761 | 0 | 30 | 1.86e-09 | 8.94e-08 | B better |
| Public-Pretrain + Uniform-FT | 15% | 0.3000 | parent | 0.8342 | 0.6056 | 0.2286 | 0.1919 | 0.2656 | 30 | 0 | 1.86e-09 | 8.94e-08 | A better |
| Public-Pretrain + Uniform-FT | 15% | 0.5000 | Public-Only | 0.8350 | 0.9109 | -0.0759 | -0.0767 | -0.0752 | 0 | 30 | 1.86e-09 | 8.94e-08 | B better |
| Public-Pretrain + Uniform-FT | 15% | 0.5000 | parent | 0.8350 | 0.6679 | 0.1671 | 0.1357 | 0.2005 | 30 | 0 | 1.86e-09 | 8.94e-08 | A better |
| Public-Pretrain + Uniform-FT | 15% | 1.0000 | Public-Only | 0.8361 | 0.9109 | -0.0748 | -0.0753 | -0.0742 | 0 | 30 | 1.86e-09 | 8.94e-08 | B better |
| Public-Pretrain + Uniform-FT | 15% | 1.0000 | parent | 0.8361 | 0.7083 | 0.1278 | 0.1022 | 0.1556 | 29 | 1 | 3.73e-09 | 8.94e-08 | A better |
| Public-Pretrain + Uniform-FT | 15% | 5.0000 | Public-Only | 0.8415 | 0.9109 | -0.0694 | -0.0700 | -0.0688 | 0 | 30 | 1.86e-09 | 8.94e-08 | B better |
| Public-Pretrain + Uniform-FT | 15% | 5.0000 | parent | 0.8415 | 0.7985 | 0.0429 | 0.0396 | 0.0463 | 30 | 0 | 1.86e-09 | 8.94e-08 | A better |
| Public-Pretrain + Uniform-FT | 15% | 10.0000 | Public-Only | 0.8441 | 0.9109 | -0.0668 | -0.0673 | -0.0663 | 0 | 30 | 1.86e-09 | 8.94e-08 | B better |
| Public-Pretrain + Uniform-FT | 15% | 10.0000 | parent | 0.8441 | 0.8010 | 0.0431 | 0.0416 | 0.0446 | 30 | 0 | 1.86e-09 | 8.94e-08 | A better |
| Public-Pretrain + Uniform-FT | 5% | 0.1000 | Public-Only | 0.8582 | 0.8970 | -0.0388 | -0.0393 | -0.0383 | 0 | 30 | 1.86e-09 | 8.94e-08 | B better |
| Public-Pretrain + Uniform-FT | 5% | 0.1000 | parent | 0.8582 | 0.5315 | 0.3267 | 0.2819 | 0.3723 | 30 | 0 | 1.86e-09 | 8.94e-08 | A better |
| Public-Pretrain + Uniform-FT | 5% | 0.3000 | Public-Only | 0.8582 | 0.8970 | -0.0388 | -0.0391 | -0.0383 | 0 | 30 | 1.86e-09 | 8.94e-08 | B better |
| Public-Pretrain + Uniform-FT | 5% | 0.3000 | parent | 0.8582 | 0.6056 | 0.2526 | 0.2162 | 0.2896 | 30 | 0 | 1.86e-09 | 8.94e-08 | A better |
| Public-Pretrain + Uniform-FT | 5% | 0.5000 | Public-Only | 0.8586 | 0.8970 | -0.0384 | -0.0388 | -0.0380 | 0 | 30 | 1.86e-09 | 8.94e-08 | B better |
| Public-Pretrain + Uniform-FT | 5% | 0.5000 | parent | 0.8586 | 0.6679 | 0.1907 | 0.1587 | 0.2254 | 30 | 0 | 1.86e-09 | 8.94e-08 | A better |
| Public-Pretrain + Uniform-FT | 5% | 1.0000 | Public-Only | 0.8591 | 0.8970 | -0.0379 | -0.0383 | -0.0375 | 0 | 30 | 1.86e-09 | 8.94e-08 | B better |
| Public-Pretrain + Uniform-FT | 5% | 1.0000 | parent | 0.8591 | 0.7083 | 0.1508 | 0.1256 | 0.1796 | 30 | 0 | 1.86e-09 | 8.94e-08 | A better |
| Public-Pretrain + Uniform-FT | 5% | 5.0000 | Public-Only | 0.8597 | 0.8970 | -0.0372 | -0.0376 | -0.0368 | 0 | 30 | 1.86e-09 | 8.94e-08 | B better |
| Public-Pretrain + Uniform-FT | 5% | 5.0000 | parent | 0.8597 | 0.7985 | 0.0612 | 0.0579 | 0.0644 | 30 | 0 | 1.86e-09 | 8.94e-08 | A better |
| Public-Pretrain + Uniform-FT | 5% | 10.0000 | Public-Only | 0.8586 | 0.8970 | -0.0384 | -0.0388 | -0.0380 | 0 | 30 | 1.86e-09 | 8.94e-08 | B better |
| Public-Pretrain + Uniform-FT | 5% | 10.0000 | parent | 0.8586 | 0.8010 | 0.0576 | 0.0561 | 0.0590 | 30 | 0 | 1.86e-09 | 8.94e-08 | A better |
| Public-Pretrain + MIC-DP-FT | 15% | 0.1000 | Public-Only | 0.8350 | 0.9109 | -0.0759 | -0.0779 | -0.0736 | 0 | 30 | 1.86e-09 | 8.94e-08 | B better |
| Public-Pretrain + MIC-DP-FT | 15% | 0.1000 | parent | 0.8350 | 0.5468 | 0.2882 | 0.2371 | 0.3399 | 30 | 0 | 1.86e-09 | 8.94e-08 | A better |
| Public-Pretrain + MIC-DP-FT | 15% | 0.3000 | Public-Only | 0.8349 | 0.9109 | -0.0761 | -0.0823 | -0.0704 | 0 | 30 | 1.86e-09 | 8.94e-08 | B better |
| Public-Pretrain + MIC-DP-FT | 15% | 0.3000 | parent | 0.8349 | 0.5722 | 0.2626 | 0.2176 | 0.3103 | 30 | 0 | 1.86e-09 | 8.94e-08 | A better |
| Public-Pretrain + MIC-DP-FT | 15% | 0.5000 | Public-Only | 0.8291 | 0.9109 | -0.0818 | -0.0899 | -0.0742 | 0 | 30 | 1.86e-09 | 8.94e-08 | B better |
| Public-Pretrain + MIC-DP-FT | 15% | 0.5000 | parent | 0.8291 | 0.6117 | 0.2174 | 0.1714 | 0.2692 | 30 | 0 | 1.86e-09 | 8.94e-08 | A better |
| Public-Pretrain + MIC-DP-FT | 15% | 1.0000 | Public-Only | 0.8103 | 0.9109 | -0.1006 | -0.1465 | -0.0732 | 0 | 30 | 1.86e-09 | 8.94e-08 | B better |
| Public-Pretrain + MIC-DP-FT | 15% | 1.0000 | parent | 0.8103 | 0.7048 | 0.1055 | 0.0450 | 0.1564 | 28 | 2 | 4.42e-06 | 1.14e-05 | A better |
| Public-Pretrain + MIC-DP-FT | 15% | 5.0000 | Public-Only | 0.8370 | 0.9109 | -0.0740 | -0.0794 | -0.0685 | 0 | 30 | 1.86e-09 | 8.94e-08 | B better |
| Public-Pretrain + MIC-DP-FT | 15% | 5.0000 | parent | 0.8370 | 0.8196 | 0.0174 | 0.0084 | 0.0261 | 25 | 5 | 0.000872 | 0.000872 | A better |
| Public-Pretrain + MIC-DP-FT | 15% | 10.0000 | Public-Only | 0.8468 | 0.9109 | -0.0641 | -0.0669 | -0.0613 | 0 | 30 | 1.86e-09 | 8.94e-08 | B better |
| Public-Pretrain + MIC-DP-FT | 15% | 10.0000 | parent | 0.8468 | 0.8227 | 0.0242 | 0.0210 | 0.0271 | 29 | 1 | 3.73e-09 | 8.94e-08 | A better |
| Public-Pretrain + MIC-DP-FT | 5% | 0.1000 | Public-Only | 0.8584 | 0.8970 | -0.0385 | -0.0397 | -0.0373 | 0 | 30 | 1.86e-09 | 8.94e-08 | B better |
| Public-Pretrain + MIC-DP-FT | 5% | 0.1000 | parent | 0.8584 | 0.5468 | 0.3117 | 0.2603 | 0.3650 | 30 | 0 | 1.86e-09 | 8.94e-08 | A better |
| Public-Pretrain + MIC-DP-FT | 5% | 0.3000 | Public-Only | 0.8565 | 0.8970 | -0.0405 | -0.0452 | -0.0365 | 0 | 30 | 1.86e-09 | 8.94e-08 | B better |
| Public-Pretrain + MIC-DP-FT | 5% | 0.3000 | parent | 0.8565 | 0.5722 | 0.2842 | 0.2374 | 0.3314 | 30 | 0 | 1.86e-09 | 8.94e-08 | A better |
| Public-Pretrain + MIC-DP-FT | 5% | 0.5000 | Public-Only | 0.8498 | 0.8970 | -0.0472 | -0.0530 | -0.0418 | 0 | 30 | 1.86e-09 | 8.94e-08 | B better |
| Public-Pretrain + MIC-DP-FT | 5% | 0.5000 | parent | 0.8498 | 0.6117 | 0.2380 | 0.1925 | 0.2885 | 30 | 0 | 1.86e-09 | 8.94e-08 | A better |
| Public-Pretrain + MIC-DP-FT | 5% | 1.0000 | Public-Only | 0.8281 | 0.8970 | -0.0689 | -0.1150 | -0.0427 | 0 | 30 | 1.86e-09 | 8.94e-08 | B better |
| Public-Pretrain + MIC-DP-FT | 5% | 1.0000 | parent | 0.8281 | 0.7048 | 0.1232 | 0.0649 | 0.1737 | 29 | 1 | 3.79e-06 | 1.14e-05 | A better |
| Public-Pretrain + MIC-DP-FT | 5% | 5.0000 | Public-Only | 0.8487 | 0.8970 | -0.0482 | -0.0514 | -0.0453 | 0 | 30 | 1.86e-09 | 8.94e-08 | B better |
| Public-Pretrain + MIC-DP-FT | 5% | 5.0000 | parent | 0.8487 | 0.8196 | 0.0292 | 0.0239 | 0.0345 | 29 | 1 | 5.59e-09 | 8.94e-08 | A better |
| Public-Pretrain + MIC-DP-FT | 5% | 10.0000 | Public-Only | 0.8513 | 0.8970 | -0.0456 | -0.0474 | -0.0438 | 0 | 30 | 1.86e-09 | 8.94e-08 | B better |
| Public-Pretrain + MIC-DP-FT | 5% | 10.0000 | parent | 0.8513 | 0.8227 | 0.0286 | 0.0265 | 0.0308 | 30 | 0 | 1.86e-09 | 8.94e-08 | A better |

### CDC-Diabetes (check)

| method | size | eps | AUC repaired | const repaired | max leaf (max over trials) | max leaf (median) | AUC stored | const stored | stored F7 vs Public-Only | stored F7 vs parent | AUC direct repaired | frac start clipped | repaired F7r vs Public-Only | repaired F7r vs parent |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Public-Pretrain + Uniform-FT | 15% | 0.100 | 0.197 ± 0.002 | 0 | 0.553 | 0.424 | 0.713 ± 0.061 | 0 | B better | A better | 0.787 ± 0.011 | 0.532 | B better | B better |
| Public-Pretrain + Uniform-FT | 15% | 0.300 | 0.197 ± 0.002 | 0 | 0.519 | 0.428 | 0.734 ± 0.055 | 0 | B better | A better | 0.777 ± 0.013 | 0.532 | B better | B better |
| Public-Pretrain + Uniform-FT | 15% | 0.500 | 0.198 ± 0.002 | 0 | 0.627 | 0.437 | 0.740 ± 0.060 | 0 | B better | A better | 0.758 ± 0.015 | 0.532 | B better | B better |
| Public-Pretrain + Uniform-FT | 15% | 1.000 | 0.199 ± 0.002 | 0 | 0.532 | 0.425 | 0.774 ± 0.019 | 0 | B better | A better | 0.759 ± 0.014 | 0.532 | B better | B better |
| Public-Pretrain + Uniform-FT | 15% | 5.000 | 0.331 ± 0.014 | 0 | 0.665 | 0.434 | 0.795 ± 0.012 | 0 | B better | A better | 0.714 ± 0.020 | 0.533 | B better | B better |
| Public-Pretrain + Uniform-FT | 15% | 10.000 | 0.661 ± 0.014 | 0 | 0.529 | 0.456 | 0.801 ± 0.002 | 0 | B better | A better | 0.716 ± 0.019 | 0.531 | B better | B better |
| Public-Pretrain + Uniform-FT | 5% | 0.100 | 0.199 ± 0.002 | 0 | 0.806 | 0.705 | 0.754 ± 0.032 | 0 | B better | A better | 0.720 ± 0.056 | 0.614 | B better | B better |
| Public-Pretrain + Uniform-FT | 5% | 0.300 | 0.200 ± 0.003 | 0 | 0.890 | 0.691 | 0.744 ± 0.044 | 0 | B better | A better | 0.661 ± 0.037 | 0.615 | B better | B better |
| Public-Pretrain + Uniform-FT | 5% | 0.500 | 0.200 ± 0.002 | 0 | 0.746 | 0.681 | 0.741 ± 0.032 | 0 | B better | A better | 0.637 ± 0.038 | 0.615 | B better | B better |
| Public-Pretrain + Uniform-FT | 5% | 1.000 | 0.201 ± 0.003 | 0 | 0.719 | 0.669 | 0.749 ± 0.031 | 0 | B better | A better | 0.626 ± 0.038 | 0.615 | B better | B better |
| Public-Pretrain + Uniform-FT | 5% | 5.000 | 0.217 ± 0.005 | 0 | 0.790 | 0.625 | 0.727 ± 0.063 | 0 | B better | B better | 0.596 ± 0.039 | 0.617 | B better | B better |
| Public-Pretrain + Uniform-FT | 5% | 10.000 | 0.292 ± 0.015 | 0 | 0.640 | 0.541 | 0.744 ± 0.054 | 0 | B better | B better | 0.613 ± 0.025 | 0.617 | B better | B better |
| Public-Pretrain + MIC-DP-FT | 15% | 0.100 | 0.197 ± 0.002 | 0 | 0.589 | 0.418 | 0.725 ± 0.062 | 0 | B better | A better | 0.788 ± 0.012 | 0.532 | B better | B better |
| Public-Pretrain + MIC-DP-FT | 15% | 0.300 | 0.201 ± 0.010 | 0 | 0.562 | 0.428 | 0.749 ± 0.029 | 0 | B better | A better | 0.779 ± 0.012 | 0.532 | B better | B better |
| Public-Pretrain + MIC-DP-FT | 15% | 0.500 | 0.202 ± 0.008 | 0 | 0.503 | 0.426 | 0.721 ± 0.062 | 0 | B better | A better | 0.770 ± 0.015 | 0.532 | B better | B better |
| Public-Pretrain + MIC-DP-FT | 15% | 1.000 | 0.205 ± 0.014 | 0 | 0.666 | 0.433 | 0.756 ± 0.031 | 0 | B better | A better | 0.761 ± 0.017 | 0.532 | B better | B better |
| Public-Pretrain + MIC-DP-FT | 15% | 5.000 | 0.367 ± 0.157 | 0 | 0.551 | 0.420 | 0.794 ± 0.006 | 0 | B better | A better | 0.719 ± 0.027 | 0.522 | B better | B better |
| Public-Pretrain + MIC-DP-FT | 15% | 10.000 | 0.750 ± 0.052 | 0 | 0.489 | 0.412 | 0.794 ± 0.017 | 0 | B better | n.s. | 0.703 ± 0.037 | 0.484 | B better | B better |
| Public-Pretrain + MIC-DP-FT | 5% | 0.100 | 0.199 ± 0.002 | 0 | 0.931 | 0.697 | 0.741 ± 0.049 | 0 | B better | A better | 0.718 ± 0.056 | 0.615 | B better | B better |
| Public-Pretrain + MIC-DP-FT | 5% | 0.300 | 0.202 ± 0.006 | 0 | 0.891 | 0.687 | 0.743 ± 0.041 | 0 | B better | A better | 0.688 ± 0.045 | 0.615 | B better | B better |
| Public-Pretrain + MIC-DP-FT | 5% | 0.500 | 0.203 ± 0.007 | 0 | 0.784 | 0.689 | 0.747 ± 0.040 | 0 | B better | A better | 0.664 ± 0.048 | 0.615 | B better | B better |
| Public-Pretrain + MIC-DP-FT | 5% | 1.000 | 0.207 ± 0.013 | 0 | 0.841 | 0.681 | 0.733 ± 0.051 | 0 | B better | n.s. | 0.646 ± 0.050 | 0.615 | B better | B better |
| Public-Pretrain + MIC-DP-FT | 5% | 5.000 | 0.263 ± 0.110 | 0 | 0.901 | 0.606 | 0.703 ± 0.111 | 2 | B better | B better | 0.599 ± 0.043 | 0.609 | B better | B better |
| Public-Pretrain + MIC-DP-FT | 5% | 10.000 | 0.498 ± 0.175 | 0 | 1.062 | 0.563 | 0.715 ± 0.096 | 3 | B better | B better | 0.604 ± 0.044 | 0.593 | B better | B better |

F7r tests (CDC-Diabetes):

| method | size | eps | vs | mean_a | mean_b | diff | ci_lo | ci_hi | n_a_wins | n_b_wins | p_raw | p_holm | outcome |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Public-Pretrain + Uniform-FT | 15% | 0.1000 | Public-Only | 0.1974 | 0.8037 | -0.6063 | -0.6071 | -0.6054 | 0 | 30 | 1.86e-09 | 8.94e-08 | B better |
| Public-Pretrain + Uniform-FT | 15% | 0.1000 | parent | 0.1974 | 0.5104 | -0.3130 | -0.3538 | -0.2694 | 0 | 30 | 1.86e-09 | 8.94e-08 | B better |
| Public-Pretrain + Uniform-FT | 15% | 0.3000 | Public-Only | 0.1971 | 0.8037 | -0.6066 | -0.6072 | -0.6059 | 0 | 30 | 1.86e-09 | 8.94e-08 | B better |
| Public-Pretrain + Uniform-FT | 15% | 0.3000 | parent | 0.1971 | 0.5827 | -0.3856 | -0.4217 | -0.3478 | 0 | 30 | 1.86e-09 | 8.94e-08 | B better |
| Public-Pretrain + Uniform-FT | 15% | 0.5000 | Public-Only | 0.1976 | 0.8037 | -0.6061 | -0.6068 | -0.6053 | 0 | 30 | 1.86e-09 | 8.94e-08 | B better |
| Public-Pretrain + Uniform-FT | 15% | 0.5000 | parent | 0.1976 | 0.6259 | -0.4283 | -0.4654 | -0.3885 | 0 | 30 | 1.86e-09 | 8.94e-08 | B better |
| Public-Pretrain + Uniform-FT | 15% | 1.0000 | Public-Only | 0.1989 | 0.8037 | -0.6048 | -0.6056 | -0.6039 | 0 | 30 | 1.86e-09 | 8.94e-08 | B better |
| Public-Pretrain + Uniform-FT | 15% | 1.0000 | parent | 0.1989 | 0.7117 | -0.5128 | -0.5296 | -0.4937 | 0 | 30 | 1.86e-09 | 8.94e-08 | B better |
| Public-Pretrain + Uniform-FT | 15% | 5.0000 | Public-Only | 0.3310 | 0.8037 | -0.4726 | -0.4778 | -0.4675 | 0 | 30 | 1.86e-09 | 8.94e-08 | B better |
| Public-Pretrain + Uniform-FT | 15% | 5.0000 | parent | 0.3310 | 0.7923 | -0.4613 | -0.4666 | -0.4558 | 0 | 30 | 1.86e-09 | 8.94e-08 | B better |
| Public-Pretrain + Uniform-FT | 15% | 10.0000 | Public-Only | 0.6606 | 0.8037 | -0.1431 | -0.1481 | -0.1381 | 0 | 30 | 1.86e-09 | 8.94e-08 | B better |
| Public-Pretrain + Uniform-FT | 15% | 10.0000 | parent | 0.6606 | 0.7976 | -0.1370 | -0.1419 | -0.1322 | 0 | 30 | 1.86e-09 | 8.94e-08 | B better |
| Public-Pretrain + Uniform-FT | 5% | 0.1000 | Public-Only | 0.1991 | 0.7881 | -0.5889 | -0.5898 | -0.5880 | 0 | 30 | 1.86e-09 | 8.94e-08 | B better |
| Public-Pretrain + Uniform-FT | 5% | 0.1000 | parent | 0.1991 | 0.5104 | -0.3113 | -0.3514 | -0.2685 | 0 | 30 | 1.86e-09 | 8.94e-08 | B better |
| Public-Pretrain + Uniform-FT | 5% | 0.3000 | Public-Only | 0.1998 | 0.7881 | -0.5882 | -0.5892 | -0.5871 | 0 | 30 | 1.86e-09 | 8.94e-08 | B better |
| Public-Pretrain + Uniform-FT | 5% | 0.3000 | parent | 0.1998 | 0.5827 | -0.3829 | -0.4187 | -0.3456 | 0 | 30 | 1.86e-09 | 8.94e-08 | B better |
| Public-Pretrain + Uniform-FT | 5% | 0.5000 | Public-Only | 0.2005 | 0.7881 | -0.5876 | -0.5884 | -0.5868 | 0 | 30 | 1.86e-09 | 8.94e-08 | B better |
| Public-Pretrain + Uniform-FT | 5% | 0.5000 | parent | 0.2005 | 0.6259 | -0.4255 | -0.4617 | -0.3875 | 0 | 30 | 1.86e-09 | 8.94e-08 | B better |
| Public-Pretrain + Uniform-FT | 5% | 1.0000 | Public-Only | 0.2011 | 0.7881 | -0.5869 | -0.5878 | -0.5860 | 0 | 30 | 1.86e-09 | 8.94e-08 | B better |
| Public-Pretrain + Uniform-FT | 5% | 1.0000 | parent | 0.2011 | 0.7117 | -0.5106 | -0.5271 | -0.4921 | 0 | 30 | 1.86e-09 | 8.94e-08 | B better |
| Public-Pretrain + Uniform-FT | 5% | 5.0000 | Public-Only | 0.2170 | 0.7881 | -0.5711 | -0.5727 | -0.5695 | 0 | 30 | 1.86e-09 | 8.94e-08 | B better |
| Public-Pretrain + Uniform-FT | 5% | 5.0000 | parent | 0.2170 | 0.7923 | -0.5754 | -0.5771 | -0.5736 | 0 | 30 | 1.86e-09 | 8.94e-08 | B better |
| Public-Pretrain + Uniform-FT | 5% | 10.0000 | Public-Only | 0.2922 | 0.7881 | -0.4958 | -0.5011 | -0.4904 | 0 | 30 | 1.86e-09 | 8.94e-08 | B better |
| Public-Pretrain + Uniform-FT | 5% | 10.0000 | parent | 0.2922 | 0.7976 | -0.5054 | -0.5105 | -0.5002 | 0 | 30 | 1.86e-09 | 8.94e-08 | B better |
| Public-Pretrain + MIC-DP-FT | 15% | 0.1000 | Public-Only | 0.1968 | 0.8037 | -0.6069 | -0.6075 | -0.6063 | 0 | 30 | 1.86e-09 | 8.94e-08 | B better |
| Public-Pretrain + MIC-DP-FT | 15% | 0.1000 | parent | 0.1968 | 0.4976 | -0.3009 | -0.3449 | -0.2587 | 0 | 30 | 1.86e-09 | 8.94e-08 | B better |
| Public-Pretrain + MIC-DP-FT | 15% | 0.3000 | Public-Only | 0.2009 | 0.8037 | -0.6028 | -0.6057 | -0.5988 | 0 | 30 | 1.86e-09 | 8.94e-08 | B better |
| Public-Pretrain + MIC-DP-FT | 15% | 0.3000 | parent | 0.2009 | 0.6069 | -0.4059 | -0.4431 | -0.3660 | 0 | 30 | 1.86e-09 | 8.94e-08 | B better |
| Public-Pretrain + MIC-DP-FT | 15% | 0.5000 | Public-Only | 0.2018 | 0.8037 | -0.6019 | -0.6047 | -0.5989 | 0 | 30 | 1.86e-09 | 8.94e-08 | B better |
| Public-Pretrain + MIC-DP-FT | 15% | 0.5000 | parent | 0.2018 | 0.5830 | -0.3812 | -0.4246 | -0.3355 | 0 | 30 | 1.86e-09 | 8.94e-08 | B better |
| Public-Pretrain + MIC-DP-FT | 15% | 1.0000 | Public-Only | 0.2048 | 0.8037 | -0.5988 | -0.6031 | -0.5934 | 0 | 30 | 1.86e-09 | 8.94e-08 | B better |
| Public-Pretrain + MIC-DP-FT | 15% | 1.0000 | parent | 0.2048 | 0.6904 | -0.4856 | -0.5062 | -0.4638 | 0 | 30 | 1.86e-09 | 8.94e-08 | B better |
| Public-Pretrain + MIC-DP-FT | 15% | 5.0000 | Public-Only | 0.3670 | 0.8037 | -0.4367 | -0.4875 | -0.3776 | 1 | 29 | 3.73e-09 | 8.94e-08 | B better |
| Public-Pretrain + MIC-DP-FT | 15% | 5.0000 | parent | 0.3670 | 0.7896 | -0.4226 | -0.4727 | -0.3618 | 2 | 28 | 9.31e-09 | 8.94e-08 | B better |
| Public-Pretrain + MIC-DP-FT | 15% | 10.0000 | Public-Only | 0.7496 | 0.8037 | -0.0541 | -0.0726 | -0.0368 | 2 | 28 | 9.31e-09 | 8.94e-08 | B better |
| Public-Pretrain + MIC-DP-FT | 15% | 10.0000 | parent | 0.7496 | 0.7983 | -0.0487 | -0.0688 | -0.0306 | 5 | 25 | 1.68e-06 | 1.68e-06 | B better |
| Public-Pretrain + MIC-DP-FT | 5% | 0.1000 | Public-Only | 0.1993 | 0.7881 | -0.5888 | -0.5896 | -0.5879 | 0 | 30 | 1.86e-09 | 8.94e-08 | B better |
| Public-Pretrain + MIC-DP-FT | 5% | 0.1000 | parent | 0.1993 | 0.4976 | -0.2984 | -0.3420 | -0.2569 | 0 | 30 | 1.86e-09 | 8.94e-08 | B better |
| Public-Pretrain + MIC-DP-FT | 5% | 0.3000 | Public-Only | 0.2022 | 0.7881 | -0.5859 | -0.5878 | -0.5838 | 0 | 30 | 1.86e-09 | 8.94e-08 | B better |
| Public-Pretrain + MIC-DP-FT | 5% | 0.3000 | parent | 0.2022 | 0.6069 | -0.4047 | -0.4421 | -0.3637 | 0 | 30 | 1.86e-09 | 8.94e-08 | B better |
| Public-Pretrain + MIC-DP-FT | 5% | 0.5000 | Public-Only | 0.2030 | 0.7881 | -0.5850 | -0.5874 | -0.5823 | 0 | 30 | 1.86e-09 | 8.94e-08 | B better |
| Public-Pretrain + MIC-DP-FT | 5% | 0.5000 | parent | 0.2030 | 0.5830 | -0.3800 | -0.4220 | -0.3355 | 0 | 30 | 1.86e-09 | 8.94e-08 | B better |
| Public-Pretrain + MIC-DP-FT | 5% | 1.0000 | Public-Only | 0.2070 | 0.7881 | -0.5810 | -0.5850 | -0.5761 | 0 | 30 | 1.86e-09 | 8.94e-08 | B better |
| Public-Pretrain + MIC-DP-FT | 5% | 1.0000 | parent | 0.2070 | 0.6904 | -0.4834 | -0.5026 | -0.4633 | 0 | 30 | 1.86e-09 | 8.94e-08 | B better |
| Public-Pretrain + MIC-DP-FT | 5% | 5.0000 | Public-Only | 0.2633 | 0.7881 | -0.5248 | -0.5551 | -0.4804 | 0 | 30 | 1.86e-09 | 8.94e-08 | B better |
| Public-Pretrain + MIC-DP-FT | 5% | 5.0000 | parent | 0.2633 | 0.7896 | -0.5263 | -0.5577 | -0.4810 | 0 | 30 | 1.86e-09 | 8.94e-08 | B better |
| Public-Pretrain + MIC-DP-FT | 5% | 10.0000 | Public-Only | 0.4978 | 0.7881 | -0.2902 | -0.3510 | -0.2281 | 0 | 30 | 1.86e-09 | 8.94e-08 | B better |
| Public-Pretrain + MIC-DP-FT | 5% | 10.0000 | parent | 0.4978 | 0.7983 | -0.3005 | -0.3634 | -0.2387 | 0 | 30 | 1.86e-09 | 8.94e-08 | B better |

### Diagnostic of the noise-aware prediction (exploratory, trial 0; diagnostics/draft3_ft_clip.py)

Range of the exact E[F_pre(x + eta)] (pretrained trees with the baseline) over the test records, share of test records with |E[F_pre]| > 3 (where the clip of the noise-aware prediction is active), and ROC-AUC of E[F_pre], of E[F_cont] (continued trees), of the reported noise-aware prediction clip(E[F_pre], -3, 3) + E[F_cont], and of E[F_pre] + E[F_cont] without the clip. Same code, seeds and models as the run; nothing was changed.

| dataset | method | size | eps | E_pre_min | E_pre_max | share_clip_active | auc_E_pre | auc_E_cont | auc_noise_aware | auc_E_pre_plus_E_cont_unclipped |
|---|---|---|---|---|---|---|---|---|---|---|
| Adult | Public-Pretrain + Uniform-FT | 15% | 0.100 | 0.235 | 0.289 | 0.000 | 0.846 | 0.156 | 0.834 | 0.834 |
| Adult | Public-Pretrain + Uniform-FT | 5% | 0.100 | 0.989 | 1.049 | 0.000 | 0.864 | 0.138 | 0.857 | 0.857 |
| Adult | Public-Pretrain + MIC-DP-FT | 15% | 0.100 | 0.237 | 0.278 | 0.000 | 0.847 | 0.143 | 0.834 | 0.834 |
| Adult | Public-Pretrain + MIC-DP-FT | 5% | 0.100 | 0.991 | 1.037 | 0.000 | 0.866 | 0.135 | 0.859 | 0.859 |
| Adult | Public-Pretrain + Uniform-FT | 15% | 0.300 | 0.219 | 0.368 | 0.000 | 0.846 | 0.144 | 0.833 | 0.833 |
| Adult | Public-Pretrain + Uniform-FT | 5% | 0.300 | 0.969 | 1.135 | 0.000 | 0.865 | 0.137 | 0.858 | 0.858 |
| Adult | Public-Pretrain + MIC-DP-FT | 15% | 0.300 | 0.225 | 0.353 | 0.000 | 0.859 | 0.138 | 0.850 | 0.850 |
| Adult | Public-Pretrain + MIC-DP-FT | 5% | 0.300 | 0.974 | 1.120 | 0.000 | 0.871 | 0.137 | 0.866 | 0.866 |
| Adult | Public-Pretrain + Uniform-FT | 15% | 0.500 | 0.204 | 0.442 | 0.000 | 0.846 | 0.154 | 0.837 | 0.837 |
| Adult | Public-Pretrain + Uniform-FT | 5% | 0.500 | 0.951 | 1.215 | 0.000 | 0.865 | 0.145 | 0.860 | 0.860 |
| Adult | Public-Pretrain + MIC-DP-FT | 15% | 0.500 | 0.212 | 0.403 | 0.000 | 0.846 | 0.149 | 0.835 | 0.835 |
| Adult | Public-Pretrain + MIC-DP-FT | 5% | 0.500 | 0.961 | 1.173 | 0.000 | 0.865 | 0.141 | 0.859 | 0.859 |
| Adult | Public-Pretrain + Uniform-FT | 15% | 1.000 | 0.169 | 0.616 | 0.000 | 0.846 | 0.162 | 0.838 | 0.838 |
| Adult | Public-Pretrain + Uniform-FT | 5% | 1.000 | 0.908 | 1.405 | 0.000 | 0.865 | 0.146 | 0.860 | 0.860 |
| Adult | Public-Pretrain + MIC-DP-FT | 15% | 1.000 | 0.159 | 0.497 | 0.000 | 0.875 | 0.131 | 0.870 | 0.870 |
| Adult | Public-Pretrain + MIC-DP-FT | 5% | 1.000 | 0.905 | 1.275 | 0.000 | 0.880 | 0.126 | 0.876 | 0.876 |
| Adult | Public-Pretrain + Uniform-FT | 15% | 5.000 | -0.067 | 1.764 | 0.000 | 0.847 | 0.257 | 0.838 | 0.838 |
| Adult | Public-Pretrain + Uniform-FT | 5% | 5.000 | 0.615 | 2.665 | 0.000 | 0.866 | 0.200 | 0.858 | 0.858 |
| Adult | Public-Pretrain + MIC-DP-FT | 15% | 5.000 | 0.236 | 1.719 | 0.000 | 0.847 | 0.344 | 0.847 | 0.847 |
| Adult | Public-Pretrain + MIC-DP-FT | 5% | 5.000 | 0.520 | 2.115 | 0.000 | 0.866 | 0.337 | 0.858 | 0.858 |
| Adult | Public-Pretrain + Uniform-FT | 15% | 10.000 | -0.301 | 2.855 | 0.000 | 0.849 | 0.404 | 0.843 | 0.843 |
| Adult | Public-Pretrain + Uniform-FT | 5% | 10.000 | 0.319 | 3.870 | 0.004 | 0.867 | 0.393 | 0.858 | 0.858 |
| Adult | Public-Pretrain + MIC-DP-FT | 15% | 10.000 | -0.028 | 2.844 | 0.000 | 0.841 | 0.463 | 0.843 | 0.843 |
| Adult | Public-Pretrain + MIC-DP-FT | 5% | 10.000 | 0.238 | 3.301 | 0.003 | 0.862 | 0.482 | 0.850 | 0.850 |
| CDC-Diabetes | Public-Pretrain + Uniform-FT | 15% | 0.100 | -3.147 | -3.129 | 1.000 | 0.807 | 0.198 | 0.198 | 0.805 |
| CDC-Diabetes | Public-Pretrain + Uniform-FT | 5% | 0.100 | -3.525 | -3.500 | 1.000 | 0.803 | 0.198 | 0.198 | 0.800 |
| CDC-Diabetes | Public-Pretrain + MIC-DP-FT | 15% | 0.100 | -3.146 | -3.131 | 1.000 | 0.807 | 0.195 | 0.195 | 0.804 |
| CDC-Diabetes | Public-Pretrain + MIC-DP-FT | 5% | 0.100 | -3.523 | -3.502 | 1.000 | 0.803 | 0.198 | 0.198 | 0.802 |
| CDC-Diabetes | Public-Pretrain + Uniform-FT | 15% | 0.300 | -3.161 | -3.111 | 1.000 | 0.807 | 0.196 | 0.196 | 0.801 |
| CDC-Diabetes | Public-Pretrain + Uniform-FT | 5% | 0.300 | -3.543 | -3.474 | 1.000 | 0.803 | 0.198 | 0.198 | 0.802 |
| CDC-Diabetes | Public-Pretrain + MIC-DP-FT | 15% | 0.300 | -3.153 | -3.121 | 1.000 | 0.806 | 0.205 | 0.205 | 0.805 |
| CDC-Diabetes | Public-Pretrain + MIC-DP-FT | 5% | 0.300 | -3.533 | -3.489 | 1.000 | 0.799 | 0.201 | 0.201 | 0.796 |
| CDC-Diabetes | Public-Pretrain + Uniform-FT | 15% | 0.500 | -3.174 | -3.094 | 1.000 | 0.807 | 0.199 | 0.199 | 0.805 |
| CDC-Diabetes | Public-Pretrain + Uniform-FT | 5% | 0.500 | -3.560 | -3.450 | 1.000 | 0.803 | 0.202 | 0.202 | 0.804 |
| CDC-Diabetes | Public-Pretrain + MIC-DP-FT | 15% | 0.500 | -3.167 | -3.103 | 1.000 | 0.807 | 0.201 | 0.201 | 0.808 |
| CDC-Diabetes | Public-Pretrain + MIC-DP-FT | 5% | 0.500 | -3.551 | -3.463 | 1.000 | 0.803 | 0.198 | 0.198 | 0.801 |
| CDC-Diabetes | Public-Pretrain + Uniform-FT | 15% | 1.000 | -3.204 | -3.053 | 1.000 | 0.807 | 0.199 | 0.199 | 0.808 |
| CDC-Diabetes | Public-Pretrain + Uniform-FT | 5% | 1.000 | -3.600 | -3.393 | 1.000 | 0.803 | 0.200 | 0.200 | 0.803 |
| CDC-Diabetes | Public-Pretrain + MIC-DP-FT | 15% | 1.000 | -3.188 | -3.065 | 1.000 | 0.810 | 0.193 | 0.193 | 0.805 |
| CDC-Diabetes | Public-Pretrain + MIC-DP-FT | 5% | 1.000 | -3.589 | -3.410 | 1.000 | 0.801 | 0.204 | 0.204 | 0.804 |
| CDC-Diabetes | Public-Pretrain + Uniform-FT | 15% | 5.000 | -3.415 | -2.777 | 0.882 | 0.808 | 0.213 | 0.314 | 0.807 |
| CDC-Diabetes | Public-Pretrain + Uniform-FT | 5% | 5.000 | -3.873 | -3.009 | 1.000 | 0.804 | 0.213 | 0.213 | 0.807 |
| CDC-Diabetes | Public-Pretrain + MIC-DP-FT | 15% | 5.000 | -3.363 | -2.843 | 0.935 | 0.807 | 0.216 | 0.262 | 0.806 |
| CDC-Diabetes | Public-Pretrain + MIC-DP-FT | 5% | 5.000 | -3.807 | -3.101 | 1.000 | 0.804 | 0.216 | 0.216 | 0.808 |
| CDC-Diabetes | Public-Pretrain + Uniform-FT | 15% | 10.000 | -3.640 | -2.498 | 0.726 | 0.808 | 0.299 | 0.642 | 0.808 |
| CDC-Diabetes | Public-Pretrain + Uniform-FT | 5% | 10.000 | -4.149 | -2.617 | 0.976 | 0.806 | 0.259 | 0.270 | 0.810 |
| CDC-Diabetes | Public-Pretrain + MIC-DP-FT | 15% | 10.000 | -3.663 | -2.438 | 0.663 | 0.806 | 0.537 | 0.763 | 0.805 |
| CDC-Diabetes | Public-Pretrain + MIC-DP-FT | 5% | 10.000 | -4.951 | -2.404 | 0.966 | 0.788 | 0.285 | 0.300 | 0.802 |

## Part D. Attribute-sampling baseline

Random-k: per record, a uniform random k-subset released with noise sqrt(k) c(eps, delta) per coordinate, the other coordinates set to the public-split mean; Random-k-fixed: one random k-subset for all records. Noise-aware prediction as in draft3_notes.md. Uniform, MIC-DP and MIC-Public: stored values. const = trials with a constant test prediction.

Why Random-k satisfies (eps, delta)-DP for the record: the subset of each record is drawn independently of the data, and given the subset the release of that record is a Gaussian mechanism on k coordinates of range one (L2 sensitivity sqrt(k)) with noise sqrt(k) c(eps, delta), which is (eps, delta)-DP by the analytic Gaussian calibration; the mean fill uses public data only, the other records do not depend on the changed record, and a mixture over a data-independent subset of (eps, delta)-DP mechanisms is (eps, delta)-DP. Random-k-fixed is the case of one subset for all records.

### Adult

| eps | Uniform | MIC-DP | MIC-Public (15%) | Random-k k=3 | const Random-k k=3 | Random-k k=6 | const Random-k k=6 | Random-k-fixed k=3 | const Random-k-fixed k=3 | Random-k-fixed k=6 | const Random-k-fixed k=6 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 0.1000 | 0.531 ± 0.129 | 0.547 ± 0.148 | 0.547 ± 0.176 | 0.547 ± 0.143 | 0 | 0.503 ± 0.118 | 0 | 0.536 ± 0.160 | 0 | 0.525 ± 0.133 | 0 |
| 0.3000 | 0.606 ± 0.104 | 0.572 ± 0.128 | 0.672 ± 0.131 | 0.564 ± 0.139 | 0 | 0.581 ± 0.125 | 0 | 0.607 ± 0.121 | 0 | 0.617 ± 0.124 | 0 |
| 0.5000 | 0.668 ± 0.094 | 0.612 ± 0.137 | 0.680 ± 0.121 | 0.654 ± 0.110 | 0 | 0.624 ± 0.114 | 0 | 0.582 ± 0.121 | 0 | 0.595 ± 0.118 | 0 |
| 1.0000 | 0.708 ± 0.077 | 0.705 ± 0.097 | 0.780 ± 0.047 | 0.711 ± 0.080 | 0 | 0.714 ± 0.066 | 0 | 0.673 ± 0.056 | 0 | 0.690 ± 0.088 | 0 |
| 5.0000 | 0.799 ± 0.010 | 0.820 ± 0.017 | 0.820 ± 0.008 | 0.792 ± 0.010 | 0 | 0.791 ± 0.009 | 0 | 0.711 ± 0.056 | 0 | 0.758 ± 0.036 | 0 |
| 10.0000 | 0.801 ± 0.005 | 0.823 ± 0.008 | 0.827 ± 0.005 | 0.800 ± 0.005 | 0 | 0.800 ± 0.005 | 0 | 0.720 ± 0.064 | 0 | 0.773 ± 0.038 | 0 |

### CDC-Diabetes

| eps | Uniform | MIC-DP | MIC-Public (15%) | Random-k k=5 | const Random-k k=5 | Random-k k=11 | const Random-k k=11 | Random-k-fixed k=5 | const Random-k-fixed k=5 | Random-k-fixed k=11 | const Random-k-fixed k=11 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 0.1000 | 0.510 ± 0.120 | 0.498 ± 0.125 | 0.546 ± 0.181 | 0.542 ± 0.097 | 0 | 0.540 ± 0.124 | 0 | 0.518 ± 0.120 | 0 | 0.562 ± 0.115 | 0 |
| 0.3000 | 0.583 ± 0.106 | 0.607 ± 0.113 | 0.698 ± 0.118 | 0.610 ± 0.110 | 0 | 0.576 ± 0.125 | 0 | 0.595 ± 0.101 | 0 | 0.590 ± 0.108 | 0 |
| 0.5000 | 0.626 ± 0.107 | 0.583 ± 0.128 | 0.716 ± 0.109 | 0.642 ± 0.099 | 0 | 0.666 ± 0.076 | 0 | 0.620 ± 0.097 | 0 | 0.649 ± 0.085 | 0 |
| 1.0000 | 0.712 ± 0.051 | 0.690 ± 0.056 | 0.792 ± 0.011 | 0.728 ± 0.038 | 0 | 0.729 ± 0.043 | 0 | 0.683 ± 0.064 | 0 | 0.713 ± 0.053 | 0 |
| 5.0000 | 0.792 ± 0.005 | 0.790 ± 0.008 | 0.799 ± 0.002 | 0.795 ± 0.005 | 0 | 0.793 ± 0.005 | 0 | 0.704 ± 0.052 | 0 | 0.760 ± 0.027 | 0 |
| 10.0000 | 0.798 ± 0.003 | 0.798 ± 0.004 | 0.800 ± 0.001 | 0.799 ± 0.002 | 0 | 0.798 ± 0.002 | 0 | 0.711 ± 0.041 | 0 | 0.766 ± 0.022 | 0 |

Family F9 (96 tests, Holm), outcome counts by comparison: vs MIC-DP: B better 12; vs MIC-DP: n.s. 36; vs Uniform: B better 8; vs Uniform: n.s. 40

| dataset | variant | k | eps | vs | mean_a | mean_b | diff | ci_lo | ci_hi | n_a_wins | n_b_wins | p_raw | p_holm | outcome |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Adult | per_record | 3 | 0.1000 | Uniform | 0.5468 | 0.5315 | 0.0153 | -0.0504 | 0.0824 | 17 | 13 | 0.8712 | 1.0000 | n.s. |
| Adult | per_record | 3 | 0.1000 | MIC-DP | 0.5468 | 0.5468 | 2.87e-05 | -0.0722 | 0.0705 | 15 | 15 | 0.7611 | 1.0000 | n.s. |
| Adult | per_record | 6 | 0.1000 | Uniform | 0.5034 | 0.5315 | -0.0281 | -0.0854 | 0.0334 | 12 | 18 | 0.2621 | 1.0000 | n.s. |
| Adult | per_record | 6 | 0.1000 | MIC-DP | 0.5034 | 0.5468 | -0.0434 | -0.1023 | 0.0208 | 13 | 17 | 0.1347 | 1.0000 | n.s. |
| Adult | fixed | 3 | 0.1000 | Uniform | 0.5358 | 0.5315 | 0.0043 | -0.0645 | 0.0754 | 14 | 16 | 1.0000 | 1.0000 | n.s. |
| Adult | fixed | 3 | 0.1000 | MIC-DP | 0.5358 | 0.5468 | -0.0110 | -0.0805 | 0.0573 | 16 | 14 | 0.8712 | 1.0000 | n.s. |
| Adult | fixed | 6 | 0.1000 | Uniform | 0.5245 | 0.5315 | -0.0070 | -0.0713 | 0.0596 | 12 | 18 | 0.8078 | 1.0000 | n.s. |
| Adult | fixed | 6 | 0.1000 | MIC-DP | 0.5245 | 0.5468 | -0.0223 | -0.0896 | 0.0451 | 15 | 15 | 0.5838 | 1.0000 | n.s. |
| Adult | per_record | 3 | 0.3000 | Uniform | 0.5635 | 0.6056 | -0.0421 | -0.1088 | 0.0234 | 13 | 17 | 0.2801 | 1.0000 | n.s. |
| Adult | per_record | 3 | 0.3000 | MIC-DP | 0.5635 | 0.5722 | -0.0087 | -0.0807 | 0.0577 | 17 | 13 | 0.7611 | 1.0000 | n.s. |
| Adult | per_record | 6 | 0.3000 | Uniform | 0.5808 | 0.6056 | -0.0248 | -0.0728 | 0.0261 | 13 | 17 | 0.2894 | 1.0000 | n.s. |
| Adult | per_record | 6 | 0.3000 | MIC-DP | 0.5808 | 0.5722 | 0.0086 | -0.0435 | 0.0612 | 16 | 14 | 0.7611 | 1.0000 | n.s. |
| Adult | fixed | 3 | 0.3000 | Uniform | 0.6073 | 0.6056 | 0.0017 | -0.0616 | 0.0584 | 16 | 14 | 0.6120 | 1.0000 | n.s. |
| Adult | fixed | 3 | 0.3000 | MIC-DP | 0.6073 | 0.5722 | 0.0351 | -0.0321 | 0.1029 | 19 | 11 | 0.2621 | 1.0000 | n.s. |
| Adult | fixed | 6 | 0.3000 | Uniform | 0.6170 | 0.6056 | 0.0114 | -0.0485 | 0.0705 | 17 | 13 | 0.7766 | 1.0000 | n.s. |
| Adult | fixed | 6 | 0.3000 | MIC-DP | 0.6170 | 0.5722 | 0.0448 | -0.0146 | 0.1006 | 20 | 10 | 0.1191 | 1.0000 | n.s. |
| Adult | per_record | 3 | 0.5000 | Uniform | 0.6539 | 0.6679 | -0.0140 | -0.0640 | 0.0341 | 14 | 16 | 0.6850 | 1.0000 | n.s. |
| Adult | per_record | 3 | 0.5000 | MIC-DP | 0.6539 | 0.6117 | 0.0422 | -0.0157 | 0.1016 | 20 | 10 | 0.3285 | 1.0000 | n.s. |
| Adult | per_record | 6 | 0.5000 | Uniform | 0.6244 | 0.6679 | -0.0435 | -0.0879 | 0.000712 | 11 | 19 | 0.0961 | 1.0000 | n.s. |
| Adult | per_record | 6 | 0.5000 | MIC-DP | 0.6244 | 0.6117 | 0.0127 | -0.0568 | 0.0838 | 13 | 17 | 0.9515 | 1.0000 | n.s. |
| Adult | fixed | 3 | 0.5000 | Uniform | 0.5815 | 0.6679 | -0.0863 | -0.1354 | -0.0393 | 9 | 21 | 0.0028 | 0.2047 | n.s. |
| Adult | fixed | 3 | 0.5000 | MIC-DP | 0.5815 | 0.6117 | -0.0302 | -0.0853 | 0.0282 | 12 | 18 | 0.2129 | 1.0000 | n.s. |
| Adult | fixed | 6 | 0.5000 | Uniform | 0.5952 | 0.6679 | -0.0726 | -0.1224 | -0.0232 | 8 | 22 | 0.0113 | 0.7912 | n.s. |
| Adult | fixed | 6 | 0.5000 | MIC-DP | 0.5952 | 0.6117 | -0.0165 | -0.0868 | 0.0554 | 13 | 17 | 0.6263 | 1.0000 | n.s. |
| Adult | per_record | 3 | 1.0000 | Uniform | 0.7113 | 0.7083 | 0.0030 | -0.0295 | 0.0345 | 18 | 12 | 0.5838 | 1.0000 | n.s. |
| Adult | per_record | 3 | 1.0000 | MIC-DP | 0.7113 | 0.7048 | 0.0064 | -0.0398 | 0.0498 | 14 | 16 | 0.8712 | 1.0000 | n.s. |
| Adult | per_record | 6 | 1.0000 | Uniform | 0.7144 | 0.7083 | 0.0062 | -0.0220 | 0.0340 | 14 | 16 | 0.8553 | 1.0000 | n.s. |
| Adult | per_record | 6 | 1.0000 | MIC-DP | 0.7144 | 0.7048 | 0.0096 | -0.0345 | 0.0569 | 14 | 16 | 0.9677 | 1.0000 | n.s. |
| Adult | fixed | 3 | 1.0000 | Uniform | 0.6734 | 0.7083 | -0.0349 | -0.0704 | 0.0031 | 11 | 19 | 0.0667 | 1.0000 | n.s. |
| Adult | fixed | 3 | 1.0000 | MIC-DP | 0.6734 | 0.7048 | -0.0314 | -0.0692 | 0.0084 | 12 | 18 | 0.1347 | 1.0000 | n.s. |
| Adult | fixed | 6 | 1.0000 | Uniform | 0.6899 | 0.7083 | -0.0184 | -0.0617 | 0.0272 | 11 | 19 | 0.2801 | 1.0000 | n.s. |
| Adult | fixed | 6 | 1.0000 | MIC-DP | 0.6899 | 0.7048 | -0.0149 | -0.0573 | 0.0290 | 15 | 15 | 0.3285 | 1.0000 | n.s. |
| Adult | per_record | 3 | 5.0000 | Uniform | 0.7924 | 0.7985 | -0.0061 | -0.0112 | -0.0015 | 11 | 19 | 0.0497 | 1.0000 | n.s. |
| Adult | per_record | 3 | 5.0000 | MIC-DP | 0.7924 | 0.8196 | -0.0271 | -0.0339 | -0.0205 | 2 | 28 | 4.66e-08 | 3.91e-06 | B better |
| Adult | per_record | 6 | 5.0000 | Uniform | 0.7905 | 0.7985 | -0.0080 | -0.0134 | -0.0027 | 11 | 19 | 0.0093 | 0.6697 | n.s. |
| Adult | per_record | 6 | 5.0000 | MIC-DP | 0.7905 | 0.8196 | -0.0290 | -0.0348 | -0.0229 | 1 | 29 | 4.66e-08 | 3.91e-06 | B better |
| Adult | fixed | 3 | 5.0000 | Uniform | 0.7113 | 0.7985 | -0.0873 | -0.1091 | -0.0673 | 1 | 29 | 5.59e-09 | 4.86e-07 | B better |
| Adult | fixed | 3 | 5.0000 | MIC-DP | 0.7113 | 0.8196 | -0.1083 | -0.1306 | -0.0876 | 0 | 30 | 1.86e-09 | 1.79e-07 | B better |
| Adult | fixed | 6 | 5.0000 | Uniform | 0.7578 | 0.7985 | -0.0407 | -0.0548 | -0.0273 | 4 | 26 | 2.35e-06 | 0.000186 | B better |
| Adult | fixed | 6 | 5.0000 | MIC-DP | 0.7578 | 0.8196 | -0.0617 | -0.0771 | -0.0462 | 2 | 28 | 1.02e-07 | 8.3e-06 | B better |
| Adult | per_record | 3 | 10.0000 | Uniform | 0.8004 | 0.8010 | -0.000618 | -0.0037 | 0.0024 | 14 | 16 | 0.6850 | 1.0000 | n.s. |
| Adult | per_record | 3 | 10.0000 | MIC-DP | 0.8004 | 0.8227 | -0.0223 | -0.0257 | -0.0189 | 0 | 30 | 1.86e-09 | 1.79e-07 | B better |
| Adult | per_record | 6 | 10.0000 | Uniform | 0.8005 | 0.8010 | -0.000517 | -0.0032 | 0.0021 | 15 | 15 | 0.9677 | 1.0000 | n.s. |
| Adult | per_record | 6 | 10.0000 | MIC-DP | 0.8005 | 0.8227 | -0.0222 | -0.0254 | -0.0190 | 0 | 30 | 1.86e-09 | 1.79e-07 | B better |
| Adult | fixed | 3 | 10.0000 | Uniform | 0.7200 | 0.8010 | -0.0810 | -0.1034 | -0.0589 | 2 | 28 | 1.3e-07 | 1.04e-05 | B better |
| Adult | fixed | 3 | 10.0000 | MIC-DP | 0.7200 | 0.8227 | -0.1026 | -0.1252 | -0.0802 | 1 | 29 | 1.3e-08 | 1.11e-06 | B better |
| Adult | fixed | 6 | 10.0000 | Uniform | 0.7725 | 0.8010 | -0.0285 | -0.0428 | -0.0152 | 6 | 24 | 0.000209 | 0.0161 | B better |
| Adult | fixed | 6 | 10.0000 | MIC-DP | 0.7725 | 0.8227 | -0.0501 | -0.0656 | -0.0362 | 4 | 26 | 8.01e-08 | 6.57e-06 | B better |
| CDC-Diabetes | per_record | 5 | 0.1000 | Uniform | 0.5422 | 0.5104 | 0.0318 | -0.0228 | 0.0893 | 17 | 13 | 0.4280 | 1.0000 | n.s. |
| CDC-Diabetes | per_record | 5 | 0.1000 | MIC-DP | 0.5422 | 0.4976 | 0.0446 | -0.0033 | 0.0940 | 17 | 13 | 0.0667 | 1.0000 | n.s. |
| CDC-Diabetes | per_record | 11 | 0.1000 | Uniform | 0.5396 | 0.5104 | 0.0292 | -0.0256 | 0.0830 | 18 | 12 | 0.2710 | 1.0000 | n.s. |
| CDC-Diabetes | per_record | 11 | 0.1000 | MIC-DP | 0.5396 | 0.4976 | 0.0420 | -0.0177 | 0.1005 | 16 | 14 | 0.2449 | 1.0000 | n.s. |
| CDC-Diabetes | fixed | 5 | 0.1000 | Uniform | 0.5178 | 0.5104 | 0.0074 | -0.0540 | 0.0709 | 12 | 18 | 0.9515 | 1.0000 | n.s. |
| CDC-Diabetes | fixed | 5 | 0.1000 | MIC-DP | 0.5178 | 0.4976 | 0.0202 | -0.0375 | 0.0809 | 14 | 16 | 0.9354 | 1.0000 | n.s. |
| CDC-Diabetes | fixed | 11 | 0.1000 | Uniform | 0.5621 | 0.5104 | 0.0517 | -0.0059 | 0.1076 | 20 | 10 | 0.0803 | 1.0000 | n.s. |
| CDC-Diabetes | fixed | 11 | 0.1000 | MIC-DP | 0.5621 | 0.4976 | 0.0645 | 0.0140 | 0.1154 | 21 | 9 | 0.0175 | 1.0000 | n.s. |
| CDC-Diabetes | per_record | 5 | 0.3000 | Uniform | 0.6098 | 0.5827 | 0.0271 | -0.0295 | 0.0853 | 16 | 14 | 0.4771 | 1.0000 | n.s. |
| CDC-Diabetes | per_record | 5 | 0.3000 | MIC-DP | 0.6098 | 0.6069 | 0.0030 | -0.0622 | 0.0661 | 16 | 14 | 0.9032 | 1.0000 | n.s. |
| CDC-Diabetes | per_record | 11 | 0.3000 | Uniform | 0.5763 | 0.5827 | -0.0064 | -0.0606 | 0.0465 | 13 | 17 | 0.6263 | 1.0000 | n.s. |
| CDC-Diabetes | per_record | 11 | 0.3000 | MIC-DP | 0.5763 | 0.6069 | -0.0306 | -0.0897 | 0.0321 | 14 | 16 | 0.2710 | 1.0000 | n.s. |
| CDC-Diabetes | fixed | 5 | 0.3000 | Uniform | 0.5951 | 0.5827 | 0.0124 | -0.0348 | 0.0581 | 18 | 12 | 0.5561 | 1.0000 | n.s. |
| CDC-Diabetes | fixed | 5 | 0.3000 | MIC-DP | 0.5951 | 0.6069 | -0.0118 | -0.0606 | 0.0359 | 14 | 16 | 0.9193 | 1.0000 | n.s. |
| CDC-Diabetes | fixed | 11 | 0.3000 | Uniform | 0.5902 | 0.5827 | 0.0075 | -0.0450 | 0.0591 | 17 | 13 | 0.7303 | 1.0000 | n.s. |
| CDC-Diabetes | fixed | 11 | 0.3000 | MIC-DP | 0.5902 | 0.6069 | -0.0167 | -0.0751 | 0.0420 | 13 | 17 | 0.5978 | 1.0000 | n.s. |
| CDC-Diabetes | per_record | 5 | 0.5000 | Uniform | 0.6417 | 0.6259 | 0.0158 | -0.0334 | 0.0645 | 15 | 15 | 0.5978 | 1.0000 | n.s. |
| CDC-Diabetes | per_record | 5 | 0.5000 | MIC-DP | 0.6417 | 0.5830 | 0.0587 | 0.0049 | 0.1131 | 17 | 13 | 0.0767 | 1.0000 | n.s. |
| CDC-Diabetes | per_record | 11 | 0.5000 | Uniform | 0.6662 | 0.6259 | 0.0402 | -0.0098 | 0.0919 | 18 | 12 | 0.1403 | 1.0000 | n.s. |
| CDC-Diabetes | per_record | 11 | 0.5000 | MIC-DP | 0.6662 | 0.5830 | 0.0832 | 0.0396 | 0.1312 | 23 | 7 | 0.000872 | 0.0663 | n.s. |
| CDC-Diabetes | fixed | 5 | 0.5000 | Uniform | 0.6199 | 0.6259 | -0.0060 | -0.0517 | 0.0403 | 13 | 17 | 0.6850 | 1.0000 | n.s. |
| CDC-Diabetes | fixed | 5 | 0.5000 | MIC-DP | 0.6199 | 0.5830 | 0.0369 | -0.0198 | 0.0958 | 16 | 14 | 0.4045 | 1.0000 | n.s. |
| CDC-Diabetes | fixed | 11 | 0.5000 | Uniform | 0.6487 | 0.6259 | 0.0228 | -0.0195 | 0.0658 | 18 | 12 | 0.2801 | 1.0000 | n.s. |
| CDC-Diabetes | fixed | 11 | 0.5000 | MIC-DP | 0.6487 | 0.5830 | 0.0657 | 0.0093 | 0.1255 | 17 | 13 | 0.0667 | 1.0000 | n.s. |
| CDC-Diabetes | per_record | 5 | 1.0000 | Uniform | 0.7284 | 0.7117 | 0.0166 | -0.0075 | 0.0414 | 16 | 14 | 0.3285 | 1.0000 | n.s. |
| CDC-Diabetes | per_record | 5 | 1.0000 | MIC-DP | 0.7284 | 0.6904 | 0.0380 | 0.0142 | 0.0624 | 21 | 9 | 0.0058 | 0.4217 | n.s. |
| CDC-Diabetes | per_record | 11 | 1.0000 | Uniform | 0.7285 | 0.7117 | 0.0168 | -0.0058 | 0.0397 | 18 | 12 | 0.2129 | 1.0000 | n.s. |
| CDC-Diabetes | per_record | 11 | 1.0000 | MIC-DP | 0.7285 | 0.6904 | 0.0381 | 0.0155 | 0.0617 | 23 | 7 | 0.0024 | 0.1776 | n.s. |
| CDC-Diabetes | fixed | 5 | 1.0000 | Uniform | 0.6831 | 0.7117 | -0.0286 | -0.0495 | -0.0069 | 8 | 22 | 0.0099 | 0.7051 | n.s. |
| CDC-Diabetes | fixed | 5 | 1.0000 | MIC-DP | 0.6831 | 0.6904 | -0.0073 | -0.0308 | 0.0165 | 16 | 14 | 0.6850 | 1.0000 | n.s. |
| CDC-Diabetes | fixed | 11 | 1.0000 | Uniform | 0.7130 | 0.7117 | 0.0013 | -0.0249 | 0.0291 | 15 | 15 | 0.8872 | 1.0000 | n.s. |
| CDC-Diabetes | fixed | 11 | 1.0000 | MIC-DP | 0.7130 | 0.6904 | 0.0226 | -0.0016 | 0.0452 | 23 | 7 | 0.0364 | 1.0000 | n.s. |
| CDC-Diabetes | per_record | 5 | 5.0000 | Uniform | 0.7945 | 0.7923 | 0.0022 | -2.37e-05 | 0.0045 | 19 | 11 | 0.1229 | 1.0000 | n.s. |
| CDC-Diabetes | per_record | 5 | 5.0000 | MIC-DP | 0.7945 | 0.7896 | 0.0049 | 0.0018 | 0.0081 | 19 | 11 | 0.0128 | 0.8855 | n.s. |
| CDC-Diabetes | per_record | 11 | 5.0000 | Uniform | 0.7934 | 0.7923 | 0.0010 | -0.0015 | 0.0037 | 15 | 15 | 0.5291 | 1.0000 | n.s. |
| CDC-Diabetes | per_record | 11 | 5.0000 | MIC-DP | 0.7934 | 0.7896 | 0.0038 | 0.000681 | 0.0072 | 20 | 10 | 0.0473 | 1.0000 | n.s. |
| CDC-Diabetes | fixed | 5 | 5.0000 | Uniform | 0.7042 | 0.7923 | -0.0881 | -0.1074 | -0.0700 | 0 | 30 | 1.86e-09 | 1.79e-07 | B better |
| CDC-Diabetes | fixed | 5 | 5.0000 | MIC-DP | 0.7042 | 0.7896 | -0.0853 | -0.1048 | -0.0669 | 0 | 30 | 1.86e-09 | 1.79e-07 | B better |
| CDC-Diabetes | fixed | 11 | 5.0000 | Uniform | 0.7599 | 0.7923 | -0.0324 | -0.0418 | -0.0237 | 0 | 30 | 1.86e-09 | 1.79e-07 | B better |
| CDC-Diabetes | fixed | 11 | 5.0000 | MIC-DP | 0.7599 | 0.7896 | -0.0297 | -0.0396 | -0.0199 | 6 | 24 | 2.76e-06 | 0.000215 | B better |
| CDC-Diabetes | per_record | 5 | 10.0000 | Uniform | 0.7987 | 0.7976 | 0.0012 | 0.000191 | 0.0021 | 21 | 9 | 0.0248 | 1.0000 | n.s. |
| CDC-Diabetes | per_record | 5 | 10.0000 | MIC-DP | 0.7987 | 0.7983 | 0.000442 | -0.0014 | 0.0023 | 16 | 14 | 0.6554 | 1.0000 | n.s. |
| CDC-Diabetes | per_record | 11 | 10.0000 | Uniform | 0.7979 | 0.7976 | 0.000309 | -0.000901 | 0.0014 | 18 | 12 | 0.3931 | 1.0000 | n.s. |
| CDC-Diabetes | per_record | 11 | 10.0000 | MIC-DP | 0.7979 | 0.7983 | -0.000421 | -0.0021 | 0.0013 | 15 | 15 | 0.5978 | 1.0000 | n.s. |
| CDC-Diabetes | fixed | 5 | 10.0000 | Uniform | 0.7108 | 0.7976 | -0.0867 | -0.1009 | -0.0732 | 0 | 30 | 1.86e-09 | 1.79e-07 | B better |
| CDC-Diabetes | fixed | 5 | 10.0000 | MIC-DP | 0.7108 | 0.7983 | -0.0875 | -0.1021 | -0.0737 | 0 | 30 | 1.86e-09 | 1.79e-07 | B better |
| CDC-Diabetes | fixed | 11 | 10.0000 | Uniform | 0.7656 | 0.7976 | -0.0320 | -0.0401 | -0.0246 | 0 | 30 | 1.86e-09 | 1.79e-07 | B better |
| CDC-Diabetes | fixed | 11 | 10.0000 | MIC-DP | 0.7656 | 0.7983 | -0.0327 | -0.0417 | -0.0244 | 1 | 29 | 5.59e-09 | 4.86e-07 | B better |

## Outcome counts of the new families

| family | A better | B better | n.s. |
|---|---|---|---|
| B_noise_aware | 19 | 0 | 5 |
| F7r_adult | 24 | 24 | 0 |
| F7r_cdc_diabetes | 0 | 48 | 0 |
| F9 | 0 | 20 | 76 |

## Runtime

python -m src.draft3: 58.1 min wall clock (3487 s); units run this time: 60; sum of unit times 115.8 min (2 workers); by part (sum over units, min): refits (A.1, A.2, B) 29.8, Part C 31.8, Part D 54.2; python -m src.draft3_report: 0.1 min. One smoke unit before the main run (Adult, trial 0): 5.0 min.

## Deviations

D1 (Part C, before the main run). The noise-aware prediction of the repaired fine-tuned model fixed in commit 8b2e018 estimated E[clip(F_pre(x + eta))] with a Monte-Carlo term of 256 draws per test record. On the smoke unit (Adult, trial 0, Uniform release, eps = 0.3, 15% split) the spread of the exact E[F_pre] across test records was a standard deviation of 0.012 in logit, below the Monte-Carlo error per record, and the estimate gave ROC-AUC 0.503 against 0.833 for clip(E[F_pre]) + E[F_cont]. The noise-aware prediction was changed to clip(E[F_pre(x + eta)], -3, 3) + E[F_cont(x + eta)], both exact, and recorded in draft3_notes.md and config.yaml (commit 6209f42) before the main run. The repair itself and every other setting are as fixed in 8b2e018. The smoke outputs are not part of the results.

D2 (presentation only, after the run). diagnostics/draft3_ft_clip.py was added and run on trial 0 (output diagnostic_ft_clip.csv, shown under Part C) to show the range of E[F_pre] and where the clip of the noise-aware prediction is active. statsmodels (used by src/stats.py for Holm) was missing from the restored environment and was installed (0.15.0) before the report ran. No reported number of a method changed.

Files in this folder: raw_trials_draft3.parquet (one row per refit, repaired FT model and Random-k model), stats_draft3.csv (all new tests: B_noise_aware, F7r_adult, F7r_cdc_diabetes, F9, and the A.3 intervals), config_used.yaml, tab_*.tex (rows only), auc_trials_ch4.pdf/png, sanity_checks_draft3.csv, sanity_draft3.log, runtime_draft3.txt, run.log.
