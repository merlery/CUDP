# Chapter 4 (MIC-DP): public-data baselines

Produced by `python -m src.public && python -m src.public_report`. Numbers are reported as they came out. Primary model: HistGradientBoosting (200 iterations, no early stopping, random_state = trial seed) with noise-aware prediction for every method that trains on a privatized release; secondary model: logistic regression. ROC-AUC and accuracy on the clean test set, 30 paired trials (seed 1000 + t), delta = 1e-5. Existing methods (Non-private, Uniform, MIC-DP, MIC-Public 15%) are taken from `results/ch4/raw_trials.parquet` of the main run; their trial 0 was re-run and reproduced (sanity checks).

## Pre-run commit

Pre-run commit: a11f76a11dbde46d2e403b481c7d8b8c3e43cf1f (2026-11-07 16:35:20 -0800), in the restored repository
(restore commit 52748a90ce250ef8340d5a584b9179d83a6aedb1). It contains config.yaml (public_baselines section),
public_reporting_rules.md, src/public.py, src/public_report.py and tests/test_public.py. No new method had been run before it.

## Privacy accounting

Public-Only touches no private record and has no (eps, delta). Every method that trains on a privatized release has exactly the (eps, delta) of that release: Public-Pretrain + Uniform-FT and Public + Uniform (pooled) use the Uniform release of the main run, Public-Pretrain + MIC-DP-FT uses the MIC-DP release, and MIC-Public at 5%, 1% and shifted 5% spend no budget on scores (rho = 0, scores from public data) and the whole (eps, delta) on the data release. Pretraining on public data, pooling with public rows, and noise-aware prediction are post-processing of the release and add no privacy cost.

## Reporting rules (verbatim, committed before the run)


## Section 5 of the task, verbatim

Family F6: for each dataset, public size in {15%, 5%, 1%}, and eps, compare
Public-Only with the best private method of the main comparison at that eps (the
best of Uniform, MIC-DP, MIC-Public by mean ROC-AUC, chosen before looking at the
new runs, from the existing results). 36 tests, Holm within the family.

Report, for each dataset:
1. the smallest public size at which Public-Only is significantly better than every
   private method at every eps <= 1 (or "none");
2. the largest eps at which the best private method is significantly better than
   Public-Only, per public size;
3. whether Public-Pretrain + FT or the pooled method is significantly better than
   both of its parents (Public-Only and the private method alone), per eps and size;
4. the ROC-AUC of MIC-Public at 15%, 5%, and 1% against MIC-DP, to show at which
   public size the allocation from public scores stops helping;
5. the shifted-split results, labeled exploratory.

These are reporting rules, not hypotheses; there is nothing to pass or fail. Do not
add, remove, or reword a rule after the run.

## How each rule is computed (fixed before the run; config.yaml, public_baselines)

All comparisons use the primary model (HistGradientBoosting) and ROC-AUC, 30 paired trials. A comparison
is "significant" if the Holm-adjusted p-value of the paired two-sided Wilcoxon test is below 0.05 within
its family, and its direction is the sign of the mean paired difference.

- Best private method (F6): per dataset and eps, the method with the highest mean ROC-AUC among Uniform,
  MIC-DP and MIC-Public (15% public split) in the existing main-run results (config.yaml,
  public_baselines.best_private). Because it has the highest mean of the three, "significantly better than
  every private method" in rule 1 is evaluated as "significantly better than the best private method" in
  F6. The means of all three are reported next to it.
- Rule 1: for each public size s in {1%, 5%, 15%}, the condition holds if, in F6, Public-Only at s is
  significantly better than the best private method at every eps in {0.1, 0.3, 0.5, 1}. Reported: the
  smallest s for which it holds ("none" if it holds for no size), and the list of sizes for which it holds.
- Rule 2: for each public size, the largest eps at which the best private method is significantly better
  than Public-Only in F6 ("none" if there is no such eps). The list of all such eps is reported with it.
- Rule 3 (family F7, Holm over its 144 tests): for each dataset, eps, public size in {15%, 5%}, and method
  in {Public-Pretrain + Uniform-FT, Public-Pretrain + MIC-DP-FT, Public + Uniform (pooled)}: "yes" if the
  method is significantly better than Public-Only at the same size and significantly better than its
  private parent (Uniform, MIC-DP, Uniform respectively); otherwise "no", with the direction of each test.
- Rule 4 (family F8, Holm over its 36 tests): mean ROC-AUC of MIC-Public at 15%, 5% and 1% and of MIC-DP,
  the mean paired difference MIC-Public - MIC-DP with its 95% bootstrap interval, and the F8 outcome per
  eps. Per dataset and eps, the public size at which the allocation from public scores stops helping is the
  largest size at which MIC-Public is not significantly better than MIC-DP while it is significantly better
  at every larger size ("none" if it is significantly better at all three sizes; "15%" if it is not
  significantly better even at 15%).
- Rule 5: shifted 5% split, exploratory: Public-Only (shifted) and MIC-Public (shifted) next to their
  unshifted 5% counterparts, MIC-DP and Uniform; mean differences with 95% bootstrap intervals and
  unadjusted Wilcoxon p-values; no family, no correction.


## Repository

The original Chapter 4 repository and its commit 0772eff did not survive a reset of the compute environment. It was restored from the archived snapshot of the v2 run (ch4_micdp_experiment_v2.zip: code, config, tests, results) as commit 52748a9; src/dp.py, fgmic.py, methods.py and data.py have the SHA-256 recorded at 0772eff, and the environment has the package versions of the main run (numpy 2.5.3, pandas 3.0.5, scikit-learn 1.9.1, scipy 1.18.1; Python 3.13.18 instead of 3.13.15). The re-run of trial 0 reproduces the stored results of the existing methods to 1e-9 (sanity checks).

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
[PASS] adult: p1 (488) subset of p5 (2442) subset of p15 (7327)
[PASS] adult: public sizes 7327, 2442, 488, shifted 2442 as in config.yaml
[PASS] adult: shifted split = 2442 public records with age <= 38 (pool 3868)
[PASS] adult: FG-MIC of the 15% split computed by the new code equals the scores of the existing MIC-Public
[PASS] adult: public split disjoint from private-train and test
[PASS] cdc_diabetes: X_train byte-identical to the split of the main run (SHA-256)
[PASS] cdc_diabetes: y_train byte-identical to the split of the main run (SHA-256)
[PASS] cdc_diabetes: X_test byte-identical to the split of the main run (SHA-256)
[PASS] cdc_diabetes: y_test byte-identical to the split of the main run (SHA-256)
[PASS] cdc_diabetes: idx_train byte-identical to the split of the main run (SHA-256)
[PASS] cdc_diabetes: idx_test byte-identical to the split of the main run (SHA-256)
[PASS] cdc_diabetes: X_public byte-identical to the split of the main run (SHA-256)
[PASS] cdc_diabetes: p1 (600) subset of p5 (3000) subset of p15 (9000)
[PASS] cdc_diabetes: public sizes 9000, 3000, 600, shifted 3000 as in config.yaml
[PASS] cdc_diabetes: shifted split = 3000 public records with Age <= 8 (pool 4607)
[PASS] cdc_diabetes: FG-MIC of the 15% split computed by the new code equals the scores of the existing MIC-Public
[PASS] cdc_diabetes: public split disjoint from private-train and test
== running 60 of 60 units (rest cached), n_jobs = 2
  1/60 units | adult t=1 (71 s) | elapsed 1.2 min | ETA 71.4 min
  2/60 units | adult t=0 (94 s) | elapsed 1.6 min | ETA 45.9 min
  3/60 units | adult t=2 (71 s) | elapsed 2.4 min | ETA 45.4 min
  4/60 units | adult t=3 (71 s) | elapsed 2.8 min | ETA 38.8 min
  5/60 units | adult t=4 (71 s) | elapsed 3.6 min | ETA 39.2 min
  6/60 units | adult t=5 (72 s) | elapsed 4.0 min | ETA 35.7 min
  7/60 units | adult t=6 (70 s) | elapsed 4.7 min | ETA 35.8 min
  8/60 units | adult t=7 (71 s) | elapsed 5.1 min | ETA 33.4 min
  9/60 units | adult t=8 (72 s) | elapsed 5.9 min | ETA 33.6 min
  10/60 units | adult t=9 (72 s) | elapsed 6.3 min | ETA 31.7 min
  11/60 units | adult t=10 (72 s) | elapsed 7.1 min | ETA 31.7 min
  12/60 units | adult t=11 (71 s) | elapsed 7.5 min | ETA 30.1 min
  13/60 units | adult t=12 (70 s) | elapsed 8.3 min | ETA 30.0 min
  14/60 units | adult t=13 (71 s) | elapsed 8.7 min | ETA 28.6 min
  15/60 units | adult t=14 (71 s) | elapsed 9.5 min | ETA 28.4 min
  16/60 units | adult t=15 (72 s) | elapsed 9.9 min | ETA 27.3 min
  17/60 units | adult t=16 (71 s) | elapsed 10.7 min | ETA 27.0 min
  18/60 units | adult t=17 (71 s) | elapsed 11.1 min | ETA 25.9 min
  19/60 units | adult t=18 (71 s) | elapsed 11.8 min | ETA 25.5 min
  20/60 units | adult t=19 (72 s) | elapsed 12.3 min | ETA 24.6 min
  21/60 units | adult t=20 (71 s) | elapsed 13.0 min | ETA 24.2 min
  22/60 units | adult t=21 (72 s) | elapsed 13.5 min | ETA 23.3 min
  23/60 units | adult t=22 (71 s) | elapsed 14.2 min | ETA 22.9 min
  24/60 units | adult t=23 (72 s) | elapsed 14.7 min | ETA 22.0 min
  25/60 units | adult t=24 (72 s) | elapsed 15.4 min | ETA 21.6 min
  26/60 units | adult t=25 (71 s) | elapsed 15.9 min | ETA 20.8 min
  27/60 units | adult t=26 (71 s) | elapsed 16.6 min | ETA 20.3 min
  28/60 units | adult t=27 (72 s) | elapsed 17.1 min | ETA 19.5 min
  29/60 units | adult t=28 (71 s) | elapsed 17.8 min | ETA 19.0 min
  30/60 units | adult t=29 (72 s) | elapsed 18.3 min | ETA 18.3 min
  31/60 units | cdc_diabetes t=1 (98 s) | elapsed 19.9 min | ETA 18.6 min
  32/60 units | cdc_diabetes t=0 (128 s) | elapsed 19.9 min | ETA 17.4 min
  33/60 units | cdc_diabetes t=2 (98 s) | elapsed 21.5 min | ETA 17.6 min
  34/60 units | cdc_diabetes t=3 (96 s) | elapsed 21.5 min | ETA 16.5 min
  35/60 units | cdc_diabetes t=4 (96 s) | elapsed 23.1 min | ETA 16.5 min
  36/60 units | cdc_diabetes t=5 (97 s) | elapsed 23.1 min | ETA 15.4 min
  37/60 units | cdc_diabetes t=6 (97 s) | elapsed 24.7 min | ETA 15.4 min
  38/60 units | cdc_diabetes t=7 (98 s) | elapsed 24.8 min | ETA 14.3 min
  39/60 units | cdc_diabetes t=8 (97 s) | elapsed 26.4 min | ETA 14.2 min
  40/60 units | cdc_diabetes t=9 (99 s) | elapsed 26.4 min | ETA 13.2 min
  41/60 units | cdc_diabetes t=10 (97 s) | elapsed 28.0 min | ETA 13.0 min
  42/60 units | cdc_diabetes t=11 (97 s) | elapsed 28.0 min | ETA 12.0 min
  43/60 units | cdc_diabetes t=12 (98 s) | elapsed 29.6 min | ETA 11.7 min
  44/60 units | cdc_diabetes t=13 (97 s) | elapsed 29.6 min | ETA 10.8 min
  45/60 units | cdc_diabetes t=14 (98 s) | elapsed 31.3 min | ETA 10.4 min
  46/60 units | cdc_diabetes t=15 (98 s) | elapsed 31.3 min | ETA 9.5 min
  47/60 units | cdc_diabetes t=17 (97 s) | elapsed 32.9 min | ETA 9.1 min
  48/60 units | cdc_diabetes t=16 (99 s) | elapsed 32.9 min | ETA 8.2 min
  49/60 units | cdc_diabetes t=18 (98 s) | elapsed 34.5 min | ETA 7.7 min
  50/60 units | cdc_diabetes t=19 (98 s) | elapsed 34.5 min | ETA 6.9 min
  51/60 units | cdc_diabetes t=20 (97 s) | elapsed 36.1 min | ETA 6.4 min
  52/60 units | cdc_diabetes t=21 (98 s) | elapsed 36.2 min | ETA 5.6 min
  53/60 units | cdc_diabetes t=22 (97 s) | elapsed 37.7 min | ETA 5.0 min
  54/60 units | cdc_diabetes t=23 (98 s) | elapsed 37.8 min | ETA 4.2 min
  55/60 units | cdc_diabetes t=24 (98 s) | elapsed 39.4 min | ETA 3.6 min
  56/60 units | cdc_diabetes t=25 (98 s) | elapsed 39.4 min | ETA 2.8 min
  57/60 units | cdc_diabetes t=26 (97 s) | elapsed 41.0 min | ETA 2.2 min
  58/60 units | cdc_diabetes t=27 (97 s) | elapsed 41.1 min | ETA 1.4 min
  59/60 units | cdc_diabetes t=28 (96 s) | elapsed 42.6 min | ETA 0.7 min
  60/60 units | cdc_diabetes t=29 (99 s) | elapsed 42.7 min | ETA 0.0 min
[PASS] existing_reproduced: 76 of 76 checks pass
[PASS] pretrained_reproduces_public_only: 2674 of 2674 checks pass
[PASS] public_only_reads_no_private_row: 240 of 240 checks pass
wrote raw_trials_public.parquet (6960 rows); total 42.8 min
```

## Notes on the outputs

Public-Only has no random component (HistGradientBoosting without early stopping or feature subsampling; deterministic logistic regression), so it returns the same model and the same ROC-AUC in every trial; its standard deviation over trials is 0.

Runs whose primary-model test predictions are constant (n_unique_pred = 1; ROC-AUC = 0.5), out of 30 trials (cells without constant predictions are omitted):

| dataset | method | eps | constant runs |
|---|---|---|---|
| Adult | Public-Pretrain + MIC-DP-FT (15%) | 0.1000 | 30 |
| Adult | Public-Pretrain + MIC-DP-FT (15%) | 0.3000 | 30 |
| Adult | Public-Pretrain + MIC-DP-FT (15%) | 0.5000 | 29 |
| Adult | Public-Pretrain + MIC-DP-FT (15%) | 1.0000 | 29 |
| Adult | Public-Pretrain + MIC-DP-FT (15%) | 5.0000 | 30 |
| Adult | Public-Pretrain + MIC-DP-FT (15%) | 10.0000 | 28 |
| Adult | Public-Pretrain + MIC-DP-FT (5%) | 0.1000 | 30 |
| Adult | Public-Pretrain + MIC-DP-FT (5%) | 0.3000 | 30 |
| Adult | Public-Pretrain + MIC-DP-FT (5%) | 0.5000 | 30 |
| Adult | Public-Pretrain + MIC-DP-FT (5%) | 1.0000 | 30 |
| Adult | Public-Pretrain + MIC-DP-FT (5%) | 5.0000 | 30 |
| Adult | Public-Pretrain + MIC-DP-FT (5%) | 10.0000 | 30 |
| Adult | Public-Pretrain + Uniform-FT (15%) | 0.1000 | 30 |
| Adult | Public-Pretrain + Uniform-FT (15%) | 0.3000 | 30 |
| Adult | Public-Pretrain + Uniform-FT (15%) | 0.5000 | 30 |
| Adult | Public-Pretrain + Uniform-FT (15%) | 1.0000 | 30 |
| Adult | Public-Pretrain + Uniform-FT (15%) | 5.0000 | 30 |
| Adult | Public-Pretrain + Uniform-FT (15%) | 10.0000 | 25 |
| Adult | Public-Pretrain + Uniform-FT (5%) | 0.1000 | 30 |
| Adult | Public-Pretrain + Uniform-FT (5%) | 0.3000 | 30 |
| Adult | Public-Pretrain + Uniform-FT (5%) | 0.5000 | 30 |
| Adult | Public-Pretrain + Uniform-FT (5%) | 1.0000 | 30 |
| Adult | Public-Pretrain + Uniform-FT (5%) | 5.0000 | 30 |
| Adult | Public-Pretrain + Uniform-FT (5%) | 10.0000 | 26 |
| CDC-Diabetes | Public-Pretrain + MIC-DP-FT (5%) | 5.0000 | 2 |
| CDC-Diabetes | Public-Pretrain + MIC-DP-FT (5%) | 10.0000 | 3 |

Diagnostic of the fine-tuned primary models (exploratory; `diagnostics/ft_leaf_values.py`, trial 0, 15% split, the same code path as the run; nothing was changed): mean and maximum |raw output| of the pretrained model on the privatized private-training rows, median logistic Hessian p(1 - p) at those values, largest |leaf value| among the 100 pretrained trees, median and maximum of the per-tree largest |leaf value| among the 100 continued trees, range of the noise-aware raw prediction on the test set, and the number of distinct test probabilities.

| dataset | eps | method | start_abs_raw_mean | start_abs_raw_max | start_hessian_median | max_abs_leaf_pretrained | median_max_abs_leaf_continued | max_abs_leaf_continued | test_raw_min | test_raw_max | test_unique_probabilities |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Adult | 0.1000 | Public-Pretrain + Uniform-FT (15%) | 5.0408 | 10.0740 | 0.0050 | 0.7601 | 5365.8438 | 131404.5038 | -6951.9787 | -6891.6916 | 1 |
| Adult | 0.1000 | Public-Pretrain + MIC-DP-FT (15%) | 5.0337 | 10.0740 | 0.0049 | 0.7601 | 8374.5002 | 209044.9611 | -10092.9710 | -10004.2103 | 1 |
| Adult | 0.3000 | Public-Pretrain + Uniform-FT (15%) | 5.0311 | 10.0740 | 0.0051 | 0.7601 | 5023.3067 | 246033.9169 | -14538.7704 | -14042.0488 | 1 |
| Adult | 0.3000 | Public-Pretrain + MIC-DP-FT (15%) | 5.0445 | 10.0740 | 0.0050 | 0.7601 | 3945.8421 | 127088.9114 | -4432.9544 | -4334.3206 | 1 |
| Adult | 0.5000 | Public-Pretrain + Uniform-FT (15%) | 5.0298 | 10.0740 | 0.0050 | 0.7601 | 5110.5160 | 175786.9180 | -6849.4259 | -6453.1618 | 1 |
| Adult | 0.5000 | Public-Pretrain + MIC-DP-FT (15%) | 5.0434 | 10.0740 | 0.0049 | 0.7601 | 7307.2817 | 254875.5631 | -9076.2835 | -8847.6292 | 1 |
| Adult | 1.0000 | Public-Pretrain + Uniform-FT (15%) | 5.0316 | 10.0876 | 0.0050 | 0.7601 | 4090.9136 | 287546.7957 | -50427.6781 | -45470.7309 | 1 |
| Adult | 1.0000 | Public-Pretrain + MIC-DP-FT (15%) | 5.0673 | 10.0876 | 0.0047 | 0.7601 | 6532.9042 | 294917.2426 | -53593.2292 | -50814.8404 | 1 |
| Adult | 5.0000 | Public-Pretrain + Uniform-FT (15%) | 5.0569 | 10.0874 | 0.0048 | 0.7601 | 6501.4302 | 249857.7912 | -29290.4271 | -18082.4761 | 1 |
| Adult | 5.0000 | Public-Pretrain + MIC-DP-FT (15%) | 5.7071 | 10.2907 | 0.0031 | 0.5789 | 8498.0872 | 304601.8773 | -17036.9740 | -14185.7841 | 1 |
| Adult | 10.0000 | Public-Pretrain + Uniform-FT (15%) | 5.0895 | 10.0851 | 0.0046 | 0.7601 | 5847.9772 | 123956.9837 | -18664.6626 | -9420.0252 | 1 |
| Adult | 10.0000 | Public-Pretrain + MIC-DP-FT (15%) | 5.7017 | 10.3859 | 0.0028 | 0.5789 | 11858.0223 | 322580.0213 | -8691.9352 | 6940.5042 | 456 |
| CDC-Diabetes | 0.1000 | Public-Pretrain + Uniform-FT (15%) | 3.1455 | 7.8346 | 0.0404 | 0.6159 | 0.4797 | 26.4941 | -2.0438 | -2.0418 | 11805 |
| CDC-Diabetes | 0.1000 | Public-Pretrain + MIC-DP-FT (15%) | 3.1563 | 7.7960 | 0.0401 | 0.6159 | 0.4529 | 30.1509 | -2.0177 | -2.0165 | 11805 |
| CDC-Diabetes | 0.3000 | Public-Pretrain + Uniform-FT (15%) | 3.1548 | 7.7582 | 0.0402 | 0.6159 | 0.4959 | 15.7186 | -2.0453 | -2.0383 | 11805 |
| CDC-Diabetes | 0.3000 | Public-Pretrain + MIC-DP-FT (15%) | 3.1414 | 7.9087 | 0.0404 | 0.6159 | 0.4729 | 15.3190 | -2.0401 | -2.0362 | 11805 |
| CDC-Diabetes | 0.5000 | Public-Pretrain + Uniform-FT (15%) | 3.1541 | 7.7507 | 0.0398 | 0.6159 | 0.4593 | 23.4390 | -2.0302 | -2.0195 | 11805 |
| CDC-Diabetes | 0.5000 | Public-Pretrain + MIC-DP-FT (15%) | 3.1540 | 7.8964 | 0.0401 | 0.6159 | 0.4397 | 29.4855 | -2.0236 | -2.0156 | 11805 |
| CDC-Diabetes | 1.0000 | Public-Pretrain + Uniform-FT (15%) | 3.1501 | 7.8526 | 0.0403 | 0.6159 | 0.4523 | 18.3216 | -2.0595 | -2.0403 | 11805 |
| CDC-Diabetes | 1.0000 | Public-Pretrain + MIC-DP-FT (15%) | 3.1471 | 7.5491 | 0.0407 | 0.6159 | 0.4442 | 15.7769 | -2.0214 | -2.0065 | 11805 |
| CDC-Diabetes | 5.0000 | Public-Pretrain + Uniform-FT (15%) | 3.1386 | 7.9653 | 0.0408 | 0.6159 | 0.4546 | 19.8512 | -2.0935 | -1.9385 | 11805 |
| CDC-Diabetes | 5.0000 | Public-Pretrain + MIC-DP-FT (15%) | 3.1500 | 7.9461 | 0.0399 | 0.6159 | 0.4904 | 21.7306 | -2.1084 | -1.9771 | 11805 |
| CDC-Diabetes | 10.0000 | Public-Pretrain + Uniform-FT (15%) | 3.1353 | 8.1170 | 0.0404 | 0.6159 | 0.5022 | 18.2963 | -2.2448 | -1.8123 | 11805 |
| CDC-Diabetes | 10.0000 | Public-Pretrain + MIC-DP-FT (15%) | 3.1637 | 8.2977 | 0.0430 | 0.5174 | 0.4217 | 795.3500 | -3.3709 | -1.9104 | 7553 |

## ROC-AUC, primary model (mean ± std over trials)

**Adult** (Non-private 0.928 ± 0.000):

| method | eps=0.1 | eps=0.3 | eps=0.5 | eps=1 | eps=5 | eps=10 |
|---|---|---|---|---|---|---|
| Public-Only (15%) | 0.911 ± 0.000 | 0.911 ± 0.000 | 0.911 ± 0.000 | 0.911 ± 0.000 | 0.911 ± 0.000 | 0.911 ± 0.000 |
| Public-Only (5%) | 0.897 ± 0.000 | 0.897 ± 0.000 | 0.897 ± 0.000 | 0.897 ± 0.000 | 0.897 ± 0.000 | 0.897 ± 0.000 |
| Public-Only (1%) | 0.863 ± 0.000 | 0.863 ± 0.000 | 0.863 ± 0.000 | 0.863 ± 0.000 | 0.863 ± 0.000 | 0.863 ± 0.000 |
| Uniform | 0.531 ± 0.129 | 0.606 ± 0.104 | 0.668 ± 0.094 | 0.708 ± 0.077 | 0.799 ± 0.010 | 0.801 ± 0.005 |
| MIC-DP | 0.547 ± 0.148 | 0.572 ± 0.128 | 0.612 ± 0.137 | 0.705 ± 0.097 | 0.820 ± 0.017 | 0.823 ± 0.008 |
| MIC-Public (15%) | 0.547 ± 0.176 | 0.672 ± 0.131 | 0.680 ± 0.121 | 0.780 ± 0.047 | 0.820 ± 0.008 | 0.827 ± 0.005 |
| MIC-Public (5%) | 0.492 ± 0.185 | 0.612 ± 0.145 | 0.660 ± 0.141 | 0.752 ± 0.109 | 0.824 ± 0.007 | 0.825 ± 0.006 |
| MIC-Public (1%) | 0.565 ± 0.149 | 0.674 ± 0.116 | 0.694 ± 0.150 | 0.772 ± 0.061 | 0.817 ± 0.011 | 0.819 ± 0.006 |
| Public-Pretrain + Uniform-FT (15%) | 0.500 ± 0.000 | 0.500 ± 0.000 | 0.500 ± 0.000 | 0.500 ± 0.000 | 0.500 ± 0.000 | 0.502 ± 0.009 |
| Public-Pretrain + Uniform-FT (5%) | 0.500 ± 0.000 | 0.500 ± 0.000 | 0.500 ± 0.000 | 0.500 ± 0.000 | 0.500 ± 0.000 | 0.504 ± 0.020 |
| Public-Pretrain + MIC-DP-FT (15%) | 0.500 ± 0.000 | 0.500 ± 0.000 | 0.505 ± 0.028 | 0.508 ± 0.044 | 0.500 ± 0.000 | 0.509 ± 0.046 |
| Public-Pretrain + MIC-DP-FT (5%) | 0.500 ± 0.000 | 0.500 ± 0.000 | 0.500 ± 0.000 | 0.500 ± 0.000 | 0.500 ± 0.000 | 0.500 ± 0.000 |
| Public + Uniform (pooled) (15%) | 0.537 ± 0.119 | 0.612 ± 0.095 | 0.680 ± 0.087 | 0.717 ± 0.076 | 0.801 ± 0.011 | 0.802 ± 0.004 |
| Public + Uniform (pooled) (5%) | 0.537 ± 0.119 | 0.607 ± 0.102 | 0.670 ± 0.087 | 0.709 ± 0.082 | 0.800 ± 0.009 | 0.802 ± 0.005 |

**CDC-Diabetes** (Non-private 0.825 ± 0.000):

| method | eps=0.1 | eps=0.3 | eps=0.5 | eps=1 | eps=5 | eps=10 |
|---|---|---|---|---|---|---|
| Public-Only (15%) | 0.804 ± 0.000 | 0.804 ± 0.000 | 0.804 ± 0.000 | 0.804 ± 0.000 | 0.804 ± 0.000 | 0.804 ± 0.000 |
| Public-Only (5%) | 0.788 ± 0.000 | 0.788 ± 0.000 | 0.788 ± 0.000 | 0.788 ± 0.000 | 0.788 ± 0.000 | 0.788 ± 0.000 |
| Public-Only (1%) | 0.789 ± 0.000 | 0.789 ± 0.000 | 0.789 ± 0.000 | 0.789 ± 0.000 | 0.789 ± 0.000 | 0.789 ± 0.000 |
| Uniform | 0.510 ± 0.120 | 0.583 ± 0.106 | 0.626 ± 0.107 | 0.712 ± 0.051 | 0.792 ± 0.005 | 0.798 ± 0.003 |
| MIC-DP | 0.498 ± 0.125 | 0.607 ± 0.113 | 0.583 ± 0.128 | 0.690 ± 0.056 | 0.790 ± 0.008 | 0.798 ± 0.004 |
| MIC-Public (15%) | 0.546 ± 0.181 | 0.698 ± 0.118 | 0.716 ± 0.109 | 0.792 ± 0.011 | 0.799 ± 0.002 | 0.800 ± 0.001 |
| MIC-Public (5%) | 0.589 ± 0.188 | 0.707 ± 0.087 | 0.732 ± 0.117 | 0.787 ± 0.023 | 0.798 ± 0.002 | 0.799 ± 0.001 |
| MIC-Public (1%) | 0.610 ± 0.166 | 0.665 ± 0.155 | 0.751 ± 0.091 | 0.789 ± 0.013 | 0.800 ± 0.002 | 0.800 ± 0.001 |
| Public-Pretrain + Uniform-FT (15%) | 0.713 ± 0.061 | 0.734 ± 0.055 | 0.740 ± 0.060 | 0.774 ± 0.019 | 0.795 ± 0.012 | 0.801 ± 0.002 |
| Public-Pretrain + Uniform-FT (5%) | 0.754 ± 0.032 | 0.744 ± 0.044 | 0.741 ± 0.032 | 0.749 ± 0.031 | 0.727 ± 0.063 | 0.744 ± 0.054 |
| Public-Pretrain + MIC-DP-FT (15%) | 0.725 ± 0.062 | 0.749 ± 0.029 | 0.721 ± 0.062 | 0.756 ± 0.031 | 0.794 ± 0.006 | 0.794 ± 0.017 |
| Public-Pretrain + MIC-DP-FT (5%) | 0.741 ± 0.049 | 0.743 ± 0.041 | 0.747 ± 0.040 | 0.733 ± 0.051 | 0.703 ± 0.111 | 0.715 ± 0.096 |
| Public + Uniform (pooled) (15%) | 0.529 ± 0.118 | 0.601 ± 0.105 | 0.649 ± 0.093 | 0.730 ± 0.036 | 0.794 ± 0.005 | 0.798 ± 0.002 |
| Public + Uniform (pooled) (5%) | 0.521 ± 0.117 | 0.591 ± 0.115 | 0.638 ± 0.098 | 0.714 ± 0.057 | 0.793 ± 0.005 | 0.798 ± 0.003 |

## ROC-AUC, secondary model (mean ± std over trials)

**Adult** (Non-private 0.882 ± 0.000):

| method | eps=0.1 | eps=0.3 | eps=0.5 | eps=1 | eps=5 | eps=10 |
|---|---|---|---|---|---|---|
| Public-Only (15%) | 0.877 ± 0.000 | 0.877 ± 0.000 | 0.877 ± 0.000 | 0.877 ± 0.000 | 0.877 ± 0.000 | 0.877 ± 0.000 |
| Public-Only (5%) | 0.872 ± 0.000 | 0.872 ± 0.000 | 0.872 ± 0.000 | 0.872 ± 0.000 | 0.872 ± 0.000 | 0.872 ± 0.000 |
| Public-Only (1%) | 0.857 ± 0.000 | 0.857 ± 0.000 | 0.857 ± 0.000 | 0.857 ± 0.000 | 0.857 ± 0.000 | 0.857 ± 0.000 |
| Uniform | 0.527 ± 0.132 | 0.604 ± 0.102 | 0.676 ± 0.089 | 0.707 ± 0.088 | 0.799 ± 0.009 | 0.802 ± 0.005 |
| MIC-DP | 0.548 ± 0.152 | 0.589 ± 0.123 | 0.608 ± 0.141 | 0.714 ± 0.093 | 0.819 ± 0.015 | 0.822 ± 0.007 |
| MIC-Public (15%) | 0.553 ± 0.177 | 0.673 ± 0.137 | 0.675 ± 0.126 | 0.781 ± 0.050 | 0.819 ± 0.007 | 0.825 ± 0.005 |
| MIC-Public (5%) | 0.478 ± 0.186 | 0.610 ± 0.158 | 0.671 ± 0.136 | 0.754 ± 0.111 | 0.823 ± 0.008 | 0.824 ± 0.005 |
| MIC-Public (1%) | 0.569 ± 0.142 | 0.682 ± 0.113 | 0.689 ± 0.148 | 0.778 ± 0.058 | 0.816 ± 0.011 | 0.818 ± 0.006 |
| Public-Pretrain + Uniform-FT (15%) | 0.527 ± 0.132 | 0.604 ± 0.102 | 0.676 ± 0.089 | 0.707 ± 0.088 | 0.799 ± 0.009 | 0.802 ± 0.005 |
| Public-Pretrain + Uniform-FT (5%) | 0.527 ± 0.132 | 0.604 ± 0.102 | 0.676 ± 0.089 | 0.707 ± 0.088 | 0.799 ± 0.009 | 0.802 ± 0.005 |
| Public-Pretrain + MIC-DP-FT (15%) | 0.548 ± 0.152 | 0.589 ± 0.123 | 0.608 ± 0.141 | 0.714 ± 0.093 | 0.819 ± 0.015 | 0.822 ± 0.007 |
| Public-Pretrain + MIC-DP-FT (5%) | 0.548 ± 0.152 | 0.589 ± 0.123 | 0.608 ± 0.141 | 0.714 ± 0.093 | 0.819 ± 0.015 | 0.822 ± 0.007 |
| Public + Uniform (pooled) (15%) | 0.535 ± 0.132 | 0.620 ± 0.097 | 0.695 ± 0.081 | 0.730 ± 0.068 | 0.801 ± 0.007 | 0.803 ± 0.004 |
| Public + Uniform (pooled) (5%) | 0.529 ± 0.132 | 0.609 ± 0.100 | 0.683 ± 0.086 | 0.715 ± 0.081 | 0.800 ± 0.008 | 0.803 ± 0.004 |

**CDC-Diabetes** (Non-private 0.827 ± 0.000):

| method | eps=0.1 | eps=0.3 | eps=0.5 | eps=1 | eps=5 | eps=10 |
|---|---|---|---|---|---|---|
| Public-Only (15%) | 0.825 ± 0.000 | 0.825 ± 0.000 | 0.825 ± 0.000 | 0.825 ± 0.000 | 0.825 ± 0.000 | 0.825 ± 0.000 |
| Public-Only (5%) | 0.819 ± 0.000 | 0.819 ± 0.000 | 0.819 ± 0.000 | 0.819 ± 0.000 | 0.819 ± 0.000 | 0.819 ± 0.000 |
| Public-Only (1%) | 0.802 ± 0.000 | 0.802 ± 0.000 | 0.802 ± 0.000 | 0.802 ± 0.000 | 0.802 ± 0.000 | 0.802 ± 0.000 |
| Uniform | 0.502 ± 0.120 | 0.587 ± 0.105 | 0.652 ± 0.091 | 0.727 ± 0.035 | 0.792 ± 0.005 | 0.798 ± 0.003 |
| MIC-DP | 0.510 ± 0.131 | 0.604 ± 0.100 | 0.600 ± 0.119 | 0.701 ± 0.046 | 0.790 ± 0.007 | 0.798 ± 0.004 |
| MIC-Public (15%) | 0.550 ± 0.186 | 0.711 ± 0.106 | 0.735 ± 0.092 | 0.794 ± 0.008 | 0.800 ± 0.002 | 0.801 ± 0.001 |
| MIC-Public (5%) | 0.603 ± 0.173 | 0.698 ± 0.087 | 0.731 ± 0.111 | 0.789 ± 0.013 | 0.799 ± 0.002 | 0.799 ± 0.001 |
| MIC-Public (1%) | 0.609 ± 0.161 | 0.661 ± 0.154 | 0.752 ± 0.085 | 0.787 ± 0.016 | 0.800 ± 0.001 | 0.800 ± 0.001 |
| Public-Pretrain + Uniform-FT (15%) | 0.502 ± 0.120 | 0.587 ± 0.105 | 0.652 ± 0.091 | 0.727 ± 0.035 | 0.792 ± 0.005 | 0.798 ± 0.003 |
| Public-Pretrain + Uniform-FT (5%) | 0.502 ± 0.120 | 0.587 ± 0.105 | 0.652 ± 0.091 | 0.727 ± 0.035 | 0.792 ± 0.005 | 0.798 ± 0.002 |
| Public-Pretrain + MIC-DP-FT (15%) | 0.510 ± 0.131 | 0.604 ± 0.100 | 0.600 ± 0.119 | 0.701 ± 0.046 | 0.790 ± 0.007 | 0.798 ± 0.004 |
| Public-Pretrain + MIC-DP-FT (5%) | 0.510 ± 0.131 | 0.604 ± 0.100 | 0.600 ± 0.119 | 0.701 ± 0.046 | 0.790 ± 0.007 | 0.798 ± 0.004 |
| Public + Uniform (pooled) (15%) | 0.511 ± 0.120 | 0.610 ± 0.101 | 0.676 ± 0.081 | 0.749 ± 0.024 | 0.794 ± 0.004 | 0.798 ± 0.002 |
| Public + Uniform (pooled) (5%) | 0.505 ± 0.120 | 0.595 ± 0.104 | 0.660 ± 0.088 | 0.736 ± 0.031 | 0.793 ± 0.004 | 0.798 ± 0.002 |

## Accuracy, primary model (mean over trials)

**Adult** (Non-private 0.8740):

| method | eps=0.1 | eps=0.3 | eps=0.5 | eps=1 | eps=5 | eps=10 |
|---|---|---|---|---|---|---|
| Public-Only (15%) | 0.8618 | 0.8618 | 0.8618 | 0.8618 | 0.8618 | 0.8618 |
| Public-Only (5%) | 0.8469 | 0.8469 | 0.8469 | 0.8469 | 0.8469 | 0.8469 |
| Public-Only (1%) | 0.8040 | 0.8040 | 0.8040 | 0.8040 | 0.8040 | 0.8040 |
| Uniform | 0.7607 | 0.7607 | 0.7607 | 0.7607 | 0.7607 | 0.7607 |
| MIC-DP | 0.7607 | 0.7607 | 0.7607 | 0.7607 | 0.7607 | 0.7607 |
| MIC-Public (15%) | 0.7607 | 0.7607 | 0.7607 | 0.7607 | 0.7607 | 0.7607 |
| MIC-Public (5%) | 0.7607 | 0.7607 | 0.7607 | 0.7607 | 0.7607 | 0.7607 |
| MIC-Public (1%) | 0.7607 | 0.7607 | 0.7607 | 0.7607 | 0.7607 | 0.7607 |
| Public-Pretrain + Uniform-FT (15%) | 0.7607 | 0.7607 | 0.7433 | 0.7433 | 0.7607 | 0.7272 |
| Public-Pretrain + MIC-DP-FT (15%) | 0.7607 | 0.7433 | 0.7433 | 0.7433 | 0.7433 | 0.7579 |
| Public + Uniform (pooled) (15%) | 0.7607 | 0.7607 | 0.7607 | 0.7607 | 0.7607 | 0.7607 |

**CDC-Diabetes** (Non-private 0.8659):

| method | eps=0.1 | eps=0.3 | eps=0.5 | eps=1 | eps=5 | eps=10 |
|---|---|---|---|---|---|---|
| Public-Only (15%) | 0.8597 | 0.8597 | 0.8597 | 0.8597 | 0.8597 | 0.8597 |
| Public-Only (5%) | 0.8523 | 0.8523 | 0.8523 | 0.8523 | 0.8523 | 0.8523 |
| Public-Only (1%) | 0.8493 | 0.8493 | 0.8493 | 0.8493 | 0.8493 | 0.8493 |
| Uniform | 0.8611 | 0.8611 | 0.8611 | 0.8611 | 0.8611 | 0.8611 |
| MIC-DP | 0.8611 | 0.8611 | 0.8611 | 0.8611 | 0.8611 | 0.8611 |
| MIC-Public (15%) | 0.8611 | 0.8611 | 0.8611 | 0.8611 | 0.8611 | 0.8611 |
| MIC-Public (5%) | 0.8611 | 0.8611 | 0.8611 | 0.8611 | 0.8611 | 0.8611 |
| MIC-Public (1%) | 0.8611 | 0.8611 | 0.8611 | 0.8611 | 0.8611 | 0.8611 |
| Public-Pretrain + Uniform-FT (15%) | 0.8611 | 0.8611 | 0.8611 | 0.8611 | 0.8611 | 0.8611 |
| Public-Pretrain + MIC-DP-FT (15%) | 0.8611 | 0.8611 | 0.8611 | 0.8611 | 0.8611 | 0.8611 |
| Public + Uniform (pooled) (15%) | 0.8611 | 0.8611 | 0.8611 | 0.8611 | 0.8611 | 0.8611 |

## Gap to Public-Only, G = AUC(method) - AUC(Public-Only at the same public size), primary model

Mean over trials, 95% bootstrap interval, unadjusted paired Wilcoxon p (the Holm-corrected comparisons are F6, F7 and F8 below). Methods that use no public data are compared with Public-Only at every size.

| dataset | eps | method | public_only | G | ci_lo | ci_hi | p_raw |
|---|---|---|---|---|---|---|---|
| Adult | 0.1000 | Non-private | Public-Only (15%) | 0.0170 | 0.0170 | 0.0170 | 4.32e-08 |
| Adult | 0.1000 | Non-private | Public-Only (5%) | 0.0310 | 0.0310 | 0.0310 | 4.32e-08 |
| Adult | 0.1000 | Non-private | Public-Only (1%) | 0.0646 | 0.0646 | 0.0646 | 4.32e-08 |
| Adult | 0.1000 | Uniform | Public-Only (15%) | -0.3794 | -0.4248 | -0.3355 | 1.863e-09 |
| Adult | 0.1000 | Uniform | Public-Only (5%) | -0.3655 | -0.4117 | -0.3219 | 1.863e-09 |
| Adult | 0.1000 | Uniform | Public-Only (1%) | -0.3318 | -0.3770 | -0.2877 | 1.863e-09 |
| Adult | 0.1000 | MIC-DP | Public-Only (15%) | -0.3641 | -0.4164 | -0.3125 | 1.863e-09 |
| Adult | 0.1000 | MIC-DP | Public-Only (5%) | -0.3502 | -0.4032 | -0.2981 | 1.863e-09 |
| Adult | 0.1000 | MIC-DP | Public-Only (1%) | -0.3165 | -0.3677 | -0.2644 | 1.863e-09 |
| Adult | 0.1000 | MIC-Public (15%) | Public-Only (15%) | -0.3643 | -0.4258 | -0.3020 | 1.863e-09 |
| Adult | 0.1000 | MIC-Public (5%) | Public-Only (5%) | -0.4046 | -0.4685 | -0.3409 | 1.863e-09 |
| Adult | 0.1000 | MIC-Public (1%) | Public-Only (1%) | -0.2986 | -0.3512 | -0.2459 | 1.863e-09 |
| Adult | 0.1000 | Public-Pretrain + Uniform-FT (15%) | Public-Only (15%) | -0.4109 | -0.4109 | -0.4109 | 4.32e-08 |
| Adult | 0.1000 | Public-Pretrain + Uniform-FT (5%) | Public-Only (5%) | -0.3970 | -0.3970 | -0.3970 | 4.32e-08 |
| Adult | 0.1000 | Public-Pretrain + MIC-DP-FT (15%) | Public-Only (15%) | -0.4109 | -0.4109 | -0.4109 | 4.32e-08 |
| Adult | 0.1000 | Public-Pretrain + MIC-DP-FT (5%) | Public-Only (5%) | -0.3970 | -0.3970 | -0.3970 | 4.32e-08 |
| Adult | 0.1000 | Public + Uniform (pooled) (15%) | Public-Only (15%) | -0.3737 | -0.4154 | -0.3333 | 1.863e-09 |
| Adult | 0.1000 | Public + Uniform (pooled) (5%) | Public-Only (5%) | -0.3601 | -0.4029 | -0.3186 | 1.863e-09 |
| Adult | 0.1000 | MIC-Public (shifted 5%) | Public-Only (shifted 5%) | -0.3396 | -0.3933 | -0.2890 | 1.863e-09 |
| Adult | 0.3000 | Non-private | Public-Only (15%) | 0.0170 | 0.0170 | 0.0170 | 4.32e-08 |
| Adult | 0.3000 | Non-private | Public-Only (5%) | 0.0310 | 0.0310 | 0.0310 | 4.32e-08 |
| Adult | 0.3000 | Non-private | Public-Only (1%) | 0.0646 | 0.0646 | 0.0646 | 4.32e-08 |
| Adult | 0.3000 | Uniform | Public-Only (15%) | -0.3053 | -0.3424 | -0.2681 | 1.863e-09 |
| Adult | 0.3000 | Uniform | Public-Only (5%) | -0.2913 | -0.3284 | -0.2549 | 1.863e-09 |
| Adult | 0.3000 | Uniform | Public-Only (1%) | -0.2577 | -0.2950 | -0.2221 | 1.863e-09 |
| Adult | 0.3000 | MIC-DP | Public-Only (15%) | -0.3387 | -0.3835 | -0.2928 | 1.863e-09 |
| Adult | 0.3000 | MIC-DP | Public-Only (5%) | -0.3247 | -0.3702 | -0.2807 | 1.863e-09 |
| Adult | 0.3000 | MIC-DP | Public-Only (1%) | -0.2911 | -0.3371 | -0.2464 | 1.863e-09 |
| Adult | 0.3000 | MIC-Public (15%) | Public-Only (15%) | -0.2390 | -0.2866 | -0.1948 | 1.863e-09 |
| Adult | 0.3000 | MIC-Public (5%) | Public-Only (5%) | -0.2846 | -0.3381 | -0.2363 | 1.863e-09 |
| Adult | 0.3000 | MIC-Public (1%) | Public-Only (1%) | -0.1892 | -0.2309 | -0.1501 | 1.863e-09 |
| Adult | 0.3000 | Public-Pretrain + Uniform-FT (15%) | Public-Only (15%) | -0.4109 | -0.4109 | -0.4109 | 4.32e-08 |
| Adult | 0.3000 | Public-Pretrain + Uniform-FT (5%) | Public-Only (5%) | -0.3970 | -0.3970 | -0.3970 | 4.32e-08 |
| Adult | 0.3000 | Public-Pretrain + MIC-DP-FT (15%) | Public-Only (15%) | -0.4109 | -0.4109 | -0.4109 | 4.32e-08 |
| Adult | 0.3000 | Public-Pretrain + MIC-DP-FT (5%) | Public-Only (5%) | -0.3970 | -0.3970 | -0.3970 | 4.32e-08 |
| Adult | 0.3000 | Public + Uniform (pooled) (15%) | Public-Only (15%) | -0.2988 | -0.3321 | -0.2647 | 1.863e-09 |
| Adult | 0.3000 | Public + Uniform (pooled) (5%) | Public-Only (5%) | -0.2901 | -0.3273 | -0.2543 | 1.863e-09 |
| Adult | 0.3000 | MIC-Public (shifted 5%) | Public-Only (shifted 5%) | -0.2742 | -0.3293 | -0.2204 | 1.863e-09 |
| Adult | 0.5000 | Non-private | Public-Only (15%) | 0.0170 | 0.0170 | 0.0170 | 4.32e-08 |
| Adult | 0.5000 | Non-private | Public-Only (5%) | 0.0310 | 0.0310 | 0.0310 | 4.32e-08 |
| Adult | 0.5000 | Non-private | Public-Only (1%) | 0.0646 | 0.0646 | 0.0646 | 4.32e-08 |
| Adult | 0.5000 | Uniform | Public-Only (15%) | -0.2430 | -0.2774 | -0.2106 | 1.863e-09 |
| Adult | 0.5000 | Uniform | Public-Only (5%) | -0.2291 | -0.2644 | -0.1980 | 1.863e-09 |
| Adult | 0.5000 | Uniform | Public-Only (1%) | -0.1954 | -0.2300 | -0.1638 | 1.863e-09 |
| Adult | 0.5000 | MIC-DP | Public-Only (15%) | -0.2992 | -0.3502 | -0.2542 | 1.863e-09 |
| Adult | 0.5000 | MIC-DP | Public-Only (5%) | -0.2852 | -0.3361 | -0.2395 | 1.863e-09 |
| Adult | 0.5000 | MIC-DP | Public-Only (1%) | -0.2516 | -0.3021 | -0.2065 | 1.863e-09 |
| Adult | 0.5000 | MIC-Public (15%) | Public-Only (15%) | -0.2311 | -0.2751 | -0.1914 | 1.863e-09 |
| Adult | 0.5000 | MIC-Public (5%) | Public-Only (5%) | -0.2368 | -0.2894 | -0.1891 | 1.863e-09 |
| Adult | 0.5000 | MIC-Public (1%) | Public-Only (1%) | -0.1696 | -0.2271 | -0.1219 | 1.863e-09 |
| Adult | 0.5000 | Public-Pretrain + Uniform-FT (15%) | Public-Only (15%) | -0.4109 | -0.4109 | -0.4109 | 4.32e-08 |
| Adult | 0.5000 | Public-Pretrain + Uniform-FT (5%) | Public-Only (5%) | -0.3970 | -0.3970 | -0.3970 | 4.32e-08 |
| Adult | 0.5000 | Public-Pretrain + MIC-DP-FT (15%) | Public-Only (15%) | -0.4057 | -0.4109 | -0.3954 | 6.799e-08 |
| Adult | 0.5000 | Public-Pretrain + MIC-DP-FT (5%) | Public-Only (5%) | -0.3970 | -0.3970 | -0.3970 | 4.32e-08 |
| Adult | 0.5000 | Public + Uniform (pooled) (15%) | Public-Only (15%) | -0.2314 | -0.2632 | -0.2015 | 1.863e-09 |
| Adult | 0.5000 | Public + Uniform (pooled) (5%) | Public-Only (5%) | -0.2268 | -0.2589 | -0.1971 | 1.863e-09 |
| Adult | 0.5000 | MIC-Public (shifted 5%) | Public-Only (shifted 5%) | -0.1988 | -0.2457 | -0.1554 | 1.863e-09 |
| Adult | 1.0000 | Non-private | Public-Only (15%) | 0.0170 | 0.0170 | 0.0170 | 4.32e-08 |
| Adult | 1.0000 | Non-private | Public-Only (5%) | 0.0310 | 0.0310 | 0.0310 | 4.32e-08 |
| Adult | 1.0000 | Non-private | Public-Only (1%) | 0.0646 | 0.0646 | 0.0646 | 4.32e-08 |
| Adult | 1.0000 | Uniform | Public-Only (15%) | -0.2026 | -0.2308 | -0.1768 | 1.863e-09 |
| Adult | 1.0000 | Uniform | Public-Only (5%) | -0.1887 | -0.2177 | -0.1629 | 1.863e-09 |
| Adult | 1.0000 | Uniform | Public-Only (1%) | -0.1550 | -0.1846 | -0.1294 | 1.863e-09 |
| Adult | 1.0000 | MIC-DP | Public-Only (15%) | -0.2061 | -0.2428 | -0.1740 | 1.863e-09 |
| Adult | 1.0000 | MIC-DP | Public-Only (5%) | -0.1921 | -0.2289 | -0.1600 | 1.863e-09 |
| Adult | 1.0000 | MIC-DP | Public-Only (1%) | -0.1585 | -0.1947 | -0.1259 | 1.863e-09 |
| Adult | 1.0000 | MIC-Public (15%) | Public-Only (15%) | -0.1313 | -0.1483 | -0.1160 | 1.863e-09 |
| Adult | 1.0000 | MIC-Public (5%) | Public-Only (5%) | -0.1451 | -0.1877 | -0.1128 | 1.863e-09 |
| Adult | 1.0000 | MIC-Public (1%) | Public-Only (1%) | -0.0912 | -0.1147 | -0.0713 | 1.863e-09 |
| Adult | 1.0000 | Public-Pretrain + Uniform-FT (15%) | Public-Only (15%) | -0.4109 | -0.4109 | -0.4109 | 4.32e-08 |
| Adult | 1.0000 | Public-Pretrain + Uniform-FT (5%) | Public-Only (5%) | -0.3970 | -0.3970 | -0.3970 | 4.32e-08 |
| Adult | 1.0000 | Public-Pretrain + MIC-DP-FT (15%) | Public-Only (15%) | -0.4029 | -0.4109 | -0.3868 | 6.799e-08 |
| Adult | 1.0000 | Public-Pretrain + MIC-DP-FT (5%) | Public-Only (5%) | -0.3970 | -0.3970 | -0.3970 | 4.32e-08 |
| Adult | 1.0000 | Public + Uniform (pooled) (15%) | Public-Only (15%) | -0.1936 | -0.2219 | -0.1686 | 1.863e-09 |
| Adult | 1.0000 | Public + Uniform (pooled) (5%) | Public-Only (5%) | -0.1878 | -0.2183 | -0.1618 | 1.863e-09 |
| Adult | 1.0000 | MIC-Public (shifted 5%) | Public-Only (shifted 5%) | -0.1493 | -0.1799 | -0.1215 | 1.863e-09 |
| Adult | 5.0000 | Non-private | Public-Only (15%) | 0.0170 | 0.0170 | 0.0170 | 4.32e-08 |
| Adult | 5.0000 | Non-private | Public-Only (5%) | 0.0310 | 0.0310 | 0.0310 | 4.32e-08 |
| Adult | 5.0000 | Non-private | Public-Only (1%) | 0.0646 | 0.0646 | 0.0646 | 4.32e-08 |
| Adult | 5.0000 | Uniform | Public-Only (15%) | -0.1124 | -0.1158 | -0.1088 | 1.863e-09 |
| Adult | 5.0000 | Uniform | Public-Only (5%) | -0.0984 | -0.1019 | -0.0948 | 1.863e-09 |
| Adult | 5.0000 | Uniform | Public-Only (1%) | -0.0648 | -0.0681 | -0.0612 | 1.863e-09 |
| Adult | 5.0000 | MIC-DP | Public-Only (15%) | -0.0914 | -0.0972 | -0.0856 | 1.863e-09 |
| Adult | 5.0000 | MIC-DP | Public-Only (5%) | -0.0774 | -0.0832 | -0.0716 | 1.863e-09 |
| Adult | 5.0000 | MIC-DP | Public-Only (1%) | -0.0437 | -0.0496 | -0.0378 | 1.863e-09 |
| Adult | 5.0000 | MIC-Public (15%) | Public-Only (15%) | -0.0907 | -0.0936 | -0.0879 | 1.863e-09 |
| Adult | 5.0000 | MIC-Public (5%) | Public-Only (5%) | -0.0727 | -0.0753 | -0.0702 | 1.863e-09 |
| Adult | 5.0000 | MIC-Public (1%) | Public-Only (1%) | -0.0466 | -0.0505 | -0.0427 | 1.863e-09 |
| Adult | 5.0000 | Public-Pretrain + Uniform-FT (15%) | Public-Only (15%) | -0.4109 | -0.4109 | -0.4109 | 4.32e-08 |
| Adult | 5.0000 | Public-Pretrain + Uniform-FT (5%) | Public-Only (5%) | -0.3970 | -0.3970 | -0.3970 | 4.32e-08 |
| Adult | 5.0000 | Public-Pretrain + MIC-DP-FT (15%) | Public-Only (15%) | -0.4109 | -0.4109 | -0.4109 | 4.32e-08 |
| Adult | 5.0000 | Public-Pretrain + MIC-DP-FT (5%) | Public-Only (5%) | -0.3970 | -0.3970 | -0.3970 | 4.32e-08 |
| Adult | 5.0000 | Public + Uniform (pooled) (15%) | Public-Only (15%) | -0.1099 | -0.1135 | -0.1063 | 1.863e-09 |
| Adult | 5.0000 | Public + Uniform (pooled) (5%) | Public-Only (5%) | -0.0971 | -0.1003 | -0.0938 | 1.863e-09 |
| Adult | 5.0000 | MIC-Public (shifted 5%) | Public-Only (shifted 5%) | -0.0837 | -0.0863 | -0.0811 | 1.863e-09 |
| Adult | 10.0000 | Non-private | Public-Only (15%) | 0.0170 | 0.0170 | 0.0170 | 4.32e-08 |
| Adult | 10.0000 | Non-private | Public-Only (5%) | 0.0310 | 0.0310 | 0.0310 | 4.32e-08 |
| Adult | 10.0000 | Non-private | Public-Only (1%) | 0.0646 | 0.0646 | 0.0646 | 4.32e-08 |
| Adult | 10.0000 | Uniform | Public-Only (15%) | -0.1099 | -0.1115 | -0.1083 | 1.863e-09 |
| Adult | 10.0000 | Uniform | Public-Only (5%) | -0.0960 | -0.0976 | -0.0943 | 1.863e-09 |
| Adult | 10.0000 | Uniform | Public-Only (1%) | -0.0623 | -0.0639 | -0.0607 | 1.863e-09 |
| Adult | 10.0000 | MIC-DP | Public-Only (15%) | -0.0882 | -0.0910 | -0.0855 | 1.863e-09 |
| Adult | 10.0000 | MIC-DP | Public-Only (5%) | -0.0743 | -0.0769 | -0.0715 | 1.863e-09 |
| Adult | 10.0000 | MIC-DP | Public-Only (1%) | -0.0406 | -0.0434 | -0.0379 | 1.863e-09 |
| Adult | 10.0000 | MIC-Public (15%) | Public-Only (15%) | -0.0834 | -0.0852 | -0.0817 | 1.863e-09 |
| Adult | 10.0000 | MIC-Public (5%) | Public-Only (5%) | -0.0717 | -0.0738 | -0.0695 | 1.863e-09 |
| Adult | 10.0000 | MIC-Public (1%) | Public-Only (1%) | -0.0438 | -0.0461 | -0.0418 | 1.863e-09 |
| Adult | 10.0000 | Public-Pretrain + Uniform-FT (15%) | Public-Only (15%) | -0.4086 | -0.4109 | -0.4048 | 2.616e-07 |
| Adult | 10.0000 | Public-Pretrain + Uniform-FT (5%) | Public-Only (5%) | -0.3926 | -0.3971 | -0.3844 | 1.979e-07 |
| Adult | 10.0000 | Public-Pretrain + MIC-DP-FT (15%) | Public-Only (15%) | -0.4023 | -0.4109 | -0.3853 | 1.014e-07 |
| Adult | 10.0000 | Public-Pretrain + MIC-DP-FT (5%) | Public-Only (5%) | -0.3970 | -0.3970 | -0.3970 | 4.32e-08 |
| Adult | 10.0000 | Public + Uniform (pooled) (15%) | Public-Only (15%) | -0.1089 | -0.1104 | -0.1073 | 1.863e-09 |
| Adult | 10.0000 | Public + Uniform (pooled) (5%) | Public-Only (5%) | -0.0954 | -0.0971 | -0.0938 | 1.863e-09 |
| Adult | 10.0000 | MIC-Public (shifted 5%) | Public-Only (shifted 5%) | -0.0788 | -0.0807 | -0.0771 | 1.863e-09 |
| CDC-Diabetes | 0.1000 | Non-private | Public-Only (15%) | 0.0210 | 0.0210 | 0.0210 | 4.32e-08 |
| CDC-Diabetes | 0.1000 | Non-private | Public-Only (5%) | 0.0366 | 0.0366 | 0.0366 | 4.32e-08 |
| CDC-Diabetes | 0.1000 | Non-private | Public-Only (1%) | 0.0355 | 0.0355 | 0.0355 | 4.32e-08 |
| CDC-Diabetes | 0.1000 | Uniform | Public-Only (15%) | -0.2933 | -0.3375 | -0.2523 | 1.863e-09 |
| CDC-Diabetes | 0.1000 | Uniform | Public-Only (5%) | -0.2777 | -0.3215 | -0.2365 | 1.863e-09 |
| CDC-Diabetes | 0.1000 | Uniform | Public-Only (1%) | -0.2788 | -0.3228 | -0.2383 | 1.863e-09 |
| CDC-Diabetes | 0.1000 | MIC-DP | Public-Only (15%) | -0.3060 | -0.3492 | -0.2611 | 1.863e-09 |
| CDC-Diabetes | 0.1000 | MIC-DP | Public-Only (5%) | -0.2904 | -0.3340 | -0.2451 | 1.863e-09 |
| CDC-Diabetes | 0.1000 | MIC-DP | Public-Only (1%) | -0.2915 | -0.3354 | -0.2457 | 1.863e-09 |
| CDC-Diabetes | 0.1000 | MIC-Public (15%) | Public-Only (15%) | -0.2577 | -0.3231 | -0.1960 | 1.863e-09 |
| CDC-Diabetes | 0.1000 | MIC-Public (5%) | Public-Only (5%) | -0.1989 | -0.2680 | -0.1332 | 3.725e-09 |
| CDC-Diabetes | 0.1000 | MIC-Public (1%) | Public-Only (1%) | -0.1793 | -0.2392 | -0.1234 | 9.313e-09 |
| CDC-Diabetes | 0.1000 | Public-Pretrain + Uniform-FT (15%) | Public-Only (15%) | -0.0911 | -0.1142 | -0.0710 | 1.863e-09 |
| CDC-Diabetes | 0.1000 | Public-Pretrain + Uniform-FT (5%) | Public-Only (5%) | -0.0338 | -0.0455 | -0.0231 | 1.863e-09 |
| CDC-Diabetes | 0.1000 | Public-Pretrain + MIC-DP-FT (15%) | Public-Only (15%) | -0.0791 | -0.1026 | -0.0592 | 1.863e-09 |
| CDC-Diabetes | 0.1000 | Public-Pretrain + MIC-DP-FT (5%) | Public-Only (5%) | -0.0473 | -0.0660 | -0.0312 | 3.725e-09 |
| CDC-Diabetes | 0.1000 | Public + Uniform (pooled) (15%) | Public-Only (15%) | -0.2747 | -0.3164 | -0.2352 | 1.863e-09 |
| CDC-Diabetes | 0.1000 | Public + Uniform (pooled) (5%) | Public-Only (5%) | -0.2674 | -0.3096 | -0.2276 | 1.863e-09 |
| CDC-Diabetes | 0.1000 | MIC-Public (shifted 5%) | Public-Only (shifted 5%) | -0.2202 | -0.2807 | -0.1616 | 1.863e-08 |
| CDC-Diabetes | 0.3000 | Non-private | Public-Only (15%) | 0.0210 | 0.0210 | 0.0210 | 4.32e-08 |
| CDC-Diabetes | 0.3000 | Non-private | Public-Only (5%) | 0.0366 | 0.0366 | 0.0366 | 4.32e-08 |
| CDC-Diabetes | 0.3000 | Non-private | Public-Only (1%) | 0.0355 | 0.0355 | 0.0355 | 4.32e-08 |
| CDC-Diabetes | 0.3000 | Uniform | Public-Only (15%) | -0.2210 | -0.2585 | -0.1844 | 1.863e-09 |
| CDC-Diabetes | 0.3000 | Uniform | Public-Only (5%) | -0.2054 | -0.2427 | -0.1687 | 1.863e-09 |
| CDC-Diabetes | 0.3000 | Uniform | Public-Only (1%) | -0.2065 | -0.2450 | -0.1698 | 1.863e-09 |
| CDC-Diabetes | 0.3000 | MIC-DP | Public-Only (15%) | -0.1968 | -0.2387 | -0.1579 | 1.863e-09 |
| CDC-Diabetes | 0.3000 | MIC-DP | Public-Only (5%) | -0.1812 | -0.2226 | -0.1428 | 1.863e-09 |
| CDC-Diabetes | 0.3000 | MIC-DP | Public-Only (1%) | -0.1823 | -0.2221 | -0.1447 | 1.863e-09 |
| CDC-Diabetes | 0.3000 | MIC-Public (15%) | Public-Only (15%) | -0.1057 | -0.1495 | -0.0679 | 1.863e-09 |
| CDC-Diabetes | 0.3000 | MIC-Public (5%) | Public-Only (5%) | -0.0814 | -0.1149 | -0.0539 | 1.863e-09 |
| CDC-Diabetes | 0.3000 | MIC-Public (1%) | Public-Only (1%) | -0.1246 | -0.1828 | -0.0748 | 2.049e-07 |
| CDC-Diabetes | 0.3000 | Public-Pretrain + Uniform-FT (15%) | Public-Only (15%) | -0.0692 | -0.0898 | -0.0512 | 1.863e-09 |
| CDC-Diabetes | 0.3000 | Public-Pretrain + Uniform-FT (5%) | Public-Only (5%) | -0.0437 | -0.0593 | -0.0287 | 4.712e-07 |
| CDC-Diabetes | 0.3000 | Public-Pretrain + MIC-DP-FT (15%) | Public-Only (15%) | -0.0545 | -0.0647 | -0.0441 | 1.863e-09 |
| CDC-Diabetes | 0.3000 | Public-Pretrain + MIC-DP-FT (5%) | Public-Only (5%) | -0.0448 | -0.0593 | -0.0308 | 2.049e-07 |
| CDC-Diabetes | 0.3000 | Public + Uniform (pooled) (15%) | Public-Only (15%) | -0.2030 | -0.2408 | -0.1672 | 1.863e-09 |
| CDC-Diabetes | 0.3000 | Public + Uniform (pooled) (5%) | Public-Only (5%) | -0.1971 | -0.2381 | -0.1586 | 1.863e-09 |
| CDC-Diabetes | 0.3000 | MIC-Public (shifted 5%) | Public-Only (shifted 5%) | -0.1087 | -0.1571 | -0.0635 | 2.689e-05 |
| CDC-Diabetes | 0.5000 | Non-private | Public-Only (15%) | 0.0210 | 0.0210 | 0.0210 | 4.32e-08 |
| CDC-Diabetes | 0.5000 | Non-private | Public-Only (5%) | 0.0366 | 0.0366 | 0.0366 | 4.32e-08 |
| CDC-Diabetes | 0.5000 | Non-private | Public-Only (1%) | 0.0355 | 0.0355 | 0.0355 | 4.32e-08 |
| CDC-Diabetes | 0.5000 | Uniform | Public-Only (15%) | -0.1777 | -0.2163 | -0.1413 | 1.863e-09 |
| CDC-Diabetes | 0.5000 | Uniform | Public-Only (5%) | -0.1621 | -0.2015 | -0.1264 | 1.863e-09 |
| CDC-Diabetes | 0.5000 | Uniform | Public-Only (1%) | -0.1632 | -0.2021 | -0.1270 | 1.863e-09 |
| CDC-Diabetes | 0.5000 | MIC-DP | Public-Only (15%) | -0.2207 | -0.2658 | -0.1770 | 1.863e-09 |
| CDC-Diabetes | 0.5000 | MIC-DP | Public-Only (5%) | -0.2050 | -0.2513 | -0.1618 | 1.863e-09 |
| CDC-Diabetes | 0.5000 | MIC-DP | Public-Only (1%) | -0.2062 | -0.2524 | -0.1623 | 1.863e-09 |
| CDC-Diabetes | 0.5000 | MIC-Public (15%) | Public-Only (15%) | -0.0879 | -0.1287 | -0.0530 | 1.863e-09 |
| CDC-Diabetes | 0.5000 | MIC-Public (5%) | Public-Only (5%) | -0.0561 | -0.1015 | -0.0201 | 0.0054 |
| CDC-Diabetes | 0.5000 | MIC-Public (1%) | Public-Only (1%) | -0.0382 | -0.0747 | -0.0125 | 0.0011 |
| CDC-Diabetes | 0.5000 | Public-Pretrain + Uniform-FT (15%) | Public-Only (15%) | -0.0637 | -0.0865 | -0.0441 | 1.863e-09 |
| CDC-Diabetes | 0.5000 | Public-Pretrain + Uniform-FT (5%) | Public-Only (5%) | -0.0468 | -0.0582 | -0.0355 | 1.304e-08 |
| CDC-Diabetes | 0.5000 | Public-Pretrain + MIC-DP-FT (15%) | Public-Only (15%) | -0.0823 | -0.1052 | -0.0617 | 1.863e-09 |
| CDC-Diabetes | 0.5000 | Public-Pretrain + MIC-DP-FT (5%) | Public-Only (5%) | -0.0406 | -0.0559 | -0.0276 | 2.608e-08 |
| CDC-Diabetes | 0.5000 | Public + Uniform (pooled) (15%) | Public-Only (15%) | -0.1545 | -0.1898 | -0.1232 | 1.863e-09 |
| CDC-Diabetes | 0.5000 | Public + Uniform (pooled) (5%) | Public-Only (5%) | -0.1501 | -0.1857 | -0.1167 | 1.863e-09 |
| CDC-Diabetes | 0.5000 | MIC-Public (shifted 5%) | Public-Only (shifted 5%) | -0.0189 | -0.0364 | -0.0039 | 0.1004 |
| CDC-Diabetes | 1.0000 | Non-private | Public-Only (15%) | 0.0210 | 0.0210 | 0.0210 | 4.32e-08 |
| CDC-Diabetes | 1.0000 | Non-private | Public-Only (5%) | 0.0366 | 0.0366 | 0.0366 | 4.32e-08 |
| CDC-Diabetes | 1.0000 | Non-private | Public-Only (1%) | 0.0355 | 0.0355 | 0.0355 | 4.32e-08 |
| CDC-Diabetes | 1.0000 | Uniform | Public-Only (15%) | -0.0919 | -0.1097 | -0.0750 | 1.863e-09 |
| CDC-Diabetes | 1.0000 | Uniform | Public-Only (5%) | -0.0763 | -0.0950 | -0.0592 | 1.863e-09 |
| CDC-Diabetes | 1.0000 | Uniform | Public-Only (1%) | -0.0774 | -0.0956 | -0.0605 | 1.863e-09 |
| CDC-Diabetes | 1.0000 | MIC-DP | Public-Only (15%) | -0.1133 | -0.1334 | -0.0945 | 1.863e-09 |
| CDC-Diabetes | 1.0000 | MIC-DP | Public-Only (5%) | -0.0976 | -0.1173 | -0.0784 | 1.863e-09 |
| CDC-Diabetes | 1.0000 | MIC-DP | Public-Only (1%) | -0.0987 | -0.1182 | -0.0796 | 1.863e-09 |
| CDC-Diabetes | 1.0000 | MIC-Public (15%) | Public-Only (15%) | -0.0112 | -0.0155 | -0.0076 | 2.552e-07 |
| CDC-Diabetes | 1.0000 | MIC-Public (5%) | Public-Only (5%) | -0.0015 | -0.0107 | 0.0051 | 0.1191 |
| CDC-Diabetes | 1.0000 | MIC-Public (1%) | Public-Only (1%) | -5.536e-05 | -0.0047 | 0.0040 | 0.3387 |
| CDC-Diabetes | 1.0000 | Public-Pretrain + Uniform-FT (15%) | Public-Only (15%) | -0.0295 | -0.0365 | -0.0232 | 1.863e-09 |
| CDC-Diabetes | 1.0000 | Public-Pretrain + Uniform-FT (5%) | Public-Only (5%) | -0.0388 | -0.0505 | -0.0284 | 1.863e-08 |
| CDC-Diabetes | 1.0000 | Public-Pretrain + MIC-DP-FT (15%) | Public-Only (15%) | -0.0478 | -0.0593 | -0.0372 | 3.725e-09 |
| CDC-Diabetes | 1.0000 | Public-Pretrain + MIC-DP-FT (5%) | Public-Only (5%) | -0.0548 | -0.0740 | -0.0383 | 1.863e-09 |
| CDC-Diabetes | 1.0000 | Public + Uniform (pooled) (15%) | Public-Only (15%) | -0.0742 | -0.0875 | -0.0625 | 1.863e-09 |
| CDC-Diabetes | 1.0000 | Public + Uniform (pooled) (5%) | Public-Only (5%) | -0.0737 | -0.0949 | -0.0551 | 1.863e-09 |
| CDC-Diabetes | 1.0000 | MIC-Public (shifted 5%) | Public-Only (shifted 5%) | 0.0180 | 0.0131 | 0.0218 | 4.422e-06 |
| CDC-Diabetes | 5.0000 | Non-private | Public-Only (15%) | 0.0210 | 0.0210 | 0.0210 | 4.32e-08 |
| CDC-Diabetes | 5.0000 | Non-private | Public-Only (5%) | 0.0366 | 0.0366 | 0.0366 | 4.32e-08 |
| CDC-Diabetes | 5.0000 | Non-private | Public-Only (1%) | 0.0355 | 0.0355 | 0.0355 | 4.32e-08 |
| CDC-Diabetes | 5.0000 | Uniform | Public-Only (15%) | -0.0114 | -0.0130 | -0.0097 | 1.863e-09 |
| CDC-Diabetes | 5.0000 | Uniform | Public-Only (5%) | 0.0043 | 0.0026 | 0.0059 | 5.593e-05 |
| CDC-Diabetes | 5.0000 | Uniform | Public-Only (1%) | 0.0032 | 0.0014 | 0.0048 | 0.0015 |
| CDC-Diabetes | 5.0000 | MIC-DP | Public-Only (15%) | -0.0141 | -0.0168 | -0.0115 | 1.863e-09 |
| CDC-Diabetes | 5.0000 | MIC-DP | Public-Only (5%) | 0.0015 | -0.0011 | 0.0041 | 0.1909 |
| CDC-Diabetes | 5.0000 | MIC-DP | Public-Only (1%) | 0.0003909 | -0.0023 | 0.0029 | 0.5978 |
| CDC-Diabetes | 5.0000 | MIC-Public (15%) | Public-Only (15%) | -0.0044 | -0.0051 | -0.0037 | 3.725e-09 |
| CDC-Diabetes | 5.0000 | MIC-Public (5%) | Public-Only (5%) | 0.0102 | 0.0096 | 0.0109 | 1.863e-09 |
| CDC-Diabetes | 5.0000 | MIC-Public (1%) | Public-Only (1%) | 0.0105 | 0.0099 | 0.0110 | 1.863e-09 |
| CDC-Diabetes | 5.0000 | Public-Pretrain + Uniform-FT (15%) | Public-Only (15%) | -0.0088 | -0.0137 | -0.0054 | 5.588e-09 |
| CDC-Diabetes | 5.0000 | Public-Pretrain + Uniform-FT (5%) | Public-Only (5%) | -0.0614 | -0.0845 | -0.0397 | 5.718e-07 |
| CDC-Diabetes | 5.0000 | Public-Pretrain + MIC-DP-FT (15%) | Public-Only (15%) | -0.0092 | -0.0114 | -0.0071 | 9.313e-09 |
| CDC-Diabetes | 5.0000 | Public-Pretrain + MIC-DP-FT (5%) | Public-Only (5%) | -0.0849 | -0.1268 | -0.0476 | 7.687e-06 |
| CDC-Diabetes | 5.0000 | Public + Uniform (pooled) (15%) | Public-Only (15%) | -0.0097 | -0.0113 | -0.0081 | 1.863e-09 |
| CDC-Diabetes | 5.0000 | Public + Uniform (pooled) (5%) | Public-Only (5%) | 0.0049 | 0.0031 | 0.0066 | 2.689e-05 |
| CDC-Diabetes | 5.0000 | MIC-Public (shifted 5%) | Public-Only (shifted 5%) | 0.0259 | 0.0251 | 0.0266 | 1.863e-09 |
| CDC-Diabetes | 10.0000 | Non-private | Public-Only (15%) | 0.0210 | 0.0210 | 0.0210 | 4.32e-08 |
| CDC-Diabetes | 10.0000 | Non-private | Public-Only (5%) | 0.0366 | 0.0366 | 0.0366 | 4.32e-08 |
| CDC-Diabetes | 10.0000 | Non-private | Public-Only (1%) | 0.0355 | 0.0355 | 0.0355 | 4.32e-08 |
| CDC-Diabetes | 10.0000 | Uniform | Public-Only (15%) | -0.0061 | -0.0070 | -0.0052 | 3.725e-09 |
| CDC-Diabetes | 10.0000 | Uniform | Public-Only (5%) | 0.0095 | 0.0086 | 0.0105 | 1.863e-09 |
| CDC-Diabetes | 10.0000 | Uniform | Public-Only (1%) | 0.0084 | 0.0075 | 0.0093 | 1.863e-09 |
| CDC-Diabetes | 10.0000 | MIC-DP | Public-Only (15%) | -0.0054 | -0.0070 | -0.0039 | 5.718e-07 |
| CDC-Diabetes | 10.0000 | MIC-DP | Public-Only (5%) | 0.0102 | 0.0087 | 0.0117 | 3.725e-09 |
| CDC-Diabetes | 10.0000 | MIC-DP | Public-Only (1%) | 0.0091 | 0.0075 | 0.0107 | 5.588e-09 |
| CDC-Diabetes | 10.0000 | MIC-Public (15%) | Public-Only (15%) | -0.0034 | -0.0038 | -0.0031 | 1.863e-09 |
| CDC-Diabetes | 10.0000 | MIC-Public (5%) | Public-Only (5%) | 0.0108 | 0.0104 | 0.0111 | 1.863e-09 |
| CDC-Diabetes | 10.0000 | MIC-Public (1%) | Public-Only (1%) | 0.0111 | 0.0108 | 0.0114 | 1.863e-09 |
| CDC-Diabetes | 10.0000 | Public-Pretrain + Uniform-FT (15%) | Public-Only (15%) | -0.0027 | -0.0035 | -0.0020 | 1.024e-07 |
| CDC-Diabetes | 10.0000 | Public-Pretrain + Uniform-FT (5%) | Public-Only (5%) | -0.0437 | -0.0635 | -0.0259 | 2.367e-05 |
| CDC-Diabetes | 10.0000 | Public-Pretrain + MIC-DP-FT (15%) | Public-Only (15%) | -0.0099 | -0.0167 | -0.0046 | 0.0004184 |
| CDC-Diabetes | 10.0000 | Public-Pretrain + MIC-DP-FT (5%) | Public-Only (5%) | -0.0735 | -0.1090 | -0.0419 | 8.174e-05 |
| CDC-Diabetes | 10.0000 | Public + Uniform (pooled) (15%) | Public-Only (15%) | -0.0053 | -0.0061 | -0.0044 | 1.863e-09 |
| CDC-Diabetes | 10.0000 | Public + Uniform (pooled) (5%) | Public-Only (5%) | 0.0102 | 0.0093 | 0.0111 | 1.863e-09 |
| CDC-Diabetes | 10.0000 | MIC-Public (shifted 5%) | Public-Only (shifted 5%) | 0.0265 | 0.0261 | 0.0268 | 1.863e-09 |

## Utility recovered UR (anchors Uniform and Non-private), primary model

| dataset | eps | method | UR | UR_lo | UR_hi |
|---|---|---|---|---|---|
| Adult | 0.1000 | Public-Only (15%) | 0.9570 | 0.9516 | 0.9616 |
| Adult | 0.1000 | Public-Only (5%) | 0.9219 | 0.9119 | 0.9301 |
| Adult | 0.1000 | Public-Only (1%) | 0.8369 | 0.8162 | 0.8542 |
| Adult | 0.1000 | Public-Only (shifted 5%) | 0.9092 | 0.8976 | 0.9188 |
| Adult | 0.1000 | MIC-DP | 0.0386 | -0.0917 | 0.1605 |
| Adult | 0.1000 | MIC-Public (15%) | 0.0381 | -0.1746 | 0.2159 |
| Adult | 0.1000 | MIC-Public (5%) | -0.0986 | -0.2956 | 0.0789 |
| Adult | 0.1000 | MIC-Public (1%) | 0.0837 | -0.1152 | 0.2603 |
| Adult | 0.1000 | MIC-Public (shifted 5%) | 0.0527 | -0.1542 | 0.2108 |
| Adult | 0.1000 | Public-Pretrain + Uniform-FT (15%) | -0.0794 | -0.2164 | 0.0346 |
| Adult | 0.1000 | Public-Pretrain + Uniform-FT (5%) | -0.0794 | -0.2164 | 0.0346 |
| Adult | 0.1000 | Public-Pretrain + MIC-DP-FT (15%) | -0.0794 | -0.2164 | 0.0346 |
| Adult | 0.1000 | Public-Pretrain + MIC-DP-FT (5%) | -0.0794 | -0.2164 | 0.0346 |
| Adult | 0.1000 | Public + Uniform (pooled) (15%) | 0.0144 | -0.0216 | 0.0540 |
| Adult | 0.1000 | Public + Uniform (pooled) (5%) | 0.0136 | -0.0186 | 0.0454 |
| Adult | 0.3000 | Public-Only (15%) | 0.9472 | 0.9406 | 0.9525 |
| Adult | 0.3000 | Public-Only (5%) | 0.9039 | 0.8920 | 0.9136 |
| Adult | 0.3000 | Public-Only (1%) | 0.7994 | 0.7745 | 0.8198 |
| Adult | 0.3000 | Public-Only (shifted 5%) | 0.8883 | 0.8744 | 0.8996 |
| Adult | 0.3000 | MIC-DP | -0.1036 | -0.2874 | 0.0642 |
| Adult | 0.3000 | MIC-Public (15%) | 0.2056 | 0.0302 | 0.3675 |
| Adult | 0.3000 | MIC-Public (5%) | 0.0208 | -0.2052 | 0.2046 |
| Adult | 0.3000 | MIC-Public (1%) | 0.2124 | 0.0272 | 0.3674 |
| Adult | 0.3000 | MIC-Public (shifted 5%) | 0.0377 | -0.1743 | 0.2174 |
| Adult | 0.3000 | Public-Pretrain + Uniform-FT (15%) | -0.3277 | -0.4924 | -0.1931 |
| Adult | 0.3000 | Public-Pretrain + Uniform-FT (5%) | -0.3277 | -0.4924 | -0.1931 |
| Adult | 0.3000 | Public-Pretrain + MIC-DP-FT (15%) | -0.3277 | -0.4924 | -0.1931 |
| Adult | 0.3000 | Public-Pretrain + MIC-DP-FT (5%) | -0.3277 | -0.4924 | -0.1931 |
| Adult | 0.3000 | Public + Uniform (pooled) (15%) | 0.0201 | -0.0058 | 0.0435 |
| Adult | 0.3000 | Public + Uniform (pooled) (5%) | 0.0038 | -0.0444 | 0.0423 |
| Adult | 0.5000 | Public-Only (15%) | 0.9345 | 0.9253 | 0.9424 |
| Adult | 0.5000 | Public-Only (5%) | 0.8809 | 0.8642 | 0.8953 |
| Adult | 0.5000 | Public-Only (1%) | 0.7514 | 0.7165 | 0.7814 |
| Adult | 0.5000 | Public-Only (shifted 5%) | 0.8615 | 0.8421 | 0.8782 |
| Adult | 0.5000 | MIC-DP | -0.2159 | -0.4887 | 0.0150 |
| Adult | 0.5000 | MIC-Public (15%) | 0.0461 | -0.2019 | 0.2532 |
| Adult | 0.5000 | MIC-Public (5%) | -0.0296 | -0.2504 | 0.1651 |
| Adult | 0.5000 | MIC-Public (1%) | 0.0994 | -0.1750 | 0.3183 |
| Adult | 0.5000 | MIC-Public (shifted 5%) | 0.0972 | -0.1555 | 0.3025 |
| Adult | 0.5000 | Public-Pretrain + Uniform-FT (15%) | -0.6454 | -0.8766 | -0.4470 |
| Adult | 0.5000 | Public-Pretrain + Uniform-FT (5%) | -0.6454 | -0.8766 | -0.4470 |
| Adult | 0.5000 | Public-Pretrain + MIC-DP-FT (15%) | -0.6255 | -0.8672 | -0.4151 |
| Adult | 0.5000 | Public-Pretrain + MIC-DP-FT (5%) | -0.6454 | -0.8766 | -0.4470 |
| Adult | 0.5000 | Public + Uniform (pooled) (15%) | 0.0449 | 0.0047 | 0.0858 |
| Adult | 0.5000 | Public + Uniform (pooled) (5%) | 0.0088 | -0.0338 | 0.0472 |
| Adult | 1.0000 | Public-Only (15%) | 0.9225 | 0.9121 | 0.9315 |
| Adult | 1.0000 | Public-Only (5%) | 0.8590 | 0.8402 | 0.8755 |
| Adult | 1.0000 | Public-Only (1%) | 0.7057 | 0.6664 | 0.7401 |
| Adult | 1.0000 | Public-Only (shifted 5%) | 0.8361 | 0.8142 | 0.8552 |
| Adult | 1.0000 | MIC-DP | -0.0158 | -0.2331 | 0.1638 |
| Adult | 1.0000 | MIC-Public (15%) | 0.3246 | 0.1966 | 0.4323 |
| Adult | 1.0000 | MIC-Public (5%) | 0.1984 | -0.0448 | 0.3870 |
| Adult | 1.0000 | MIC-Public (1%) | 0.2906 | 0.1519 | 0.4086 |
| Adult | 1.0000 | MIC-Public (shifted 5%) | 0.1565 | -0.0709 | 0.3327 |
| Adult | 1.0000 | Public-Pretrain + Uniform-FT (15%) | -0.9483 | -1.2080 | -0.7205 |
| Adult | 1.0000 | Public-Pretrain + Uniform-FT (5%) | -0.9483 | -1.2080 | -0.7205 |
| Adult | 1.0000 | Public-Pretrain + MIC-DP-FT (15%) | -0.9117 | -1.1762 | -0.6800 |
| Adult | 1.0000 | Public-Pretrain + MIC-DP-FT (5%) | -0.9483 | -1.2080 | -0.7205 |
| Adult | 1.0000 | Public + Uniform (pooled) (15%) | 0.0411 | 0.0095 | 0.0745 |
| Adult | 1.0000 | Public + Uniform (pooled) (5%) | 0.0040 | -0.0344 | 0.0414 |
| Adult | 5.0000 | Public-Only (15%) | 0.8684 | 0.8647 | 0.8717 |
| Adult | 5.0000 | Public-Only (5%) | 0.7606 | 0.7539 | 0.7667 |
| Adult | 5.0000 | Public-Only (1%) | 0.5004 | 0.4864 | 0.5132 |
| Adult | 5.0000 | Public-Only (shifted 5%) | 0.7217 | 0.7139 | 0.7288 |
| Adult | 5.0000 | MIC-DP | 0.1624 | 0.1107 | 0.2122 |
| Adult | 5.0000 | MIC-Public (15%) | 0.1673 | 0.1386 | 0.1952 |
| Adult | 5.0000 | MIC-Public (5%) | 0.1985 | 0.1671 | 0.2269 |
| Adult | 5.0000 | MIC-Public (1%) | 0.1407 | 0.1032 | 0.1769 |
| Adult | 5.0000 | MIC-Public (shifted 5%) | 0.0751 | 0.0445 | 0.1057 |
| Adult | 5.0000 | Public-Pretrain + Uniform-FT (15%) | -2.3070 | -2.3999 | -2.2226 |
| Adult | 5.0000 | Public-Pretrain + Uniform-FT (5%) | -2.3070 | -2.3999 | -2.2226 |
| Adult | 5.0000 | Public-Pretrain + MIC-DP-FT (15%) | -2.3070 | -2.3999 | -2.2226 |
| Adult | 5.0000 | Public-Pretrain + MIC-DP-FT (5%) | -2.3070 | -2.3999 | -2.2226 |
| Adult | 5.0000 | Public + Uniform (pooled) (15%) | 0.0193 | 0.0115 | 0.0274 |
| Adult | 5.0000 | Public + Uniform (pooled) (5%) | 0.0106 | 0.0032 | 0.0175 |
| Adult | 10.0000 | Public-Only (15%) | 0.8658 | 0.8641 | 0.8675 |
| Adult | 10.0000 | Public-Only (5%) | 0.7559 | 0.7529 | 0.7591 |
| Adult | 10.0000 | Public-Only (1%) | 0.4907 | 0.4842 | 0.4972 |
| Adult | 10.0000 | Public-Only (shifted 5%) | 0.7163 | 0.7127 | 0.7199 |
| Adult | 10.0000 | MIC-DP | 0.1708 | 0.1486 | 0.1932 |
| Adult | 10.0000 | MIC-Public (15%) | 0.2084 | 0.1926 | 0.2248 |
| Adult | 10.0000 | MIC-Public (5%) | 0.1911 | 0.1743 | 0.2078 |
| Adult | 10.0000 | MIC-Public (1%) | 0.1453 | 0.1284 | 0.1610 |
| Adult | 10.0000 | MIC-Public (shifted 5%) | 0.0956 | 0.0777 | 0.1111 |
| Adult | 10.0000 | Public-Pretrain + Uniform-FT (15%) | -2.3534 | -2.3962 | -2.3093 |
| Adult | 10.0000 | Public-Pretrain + Uniform-FT (5%) | -2.3368 | -2.4020 | -2.2554 |
| Adult | 10.0000 | Public-Pretrain + MIC-DP-FT (15%) | -2.3038 | -2.4075 | -2.1404 |
| Adult | 10.0000 | Public-Pretrain + MIC-DP-FT (5%) | -2.3715 | -2.4142 | -2.3281 |
| Adult | 10.0000 | Public + Uniform (pooled) (15%) | 0.0082 | 0.0049 | 0.0116 |
| Adult | 10.0000 | Public + Uniform (pooled) (5%) | 0.0045 | 0.0016 | 0.0074 |
| CDC-Diabetes | 0.1000 | Public-Only (15%) | 0.9332 | 0.9233 | 0.9414 |
| CDC-Diabetes | 0.1000 | Public-Only (5%) | 0.8834 | 0.8661 | 0.8977 |
| CDC-Diabetes | 0.1000 | Public-Only (1%) | 0.8870 | 0.8702 | 0.9009 |
| CDC-Diabetes | 0.1000 | Public-Only (shifted 5%) | 0.8381 | 0.8140 | 0.8579 |
| CDC-Diabetes | 0.1000 | MIC-DP | -0.0406 | -0.2145 | 0.1153 |
| CDC-Diabetes | 0.1000 | MIC-Public (15%) | 0.1134 | -0.1313 | 0.3173 |
| CDC-Diabetes | 0.1000 | MIC-Public (5%) | 0.2506 | 0.0115 | 0.4597 |
| CDC-Diabetes | 0.1000 | MIC-Public (1%) | 0.3167 | 0.1205 | 0.4965 |
| CDC-Diabetes | 0.1000 | MIC-Public (shifted 5%) | 0.1376 | -0.1212 | 0.3581 |
| CDC-Diabetes | 0.1000 | Public-Pretrain + Uniform-FT (15%) | 0.6434 | 0.5998 | 0.6855 |
| CDC-Diabetes | 0.1000 | Public-Pretrain + Uniform-FT (5%) | 0.7759 | 0.7270 | 0.8153 |
| CDC-Diabetes | 0.1000 | Public-Pretrain + MIC-DP-FT (15%) | 0.6816 | 0.6082 | 0.7437 |
| CDC-Diabetes | 0.1000 | Public-Pretrain + MIC-DP-FT (5%) | 0.7330 | 0.6474 | 0.7993 |
| CDC-Diabetes | 0.1000 | Public + Uniform (pooled) (15%) | 0.0591 | 0.0024 | 0.1062 |
| CDC-Diabetes | 0.1000 | Public + Uniform (pooled) (5%) | 0.0326 | -0.0088 | 0.0738 |
| CDC-Diabetes | 0.3000 | Public-Only (15%) | 0.9132 | 0.8971 | 0.9250 |
| CDC-Diabetes | 0.3000 | Public-Only (5%) | 0.8486 | 0.8205 | 0.8693 |
| CDC-Diabetes | 0.3000 | Public-Only (1%) | 0.8533 | 0.8260 | 0.8733 |
| CDC-Diabetes | 0.3000 | Public-Only (shifted 5%) | 0.7897 | 0.7506 | 0.8184 |
| CDC-Diabetes | 0.3000 | MIC-DP | 0.0998 | -0.1095 | 0.2665 |
| CDC-Diabetes | 0.3000 | MIC-Public (15%) | 0.4762 | 0.2645 | 0.6401 |
| CDC-Diabetes | 0.3000 | MIC-Public (5%) | 0.5124 | 0.3444 | 0.6425 |
| CDC-Diabetes | 0.3000 | MIC-Public (1%) | 0.3385 | 0.0942 | 0.5478 |
| CDC-Diabetes | 0.3000 | MIC-Public (shifted 5%) | 0.3405 | 0.0830 | 0.5393 |
| CDC-Diabetes | 0.3000 | Public-Pretrain + Uniform-FT (15%) | 0.6271 | 0.5605 | 0.6917 |
| CDC-Diabetes | 0.3000 | Public-Pretrain + Uniform-FT (5%) | 0.6680 | 0.5694 | 0.7503 |
| CDC-Diabetes | 0.3000 | Public-Pretrain + MIC-DP-FT (15%) | 0.6880 | 0.6396 | 0.7284 |
| CDC-Diabetes | 0.3000 | Public-Pretrain + MIC-DP-FT (5%) | 0.6635 | 0.5606 | 0.7403 |
| CDC-Diabetes | 0.3000 | Public + Uniform (pooled) (15%) | 0.0743 | 0.0271 | 0.1243 |
| CDC-Diabetes | 0.3000 | Public + Uniform (pooled) (5%) | 0.0341 | -0.0182 | 0.0849 |
| CDC-Diabetes | 0.5000 | Public-Only (15%) | 0.8943 | 0.8710 | 0.9113 |
| CDC-Diabetes | 0.5000 | Public-Only (5%) | 0.8157 | 0.7749 | 0.8452 |
| CDC-Diabetes | 0.5000 | Public-Only (1%) | 0.8213 | 0.7818 | 0.8500 |
| CDC-Diabetes | 0.5000 | Public-Only (shifted 5%) | 0.7439 | 0.6873 | 0.7850 |
| CDC-Diabetes | 0.5000 | MIC-DP | -0.2160 | -0.5417 | 0.0437 |
| CDC-Diabetes | 0.5000 | MIC-Public (15%) | 0.4518 | 0.1848 | 0.6445 |
| CDC-Diabetes | 0.5000 | MIC-Public (5%) | 0.5334 | 0.2890 | 0.7162 |
| CDC-Diabetes | 0.5000 | MIC-Public (1%) | 0.6293 | 0.4253 | 0.7688 |
| CDC-Diabetes | 0.5000 | MIC-Public (shifted 5%) | 0.6490 | 0.5542 | 0.7291 |
| CDC-Diabetes | 0.5000 | Public-Pretrain + Uniform-FT (15%) | 0.5739 | 0.4943 | 0.6483 |
| CDC-Diabetes | 0.5000 | Public-Pretrain + Uniform-FT (5%) | 0.5800 | 0.4841 | 0.6572 |
| CDC-Diabetes | 0.5000 | Public-Pretrain + MIC-DP-FT (15%) | 0.4801 | 0.3198 | 0.6025 |
| CDC-Diabetes | 0.5000 | Public-Pretrain + MIC-DP-FT (5%) | 0.6115 | 0.4949 | 0.6990 |
| CDC-Diabetes | 0.5000 | Public + Uniform (pooled) (15%) | 0.1169 | 0.0678 | 0.1591 |
| CDC-Diabetes | 0.5000 | Public + Uniform (pooled) (5%) | 0.0602 | 0.0198 | 0.0980 |
| CDC-Diabetes | 1.0000 | Public-Only (15%) | 0.8140 | 0.7802 | 0.8403 |
| CDC-Diabetes | 1.0000 | Public-Only (5%) | 0.6756 | 0.6167 | 0.7215 |
| CDC-Diabetes | 1.0000 | Public-Only (1%) | 0.6856 | 0.6284 | 0.7301 |
| CDC-Diabetes | 1.0000 | Public-Only (shifted 5%) | 0.5494 | 0.4675 | 0.6131 |
| CDC-Diabetes | 1.0000 | MIC-DP | -0.1887 | -0.4714 | 0.0491 |
| CDC-Diabetes | 1.0000 | MIC-Public (15%) | 0.7147 | 0.6575 | 0.7617 |
| CDC-Diabetes | 1.0000 | MIC-Public (5%) | 0.6624 | 0.5457 | 0.7434 |
| CDC-Diabetes | 1.0000 | MIC-Public (1%) | 0.6851 | 0.6151 | 0.7392 |
| CDC-Diabetes | 1.0000 | MIC-Public (shifted 5%) | 0.7088 | 0.6409 | 0.7607 |
| CDC-Diabetes | 1.0000 | Public-Pretrain + Uniform-FT (15%) | 0.5525 | 0.4691 | 0.6195 |
| CDC-Diabetes | 1.0000 | Public-Pretrain + Uniform-FT (5%) | 0.3318 | 0.1902 | 0.4466 |
| CDC-Diabetes | 1.0000 | Public-Pretrain + MIC-DP-FT (15%) | 0.3907 | 0.2645 | 0.4971 |
| CDC-Diabetes | 1.0000 | Public-Pretrain + MIC-DP-FT (5%) | 0.1901 | 0.0222 | 0.3397 |
| CDC-Diabetes | 1.0000 | Public + Uniform (pooled) (15%) | 0.1574 | 0.0885 | 0.2199 |
| CDC-Diabetes | 1.0000 | Public + Uniform (pooled) (5%) | 0.0235 | -0.0691 | 0.1043 |
| CDC-Diabetes | 5.0000 | Public-Only (15%) | 0.3509 | 0.3155 | 0.3824 |
| CDC-Diabetes | 5.0000 | Public-Only (5%) | -0.1322 | -0.1939 | -0.0771 |
| CDC-Diabetes | 5.0000 | Public-Only (1%) | -0.0975 | -0.1573 | -0.0441 |
| CDC-Diabetes | 5.0000 | Public-Only (shifted 5%) | -0.5730 | -0.6588 | -0.4965 |
| CDC-Diabetes | 5.0000 | MIC-DP | -0.0854 | -0.1743 | -0.0080 |
| CDC-Diabetes | 5.0000 | MIC-Public (15%) | 0.2157 | 0.1656 | 0.2612 |
| CDC-Diabetes | 5.0000 | MIC-Public (5%) | 0.1842 | 0.1360 | 0.2278 |
| CDC-Diabetes | 5.0000 | MIC-Public (1%) | 0.2256 | 0.1850 | 0.2630 |
| CDC-Diabetes | 5.0000 | MIC-Public (shifted 5%) | 0.2278 | 0.1720 | 0.2776 |
| CDC-Diabetes | 5.0000 | Public-Pretrain + Uniform-FT (15%) | 0.0802 | -0.0611 | 0.1642 |
| CDC-Diabetes | 5.0000 | Public-Pretrain + Uniform-FT (5%) | -2.0285 | -2.7924 | -1.3306 |
| CDC-Diabetes | 5.0000 | Public-Pretrain + MIC-DP-FT (15%) | 0.0652 | -0.0098 | 0.1308 |
| CDC-Diabetes | 5.0000 | Public-Pretrain + MIC-DP-FT (5%) | -2.7550 | -4.0888 | -1.6071 |
| CDC-Diabetes | 5.0000 | Public + Uniform (pooled) (15%) | 0.0522 | 0.0387 | 0.0652 |
| CDC-Diabetes | 5.0000 | Public + Uniform (pooled) (5%) | 0.0178 | 0.0031 | 0.0321 |
| CDC-Diabetes | 10.0000 | Public-Only (15%) | 0.2254 | 0.1982 | 0.2505 |
| CDC-Diabetes | 10.0000 | Public-Only (5%) | -0.3510 | -0.3985 | -0.3073 |
| CDC-Diabetes | 10.0000 | Public-Only (1%) | -0.3096 | -0.3556 | -0.2672 |
| CDC-Diabetes | 10.0000 | Public-Only (shifted 5%) | -0.8770 | -0.9430 | -0.8163 |
| CDC-Diabetes | 10.0000 | MIC-DP | 0.0269 | -0.0311 | 0.0871 |
| CDC-Diabetes | 10.0000 | MIC-Public (15%) | 0.0991 | 0.0628 | 0.1314 |
| CDC-Diabetes | 10.0000 | MIC-Public (5%) | 0.0460 | 0.0117 | 0.0787 |
| CDC-Diabetes | 10.0000 | MIC-Public (1%) | 0.0994 | 0.0666 | 0.1283 |
| CDC-Diabetes | 10.0000 | MIC-Public (shifted 5%) | 0.0997 | 0.0658 | 0.1305 |
| CDC-Diabetes | 10.0000 | Public-Pretrain + Uniform-FT (15%) | 0.1244 | 0.1085 | 0.1404 |
| CDC-Diabetes | 10.0000 | Public-Pretrain + Uniform-FT (5%) | -1.9642 | -2.6778 | -1.3363 |
| CDC-Diabetes | 10.0000 | Public-Pretrain + MIC-DP-FT (15%) | -0.1391 | -0.3964 | 0.0676 |
| CDC-Diabetes | 10.0000 | Public-Pretrain + MIC-DP-FT (5%) | -3.0625 | -4.4049 | -1.8581 |
| CDC-Diabetes | 10.0000 | Public + Uniform (pooled) (15%) | 0.0316 | 0.0214 | 0.0422 |
| CDC-Diabetes | 10.0000 | Public + Uniform (pooled) (5%) | 0.0256 | 0.0178 | 0.0340 |

## Family F6: best private method vs Public-Only (Holm over 36 tests)

| dataset | size | eps | A | B | mean_a | mean_b | diff | ci_lo | ci_hi | n_a_wins | n_b_wins | p_raw | p_holm | outcome |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Adult | 15% | 0.1000 | MIC-DP | Public-Only (15%) | 0.5468 | 0.9109 | -0.3641 | -0.4161 | -0.3128 | 0 | 30 | 1.863e-09 | 6.706e-08 | B better |
| Adult | 15% | 0.3000 | MIC-Public (15%) | Public-Only (15%) | 0.6719 | 0.9109 | -0.2390 | -0.2874 | -0.1946 | 0 | 30 | 1.863e-09 | 6.706e-08 | B better |
| Adult | 15% | 0.5000 | MIC-Public (15%) | Public-Only (15%) | 0.6799 | 0.9109 | -0.2311 | -0.2749 | -0.1904 | 0 | 30 | 1.863e-09 | 6.706e-08 | B better |
| Adult | 15% | 1.0000 | MIC-Public (15%) | Public-Only (15%) | 0.7796 | 0.9109 | -0.1313 | -0.1484 | -0.1157 | 0 | 30 | 1.863e-09 | 6.706e-08 | B better |
| Adult | 15% | 5.0000 | MIC-Public (15%) | Public-Only (15%) | 0.8202 | 0.9109 | -0.0907 | -0.0936 | -0.0879 | 0 | 30 | 1.863e-09 | 6.706e-08 | B better |
| Adult | 15% | 10.0000 | MIC-Public (15%) | Public-Only (15%) | 0.8275 | 0.9109 | -0.0834 | -0.0852 | -0.0817 | 0 | 30 | 1.863e-09 | 6.706e-08 | B better |
| Adult | 5% | 0.1000 | MIC-DP | Public-Only (5%) | 0.5468 | 0.8970 | -0.3502 | -0.4022 | -0.2976 | 0 | 30 | 1.863e-09 | 6.706e-08 | B better |
| Adult | 5% | 0.3000 | MIC-Public (15%) | Public-Only (5%) | 0.6719 | 0.8970 | -0.2251 | -0.2734 | -0.1804 | 0 | 30 | 1.863e-09 | 6.706e-08 | B better |
| Adult | 5% | 0.5000 | MIC-Public (15%) | Public-Only (5%) | 0.6799 | 0.8970 | -0.2171 | -0.2622 | -0.1772 | 0 | 30 | 1.863e-09 | 6.706e-08 | B better |
| Adult | 5% | 1.0000 | MIC-Public (15%) | Public-Only (5%) | 0.7796 | 0.8970 | -0.1174 | -0.1349 | -0.1018 | 0 | 30 | 1.863e-09 | 6.706e-08 | B better |
| Adult | 5% | 5.0000 | MIC-Public (15%) | Public-Only (5%) | 0.8202 | 0.8970 | -0.0768 | -0.0796 | -0.0739 | 0 | 30 | 1.863e-09 | 6.706e-08 | B better |
| Adult | 5% | 10.0000 | MIC-Public (15%) | Public-Only (5%) | 0.8275 | 0.8970 | -0.0695 | -0.0712 | -0.0677 | 0 | 30 | 1.863e-09 | 6.706e-08 | B better |
| Adult | 1% | 0.1000 | MIC-DP | Public-Only (1%) | 0.5468 | 0.8633 | -0.3165 | -0.3694 | -0.2641 | 0 | 30 | 1.863e-09 | 6.706e-08 | B better |
| Adult | 1% | 0.3000 | MIC-Public (15%) | Public-Only (1%) | 0.6719 | 0.8633 | -0.1914 | -0.2388 | -0.1485 | 0 | 30 | 1.863e-09 | 6.706e-08 | B better |
| Adult | 1% | 0.5000 | MIC-Public (15%) | Public-Only (1%) | 0.6799 | 0.8633 | -0.1834 | -0.2281 | -0.1438 | 0 | 30 | 1.863e-09 | 6.706e-08 | B better |
| Adult | 1% | 1.0000 | MIC-Public (15%) | Public-Only (1%) | 0.7796 | 0.8633 | -0.0837 | -0.1008 | -0.0680 | 0 | 30 | 1.863e-09 | 6.706e-08 | B better |
| Adult | 1% | 5.0000 | MIC-Public (15%) | Public-Only (1%) | 0.8202 | 0.8633 | -0.0431 | -0.0460 | -0.0403 | 0 | 30 | 1.863e-09 | 6.706e-08 | B better |
| Adult | 1% | 10.0000 | MIC-Public (15%) | Public-Only (1%) | 0.8275 | 0.8633 | -0.0358 | -0.0375 | -0.0340 | 0 | 30 | 1.863e-09 | 6.706e-08 | B better |
| CDC-Diabetes | 15% | 0.1000 | MIC-Public (15%) | Public-Only (15%) | 0.5460 | 0.8037 | -0.2577 | -0.3219 | -0.1957 | 0 | 30 | 1.863e-09 | 6.706e-08 | B better |
| CDC-Diabetes | 15% | 0.3000 | MIC-Public (15%) | Public-Only (15%) | 0.6979 | 0.8037 | -0.1057 | -0.1508 | -0.0683 | 0 | 30 | 1.863e-09 | 6.706e-08 | B better |
| CDC-Diabetes | 15% | 0.5000 | MIC-Public (15%) | Public-Only (15%) | 0.7157 | 0.8037 | -0.0879 | -0.1302 | -0.0534 | 0 | 30 | 1.863e-09 | 6.706e-08 | B better |
| CDC-Diabetes | 15% | 1.0000 | MIC-Public (15%) | Public-Only (15%) | 0.7925 | 0.8037 | -0.0112 | -0.0153 | -0.0076 | 2 | 28 | 2.552e-07 | 1.276e-06 | B better |
| CDC-Diabetes | 15% | 5.0000 | MIC-Public (15%) | Public-Only (15%) | 0.7993 | 0.8037 | -0.0044 | -0.0051 | -0.0036 | 1 | 29 | 3.725e-09 | 6.706e-08 | B better |
| CDC-Diabetes | 15% | 10.0000 | MIC-Public (15%) | Public-Only (15%) | 0.8003 | 0.8037 | -0.0034 | -0.0038 | -0.0031 | 0 | 30 | 1.863e-09 | 6.706e-08 | B better |
| CDC-Diabetes | 5% | 0.1000 | MIC-Public (15%) | Public-Only (5%) | 0.5460 | 0.7881 | -0.2420 | -0.3068 | -0.1807 | 0 | 30 | 1.863e-09 | 6.706e-08 | B better |
| CDC-Diabetes | 5% | 0.3000 | MIC-Public (15%) | Public-Only (5%) | 0.6979 | 0.7881 | -0.0901 | -0.1348 | -0.0522 | 1 | 29 | 1.304e-08 | 7.823e-08 | B better |
| CDC-Diabetes | 5% | 0.5000 | MIC-Public (15%) | Public-Only (5%) | 0.7157 | 0.7881 | -0.0723 | -0.1139 | -0.0374 | 5 | 25 | 7.994e-06 | 2.398e-05 | B better |
| CDC-Diabetes | 5% | 1.0000 | MIC-Public (15%) | Public-Only (5%) | 0.7925 | 0.7881 | 0.0044 | 0.0003325 | 0.0080 | 24 | 6 | 0.0071 | 0.0142 | A better |
| CDC-Diabetes | 5% | 5.0000 | MIC-Public (15%) | Public-Only (5%) | 0.7993 | 0.7881 | 0.0113 | 0.0106 | 0.0120 | 30 | 0 | 1.863e-09 | 6.706e-08 | A better |
| CDC-Diabetes | 5% | 10.0000 | MIC-Public (15%) | Public-Only (5%) | 0.8003 | 0.7881 | 0.0122 | 0.0119 | 0.0126 | 30 | 0 | 1.863e-09 | 6.706e-08 | A better |
| CDC-Diabetes | 1% | 0.1000 | MIC-Public (15%) | Public-Only (1%) | 0.5460 | 0.7892 | -0.2431 | -0.3080 | -0.1813 | 0 | 30 | 1.863e-09 | 6.706e-08 | B better |
| CDC-Diabetes | 1% | 0.3000 | MIC-Public (15%) | Public-Only (1%) | 0.6979 | 0.7892 | -0.0912 | -0.1365 | -0.0535 | 1 | 29 | 3.725e-09 | 6.706e-08 | B better |
| CDC-Diabetes | 1% | 0.5000 | MIC-Public (15%) | Public-Only (1%) | 0.7157 | 0.7892 | -0.0734 | -0.1156 | -0.0387 | 5 | 25 | 2.349e-06 | 9.395e-06 | B better |
| CDC-Diabetes | 1% | 1.0000 | MIC-Public (15%) | Public-Only (1%) | 0.7925 | 0.7892 | 0.0033 | -0.000817 | 0.0070 | 23 | 7 | 0.0221 | 0.0221 | A better |
| CDC-Diabetes | 1% | 5.0000 | MIC-Public (15%) | Public-Only (1%) | 0.7993 | 0.7892 | 0.0101 | 0.0094 | 0.0108 | 30 | 0 | 1.863e-09 | 6.706e-08 | A better |
| CDC-Diabetes | 1% | 10.0000 | MIC-Public (15%) | Public-Only (1%) | 0.8003 | 0.7892 | 0.0111 | 0.0107 | 0.0114 | 30 | 0 | 1.863e-09 | 6.706e-08 | A better |

## Answers to the reporting rules

### Adult

1. Smallest public size at which Public-Only is significantly better than the best private method at every eps <= 1: **1%** (sizes at which this holds: 1%, 5%, 15%).

2. Largest eps at which the best private method is significantly better than Public-Only: 15%: none (all such eps: none); 5%: none (all such eps: none); 1%: none (all such eps: none).

3. Fine-tuned and pooled methods against both parents (F7, Holm over 144 tests):

| eps | size | method | vs_public_only | vs_private | better_than_both |
|---|---|---|---|---|---|
| 0.1000 | 15% | Public-Pretrain + Uniform-FT | B better | n.s. | no |
| 0.1000 | 15% | Public-Pretrain + MIC-DP-FT | B better | n.s. | no |
| 0.1000 | 15% | Public + Uniform | B better | n.s. | no |
| 0.1000 | 5% | Public-Pretrain + Uniform-FT | B better | n.s. | no |
| 0.1000 | 5% | Public-Pretrain + MIC-DP-FT | B better | n.s. | no |
| 0.1000 | 5% | Public + Uniform | B better | n.s. | no |
| 0.3000 | 15% | Public-Pretrain + Uniform-FT | B better | B better | no |
| 0.3000 | 15% | Public-Pretrain + MIC-DP-FT | B better | n.s. | no |
| 0.3000 | 15% | Public + Uniform | B better | n.s. | no |
| 0.3000 | 5% | Public-Pretrain + Uniform-FT | B better | B better | no |
| 0.3000 | 5% | Public-Pretrain + MIC-DP-FT | B better | n.s. | no |
| 0.3000 | 5% | Public + Uniform | B better | n.s. | no |
| 0.5000 | 15% | Public-Pretrain + Uniform-FT | B better | B better | no |
| 0.5000 | 15% | Public-Pretrain + MIC-DP-FT | B better | B better | no |
| 0.5000 | 15% | Public + Uniform | B better | n.s. | no |
| 0.5000 | 5% | Public-Pretrain + Uniform-FT | B better | B better | no |
| 0.5000 | 5% | Public-Pretrain + MIC-DP-FT | B better | B better | no |
| 0.5000 | 5% | Public + Uniform | B better | n.s. | no |
| 1.0000 | 15% | Public-Pretrain + Uniform-FT | B better | B better | no |
| 1.0000 | 15% | Public-Pretrain + MIC-DP-FT | B better | B better | no |
| 1.0000 | 15% | Public + Uniform | B better | n.s. | no |
| 1.0000 | 5% | Public-Pretrain + Uniform-FT | B better | B better | no |
| 1.0000 | 5% | Public-Pretrain + MIC-DP-FT | B better | B better | no |
| 1.0000 | 5% | Public + Uniform | B better | n.s. | no |
| 5.0000 | 15% | Public-Pretrain + Uniform-FT | B better | B better | no |
| 5.0000 | 15% | Public-Pretrain + MIC-DP-FT | B better | B better | no |
| 5.0000 | 15% | Public + Uniform | B better | A better | no |
| 5.0000 | 5% | Public-Pretrain + Uniform-FT | B better | B better | no |
| 5.0000 | 5% | Public-Pretrain + MIC-DP-FT | B better | B better | no |
| 5.0000 | 5% | Public + Uniform | B better | n.s. | no |
| 10.0000 | 15% | Public-Pretrain + Uniform-FT | B better | B better | no |
| 10.0000 | 15% | Public-Pretrain + MIC-DP-FT | B better | B better | no |
| 10.0000 | 15% | Public + Uniform | B better | A better | no |
| 10.0000 | 5% | Public-Pretrain + Uniform-FT | B better | B better | no |
| 10.0000 | 5% | Public-Pretrain + MIC-DP-FT | B better | B better | no |
| 10.0000 | 5% | Public + Uniform | B better | n.s. | no |

4. MIC-Public at 15%, 5% and 1% against MIC-DP (F8, Holm over 36 tests):

| eps | mic_dp | mic_public_15% | mic_public_5% | mic_public_1% | outcome_15% | outcome_5% | outcome_1% | stops_helping_at |
|---|---|---|---|---|---|---|---|---|
| 0.1000 | 0.5468 | 0.5466 | 0.4924 | 0.5647 | n.s. | n.s. | n.s. | 15% |
| 0.3000 | 0.5722 | 0.6719 | 0.6123 | 0.6741 | n.s. | n.s. | n.s. | 15% |
| 0.5000 | 0.6117 | 0.6799 | 0.6602 | 0.6937 | n.s. | n.s. | n.s. | 15% |
| 1.0000 | 0.7048 | 0.7796 | 0.7519 | 0.7721 | A better | n.s. | A better | 5% |
| 5.0000 | 0.8196 | 0.8202 | 0.8242 | 0.8167 | n.s. | n.s. | n.s. | 15% |
| 10.0000 | 0.8227 | 0.8275 | 0.8253 | 0.8195 | n.s. | n.s. | n.s. | 15% |

### CDC-Diabetes

1. Smallest public size at which Public-Only is significantly better than the best private method at every eps <= 1: **15%** (sizes at which this holds: 15%).

2. Largest eps at which the best private method is significantly better than Public-Only: 15%: none (all such eps: none); 5%: 10 (all such eps: 1, 5, 10); 1%: 10 (all such eps: 1, 5, 10).

3. Fine-tuned and pooled methods against both parents (F7, Holm over 144 tests):

| eps | size | method | vs_public_only | vs_private | better_than_both |
|---|---|---|---|---|---|
| 0.1000 | 15% | Public-Pretrain + Uniform-FT | B better | A better | no |
| 0.1000 | 15% | Public-Pretrain + MIC-DP-FT | B better | A better | no |
| 0.1000 | 15% | Public + Uniform | B better | A better | no |
| 0.1000 | 5% | Public-Pretrain + Uniform-FT | B better | A better | no |
| 0.1000 | 5% | Public-Pretrain + MIC-DP-FT | B better | A better | no |
| 0.1000 | 5% | Public + Uniform | B better | n.s. | no |
| 0.3000 | 15% | Public-Pretrain + Uniform-FT | B better | A better | no |
| 0.3000 | 15% | Public-Pretrain + MIC-DP-FT | B better | A better | no |
| 0.3000 | 15% | Public + Uniform | B better | n.s. | no |
| 0.3000 | 5% | Public-Pretrain + Uniform-FT | B better | A better | no |
| 0.3000 | 5% | Public-Pretrain + MIC-DP-FT | B better | A better | no |
| 0.3000 | 5% | Public + Uniform | B better | n.s. | no |
| 0.5000 | 15% | Public-Pretrain + Uniform-FT | B better | A better | no |
| 0.5000 | 15% | Public-Pretrain + MIC-DP-FT | B better | A better | no |
| 0.5000 | 15% | Public + Uniform | B better | A better | no |
| 0.5000 | 5% | Public-Pretrain + Uniform-FT | B better | A better | no |
| 0.5000 | 5% | Public-Pretrain + MIC-DP-FT | B better | A better | no |
| 0.5000 | 5% | Public + Uniform | B better | n.s. | no |
| 1.0000 | 15% | Public-Pretrain + Uniform-FT | B better | A better | no |
| 1.0000 | 15% | Public-Pretrain + MIC-DP-FT | B better | A better | no |
| 1.0000 | 15% | Public + Uniform | B better | A better | no |
| 1.0000 | 5% | Public-Pretrain + Uniform-FT | B better | A better | no |
| 1.0000 | 5% | Public-Pretrain + MIC-DP-FT | B better | n.s. | no |
| 1.0000 | 5% | Public + Uniform | B better | n.s. | no |
| 5.0000 | 15% | Public-Pretrain + Uniform-FT | B better | A better | no |
| 5.0000 | 15% | Public-Pretrain + MIC-DP-FT | B better | A better | no |
| 5.0000 | 15% | Public + Uniform | B better | A better | no |
| 5.0000 | 5% | Public-Pretrain + Uniform-FT | B better | B better | no |
| 5.0000 | 5% | Public-Pretrain + MIC-DP-FT | B better | B better | no |
| 5.0000 | 5% | Public + Uniform | A better | n.s. | no |
| 10.0000 | 15% | Public-Pretrain + Uniform-FT | B better | A better | no |
| 10.0000 | 15% | Public-Pretrain + MIC-DP-FT | B better | n.s. | no |
| 10.0000 | 15% | Public + Uniform | B better | A better | no |
| 10.0000 | 5% | Public-Pretrain + Uniform-FT | B better | B better | no |
| 10.0000 | 5% | Public-Pretrain + MIC-DP-FT | B better | B better | no |
| 10.0000 | 5% | Public + Uniform | A better | A better | yes |

4. MIC-Public at 15%, 5% and 1% against MIC-DP (F8, Holm over 36 tests):

| eps | mic_dp | mic_public_15% | mic_public_5% | mic_public_1% | outcome_15% | outcome_5% | outcome_1% | stops_helping_at |
|---|---|---|---|---|---|---|---|---|
| 0.1000 | 0.4976 | 0.5460 | 0.5891 | 0.6099 | n.s. | n.s. | n.s. | 15% |
| 0.3000 | 0.6069 | 0.6979 | 0.7067 | 0.6646 | A better | A better | n.s. | 1% |
| 0.5000 | 0.5830 | 0.7157 | 0.7319 | 0.7510 | A better | A better | A better | none |
| 1.0000 | 0.6904 | 0.7925 | 0.7866 | 0.7891 | A better | A better | A better | none |
| 5.0000 | 0.7896 | 0.7993 | 0.7983 | 0.7996 | A better | A better | A better | none |
| 10.0000 | 0.7983 | 0.8003 | 0.7988 | 0.8003 | n.s. | n.s. | n.s. | 15% |

5. Shifted public split (EXPLORATORY; unadjusted Wilcoxon p, no family):

| dataset | eps | Public-Only (shifted 5%) | Public-Only (5%) | MIC-Public (shifted 5%) | MIC-Public (5%) | MIC-DP | Uniform |
|---|---|---|---|---|---|---|---|
| Adult | 0.1000 | 0.892 ± 0.000 | 0.897 ± 0.000 | 0.552 ± 0.147 | 0.492 ± 0.185 | 0.547 ± 0.148 | 0.531 ± 0.129 |
| Adult | 0.3000 | 0.892 ± 0.000 | 0.897 ± 0.000 | 0.618 ± 0.157 | 0.612 ± 0.145 | 0.572 ± 0.128 | 0.606 ± 0.104 |
| Adult | 0.5000 | 0.892 ± 0.000 | 0.897 ± 0.000 | 0.693 ± 0.130 | 0.660 ± 0.141 | 0.612 ± 0.137 | 0.668 ± 0.094 |
| Adult | 1.0000 | 0.892 ± 0.000 | 0.897 ± 0.000 | 0.743 ± 0.083 | 0.752 ± 0.109 | 0.705 ± 0.097 | 0.708 ± 0.077 |
| Adult | 5.0000 | 0.892 ± 0.000 | 0.897 ± 0.000 | 0.808 ± 0.007 | 0.824 ± 0.007 | 0.820 ± 0.017 | 0.799 ± 0.010 |
| Adult | 10.0000 | 0.892 ± 0.000 | 0.897 ± 0.000 | 0.813 ± 0.005 | 0.825 ± 0.006 | 0.823 ± 0.008 | 0.801 ± 0.005 |
| CDC-Diabetes | 0.1000 | 0.774 ± 0.000 | 0.788 ± 0.000 | 0.554 ± 0.169 | 0.589 ± 0.188 | 0.498 ± 0.125 | 0.510 ± 0.120 |
| CDC-Diabetes | 0.3000 | 0.774 ± 0.000 | 0.788 ± 0.000 | 0.665 ± 0.135 | 0.707 ± 0.087 | 0.607 ± 0.113 | 0.583 ± 0.106 |
| CDC-Diabetes | 0.5000 | 0.774 ± 0.000 | 0.788 ± 0.000 | 0.755 ± 0.046 | 0.732 ± 0.117 | 0.583 ± 0.128 | 0.626 ± 0.107 |
| CDC-Diabetes | 1.0000 | 0.774 ± 0.000 | 0.788 ± 0.000 | 0.792 ± 0.012 | 0.787 ± 0.023 | 0.690 ± 0.056 | 0.712 ± 0.051 |
| CDC-Diabetes | 5.0000 | 0.774 ± 0.000 | 0.788 ± 0.000 | 0.800 ± 0.002 | 0.798 ± 0.002 | 0.790 ± 0.008 | 0.792 ± 0.005 |
| CDC-Diabetes | 10.0000 | 0.774 ± 0.000 | 0.788 ± 0.000 | 0.800 ± 0.001 | 0.799 ± 0.001 | 0.798 ± 0.004 | 0.798 ± 0.003 |

| dataset | eps | comparison | A | B | mean_a | mean_b | diff | ci_lo | ci_hi | n_a_wins | n_b_wins | p_raw | p_holm | outcome |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Adult | 0.1000 | best private - Public-Only (shifted 5%) | MIC-DP | Public-Only (shifted 5%) | 0.5468 | 0.8919 | -0.3452 | -0.3983 | -0.2937 | 0 | 30 | 1.863e-09 | nan | B better (unadjusted) |
| Adult | 0.1000 | Public-Only (shifted 5%) - Public-Only (5%) | Public-Only (shifted 5%) | Public-Only (5%) | 0.8919 | 0.8970 | -0.0050 | -0.0050 | -0.0050 | 0 | 30 | 4.32e-08 | nan | B better (unadjusted) |
| Adult | 0.1000 | MIC-Public (shifted 5%) - MIC-Public (5%) | MIC-Public (shifted 5%) | MIC-Public (5%) | 0.5524 | 0.4924 | 0.0600 | -0.0201 | 0.1357 | 20 | 10 | 0.1191 | nan | n.s. (unadjusted) |
| Adult | 0.1000 | MIC-Public (shifted 5%) - MIC-DP | MIC-Public (shifted 5%) | MIC-DP | 0.5524 | 0.5468 | 0.0056 | -0.0818 | 0.0882 | 17 | 13 | 0.8236 | nan | n.s. (unadjusted) |
| Adult | 0.3000 | best private - Public-Only (shifted 5%) | MIC-Public (15%) | Public-Only (shifted 5%) | 0.6719 | 0.8919 | -0.2200 | -0.2685 | -0.1766 | 0 | 30 | 1.863e-09 | nan | B better (unadjusted) |
| Adult | 0.3000 | Public-Only (shifted 5%) - Public-Only (5%) | Public-Only (shifted 5%) | Public-Only (5%) | 0.8919 | 0.8970 | -0.0050 | -0.0050 | -0.0050 | 0 | 30 | 4.32e-08 | nan | B better (unadjusted) |
| Adult | 0.3000 | MIC-Public (shifted 5%) - MIC-Public (5%) | MIC-Public (shifted 5%) | MIC-Public (5%) | 0.6178 | 0.6123 | 0.0054 | -0.0755 | 0.0908 | 15 | 15 | 0.9193 | nan | n.s. (unadjusted) |
| Adult | 0.3000 | MIC-Public (shifted 5%) - MIC-DP | MIC-Public (shifted 5%) | MIC-DP | 0.6178 | 0.5722 | 0.0455 | -0.0327 | 0.1230 | 17 | 13 | 0.3085 | nan | n.s. (unadjusted) |
| Adult | 0.5000 | best private - Public-Only (shifted 5%) | MIC-Public (15%) | Public-Only (shifted 5%) | 0.6799 | 0.8919 | -0.2121 | -0.2562 | -0.1726 | 0 | 30 | 1.863e-09 | nan | B better (unadjusted) |
| Adult | 0.5000 | Public-Only (shifted 5%) - Public-Only (5%) | Public-Only (shifted 5%) | Public-Only (5%) | 0.8919 | 0.8970 | -0.0050 | -0.0050 | -0.0050 | 0 | 30 | 4.32e-08 | nan | B better (unadjusted) |
| Adult | 0.5000 | MIC-Public (shifted 5%) - MIC-Public (5%) | MIC-Public (shifted 5%) | MIC-Public (5%) | 0.6932 | 0.6602 | 0.0330 | -0.0440 | 0.1091 | 18 | 12 | 0.3387 | nan | n.s. (unadjusted) |
| Adult | 0.5000 | MIC-Public (shifted 5%) - MIC-DP | MIC-Public (shifted 5%) | MIC-DP | 0.6932 | 0.6117 | 0.0814 | 0.0203 | 0.1483 | 20 | 10 | 0.0208 | nan | A better (unadjusted) |
| Adult | 1.0000 | best private - Public-Only (shifted 5%) | MIC-Public (15%) | Public-Only (shifted 5%) | 0.7796 | 0.8919 | -0.1123 | -0.1296 | -0.0969 | 0 | 30 | 1.863e-09 | nan | B better (unadjusted) |
| Adult | 1.0000 | Public-Only (shifted 5%) - Public-Only (5%) | Public-Only (shifted 5%) | Public-Only (5%) | 0.8919 | 0.8970 | -0.0050 | -0.0050 | -0.0050 | 0 | 30 | 4.32e-08 | nan | B better (unadjusted) |
| Adult | 1.0000 | MIC-Public (shifted 5%) - MIC-Public (5%) | MIC-Public (shifted 5%) | MIC-Public (5%) | 0.7427 | 0.7519 | -0.0092 | -0.0547 | 0.0450 | 14 | 16 | 0.4771 | nan | n.s. (unadjusted) |
| Adult | 1.0000 | MIC-Public (shifted 5%) - MIC-DP | MIC-Public (shifted 5%) | MIC-DP | 0.7427 | 0.7048 | 0.0379 | -0.0047 | 0.0772 | 21 | 9 | 0.0405 | nan | A better (unadjusted) |
| Adult | 5.0000 | best private - Public-Only (shifted 5%) | MIC-Public (15%) | Public-Only (shifted 5%) | 0.8202 | 0.8919 | -0.0717 | -0.0745 | -0.0689 | 0 | 30 | 1.863e-09 | nan | B better (unadjusted) |
| Adult | 5.0000 | Public-Only (shifted 5%) - Public-Only (5%) | Public-Only (shifted 5%) | Public-Only (5%) | 0.8919 | 0.8970 | -0.0050 | -0.0050 | -0.0050 | 0 | 30 | 4.32e-08 | nan | B better (unadjusted) |
| Adult | 5.0000 | MIC-Public (shifted 5%) - MIC-Public (5%) | MIC-Public (shifted 5%) | MIC-Public (5%) | 0.8083 | 0.8242 | -0.0160 | -0.0198 | -0.0121 | 3 | 27 | 2.049e-07 | nan | B better (unadjusted) |
| Adult | 5.0000 | MIC-Public (shifted 5%) - MIC-DP | MIC-Public (shifted 5%) | MIC-DP | 0.8083 | 0.8196 | -0.0113 | -0.0171 | -0.0056 | 8 | 22 | 0.0024 | nan | B better (unadjusted) |
| Adult | 10.0000 | best private - Public-Only (shifted 5%) | MIC-Public (15%) | Public-Only (shifted 5%) | 0.8275 | 0.8919 | -0.0645 | -0.0662 | -0.0626 | 0 | 30 | 1.863e-09 | nan | B better (unadjusted) |
| Adult | 10.0000 | Public-Only (shifted 5%) - Public-Only (5%) | Public-Only (shifted 5%) | Public-Only (5%) | 0.8919 | 0.8970 | -0.0050 | -0.0050 | -0.0050 | 0 | 30 | 4.32e-08 | nan | B better (unadjusted) |
| Adult | 10.0000 | MIC-Public (shifted 5%) - MIC-Public (5%) | MIC-Public (shifted 5%) | MIC-Public (5%) | 0.8131 | 0.8253 | -0.0121 | -0.0151 | -0.0092 | 2 | 28 | 1.304e-08 | nan | B better (unadjusted) |
| Adult | 10.0000 | MIC-Public (shifted 5%) - MIC-DP | MIC-Public (shifted 5%) | MIC-DP | 0.8131 | 0.8227 | -0.0095 | -0.0134 | -0.0062 | 3 | 27 | 9.984e-07 | nan | B better (unadjusted) |
| CDC-Diabetes | 0.1000 | best private - Public-Only (shifted 5%) | MIC-Public (15%) | Public-Only (shifted 5%) | 0.5460 | 0.7738 | -0.2278 | -0.2934 | -0.1650 | 2 | 28 | 3.539e-08 | nan | B better (unadjusted) |
| CDC-Diabetes | 0.1000 | Public-Only (shifted 5%) - Public-Only (5%) | Public-Only (shifted 5%) | Public-Only (5%) | 0.7738 | 0.7881 | -0.0143 | -0.0143 | -0.0143 | 0 | 30 | 4.32e-08 | nan | B better (unadjusted) |
| CDC-Diabetes | 0.1000 | MIC-Public (shifted 5%) - MIC-Public (5%) | MIC-Public (shifted 5%) | MIC-Public (5%) | 0.5536 | 0.5891 | -0.0355 | -0.1170 | 0.0461 | 13 | 17 | 0.4045 | nan | n.s. (unadjusted) |
| CDC-Diabetes | 0.1000 | MIC-Public (shifted 5%) - MIC-DP | MIC-Public (shifted 5%) | MIC-DP | 0.5536 | 0.4976 | 0.0560 | -0.0133 | 0.1251 | 20 | 10 | 0.1191 | nan | n.s. (unadjusted) |
| CDC-Diabetes | 0.3000 | best private - Public-Only (shifted 5%) | MIC-Public (15%) | Public-Only (shifted 5%) | 0.6979 | 0.7738 | -0.0759 | -0.1214 | -0.0385 | 7 | 23 | 1.397e-05 | nan | B better (unadjusted) |
| CDC-Diabetes | 0.3000 | Public-Only (shifted 5%) - Public-Only (5%) | Public-Only (shifted 5%) | Public-Only (5%) | 0.7738 | 0.7881 | -0.0143 | -0.0143 | -0.0143 | 0 | 30 | 4.32e-08 | nan | B better (unadjusted) |
| CDC-Diabetes | 0.3000 | MIC-Public (shifted 5%) - MIC-Public (5%) | MIC-Public (shifted 5%) | MIC-Public (5%) | 0.6651 | 0.7067 | -0.0416 | -0.0931 | 0.0125 | 14 | 16 | 0.1048 | nan | n.s. (unadjusted) |
| CDC-Diabetes | 0.3000 | MIC-Public (shifted 5%) - MIC-DP | MIC-Public (shifted 5%) | MIC-DP | 0.6651 | 0.6069 | 0.0582 | -0.0041 | 0.1186 | 18 | 12 | 0.0879 | nan | n.s. (unadjusted) |
| CDC-Diabetes | 0.5000 | best private - Public-Only (shifted 5%) | MIC-Public (15%) | Public-Only (shifted 5%) | 0.7157 | 0.7738 | -0.0581 | -0.0997 | -0.0238 | 12 | 18 | 0.0058 | nan | B better (unadjusted) |
| CDC-Diabetes | 0.5000 | Public-Only (shifted 5%) - Public-Only (5%) | Public-Only (shifted 5%) | Public-Only (5%) | 0.7738 | 0.7881 | -0.0143 | -0.0143 | -0.0143 | 0 | 30 | 4.32e-08 | nan | B better (unadjusted) |
| CDC-Diabetes | 0.5000 | MIC-Public (shifted 5%) - MIC-Public (5%) | MIC-Public (shifted 5%) | MIC-Public (5%) | 0.7549 | 0.7319 | 0.0230 | -0.0152 | 0.0681 | 17 | 13 | 0.7457 | nan | n.s. (unadjusted) |
| CDC-Diabetes | 0.5000 | MIC-Public (shifted 5%) - MIC-DP | MIC-Public (shifted 5%) | MIC-DP | 0.7549 | 0.5830 | 0.1719 | 0.1253 | 0.2231 | 27 | 3 | 8.009e-08 | nan | A better (unadjusted) |
| CDC-Diabetes | 1.0000 | best private - Public-Only (shifted 5%) | MIC-Public (15%) | Public-Only (shifted 5%) | 0.7925 | 0.7738 | 0.0187 | 0.0145 | 0.0224 | 29 | 1 | 4.712e-07 | nan | A better (unadjusted) |
| CDC-Diabetes | 1.0000 | Public-Only (shifted 5%) - Public-Only (5%) | Public-Only (shifted 5%) | Public-Only (5%) | 0.7738 | 0.7881 | -0.0143 | -0.0143 | -0.0143 | 0 | 30 | 4.32e-08 | nan | B better (unadjusted) |
| CDC-Diabetes | 1.0000 | MIC-Public (shifted 5%) - MIC-Public (5%) | MIC-Public (shifted 5%) | MIC-Public (5%) | 0.7918 | 0.7866 | 0.0052 | -0.0039 | 0.0156 | 18 | 12 | 0.1981 | nan | n.s. (unadjusted) |
| CDC-Diabetes | 1.0000 | MIC-Public (shifted 5%) - MIC-DP | MIC-Public (shifted 5%) | MIC-DP | 0.7918 | 0.6904 | 0.1014 | 0.0812 | 0.1217 | 30 | 0 | 1.863e-09 | nan | A better (unadjusted) |
| CDC-Diabetes | 5.0000 | best private - Public-Only (shifted 5%) | MIC-Public (15%) | Public-Only (shifted 5%) | 0.7993 | 0.7738 | 0.0255 | 0.0248 | 0.0262 | 30 | 0 | 1.863e-09 | nan | A better (unadjusted) |
| CDC-Diabetes | 5.0000 | Public-Only (shifted 5%) - Public-Only (5%) | Public-Only (shifted 5%) | Public-Only (5%) | 0.7738 | 0.7881 | -0.0143 | -0.0143 | -0.0143 | 0 | 30 | 4.32e-08 | nan | B better (unadjusted) |
| CDC-Diabetes | 5.0000 | MIC-Public (shifted 5%) - MIC-Public (5%) | MIC-Public (shifted 5%) | MIC-Public (5%) | 0.7997 | 0.7983 | 0.0014 | 0.0004773 | 0.0024 | 19 | 11 | 0.0128 | nan | A better (unadjusted) |
| CDC-Diabetes | 5.0000 | MIC-Public (shifted 5%) - MIC-DP | MIC-Public (shifted 5%) | MIC-DP | 0.7997 | 0.7896 | 0.0101 | 0.0073 | 0.0130 | 28 | 2 | 5.718e-07 | nan | A better (unadjusted) |
| CDC-Diabetes | 10.0000 | best private - Public-Only (shifted 5%) | MIC-Public (15%) | Public-Only (shifted 5%) | 0.8003 | 0.7738 | 0.0265 | 0.0261 | 0.0268 | 30 | 0 | 1.863e-09 | nan | A better (unadjusted) |
| CDC-Diabetes | 10.0000 | Public-Only (shifted 5%) - Public-Only (5%) | Public-Only (shifted 5%) | Public-Only (5%) | 0.7738 | 0.7881 | -0.0143 | -0.0143 | -0.0143 | 0 | 30 | 4.32e-08 | nan | B better (unadjusted) |
| CDC-Diabetes | 10.0000 | MIC-Public (shifted 5%) - MIC-Public (5%) | MIC-Public (shifted 5%) | MIC-Public (5%) | 0.8003 | 0.7988 | 0.0015 | 0.000971 | 0.0019 | 26 | 4 | 3.79e-06 | nan | A better (unadjusted) |
| CDC-Diabetes | 10.0000 | MIC-Public (shifted 5%) - MIC-DP | MIC-Public (shifted 5%) | MIC-DP | 0.8003 | 0.7983 | 0.0020 | 0.0003761 | 0.0036 | 21 | 9 | 0.0310 | nan | A better (unadjusted) |

## Family F7 (all tests)

| dataset | size | eps | method | parent | A | B | mean_a | mean_b | diff | ci_lo | ci_hi | n_a_wins | n_b_wins | p_raw | p_holm | outcome |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Adult | 15% | 0.1000 | Public-Pretrain + Uniform-FT | Public-Only | Public-Pretrain + Uniform-FT (15%) | Public-Only (15%) | 0.5000 | 0.9109 | -0.4109 | -0.4109 | -0.4109 | 0 | 30 | 4.32e-08 | 3.37e-06 | B better |
| Adult | 15% | 0.1000 | Public-Pretrain + Uniform-FT | private | Public-Pretrain + Uniform-FT (15%) | Uniform | 0.5000 | 0.5315 | -0.0315 | -0.0755 | 0.0152 | 12 | 18 | 0.1094 | 1.0000 | n.s. |
| Adult | 15% | 0.1000 | Public-Pretrain + MIC-DP-FT | Public-Only | Public-Pretrain + MIC-DP-FT (15%) | Public-Only (15%) | 0.5000 | 0.9109 | -0.4109 | -0.4109 | -0.4109 | 0 | 30 | 4.32e-08 | 3.37e-06 | B better |
| Adult | 15% | 0.1000 | Public-Pretrain + MIC-DP-FT | private | Public-Pretrain + MIC-DP-FT (15%) | MIC-DP | 0.5000 | 0.5468 | -0.0468 | -0.0973 | 0.0060 | 12 | 18 | 0.1142 | 1.0000 | n.s. |
| Adult | 15% | 0.1000 | Public + Uniform | Public-Only | Public + Uniform (pooled) (15%) | Public-Only (15%) | 0.5372 | 0.9109 | -0.3737 | -0.4173 | -0.3335 | 0 | 30 | 1.863e-09 | 2.682e-07 | B better |
| Adult | 15% | 0.1000 | Public + Uniform | private | Public + Uniform (pooled) (15%) | Uniform | 0.5372 | 0.5315 | 0.0057 | -0.0089 | 0.0221 | 16 | 14 | 0.7000 | 1.0000 | n.s. |
| Adult | 5% | 0.1000 | Public-Pretrain + Uniform-FT | Public-Only | Public-Pretrain + Uniform-FT (5%) | Public-Only (5%) | 0.5000 | 0.8970 | -0.3970 | -0.3970 | -0.3970 | 0 | 30 | 4.32e-08 | 3.37e-06 | B better |
| Adult | 5% | 0.1000 | Public-Pretrain + Uniform-FT | private | Public-Pretrain + Uniform-FT (5%) | Uniform | 0.5000 | 0.5315 | -0.0315 | -0.0746 | 0.0150 | 12 | 18 | 0.1094 | 1.0000 | n.s. |
| Adult | 5% | 0.1000 | Public-Pretrain + MIC-DP-FT | Public-Only | Public-Pretrain + MIC-DP-FT (5%) | Public-Only (5%) | 0.5000 | 0.8970 | -0.3970 | -0.3970 | -0.3970 | 0 | 30 | 4.32e-08 | 3.37e-06 | B better |
| Adult | 5% | 0.1000 | Public-Pretrain + MIC-DP-FT | private | Public-Pretrain + MIC-DP-FT (5%) | MIC-DP | 0.5000 | 0.5468 | -0.0468 | -0.0991 | 0.0058 | 12 | 18 | 0.1142 | 1.0000 | n.s. |
| Adult | 5% | 0.1000 | Public + Uniform | Public-Only | Public + Uniform (pooled) (5%) | Public-Only (5%) | 0.5369 | 0.8970 | -0.3601 | -0.4041 | -0.3182 | 0 | 30 | 1.863e-09 | 2.682e-07 | B better |
| Adult | 5% | 0.1000 | Public + Uniform | private | Public + Uniform (pooled) (5%) | Uniform | 0.5369 | 0.5315 | 0.0054 | -0.0071 | 0.0189 | 15 | 15 | 0.6263 | 1.0000 | n.s. |
| Adult | 15% | 0.3000 | Public-Pretrain + Uniform-FT | Public-Only | Public-Pretrain + Uniform-FT (15%) | Public-Only (15%) | 0.5000 | 0.9109 | -0.4109 | -0.4109 | -0.4109 | 0 | 30 | 4.32e-08 | 3.37e-06 | B better |
| Adult | 15% | 0.3000 | Public-Pretrain + Uniform-FT | private | Public-Pretrain + Uniform-FT (15%) | Uniform | 0.5000 | 0.6056 | -0.1056 | -0.1418 | -0.0685 | 4 | 26 | 4.422e-06 | 0.0001769 | B better |
| Adult | 15% | 0.3000 | Public-Pretrain + MIC-DP-FT | Public-Only | Public-Pretrain + MIC-DP-FT (15%) | Public-Only (15%) | 0.5000 | 0.9109 | -0.4109 | -0.4109 | -0.4109 | 0 | 30 | 4.32e-08 | 3.37e-06 | B better |
| Adult | 15% | 0.3000 | Public-Pretrain + MIC-DP-FT | private | Public-Pretrain + MIC-DP-FT (15%) | MIC-DP | 0.5000 | 0.5722 | -0.0722 | -0.1173 | -0.0277 | 10 | 20 | 0.0058 | 0.1271 | n.s. |
| Adult | 15% | 0.3000 | Public + Uniform | Public-Only | Public + Uniform (pooled) (15%) | Public-Only (15%) | 0.6121 | 0.9109 | -0.2988 | -0.3335 | -0.2648 | 0 | 30 | 1.863e-09 | 2.682e-07 | B better |
| Adult | 15% | 0.3000 | Public + Uniform | private | Public + Uniform (pooled) (15%) | Uniform | 0.6121 | 0.6056 | 0.0065 | -0.0020 | 0.0145 | 17 | 13 | 0.1191 | 1.0000 | n.s. |
| Adult | 5% | 0.3000 | Public-Pretrain + Uniform-FT | Public-Only | Public-Pretrain + Uniform-FT (5%) | Public-Only (5%) | 0.5000 | 0.8970 | -0.3970 | -0.3970 | -0.3970 | 0 | 30 | 4.32e-08 | 3.37e-06 | B better |
| Adult | 5% | 0.3000 | Public-Pretrain + Uniform-FT | private | Public-Pretrain + Uniform-FT (5%) | Uniform | 0.5000 | 0.6056 | -0.1056 | -0.1422 | -0.0682 | 4 | 26 | 4.422e-06 | 0.0001769 | B better |
| Adult | 5% | 0.3000 | Public-Pretrain + MIC-DP-FT | Public-Only | Public-Pretrain + MIC-DP-FT (5%) | Public-Only (5%) | 0.5000 | 0.8970 | -0.3970 | -0.3970 | -0.3970 | 0 | 30 | 4.32e-08 | 3.37e-06 | B better |
| Adult | 5% | 0.3000 | Public-Pretrain + MIC-DP-FT | private | Public-Pretrain + MIC-DP-FT (5%) | MIC-DP | 0.5000 | 0.5722 | -0.0722 | -0.1172 | -0.0278 | 10 | 20 | 0.0058 | 0.1271 | n.s. |
| Adult | 5% | 0.3000 | Public + Uniform | Public-Only | Public + Uniform (pooled) (5%) | Public-Only (5%) | 0.6069 | 0.8970 | -0.2901 | -0.3273 | -0.2546 | 0 | 30 | 1.863e-09 | 2.682e-07 | B better |
| Adult | 5% | 0.3000 | Public + Uniform | private | Public + Uniform (pooled) (5%) | Uniform | 0.6069 | 0.6056 | 0.0012 | -0.0138 | 0.0138 | 17 | 13 | 0.3931 | 1.0000 | n.s. |
| Adult | 15% | 0.5000 | Public-Pretrain + Uniform-FT | Public-Only | Public-Pretrain + Uniform-FT (15%) | Public-Only (15%) | 0.5000 | 0.9109 | -0.4109 | -0.4109 | -0.4109 | 0 | 30 | 4.32e-08 | 3.37e-06 | B better |
| Adult | 15% | 0.5000 | Public-Pretrain + Uniform-FT | private | Public-Pretrain + Uniform-FT (15%) | Uniform | 0.5000 | 0.6679 | -0.1679 | -0.1998 | -0.1332 | 2 | 28 | 1.863e-08 | 1.565e-06 | B better |
| Adult | 15% | 0.5000 | Public-Pretrain + MIC-DP-FT | Public-Only | Public-Pretrain + MIC-DP-FT (15%) | Public-Only (15%) | 0.5052 | 0.9109 | -0.4057 | -0.4109 | -0.3954 | 0 | 30 | 6.799e-08 | 3.943e-06 | B better |
| Adult | 15% | 0.5000 | Public-Pretrain + MIC-DP-FT | private | Public-Pretrain + MIC-DP-FT (15%) | MIC-DP | 0.5052 | 0.6117 | -0.1065 | -0.1527 | -0.0560 | 6 | 24 | 0.0010 | 0.0260 | B better |
| Adult | 15% | 0.5000 | Public + Uniform | Public-Only | Public + Uniform (pooled) (15%) | Public-Only (15%) | 0.6796 | 0.9109 | -0.2314 | -0.2632 | -0.2017 | 0 | 30 | 1.863e-09 | 2.682e-07 | B better |
| Adult | 15% | 0.5000 | Public + Uniform | private | Public + Uniform (pooled) (15%) | Uniform | 0.6796 | 0.6679 | 0.0117 | 0.0010 | 0.0230 | 20 | 10 | 0.0549 | 0.8238 | n.s. |
| Adult | 5% | 0.5000 | Public-Pretrain + Uniform-FT | Public-Only | Public-Pretrain + Uniform-FT (5%) | Public-Only (5%) | 0.5000 | 0.8970 | -0.3970 | -0.3970 | -0.3970 | 0 | 30 | 4.32e-08 | 3.37e-06 | B better |
| Adult | 5% | 0.5000 | Public-Pretrain + Uniform-FT | private | Public-Pretrain + Uniform-FT (5%) | Uniform | 0.5000 | 0.6679 | -0.1679 | -0.1995 | -0.1333 | 2 | 28 | 1.863e-08 | 1.565e-06 | B better |
| Adult | 5% | 0.5000 | Public-Pretrain + MIC-DP-FT | Public-Only | Public-Pretrain + MIC-DP-FT (5%) | Public-Only (5%) | 0.5000 | 0.8970 | -0.3970 | -0.3970 | -0.3970 | 0 | 30 | 4.32e-08 | 3.37e-06 | B better |
| Adult | 5% | 0.5000 | Public-Pretrain + MIC-DP-FT | private | Public-Pretrain + MIC-DP-FT (5%) | MIC-DP | 0.5000 | 0.6117 | -0.1117 | -0.1568 | -0.0599 | 5 | 25 | 0.0007979 | 0.0207 | B better |
| Adult | 5% | 0.5000 | Public + Uniform | Public-Only | Public + Uniform (pooled) (5%) | Public-Only (5%) | 0.6702 | 0.8970 | -0.2268 | -0.2581 | -0.1964 | 0 | 30 | 1.863e-09 | 2.682e-07 | B better |
| Adult | 5% | 0.5000 | Public + Uniform | private | Public + Uniform (pooled) (5%) | Uniform | 0.6702 | 0.6679 | 0.0023 | -0.0081 | 0.0126 | 16 | 14 | 0.7151 | 1.0000 | n.s. |
| Adult | 15% | 1.0000 | Public-Pretrain + Uniform-FT | Public-Only | Public-Pretrain + Uniform-FT (15%) | Public-Only (15%) | 0.5000 | 0.9109 | -0.4109 | -0.4109 | -0.4109 | 0 | 30 | 4.32e-08 | 3.37e-06 | B better |
| Adult | 15% | 1.0000 | Public-Pretrain + Uniform-FT | private | Public-Pretrain + Uniform-FT (15%) | Uniform | 0.5000 | 0.7083 | -0.2083 | -0.2340 | -0.1802 | 0 | 30 | 1.863e-09 | 2.682e-07 | B better |
| Adult | 15% | 1.0000 | Public-Pretrain + MIC-DP-FT | Public-Only | Public-Pretrain + MIC-DP-FT (15%) | Public-Only (15%) | 0.5080 | 0.9109 | -0.4029 | -0.4109 | -0.3868 | 0 | 30 | 6.799e-08 | 3.943e-06 | B better |
| Adult | 15% | 1.0000 | Public-Pretrain + MIC-DP-FT | private | Public-Pretrain + MIC-DP-FT (15%) | MIC-DP | 0.5080 | 0.7048 | -0.1968 | -0.2325 | -0.1574 | 3 | 27 | 3.539e-08 | 2.796e-06 | B better |
| Adult | 15% | 1.0000 | Public + Uniform | Public-Only | Public + Uniform (pooled) (15%) | Public-Only (15%) | 0.7173 | 0.9109 | -0.1936 | -0.2221 | -0.1684 | 0 | 30 | 1.863e-09 | 2.682e-07 | B better |
| Adult | 15% | 1.0000 | Public + Uniform | private | Public + Uniform (pooled) (15%) | Uniform | 0.7173 | 0.7083 | 0.0090 | 0.0020 | 0.0168 | 19 | 11 | 0.0449 | 0.7185 | n.s. |
| Adult | 5% | 1.0000 | Public-Pretrain + Uniform-FT | Public-Only | Public-Pretrain + Uniform-FT (5%) | Public-Only (5%) | 0.5000 | 0.8970 | -0.3970 | -0.3970 | -0.3970 | 0 | 30 | 4.32e-08 | 3.37e-06 | B better |
| Adult | 5% | 1.0000 | Public-Pretrain + Uniform-FT | private | Public-Pretrain + Uniform-FT (5%) | Uniform | 0.5000 | 0.7083 | -0.2083 | -0.2348 | -0.1800 | 0 | 30 | 1.863e-09 | 2.682e-07 | B better |
| Adult | 5% | 1.0000 | Public-Pretrain + MIC-DP-FT | Public-Only | Public-Pretrain + MIC-DP-FT (5%) | Public-Only (5%) | 0.5000 | 0.8970 | -0.3970 | -0.3970 | -0.3970 | 0 | 30 | 4.32e-08 | 3.37e-06 | B better |
| Adult | 5% | 1.0000 | Public-Pretrain + MIC-DP-FT | private | Public-Pretrain + MIC-DP-FT (5%) | MIC-DP | 0.5000 | 0.7048 | -0.2048 | -0.2371 | -0.1682 | 2 | 28 | 1.304e-08 | 1.134e-06 | B better |
| Adult | 5% | 1.0000 | Public + Uniform | Public-Only | Public + Uniform (pooled) (5%) | Public-Only (5%) | 0.7092 | 0.8970 | -0.1878 | -0.2176 | -0.1617 | 0 | 30 | 1.863e-09 | 2.682e-07 | B better |
| Adult | 5% | 1.0000 | Public + Uniform | private | Public + Uniform (pooled) (5%) | Uniform | 0.7092 | 0.7083 | 0.0008778 | -0.0077 | 0.0090 | 10 | 20 | 0.7000 | 1.0000 | n.s. |
| Adult | 15% | 5.0000 | Public-Pretrain + Uniform-FT | Public-Only | Public-Pretrain + Uniform-FT (15%) | Public-Only (15%) | 0.5000 | 0.9109 | -0.4109 | -0.4109 | -0.4109 | 0 | 30 | 4.32e-08 | 3.37e-06 | B better |
| Adult | 15% | 5.0000 | Public-Pretrain + Uniform-FT | private | Public-Pretrain + Uniform-FT (15%) | Uniform | 0.5000 | 0.7985 | -0.2985 | -0.3021 | -0.2951 | 0 | 30 | 1.863e-09 | 2.682e-07 | B better |
| Adult | 15% | 5.0000 | Public-Pretrain + MIC-DP-FT | Public-Only | Public-Pretrain + MIC-DP-FT (15%) | Public-Only (15%) | 0.5000 | 0.9109 | -0.4109 | -0.4109 | -0.4109 | 0 | 30 | 4.32e-08 | 3.37e-06 | B better |
| Adult | 15% | 5.0000 | Public-Pretrain + MIC-DP-FT | private | Public-Pretrain + MIC-DP-FT (15%) | MIC-DP | 0.5000 | 0.8196 | -0.3196 | -0.3252 | -0.3137 | 0 | 30 | 1.863e-09 | 2.682e-07 | B better |
| Adult | 15% | 5.0000 | Public + Uniform | Public-Only | Public + Uniform (pooled) (15%) | Public-Only (15%) | 0.8010 | 0.9109 | -0.1099 | -0.1135 | -0.1062 | 0 | 30 | 1.863e-09 | 2.682e-07 | B better |
| Adult | 15% | 5.0000 | Public + Uniform | private | Public + Uniform (pooled) (15%) | Uniform | 0.8010 | 0.7985 | 0.0025 | 0.0015 | 0.0035 | 25 | 5 | 6.287e-05 | 0.0021 | A better |
| Adult | 5% | 5.0000 | Public-Pretrain + Uniform-FT | Public-Only | Public-Pretrain + Uniform-FT (5%) | Public-Only (5%) | 0.5000 | 0.8970 | -0.3970 | -0.3970 | -0.3970 | 0 | 30 | 4.32e-08 | 3.37e-06 | B better |
| Adult | 5% | 5.0000 | Public-Pretrain + Uniform-FT | private | Public-Pretrain + Uniform-FT (5%) | Uniform | 0.5000 | 0.7985 | -0.2985 | -0.3021 | -0.2952 | 0 | 30 | 1.863e-09 | 2.682e-07 | B better |
| Adult | 5% | 5.0000 | Public-Pretrain + MIC-DP-FT | Public-Only | Public-Pretrain + MIC-DP-FT (5%) | Public-Only (5%) | 0.5000 | 0.8970 | -0.3970 | -0.3970 | -0.3970 | 0 | 30 | 4.32e-08 | 3.37e-06 | B better |
| Adult | 5% | 5.0000 | Public-Pretrain + MIC-DP-FT | private | Public-Pretrain + MIC-DP-FT (5%) | MIC-DP | 0.5000 | 0.8196 | -0.3196 | -0.3252 | -0.3137 | 0 | 30 | 1.863e-09 | 2.682e-07 | B better |
| Adult | 5% | 5.0000 | Public + Uniform | Public-Only | Public + Uniform (pooled) (5%) | Public-Only (5%) | 0.7999 | 0.8970 | -0.0971 | -0.1003 | -0.0939 | 0 | 30 | 1.863e-09 | 2.682e-07 | B better |
| Adult | 5% | 5.0000 | Public + Uniform | private | Public + Uniform (pooled) (5%) | Uniform | 0.7999 | 0.7985 | 0.0014 | 0.0004046 | 0.0023 | 22 | 8 | 0.0145 | 0.2908 | n.s. |
| Adult | 15% | 10.0000 | Public-Pretrain + Uniform-FT | Public-Only | Public-Pretrain + Uniform-FT (15%) | Public-Only (15%) | 0.5023 | 0.9109 | -0.4086 | -0.4109 | -0.4048 | 0 | 30 | 2.616e-07 | 1.203e-05 | B better |
| Adult | 15% | 10.0000 | Public-Pretrain + Uniform-FT | private | Public-Pretrain + Uniform-FT (15%) | Uniform | 0.5023 | 0.8010 | -0.2987 | -0.3015 | -0.2952 | 0 | 30 | 1.863e-09 | 2.682e-07 | B better |
| Adult | 15% | 10.0000 | Public-Pretrain + MIC-DP-FT | Public-Only | Public-Pretrain + MIC-DP-FT (15%) | Public-Only (15%) | 0.5086 | 0.9109 | -0.4023 | -0.4109 | -0.3853 | 0 | 30 | 1.014e-07 | 5.478e-06 | B better |
| Adult | 15% | 10.0000 | Public-Pretrain + MIC-DP-FT | private | Public-Pretrain + MIC-DP-FT (15%) | MIC-DP | 0.5086 | 0.8227 | -0.3141 | -0.3250 | -0.2950 | 0 | 30 | 1.863e-09 | 2.682e-07 | B better |
| Adult | 15% | 10.0000 | Public + Uniform | Public-Only | Public + Uniform (pooled) (15%) | Public-Only (15%) | 0.8021 | 0.9109 | -0.1089 | -0.1104 | -0.1074 | 0 | 30 | 1.863e-09 | 2.682e-07 | B better |
| Adult | 15% | 10.0000 | Public + Uniform | private | Public + Uniform (pooled) (15%) | Uniform | 0.8021 | 0.8010 | 0.0010 | 0.0006176 | 0.0015 | 24 | 6 | 6.287e-05 | 0.0021 | A better |
| Adult | 5% | 10.0000 | Public-Pretrain + Uniform-FT | Public-Only | Public-Pretrain + Uniform-FT (5%) | Public-Only (5%) | 0.5044 | 0.8970 | -0.3926 | -0.3971 | -0.3844 | 0 | 30 | 1.979e-07 | 9.697e-06 | B better |
| Adult | 5% | 10.0000 | Public-Pretrain + Uniform-FT | private | Public-Pretrain + Uniform-FT (5%) | Uniform | 0.5044 | 0.8010 | -0.2966 | -0.3020 | -0.2881 | 0 | 30 | 1.863e-09 | 2.682e-07 | B better |
| Adult | 5% | 10.0000 | Public-Pretrain + MIC-DP-FT | Public-Only | Public-Pretrain + MIC-DP-FT (5%) | Public-Only (5%) | 0.5000 | 0.8970 | -0.3970 | -0.3970 | -0.3970 | 0 | 30 | 4.32e-08 | 3.37e-06 | B better |
| Adult | 5% | 10.0000 | Public-Pretrain + MIC-DP-FT | private | Public-Pretrain + MIC-DP-FT (5%) | MIC-DP | 0.5000 | 0.8227 | -0.3227 | -0.3254 | -0.3200 | 0 | 30 | 1.863e-09 | 2.682e-07 | B better |
| Adult | 5% | 10.0000 | Public + Uniform | Public-Only | Public + Uniform (pooled) (5%) | Public-Only (5%) | 0.8016 | 0.8970 | -0.0954 | -0.0971 | -0.0938 | 0 | 30 | 1.863e-09 | 2.682e-07 | B better |
| Adult | 5% | 10.0000 | Public + Uniform | private | Public + Uniform (pooled) (5%) | Uniform | 0.8016 | 0.8010 | 0.0005761 | 0.0002019 | 0.0009492 | 23 | 7 | 0.0054 | 0.1238 | n.s. |
| CDC-Diabetes | 15% | 0.1000 | Public-Pretrain + Uniform-FT | Public-Only | Public-Pretrain + Uniform-FT (15%) | Public-Only (15%) | 0.7126 | 0.8037 | -0.0911 | -0.1138 | -0.0704 | 0 | 30 | 1.863e-09 | 2.682e-07 | B better |
| CDC-Diabetes | 15% | 0.1000 | Public-Pretrain + Uniform-FT | private | Public-Pretrain + Uniform-FT (15%) | Uniform | 0.7126 | 0.5104 | 0.2022 | 0.1745 | 0.2308 | 30 | 0 | 1.863e-09 | 2.682e-07 | A better |
| CDC-Diabetes | 15% | 0.1000 | Public-Pretrain + MIC-DP-FT | Public-Only | Public-Pretrain + MIC-DP-FT (15%) | Public-Only (15%) | 0.7246 | 0.8037 | -0.0791 | -0.1023 | -0.0589 | 0 | 30 | 1.863e-09 | 2.682e-07 | B better |
| CDC-Diabetes | 15% | 0.1000 | Public-Pretrain + MIC-DP-FT | private | Public-Pretrain + MIC-DP-FT (15%) | MIC-DP | 0.7246 | 0.4976 | 0.2270 | 0.1890 | 0.2626 | 30 | 0 | 1.863e-09 | 2.682e-07 | A better |
| CDC-Diabetes | 15% | 0.1000 | Public + Uniform | Public-Only | Public + Uniform (pooled) (15%) | Public-Only (15%) | 0.5290 | 0.8037 | -0.2747 | -0.3172 | -0.2342 | 0 | 30 | 1.863e-09 | 2.682e-07 | B better |
| CDC-Diabetes | 15% | 0.1000 | Public + Uniform | private | Public + Uniform (pooled) (15%) | Uniform | 0.5290 | 0.5104 | 0.0186 | 7.502e-05 | 0.0336 | 26 | 4 | 0.0005055 | 0.0147 | A better |
| CDC-Diabetes | 5% | 0.1000 | Public-Pretrain + Uniform-FT | Public-Only | Public-Pretrain + Uniform-FT (5%) | Public-Only (5%) | 0.7543 | 0.7881 | -0.0338 | -0.0458 | -0.0234 | 0 | 30 | 1.863e-09 | 2.682e-07 | B better |
| CDC-Diabetes | 5% | 0.1000 | Public-Pretrain + Uniform-FT | private | Public-Pretrain + Uniform-FT (5%) | Uniform | 0.7543 | 0.5104 | 0.2439 | 0.2028 | 0.2881 | 30 | 0 | 1.863e-09 | 2.682e-07 | A better |
| CDC-Diabetes | 5% | 0.1000 | Public-Pretrain + MIC-DP-FT | Public-Only | Public-Pretrain + MIC-DP-FT (5%) | Public-Only (5%) | 0.7408 | 0.7881 | -0.0473 | -0.0655 | -0.0312 | 1 | 29 | 3.725e-09 | 3.539e-07 | B better |
| CDC-Diabetes | 5% | 0.1000 | Public-Pretrain + MIC-DP-FT | private | Public-Pretrain + MIC-DP-FT (5%) | MIC-DP | 0.7408 | 0.4976 | 0.2431 | 0.1922 | 0.2939 | 29 | 1 | 3.725e-09 | 3.539e-07 | A better |
| CDC-Diabetes | 5% | 0.1000 | Public + Uniform | Public-Only | Public + Uniform (pooled) (5%) | Public-Only (5%) | 0.5206 | 0.7881 | -0.2674 | -0.3094 | -0.2283 | 0 | 30 | 1.863e-09 | 2.682e-07 | B better |
| CDC-Diabetes | 5% | 0.1000 | Public + Uniform | private | Public + Uniform (pooled) (5%) | Uniform | 0.5206 | 0.5104 | 0.0102 | -0.0027 | 0.0233 | 18 | 12 | 0.1981 | 1.0000 | n.s. |
| CDC-Diabetes | 15% | 0.3000 | Public-Pretrain + Uniform-FT | Public-Only | Public-Pretrain + Uniform-FT (15%) | Public-Only (15%) | 0.7345 | 0.8037 | -0.0692 | -0.0898 | -0.0509 | 0 | 30 | 1.863e-09 | 2.682e-07 | B better |
| CDC-Diabetes | 15% | 0.3000 | Public-Pretrain + Uniform-FT | private | Public-Pretrain + Uniform-FT (15%) | Uniform | 0.7345 | 0.5827 | 0.1518 | 0.1215 | 0.1835 | 30 | 0 | 1.863e-09 | 2.682e-07 | A better |
| CDC-Diabetes | 15% | 0.3000 | Public-Pretrain + MIC-DP-FT | Public-Only | Public-Pretrain + MIC-DP-FT (15%) | Public-Only (15%) | 0.7492 | 0.8037 | -0.0545 | -0.0648 | -0.0442 | 0 | 30 | 1.863e-09 | 2.682e-07 | B better |
| CDC-Diabetes | 15% | 0.3000 | Public-Pretrain + MIC-DP-FT | private | Public-Pretrain + MIC-DP-FT (15%) | MIC-DP | 0.7492 | 0.6069 | 0.1423 | 0.1092 | 0.1781 | 30 | 0 | 1.863e-09 | 2.682e-07 | A better |
| CDC-Diabetes | 15% | 0.3000 | Public + Uniform | Public-Only | Public + Uniform (pooled) (15%) | Public-Only (15%) | 0.6007 | 0.8037 | -0.2030 | -0.2408 | -0.1672 | 0 | 30 | 1.863e-09 | 2.682e-07 | B better |
| CDC-Diabetes | 15% | 0.3000 | Public + Uniform | private | Public + Uniform (pooled) (15%) | Uniform | 0.6007 | 0.5827 | 0.0180 | 0.0063 | 0.0302 | 19 | 11 | 0.0185 | 0.3521 | n.s. |
| CDC-Diabetes | 5% | 0.3000 | Public-Pretrain + Uniform-FT | Public-Only | Public-Pretrain + Uniform-FT (5%) | Public-Only (5%) | 0.7443 | 0.7881 | -0.0437 | -0.0596 | -0.0290 | 4 | 26 | 4.712e-07 | 2.121e-05 | B better |
| CDC-Diabetes | 5% | 0.3000 | Public-Pretrain + Uniform-FT | private | Public-Pretrain + Uniform-FT (5%) | Uniform | 0.7443 | 0.5827 | 0.1616 | 0.1200 | 0.2043 | 27 | 3 | 1.304e-07 | 6.78e-06 | A better |
| CDC-Diabetes | 5% | 0.3000 | Public-Pretrain + MIC-DP-FT | Public-Only | Public-Pretrain + MIC-DP-FT (5%) | Public-Only (5%) | 0.7433 | 0.7881 | -0.0448 | -0.0591 | -0.0310 | 3 | 27 | 2.049e-07 | 9.835e-06 | B better |
| CDC-Diabetes | 5% | 0.3000 | Public-Pretrain + MIC-DP-FT | private | Public-Pretrain + MIC-DP-FT (5%) | MIC-DP | 0.7433 | 0.6069 | 0.1364 | 0.0968 | 0.1781 | 29 | 1 | 8.009e-08 | 4.485e-06 | A better |
| CDC-Diabetes | 5% | 0.3000 | Public + Uniform | Public-Only | Public + Uniform (pooled) (5%) | Public-Only (5%) | 0.5910 | 0.7881 | -0.1971 | -0.2400 | -0.1575 | 0 | 30 | 1.863e-09 | 2.682e-07 | B better |
| CDC-Diabetes | 5% | 0.3000 | Public + Uniform | private | Public + Uniform (pooled) (5%) | Uniform | 0.5910 | 0.5827 | 0.0083 | -0.0044 | 0.0203 | 17 | 13 | 0.1142 | 1.0000 | n.s. |
| CDC-Diabetes | 15% | 0.5000 | Public-Pretrain + Uniform-FT | Public-Only | Public-Pretrain + Uniform-FT (15%) | Public-Only (15%) | 0.7400 | 0.8037 | -0.0637 | -0.0865 | -0.0440 | 0 | 30 | 1.863e-09 | 2.682e-07 | B better |
| CDC-Diabetes | 15% | 0.5000 | Public-Pretrain + Uniform-FT | private | Public-Pretrain + Uniform-FT (15%) | Uniform | 0.7400 | 0.6259 | 0.1141 | 0.0887 | 0.1435 | 30 | 0 | 1.863e-09 | 2.682e-07 | A better |
| CDC-Diabetes | 15% | 0.5000 | Public-Pretrain + MIC-DP-FT | Public-Only | Public-Pretrain + MIC-DP-FT (15%) | Public-Only (15%) | 0.7214 | 0.8037 | -0.0823 | -0.1052 | -0.0614 | 0 | 30 | 1.863e-09 | 2.682e-07 | B better |
| CDC-Diabetes | 15% | 0.5000 | Public-Pretrain + MIC-DP-FT | private | Public-Pretrain + MIC-DP-FT (15%) | MIC-DP | 0.7214 | 0.5830 | 0.1383 | 0.1056 | 0.1744 | 29 | 1 | 3.725e-09 | 3.539e-07 | A better |
| CDC-Diabetes | 15% | 0.5000 | Public + Uniform | Public-Only | Public + Uniform (pooled) (15%) | Public-Only (15%) | 0.6492 | 0.8037 | -0.1545 | -0.1881 | -0.1228 | 0 | 30 | 1.863e-09 | 2.682e-07 | B better |
| CDC-Diabetes | 15% | 0.5000 | Public + Uniform | private | Public + Uniform (pooled) (15%) | Uniform | 0.6492 | 0.6259 | 0.0232 | 0.0124 | 0.0338 | 24 | 6 | 0.0001106 | 0.0034 | A better |
| CDC-Diabetes | 5% | 0.5000 | Public-Pretrain + Uniform-FT | Public-Only | Public-Pretrain + Uniform-FT (5%) | Public-Only (5%) | 0.7412 | 0.7881 | -0.0468 | -0.0579 | -0.0355 | 2 | 28 | 1.304e-08 | 1.134e-06 | B better |
| CDC-Diabetes | 5% | 0.5000 | Public-Pretrain + Uniform-FT | private | Public-Pretrain + Uniform-FT (5%) | Uniform | 0.7412 | 0.6259 | 0.1153 | 0.0799 | 0.1539 | 28 | 2 | 2.608e-08 | 2.112e-06 | A better |
| CDC-Diabetes | 5% | 0.5000 | Public-Pretrain + MIC-DP-FT | Public-Only | Public-Pretrain + MIC-DP-FT (5%) | Public-Only (5%) | 0.7475 | 0.7881 | -0.0406 | -0.0558 | -0.0276 | 3 | 27 | 2.608e-08 | 2.112e-06 | B better |
| CDC-Diabetes | 5% | 0.5000 | Public-Pretrain + MIC-DP-FT | private | Public-Pretrain + MIC-DP-FT (5%) | MIC-DP | 0.7475 | 0.5830 | 0.1645 | 0.1184 | 0.2142 | 27 | 3 | 4.657e-08 | 3.37e-06 | A better |
| CDC-Diabetes | 5% | 0.5000 | Public + Uniform | Public-Only | Public + Uniform (pooled) (5%) | Public-Only (5%) | 0.6379 | 0.7881 | -0.1501 | -0.1858 | -0.1174 | 0 | 30 | 1.863e-09 | 2.682e-07 | B better |
| CDC-Diabetes | 5% | 0.5000 | Public + Uniform | private | Public + Uniform (pooled) (5%) | Uniform | 0.6379 | 0.6259 | 0.0120 | 0.0034 | 0.0208 | 18 | 12 | 0.0208 | 0.3753 | n.s. |
| CDC-Diabetes | 15% | 1.0000 | Public-Pretrain + Uniform-FT | Public-Only | Public-Pretrain + Uniform-FT (15%) | Public-Only (15%) | 0.7741 | 0.8037 | -0.0295 | -0.0367 | -0.0232 | 0 | 30 | 1.863e-09 | 2.682e-07 | B better |
| CDC-Diabetes | 15% | 1.0000 | Public-Pretrain + Uniform-FT | private | Public-Pretrain + Uniform-FT (15%) | Uniform | 0.7741 | 0.7117 | 0.0624 | 0.0464 | 0.0793 | 28 | 2 | 1.304e-08 | 1.134e-06 | A better |
| CDC-Diabetes | 15% | 1.0000 | Public-Pretrain + MIC-DP-FT | Public-Only | Public-Pretrain + MIC-DP-FT (15%) | Public-Only (15%) | 0.7559 | 0.8037 | -0.0478 | -0.0596 | -0.0373 | 1 | 29 | 3.725e-09 | 3.539e-07 | B better |
| CDC-Diabetes | 15% | 1.0000 | Public-Pretrain + MIC-DP-FT | private | Public-Pretrain + MIC-DP-FT (15%) | MIC-DP | 0.7559 | 0.6904 | 0.0654 | 0.0515 | 0.0798 | 29 | 1 | 3.725e-09 | 3.539e-07 | A better |
| CDC-Diabetes | 15% | 1.0000 | Public + Uniform | Public-Only | Public + Uniform (pooled) (15%) | Public-Only (15%) | 0.7295 | 0.8037 | -0.0742 | -0.0872 | -0.0622 | 0 | 30 | 1.863e-09 | 2.682e-07 | B better |
| CDC-Diabetes | 15% | 1.0000 | Public + Uniform | private | Public + Uniform (pooled) (15%) | Uniform | 0.7295 | 0.7117 | 0.0178 | 0.0091 | 0.0273 | 22 | 8 | 0.0005055 | 0.0147 | A better |
| CDC-Diabetes | 5% | 1.0000 | Public-Pretrain + Uniform-FT | Public-Only | Public-Pretrain + Uniform-FT (5%) | Public-Only (5%) | 0.7492 | 0.7881 | -0.0388 | -0.0497 | -0.0286 | 2 | 28 | 1.863e-08 | 1.565e-06 | B better |
| CDC-Diabetes | 5% | 1.0000 | Public-Pretrain + Uniform-FT | private | Public-Pretrain + Uniform-FT (5%) | Uniform | 0.7492 | 0.7117 | 0.0375 | 0.0192 | 0.0575 | 23 | 7 | 0.0005548 | 0.0150 | A better |
| CDC-Diabetes | 5% | 1.0000 | Public-Pretrain + MIC-DP-FT | Public-Only | Public-Pretrain + MIC-DP-FT (5%) | Public-Only (5%) | 0.7332 | 0.7881 | -0.0548 | -0.0743 | -0.0382 | 0 | 30 | 1.863e-09 | 2.682e-07 | B better |
| CDC-Diabetes | 5% | 1.0000 | Public-Pretrain + MIC-DP-FT | private | Public-Pretrain + MIC-DP-FT (5%) | MIC-DP | 0.7332 | 0.6904 | 0.0428 | 0.0158 | 0.0695 | 20 | 10 | 0.0050 | 0.1203 | n.s. |
| CDC-Diabetes | 5% | 1.0000 | Public + Uniform | Public-Only | Public + Uniform (pooled) (5%) | Public-Only (5%) | 0.7144 | 0.7881 | -0.0737 | -0.0949 | -0.0551 | 0 | 30 | 1.863e-09 | 2.682e-07 | B better |
| CDC-Diabetes | 5% | 1.0000 | Public + Uniform | private | Public + Uniform (pooled) (5%) | Uniform | 0.7144 | 0.7117 | 0.0027 | -0.0080 | 0.0118 | 16 | 14 | 0.4771 | 1.0000 | n.s. |
| CDC-Diabetes | 15% | 5.0000 | Public-Pretrain + Uniform-FT | Public-Only | Public-Pretrain + Uniform-FT (15%) | Public-Only (15%) | 0.7949 | 0.8037 | -0.0088 | -0.0138 | -0.0054 | 1 | 29 | 5.588e-09 | 5.029e-07 | B better |
| CDC-Diabetes | 15% | 5.0000 | Public-Pretrain + Uniform-FT | private | Public-Pretrain + Uniform-FT (15%) | Uniform | 0.7949 | 0.7923 | 0.0026 | -0.0020 | 0.0053 | 27 | 3 | 7.994e-06 | 0.0002958 | A better |
| CDC-Diabetes | 15% | 5.0000 | Public-Pretrain + MIC-DP-FT | Public-Only | Public-Pretrain + MIC-DP-FT (15%) | Public-Only (15%) | 0.7944 | 0.8037 | -0.0092 | -0.0114 | -0.0071 | 2 | 28 | 9.313e-09 | 8.289e-07 | B better |
| CDC-Diabetes | 15% | 5.0000 | Public-Pretrain + MIC-DP-FT | private | Public-Pretrain + MIC-DP-FT (15%) | MIC-DP | 0.7944 | 0.7896 | 0.0049 | 0.0034 | 0.0064 | 29 | 1 | 4.712e-07 | 2.121e-05 | A better |
| CDC-Diabetes | 15% | 5.0000 | Public + Uniform | Public-Only | Public + Uniform (pooled) (15%) | Public-Only (15%) | 0.7940 | 0.8037 | -0.0097 | -0.0113 | -0.0081 | 0 | 30 | 1.863e-09 | 2.682e-07 | B better |
| CDC-Diabetes | 15% | 5.0000 | Public + Uniform | private | Public + Uniform (pooled) (15%) | Uniform | 0.7940 | 0.7923 | 0.0017 | 0.0013 | 0.0021 | 27 | 3 | 2.552e-07 | 1.199e-05 | A better |
| CDC-Diabetes | 5% | 5.0000 | Public-Pretrain + Uniform-FT | Public-Only | Public-Pretrain + Uniform-FT (5%) | Public-Only (5%) | 0.7267 | 0.7881 | -0.0614 | -0.0842 | -0.0400 | 4 | 26 | 5.718e-07 | 2.459e-05 | B better |
| CDC-Diabetes | 5% | 5.0000 | Public-Pretrain + Uniform-FT | private | Public-Pretrain + Uniform-FT (5%) | Uniform | 0.7267 | 0.7923 | -0.0656 | -0.0896 | -0.0442 | 3 | 27 | 1.304e-07 | 6.78e-06 | B better |
| CDC-Diabetes | 5% | 5.0000 | Public-Pretrain + MIC-DP-FT | Public-Only | Public-Pretrain + MIC-DP-FT (5%) | Public-Only (5%) | 0.7032 | 0.7881 | -0.0849 | -0.1267 | -0.0476 | 4 | 26 | 7.687e-06 | 0.0002921 | B better |
| CDC-Diabetes | 5% | 5.0000 | Public-Pretrain + MIC-DP-FT | private | Public-Pretrain + MIC-DP-FT (5%) | MIC-DP | 0.7032 | 0.7896 | -0.0864 | -0.1273 | -0.0497 | 3 | 27 | 1.304e-07 | 6.78e-06 | B better |
| CDC-Diabetes | 5% | 5.0000 | Public + Uniform | Public-Only | Public + Uniform (pooled) (5%) | Public-Only (5%) | 0.7929 | 0.7881 | 0.0049 | 0.0030 | 0.0066 | 24 | 6 | 2.689e-05 | 0.000941 | A better |
| CDC-Diabetes | 5% | 5.0000 | Public + Uniform | private | Public + Uniform (pooled) (5%) | Uniform | 0.7929 | 0.7923 | 0.0005748 | 0.0001127 | 0.0010 | 19 | 11 | 0.0262 | 0.4459 | n.s. |
| CDC-Diabetes | 15% | 10.0000 | Public-Pretrain + Uniform-FT | Public-Only | Public-Pretrain + Uniform-FT (15%) | Public-Only (15%) | 0.8009 | 0.8037 | -0.0027 | -0.0034 | -0.0020 | 2 | 28 | 1.024e-07 | 5.478e-06 | B better |
| CDC-Diabetes | 15% | 10.0000 | Public-Pretrain + Uniform-FT | private | Public-Pretrain + Uniform-FT (15%) | Uniform | 0.8009 | 0.7976 | 0.0034 | 0.0029 | 0.0039 | 30 | 0 | 1.863e-09 | 2.682e-07 | A better |
| CDC-Diabetes | 15% | 10.0000 | Public-Pretrain + MIC-DP-FT | Public-Only | Public-Pretrain + MIC-DP-FT (15%) | Public-Only (15%) | 0.7938 | 0.8037 | -0.0099 | -0.0165 | -0.0045 | 8 | 22 | 0.0004184 | 0.0126 | B better |
| CDC-Diabetes | 15% | 10.0000 | Public-Pretrain + MIC-DP-FT | private | Public-Pretrain + MIC-DP-FT (15%) | MIC-DP | 0.7938 | 0.7983 | -0.0045 | -0.0112 | 0.0006328 | 18 | 12 | 0.9515 | 1.0000 | n.s. |
| CDC-Diabetes | 15% | 10.0000 | Public + Uniform | Public-Only | Public + Uniform (pooled) (15%) | Public-Only (15%) | 0.7984 | 0.8037 | -0.0053 | -0.0061 | -0.0044 | 0 | 30 | 1.863e-09 | 2.682e-07 | B better |
| CDC-Diabetes | 15% | 10.0000 | Public + Uniform | private | Public + Uniform (pooled) (15%) | Uniform | 0.7984 | 0.7976 | 0.0008581 | 0.000576 | 0.0012 | 25 | 5 | 2.762e-06 | 0.0001133 | A better |
| CDC-Diabetes | 5% | 10.0000 | Public-Pretrain + Uniform-FT | Public-Only | Public-Pretrain + Uniform-FT (5%) | Public-Only (5%) | 0.7443 | 0.7881 | -0.0437 | -0.0640 | -0.0265 | 7 | 23 | 2.367e-05 | 0.000852 | B better |
| CDC-Diabetes | 5% | 10.0000 | Public-Pretrain + Uniform-FT | private | Public-Pretrain + Uniform-FT (5%) | Uniform | 0.7443 | 0.7976 | -0.0533 | -0.0726 | -0.0359 | 2 | 28 | 9.313e-09 | 8.289e-07 | B better |
| CDC-Diabetes | 5% | 10.0000 | Public-Pretrain + MIC-DP-FT | Public-Only | Public-Pretrain + MIC-DP-FT (5%) | Public-Only (5%) | 0.7145 | 0.7881 | -0.0735 | -0.1082 | -0.0416 | 7 | 23 | 8.174e-05 | 0.0026 | B better |
| CDC-Diabetes | 5% | 10.0000 | Public-Pretrain + MIC-DP-FT | private | Public-Pretrain + MIC-DP-FT (5%) | MIC-DP | 0.7145 | 0.7983 | -0.0838 | -0.1192 | -0.0518 | 3 | 27 | 8.009e-08 | 4.485e-06 | B better |
| CDC-Diabetes | 5% | 10.0000 | Public + Uniform | Public-Only | Public + Uniform (pooled) (5%) | Public-Only (5%) | 0.7983 | 0.7881 | 0.0102 | 0.0093 | 0.0111 | 30 | 0 | 1.863e-09 | 2.682e-07 | A better |
| CDC-Diabetes | 5% | 10.0000 | Public + Uniform | private | Public + Uniform (pooled) (5%) | Uniform | 0.7983 | 0.7976 | 0.0006955 | 0.0004824 | 0.000928 | 28 | 2 | 9.984e-07 | 4.193e-05 | A better |

## Family F8 (all tests)

| dataset | size | eps | A | B | mean_a | mean_b | diff | ci_lo | ci_hi | n_a_wins | n_b_wins | p_raw | p_holm | outcome |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Adult | 15% | 0.1000 | MIC-Public (15%) | MIC-DP | 0.5466 | 0.5468 | -0.0001945 | -0.0804 | 0.0766 | 16 | 14 | 0.9515 | 1.0000 | n.s. |
| Adult | 15% | 0.3000 | MIC-Public (15%) | MIC-DP | 0.6719 | 0.5722 | 0.0997 | 0.0360 | 0.1630 | 23 | 7 | 0.0022 | 0.0503 | n.s. |
| Adult | 15% | 0.5000 | MIC-Public (15%) | MIC-DP | 0.6799 | 0.6117 | 0.0681 | -0.0044 | 0.1456 | 17 | 13 | 0.1840 | 1.0000 | n.s. |
| Adult | 15% | 1.0000 | MIC-Public (15%) | MIC-DP | 0.7796 | 0.7048 | 0.0748 | 0.0419 | 0.1093 | 23 | 7 | 0.0001233 | 0.0033 | A better |
| Adult | 15% | 5.0000 | MIC-Public (15%) | MIC-DP | 0.8202 | 0.8196 | 0.0006338 | -0.0061 | 0.0073 | 18 | 12 | 0.7000 | 1.0000 | n.s. |
| Adult | 15% | 10.0000 | MIC-Public (15%) | MIC-DP | 0.8275 | 0.8227 | 0.0048 | 0.0016 | 0.0080 | 19 | 11 | 0.0113 | 0.2261 | n.s. |
| Adult | 5% | 0.1000 | MIC-Public (5%) | MIC-DP | 0.4924 | 0.5468 | -0.0544 | -0.1311 | 0.0201 | 14 | 16 | 0.3285 | 1.0000 | n.s. |
| Adult | 5% | 0.3000 | MIC-Public (5%) | MIC-DP | 0.6123 | 0.5722 | 0.0401 | -0.0280 | 0.1049 | 17 | 13 | 0.2449 | 1.0000 | n.s. |
| Adult | 5% | 0.5000 | MIC-Public (5%) | MIC-DP | 0.6602 | 0.6117 | 0.0484 | -0.0140 | 0.1127 | 18 | 12 | 0.1772 | 1.0000 | n.s. |
| Adult | 5% | 1.0000 | MIC-Public (5%) | MIC-DP | 0.7519 | 0.7048 | 0.0471 | 9.896e-05 | 0.0898 | 21 | 9 | 0.0040 | 0.0847 | n.s. |
| Adult | 5% | 5.0000 | MIC-Public (5%) | MIC-DP | 0.8242 | 0.8196 | 0.0047 | -0.0017 | 0.0112 | 16 | 14 | 0.2534 | 1.0000 | n.s. |
| Adult | 5% | 10.0000 | MIC-Public (5%) | MIC-DP | 0.8253 | 0.8227 | 0.0026 | -0.0009231 | 0.0061 | 17 | 13 | 0.1981 | 1.0000 | n.s. |
| Adult | 1% | 0.1000 | MIC-Public (1%) | MIC-DP | 0.5647 | 0.5468 | 0.0179 | -0.0630 | 0.0998 | 17 | 13 | 0.6263 | 1.0000 | n.s. |
| Adult | 1% | 0.3000 | MIC-Public (1%) | MIC-DP | 0.6741 | 0.5722 | 0.1018 | 0.0388 | 0.1658 | 18 | 12 | 0.0145 | 0.2471 | n.s. |
| Adult | 1% | 0.5000 | MIC-Public (1%) | MIC-DP | 0.6937 | 0.6117 | 0.0820 | 0.0022 | 0.1559 | 23 | 7 | 0.0113 | 0.2261 | n.s. |
| Adult | 1% | 1.0000 | MIC-Public (1%) | MIC-DP | 0.7721 | 0.7048 | 0.0673 | 0.0289 | 0.1088 | 24 | 6 | 0.0010 | 0.0260 | A better |
| Adult | 1% | 5.0000 | MIC-Public (1%) | MIC-DP | 0.8167 | 0.8196 | -0.0028 | -0.0100 | 0.0045 | 14 | 16 | 0.4161 | 1.0000 | n.s. |
| Adult | 1% | 10.0000 | MIC-Public (1%) | MIC-DP | 0.8195 | 0.8227 | -0.0032 | -0.0071 | 0.0002603 | 12 | 18 | 0.2054 | 1.0000 | n.s. |
| CDC-Diabetes | 15% | 0.1000 | MIC-Public (15%) | MIC-DP | 0.5460 | 0.4976 | 0.0484 | -0.0221 | 0.1153 | 18 | 12 | 0.2129 | 1.0000 | n.s. |
| CDC-Diabetes | 15% | 0.3000 | MIC-Public (15%) | MIC-DP | 0.6979 | 0.6069 | 0.0911 | 0.0353 | 0.1443 | 25 | 5 | 0.0006666 | 0.0173 | A better |
| CDC-Diabetes | 15% | 0.5000 | MIC-Public (15%) | MIC-DP | 0.7157 | 0.5830 | 0.1327 | 0.0678 | 0.1943 | 27 | 3 | 6.287e-05 | 0.0018 | A better |
| CDC-Diabetes | 15% | 1.0000 | MIC-Public (15%) | MIC-DP | 0.7925 | 0.6904 | 0.1020 | 0.0840 | 0.1209 | 30 | 0 | 1.863e-09 | 6.706e-08 | A better |
| CDC-Diabetes | 15% | 5.0000 | MIC-Public (15%) | MIC-DP | 0.7993 | 0.7896 | 0.0097 | 0.0070 | 0.0126 | 28 | 2 | 6.147e-08 | 2.028e-06 | A better |
| CDC-Diabetes | 15% | 10.0000 | MIC-Public (15%) | MIC-DP | 0.8003 | 0.7983 | 0.0020 | 0.000293 | 0.0036 | 20 | 10 | 0.0364 | 0.5465 | n.s. |
| CDC-Diabetes | 5% | 0.1000 | MIC-Public (5%) | MIC-DP | 0.5891 | 0.4976 | 0.0915 | 0.0150 | 0.1627 | 21 | 9 | 0.0185 | 0.2965 | n.s. |
| CDC-Diabetes | 5% | 0.3000 | MIC-Public (5%) | MIC-DP | 0.7067 | 0.6069 | 0.0998 | 0.0506 | 0.1515 | 24 | 6 | 0.0013 | 0.0322 | A better |
| CDC-Diabetes | 5% | 0.5000 | MIC-Public (5%) | MIC-DP | 0.7319 | 0.5830 | 0.1489 | 0.0848 | 0.2126 | 25 | 5 | 7.911e-05 | 0.0022 | A better |
| CDC-Diabetes | 5% | 1.0000 | MIC-Public (5%) | MIC-DP | 0.7866 | 0.6904 | 0.0961 | 0.0736 | 0.1186 | 28 | 2 | 6.147e-08 | 2.028e-06 | A better |
| CDC-Diabetes | 5% | 5.0000 | MIC-Public (5%) | MIC-DP | 0.7983 | 0.7896 | 0.0087 | 0.0060 | 0.0116 | 28 | 2 | 9.984e-07 | 3.095e-05 | A better |
| CDC-Diabetes | 5% | 10.0000 | MIC-Public (5%) | MIC-DP | 0.7988 | 0.7983 | 0.0005165 | -0.0011 | 0.0021 | 19 | 11 | 0.4280 | 1.0000 | n.s. |
| CDC-Diabetes | 1% | 0.1000 | MIC-Public (1%) | MIC-DP | 0.6099 | 0.4976 | 0.1123 | 0.0437 | 0.1786 | 22 | 8 | 0.0035 | 0.0765 | n.s. |
| CDC-Diabetes | 1% | 0.3000 | MIC-Public (1%) | MIC-DP | 0.6646 | 0.6069 | 0.0578 | -0.0105 | 0.1233 | 21 | 9 | 0.0364 | 0.5465 | n.s. |
| CDC-Diabetes | 1% | 0.5000 | MIC-Public (1%) | MIC-DP | 0.7510 | 0.5830 | 0.1680 | 0.1118 | 0.2237 | 28 | 2 | 1.061e-05 | 0.0003183 | A better |
| CDC-Diabetes | 1% | 1.0000 | MIC-Public (1%) | MIC-DP | 0.7891 | 0.6904 | 0.0987 | 0.0786 | 0.1194 | 30 | 0 | 1.863e-09 | 6.706e-08 | A better |
| CDC-Diabetes | 1% | 5.0000 | MIC-Public (1%) | MIC-DP | 0.7996 | 0.7896 | 0.0101 | 0.0074 | 0.0127 | 28 | 2 | 2.608e-08 | 8.866e-07 | A better |
| CDC-Diabetes | 1% | 10.0000 | MIC-Public (1%) | MIC-DP | 0.8003 | 0.7983 | 0.0020 | 0.0004962 | 0.0035 | 20 | 10 | 0.0120 | 0.2261 | n.s. |

## Runtime

python -m src.public: 42.8 min wall clock (2571 s); units run this time: 60; sum of unit times 85.3 min (2 workers)


## Deviations

No method, setting, split, seed, hyperparameter or reporting rule changed after the pre-run commit a11f76a;
config.yaml, public_reporting_rules.md and src/public.py are as committed.

D1 (presentation only). After the run, `src/public_report.py` was extended with the sections "Repository"
and "Notes on the outputs" of this file (Public-Only has zero variance across trials; counts of runs with
constant predictions; a diagnostic table of the fine-tuned models), and `diagnostics/ft_leaf_values.py`
was added to produce that diagnostic. The dash pattern of the 1% Public-Only line in `public_ch4.pdf` was
changed so that it is not confused with the dotted Non-private line. Reason: the fine-tuned primary models returned constant predictions
on Adult, and the table shows how that arises numerically. No reported number of a method changed.
