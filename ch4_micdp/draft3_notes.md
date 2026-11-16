# Draft 3 follow-up runs: notes fixed before each part (committed before any part ran)

Scope: Adult and CDC-Diabetes; eps in {0.1, 0.3, 0.5, 1, 5, 10}; delta = 1e-5; 30 paired trials (seed 1000 + t);
primary model HistGradientBoostingClassifier(max_iter=200, early_stopping=False, random_state=seed). No stored
result, split, seed or hyperparameter is changed. All settings: config.yaml, section draft3.

## Part A. Analyses of stored results

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

## Part B. Noise-aware prediction, with and without

Same refits as A.2 (Uniform and MIC-DP, six budgets, 30 trials). Reported: ROC-AUC mean and std with direct and
with noise-aware prediction, the paired difference (noise-aware minus direct) with its 95% bootstrap interval,
the paired two-sided Wilcoxon p-value with Holm over the 24 tests, and the number of trials with a constant
prediction under each.

Claim to check, fixed before the run: "noise-aware prediction improves ROC-AUC for Uniform at every budget on
both datasets". It holds if and only if all 12 Uniform tests are significant (Holm-adjusted p < 0.05) with a
positive mean difference.

## Part C. Fine-tuning, repaired

Repair, chosen and recorded before any rerun: the raw score of the pretrained model on the privatized rows is
clipped to [-3, 3] before the continued boosting stage (it is the starting score of trees 101-200), and
l2_regularization = 1.0 for the continued trees (101-200). The pretrained trees are unchanged. The repaired
model predicts clip(F_pre(x), -3, 3) + F_cont(x); its noise-aware prediction is the expectation of that function
over the release noise (config.yaml, draft3.part_c). Rerun: Public-Pretrain + Uniform-FT and Public-Pretrain +
MIC-DP-FT, 15% and 5% public splits, six budgets, 30 trials, on Adult and on CDC-Diabetes (the latter as a check).
Reported: ROC-AUC, number of constant predictions, largest leaf value, and F7r: against Public-Only at the same
size and against the private parent (Uniform, MIC-DP), Holm within the 48 new tests of each dataset. The stored
(unrepaired) results and their F7 outcomes (Holm over 144) are printed next to the repaired ones.

## Part D. Attribute-sampling baseline

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

## Amendment before the main run (Part C prediction only)

Recorded after a smoke unit (Adult, trial 0, written to results/ch4/draft3_smoke/, not part of the results) and
before the main run. The noise-aware prediction of the repaired model fixed in commit 8b2e018 estimated
E[clip(F_pre(x + eta))] with a Monte-Carlo term of 256 draws per test record. On the smoke unit (Uniform release,
eps = 0.3, 15% split) the spread of the exact E[F_pre(x + eta)] across test records was a standard deviation of
0.012 in logit, far below the Monte-Carlo error per record, so the estimate was dominated by simulation noise
(ROC-AUC 0.503 with it, 0.833 with the deterministic form below). The noise-aware prediction is therefore
clip(E[F_pre(x + eta)], -3, 3) + E[F_cont(x + eta)] with both expectations exact. Nothing else changed: the
repair (clipped start, l2_regularization = 1.0 for the continued trees), the direct prediction
clip(F_pre(x), -3, 3) + F_cont(x), and every other part.
