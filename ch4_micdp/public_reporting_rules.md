# Public-data baselines: reporting rules (committed before the first run)

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
