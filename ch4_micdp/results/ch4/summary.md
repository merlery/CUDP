# Chapter 4 (MIC-DP) experiment: summary

Generated 2026-10-01 17:37 by `python -m src.report`. 30 paired trials, eps in [0.1, 0.3, 0.5, 1.0, 5.0, 10.0], delta = 1e-05. Primary model: HistGradientBoosting(max_iter=200, early_stopping=False); secondary: LogisticRegression / Ridge. Primary metric: ROC-AUC (classification) or MAE (regression). Utility U = AUC or -MAE (higher is better). "Significant" means a paired two-sided Wilcoxon test with Holm-adjusted p < 0.05 within its family.

## Datasets

| name | task | n_public | n_train | n_test | d | target | max_MIC_public | mean_MIC_public |
|---|---|---|---|---|---|---|---|---|
| Adult | classification | 7327 | 31746 | 9769 | 12 | pos. rate 0.239 | 0.170 | 0.071 |
| CDC-Diabetes | classification | 9000 | 39000 | 12000 | 21 | pos. rate 0.139 | 0.062 | 0.017 |
| CalHousing | regression | 3096 | 13416 | 4128 | 8 | std 1.154 | 0.278 | 0.101 |
| Appliances | regression | 2961 | 12827 | 3947 | 27 | std 102.522 | 0.084 | 0.050 |
| Synthetic-lin | regression | 3000 | 13000 | 4000 | 20 | std 1.416 | 0.069 | 0.030 |
| Synthetic-nl-low | regression | 3000 | 13000 | 4000 | 20 | std 1.120 | 0.042 | 0.016 |
| Synthetic-nl-mid | regression | 3000 | 13000 | 4000 | 20 | std 1.418 | 0.094 | 0.026 |
| Synthetic-nl-high | regression | 3000 | 13000 | 4000 | 20 | std 2.242 | 0.158 | 0.038 |

MIC statistics are FG-MIC (fixed-grid MIC, src/fgmic.py) on the full public split. CDC-Diabetes was subsampled from 253,680 to 60,000 rows before splitting.

## Hypotheses (pre-registered decision rules in `config.yaml`)

### H1 - MIC-DP beats Uniform at eps <= 1 on >= 3 of 4 real datasets: **PARTIAL**

Datasets where MIC-DP is significantly better than Uniform at every eps in [0.1, 0.3, 1.0]: 0 of 4 (Adult: False, CDC-Diabetes: False, CalHousing: False, Appliances: False). Across all 24 (dataset, eps) cells: 4 significant wins, 0 significant losses vs Uniform.

| dataset | eps | U MIC-DP | U Uniform | diff | CI lo | CI hi | p_holm | outcome |
|---|---|---|---|---|---|---|---|---|
| Adult | 0.1 | 0.5468 | 0.5315 | 0.0153 | -0.0352 | 0.0658 | 1 | n.s. |
| Adult | 0.3 | 0.5722 | 0.6056 | -0.0334 | -0.0865 | 0.0237 | 1 | n.s. |
| Adult | 0.5 | 0.6117 | 0.6679 | -0.0561 | -0.1167 | 0.0033 | 1 | n.s. |
| Adult | 1.0 | 0.7048 | 0.7083 | -0.0035 | -0.0488 | 0.0406 | 1 | n.s. |
| Adult | 5.0 | 0.8196 | 0.7985 | 0.0210 | 0.0142 | 0.0278 | 0.000656 | A better |
| Adult | 10.0 | 0.8227 | 0.8010 | 0.0217 | 0.0188 | 0.0246 | 1.79e-07 | A better |
| CDC-Diabetes | 0.1 | 0.4976 | 0.5104 | -0.0127 | -0.0634 | 0.0379 | 1 | n.s. |
| CDC-Diabetes | 0.3 | 0.6069 | 0.5827 | 0.0242 | -0.0237 | 0.0693 | 1 | n.s. |
| CDC-Diabetes | 0.5 | 0.5830 | 0.6259 | -0.0429 | -0.0949 | 0.0103 | 1 | n.s. |
| CDC-Diabetes | 1.0 | 0.6904 | 0.7117 | -0.0213 | -0.0473 | 0.0066 | 1 | n.s. |
| CDC-Diabetes | 5.0 | 0.7896 | 0.7923 | -0.0028 | -0.0055 | -0.0002 | 1 | n.s. |
| CDC-Diabetes | 10.0 | 0.7983 | 0.7976 | 0.0007 | -0.0009 | 0.0024 | 1 | n.s. |
| CalHousing | 0.1 | -0.9252 | -0.9251 | -0.0001 | -0.0003 | 0.0001 | 1 | n.s. |
| CalHousing | 0.3 | -0.9252 | -0.9251 | -0.0001 | -0.0003 | 0.0000 | 1 | n.s. |
| CalHousing | 0.5 | -0.9252 | -0.9251 | -0.0001 | -0.0003 | 0.0001 | 1 | n.s. |
| CalHousing | 1.0 | -0.9250 | -0.9250 | 0.0001 | -0.0001 | 0.0003 | 1 | n.s. |
| CalHousing | 5.0 | -0.9217 | -0.9233 | 0.0016 | 0.0012 | 0.0019 | 3.96e-06 | A better |
| CalHousing | 10.0 | -0.9147 | -0.9197 | 0.0050 | 0.0046 | 0.0055 | 1.79e-07 | A better |
| Appliances | 0.1 | -61.7815 | -61.7992 | 0.0177 | -0.0328 | 0.0739 | 1 | n.s. |
| Appliances | 0.3 | -61.7915 | -61.7732 | -0.0183 | -0.0828 | 0.0441 | 1 | n.s. |
| Appliances | 0.5 | -61.7509 | -61.8022 | 0.0512 | 0.0057 | 0.0994 | 1 | n.s. |
| Appliances | 1.0 | -61.8099 | -61.7752 | -0.0347 | -0.0913 | 0.0212 | 1 | n.s. |
| Appliances | 5.0 | -61.7614 | -61.7483 | -0.0131 | -0.0634 | 0.0347 | 1 | n.s. |
| Appliances | 10.0 | -61.7152 | -61.6951 | -0.0201 | -0.0662 | 0.0265 | 1 | n.s. |

### H2 - MIC-DP beats Pearson-Alloc on nonlinear data, not on linear: **PARTIAL**

Synthetic-nl-mid: 0 of 6 eps significantly better, 0 significantly worse (needed >= 4 wins). Synthetic-lin: 0 of 6 eps with a significant difference (needed 0).

| dataset | eps | diff U (MIC-DP - Pearson-Alloc) | ci_lo | ci_hi | p_holm | outcome |
|---|---|---|---|---|---|---|
| Synthetic-lin | 0.1 | 0.0000 | -0.0000 | 0.0000 | 1 | n.s. |
| Synthetic-lin | 0.3 | -0.0000 | -0.0000 | 0.0000 | 1 | n.s. |
| Synthetic-lin | 0.5 | 0.0000 | -0.0000 | 0.0000 | 1 | n.s. |
| Synthetic-lin | 1.0 | -0.0000 | -0.0000 | 0.0000 | 1 | n.s. |
| Synthetic-lin | 5.0 | 0.0001 | -0.0001 | 0.0002 | 1 | n.s. |
| Synthetic-lin | 10.0 | 0.0002 | -0.0001 | 0.0006 | 1 | n.s. |
| Synthetic-nl-mid | 0.1 | 0.0000 | -0.0000 | 0.0001 | 1 | n.s. |
| Synthetic-nl-mid | 0.3 | -0.0000 | -0.0001 | 0.0000 | 1 | n.s. |
| Synthetic-nl-mid | 0.5 | 0.0000 | -0.0000 | 0.0001 | 1 | n.s. |
| Synthetic-nl-mid | 1.0 | 0.0000 | -0.0000 | 0.0001 | 0.635 | n.s. |
| Synthetic-nl-mid | 5.0 | 0.0001 | -0.0000 | 0.0002 | 1 | n.s. |
| Synthetic-nl-mid | 10.0 | 0.0002 | 0.0001 | 0.0004 | 0.222 | n.s. |

### H3 - MIC-DP beats Random-Alloc: **PARTIAL**

6 significant wins and 0 significant losses out of 24 (dataset, eps) cells (needed >= 13 wins, 0 losses).

Mean paired difference U(MIC-DP) - U(Random-Alloc):

| dataset | 0.1 | 0.3 | 0.5 | 1.0 | 5.0 | 10.0 |
|---|---|---|---|---|---|---|
| Adult | +0.0422 | -0.0197 | -0.0085 | +0.0508 | +0.0624 | +0.0512 |
| Appliances | +0.0210 | -0.0214 | +0.0056 | -0.0333 | -0.0086 | +0.0127 |
| CalHousing | -0.0002 | -0.0002 | +0.0000 | +0.0001 | +0.0020 | +0.0063 |
| CDC-Diabetes | -0.0431 | +0.0191 | -0.0125 | +0.0162 | +0.0281 | +0.0552 |

### H4 - gain of MIC-DP over Uniform grows with signal strength: **PARTIAL**

Slope of UR(MIC-DP) (ratio of trial means) on log(lambda) over nl-low/mid/high, 95% bootstrap CI resampling trial seeds. CI above 0 at 2 of 6 eps (needed >= 4); CI below 0 at 0.

| eps | slope | ci_lo | ci_hi | UR_syn_nl_low | UR_syn_nl_mid | UR_syn_nl_high |
|---|---|---|---|---|---|---|
| 0.1000 | 0.0000 | -0.0003 | 0.0003 | -0.0000 | 0.0002 | 0.0000 |
| 0.3000 | 0.0001 | -0.0002 | 0.0004 | -0.0003 | -0.0001 | -0.0001 |
| 0.5000 | 0.0001 | -0.0001 | 0.0004 | -0.0001 | 0.0000 | 0.0001 |
| 1.0000 | -0.0001 | -0.0004 | 0.0002 | 0.0002 | 0.0002 | 0.0001 |
| 5.0000 | 0.0005 | 0.0001 | 0.0008 | -0.0006 | -0.0003 | 0.0000 |
| 10.0000 | 0.0018 | 0.0010 | 0.0026 | -0.0005 | 0.0004 | 0.0019 |

### H5 - MIC-Public >= MIC-DP, gap largest at small eps: **PARTIAL**

(a) MIC-Public significantly better in 10 and significantly worse in 0 of 24 cells -> never worse: True. (b) Mean gap U(MIC-Public) - U(MIC-DP), eps in [0.1, 0.3, 0.5] vs [5.0, 10.0]: Adult +0.0559 vs +0.0027; CDC-Diabetes +0.0907 vs +0.0059; CalHousing +0.0003 vs +0.0040; Appliances +0.0159 vs +0.0353 -> larger at small eps for 2 of 4 datasets (needed >= 3).

### H6 - demographic override costs little utility on Adult: **PARTIAL**

Loss = AUC(MIC-DP) - AUC(MIC-DP+Override); equivalence margin 0.01 AUC. Upper 95% CI bound within the margin at 3 of 6 eps. Holm-adjusted Wilcoxon (F5): override significantly worse at 0 eps, significantly better at 0 eps.

| eps | mean_loss | ci_lo | ci_hi | within_margin |
|---|---|---|---|---|
| 0.1000 | 0.0271 | -0.0393 | 0.0975 | False |
| 0.3000 | -0.0578 | -0.1213 | 0.0056 | True |
| 0.5000 | -0.0143 | -0.0767 | 0.0450 | False |
| 1.0000 | 0.0011 | -0.0400 | 0.0410 | False |
| 5.0000 | 0.0001 | -0.0058 | 0.0058 | True |
| 10.0000 | -0.0022 | -0.0052 | 0.0009 | True |

## Utility recovered and relative improvement over Uniform (primary model)

UR = (U_m - U_Uniform) / (U_NonPrivate - U_Uniform) from trial means (0 = Uniform, 1 = non-private). Rel. = (U_m - U_Uniform) / |U_Uniform| in %, i.e. AUC gain or MAE reduction relative to Uniform.

**Adult** (non-private ROC-AUC = 0.9279)

| method | eps=0.1 | eps=0.3 | eps=0.5 | eps=1 | eps=5 | eps=10 |
|---|---|---|---|---|---|---|
| MIC-DP | +0.039 (+2.88%) | -0.104 (-5.51%) | -0.216 (-8.41%) | -0.016 (-0.49%) | +0.162 (+2.63%) | +0.171 (+2.71%) |
| MIC-Public | +0.038 (+2.84%) | +0.206 (+10.94%) | +0.046 (+1.80%) | +0.325 (+10.06%) | +0.167 (+2.71%) | +0.208 (+3.30%) |
| Pearson-Alloc | -0.044 (-3.31%) | -0.131 (-6.97%) | -0.073 (-2.83%) | +0.190 (+5.90%) | +0.171 (+2.76%) | +0.174 (+2.76%) |
| Pearson-Adj | -0.043 (-3.20%) | -0.181 (-9.61%) | -0.331 (-12.90%) | -0.174 (-5.41%) | -0.134 (-2.17%) | -0.098 (-1.55%) |
| PartialCorr-Adj | -0.017 (-1.29%) | -0.230 (-12.25%) | -0.231 (-9.01%) | -0.001 (-0.02%) | -0.155 (-2.52%) | -0.098 (-1.55%) |
| Random-Alloc | -0.068 (-5.06%) | -0.043 (-2.27%) | -0.183 (-7.14%) | -0.247 (-7.66%) | -0.320 (-5.18%) | -0.232 (-3.68%) |

**CDC-Diabetes** (non-private ROC-AUC = 0.8247)

| method | eps=0.1 | eps=0.3 | eps=0.5 | eps=1 | eps=5 | eps=10 |
|---|---|---|---|---|---|---|
| MIC-DP | -0.041 (-2.50%) | +0.100 (+4.15%) | -0.216 (-6.86%) | -0.189 (-2.99%) | -0.085 (-0.35%) | +0.027 (+0.09%) |
| MIC-Public | +0.113 (+6.98%) | +0.476 (+19.78%) | +0.452 (+14.35%) | +0.715 (+11.34%) | +0.216 (+0.88%) | +0.099 (+0.34%) |
| Pearson-Alloc | +0.052 (+3.20%) | -0.009 (-0.38%) | -0.021 (-0.65%) | +0.094 (+1.48%) | +0.226 (+0.92%) | +0.112 (+0.38%) |
| Pearson-Adj | +0.059 (+3.62%) | -0.126 (-5.23%) | -0.061 (-1.95%) | -0.298 (-4.73%) | -0.250 (-1.02%) | -0.174 (-0.59%) |
| PartialCorr-Adj | +0.055 (+3.38%) | +0.040 (+1.66%) | -0.050 (-1.57%) | -0.277 (-4.40%) | -0.074 (-0.30%) | -0.105 (-0.36%) |
| Random-Alloc | +0.097 (+5.95%) | +0.021 (+0.87%) | -0.153 (-4.86%) | -0.332 (-5.26%) | -0.953 (-3.89%) | -2.007 (-6.82%) |

**CalHousing** (non-private MAE = 0.3071)

| method | eps=0.1 | eps=0.3 | eps=0.5 | eps=1 | eps=5 | eps=10 |
|---|---|---|---|---|---|---|
| MIC-DP | -0.000 (-0.01%) | -0.000 (-0.01%) | -0.000 (-0.01%) | +0.000 (+0.01%) | +0.003 (+0.17%) | +0.008 (+0.55%) |
| MIC-Public | +0.000 (+0.01%) | +0.000 (+0.03%) | +0.000 (+0.03%) | +0.001 (+0.04%) | +0.006 (+0.38%) | +0.018 (+1.21%) |
| Pearson-Alloc | +0.000 (+0.00%) | +0.000 (+0.00%) | +0.000 (+0.00%) | +0.000 (+0.02%) | +0.003 (+0.22%) | +0.016 (+1.03%) |
| Pearson-Adj | -0.000 (-0.01%) | -0.000 (-0.01%) | +0.000 (+0.02%) | -0.000 (-0.01%) | -0.002 (-0.10%) | -0.005 (-0.32%) |
| PartialCorr-Adj | -0.000 (-0.00%) | -0.000 (-0.01%) | +0.000 (+0.01%) | -0.000 (-0.00%) | -0.001 (-0.08%) | -0.003 (-0.19%) |
| Random-Alloc | +0.000 (+0.00%) | +0.000 (+0.00%) | -0.000 (-0.01%) | -0.000 (-0.01%) | -0.001 (-0.05%) | -0.002 (-0.14%) |

**Appliances** (non-private MAE = 38.1873)

| method | eps=0.1 | eps=0.3 | eps=0.5 | eps=1 | eps=5 | eps=10 |
|---|---|---|---|---|---|---|
| MIC-DP | +0.001 (+0.03%) | -0.001 (-0.03%) | +0.002 (+0.08%) | -0.001 (-0.06%) | -0.001 (-0.02%) | -0.001 (-0.03%) |
| MIC-Public | +0.001 (+0.02%) | +0.001 (+0.03%) | +0.003 (+0.11%) | +0.000 (+0.00%) | -0.000 (-0.01%) | +0.002 (+0.07%) |
| Pearson-Alloc | -0.001 (-0.02%) | +0.001 (+0.04%) | +0.001 (+0.05%) | -0.001 (-0.05%) | -0.001 (-0.05%) | -0.003 (-0.10%) |
| Pearson-Adj | +0.000 (+0.01%) | -0.001 (-0.02%) | +0.002 (+0.08%) | +0.000 (+0.01%) | +0.000 (+0.02%) | -0.002 (-0.07%) |
| PartialCorr-Adj | +0.001 (+0.04%) | +0.000 (+0.01%) | +0.001 (+0.05%) | +0.000 (+0.01%) | -0.001 (-0.05%) | -0.002 (-0.06%) |
| Random-Alloc | -0.000 (-0.01%) | +0.000 (+0.01%) | +0.002 (+0.07%) | -0.000 (-0.00%) | -0.000 (-0.01%) | -0.001 (-0.05%) |

**Synthetic-lin** (non-private MAE = 0.8321)

| method | eps=0.1 | eps=0.3 | eps=0.5 | eps=1 | eps=5 | eps=10 |
|---|---|---|---|---|---|---|
| MIC-DP | +0.000 (+0.00%) | -0.000 (-0.00%) | -0.000 (-0.00%) | +0.000 (+0.00%) | -0.001 (-0.02%) | -0.002 (-0.04%) |
| MIC-Public | +0.000 (+0.00%) | -0.000 (-0.00%) | -0.000 (-0.00%) | +0.000 (+0.01%) | +0.003 (+0.07%) | +0.009 (+0.23%) |
| Pearson-Alloc | -0.000 (-0.00%) | +0.000 (+0.00%) | -0.000 (-0.00%) | +0.000 (+0.00%) | -0.001 (-0.02%) | -0.002 (-0.06%) |
| Pearson-Adj | +0.000 (+0.00%) | -0.000 (-0.00%) | -0.000 (-0.00%) | +0.000 (+0.00%) | -0.001 (-0.02%) | -0.003 (-0.08%) |
| PartialCorr-Adj | +0.000 (+0.00%) | +0.000 (+0.00%) | -0.000 (-0.00%) | +0.000 (+0.00%) | -0.001 (-0.02%) | -0.003 (-0.07%) |
| Random-Alloc | +0.000 (+0.00%) | +0.000 (+0.00%) | -0.000 (-0.00%) | +0.000 (+0.00%) | -0.001 (-0.03%) | -0.003 (-0.07%) |

**Synthetic-nl-low** (non-private MAE = 0.8258)

| method | eps=0.1 | eps=0.3 | eps=0.5 | eps=1 | eps=5 | eps=10 |
|---|---|---|---|---|---|---|
| MIC-DP | -0.000 (-0.00%) | -0.000 (-0.00%) | -0.000 (-0.00%) | +0.000 (+0.00%) | -0.001 (-0.00%) | -0.000 (-0.00%) |
| MIC-Public | -0.000 (-0.00%) | +0.000 (+0.00%) | +0.000 (+0.00%) | +0.001 (+0.00%) | +0.001 (+0.01%) | +0.005 (+0.04%) |
| Pearson-Alloc | -0.000 (-0.00%) | +0.000 (+0.00%) | +0.000 (+0.00%) | -0.000 (-0.00%) | -0.001 (-0.00%) | -0.001 (-0.01%) |
| Pearson-Adj | +0.000 (+0.00%) | -0.000 (-0.00%) | +0.000 (+0.00%) | +0.000 (+0.00%) | -0.001 (-0.01%) | -0.000 (-0.00%) |
| PartialCorr-Adj | -0.000 (-0.00%) | -0.000 (-0.00%) | -0.000 (-0.00%) | -0.000 (-0.00%) | -0.000 (-0.00%) | -0.001 (-0.01%) |
| Random-Alloc | -0.000 (-0.00%) | -0.000 (-0.00%) | -0.000 (-0.00%) | +0.000 (+0.00%) | -0.001 (-0.01%) | -0.000 (-0.00%) |

**Synthetic-nl-mid** (non-private MAE = 0.8339)

| method | eps=0.1 | eps=0.3 | eps=0.5 | eps=1 | eps=5 | eps=10 |
|---|---|---|---|---|---|---|
| MIC-DP | +0.000 (+0.00%) | -0.000 (-0.00%) | +0.000 (+0.00%) | +0.000 (+0.00%) | -0.000 (-0.01%) | +0.000 (+0.01%) |
| MIC-Public | +0.000 (+0.01%) | -0.000 (-0.00%) | -0.000 (-0.00%) | +0.000 (+0.00%) | +0.002 (+0.06%) | +0.007 (+0.18%) |
| Pearson-Alloc | +0.000 (+0.00%) | -0.000 (-0.00%) | -0.000 (-0.00%) | +0.000 (+0.00%) | -0.001 (-0.02%) | -0.000 (-0.01%) |
| Pearson-Adj | +0.000 (+0.00%) | -0.000 (-0.00%) | +0.000 (+0.00%) | +0.000 (+0.00%) | -0.001 (-0.01%) | -0.001 (-0.02%) |
| PartialCorr-Adj | +0.000 (+0.00%) | -0.000 (-0.00%) | +0.000 (+0.00%) | -0.000 (-0.00%) | -0.000 (-0.01%) | -0.001 (-0.02%) |
| Random-Alloc | +0.000 (+0.00%) | +0.000 (+0.00%) | -0.000 (-0.00%) | -0.000 (-0.00%) | -0.001 (-0.02%) | -0.001 (-0.03%) |

**Synthetic-nl-high** (non-private MAE = 0.8523)

| method | eps=0.1 | eps=0.3 | eps=0.5 | eps=1 | eps=5 | eps=10 |
|---|---|---|---|---|---|---|
| MIC-DP | +0.000 (+0.00%) | -0.000 (-0.00%) | +0.000 (+0.00%) | +0.000 (+0.00%) | +0.000 (+0.00%) | +0.002 (+0.10%) |
| MIC-Public | +0.000 (+0.01%) | +0.000 (+0.00%) | +0.000 (+0.01%) | +0.000 (+0.02%) | +0.002 (+0.13%) | +0.007 (+0.35%) |
| Pearson-Alloc | +0.000 (+0.00%) | -0.000 (-0.00%) | -0.000 (-0.00%) | -0.000 (-0.00%) | -0.000 (-0.03%) | -0.000 (-0.00%) |
| Pearson-Adj | +0.000 (+0.01%) | -0.000 (-0.01%) | +0.000 (+0.01%) | +0.000 (+0.00%) | -0.000 (-0.02%) | -0.001 (-0.04%) |
| PartialCorr-Adj | +0.000 (+0.00%) | -0.000 (-0.01%) | +0.000 (+0.00%) | +0.000 (+0.00%) | -0.000 (-0.02%) | -0.001 (-0.05%) |
| Random-Alloc | +0.000 (+0.00%) | -0.000 (-0.01%) | +0.000 (+0.00%) | +0.000 (+0.00%) | -0.000 (-0.02%) | -0.001 (-0.05%) |

## Significant wins per family (Holm within family)

| family | comparison | tests | A sig. better | A sig. worse | n.s. |
|---|---|---|---|---|---|
| F1 | MIC-DP vs PartialCorr-Adj, Pearson-Adj, Pearson-Alloc, Uniform | 96 | 14 | 2 | 80 |
| F2 | MIC-DP vs Pearson-Alloc | 12 | 0 | 0 | 12 |
| F3 | MIC-DP vs Random-Alloc | 24 | 6 | 0 | 18 |
| F4 | MIC-Public vs MIC-DP | 24 | 10 | 0 | 14 |
| F5 | MIC-DP+Override vs MIC-DP | 6 | 0 | 0 | 6 |
| F1_secondary | MIC-DP vs PartialCorr-Adj, Pearson-Adj, Pearson-Alloc, Uniform | 96 | 14 | 2 | 80 |

F1 broken down by baseline (24 cells each):

| baseline | MIC-DP sig. better | MIC-DP sig. worse | n.s. |
|---|---|---|---|
| Uniform | 4 | 0 | 20 |
| Pearson-Alloc | 0 | 2 | 22 |
| Pearson-Adj | 5 | 0 | 19 |
| PartialCorr-Adj | 5 | 0 | 19 |

## Across-trial standard deviation of the primary metric: MIC-DP vs Uniform

| dataset | MIC-DP std range | Uniform std range | median ratio MIC-DP/Uniform |
|---|---|---|---|
| Adult | 0.0078 - 0.1482 | 0.0046 - 0.1288 | 1.35 |
| CDC-Diabetes | 0.0044 - 0.1279 | 0.0026 - 0.1201 | 1.14 |
| CalHousing | 0.0003 - 0.0014 | 0.0003 - 0.0008 | 1.17 |
| Appliances | 0.0974 - 0.1086 | 0.0845 - 0.1233 | 0.99 |
| Synthetic-lin | 0.0000 - 0.0007 | 0.0000 - 0.0004 | 0.87 |
| Synthetic-nl-low | 0.0001 - 0.0002 | 0.0000 - 0.0002 | 1.00 |
| Synthetic-nl-mid | 0.0001 - 0.0004 | 0.0001 - 0.0003 | 1.04 |
| Synthetic-nl-high | 0.0003 - 0.0016 | 0.0002 - 0.0005 | 1.26 |

## Score-privatization noise sigma_s (rho = 0.2)

sigma_s = sqrt(d) * 3(1 + ln n)/(n ln 2) * c(rho*eps, rho*delta), with n the private-train size. For scale, the last column gives the standard deviation of the exact FG-MIC scores across features (private-train split): a ranking is only informative where sigma_s is not much larger than this.

| dataset | d | eps=0.1 | eps=0.3 | eps=0.5 | eps=1 | eps=5 | eps=10 | sd of FG-MIC |
|---|---|---|---|---|---|---|---|---|
| Adult | 12 | 0.8225 | 0.2990 | 0.1862 | 0.0977 | 0.0219 | 0.0116 | 0.0536 |
| CDC-Diabetes | 21 | 0.9018 | 0.3278 | 0.2042 | 0.1072 | 0.0240 | 0.0127 | 0.0174 |
| CalHousing | 8 | 1.4688 | 0.5340 | 0.3325 | 0.1745 | 0.0391 | 0.0207 | 0.0994 |
| Appliances | 27 | 2.8101 | 1.0216 | 0.6362 | 0.3339 | 0.0748 | 0.0396 | 0.0197 |
| Synthetic-lin | 20 | 2.3894 | 0.8687 | 0.5410 | 0.2839 | 0.0636 | 0.0337 | 0.0160 |
| Synthetic-nl-low | 20 | 2.3894 | 0.8687 | 0.5410 | 0.2839 | 0.0636 | 0.0337 | 0.0086 |
| Synthetic-nl-mid | 20 | 2.3894 | 0.8687 | 0.5410 | 0.2839 | 0.0636 | 0.0337 | 0.0227 |
| Synthetic-nl-high | 20 | 2.3894 | 0.8687 | 0.5410 | 0.2839 | 0.0636 | 0.0337 | 0.0385 |

MIC-DP on the four real datasets (B = 0: weights fall back to uniform; B = 1: exact scores):

| eps | mean shrinkage factor B | share of trials with pruning (B >= 0.5) | mean share of columns released |
|---|---|---|---|
| 0.1 | 0.005 | 0.00 | 1.00 |
| 0.3 | 0.037 | 0.00 | 1.00 |
| 0.5 | 0.079 | 0.00 | 1.00 |
| 1 | 0.163 | 0.06 | 0.98 |
| 5 | 0.575 | 0.57 | 0.80 |
| 10 | 0.713 | 0.76 | 0.71 |

## Diagnostic: constant predictions on the clean test set

Before the noise-aware prediction, HistGradientBoosting returned one constant prediction for the whole clean test set in many fits at eps <= 1 (all clean test values fall into one histogram bin). Share of primary-model fits with a single distinct test prediction in this run:

| dataset | 0.1 | 0.3 | 0.5 | 1.0 | 5.0 | 10.0 |
|---|---|---|---|---|---|---|
| Adult | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| Appliances | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| CalHousing | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| CDC-Diabetes | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| Synthetic-lin | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| Synthetic-nl-high | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| Synthetic-nl-low | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| Synthetic-nl-mid | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |

## Reading the results

1. **MIC-DP vs Uniform, primary model:** 4 significant wins and 0 significant losses in 24 (dataset, eps) cells. Wins: Adult eps=5 (+0.0210), Adult eps=10 (+0.0217), CalHousing eps=5 (+0.0016), CalHousing eps=10 (+0.0050).
2. **MIC-DP vs Uniform, linear model (F1_secondary):** 4 significant wins, 0 significant losses.
3. **Distortion of the released columns.** MIC-DP's mean |X_tilde - X| over released entries is 1.50-1.67 times Uniform's at eps = 1 and 1.23-1.43 times at eps = 10 on the real datasets. Below 1 means the released columns are less noisy than under Uniform (fewer columns share the budget).
4. **Power.** 32 of 162 pre-registered comparisons are significant after Holm correction. Where a hypothesis is PARTIAL because a 'no difference' or 'never worse' part holds, that part rests on non-significant tests and is weak evidence on its own.

## Deviations from the specification, and why

1. **Enhanced MIC-DP (v2), at the author's request after the first run.** The first run showed that the specified score release (sensitivity sqrt(d), sigma_s = 6-796 on scores in [0, 1]) leaves no usable ranking. This run changes the method; every setting below was fixed before it ran and is in `config.yaml`:
   - **Scores (Phase 1):** fixed-grid MIC (FG-MIC, `src/fgmic.py`) on the full private-train split instead of MIC on a 3,000-row subsample. Grid cut points are quantiles of the public split (32 candidate bins per feature; target rows = classes, or 2-16 public-quantile rows for regression), so the grid family does not depend on the private data and each score has sensitivity 3(1 + ln n)/(n ln 2) (proof sketch in `src/fgmic.py`, empirical check in `tests/test_fgmic.py`) instead of 1.
   - **Pearson / partial-correlation baselines** get the same treatment: one Gaussian release of the first and second moments of [x, y] (sensitivity sqrt((d+1) + (d+1)(d+2)/2)/n), correlations by post-processing. Regression targets are clipped to the public-split 0.5-99.5 percentiles for this release only.
   - **Allocation (Phase 2):** noisy scores are shrunk toward their mean (B = min(positive-part James-Stein factor, tau0^2/(tau0^2 + sigma^2)), tau0 = 0.1); weights w_j ~ (s_j / max s)^alpha with alpha = 2; columns with relative score below theta = 0.2 are not released once B >= 0.5. The specification required w_j > 0; not releasing a column is allowed by DP and keeps sum over released columns of 1/sigma_j^2 = 1/c^2. Pearson-Adj and PartialCorr-Adj keep their (1 + s)^-2 rule on shrunken scores.
   - **Random-Alloc** is MIC-DP's weight vector of the same trial, randomly permuted (same shape, signal removed), instead of softmax of U(0,1) scores.
   - **Prediction:** for every private method, the primary model predicts E[f(x + eta)] under the public release noise, computed exactly from the trees (`src/noiseaware.py`). This is post-processing; non-private and linear models are unchanged.
   - **Sweeps:** alpha in {0, 1, 2, 5, 10} is now the exponent of the relevance rule (alpha = 0 is uniform over released columns).
2. **MIC code.** `src/mic.py` (own MIC_e implementation, cross-checked against minepy in `mic_crosscheck.csv`) is kept and tested but no longer used by the pipeline; FG-MIC has its own exact dynamic programme (`tests/test_fgmic.py` checks it against brute force).
3. **Public split usage.** Beyond MIC-Public, the public split now supplies the grid cut points (feature and target quantiles) and the target range for the moment release. MIC-Public uses FG-MIC on the public split.
4. **Data source.** `ucimlrepo` (archive.ics.uci.edu) and sklearn's `fetch_california_housing` (figshare) are blocked here. The same files were downloaded from GitHub mirrors and pinned by SHA-256 in `config.yaml`: UCI Adult `adult.data` + `adult.test` (48,842 rows, byte-identical across two mirrors), the CDC BRFSS 2015 binary file behind UCI id 891 (253,680 rows, byte-identical across two mirrors), the original authors' Appliances repository (19,735 rows), and the StatLib `cal_housing.data` file that sklearn downloads. CalHousing is rebuilt with sklearn's exact transformation.
5. **Appliances temperature ranges.** T1-T9 use the DHT-22 operating range [-40, 80] C, recalled from Candanedo et al. (2017) and not re-checked from this sandbox; please confirm the citation.
6. **Adult ranges.** capital-gain uses [0, 99999] (the Census top-code); capital-loss uses the public-split fallback. Categorical codes follow `adult.names`; '?' is an extra category for workclass, occupation and native-country.
7. **MIC-DP+Override and Random-Alloc draws.** Both reuse the MIC-DP trial's privatized scores (stream id 2) and draw their own mechanism noise (ids 9 and 7).
8. **Extra outputs.** Tables for the synthetic datasets (no daggers), the `F1_secondary` family, `scores_raw.csv`, `sanity.log`, `runtime.txt`, `captions.tex`, and the columns `n_unique_pred`, `n_released`, `shrink_B`, `s_tilde` in `raw_trials.parquet`.


_Report generation time: 31 s._
