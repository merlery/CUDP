# Chapter 7 (PrivAgent): pre-registered hypotheses and decision rules

Written before the main run and committed to git together with `config.yaml`, `src/` and `tests/`. The
commit hash and date are recorded in `results/ch7/summary.md`. After that commit nothing in the
hypotheses, decision rules, methods or settings changes; any later change is listed under "Deviations"
in `summary.md`, and every result it affects is labeled exploratory.

No human participants are involved. All users are simulated. The simulated users answer according to
fixed true weights with logistic response noise; they have no anchoring in the dialogue, no present bias
and no context-dependence.

## Common definitions

- Primary population: true weights w* ~ Dirichlet(1, 1, 1), 1,000 users per replication, 20
  replications (20,000 users).
- Primary decision settings: the Adult and CDC-Diabetes decision matrices of Chapter 6 (Table 6.1) with
  the linear LP map.
- Primary user temperature tau = 0.1; anchored slider with lambda = 0.4 and report noise sd 0.1.
- Budget agreement of a method: share of users for whom the weights of the method lead ULDP (TOPSIS,
  Chapter 6) to the same eps* as the true weights. An undefined selection counts as disagreement.
- Privacy share pi = w_LP / (w_LP + w_LA) (0.5 when w_LP + w_LA = 0); privacy-share error |pi_method - pi_true|.
- Cosine agreement: cosine similarity between the weights of the method and the true weights.
- Point estimates pool all users of all replications. Intervals: 95% percentile cluster bootstrap over
  replications (10,000 resamples).
- Tests are pooled over the 20 replications (20,000 paired users): McNemar on paired per-user agreement
  (exact binomial if the discordant count is below 25, otherwise chi-square with continuity correction),
  paired two-sided Wilcoxon signed-rank on per-user differences. Holm correction within each family;
  significant means Holm-adjusted p < 0.05. A test counts "in the stated direction" if it is significant
  and the pooled mean difference A - B has the stated sign.
- Decision rule for every hypothesis: pass if every test of its family is significant in the stated
  direction; partial if some are; fail if none are. Effect sizes (difference with its 95% interval) are
  reported for every test, because 20,000 users make small differences significant.

## H1 (family G1)

On both datasets (Adult, CDC-Diabetes; linear LP), PrivAgent-6 has significantly higher budget agreement
than the expert default (0.4, 0.4, 0.2). Two McNemar tests, A = PrivAgent-6, B = expert default,
direction A > B.

## H2 (family G2)

On both datasets, PrivAgent-LPLA has significantly higher budget agreement than PrivAgent-6. Two McNemar
tests, A = PrivAgent-LPLA, B = PrivAgent-6, direction A > B.

## H3 (family G3)

For the same prompts and answers (PrivAgent-6 prompts), Bayesian inference has a significantly smaller
mean absolute privacy-share error than vote inference. One Wilcoxon test, A = PrivAgent-6-Bayes,
B = PrivAgent-6, direction A < B.

## H4 (family G4)

On both datasets, PrivAgent-6 has significantly higher budget agreement than the anchored slider
(lambda = 0.4). Two McNemar tests, A = PrivAgent-6, B = anchored slider, direction A > B. This result
depends on the assumed anchoring model (anchor at the slider default (1/3, 1/3, 1/3) with strength
lambda, Gaussian report noise with sd 0.1); the lambda sensitivity analysis is reported with it.

## H5 (family G5; replication of the earlier result)

The mean cosine agreement of PrivAgent-6 is lower than that of the expert default. One Wilcoxon test,
A = PrivAgent-6, B = expert default, direction A < B. This hypothesis checks whether the earlier finding
reproduces; it is not a claim in favor of PrivAgent.

## Descriptive analyses (no test)

Profile populations (w* ~ Dirichlet(20 c) around the four Chapter 6 profile centers, 250 users per profile
per replication); the exponential LP map; the rounds analysis (PrivAgent-6 with 1 to 6 prompts); the
sensitivity to the user temperature tau in {0.05, 0.1, 0.2, 0.4} (the analyst's tau of the Bayesian
inference stays 0.1); the sensitivity to the anchoring strength lambda in {0, 0.2, 0.4, 0.6}; repair rate;
closeness loss; share of undefined selections.
