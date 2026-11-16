# Central-on-Privatized baseline (Chapter 5 study)

## Decision-log entry and commit recorded before the run

Plan written at commit `151dbe1ed5c93b1b5d5e22efff6a665e89e8c8e4` (HEAD when D53 was written); plan, rules and `config.yaml` committed before the first run at `12f1a14b8851e661961b81ac367077e09d4d50b7`.

```
## D53 — Central-on-Privatized baseline (plan, recorded before the run)

Author task 2026-11-08 (dissertation study, separate from the S&P paper
unless promising): each client privatizes its records once (Phases 1 and 3
unchanged, same seeds, byte-identical arrays, asserted by SHA-256), uploads
them once, and the server trains the same two-hidden-layer MLP (64, 32)
centrally; Central-on-Clean is the non-private central reference.  Central
protocol (fixed here): SGD without momentum, lr 0.1 (the non-private
federated Adult model's lr, D17), batch 128, epochs = round(total federated
gradient steps / central steps per epoch) where the federated total is
sum over rounds and participants of ceil(m_i / 128) from the recorded
participation schedule; no early stopping; the test set is never read
during training; no update clipping.  Settings: Adult (occupation, 50
clients), seeds 0-4: all-features uniform release at eps in {1, 10} (the
recorded `adult_occ_iso` runs) and {50, 100, 200} (`adult_occ_v7_iso`);
private top-20 selection (`adult_occ_v7_sel_mi_k20`) and oracle top-20
(`adult_occ_v7_sel_mi_oracle_k20`) at eps in {50, 100, 200}; extension:
Bank (job) private top-10 at eps = 200 (`bank_occ_v9_sel_k10`).
Communication: federated bytes = sum over rounds of bytes_per_client x
n_participants from the per-round CSVs; central bytes = records x released
features x 4 + the Phase 1 score release (phase1_bytes_per_client x N);
both from recorded files.  Outputs in results/ch5/central/.
Commit before the run: 151dbe1ed5c93b1b5d5e22efff6a665e89e8c8e4.

Reporting rules (verbatim, fixed before the run):
1. For each setting, state whether Central-on-Privatized is above, within,
   or below one seed standard deviation of the federated result of the
   same release (standard deviation of the federated result).
2. State whether Central-on-Privatized leaves the majority rate at any
   eps <= 10 for the all-features release. (The noise law predicts that it
   does not.)
3. Report the communication ratio (federated bytes divided by central
   bytes) per setting.
4. Report the accuracy of Central-on-Clean against the federated
   non-private model.
These are reporting rules, not hypotheses.
Sanity checks (the run stops if any fails): SHA-256 equality of the
privatized arrays with the federated runs; no test record read during
training and no choice depending on test accuracy; central gradient steps
within 5 % of the federated total (both printed); re-running one federated
setting for one seed reproduces its stored accuracy.
Operational definitions and all remaining choices, fixed before the run in
results/ch5/central/config.yaml: "within one std" means
|central mean - federated mean| <= federated sample std over the 5 seeds;
"leaves the majority rate" means mean balanced accuracy exceeds 0.5 by more
than its seed std; central shuffling uses a generator seeded by
derive_seed(seed, "central"); the model is initialised after
seed_everything(seed) as in the federated run.
```

## Reporting rules (verbatim)

1. For each setting, state whether Central-on-Privatized is above, within,
   or below one seed standard deviation of the federated result of the
   same release (standard deviation of the federated result).
2. State whether Central-on-Privatized leaves the majority rate at any
   eps <= 10 for the all-features release. (The noise law predicts that it
   does not.)
3. Report the communication ratio (federated bytes divided by central
   bytes) per setting.
4. Report the accuracy of Central-on-Clean against the federated
   non-private model.

## Results

| release / budget | Federated accuracy | Central accuracy | Difference (mean [min, max]) | Federated MB | Central MB | Ratio |
|---|---|---|---|---|---|---|
| **All features, uniform shares** | | | | | | |
| eps = 1 | 0.658 ± 0.236 | 0.764 ± 0.000 | +0.106 [+0.000, +0.528] | 71.440 | 13.676 | 5.224 |
| eps = 10 | 0.764 ± 0.000 | 0.335 ± 0.219 | -0.429 [-0.527, -0.038] | 71.440 | 13.676 | 5.224 |
| eps = 50 | 0.771 ± 0.009 | 0.372 ± 0.078 | -0.399 [-0.548, -0.329] | 71.440 | 13.676 | 5.224 |
| eps = 100 | 0.770 ± 0.041 | 0.428 ± 0.140 | -0.342 [-0.534, -0.168] | 71.440 | 13.676 | 5.224 |
| eps = 200 | 0.793 ± 0.035 | 0.618 ± 0.094 | -0.175 [-0.331, -0.100] | 71.440 | 13.676 | 5.224 |
| **Private selection, k=20** | | | | | | |
| eps = 50 | 0.764 ± 0.000 | 0.777 ± 0.016 | +0.013 [+0.000, +0.034] | 27.920 | 2.647 | 10.548 |
| eps = 100 | 0.777 ± 0.012 | 0.797 ± 0.006 | +0.020 [+0.009, +0.031] | 27.920 | 2.647 | 10.548 |
| eps = 200 | 0.802 ± 0.016 | 0.814 ± 0.007 | +0.011 [-0.001, +0.031] | 27.920 | 2.647 | 10.548 |
| **Oracle selection, k=20 (non-private scores)** | | | | | | |
| eps = 50 | 0.764 ± 0.001 | 0.787 ± 0.019 | +0.023 [+0.002, +0.047] | 27.920 | 2.647 | 10.548 |
| eps = 100 | 0.779 ± 0.019 | 0.816 ± 0.003 | +0.037 [+0.007, +0.051] | 27.920 | 2.647 | 10.548 |
| eps = 200 | 0.806 ± 0.017 | 0.820 ± 0.001 | +0.014 [+0.000, +0.032] | 27.920 | 2.647 | 10.548 |
| **Extension: Bank (job), private selection, k=10** | | | | | | |
| eps = 200 | 0.895 ± 0.001 | 0.895 ± 0.000 | -0.001 [-0.002, +0.000] | 22.800 | 1.344 | 16.964 |
| **Non-private references (Adult)** | | | | | | |
| no privacy | 0.842 ± 0.001 | 0.849 ± 0.005 | +0.007 [-0.000, +0.012] | 71.440 | 13.676 | 5.224 |

Accuracy is the final test accuracy on the clean test set, mean ± sample std over seeds 0-4; Difference is central minus federated per seed. Bytes are totals over the whole run (federated: all clients, all rounds; central: the one-time upload of privatized records plus the score release).

Majority-class rate of the test set: Adult 0.764, Bank 0.887.

## Answers to the reporting items

**Rule 1 (above / within / below one federated seed std):**

- All features, uniform shares, eps = 1: central 0.764, federated 0.658 ± 0.236 -> **within**
- All features, uniform shares, eps = 10: central 0.335, federated 0.764 ± 0.000 -> **below**
- All features, uniform shares, eps = 50: central 0.372, federated 0.771 ± 0.009 -> **below**
- All features, uniform shares, eps = 100: central 0.428, federated 0.770 ± 0.041 -> **below**
- All features, uniform shares, eps = 200: central 0.618, federated 0.793 ± 0.035 -> **below**
- Private selection, k=20, eps = 50: central 0.777, federated 0.764 ± 0.000 -> **above**
- Private selection, k=20, eps = 100: central 0.797, federated 0.777 ± 0.012 -> **above**
- Private selection, k=20, eps = 200: central 0.814, federated 0.802 ± 0.016 -> **within**
- Oracle selection, k=20 (non-private scores), eps = 50: central 0.787, federated 0.764 ± 0.001 -> **above**
- Oracle selection, k=20 (non-private scores), eps = 100: central 0.816, federated 0.779 ± 0.019 -> **above**
- Oracle selection, k=20 (non-private scores), eps = 200: central 0.820, federated 0.806 ± 0.017 -> **within**
- Extension: Bank (job), private selection, k=10, eps = 200: central 0.895, federated 0.895 ± 0.001 -> **within**

**Rule 2 (does Central-on-Privatized leave the majority rate at eps <= 10, all features):** no -- eps = 1: mean balanced accuracy 0.500 (std 0.000), accuracy 0.764 vs majority rate 0.764; eps = 10: mean balanced accuracy 0.511 (std 0.024), accuracy 0.335 vs majority rate 0.764

**Rule 3 (communication ratio, federated bytes / central bytes):**

- All features, uniform shares, eps = 1: 71.440 MB / 13.676 MB = 5.224
- All features, uniform shares, eps = 10: 71.440 MB / 13.676 MB = 5.224
- All features, uniform shares, eps = 50: 71.440 MB / 13.676 MB = 5.224
- All features, uniform shares, eps = 100: 71.440 MB / 13.676 MB = 5.224
- All features, uniform shares, eps = 200: 71.440 MB / 13.676 MB = 5.224
- Private selection, k=20, eps = 50: 27.920 MB / 2.647 MB = 10.548
- Private selection, k=20, eps = 100: 27.920 MB / 2.647 MB = 10.548
- Private selection, k=20, eps = 200: 27.920 MB / 2.647 MB = 10.548
- Oracle selection, k=20 (non-private scores), eps = 50: 27.920 MB / 2.647 MB = 10.548
- Oracle selection, k=20 (non-private scores), eps = 100: 27.920 MB / 2.647 MB = 10.548
- Oracle selection, k=20 (non-private scores), eps = 200: 27.920 MB / 2.647 MB = 10.548
- Extension: Bank (job), private selection, k=10, eps = 200: 22.800 MB / 1.344 MB = 16.964

**Rule 4 (Central-on-Clean vs the federated non-private model, Adult):** central 0.849 ± 0.005, federated 0.842 ± 0.001; paired difference +0.007 [-0.000, +0.012].

## Sanity checks

1. SHA-256 of the privatized arrays: identical to the federated run for every client in all 60 Central-on-Privatized runs (`sha256_match` in central_runs.csv; a mismatch would have stopped the run).
2. The test set is built only after central training finishes; the training call receives only the concatenated client arrays; no choice depends on test accuracy (all choices fixed in config.yaml before the run).
3. Gradient steps: largest relative difference central vs federated 1.13 % (tolerance 5 %); per run in central_runs.csv (federated_steps, central_steps).
4. Re-run of `adult_occ_v7_sel_mi_k20_eps200_s0_544ce0ea`: stored accuracy 0.814938, re-run 0.814938 -> reproduced exactly.

## Runtime

- 65 central runs; total central training time 334 s (mean 5.1 s per run).
- Wall-clock from the first to the last central run: 5.8 min (including client privatization).
- Server-side training time and federated wall-clock per setting: results/ch5/central/central_settings.csv (cen_train_s, fed_duration_s).

## Deviations

1. `config.yaml`, committed before the run, is not machine-parseable YAML (unquoted colons inside two values). It was not edited after the run; `config_used.yaml` is its verbatim text followed by the resolved per-run values (epochs, steps, learning rate) as valid YAML.
2. `tab_central.tex` is a bare booktabs `tabular` (header row and data rows, no float and no caption), which is how "rows only" was read.

Note (not a deviation): as specified, every federated reference is the recorded run of the same release; for the all-features release these are the main-sweep runs `adult_occ_iso` at eps 1 and 10 (no update clipping) and the selection-sweep runs `adult_occ_v7_iso` at eps 50-200 (update clipping C = 0.2).
