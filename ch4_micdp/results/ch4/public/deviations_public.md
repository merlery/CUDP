No method, setting, split, seed, hyperparameter or reporting rule changed after the pre-run commit a11f76a;
config.yaml, public_reporting_rules.md and src/public.py are as committed.

D1 (presentation only). After the run, `src/public_report.py` was extended with the sections "Repository"
and "Notes on the outputs" of this file (Public-Only has zero variance across trials; counts of runs with
constant predictions; a diagnostic table of the fine-tuned models), and `diagnostics/ft_leaf_values.py`
was added to produce that diagnostic. The dash pattern of the 1% Public-Only line in `public_ch4.pdf` was
changed so that it is not confused with the dotted Non-private line. Reason: the fine-tuned primary models returned constant predictions
on Adult, and the table shows how that arises numerically. No reported number of a method changed.
