# Chapter 8: User Study of the Preference-Based Front End

## Purpose

Code, de-identified data and survey instrument for the Chapter 8 user study (100 Prolific participants, IRB STUDY00025867, exempt).
This part covers the exploratory analyses added after Draft 2 (slider anchoring, direct budget choice, variance split by person and kind of data, and the N = 97 check); the main analysis script is in another part.

## Table and figure map

Labels are those in the chapter source; figure files keep their `_ch9` names from an earlier chapter numbering.

| Thesis table or figure | Script | Output file |
|---|---|---|
| Table: Participants of the user study (`tab:ch9:participants`) | `analysis.py` (another part) | table in the text |
| Figure: Self-rated understanding before and after the demo (`fig:ch9:comprehension`) | `analysis.py` (another part) | `comprehension_ch9.pdf` |
| Figure: Weights from sliders and from PrivAgent (`fig:ch9:weights`) | `analysis.py` (another part) | `weights_ch9.pdf` |
| Table: Budgets selected from the participants' weights (`tab:ch9:selection`) | `analysis.py` (another part) | table in the text |
| Figure: Recommended, selected and chosen budgets (`fig:ch9:budgets`) | `analysis.py` (another part) | `budgets_ch9.pdf` |
| Figure: Ratings of the tool (`fig:ch9:ratings`) | `analysis.py` (another part) | `ratings_ch9.pdf` |
| Slider movement (13, 19, 39 participants; median move 0.27; privacy share 31 / 57 / 12), paragraph "Slider Weights Stay Near Their Start Position" | `code/ch8_extra_analysis.py` | `outputs/n100/extra_analysis.json` (`anchoring`) |
| ε* = 1 interval and direct choice (68 moved, 12 and 3 crossed; 57 / 29 / 9 / 5), paragraph "With Sliders That Started at 40/40/20" | `code/ch8_extra_analysis.py` | `outputs/n100/extra_analysis.json` (`anchoring.left_eps1_interval_*`, `anchoring.C2_dist`) |
| Variance of log10 ε (person 41%, kind of data 4.7%; sums of squares 60%, 3.6%, 37%), paragraph "The Chosen Budget Depended on the Kind of Data, but Far More on the Person" | `code/ch8_extra_analysis.py` | `outputs/n100/extra_analysis.json` (`variance_log_eps`) |
| N = 97 check (82, 56, 41%, 4.8%), Sensitivity Analysis | `code/check_ch8.py` | `outputs/n97/extra_analysis.json` |

The N = 97 check removes the three participants flagged on three or more of five quality checks: completion under five minutes, the same answer on all ten SUS items, on all four trust items, or on all three load items, and sliders left at the start position. All three finished in under five minutes.

Bootstrap intervals depend on the order of the rows, and the rows were reordered by the random participant IDs. Intervals computed from the public data may therefore differ from the thesis in the last digit. For example, the kind-of-data variance component is [1.2%, 10.7%] here and [1.3%, 10.8%] in the thesis. Point estimates are identical.

## Check command

From `part_B/ch8_user_study/`, on a CPU, in about 3 seconds:

```
pip install -r code/requirements.txt
python3 -I code/check_ch8.py data/ch8_user_study_deidentified.csv outputs
```

It runs `ch8_extra_analysis.py` on all 100 participants and on the N = 97 check, compares 16 reported numbers with the thesis, prints OK or FAIL for each, and exits with status 1 if any differ.

## Requirements

- Python 3.13 (tested with 3.13.18)
- numpy 2.5.3
- pandas 3.0.5

Both packages are pinned in `code/requirements.txt`. `openpyxl` is needed only to read an `.xlsx` file, not the public `.csv`.

## Data

- `data/ch8_user_study_deidentified.csv`: one row per participant, 100 rows, 55 columns (answers, slider weights, inferred weights, chosen and recommended budgets, completion time, age bracket, technical background, privacy-tool use).
- `data/README.md`: what was removed, and one line per column.
- `instrument/survey_index.html`: the survey page as deployed on Prolific (July 2026), with the submission endpoint, the Prolific completion code and all contact details redacted. With the endpoint left as a placeholder, the page submits nowhere and copies the answers to the clipboard instead.
- `instrument/apps_script_backend.gs`: the Google Apps Script that wrote each submission to a Google Sheet. It contains no keys or addresses.

## Development commits

None. This code was written in chat sessions, not in a repository. `ch8_extra_analysis.py` dates from November 12, 2026. The only change for this snapshot is that it now reads `.csv` as well as `.xlsx`. `check_ch8.py` was written for this snapshot.

## Excluded

- **The collected response file (`UCLDP_results.xlsx`).** It contains session IDs, completion codes, timestamps, user agents, locales and free-text answers; only the de-identified file is public.
- **The 18 demo rows.** They are survey tests, not participants.
- **The free-text answers (OE2 to OE4).** Participants may identify themselves in free text. The open-answer theme counts in the chapter, made by `code_open.py` in another part, therefore cannot be reproduced from the public data.
- **Browser locale.** It is location data. The sentence that most participants used a US English locale cannot be checked from the public data.
- **The de-identification script.** It needs the collected file, and its random ID order is deliberately not reproducible.
- **The live submission endpoint and the Prolific completion code.** The first would let anyone write to the study sheet; the second would let anyone claim completion.
- **The contact emails and phone number on the consent and thank-you screens.** Redacted at the author's request.
- **Recruitment screenshots and IRB forms.** These are the MTurk and Prolific setup screenshots, the UW IRB templates and the draft protocol. They hold account details and are not needed to check the chapter.
- **The survey assessment document.** It is an internal review of an older survey version.
- **Earlier drafts and figures.** Synthetic test rows, an earlier draft of the chapter, and three figures from the simulation chapter that the user study replaced. None is used by the current chapter.
