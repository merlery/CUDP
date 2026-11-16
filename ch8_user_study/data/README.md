# Chapter 8 user-study data (de-identified)

`ch8_user_study_deidentified.csv`: one row per participant, 100 rows, 55 columns.

The study was reviewed by the University of Washington Human Subjects Division and determined to be exempt (IRB ID STUDY00025867). Participants were recruited on Prolific.

The data are de-identified. Removed from the collected file:

- the 18 demo rows used to test the survey
- the browser session ID and the completion code
- submission and consent timestamps
- browser user agent and locale
- the three free-text answers (OE2 to OE4), and the raw JSON copy of each submission
- demo interaction logs and fixed demo values that the chapter does not analyze

Prolific IDs, IP addresses and email addresses were never recorded. The participant IDs P001 to P100 were assigned in a random order that was not stored, so rows cannot be linked back to the collected file. Item wording is in `../instrument/survey_index.html` and in the thesis appendix.

## Columns

Ratings marked 1–7 run from strongly disagree to strongly agree unless noted. Budget columns take one of 0.1, 0.3, 1, 5, 10.

| Column | Meaning |
|---|---|
| `participant_id` | P001 to P100, random, not linked to anything else |
| `recruitment_source` | Always `prolific`; kept so the scripts' filter runs unchanged |
| `time_to_complete_seconds` | Seconds from consent to submission |
| `A1_iuipc_collection` | 1–7, concern about data collection (adapted from IUIPC) |
| `A2_iuipc_awareness` | 1–7, awareness of data collection (adapted from IUIPC) |
| `A3_prior_eps_comprehension` | 1–7, understanding of ε before the demo (1 = not at all, 7 = very clearly) |
| `A4_age_range` | `18-24`, `25-34`, `35-44`, `45-54`, `55+`, `prefer_not_to_say` |
| `A5_technical_background` | `non_technical`, `some_background`, `cs_or_data_professional` |
| `A6_privacy_tool_experience` | `never`, `once_or_twice`, `regularly` |
| `slider_LP` | Privacy weight from the sliders (the three slider weights sum to 1; start 0.4) |
| `slider_LA` | Accuracy weight from the sliders (start 0.4) |
| `slider_CR` | Cost weight from the sliders (start 0.2) |
| `topsis_eps` | Budget recommended by the study prototype and shown to the participant |
| `pa_LP` | Privacy weight inferred from the six pairwise answers (earlier vote rule of PrivAgent) |
| `pa_LA` | Accuracy weight inferred from the pairwise answers |
| `pa_CR` | Cost weight inferred from the pairwise answers |
| `pa_r1` | Answer `a` or `b` to pairwise prompt 1 (fitness app, privacy vs accuracy) |
| `pa_r2` | Answer to prompt 2 (music or streaming app, privacy vs accuracy) |
| `pa_r3` | Answer to prompt 3 (home device, privacy vs cost) |
| `pa_r4` | Answer to prompt 4 (phone battery, privacy vs cost) |
| `pa_r5` | Answer to prompt 5 (live billing display, accuracy vs cost) |
| `pa_r6` | Answer to prompt 6 (health summary, accuracy vs cost) |
| `B1_post_demo_eps_comprehension` | 1–7, understanding of ε after the demo |
| `B2_demo_tradeoff_clarity` | 1–7, the demo made the trade-off clear |
| `B3_linkage_attack_comprehension` | 1–7, understood how linking data sources can reveal information |
| `B4_meter` | Budget chosen for smart-meter readings |
| `B4_identity` | Budget chosen for identity data (health record card) |
| `B4_ecg` | Budget chosen for an electrocardiogram |
| `B5_self_identified_archetype` | `privacy_first`, `accuracy_first`, `balanced`, `efficient` |
| `C2_forced_eps_choice` | One budget chosen directly for the participant's own data |
| `C3_topsis_satisfaction` | 1–7 (1 = completely wrong, 7 = exactly right), match of the recommendation |
| `C4_explanation_quality` | 1–7, the tool explained why it recommended the setting |
| `C5_perceived_control` | 1–7, sense of control |
| `C6_archetype_match` | `match`, `close`, `mismatch`: sliders against the B5 profile |
| `SUS1` | 1–5, System Usability Scale item 1 (Brooke 1996 wording) |
| `SUS2` | 1–5, SUS item 2 |
| `SUS3` | 1–5, SUS item 3 |
| `SUS4` | 1–5, SUS item 4 |
| `SUS5` | 1–5, SUS item 5 |
| `SUS6` | 1–5, SUS item 6 |
| `SUS7` | 1–5, SUS item 7 |
| `SUS8` | 1–5, SUS item 8 |
| `SUS9` | 1–5, SUS item 9 |
| `SUS10` | 1–5, SUS item 10 |
| `SUS_composite_0_100` | SUS score, 0–100, computed by the survey from SUS1 to SUS10 |
| `T1_trust_recommendation` | 1–7, trust in the recommendation |
| `T2_transparency` | 1–7, transparency of the reasoning |
| `T3_stability` | 1–7, expected stability of the recommendation |
| `T4_adoption_intention` | 1–7, willingness to use such a tool for real data |
| `CL1_mental_demand` | 1–7 (1 = very low, 7 = very high), mental demand |
| `CL2_effort` | 1–7 (1 = very low, 7 = very high), effort |
| `CL3_frustration` | 1–7 (1 = very relaxed, 7 = very tense); the column name keeps the NASA-TLX construct |
| `CL_mean` | Mean of CL1 to CL3, computed by the survey |
| `OE1_elicitation_preference` | Preferred way to set privacy: `preference_questions`, `sliders`, `ai_chat`, `default` |
