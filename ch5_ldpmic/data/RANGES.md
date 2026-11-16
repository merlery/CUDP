# Public feature ranges

Every feature `j` has a public clipping range `[a_j, b_j]` that is fixed
BEFORE looking at the private data.  The mechanism clips each record to this
box and calibrates its noise to `Delta_j = b_j - a_j`.  Nothing below was
computed from the data files.  Source codes:

- **schema** — one-hot indicator over the category list given in the dataset's
  own documentation file; range `[0, 1]`.
- **doc** — a bound stated in the dataset documentation or in the census /
  survey definition it cites.
- **declared** — no bound is documented; the range was declared a priori from
  general public knowledge of the quantity (NOT from the data).  Values that
  fall outside are clipped.  These need the author's confirmation
  (QUESTIONS.md Q12).

Encoded features are in the order of `FEATURES` in `ldpmic/data/adult.py` and
`ldpmic/data/bank.py`.  Missing-value tokens (`?`, `unknown`) are listed in
the schema where the documentation lists them; an unlisted value encodes as
all-zeros.

## adult (UCI Adult, `adult.data` / `adult.test`)

| feature | type | range | source |
|---|---|---|---|
| age | numeric | [17, 90] | doc: extraction rule `AAGE > 16` in adult.names; age top-coded at 90 in the 1994 CPS |
| workclass | 8 one-hot (`?` = all zeros) | [0,1] | schema (adult.names) |
| fnlwgt | numeric | [10000, 1500000] | declared (CPS final sampling weight) |
| education | 16 one-hot | [0,1] | schema |
| education-num | numeric | [1, 16] | doc (16 education levels) |
| marital-status | 7 one-hot | [0,1] | schema |
| occupation | 14 one-hot (`?` = all zeros) | [0,1] | schema |
| relationship | 6 one-hot | [0,1] | schema |
| race | 5 one-hot | [0,1] | schema |
| sex | 2 one-hot | [0,1] | schema |
| capital-gain | numeric | [0, 99999] | doc: CPS top-code value 99999 |
| capital-loss | numeric | [0, 5000] | declared |
| hours-per-week | numeric | [1, 99] | doc: CPS top-code 99 |
| native-country | 41 one-hot (`?` = all zeros) | [0,1] | schema |

d = 6 numeric + 99 one-hot = 105.  Label: `income > 50K`.  Partition group
attribute: `occupation` (15 groups including `?`).

## bank (UCI Bank Marketing, `bank-additional-full.csv`)

`duration` is dropped (the documentation says it must be discarded for a
realistic model).  `pdays` uses the documented sentinel 999 = "not previously
contacted" and is encoded as two features.

| feature | type | range | source |
|---|---|---|---|
| age | numeric | [17, 100] | declared |
| job | 12 one-hot | [0,1] | schema (bank-additional-names.txt) |
| marital | 4 one-hot | [0,1] | schema |
| education | 8 one-hot | [0,1] | schema |
| default | 3 one-hot | [0,1] | schema |
| housing | 3 one-hot | [0,1] | schema |
| loan | 3 one-hot | [0,1] | schema |
| contact | 2 one-hot | [0,1] | schema |
| month | 12 one-hot | [0,1] | schema (all calendar months) |
| day_of_week | 5 one-hot | [0,1] | schema |
| campaign | numeric | [1, 60] | declared |
| pdays_contacted | numeric {0,1} | [0, 1] | doc (sentinel 999) |
| pdays_days | numeric | [0, 30] | declared cap (days since last contact) |
| previous | numeric | [0, 10] | declared |
| poutcome | 3 one-hot | [0,1] | schema |
| emp.var.rate | numeric | [-4, 2] | declared (quarterly employment variation rate, Portugal 2008–2013) |
| cons.price.idx | numeric | [90, 96] | declared (monthly CPI) |
| cons.conf.idx | numeric | [-60, -20] | declared (monthly consumer confidence index) |
| euribor3m | numeric | [0, 6] | declared (3-month Euribor, %) |
| nr.employed | numeric | [4900, 5300] | declared (quarterly, thousands) |

d = 10 numeric + 55 one-hot = 65.  Label: `y == yes`.  Partition group
attribute: `job` (12 groups).  Train/test: stratified 80/20 split with the
constant seed in `ldpmic/data/bank.py` (no standard split exists).

## Model input scaling

The model receives `(x - a_j) / (b_j - a_j)`, an affine map that depends only
on the public ranges, applied AFTER sanitization (post-processing).  In those
units every coordinate's noise std is `sigma0 / sqrt(p_j)`.

## kdd (UCI Census-Income (KDD), `census-income.data` / `census-income.test`; DECISIONS.md D50)

Schema (attribute list and category values) parsed from `census-income.names`.
Dropped attributes (D50): detailed industry recode, detailed occupation recode,
state of previous residence, detailed household and family stat, country of
birth father / mother / self (all > 30 categories), instance weight (sampling
weight), year (survey year).  Every remaining nominal attribute is one-hot over
its documented value list (range [0,1], source: schema); the numeric ones:

| feature | range | source |
|---|---|---|
| age | [0, 90] | doc: CPS top-code 90 |
| wage per hour | [0, 9999] | declared: 4-digit field |
| capital gains | [0, 99999] | doc: CPS top-code 99999 |
| capital losses | [0, 9999] | declared: 4-digit field |
| dividends from stocks | [0, 99999] | doc: CPS top-code 99999 |
| num persons worked for employer | [0, 6] | doc: 7 codes 0..6 |
| weeks worked in year | [0, 52] | doc: weeks in a year |

Training split: fixed-seed (seed 0) subsample of 40,000 of the 199,523 records;
test split: 10,000 of the 99,762 (CPU budget, D50).  Partition attribute:
`major occupation code` (15 values) -> 50 clients, as for Adult.

## cdc (UCI CDC Diabetes Health Indicators, `data.csv`; DECISIONS.md D58)

Codes and ranges from the UCI variable descriptions (BRFSS codebook).  Age
(13 five-year groups), Income (8 levels) and Sex (0 = female, 1 = male) are
one-hot (range [0,1]); the other features are numeric:

| feature | range | source |
|---|---|---|
| HighBP, HighChol, CholCheck, Smoker, Stroke, HeartDiseaseorAttack, PhysActivity, Fruits, Veggies, HvyAlcoholConsump, AnyHealthcare, NoDocbcCost, DiffWalk | [0, 1] | doc: binary indicators |
| BMI | [10, 100] | declared a priori (no documented range) |
| GenHlth | [1, 5] | doc: 5-level scale |
| MentHlth, PhysHlth | [0, 30] | doc: days in the past 30 |
| Education | [1, 6] | doc: 6-level scale |

No official split: a fixed-seed (seed 0) permutation of the 253,680 records
gives 40,000 training and the next 10,000 test records (CPU budget).
Partition attribute: Education (6 levels) -> 50 clients.
