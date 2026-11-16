"""Check that the public data reproduce the Chapter 8 numbers of the extra analysis.

Usage, from ch8_user_study/:
    python3 -I code/check_ch8.py data/ch8_user_study_deidentified.csv outputs

Runs code/ch8_extra_analysis.py on all 100 participants and on the N = 97 check
(the three participants flagged on three or more quality checks removed),
writes both JSON files to outputs/, and fails if any reported number differs.
"""
import sys, json, subprocess, pathlib
import pandas as pd

DATA, OUT = pathlib.Path(sys.argv[1]), pathlib.Path(sys.argv[2])
OUT.mkdir(parents=True, exist_ok=True)
HERE = pathlib.Path(__file__).resolve().parent
df = pd.read_csv(DATA)

# Quality flags (Section 8.1 sensitivity analysis).
sus = [f"SUS{i}" for i in range(1, 11)]
trust = ["T1_trust_recommendation", "T2_transparency", "T3_stability", "T4_adoption_intention"]
load = ["CL1_mental_demand", "CL2_effort", "CL3_frustration"]
flags = ((df.time_to_complete_seconds < 300).astype(int)
         + (df[sus].nunique(axis=1) == 1)
         + (df[trust].nunique(axis=1) == 1)
         + (df[load].nunique(axis=1) == 1)
         + ((df.slider_LP.round(2) == 0.4) & (df.slider_LA.round(2) == 0.4) & (df.slider_CR.round(2) == 0.2)))
n97 = OUT / "n97_input.csv"
df[flags < 3].to_csv(n97, index=False)

def run(src, sub):
    d = OUT / sub; d.mkdir(exist_ok=True)
    subprocess.run([sys.executable, "-I", str(HERE / "ch8_extra_analysis.py"), str(src), str(d)],
                   check=True, stdout=subprocess.DEVNULL)
    return json.load(open(d / "extra_analysis.json"))

a, b = run(DATA, "n100"), run(n97, "n97")
c2 = df.C2_forced_eps_choice.astype(float)
c2_97 = df[flags < 3].C2_forced_eps_choice.astype(float)

rows = [  # (what, got, expected)
    ("N", a["N"], 100),
    ("direct choice eps = 1", int((c2 == 1).sum()), 57),
    ("direct choice eps = 0.3", int((c2 == 0.3).sum()), 29),
    ("direct choice eps = 0.1", int((c2 == 0.1).sum()), 9),
    ("direct choice weaker than 1", int((c2 > 1).sum()), 5),
    ("person, share of variance (%)", round(100 * a["variance_log_eps"]["variance_component_share"]["participant"]), 41),
    ("kind of data, share of variance (%)", round(100 * a["variance_log_eps"]["variance_component_share"]["data_type"], 1), 4.7),
    ("median total slider move", round(a["anchoring"]["L1_move_median"], 2), 0.27),
    ("privacy share unchanged", a["anchoring"]["share_equal_0.5"], 31),
    ("privacy share up", a["anchoring"]["share_above_0.5"], 57),
    ("privacy share down", a["anchoring"]["share_below_0.5"], 12),
    ("N = 97 check: N", b["N"], 97),
    ("N = 97 check: eps* = 1 from sliders", b["anchoring"]["left_eps1_interval_Adult"]["stayed_inside"], 82),
    ("N = 97 check: direct choice eps = 1", int((c2_97 == 1).sum()), 56),
    ("N = 97 check: person (%)", round(100 * b["variance_log_eps"]["variance_component_share"]["participant"]), 41),
    ("N = 97 check: kind of data (%)", round(100 * b["variance_log_eps"]["variance_component_share"]["data_type"], 1), 4.8),
]
bad = 0
for what, got, exp in rows:
    ok = got == exp; bad += not ok
    print(f"{'OK  ' if ok else 'FAIL'} {what:40} {got!s:>6}  (thesis {exp})")
print("all numbers reproduced" if not bad else f"{bad} number(s) differ")
sys.exit(1 if bad else 0)
