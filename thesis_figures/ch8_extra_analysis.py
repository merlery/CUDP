"""Extra user-study analysis for the Draft 2 feedback (Chapter 8, comments 3 and 6).

Usage:
    python3 -I ch8_extra_analysis.py data/ch8_user_study_deidentified.csv outdir
(.csv or .xlsx; the public data are the de-identified .csv)

Writes outdir/extra_analysis.json and prints a short report. Nothing here was
planned before the data were collected; all of it is exploratory.

Part 1. Anchoring on the slider start position (40/40/20).
Part 2. Variance of log10(eps) in the three domain choices, split into
        participant, kind of data, and residual (two-way layout, one
        observation per cell).
"""
import sys, json, numpy as np, pandas as pd

df = (pd.read_csv if sys.argv[1].endswith(".csv") else pd.read_excel)(sys.argv[1]); OUT = sys.argv[2]
r = df[df.recruitment_source == "prolific"].copy().reset_index(drop=True)
rng = np.random.default_rng(20261112)
R = {"N": len(r)}

# ------------------------------------------------------------------ Part 1
START = np.array([0.4, 0.4, 0.2])                 # slider start, normalized
S = r[["slider_LP", "slider_LA", "slider_CR"]].values.astype(float)
S = S / S.sum(1, keepdims=True)
move = np.abs(S - START)                          # per-slider move, in weight
l1 = move.sum(1)
pi = S[:, 0] / np.maximum(S[:, 0] + S[:, 1], 1e-12)   # privacy share
PI0 = 0.5                                         # share of the start position

# selection intervals of eps* = 1 with the linear LP map (Table A.x, tab:ch6:switch)
INT = {"Adult": (0.472, 0.887), "CDC-Diabetes": (0.472, 0.857)}

def boot(x, B=10000):
    x = np.asarray(x, float); i = rng.integers(0, len(x), (B, len(x))); m = x[i].mean(1)
    return [float(x.mean()), float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))]

R["anchoring"] = {
    "start_weights": START.tolist(), "start_privacy_share": PI0,
    "mean_weights": S.mean(0).round(3).tolist(),
    "unmoved_all_three(|d|<0.01)": int((move.max(1) < 0.01).sum()),
    "moved_less_than_0.05_on_every_slider": int((move.max(1) < 0.05).sum()),
    "moved_less_than_0.10_on_every_slider": int((move.max(1) < 0.10).sum()),
    "mean_abs_move_per_slider": move.mean(0).round(3).tolist(),
    "median_abs_move_per_slider": np.median(move, 0).round(3).tolist(),
    "L1_move": boot(l1), "L1_move_median": float(np.median(l1)),
    "privacy_share_mean": boot(pi), "privacy_share_sd": float(pi.std(ddof=1)),
    "abs_share_move_from_0.5": boot(np.abs(pi - PI0)),
    "share_above_0.5": int((pi > PI0 + 1e-9).sum()), "share_below_0.5": int((pi < PI0 - 1e-9).sum()),
    "share_equal_0.5": int((np.abs(pi - PI0) <= 1e-9).sum()),
}
for ds, (lo, hi) in INT.items():
    inside = (pi >= lo) & (pi < hi)
    R["anchoring"][f"left_eps1_interval_{ds}"] = {
        "interval": [lo, hi], "stayed_inside": int(inside.sum()),
        "crossed_below(to eps=5 or 10)": int((pi < lo).sum()),
        "crossed_above(to eps<1)": int((pi >= hi).sum()),
        # how far the start position is from the nearest boundary
        "distance_start_to_lower_boundary": round(PI0 - lo, 3),
        "participants_who_moved_share_by_more_than_that": int((np.abs(pi - PI0) > PI0 - lo).sum()),
    }
# does the direct choice C2 agree with the anchored selection?
if "C2_forced_eps_choice" in r:
    c2 = r.C2_forced_eps_choice.astype(float)
    R["anchoring"]["C2_direct_choice_equals_1"] = int((c2 == 1.0).sum())
    R["anchoring"]["C2_dist"] = {str(k): int(v) for k, v in c2.value_counts().sort_index().items()}

# ------------------------------------------------------------------ Part 2
L = np.log10(r[["B4_meter", "B4_identity", "B4_ecg"]].astype(float).values)   # n x 3
n, k = L.shape
g = L.mean(); rowm = L.mean(1, keepdims=True); colm = L.mean(0, keepdims=True)
SS_tot = ((L - g) ** 2).sum()
SS_part = k * ((rowm - g) ** 2).sum()
SS_type = n * ((colm - g) ** 2).sum()
SS_res = SS_tot - SS_part - SS_type
df_p, df_t, df_r = n - 1, k - 1, (n - 1) * (k - 1)
MS_p, MS_t, MS_r = SS_part / df_p, SS_type / df_t, SS_res / df_r
# variance components of the two-way random-effects layout (negative estimates set to 0)
var_res = MS_r
var_part = max((MS_p - MS_r) / k, 0.0)
var_type = max((MS_t - MS_r) / n, 0.0)
tot = var_part + var_type + var_res

def shares(Lb):
    nb = Lb.shape[0]; gb = Lb.mean(); rm = Lb.mean(1, keepdims=True); cm = Lb.mean(0, keepdims=True)
    st = ((Lb - gb) ** 2).sum(); sp = k * ((rm - gb) ** 2).sum(); sy = nb * ((cm - gb) ** 2).sum(); sr = st - sp - sy
    mp, mt, mr = sp / (nb - 1), sy / (k - 1), sr / ((nb - 1) * (k - 1))
    vp, vt = max((mp - mr) / k, 0), max((mt - mr) / nb, 0); t = vp + vt + mr
    return sp / st, sy / st, vp / t, vt / t

bs = np.array([shares(L[rng.integers(0, n, n)]) for _ in range(5000)])   # resample participants
ci = lambda j: [float(np.percentile(bs[:, j], 2.5)), float(np.percentile(bs[:, j], 97.5))]
R["variance_log_eps"] = {
    "sum_of_squares_share": {"participant": float(SS_part / SS_tot), "data_type": float(SS_type / SS_tot),
                             "residual": float(SS_res / SS_tot),
                             "participant_ci": ci(0), "data_type_ci": ci(1)},
    "variance_component_share": {"participant": float(var_part / tot), "data_type": float(var_type / tot),
                                 "residual": float(var_res / tot),
                                 "participant_ci": ci(2), "data_type_ci": ci(3)},
    "share_of_bootstraps_with_data_type_component_larger_than_participant": float((bs[:, 3] > bs[:, 2]).mean()),
    "domain_means_log10": dict(zip(["meter", "identity", "ecg"], colm.ravel().round(3).tolist())),
    "sd_of_participant_means_log10": float(rowm.std(ddof=1)),
}

json.dump(R, open(f"{OUT}/extra_analysis.json", "w"), indent=2)
print(json.dumps(R, indent=2))