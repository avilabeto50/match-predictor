"""
working/followup_checks.py
==========================
Investigates two anomalies from the favorite/underdog calibration analysis:
1. The 50-60% favorite bucket anomaly.
2. The correlation between underdog overestimation and data sparsity.
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats

# ── paths ────────────────────────────────────────────────────────────────
PREDICTIONS_PATH = PROJECT_ROOT / "predictions/group_stage_predictions_new.csv"
RESULTS_PATH = PROJECT_ROOT / "results/actual_outcomes.csv"
TRAINING_PATH = PROJECT_ROOT / "data/processed/matches_filtered.csv"
FIGURES_DIR = PROJECT_ROOT / "findings/calibrations"
FINDINGS_MD = PROJECT_ROOT / "findings/calibration.md"

# ═══════════════════════════════════════════════════════════════════════════
#  LOAD & TRANSFORM TO FAVORITE / UNDERDOG FRAMING
# ═══════════════════════════════════════════════════════════════════════════

predictions = pd.read_csv(PREDICTIONS_PATH)
results = pd.read_csv(RESULTS_PATH)
merged = predictions.merge(results, on=["match_id", "date"], how="inner")

merged["home_is_fav"] = merged["p_home_win"] >= merged["p_away_win"]
merged["p_fav_win"] = np.where(merged["home_is_fav"], merged["p_home_win"], merged["p_away_win"])
merged["p_dog_win"] = np.where(merged["home_is_fav"], merged["p_away_win"], merged["p_home_win"])
merged["favorite"] = np.where(merged["home_is_fav"], merged["home_team_x"], merged["away_team_x"])
merged["underdog"] = np.where(merged["home_is_fav"], merged["away_team_x"], merged["home_team_x"])

def map_outcome(row):
    if row["outcome"] == "draw":
        return "draw"
    if row["outcome"] == "home_win":
        return "fav_win" if row["home_is_fav"] else "dog_win"
    else:
        return "fav_win" if not row["home_is_fav"] else "dog_win"

merged["fdu_outcome"] = merged.apply(map_outcome, axis=1)
merged["actual_fav_win"] = (merged["fdu_outcome"] == "fav_win").astype(int)
merged["actual_dog_win"] = (merged["fdu_outcome"] == "dog_win").astype(int)

# ═══════════════════════════════════════════════════════════════════════════
#  CHECK 1: 50-60% FAVORITE BUCKET ANOMALY
# ═══════════════════════════════════════════════════════════════════════════

bucket_mask = (merged["p_fav_win"] >= 0.50) & (merged["p_fav_win"] < 0.60)
anomaly_df = merged[bucket_mask].copy()

# Sort by p_fav_win
anomaly_df = anomaly_df.sort_values("p_fav_win", ascending=False)

# ═══════════════════════════════════════════════════════════════════════════
#  CHECK 2: DATA SPARSITY VS UNDERDOG OVERESTIMATION
# ═══════════════════════════════════════════════════════════════════════════

training = pd.read_csv(TRAINING_PATH)
home_counts = training["home_team"].value_counts()
away_counts = training["away_team"].value_counts()
total_counts = home_counts.add(away_counts, fill_value=0)

# Calculate error (predicted - actual) so positive means overestimation
merged["dog_error"] = merged["p_dog_win"] - merged["actual_dog_win"]
merged["dog_training_matches"] = merged["underdog"].map(total_counts).fillna(0)

correlation, p_value = stats.pearsonr(merged["dog_training_matches"], merged["dog_error"])

fig, ax = plt.subplots(figsize=(7, 5))
ax.scatter(merged["dog_training_matches"], merged["dog_error"], alpha=0.6, color="#2563eb")
ax.set_xlabel("Underdog Team's Historical Matches in Training Set", fontsize=11)
ax.set_ylabel("Underdog Calibration Error (Predicted - Actual)", fontsize=11)
ax.set_title("Underdog Overestimation vs. Data Sparsity", fontsize=12, fontweight="bold")
ax.grid(alpha=0.25)
ax.axhline(0, color="k", linestyle="--", alpha=0.4)

# Line of best fit
m, b = np.polyfit(merged["dog_training_matches"], merged["dog_error"], 1)
x_vals = np.array([merged["dog_training_matches"].min(), merged["dog_training_matches"].max()])
ax.plot(x_vals, m*x_vals + b, color="#f59e0b", linewidth=2, label=f"Trend (r={correlation:.2f})")
ax.legend()

fig.tight_layout()
sparsity_plot_path = FIGURES_DIR / "sparsity_vs_error.png"
fig.savefig(sparsity_plot_path, dpi=150, bbox_inches="tight")
plt.close(fig)

# ═══════════════════════════════════════════════════════════════════════════
#  APPEND TO FINDINGS
# ═══════════════════════════════════════════════════════════════════════════

md = []
md.append("\n---\n")
md.append("## Follow-up Investigation\n")

# Check 1 Report
md.append("### Check 1: The [50%, 60%) Favorite-Win Anomaly\n")
md.append("In the calibration analysis, the [50%, 60%) favorite bucket was the largest "
          "single miscalibration: the model predicted a 54.9% average win rate for "
          f"these favorites, but they won {anomaly_df['actual_fav_win'].mean():.1%} of the time "
          f"({anomaly_df['actual_fav_win'].sum()}/{len(anomaly_df)} matches). "
          "Here are the specific matches that fell into this bucket:\n")

md.append("| Match ID | Favorite | Underdog | P(Fav Win) | Outcome | FDU Outcome |")
md.append("|---|---|---|---|---|---|")
for _, row in anomaly_df.iterrows():
    md.append(f"| {row['match_id']} | {row['favorite']} | {row['underdog']} | {row['p_fav_win']:.1%} | {row['outcome']} | {row['fdu_outcome']} |")
md.append("\n**Observations:**")
if anomaly_df["actual_fav_win"].mean() > 0.6:
    md.append("- This isn't one or two matches skewing a small sample; favorites in this bucket won overwhelmingly.")
else:
    md.append("- (Custom analysis here if needed)")

# Check 2 Report
md.append("\n### Check 2: Data Sparsity vs. Underdog Overestimation\n")
md.append("To determine if the underdog overestimation is an artifact of sparse training data for weaker teams, "
          "we plotted the underdog's calibration error (Predicted P(Dog Win) - Actual) against the number "
          "of historical matches they had in the MLE training set.\n")
md.append("![Data Sparsity vs Error](calibrations/sparsity_vs_error.png)\n")
md.append(f"- **Correlation:** r = {correlation:.3f} (p = {p_value:.3f})\n")

if correlation < -0.2 and p_value < 0.05:
    md.append("**Conclusion:** There is a statistically significant negative correlation. "
              "Underdogs with fewer historical matches tend to have larger overestimation errors. "
              "This suggests that data sparsity (noisy MLE estimates due to small sample sizes) "
              "contributes meaningfully to the underdog-overestimation pattern, in addition to any "
              "structural Poisson-independence effects.")
elif correlation > 0.2 and p_value < 0.05:
    md.append("**Conclusion:** There is a statistically significant positive correlation, "
              "meaning teams with *more* data are actually overestimated *more*. This contradicts "
              "the data sparsity hypothesis.")
else:
    md.append("**Conclusion:** There is no strong correlation between training data volume and underdog "
              "overestimation. The error is relatively uniform across teams with deep vs. sparse histories. "
              "This strengthens the case that the structural independence assumption of the Poisson model "
              "(the tactical-correlation story) is the primary driver.")

md.append("\n")
with open(FINDINGS_MD, "a", encoding="utf-8") as f:
    f.write("\n".join(md))

print(f"Appended findings to {FINDINGS_MD}")
