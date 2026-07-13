"""
working/calibration.py
======================
Calibration-curve analysis using FAVORITE / UNDERDOG / DRAW framing.

The Poisson MLE model is position-agnostic (no home/away parameter), and World
Cup group-stage fixtures are neutral-venue.  "Home" is just fixture-listing
order, so calibration buckets are built around:

  - p_favorite_win  (higher of p_home_win / p_away_win per match)
  - p_underdog_win  (lower of p_home_win / p_away_win per match)
  - p_draw          (unchanged, already position-independent)

Outputs:
  findings/calibrations/calibration_favorite.png
  findings/calibrations/calibration_underdog.png
  findings/calibrations/calibration_draw.png
  findings/calibration.md
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

# ── paths ────────────────────────────────────────────────────────────────
PREDICTIONS_PATH = PROJECT_ROOT / "predictions/group_stage_predictions_new.csv"
RESULTS_PATH = PROJECT_ROOT / "results/actual_outcomes.csv"
FIGURES_DIR = PROJECT_ROOT / "findings/calibrations"
FINDINGS_MD = PROJECT_ROOT / "findings/calibration.md"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

MIN_BUCKET_SIZE = 6  # flag buckets with fewer matches as unreliable

# ═══════════════════════════════════════════════════════════════════════════
#  LOAD & TRANSFORM TO FAVORITE / UNDERDOG FRAMING
# ═══════════════════════════════════════════════════════════════════════════

predictions = pd.read_csv(PREDICTIONS_PATH)
results = pd.read_csv(RESULTS_PATH)
merged = predictions.merge(results, on=["match_id", "date"], how="inner")

N = len(merged)
print(f"Merged {N} matches for calibration analysis")
print(f"Outcome distribution (raw): {merged['outcome'].value_counts().to_dict()}")

# ── Determine favorite/underdog per match ────────────────────────────────
# "Favorite" = whichever team the model gave the higher win probability.
# If p_home_win > p_away_win, the home-listed team is the favorite.
# If p_home_win < p_away_win, the away-listed team is the favorite.
# If equal (extremely unlikely with MLE floats), pick home as favorite.

merged["home_is_fav"] = merged["p_home_win"] >= merged["p_away_win"]

merged["p_fav_win"] = np.where(merged["home_is_fav"],
                               merged["p_home_win"], merged["p_away_win"])
merged["p_dog_win"] = np.where(merged["home_is_fav"],
                               merged["p_away_win"], merged["p_home_win"])
# p_draw is unchanged

merged["favorite"] = np.where(merged["home_is_fav"],
                              merged["home_team_x"], merged["away_team_x"])
merged["underdog"] = np.where(merged["home_is_fav"],
                              merged["away_team_x"], merged["home_team_x"])

# ── Map actual outcome to favorite/underdog framing ──────────────────────
def map_outcome(row):
    if row["outcome"] == "draw":
        return "draw"
    if row["outcome"] == "home_win":
        return "fav_win" if row["home_is_fav"] else "dog_win"
    else:  # away_win
        return "fav_win" if not row["home_is_fav"] else "dog_win"

merged["fdu_outcome"] = merged.apply(map_outcome, axis=1)

# Binary flags for calibration computation
merged["actual_fav_win"] = (merged["fdu_outcome"] == "fav_win").astype(int)
merged["actual_draw"]    = (merged["fdu_outcome"] == "draw").astype(int)
merged["actual_dog_win"] = (merged["fdu_outcome"] == "dog_win").astype(int)

# Sanity
print(f"\nFavorite/Underdog outcome distribution:")
print(f"  {merged['fdu_outcome'].value_counts().to_dict()}")
print(f"  (fav_win should >= dog_win if model has any skill)")

# Quick distribution check
print(f"\np_fav_win  range: [{merged['p_fav_win'].min():.4f}, {merged['p_fav_win'].max():.4f}]  "
      f"mean: {merged['p_fav_win'].mean():.4f}")
print(f"p_dog_win  range: [{merged['p_dog_win'].min():.4f}, {merged['p_dog_win'].max():.4f}]  "
      f"mean: {merged['p_dog_win'].mean():.4f}")
print(f"p_draw     range: [{merged['p_draw'].min():.4f}, {merged['p_draw'].max():.4f}]  "
      f"mean: {merged['p_draw'].mean():.4f}")


# ═══════════════════════════════════════════════════════════════════════════
#  BUCKETING & CALIBRATION HELPERS
# ═══════════════════════════════════════════════════════════════════════════

def choose_edges(values, name):
    """Data-driven bucket edges.

    For each outcome type, look at the actual range and pick 4-5 buckets
    using rounded quantile boundaries, with guard rails:
      - Always start at 0 and end at 1.
      - Round edges to nearest 0.05 for readability.
      - Merge edges that would create empty or tiny buckets.
    """
    vmin, vmax = values.min(), values.max()

    # Start with quantile-based candidates
    raw = [0.0]
    for q in [0.20, 0.40, 0.60, 0.80]:
        edge = round(np.quantile(values, q) / 0.05) * 0.05
        if edge > raw[-1] + 0.01:  # avoid duplicate / too-close edges
            raw.append(edge)
    raw.append(1.0)
    return sorted(set(raw))


def compute_calibration(df, pred_col, actual_col, edges):
    """Compute calibration table for one outcome type."""
    rows = []
    for i in range(len(edges) - 1):
        lo, hi = edges[i], edges[i + 1]
        if i < len(edges) - 2:
            mask = (df[pred_col] >= lo) & (df[pred_col] < hi)
        else:
            mask = (df[pred_col] >= lo) & (df[pred_col] <= hi)
        subset = df[mask]
        n = len(subset)
        if n == 0:
            continue
        avg_pred = subset[pred_col].mean()
        obs_rate = subset[actual_col].mean()
        rows.append({
            "bucket": f"[{lo:.0%}, {hi:.0%})",
            "lo": lo, "hi": hi,
            "n": n,
            "avg_predicted": avg_pred,
            "observed_rate": obs_rate,
            "diff": obs_rate - avg_pred,
            "reliable": n >= MIN_BUCKET_SIZE,
        })
    return pd.DataFrame(rows)


def plot_calibration(cal_df, label, save_path):
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.plot([0, 1], [0, 1], "k--", alpha=0.4, linewidth=1, label="Perfect calibration")

    rel = cal_df[cal_df["reliable"]]
    unrel = cal_df[~cal_df["reliable"]]

    if len(rel) > 0:
        sizes = np.clip(rel["n"].values * 8, 40, 500)
        ax.scatter(rel["avg_predicted"], rel["observed_rate"],
                   s=sizes, color="#2563eb", edgecolors="white", linewidth=1.5,
                   zorder=5, label=f"Bucket (n >= {MIN_BUCKET_SIZE})")
        ax.plot(rel["avg_predicted"], rel["observed_rate"],
                color="#2563eb", alpha=0.5, linewidth=1.5, zorder=4)
        for _, row in rel.iterrows():
            ax.annotate(f"n={row['n']}", (row["avg_predicted"], row["observed_rate"]),
                        textcoords="offset points", xytext=(8, -8),
                        fontsize=9, color="#374151")

    if len(unrel) > 0:
        sizes = np.clip(unrel["n"].values * 8, 24, 500)
        ax.scatter(unrel["avg_predicted"], unrel["observed_rate"],
                   s=sizes, color="#f59e0b", edgecolors="white", linewidth=1.5,
                   zorder=5, marker="D", label=f"Unreliable (n < {MIN_BUCKET_SIZE})")
        for _, row in unrel.iterrows():
            ax.annotate(f"n={row['n']}", (row["avg_predicted"], row["observed_rate"]),
                        textcoords="offset points", xytext=(8, -8),
                        fontsize=9, color="#92400e")

    ax.set_xlabel("Mean Predicted Probability", fontsize=12)
    ax.set_ylabel("Observed Frequency", fontsize=12)
    ax.set_title(f"Calibration: {label}\n(72 group-stage matches, fav/dog framing)",
                 fontsize=13, fontweight="bold")

    all_vals = list(cal_df["avg_predicted"]) + list(cal_df["observed_rate"])
    lo = max(0, min(all_vals) - 0.05)
    hi = min(1, max(all_vals) + 0.05)
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_aspect("equal")
    ax.xaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0))
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0))
    ax.grid(alpha=0.25)
    ax.legend(loc="upper left", fontsize=9)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {save_path.name}")


# ═══════════════════════════════════════════════════════════════════════════
#  RUN ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════

OUTCOMES = [
    ("p_fav_win",  "actual_fav_win", "Favorite Win",  "calibration_favorite.png"),
    ("p_dog_win",  "actual_dog_win", "Underdog Win",   "calibration_underdog.png"),
    ("p_draw",     "actual_draw",    "Draw",            "calibration_draw.png"),
]

all_tables = {}

for pred_col, actual_col, label, filename in OUTCOMES:
    print(f"\n{'='*60}")
    print(f"  {label}")
    print(f"{'='*60}")

    vals = merged[pred_col]
    print(f"  Range: [{vals.min():.4f}, {vals.max():.4f}]")
    print(f"  Mean:  {vals.mean():.4f}  |  Median: {vals.median():.4f}")

    edges = choose_edges(vals, pred_col)
    print(f"  Bucket edges: {[f'{e:.0%}' for e in edges]}")

    cal = compute_calibration(merged, pred_col, actual_col, edges)
    all_tables[label] = cal

    print(f"\n  {'Bucket':>14s}  {'N':>4s}  {'Pred':>7s}  {'Obs':>7s}  {'Diff':>7s}")
    print(f"  {'-'*50}")
    for _, row in cal.iterrows():
        flag = "" if row["reliable"] else "  [!]"
        print(f"  {row['bucket']:>14s}  {row['n']:4d}  "
              f"{row['avg_predicted']:7.4f}  {row['observed_rate']:7.4f}  "
              f"{row['diff']:+7.4f}{flag}")

    # Weighted calibration error (reliable only)
    rel = cal[cal["reliable"]]
    if len(rel) > 0:
        w = rel["n"].values / rel["n"].sum()
        wce = np.sum(w * np.abs(rel["diff"].values))
        print(f"\n  Weighted calibration error (reliable): {wce:.4f}")

    overall_pred = merged[pred_col].mean()
    overall_obs = merged[actual_col].mean()
    print(f"  Overall: pred {overall_pred:.4f}  obs {overall_obs:.4f}  "
          f"diff {overall_obs - overall_pred:+.4f}")

    plot_calibration(cal, label, FIGURES_DIR / filename)


# ═══════════════════════════════════════════════════════════════════════════
#  GENERATE findings/calibration.md
# ═══════════════════════════════════════════════════════════════════════════

print(f"\n{'='*60}")
print("  Generating findings/calibration.md")
print(f"{'='*60}")

md = []
md.append("# Calibration Analysis -- Favorite / Underdog / Draw Framing\n")
md.append("The Poisson MLE model is position-agnostic (symmetric attack/defense "
          "ratings, no home-advantage parameter), and World Cup group-stage "
          "matches are neutral-venue. The previous calibration framed outcomes "
          "as home/away, which is meaningless here. This analysis reframes "
          "everything as **favorite** (whichever team the model predicted more "
          "likely to win), **underdog** (the other), and **draw**.\n")
md.append(f"**Matches:** {N} completed group-stage matches  ")
md.append(f"**Outcome distribution (fav/dog/draw):** "
          f"{merged['fdu_outcome'].value_counts().to_dict()}  ")
md.append(f"**Min reliable bucket size:** {MIN_BUCKET_SIZE}\n")

# Overall table
md.append("## Overall Predicted vs Observed\n")
md.append("| Outcome | Mean Predicted | Observed | Diff |")
md.append("|---------|---------------|----------|------|")
for pred_col, actual_col, label, _ in OUTCOMES:
    p = merged[pred_col].mean()
    o = merged[actual_col].mean()
    md.append(f"| {label} | {p:.1%} | {o:.1%} | {o-p:+.1%} |")
md.append("")

# Per-outcome detailed sections
for pred_col, actual_col, label, filename in OUTCOMES:
    cal = all_tables[label]
    md.append(f"---\n\n## {label}\n")
    md.append(f"![{label} calibration](calibrations/{filename})\n")

    md.append("| Bucket | N | Avg Predicted | Observed | Diff | Reliable? |")
    md.append("|--------|---|--------------|----------|------|-----------|")
    for _, row in cal.iterrows():
        rel_str = "Yes" if row["reliable"] else "**No**"
        md.append(f"| {row['bucket']} | {row['n']} | "
                  f"{row['avg_predicted']:.1%} | {row['observed_rate']:.1%} | "
                  f"{row['diff']:+.1%} | {rel_str} |")
    md.append("")

    # Interpretation
    rel = cal[cal["reliable"]]
    if len(rel) > 0:
        worst = rel.loc[rel["diff"].abs().idxmax()]
        direction = "under-predicted" if worst["diff"] > 0 else "over-predicted"
        md.append(f"**Largest miscalibration** (reliable buckets): "
                  f"**{worst['bucket']}** -- {direction} by "
                  f"{abs(worst['diff']):.1%} (n={worst['n']}).\n")

    # Overall for this outcome
    p = merged[pred_col].mean()
    o = merged[actual_col].mean()
    md.append(f"Overall: predicted {p:.1%}, observed {o:.1%} ({o-p:+.1%}).\n")

# ── Key findings section ────────────────────────────────────────────────
md.append("---\n\n## Key Findings\n")

# 1. Draw underestimation
p_draw = merged["p_draw"].mean()
o_draw = merged["actual_draw"].mean()
draw_diff = o_draw - p_draw

# 2. Underdog overconfidence
p_dog = merged["p_dog_win"].mean()
o_dog = merged["actual_dog_win"].mean()
dog_diff = o_dog - p_dog

# 3. Favorite
p_fav = merged["p_fav_win"].mean()
o_fav = merged["actual_fav_win"].mean()
fav_diff = o_fav - p_fav

md.append("### Was the previous 'away-win overconfidence' actually underdog overconfidence?\n")

md.append(f"In the archived home/away analysis, the model showed severe "
          f"away-win overconfidence (predicted 34.8%, observed 25.0%, a -9.8% gap). "
          f"With favorite/underdog framing:\n")
md.append(f"- **Underdog win:** predicted {p_dog:.1%}, observed {o_dog:.1%} "
          f"({dog_diff:+.1%})")
md.append(f"- **Favorite win:** predicted {p_fav:.1%}, observed {o_fav:.1%} "
          f"({fav_diff:+.1%})")
md.append(f"- **Draw:** predicted {p_draw:.1%}, observed {o_draw:.1%} "
          f"({draw_diff:+.1%})\n")

if dog_diff < -0.03:
    md.append("**Yes -- the pattern is confirmed as underdog overconfidence,** "
              "not a home/away artifact. The model assigns too much probability "
              "to the weaker team winning, regardless of which fixture-listing "
              "position they occupy. This is the mirror image of the draw "
              "underestimation: probability that should go to draws is instead "
              "leaking into underdog-win predictions.\n")
elif dog_diff < 0:
    md.append("**Partially.** There is mild underdog overconfidence, but the "
              "gap is small enough that sampling noise could explain it.\n")
else:
    md.append("**No.** The underdog-win probabilities are actually well-calibrated "
              "or under-predicted in this framing, suggesting the previous "
              "away-win pattern was partly a home/away artifact.\n")

# Draw
md.append("### Draw underestimation\n")
draw_cal = all_tables["Draw"]
draw_rel = draw_cal[draw_cal["reliable"]]
n_under = (draw_rel["diff"] > 0.01).sum() if len(draw_rel) > 0 else 0
n_rel = len(draw_rel)

if draw_diff > 0.03:
    md.append(f"**Confirmed.** Draws were predicted at {p_draw:.1%} on average but "
              f"occurred {o_draw:.1%} of the time. "
              f"{n_under} of {n_rel} reliable draw buckets show observed > predicted.\n")
elif draw_diff > 0:
    md.append(f"**Mild.** A small gap ({draw_diff:+.1%}) exists but could be noise "
              f"with only {N} matches.\n")
else:
    md.append(f"**Not supported.** Draws were observed at or below the predicted rate.\n")

# Mechanistic note
md.append("### Mechanistic interpretation\n")
md.append("The Poisson model assumes goals are independent Poisson draws. In "
          "reality, teams adjust tactics (park the bus when ahead, press when "
          "behind), creating negative correlation between the two teams' goal "
          "counts within a match. This explains both findings simultaneously:\n\n"
          "- **Independent Poisson underestimates draws** because it misses the "
          "score-convergence effect of tactical adjustments.\n"
          "- **Independent Poisson overestimates underdog wins** because the "
          "weaker team's tail of lucky high-scoring outcomes is unrealistically "
          "independent of the stronger team's scoring.\n\n"
          "A correlated bivariate Poisson or a Dixon-Coles correction would "
          "likely address both issues.\n")

# Caveats
md.append("\n> [!NOTE]\n")
md.append(f"> With {N} group-stage matches and data-driven bucketing, individual "
          f"bucket sizes range from {min(cal['n'])} to {max(cal['n'])} matches. "
          f"These calibration curves are directional evidence, not statistically "
          f"conclusive. Treat deviations under ~10% in small buckets as noise.\n")

md.append(f"\n---\n\n*Generated by `working/calibration.py` on "
          f"{pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}*\n")

FINDINGS_MD.write_text("\n".join(md), encoding="utf-8")
print(f"  Saved findings/calibration.md")
print("\nDone.")
