"""
working/calibration.py
======================
Calibration-curve analysis for the Poisson MLE group-stage predictions.

Produces:
  findings/calibrations/calibration_draw.png
  findings/calibrations/calibration_home_win.png
  findings/calibrations/calibration_away_win.png
  findings/calibration.md   (auto-generated summary)
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

# ── load & merge ─────────────────────────────────────────────────────────
predictions = pd.read_csv(PREDICTIONS_PATH)
results = pd.read_csv(RESULTS_PATH)
merged = predictions.merge(results, on=["match_id", "date"], how="inner")

# Binary outcome columns
merged["actual_home_win"] = (merged["outcome"] == "home_win").astype(int)
merged["actual_draw"] = (merged["outcome"] == "draw").astype(int)
merged["actual_away_win"] = (merged["outcome"] == "away_win").astype(int)

N = len(merged)
print(f"Merged {N} matches for calibration analysis")
print(f"Outcome distribution: {merged['outcome'].value_counts().to_dict()}")

# ═══════════════════════════════════════════════════════════════════════════
#  HELPER: data-driven bucketing & calibration computation
# ═══════════════════════════════════════════════════════════════════════════

MIN_BUCKET_SIZE = 5  # flag buckets smaller than this


def choose_bucket_edges(values, name):
    """Choose bucket edges based on the actual distribution of predicted probs.

    Strategy:
    - For narrow-range columns (like p_draw, range < 0.35): use ~4-5 buckets
      with edges at rounded quantile boundaries.
    - For wide-range columns (like p_home_win / p_away_win): use fixed
      probability bands that align with intuitive thresholds.
    - Always include 0.0 and 1.0 as endpoints.
    """
    vmin, vmax = values.min(), values.max()
    spread = vmax - vmin

    if spread < 0.35:
        # Narrow range (draws): quantile-based edges, rounded to 0.05
        raw_edges = [0.0]
        for q in [0.25, 0.50, 0.75]:
            edge = round(np.quantile(values, q) / 0.05) * 0.05
            if edge > raw_edges[-1]:
                raw_edges.append(edge)
        raw_edges.append(1.0)
        edges = sorted(set(raw_edges))
    else:
        # Wide range (home_win / away_win): fixed intuitive bands
        candidates = [0.0, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 1.0]
        # Keep only edges that fall within [vmin-0.05, vmax+0.05] to avoid
        # empty leading/trailing buckets, but always include 0 and 1
        edges = [0.0]
        for c in candidates[1:-1]:
            if vmin - 0.05 <= c <= vmax + 0.05:
                edges.append(c)
        edges.append(1.0)

    return edges


def compute_calibration(merged_df, pred_col, actual_col, edges):
    """Compute calibration table for one outcome type."""
    rows = []
    for i in range(len(edges) - 1):
        lo, hi = edges[i], edges[i + 1]
        if i < len(edges) - 2:
            mask = (merged_df[pred_col] >= lo) & (merged_df[pred_col] < hi)
        else:
            # Last bucket is inclusive on right
            mask = (merged_df[pred_col] >= lo) & (merged_df[pred_col] <= hi)

        subset = merged_df[mask]
        n = len(subset)
        if n == 0:
            continue

        avg_predicted = subset[pred_col].mean()
        observed_rate = subset[actual_col].mean()
        reliable = n >= MIN_BUCKET_SIZE

        rows.append({
            "bucket": f"[{lo:.2f}, {hi:.2f})",
            "lo": lo,
            "hi": hi,
            "n": n,
            "avg_predicted": avg_predicted,
            "observed_rate": observed_rate,
            "diff": observed_rate - avg_predicted,
            "reliable": reliable,
        })

    return pd.DataFrame(rows)


# ═══════════════════════════════════════════════════════════════════════════
#  HELPER: plot calibration curve
# ═══════════════════════════════════════════════════════════════════════════

def plot_calibration(cal_df, outcome_label, save_path):
    """Plot predicted-vs-observed calibration curve with perfect-calibration line."""
    fig, ax = plt.subplots(figsize=(7, 7))

    # Perfect calibration line
    ax.plot([0, 1], [0, 1], "k--", alpha=0.4, linewidth=1, label="Perfect calibration")

    # Reliable points
    rel = cal_df[cal_df["reliable"]]
    unrel = cal_df[~cal_df["reliable"]]

    if len(rel) > 0:
        sizes = rel["n"].values * 8
        ax.scatter(
            rel["avg_predicted"], rel["observed_rate"],
            s=sizes, color="#2563eb", edgecolors="white", linewidth=1.5,
            zorder=5, label=f"Bucket (n >= {MIN_BUCKET_SIZE})"
        )
        # Connect with line
        ax.plot(
            rel["avg_predicted"], rel["observed_rate"],
            color="#2563eb", alpha=0.5, linewidth=1.5, zorder=4
        )
        # Annotate counts
        for _, row in rel.iterrows():
            ax.annotate(
                f"n={row['n']}", (row["avg_predicted"], row["observed_rate"]),
                textcoords="offset points", xytext=(8, -8),
                fontsize=8, color="#374151"
            )

    if len(unrel) > 0:
        sizes = unrel["n"].values * 8
        ax.scatter(
            unrel["avg_predicted"], unrel["observed_rate"],
            s=sizes, color="#f59e0b", edgecolors="white", linewidth=1.5,
            zorder=5, marker="D", label=f"Unreliable (n < {MIN_BUCKET_SIZE})"
        )
        for _, row in unrel.iterrows():
            ax.annotate(
                f"n={row['n']}", (row["avg_predicted"], row["observed_rate"]),
                textcoords="offset points", xytext=(8, -8),
                fontsize=8, color="#92400e"
            )

    # Axis formatting
    ax.set_xlabel("Mean Predicted Probability", fontsize=12)
    ax.set_ylabel("Observed Frequency", fontsize=12)
    ax.set_title(f"Calibration Curve: {outcome_label}\n(72 group-stage matches)",
                 fontsize=13, fontweight="bold")

    # Set axis limits with padding
    all_x = cal_df["avg_predicted"]
    all_y = cal_df["observed_rate"]
    lo = max(0, min(all_x.min(), all_y.min()) - 0.05)
    hi = min(1, max(all_x.max(), all_y.max()) + 0.05)
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
#  RUN ANALYSIS FOR EACH OUTCOME TYPE
# ═══════════════════════════════════════════════════════════════════════════

OUTCOMES = [
    ("p_draw",     "actual_draw",     "Draw (p_draw)",          "calibration_draw.png"),
    ("p_home_win", "actual_home_win", "Home Win (p_home_win)",  "calibration_home_win.png"),
    ("p_away_win", "actual_away_win", "Away Win (p_away_win)",  "calibration_away_win.png"),
]

all_tables = {}  # outcome_label -> cal_df

for pred_col, actual_col, label, filename in OUTCOMES:
    print(f"\n{'='*60}")
    print(f"  {label}")
    print(f"{'='*60}")

    vals = merged[pred_col]
    print(f"  Range: [{vals.min():.4f}, {vals.max():.4f}]")
    print(f"  Mean:  {vals.mean():.4f}  |  Median: {vals.median():.4f}")

    edges = choose_bucket_edges(vals, pred_col)
    print(f"  Bucket edges: {[f'{e:.2f}' for e in edges]}")

    cal = compute_calibration(merged, pred_col, actual_col, edges)
    all_tables[label] = cal

    # Print table
    print(f"\n  {'Bucket':>16s}  {'N':>4s}  {'Pred':>7s}  {'Obs':>7s}  {'Diff':>7s}  {'Flag':>6s}")
    print(f"  {'-'*55}")
    for _, row in cal.iterrows():
        flag = "" if row["reliable"] else " [!]"
        print(
            f"  {row['bucket']:>16s}  {row['n']:4d}  "
            f"{row['avg_predicted']:7.4f}  {row['observed_rate']:7.4f}  "
            f"{row['diff']:+7.4f}{flag}"
        )

    # Overall calibration error (weighted by bucket size, reliable only)
    rel = cal[cal["reliable"]]
    if len(rel) > 0:
        weights = rel["n"].values / rel["n"].sum()
        wce = np.sum(weights * np.abs(rel["diff"].values))
        print(f"\n  Weighted calibration error (reliable buckets): {wce:.4f}")

    # Observed vs predicted overall
    overall_pred = merged[pred_col].mean()
    overall_obs = merged[actual_col].mean()
    print(f"  Overall: predicted {overall_pred:.4f}  vs observed {overall_obs:.4f}  (diff {overall_obs - overall_pred:+.4f})")

    plot_calibration(cal, label, FIGURES_DIR / filename)


# ═══════════════════════════════════════════════════════════════════════════
#  GENERATE findings/calibration.md
# ═══════════════════════════════════════════════════════════════════════════

print(f"\n{'='*60}")
print("  Generating findings/calibration.md")
print(f"{'='*60}")

md_lines = []
md_lines.append("# Calibration Curve Analysis -- Poisson MLE Group Stage Predictions\n")
md_lines.append(f"**Matches scored:** {N} completed group-stage matches  ")
md_lines.append(f"**Outcome distribution:** {merged['outcome'].value_counts().to_dict()}  ")
md_lines.append(f"**Minimum reliable bucket size:** {MIN_BUCKET_SIZE} matches\n")

# Overall outcome summary
overall_pred_draw = merged["p_draw"].mean()
overall_obs_draw = merged["actual_draw"].mean()
overall_pred_home = merged["p_home_win"].mean()
overall_obs_home = merged["actual_home_win"].mean()
overall_pred_away = merged["p_away_win"].mean()
overall_obs_away = merged["actual_away_win"].mean()

md_lines.append("## Overall Predicted vs Observed Rates\n")
md_lines.append("| Outcome | Mean Predicted | Observed Rate | Difference |")
md_lines.append("|---------|---------------|---------------|------------|")
md_lines.append(f"| Home Win | {overall_pred_home:.4f} | {overall_obs_home:.4f} | {overall_obs_home - overall_pred_home:+.4f} |")
md_lines.append(f"| Draw | {overall_pred_draw:.4f} | {overall_obs_draw:.4f} | {overall_obs_draw - overall_pred_draw:+.4f} |")
md_lines.append(f"| Away Win | {overall_pred_away:.4f} | {overall_obs_away:.4f} | {overall_obs_away - overall_pred_away:+.4f} |")
md_lines.append("")

# Per-outcome sections
for pred_col, actual_col, label, filename in OUTCOMES:
    cal = all_tables[label]

    md_lines.append(f"---\n")
    md_lines.append(f"## {label}\n")
    md_lines.append(f"![{label} calibration curve](calibrations/{filename})\n")

    md_lines.append("| Bucket | N | Avg Predicted | Observed Rate | Diff | Reliable? |")
    md_lines.append("|--------|---|--------------|---------------|------|-----------|")
    for _, row in cal.iterrows():
        rel_str = "Yes" if row["reliable"] else "**No**"
        md_lines.append(
            f"| {row['bucket']} | {row['n']} | "
            f"{row['avg_predicted']:.4f} | {row['observed_rate']:.4f} | "
            f"{row['diff']:+.4f} | {rel_str} |"
        )
    md_lines.append("")

    # Commentary
    rel = cal[cal["reliable"]]
    if len(rel) > 0:
        worst = rel.loc[rel["diff"].abs().idxmax()]
        direction = "under-predicted" if worst["diff"] > 0 else "over-predicted"
        md_lines.append(
            f"**Largest miscalibration** (reliable buckets): "
            f"bucket {worst['bucket']} -- model {direction} by "
            f"{abs(worst['diff']):.1%} ({worst['n']} matches).\n"
        )

    md_lines.append("")

# ── Draw-specific hypothesis section ────────────────────────────────────
md_lines.append("---\n")
md_lines.append("## Draw-Underestimation Hypothesis\n")

draw_cal = all_tables["Draw (p_draw)"]
draw_rel = draw_cal[draw_cal["reliable"]]

n_draws = merged["actual_draw"].sum()
n_total = len(merged)
draw_rate = n_draws / n_total
mean_pred_draw = merged["p_draw"].mean()
diff = draw_rate - mean_pred_draw

if diff > 0.03:
    verdict = (
        f"**Confirmed.** The model predicted an average draw probability of "
        f"{mean_pred_draw:.1%}, but draws occurred {draw_rate:.1%} of the time "
        f"(+{diff:.1%} gap). "
    )
elif diff > 0:
    verdict = (
        f"**Mild support.** The model predicted an average draw probability of "
        f"{mean_pred_draw:.1%}, but draws occurred {draw_rate:.1%} of the time "
        f"(+{diff:.1%} gap). The gap exists but is small enough that sampling noise "
        f"could explain it with only {n_total} matches. "
    )
else:
    verdict = (
        f"**Not supported.** The model predicted an average draw probability of "
        f"{mean_pred_draw:.1%} and draws occurred {draw_rate:.1%} of the time. "
        f"The model actually slightly over-predicted draws. "
    )

# Check per-bucket pattern
if len(draw_rel) > 0:
    n_under = (draw_rel["diff"] > 0.01).sum()
    n_over = (draw_rel["diff"] < -0.01).sum()
    if n_under > n_over:
        verdict += (
            f"The bucket-level pattern reinforces this: {n_under} of "
            f"{len(draw_rel)} reliable buckets show the observed draw rate "
            f"exceeding the predicted rate."
        )
    elif n_over > n_under:
        verdict += (
            f"However, at the bucket level, {n_over} of {len(draw_rel)} "
            f"reliable buckets show the model over-predicting draws, "
            f"complicating the narrative."
        )
    else:
        verdict += (
            f"At the bucket level, the pattern is mixed -- equal numbers of "
            f"buckets show over- and under-prediction."
        )

md_lines.append(verdict + "\n")

# Caveats
md_lines.append("\n> [!NOTE]\n")
md_lines.append(
    f"> With only {n_total} group-stage matches, individual bucket sizes are small "
    f"({draw_cal['n'].min()}-{draw_cal['n'].max()} matches per bucket). "
    f"These calibration curves are directional indicators, not statistically "
    f"conclusive. A 10-percentage-point deviation in a bucket of 15 matches "
    f"is well within the sampling noise range.\n"
)

md_lines.append("\n---\n")
md_lines.append(f"*Generated by `working/calibration.py` on {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}*\n")

FINDINGS_MD.write_text("\n".join(md_lines), encoding="utf-8")
print(f"  Saved findings/calibration.md")
print("\nDone.")
