"""
working/calibration_fixed_buckets.py
====================================
Calibration analysis with user-specified coarser buckets:
  0-20%, 20-30%, 30-40%, 40%+

Keeps the existing auto-bucketed analysis untouched; this is a separate view.

Outputs:
  findings/calibrations/fixed_home_win.png
  findings/calibrations/fixed_away_win.png
  findings/calibrations/fixed_draw.png
  findings/calibration_fixed_buckets.md
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

# ── paths ────────────────────────────────────────────────────────────────
PREDICTIONS_PATH = PROJECT_ROOT / "predictions/group_stage_predictions_new.csv"
RESULTS_PATH = PROJECT_ROOT / "results/actual_outcomes.csv"
FIGURES_DIR = PROJECT_ROOT / "findings/calibrations"
FINDINGS_MD = PROJECT_ROOT / "findings/calibration_fixed_buckets.md"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# ── load & merge ─────────────────────────────────────────────────────────
predictions = pd.read_csv(PREDICTIONS_PATH)
results = pd.read_csv(RESULTS_PATH)
merged = predictions.merge(results, on=["match_id", "date"], how="inner")

merged["actual_home_win"] = (merged["outcome"] == "home_win").astype(int)
merged["actual_draw"] = (merged["outcome"] == "draw").astype(int)
merged["actual_away_win"] = (merged["outcome"] == "away_win").astype(int)

N = len(merged)
FIXED_EDGES = [0.0, 0.20, 0.30, 0.40, 1.0]
MIN_BUCKET_SIZE = 5

print(f"Merged {N} matches | Fixed buckets: {FIXED_EDGES}")
print(f"Outcome distribution: {merged['outcome'].value_counts().to_dict()}")


# ═══════════════════════════════════════════════════════════════════════════

def compute_calibration(df, pred_col, actual_col, edges):
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
        sizes = rel["n"].values * 8
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
        sizes = unrel["n"].values * 8
        ax.scatter(unrel["avg_predicted"], unrel["observed_rate"],
                   s=sizes, color="#f59e0b", edgecolors="white", linewidth=1.5,
                   zorder=5, marker="D", label=f"Unreliable (n < {MIN_BUCKET_SIZE})")
        for _, row in unrel.iterrows():
            ax.annotate(f"n={row['n']}", (row["avg_predicted"], row["observed_rate"]),
                        textcoords="offset points", xytext=(8, -8),
                        fontsize=9, color="#92400e")

    ax.set_xlabel("Mean Predicted Probability", fontsize=12)
    ax.set_ylabel("Observed Frequency", fontsize=12)
    ax.set_title(f"Calibration: {label}\n(72 group-stage matches, fixed buckets)",
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
#  RUN
# ═══════════════════════════════════════════════════════════════════════════

OUTCOMES = [
    ("p_draw",     "actual_draw",     "Draw",      "fixed_draw.png"),
    ("p_home_win", "actual_home_win", "Home Win",  "fixed_home_win.png"),
    ("p_away_win", "actual_away_win", "Away Win",  "fixed_away_win.png"),
]

all_tables = {}

for pred_col, actual_col, label, filename in OUTCOMES:
    print(f"\n{'='*55}")
    print(f"  {label}  (buckets: 0-20%, 20-30%, 30-40%, 40%+)")
    print(f"{'='*55}")

    cal = compute_calibration(merged, pred_col, actual_col, FIXED_EDGES)
    all_tables[label] = cal

    print(f"\n  {'Bucket':>14s}  {'N':>4s}  {'Pred':>7s}  {'Obs':>7s}  {'Diff':>7s}")
    print(f"  {'-'*48}")
    for _, row in cal.iterrows():
        flag = "" if row["reliable"] else " [!]"
        print(f"  {row['bucket']:>14s}  {row['n']:4d}  "
              f"{row['avg_predicted']:7.4f}  {row['observed_rate']:7.4f}  "
              f"{row['diff']:+7.4f}{flag}")

    overall_pred = merged[pred_col].mean()
    overall_obs = merged[actual_col].mean()
    print(f"\n  Overall: pred {overall_pred:.4f}  obs {overall_obs:.4f}  diff {overall_obs - overall_pred:+.4f}")

    plot_calibration(cal, label, FIGURES_DIR / filename)


# ═══════════════════════════════════════════════════════════════════════════
#  GENERATE findings/calibration_fixed_buckets.md
# ═══════════════════════════════════════════════════════════════════════════

md = []
md.append("# Calibration Analysis -- Fixed Buckets (0-20%, 20-30%, 30-40%, 40%+)\n")
md.append(f"**Matches:** {N} completed group-stage matches  ")
md.append(f"**Outcomes:** {merged['outcome'].value_counts().to_dict()}  ")
md.append(f"**Bucket edges:** {', '.join(f'{e:.0%}' for e in FIXED_EDGES)}\n")

# Overall table
md.append("## Overall Predicted vs Observed\n")
md.append("| Outcome | Predicted | Observed | Diff |")
md.append("|---------|-----------|----------|------|")
for pred_col, actual_col, label, _ in OUTCOMES:
    p = merged[pred_col].mean()
    o = merged[actual_col].mean()
    md.append(f"| {label} | {p:.1%} | {o:.1%} | {o-p:+.1%} |")
md.append("")

# Per-outcome
for pred_col, actual_col, label, filename in OUTCOMES:
    cal = all_tables[label]
    md.append(f"---\n\n## {label}\n")
    md.append(f"![{label} calibration](calibrations/{filename})\n")
    md.append("| Bucket | N | Avg Predicted | Observed | Diff | Reliable? |")
    md.append("|--------|---|--------------|----------|------|-----------|")
    for _, row in cal.iterrows():
        rel = "Yes" if row["reliable"] else "**No**"
        md.append(f"| {row['bucket']} | {row['n']} | "
                  f"{row['avg_predicted']:.1%} | {row['observed_rate']:.1%} | "
                  f"{row['diff']:+.1%} | {rel} |")
    md.append("")

    # Interpretation
    rel = cal[cal["reliable"]]
    if len(rel) > 0:
        worst = rel.loc[rel["diff"].abs().idxmax()]
        direction = "under-predicted" if worst["diff"] > 0 else "over-predicted"
        md.append(f"Largest miscalibration: **{worst['bucket']}** -- "
                  f"{direction} by {abs(worst['diff']):.1%} (n={worst['n']}).\n")

# Draw hypothesis
md.append("---\n\n## Draw-Underestimation Assessment\n")
draw_cal = all_tables["Draw"]
p_draw = merged["p_draw"].mean()
o_draw = merged["actual_draw"].mean()
draw_rel = draw_cal[draw_cal["reliable"]]
n_under = (draw_rel["diff"] > 0.01).sum() if len(draw_rel) > 0 else 0

md.append(f"The model predicted draws at {p_draw:.1%} on average; "
          f"they occurred {o_draw:.1%} of the time ({o_draw - p_draw:+.1%}).\n")

if len(draw_rel) > 0:
    md.append(f"With these coarser buckets, **{n_under} of {len(draw_rel)}** "
              f"reliable draw buckets show observed > predicted, "
              f"{'reinforcing' if n_under > len(draw_rel)/2 else 'weakening'} "
              f"the draw-underestimation finding.\n")

md.append("\n> [!NOTE]\n> The away-win curve is also revealing: the model "
          "over-predicts away wins below ~40%, meaning probability mass that "
          "should go to draws (or home wins) is leaking into away-win predictions. "
          "This is the flip side of draw underestimation.\n")

md.append(f"\n---\n\n*Generated by `working/calibration_fixed_buckets.py` "
          f"on {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}*\n")

FINDINGS_MD.write_text("\n".join(md), encoding="utf-8")
print(f"\n  Saved findings/calibration_fixed_buckets.md")
print("Done.")
