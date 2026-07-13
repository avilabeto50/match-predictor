"""
04_elo_gap_baseline.py
======================
Run from the project root (or notebooks/ — paths are resolved relative to this
script's location).

Outputs:
  data/processed/elo_match_log.csv   – per-match Elo snapshot (before update)
  data/processed/elo_ratings_new.csv – final Elo ratings (unchanged output)
  Console: Elo-gap bucket analysis + full baseline comparison table
"""

import sys, os
from pathlib import Path

# ── resolve project root regardless of cwd ─────────────────────────────────
SCRIPT_DIR = Path(__file__).resolve().parent          # notebooks/
PROJECT_ROOT = SCRIPT_DIR.parent                      # match-predictor/
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import pandas as pd
import numpy as np

# ═══════════════════════════════════════════════════════════════════════════
# 1. LOAD DATA
# ═══════════════════════════════════════════════════════════════════════════

matches = pd.read_csv(PROJECT_ROOT / "data/processed/matches_filtered.csv")
teams_df = pd.read_csv(PROJECT_ROOT / "data/processed/wc2026_teams.csv")
teams = teams_df["team"].tolist()

predictions_df = pd.read_csv(PROJECT_ROOT / "predictions/group_stage_predictions_new.csv")
results_df = pd.read_csv(PROJECT_ROOT / "results/actual_outcomes.csv")

print(f"Loaded {len(matches)} historical matches for {len(teams)} teams")
print(f"Date range: {matches['date'].min()} -> {matches['date'].max()}")
print(f"2026 group-stage predictions: {len(predictions_df)}")
print(f"2026 actual outcomes:         {len(results_df)}")

# ═══════════════════════════════════════════════════════════════════════════
# 2. K-FACTOR MAP  (exact copy from notebook 02)
# ═══════════════════════════════════════════════════════════════════════════

K_FACTOR_MAP = {
    "FIFA World Cup": 60,
    "Confederations Cup": 55,
    "UEFA Euro": 50,
    "Copa América": 50,
    "African Cup of Nations": 50,
    "AFC Asian Cup": 50,
    "Gold Cup": 50,
    "Arab Cup": 45,
    "Gulf Cup": 45,
    "EAFF Championship": 45,
    "WAFF Championship": 45,
    "COSAFA Cup": 40,
    "CAFA Nations Cup": 40,
    "UEFA Nations League": 45,
    "CONCACAF Nations League": 45,
    "FIFA World Cup qualification": 40,
    "UEFA Euro qualification": 35,
    "African Cup of Nations qualification": 35,
    "FIFA Series": 30,
    "Superclásico de las Américas": 35,
    "Kirin Challenge Cup": 20,
    "Kirin Cup": 20,
    "Al Ain International Cup": 20,
    "Canadian Shield": 20,
    "Soccer Ashes": 20,
    "Friendly": 20,
}


def get_k_factor(tournament):
    if tournament not in K_FACTOR_MAP:
        raise ValueError(
            f"Unknown tournament: {tournament!r}. "
            "Add it to K_FACTOR_MAP with an appropriate K-factor."
        )
    return K_FACTOR_MAP[tournament]


# ═══════════════════════════════════════════════════════════════════════════
# 3. ELO LOOP — with pre-match snapshot logging
# ═══════════════════════════════════════════════════════════════════════════

elo_ratings = {team: 1500 for team in teams}
matches_sorted = matches.sort_values("date").reset_index(drop=True)

match_log = []  # ← NEW: capture pre-match Elo for every historical match

for _, row in matches_sorted.iterrows():
    home_team = row["home_team"]
    away_team = row["away_team"]
    home_goals = row["home_score"]
    away_goals = row["away_score"]

    # Current ratings *before* this match
    home_rating = elo_ratings[home_team]
    away_rating = elo_ratings[away_team]

    # Determine actual outcome
    if home_goals > away_goals:
        outcome = "home_win"
        actual_home, actual_away = 1, 0
    elif home_goals == away_goals:
        outcome = "draw"
        actual_home, actual_away = 0.5, 0.5
    else:
        outcome = "away_win"
        actual_home, actual_away = 0, 1

    # Log the snapshot *before* update
    match_log.append(
        {
            "date": row["date"],
            "home_team": home_team,
            "away_team": away_team,
            "home_elo_before": home_rating,
            "away_elo_before": away_rating,
            "outcome": outcome,
            "tournament": row["tournament"],
        }
    )

    # Standard Elo update
    expected_home = 1 / (1 + 10 ** ((away_rating - home_rating) / 400))
    expected_away = 1 - expected_home
    k = get_k_factor(row["tournament"])
    elo_ratings[home_team] = home_rating + k * (actual_home - expected_home)
    elo_ratings[away_team] = away_rating + k * (actual_away - expected_away)

match_log_df = pd.DataFrame(match_log)
match_log_df.to_csv(
    PROJECT_ROOT / "data/processed/elo_match_log.csv", index=False
)

# Also save final ratings (unchanged behaviour)
elo_final = (
    pd.DataFrame(list(elo_ratings.items()), columns=["team", "elo_rating"])
    .sort_values("elo_rating", ascending=False)
    .reset_index(drop=True)
)
elo_final.to_csv(
    PROJECT_ROOT / "data/processed/elo_ratings_new.csv", index=False
)

print(f"\nProcessed {len(matches_sorted)} matches")
print(f"Saved elo_match_log.csv  ({len(match_log_df)} rows)")
print(f"Saved elo_ratings_new.csv ({len(elo_final)} teams)")

# Top 10
print("\nTop 10 teams by final Elo rating:")
for _, r in elo_final.head(10).iterrows():
    print(f"  {r['team']:25s} {r['elo_rating']:7.1f}")

# ═══════════════════════════════════════════════════════════════════════════
# 4. BUILD ELO-GAP BASELINE (training data only = pre-2026 matches)
# ═══════════════════════════════════════════════════════════════════════════

# Training set: everything before the WC (date < 2026-06-11)
train = match_log_df[match_log_df["date"] < "2026-06-11"].copy()
print(f"\n{'='*70}")
print(f"ELO-GAP BASELINE — Training on {len(train)} pre-tournament matches")
print(f"{'='*70}")

# Compute absolute elo gap + favored/underdog outcome
train["elo_gap"] = (train["home_elo_before"] - train["away_elo_before"]).abs()
train["home_is_favored"] = train["home_elo_before"] >= train["away_elo_before"]

# Map outcome to favored/draw/underdog
def map_outcome(row):
    if row["outcome"] == "draw":
        return "draw"
    elif row["outcome"] == "home_win":
        return "favored_win" if row["home_is_favored"] else "underdog_win"
    else:  # away_win
        return "favored_win" if not row["home_is_favored"] else "underdog_win"

train["fdu_outcome"] = train.apply(map_outcome, axis=1)

# ── Choose bin edges based on actual distribution ────────────────────────
print("\nElo-gap distribution (training data):")
print(f"  Count: {len(train)}")
print(f"  Mean:  {train['elo_gap'].mean():.1f}")
print(f"  Median:{train['elo_gap'].median():.1f}")
for pct in [25, 50, 75, 90, 95, 99]:
    print(f"  P{pct:02d}:   {train['elo_gap'].quantile(pct/100):.1f}")
print(f"  Max:   {train['elo_gap'].max():.1f}")

# Bin edges: 0–50, 50–100, 100–150, 150–200, 200+
# (chosen to give roughly balanced buckets given typical WC-team elo spreads)
BIN_EDGES = [0, 50, 100, 150, 200, float("inf")]
BIN_LABELS = ["0–50", "50–100", "100–150", "150–200", "200+"]

train["gap_bucket"] = pd.cut(
    train["elo_gap"], bins=BIN_EDGES, labels=BIN_LABELS, right=False
)

# ── Bucket frequency table ───────────────────────────────────────────────
print("\n-- Bucket counts & outcome frequencies (training data) --")
print(f"{'Bucket':>12s}  {'N':>5s}  {'Fav Win':>8s}  {'Draw':>8s}  {'Dog Win':>8s}")
print("-" * 55)

bucket_probs = {}
for label in BIN_LABELS:
    subset = train[train["gap_bucket"] == label]
    n = len(subset)
    if n == 0:
        bucket_probs[label] = {"favored_win": 1 / 3, "draw": 1 / 3, "underdog_win": 1 / 3}
        print(f"{label:>12s}  {n:5d}  {'(empty — using 1/3 each)':>30s}")
        continue
    counts = subset["fdu_outcome"].value_counts()
    p_fav = counts.get("favored_win", 0) / n
    p_draw = counts.get("draw", 0) / n
    p_dog = counts.get("underdog_win", 0) / n
    bucket_probs[label] = {"favored_win": p_fav, "draw": p_draw, "underdog_win": p_dog}
    flag = " [!] LOW N" if n < 30 else ""
    print(f"{label:>12s}  {n:5d}  {p_fav:8.4f}  {p_draw:8.4f}  {p_dog:8.4f}{flag}")

# ── Sanity check ─────────────────────────────────────────────────────────
low_n_buckets = [
    label for label in BIN_LABELS
    if 0 < len(train[train["gap_bucket"] == label]) < 30
]
if low_n_buckets:
    print(f"\n[!] WARNING: Buckets with < 30 matches (unstable estimates): {low_n_buckets}")
else:
    print("\n[OK] All buckets have >= 30 matches -- frequency estimates should be stable.")

# ═══════════════════════════════════════════════════════════════════════════
# 5. APPLY ELO-GAP BASELINE TO 2026 GROUP-STAGE MATCHES
# ═══════════════════════════════════════════════════════════════════════════

# For 2026 predictions we use each team's *current* (post-training) Elo.
# This is exactly what a real predictor would have at tournament start.

print(f"\n{'='*70}")
print("APPLYING ELO-GAP BASELINE TO 2026 GROUP STAGE")
print(f"{'='*70}")

elo_gap_preds = []

for _, row in predictions_df.iterrows():
    home = row["home_team"]
    away = row["away_team"]

    home_elo = elo_ratings[home]
    away_elo = elo_ratings[away]
    gap = abs(home_elo - away_elo)
    home_is_fav = home_elo >= away_elo

    # Find bucket
    bucket = None
    for i in range(len(BIN_EDGES) - 1):
        if BIN_EDGES[i] <= gap < BIN_EDGES[i + 1]:
            bucket = BIN_LABELS[i]
            break

    probs = bucket_probs[bucket]

    # Map favored/draw/underdog → home/draw/away
    if home_is_fav:
        p_home_win = probs["favored_win"]
        p_draw = probs["draw"]
        p_away_win = probs["underdog_win"]
    else:
        p_home_win = probs["underdog_win"]
        p_draw = probs["draw"]
        p_away_win = probs["favored_win"]

    elo_gap_preds.append(
        {
            "match_id": row["match_id"],
            "date": row["date"],
            "home_team": home,
            "away_team": away,
            "home_elo": round(home_elo, 1),
            "away_elo": round(away_elo, 1),
            "elo_gap": round(gap, 1),
            "bucket": bucket,
            "favored": home if home_is_fav else away,
            "p_home_win": p_home_win,
            "p_draw": p_draw,
            "p_away_win": p_away_win,
        }
    )

elo_gap_df = pd.DataFrame(elo_gap_preds)

# Quick sanity: probabilities must sum to 1
prob_sums = elo_gap_df["p_home_win"] + elo_gap_df["p_draw"] + elo_gap_df["p_away_win"]
assert np.allclose(prob_sums, 1.0), f"Probabilities don't sum to 1: {prob_sums.describe()}"

print("\nSample Elo-gap predictions (first 10):")
print(
    elo_gap_df[
        ["date", "home_team", "away_team", "elo_gap", "bucket", "favored",
         "p_home_win", "p_draw", "p_away_win"]
    ].head(10).to_string(index=False)
)

# Save for reference
elo_gap_df.to_csv(
    PROJECT_ROOT / "predictions/elo_gap_baseline_predictions.csv", index=False
)
print(f"\nSaved predictions/elo_gap_baseline_predictions.csv")

# ═══════════════════════════════════════════════════════════════════════════
# 6. SCORE ALL BASELINES + POISSON MODEL
# ═══════════════════════════════════════════════════════════════════════════

# Reuse the exact scoring functions from app.py
def brier_score(preds, results):
    """Calculate Brier score for all completed matches."""
    if len(results) == 0:
        return None
    merged = preds.merge(results, on=["match_id", "date"], how="inner")
    p_win = merged["p_home_win"].values
    p_draw = merged["p_draw"].values
    p_loss = merged["p_away_win"].values
    outcomes = merged["outcome"].values
    o_win = (outcomes == "home_win").astype(int)
    o_draw = (outcomes == "draw").astype(int)
    o_loss = (outcomes == "away_win").astype(int)
    return float(np.mean((p_win - o_win) ** 2 + (p_draw - o_draw) ** 2 + (p_loss - o_loss) ** 2))


def log_loss_fn(preds, results):
    """Calculate log-loss for all completed matches."""
    if len(results) == 0:
        return None
    merged = preds.merge(results, on=["match_id", "date"], how="inner")
    p_win = np.clip(merged["p_home_win"].values, 1e-15, 1 - 1e-15)
    p_draw = np.clip(merged["p_draw"].values, 1e-15, 1 - 1e-15)
    p_loss = np.clip(merged["p_away_win"].values, 1e-15, 1 - 1e-15)
    outcomes = merged["outcome"].values
    o_win = (outcomes == "home_win").astype(int)
    o_draw = (outcomes == "draw").astype(int)
    o_loss = (outcomes == "away_win").astype(int)
    return float(-np.mean(o_win * np.log(p_win) + o_draw * np.log(p_draw) + o_loss * np.log(p_loss)))


# Build all baseline prediction DataFrames
uniform = predictions_df[["match_id", "date"]].copy()
uniform["p_home_win"] = 1 / 3
uniform["p_draw"] = 1 / 3
uniform["p_away_win"] = 1 / 3

# Base-rate baseline uses training-set outcome proportions
# Training outcomes = historical matches from matches_filtered (pre-2026)
train_outcomes = match_log_df[match_log_df["date"] < "2026-06-11"]["outcome"]
rates = train_outcomes.value_counts(normalize=True)
base_rate = predictions_df[["match_id", "date"]].copy()
base_rate["p_home_win"] = rates.get("home_win", 0)
base_rate["p_draw"] = rates.get("draw", 0)
base_rate["p_away_win"] = rates.get("away_win", 0)

# Elo-gap baseline (just the p_ columns + match_id + date for scoring)
elo_gap_for_scoring = elo_gap_df[["match_id", "date", "p_home_win", "p_draw", "p_away_win"]].copy()

# Poisson model predictions (already saved)
poisson = predictions_df[["match_id", "date", "p_home_win", "p_draw", "p_away_win"]].copy()

# ── Score everything ─────────────────────────────────────────────────────
models = {
    "Uniform (1/3 each)": uniform,
    "Base-Rate (hist. %)": base_rate,
    "Elo-Gap Baseline":    elo_gap_for_scoring,
    "Poisson MLE":         poisson,
}

print(f"\n{'='*70}")
print("BASELINE COMPARISON — 2026 GROUP STAGE")
n_matched = len(predictions_df.merge(results_df, on=["match_id", "date"], how="inner"))
print(f"Scored on {n_matched} completed matches")
print(f"{'='*70}")
print(f"\n{'Model':>25s}  {'Brier':>8s}  {'Log Loss':>10s}")
print("-" * 50)

for name, pred_df in models.items():
    bs = brier_score(pred_df, results_df)
    ll = log_loss_fn(pred_df, results_df)
    print(f"{name:>25s}  {bs:8.4f}  {ll:10.4f}")

# ── Also print base-rate training proportions for reference ──────────────
print(f"\nTraining-set outcome rates (N={len(train_outcomes)}):")
for outcome in ["home_win", "draw", "away_win"]:
    print(f"  {outcome:12s}  {rates.get(outcome, 0):.4f}")

# ── Elo gap distribution in 2026 matches for context ─────────────────────
print(f"\nElo-gap distribution in 2026 group-stage matches:")
print(f"  Mean:   {elo_gap_df['elo_gap'].mean():.1f}")
print(f"  Median: {elo_gap_df['elo_gap'].median():.1f}")
print(f"  Min:    {elo_gap_df['elo_gap'].min():.1f}")
print(f"  Max:    {elo_gap_df['elo_gap'].max():.1f}")
print(f"\n  Bucket distribution:")
print(elo_gap_df["bucket"].value_counts().sort_index().to_string())
