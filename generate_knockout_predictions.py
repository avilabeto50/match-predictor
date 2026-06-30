"""
generate_knockout_predictions.py
═════════════════════════════════════════════════════════════════════════════
Knockout Stage Prediction Pipeline for the 2026 FIFA World Cup.

Objectives:
  1. FROZEN MODEL  — Re-fit on historical data only (matches_filtered.csv).
                     Generate Round of 32 predictions using these frozen params.
                     Save: predictions/knockout_predictions_frozen_model.csv

  2. RE-FIT MODEL  — Re-fit on historical + 72 group stage results combined.
                     Generate Round of 32 predictions using updated params.
                     Save: predictions/knockout_predictions.csv

  3. DIAGNOSTICS   — Compare parameters, log-likelihood improvement, and
                     count prediction flips between the two models.
                     Save: predictions/knockout_prediction_diff.md

Usage:
  python generate_knockout_predictions.py

Do NOT modify predictions/group_stage_predictions.csv.
Do NOT modify results/actual_outcomes.csv (read-only source).
"""

import sys
import os
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime

# Allow importing from src/
sys.path.insert(0, str(Path(__file__).parent / "src"))
from poisson_model import (
    load_and_weight_data,
    build_team_index,
    fit_poisson_model,
    predict_match,
    neg_log_likelihood,
)

# ─────────────────────────────────────────────────────────────────────────────
# PATHS
# ─────────────────────────────────────────────────────────────────────────────
BASE = Path(__file__).parent
HISTORICAL_DATA   = BASE / "data/processed/matches_filtered.csv"
GROUP_RESULTS     = BASE / "results/actual_outcomes.csv"
OUTPUT_DIR        = BASE / "predictions"
PARAMS_FROZEN     = BASE / "data/processed/team_ratings_frozen.csv"
PARAMS_REFITTED   = BASE / "data/processed/team_ratings_knockout_refitted.csv"
OUTPUT_FROZEN     = OUTPUT_DIR / "knockout_predictions_frozen_model.csv"
OUTPUT_REFITTED   = OUTPUT_DIR / "knockout_predictions.csv"
OUTPUT_DIFF       = OUTPUT_DIR / "knockout_prediction_diff.md"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ─────────────────────────────────────────────────────────────────────────────
# ROUND OF 32 FIXTURES (hardcoded from official bracket)
# Canonical team names matching the Poisson model's team index.
# match_id continues from 73 (group stage was 1-72).
# ─────────────────────────────────────────────────────────────────────────────
R32_FIXTURES = [
    # match_id  date          home_team                away_team                   location
    (73,  "2026-06-28", "Germany",          "Paraguay",                "Los Angeles Stadium"),
    (74,  "2026-06-29", "France",           "Sweden",                  "Boston Stadium"),
    (75,  "2026-06-29", "South Africa",     "Canada",                  "Estadio Monterrey"),
    (76,  "2026-06-29", "Netherlands",      "Morocco",                 "Houston Stadium"),
    (77,  "2026-06-30", "Portugal",         "Croatia",                 "New York New Jersey Stadium"),
    (78,  "2026-06-30", "Spain",            "Austria",                 "Dallas Stadium"),
    (79,  "2026-06-30", "United States",    "Bosnia and Herzegovina",  "Mexico City Stadium"),
    (80,  "2026-07-01", "Belgium",          "Senegal",                 "Atlanta Stadium"),
    (81,  "2026-07-01", "Brazil",           "Japan",                   "San Francisco Bay Area Stadium"),
    (82,  "2026-07-01", "Côte d'Ivoire",    "Norway",                  "Seattle Stadium"),
    (83,  "2026-07-02", "Mexico",           "Ecuador",                 "Toronto Stadium"),
    (84,  "2026-07-02", "England",          "DR Congo",                "Los Angeles Stadium"),
    (85,  "2026-07-02", "Argentina",        "Cabo Verde",              "BC Place Vancouver"),
    (86,  "2026-07-03", "Australia",        "Egypt",                   "Miami Stadium"),
    (87,  "2026-07-03", "Switzerland",      "Algeria",                 "Kansas City Stadium"),
    (88,  "2026-07-03", "Colombia",         "Ghana",                   "Dallas Stadium"),
]

# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def build_prediction_row(match_id, date, home_team, away_team, location,
                          mu, alphas, betas, teams, team_to_idx) -> dict:
    """Call predict_match and format a single output row."""
    pred = predict_match(home_team, away_team, mu, alphas, betas, teams, team_to_idx)

    p_h = pred["p_win_a"]
    p_d = pred["p_draw"]
    p_a = pred["p_win_b"]

    if p_h > p_d and p_h > p_a:
        winner = home_team
    elif p_a > p_h and p_a > p_d:
        winner = away_team
    else:
        winner = "draw"

    top = pred["top_scorelines"][0]
    return {
        "match_id":        match_id,
        "date":            date,
        "round":           "Round of 32",
        "home_team":       home_team,
        "away_team":       away_team,
        "p_home_win":      p_h,
        "p_draw":          p_d,
        "p_away_win":      p_a,
        "predicted_winner": winner,
        "top_scoreline":   f"{int(top[0])}-{int(top[1])}",
        "location":        location,
    }


def compute_nll(df, mu, alphas, betas, team_to_idx, n_teams) -> float:
    """Compute the raw negative log-likelihood (no regularization) for reporting."""
    params = np.concatenate([[mu], alphas, betas])
    return neg_log_likelihood(params, n_teams, df, team_to_idx)


def save_team_ratings(teams, alphas, betas, mu, path: Path) -> None:
    """Persist fitted parameters as a CSV for reference."""
    rows = [
        {"team": t, "attack": alphas[i], "defense": betas[i], "mu": mu}
        for i, t in enumerate(teams)
    ]
    pd.DataFrame(rows).to_csv(path, index=False)
    print(f"  ✓ Saved team ratings to {path.name}")


def generate_predictions(fixtures, mu, alphas, betas, teams, team_to_idx,
                          label: str) -> pd.DataFrame:
    """Generate predictions for all fixtures using the given model parameters."""
    rows = []
    failed = []
    for match_id, date, home_team, away_team, location in fixtures:
        try:
            row = build_prediction_row(
                match_id, date, home_team, away_team, location,
                mu, alphas, betas, teams, team_to_idx
            )
            rows.append(row)
        except KeyError as e:
            failed.append((home_team, away_team, str(e)))
            print(f"  ⚠ ERROR [{label}]: Cannot predict {home_team} vs {away_team} — {e}")

    if failed:
        print(f"\n  ⚠ {len(failed)} fixture(s) failed to generate predictions.")
    else:
        print(f"  ✓ {len(rows)} predictions generated ({label})")

    return pd.DataFrame(rows)


def validate_predictions(df: pd.DataFrame, label: str) -> bool:
    """Run sanity checks on a predictions dataframe. Returns True if all pass."""
    print(f"\n── Validating: {label} ──")
    ok = True

    if len(df) != 16:
        print(f"  ✗ Expected 16 rows, got {len(df)}")
        ok = False
    else:
        print(f"  ✓ Row count: {len(df)}")

    nan_cols = df[["p_home_win", "p_draw", "p_away_win"]].isnull().sum()
    if nan_cols.any():
        print(f"  ✗ NaN values found:\n{nan_cols}")
        ok = False
    else:
        print(f"  ✓ No NaN values in probability columns")

    prob_sums = df["p_home_win"] + df["p_draw"] + df["p_away_win"]
    max_dev = (prob_sums - 1.0).abs().max()
    if max_dev > 1e-4:
        print(f"  ✗ Probability sum deviation too large: max={max_dev:.6f}")
        ok = False
    else:
        print(f"  ✓ Probability sums ≈ 1.0 (max deviation: {max_dev:.2e})")

    # Verify predicted_winner is consistent with the probabilities
    for _, row in df.iterrows():
        probs = {"home": row["p_home_win"], "draw": row["p_draw"], "away": row["p_away_win"]}
        best = max(probs, key=probs.get)
        expected = {
            "home": row["home_team"],
            "draw": "draw",
            "away": row["away_team"],
        }[best]
        if row["predicted_winner"] != expected:
            print(f"  ✗ predicted_winner mismatch for match {row['match_id']}: "
                  f"got {row['predicted_winner']!r}, expected {expected!r}")
            ok = False
    if ok:
        print(f"  ✓ predicted_winner consistent with probabilities")

    print(f"  {'✓ All checks passed' if ok else '✗ Some checks FAILED'} for {label}")
    return ok


# ─────────────────────────────────────────────────────────────────────────────
# DIFF REPORT
# ─────────────────────────────────────────────────────────────────────────────

def build_diff_report(
    frozen_df: pd.DataFrame,
    refitted_df: pd.DataFrame,
    teams,
    alphas_frozen, betas_frozen, mu_frozen,
    alphas_new, betas_new, mu_new,
    team_to_idx,
    n_teams,
    hist_df,
    combined_df,
) -> str:
    """Build a markdown diff report comparing frozen vs re-fitted model outputs."""

    nll_hist     = compute_nll(hist_df,     mu_frozen, alphas_frozen, betas_frozen, team_to_idx, n_teams)
    nll_combined = compute_nll(combined_df, mu_new,    alphas_new,    betas_new,    team_to_idx, n_teams)
    nll_frozen_on_combined = compute_nll(combined_df, mu_frozen, alphas_frozen, betas_frozen, team_to_idx, n_teams)
    ll_improvement = nll_frozen_on_combined - nll_combined  # positive = better

    # Parameter changes
    alpha_changes = []
    beta_changes  = []
    for i, team in enumerate(teams):
        da = alphas_new[i] - alphas_frozen[i]
        db = betas_new[i]  - betas_frozen[i]
        alpha_changes.append((abs(da), da, team, alphas_frozen[i], alphas_new[i]))
        beta_changes.append((abs(db),  db, team, betas_frozen[i],  betas_new[i]))

    alpha_changes.sort(reverse=True)
    beta_changes.sort(reverse=True)

    top5_alpha = alpha_changes[:5]
    top5_beta  = beta_changes[:5]

    # Prediction flips
    flips = []
    for (_, fr), (_, nr) in zip(frozen_df.iterrows(), refitted_df.iterrows()):
        assert fr["match_id"] == nr["match_id"]
        if fr["predicted_winner"] != nr["predicted_winner"]:
            flips.append({
                "match_id":   fr["match_id"],
                "home":       fr["home_team"],
                "away":       fr["away_team"],
                "frozen":     fr["predicted_winner"],
                "refitted":   nr["predicted_winner"],
                "frozen_ph":  fr["p_home_win"],
                "frozen_pa":  fr["p_away_win"],
                "new_ph":     nr["p_home_win"],
                "new_pa":     nr["p_away_win"],
            })

    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    lines = [
        "# Knockout Stage Model Comparison Report",
        f"*Generated: {ts}*",
        "",
        "## Overview",
        "",
        f"- **Frozen model**: fitted on {len(hist_df):,} historical matches (2014–2025)",
        f"- **Re-fitted model**: fitted on {len(combined_df):,} matches (historical + 72 group stage results)",
        f"- **Group stage weight**: 1.0 (World Cup tier)",
        "",
        "## Log-Likelihood Comparison",
        "",
        "| Metric | Value |",
        "|--------|-------|",
        f"| Frozen model NLL (on historical data) | {nll_hist:.2f} |",
        f"| Frozen model NLL (on combined data) | {nll_frozen_on_combined:.2f} |",
        f"| Re-fitted model NLL (on combined data) | {nll_combined:.2f} |",
        f"| Log-likelihood improvement from re-fit | **{ll_improvement:.2f}** ({'↑ better' if ll_improvement > 0 else '↓ worse'}) |",
        "",
        "## Model Parameters",
        "",
        f"| Parameter | Frozen | Re-Fitted | Change |",
        "|-----------|--------|-----------|--------|",
        f"| μ (global rate) | {mu_frozen:.4f} | {mu_new:.4f} | {mu_new - mu_frozen:+.4f} |",
        f"| α mean (attack) | {np.mean(alphas_frozen):.4f} | {np.mean(alphas_new):.4f} | {np.mean(alphas_new) - np.mean(alphas_frozen):+.4f} |",
        f"| β mean (defense) | {np.mean(betas_frozen):.4f} | {np.mean(betas_new):.4f} | {np.mean(betas_new) - np.mean(betas_frozen):+.4f} |",
        f"| α std | {np.std(alphas_frozen):.4f} | {np.std(alphas_new):.4f} | — |",
        f"| β std | {np.std(betas_frozen):.4f} | {np.std(betas_new):.4f} | — |",
        "",
        "## Top 5 Attack (α) Movers",
        "",
        "| Team | Frozen α | New α | Change |",
        "|------|----------|-------|--------|",
    ]
    for _, da, team, old, new in top5_alpha:
        direction = "↑" if da > 0 else "↓"
        lines.append(f"| {team} | {old:.3f} | {new:.3f} | {da:+.3f} {direction} |")

    lines += [
        "",
        "## Top 5 Defense (β) Movers",
        "",
        "| Team | Frozen β | New β | Change |",
        "|------|----------|-------|--------|",
    ]
    for _, db, team, old, new in top5_beta:
        direction = "↑" if db > 0 else "↓"
        lines.append(f"| {team} | {old:.3f} | {new:.3f} | {db:+.3f} {direction} |")

    lines += [
        "",
        f"## Prediction Flips ({len(flips)} of 16 matches changed)",
        "",
    ]
    if flips:
        lines += [
            "| Match | Home | Away | Frozen Winner | Re-Fitted Winner | Notes |",
            "|-------|------|------|---------------|-----------------|-------|",
        ]
        for f in flips:
            notes = (
                f"P(H): {f['frozen_ph']:.3f}→{f['new_ph']:.3f}, "
                f"P(A): {f['frozen_pa']:.3f}→{f['new_pa']:.3f}"
            )
            lines.append(
                f"| {f['match_id']} | {f['home']} | {f['away']} "
                f"| {f['frozen']} | {f['refitted']} | {notes} |"
            )
    else:
        lines.append("*No prediction flips — both models agree on all 16 Round of 32 outcomes.*")

    lines += [
        "",
        "## Full Prediction Comparison",
        "",
        "| Match | Home | Away | Frozen P(H)/P(D)/P(A) | Winner (frozen) | New P(H)/P(D)/P(A) | Winner (new) | Flip? |",
        "|-------|------|------|-----------------------|-----------------|---------------------|--------------|-------|",
    ]
    for (_, fr), (_, nr) in zip(frozen_df.iterrows(), refitted_df.iterrows()):
        flip = "🔄" if fr["predicted_winner"] != nr["predicted_winner"] else ""
        lines.append(
            f"| {fr['match_id']} | {fr['home_team']} | {fr['away_team']} "
            f"| {fr['p_home_win']:.3f}/{fr['p_draw']:.3f}/{fr['p_away_win']:.3f} "
            f"| **{fr['predicted_winner']}** "
            f"| {nr['p_home_win']:.3f}/{nr['p_draw']:.3f}/{nr['p_away_win']:.3f} "
            f"| **{nr['predicted_winner']}** "
            f"| {flip} |"
        )

    lines += [
        "",
        "---",
        "*Frozen model: predictions based on pre-tournament data only (locked baseline).*",
        "*Re-fitted model: PRIMARY predictions incorporating group stage evidence — use for knockout tracking.*",
    ]

    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    print("=" * 70)
    print("2026 FIFA World Cup — Knockout Stage Prediction Pipeline")
    print("=" * 70)

    # ── OBJECTIVE 1: Frozen Model ──────────────────────────────────────────
    print("\n[1/3] FROZEN MODEL — fitting on historical data only...")
    print(f"  Loading {HISTORICAL_DATA.name}...")

    hist_df = load_and_weight_data(str(HISTORICAL_DATA))
    teams_frozen, team_to_idx_frozen, idx_to_team_frozen = build_team_index(hist_df)
    n_teams_frozen = len(teams_frozen)

    print(f"  Fitting Poisson model on {len(hist_df)} historical matches, {n_teams_frozen} teams...")
    mu_frozen, alphas_frozen, betas_frozen = fit_poisson_model(hist_df, team_to_idx_frozen, n_teams_frozen)
    print(f"  Frozen: μ={mu_frozen:.4f}, ᾱ={np.mean(alphas_frozen):.4f}, β̄={np.mean(betas_frozen):.4f}")

    save_team_ratings(teams_frozen, alphas_frozen, betas_frozen, mu_frozen, PARAMS_FROZEN)

    print("\n  Generating frozen-model predictions...")
    frozen_df = generate_predictions(
        R32_FIXTURES, mu_frozen, alphas_frozen, betas_frozen,
        teams_frozen, team_to_idx_frozen, label="frozen"
    )
    validate_predictions(frozen_df, "Frozen Model")
    frozen_df.to_csv(OUTPUT_FROZEN, index=False)
    print(f"  ✓ Saved → {OUTPUT_FROZEN.name}")

    # ── OBJECTIVE 2: Re-Fit Model ──────────────────────────────────────────
    print("\n[2/3] RE-FIT MODEL — incorporating 72 group stage results...")
    print(f"  Loading group stage results from {GROUP_RESULTS.name}...")

    gs_raw = pd.read_csv(GROUP_RESULTS)
    gs_raw = gs_raw[gs_raw["match_id"] <= 72].copy()  # group stage only

    # Rename columns to match matches_filtered schema
    gs_matches = gs_raw.rename(columns={
        "home_goals": "home_score",
        "away_goals": "away_score",
    })[["date", "home_team", "away_team", "home_score", "away_score"]].copy()

    gs_matches["tournament"] = "World Cup"
    gs_matches["city"]       = ""
    gs_matches["country"]    = ""
    gs_matches["neutral"]    = True
    gs_matches["weight"]     = 1.0  # World Cup weight (highest importance)

    print(f"  Group stage matches to add: {len(gs_matches)}")

    # Combine: historical + group stage
    combined_df = pd.concat([hist_df, gs_matches], ignore_index=True)
    print(f"  Combined dataset: {len(combined_df)} total matches")

    # Build team index from combined data (same 48 teams, no new ones)
    teams_new, team_to_idx_new, idx_to_team_new = build_team_index(combined_df)
    n_teams_new = len(teams_new)

    print(f"  Fitting re-fitted Poisson model ({n_teams_new} teams)...")
    mu_new, alphas_new, betas_new = fit_poisson_model(combined_df, team_to_idx_new, n_teams_new)
    print(f"  Re-fitted: μ={mu_new:.4f}, ᾱ={np.mean(alphas_new):.4f}, β̄={np.mean(betas_new):.4f}")

    save_team_ratings(teams_new, alphas_new, betas_new, mu_new, PARAMS_REFITTED)

    print("\n  Generating re-fitted-model predictions...")
    refitted_df = generate_predictions(
        R32_FIXTURES, mu_new, alphas_new, betas_new,
        teams_new, team_to_idx_new, label="re-fitted"
    )
    validate_predictions(refitted_df, "Re-Fitted Model")
    refitted_df.to_csv(OUTPUT_REFITTED, index=False)
    print(f"  ✓ Saved → {OUTPUT_REFITTED.name}")

    # ── OBJECTIVE 3: Diagnostics & Diff Report ─────────────────────────────
    print("\n[3/3] DIAGNOSTICS — comparing models...")

    # Align teams arrays (both models should have same 48 teams in same sorted order)
    assert list(teams_frozen) == list(teams_new), \
        "Team list mismatch between frozen and re-fitted models!"

    diff_md = build_diff_report(
        frozen_df, refitted_df,
        teams_frozen,
        alphas_frozen, betas_frozen, mu_frozen,
        alphas_new,    betas_new,    mu_new,
        team_to_idx_frozen,
        n_teams_frozen,
        hist_df,
        combined_df,
    )
    OUTPUT_DIFF.write_text(diff_md, encoding="utf-8")
    print(f"  ✓ Diff report saved → {OUTPUT_DIFF.name}")

    # Print a quick summary
    flips_count = sum(
        1 for (_, fr), (_, nr) in zip(frozen_df.iterrows(), refitted_df.iterrows())
        if fr["predicted_winner"] != nr["predicted_winner"]
    )
    print(f"\n  Prediction flips (frozen→refitted): {flips_count}/16")

    # ── FINAL SUMMARY ──────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("✅ PIPELINE COMPLETE")
    print("=" * 70)
    print(f"\nOutputs:")
    print(f"  • {OUTPUT_FROZEN.name}      — baseline (frozen pre-tournament model)")
    print(f"  • {OUTPUT_REFITTED.name}   — PRIMARY (re-fitted with group stage data)")
    print(f"  • {OUTPUT_DIFF.name}       — parameter & prediction comparison")
    print(f"  • {PARAMS_FROZEN.name}     — frozen parameters (reference)")
    print(f"  • {PARAMS_REFITTED.name}   — re-fitted parameters")
    print(f"\nRound of 32 schedule: June 28 – July 3, 2026")
    print(f"Next phase: log actual results to results/knockout_outcomes.csv")
    print()

    print("\nKnockout Predictions (Re-Fitted Model — PRIMARY):")
    print(f"  {'#':>2}  {'Home':25s}  {'P(H)':>6}  {'P(D)':>6}  {'P(A)':>6}  {'Away':25s}  {'Winner'}")
    print(f"  {'─'*90}")
    for _, row in refitted_df.iterrows():
        print(
            f"  {int(row['match_id']):>2}  {row['home_team']:25s}  "
            f"{row['p_home_win']:>6.3f}  {row['p_draw']:>6.3f}  {row['p_away_win']:>6.3f}  "
            f"{row['away_team']:25s}  {row['predicted_winner']}"
        )


if __name__ == "__main__":
    main()
