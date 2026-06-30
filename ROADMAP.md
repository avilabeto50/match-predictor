# ROADMAP — Copa 2026 Match Predictor

This document is the authoritative development plan for the project. It is organized into phases that follow the logical dependency chain: data before modeling, modeling before evaluation, evaluation before dashboard. Each phase lists exactly what needs to exist at the end of it, what decisions need to be made, and what the known hard parts are.

The 2026 FIFA World Cup begins June 11, 2026. All modeling work must be complete and validated before that date.

---

## Phase 0 — Repository and Environment ✅ COMPLETE

**Goal:** A working Python environment on the Windows 11 machine, a clean repository structure, and all data on disk before any code is written.

**Status:** Environment set up, raw data downloaded and organized. Fixture list compiled with all 104 matches (72 group stage + 32 knockout).

### Repository structure to establish

```
copa2026/
├── data/
│   ├── raw/                   # untouched source files, never modified
│   └── processed/             # cleaned, filtered outputs ready for modeling
├── notebooks/                 # exploratory Jupyter notebooks, numbered sequentially
├── src/
│   ├── data_prep.py           # data loading, cleaning, filtering
│   ├── elo.py                 # Elo rating computation
│   ├── poisson_model.py       # attack/defense MLE + probability derivation
│   └── calibration.py         # Brier score, reliability diagrams
├── predictions/               # one CSV per round, generated before each matchday
├── results/                   # actual outcomes logged after each round
├── README.md
└── ROADMAP.md
```

### Environment

- Python 3.11 via Anaconda or Miniconda (Miniconda preferred for a clean install)
- Create a dedicated conda environment: `conda create -n copa2026 python=3.11`
- Required packages: `pandas`, `numpy`, `scipy`, `scikit-learn`, `matplotlib`, `seaborn`, `jupyter`, `notebook`
- Optional for dashboard phase: `streamlit`, `plotly`
- The RTX 4060 is not needed for any phase of this project. All computation is closed-form or lightweight optimization, not neural training.

### Data to download

1. **Historical match results**: download `results.csv` from `https://github.com/martj42/international-football-results` (or the Kaggle mirror). Save to `data/raw/results.csv`. Do not modify this file.
2. **FIFA World Rankings**: download the historical rankings CSV from Kaggle ("FIFA World Rankings 1993–2023") or scrape the current ranking from `https://www.fifa.com/fifa-world-ranking`. Save to `data/raw/fifa_rankings.csv`.
3. **2026 World Cup fixture list**: all 104 matches (72 group stage + 32 knockout) with dates, venues, and team matchups compiled into `data/raw/wc2026_fixtures.csv`. ✅ COMPLETE

### Deliverable at end of Phase 0

- Conda environment activates cleanly
- All three raw data files are on disk
- Folder structure exists
- README.md and ROADMAP.md committed to the repo

---

## Phase 1 — Data Preparation ✅ COMPLETE

**Goal:** A single clean, filtered dataframe of international match results ready for Elo computation and attack/defense estimation, plus a team name mapping that reconciles inconsistencies across the historical record.

**Status:** Data preparation notebook created (`notebooks/01_data_preparation.ipynb`). Historical match data filtered and team name normalization applied for all 48 qualified teams.

### What to build: `src/data_prep.py`

**Filtering decisions:**

The full historical dataset goes back to 1872. Using all of it would give undue weight to matches that are essentially irrelevant to predicting modern teams. The recommended filter is matches from **January 1, 2014 onward** (approximately three World Cup cycles). This preserves enough history for statistical estimation while keeping the team strengths reflective of current squads.

Additionally, weight matches by recency and importance. A friendly played in 2014 should matter less than a World Cup qualifier played in 2024. A simple weighting scheme:
- World Cup matches: weight 1.0
- Continental championship matches (CONMEBOL, UEFA, etc.): weight 0.85
- World Cup qualifiers: weight 0.75
- Friendlies: weight 0.5

These weights are applied during attack/defense estimation (Phase 3), not during Elo computation (Phase 2 has its own k-factor scheme).

**Team name normalization:**

The historical dataset uses inconsistent team names across decades. The 48 qualified teams for 2026 must have a canonical name that maps to all variants in the historical record. Known issues to resolve manually:
- "United States" vs "USA"
- "IR Iran" vs "Iran"
- "Korea Republic" vs "South Korea"
- "Ivory Coast" vs "Côte d'Ivoire"
- "Czech Republic" vs "Czechia"
- "Curaçao" variants (accents, caps)
- "Cabo Verde" vs "Cape Verde"

Build a `TEAM_NAME_MAP` dictionary in `data_prep.py` that normalizes all variants to a single canonical string. Apply this to both home and away team columns. ✅ COMPLETE

**Output:**

- `data/processed/matches_filtered.csv` — filtered, normalized match history
- `data/processed/wc2026_teams.csv` — list of 48 qualified teams with canonical names ✅ COMPLETE

### Known hard parts

Getting team name normalization right is the most tedious part of the whole project. Budget a few hours for this. The safest approach is to filter the dataset to only matches involving the 32 qualified teams before doing anything else, then inspect the unique team name strings for variants.

---

## Phase 2 — Elo Ratings 🔄 IN PROGRESS

**Goal:** A single Elo rating for each of the 48 qualified teams, computed from the filtered historical match data.

**Status:** Elo computation notebook created (`notebooks/02_elo_computation.ipynb`). Elo ratings being computed using K-factor scheme weighted by match importance.

### What to build: `src/elo.py`

**How Elo works in this context:**

Elo is an iterative rating system. Every team starts with a base rating (conventionally 1500). After each match, ratings are updated based on the outcome and the expected outcome given pre-match ratings. The update rule is:

```
R_new = R_old + K * (S - E)
```

Where:
- `S` is the actual score: 1 for a win, 0.5 for a draw, 0 for a loss
- `E` is the expected score: `1 / (1 + 10^((R_opponent - R_own) / 400))`
- `K` is the sensitivity factor, which should vary by match importance

**K-factor scheme:**

- World Cup matches: K = 60
- Continental championships: K = 50
- World Cup qualifiers: K = 40
- Friendlies: K = 20

These values are consistent with the widely cited Club Elo and FiveThirtyEight international Elo implementations.

**Implementation approach:**

Process the filtered match history chronologically. For each match, look up the current Elo of both teams, compute the update, and apply it. At the end of processing, read off the final rating for each of the 32 qualified teams.

**Home advantage:**

International matches played on neutral ground (which is typical for tournaments) have no home advantage adjustment. For historical club matches played at home, add a fixed offset of +100 rating points to the home team's effective rating when computing the expected outcome. The 2026 World Cup is played on neutral ground (across USA, Canada, Mexico), so home advantage does not apply to tournament predictions.

**Output:**

- `data/processed/elo_ratings.csv` — one row per team (48 teams), columns: `team`, `elo_rating`
- A saved plot: `notebooks/elo_distribution.png` — bar chart of all 48 teams sorted by Elo rating, useful for a sanity check

### Sanity checks

After computing ratings, verify:
- Argentina, France, Brazil, England, and Germany should be near the top
- The spread should be roughly 1200–1900 for qualified teams
- No team should have a rating below 1000 or above 2200

If results look implausible, the most likely cause is incorrect team name normalization bleeding through from Phase 1.

---

## Phase 3 — Attack / Defense Strength Model (Poisson Model)

**Goal:** Two parameters per team — an attack rating and a defense rating — estimated from historical data via maximum likelihood. From these, derive win/draw/loss probabilities for any match.

**Status:** ✅ COMPLETE. Model fitted on 1505 filtered historical matches; team attack/defense ratings saved to `data/processed/team_ratings.csv` and sanity checks passed.

### The model

For a match between team $i$ (home) and team $j$ (away), the number of goals scored by each team is modeled as:

```
Goals_i ~ Poisson(lambda_ij)
Goals_j ~ Poisson(lambda_ji)

lambda_ij = mu * alpha_i * beta_j
lambda_ji = mu * alpha_j * beta_i
```

Where:
- `mu` is the average number of goals scored per team per match across all matches in the dataset (typically around 1.3–1.5 for international football)
- `alpha_i` is team $i$'s attack strength (values above 1.0 = above-average attack)
- `beta_i` is team $i$'s defensive vulnerability (values above 1.0 = above-average leakiness)

All four variables are positive real numbers. By convention, attack and defense ratings are scaled so that their product across all teams equals 1 (i.e., they are multiplicative adjustments relative to the mean).

### What to build: `src/poisson_model.py`

**Part 1 — MLE estimation**

Fit `alpha` and `beta` for all teams simultaneously using maximum likelihood. The log-likelihood function over all historical matches is:

```
log L = sum over matches [ log P(goals_i | lambda_ij) + log P(goals_j | lambda_ji) ]
      = sum over matches [ goals_i * log(lambda_ij) - lambda_ij - log(goals_i!)
                         + goals_j * log(lambda_ji) - lambda_ji - log(goals_j!) ]
```

Minimize the negative log-likelihood using `scipy.optimize.minimize` with the L-BFGS-B method. Initialize all attack and defense parameters at 1.0. The optimization should converge in seconds.

Apply match weights (from Phase 1) by multiplying each term in the log-likelihood sum by its corresponding weight before summing. This is the mechanism by which recent, high-stakes matches carry more influence.

**Part 2 — Probability derivation**

Given `lambda_ij` and `lambda_ji` for a specific match, compute win/draw/loss probabilities analytically:

```
P(win for i) = sum over g_i, g_j where g_i > g_j: Poisson_PMF(g_i, lambda_ij) * Poisson_PMF(g_j, lambda_ji)
P(draw)      = sum over g: Poisson_PMF(g, lambda_ij) * Poisson_PMF(g, lambda_ji)
P(loss for i) = 1 - P(win) - P(draw)
```

Truncate the sum at 10 goals per team. The probability mass beyond 10 goals is negligible (less than 0.001% at typical scoring rates). This gives a 10×10 joint probability matrix from which win/draw/loss probabilities are read off directly.

**Part 3 — Expected scoreline**

Also output the most likely scoreline(s) by returning the top 5 (g_i, g_j) pairs by joint probability. This is a useful human-readable summary of the model's expectations and a natural dashboard feature.

**Output:**

- `data/processed/team_ratings.csv` — columns: `team`, `attack`, `defense`, `mu`
- A function `predict_match(team_a, team_b) -> dict` that returns `{p_win, p_draw, p_loss, top_scorelines}`

### Sanity checks

- `predict_match("Argentina", "Saudi Arabia")` should give Argentina a win probability above 0.70
- `predict_match("France", "Brazil")` should be competitive, with neither team above 0.55 win probability
- All three probabilities should sum to 1.0 (allow floating point tolerance of 1e-6)
- No probability should be negative or above 1.0

---

## Phase 4 — Combining Elo and Poisson (Optional Blending)

**Goal:** Decide whether to blend the Elo-based win probabilities with the Poisson model's probabilities, and if so, how.

**Status:** ⏭️ SKIPPED. Using the pure Poisson model for the locked pre-tournament predictions; revisit blending after the tournament.

### The design choice

Elo and the Poisson attack/defense model answer slightly different questions. Elo is a pure strength signal with no goal-scoring structure. The Poisson model captures offensive and defensive tendencies separately but may be miscalibrated for rare matchups (teams with little shared history).

The simplest approach is to use the Poisson model as the primary output and use the Elo rating purely as a feature — specifically, include the Elo rating difference between the two teams as an input feature that shifts `mu` slightly upward or downward. This keeps the probabilistic structure of the Poisson model while grounding it in the Elo signal.

A concrete implementation:

```
mu_adjusted = mu * (1 + gamma * (elo_i - elo_j) / 1000)
```

Where `gamma` is a small scalar (0.05–0.15) that modulates how much the Elo gap shifts the expected scoring rate. The optimal value of `gamma` can be estimated via cross-validation on held-out historical matches, or simply fixed at 0.10 as a reasonable default.

This is optional. The model is informative without it. Revisit after Phase 3 is producing sensible outputs.

---

## Phase 5 — Generating Tournament Predictions

**Goal:** Before the tournament begins, generate win/draw/loss probability predictions for all 72 group-stage matches and commit them to the repository. This creates a locked-in set of pre-tournament predictions to evaluate against real outcomes.

**Status:** ✅ COMPLETE. All 72 group-stage match predictions are locked in `predictions/group_stage_predictions.csv`, probabilities verified, and team name mismatches corrected.

### Process

1. Load the fixture list from `data/raw/wc2026_fixtures.csv` (✅ already compiled)
2. For each group-stage match (72 total), call `predict_match(team_a, team_b)`
3. Write results to `predictions/group_stage_predictions.csv`

Column schema for predictions CSV:
```
match_id, date, group, team_a, team_b, p_win_a, p_draw, p_win_b, predicted_winner
```

Where `predicted_winner` is `team_a` if `p_win_a > p_win_b`, `team_b` if `p_win_b > p_win_a`, or `"draw"` if `p_draw` is the maximum of the three.

**Do not go back and modify prediction files after matches are played.** The integrity of the calibration analysis depends on predictions being locked before outcomes are known.

For knockout rounds, generate predictions round by round after the bracket is determined.

---

## Phase 6 — Live Calibration Tracking

**Status:** 🔜 IN PROGRESS. Preparation is complete; calibration tracking will begin once actual outcomes are logged during the tournament.

**Goal:** After each round of matches, record actual outcomes and compute calibration metrics against locked-in predictions. This is the analytical centerpiece of the project.

### What to build: `src/calibration.py`

**Results logging:**

After each matchday, create or append to `results/actual_outcomes.csv`:
```
match_id, team_a, team_b, goals_a, goals_b, outcome
```

Where `outcome` is `win_a`, `draw`, or `win_b`.

**Brier Score:**

The Brier score measures the mean squared error of probability predictions over a set of outcomes. For a three-outcome problem (win/draw/loss):

```
BS = (1/N) * sum over matches [ (p_win - o_win)^2 + (p_draw - o_draw)^2 + (p_loss - o_loss)^2 ]
```

Where `o_win`, `o_draw`, `o_loss` are 1 or 0 depending on the actual outcome. Lower is better. A random model predicting 1/3 for each outcome has a Brier score of approximately 0.667. A perfect model has 0.0. In practice, good sports forecasting models score in the 0.55–0.62 range for football.

**Reliability diagram:**

Bin all win probability predictions into 10 buckets of width 0.1 (i.e., 0.0–0.1, 0.1–0.2, ..., 0.9–1.0). For each bucket, compute the mean predicted probability and the actual win rate across the matches in that bucket. Plot predicted probability (x-axis) against observed frequency (y-axis). A perfectly calibrated model lies on the diagonal. Points above the diagonal indicate overconfidence; points below indicate underconfidence.

This plot should be regenerated and saved after each round.

**Log-loss (supplementary metric):**

```
Log-loss = -(1/N) * sum [ o_win * log(p_win) + o_draw * log(p_draw) + o_loss * log(p_loss) ]
```

Add a small epsilon (1e-15) inside the log to prevent numerical issues when a predicted probability is exactly 0. Log-loss heavily penalizes confident wrong predictions — a useful complement to Brier score.

**Output after each round:**

- Updated `results/calibration_summary.csv` with cumulative Brier score and log-loss
- Updated reliability diagram saved as `results/reliability_diagram_after_round_N.png`

---

## Phase 7 — Streamlit Live Calibration Dashboard

**Goal:** A local web app for real-time match result entry and live calibration visualization during the tournament.

**Status:** ✅ IMPLEMENTED. Functional and ready for tournament (June 11–July 19, 2026).

### How to run

```bash
streamlit run app.py
```

Opens at `localhost:8501`

### Three main tabs

**Dashboard Tab**

- Tournament progress (matches completed / total, percentage)
- Brier Score (measures prediction accuracy; lower is better)
- Log-Loss (penalizes confident wrong predictions)
- Calibration Reliability Diagram (visual check: perfectly calibrated model lies on diagonal)
- List of 10 most recent completed matches with predictions vs actual outcomes

**Enter Results Tab**

- Dropdown selector for next unplayed match
- Input fields for home and away goals
- Shows predicted winner and top scoreline from locked predictions
- Submit button saves result to `results/actual_outcomes.csv`
- Dashboard auto-updates with new metrics

**Match History Tab**

- Browse all completed matches
- Filter by group (A–L) or outcome (home win / draw / away win)
- Side-by-side view of predicted vs actual outcomes
- All predictions and goals displayed

### Technical details

- Reads from: `predictions/group_stage_predictions.csv` (locked before tournament)
- Writes to: `results/actual_outcomes.csv` (appends one row per match)
- Metrics computed on-the-fly using pandas and numpy
- Reliability diagrams generated with matplotlib/seaborn
- Requires: streamlit, pandas, numpy, matplotlib, seaborn

### Calibration workflow during tournament

1. After each matchday, open dashboard
2. Go to "Enter Results" tab
3. Select match, enter score, click Submit
4. Results auto-saved and Brier score updates
5. Reliability diagram updates after 5+ matches completed

### Integration with Phase 6

The dashboard IS the Phase 6 implementation. As matches are entered, it computes all calibration metrics on-the-fly (Brier score, log-loss, reliability diagram), replacing the need for manual Phase 6 calibration.py code.

### Post-tournament analysis

After July 19, final calibration metrics and reliability diagram inform whether Phase 4 (Elo blending) would improve the model for a v2 version.

---

## Phase 8 — Knockout Stage Re-Fit and Predictions

**Status:** ✅ COMPLETE (June 30, 2026). Knockout predictions generated and locked.

**Goal:** Re-fit the Poisson model incorporating all 72 group stage results, then generate Round of 32 predictions under both frozen (pre-tournament) and re-fitted (post-group-stage) model states for a before/after calibration comparison.

### What was built

- `src/standings.py` — Group standings utility (reference; not used for bracket resolution since bracket was provided directly)
- `generate_knockout_predictions.py` — Orchestration script: runs both model fits, generates both CSVs, produces the diff report
- `data/processed/team_ratings_frozen.csv` — Pre-tournament parameters saved as reference
- `data/processed/team_ratings_knockout_refitted.csv` — Re-fitted parameters (historical + 72 WC group stage results)

### Outputs

| File | Description |
|------|-------------|
| `predictions/knockout_predictions.csv` | **PRIMARY** — 16 Round of 32 predictions (re-fitted model) |
| `predictions/knockout_predictions_frozen_model.csv` | Baseline — 16 Round of 32 predictions (frozen pre-tournament model) |
| `predictions/knockout_prediction_diff.md` | Parameter comparison + prediction flips report |

### Key findings from re-fit

- **Log-likelihood improvement**: +5.66 (frozen → re-fitted on combined data)
- **μ change**: 1.2189 → 1.2285 (+0.01, slightly more goals expected after WC evidence)
- **Biggest attack movers**: Norway (+0.126), Senegal (+0.101), Netherlands (+0.087), Canada (+0.086), Germany (+0.077)
- **Biggest defense movers**: Iraq (+0.181 leakier), New Zealand (+0.178 leakier), Cabo Verde (−0.136 tighter)
- **Prediction flips**: 0/16 — both models agreed on all Round of 32 predicted winners
- **Closest match**: Mexico vs Ecuador (Mexico 38.1% vs Ecuador 33.1%, within 5 points)
- **Biggest favorite**: Argentina vs Cabo Verde (Argentina 68.2%)

### Round of 32 Predicted Winners (Re-Fitted Model)

| Match | Home | Away | Predicted |
|-------|------|------|-----------|
| 73 | Germany | Paraguay | **Germany** (62.2%) |
| 74 | France | Sweden | **France** (66.8%) |
| 75 | South Africa | Canada | **Canada** (42.0% away) |
| 76 | Netherlands | Morocco | **Netherlands** (42.4%) |
| 77 | Portugal | Croatia | **Portugal** (54.0%) |
| 78 | Spain | Austria | **Spain** (52.6%) |
| 79 | United States | Bosnia and Herzegovina | **United States** (48.1%) |
| 80 | Belgium | Senegal | **Belgium** (53.9%) |
| 81 | Brazil | Japan | **Brazil** (61.9%) |
| 82 | Côte d'Ivoire | Norway | **Côte d'Ivoire** (43.7%) |
| 83 | Mexico | Ecuador | **Mexico** (38.1%) |
| 84 | England | DR Congo | **England** (66.4%) |
| 85 | Argentina | Cabo Verde | **Argentina** (68.2%) |
| 86 | Australia | Egypt | **Egypt** (37.7% away) |
| 87 | Switzerland | Algeria | **Switzerland** (45.2%) |
| 88 | Colombia | Ghana | **Colombia** (56.8%) |

---

## Phase 9 — Knockout Calibration Tracking

**Status:** 🔜 ACTIVE (June 28 – July 3, 2026).

**Goal:** As Round of 32 matches conclude, log actual results and track calibration metrics separately for both frozen and re-fitted models.

### Process

1. After each knockout match, open dashboard → "🔴 Knockout Stage" tab
2. Select match, enter FT score (including extra time if applicable), click Submit
3. Results saved to `results/knockout_outcomes.csv`
4. Brier Score updates live for both models
5. At end of Round of 32, the side-by-side Brier score comparison will show whether the re-fit improved calibration

### Key difference from group stage tracking

- Goals include extra time (draw in 90 min → goes to penalties, log as "draw" for the model since it only sees 90+ET score)
- Knockout results saved to `results/knockout_outcomes.csv` (separate from group stage `results/actual_outcomes.csv`)
- Both frozen and re-fitted models tracked simultaneously for direct comparison

---

## Milestone Checklist

| Milestone | Target Date |
|---|---|
| Phase 0 complete: repo, environment, raw data downloaded | June 8, 2026 |
| Phase 1 complete: filtered dataset, team name normalization done | June 9, 2026 |
| Phase 2 complete: Elo ratings computed, sanity checks passed | June 9, 2026 |
| Phase 3 complete: Poisson model fit, predict_match function working | June 6, 2026 |
| Phase 4 skipped: defer Elo blending to post-tournament v2 | June 6, 2026 |
| Phase 5 complete: all group-stage predictions locked in pre-tournament | June 6, 2026 |
| Phase 7 complete: Streamlit dashboard running locally | June 6, 2026 |
| Phase 6 active: results logged, calibration updated after each round | June 11 – July 19, 2026 |
| Post-tournament: final calibration analysis, README updated | July 20, 2026 |

---

## Known Risks and Mitigations

**Team name normalization takes longer than expected.** Mitigation: do this first, before writing any modeling code. Get the 32-team name map fully correct before touching Elo or Poisson code.

**Some qualified teams have little recent match history.** A team that has rarely played strong opponents in the last three years will have noisy attack/defense estimates. Mitigation: fall back to Elo rating as the primary signal for such teams. When MLE produces extreme parameter values (attack or defense rating below 0.4 or above 2.5), treat the estimate as unreliable and clip it.

**The fixture list may not be fully finalized.** Some kickoff times and venues may shift. Mitigation: the prediction model only needs team names, not venues, so venue changes do not affect predictions. Update kickoff dates in the fixture CSV as they are confirmed.

**Calibration analysis requires enough predictions to be statistically meaningful.** The group stage produces 72 matches, which is solid for calibration but interpretation should remain grounded in the uncertainty that inherently exists with any forecasting model until the full tournament is complete.

---

*This roadmap should be treated as a living document during Phase 0 only. Once modeling begins in Phase 1, the analytical design (Phases 2–6) should not be changed mid-stream, as doing so would invalidate the calibration analysis. Feature additions belong in Phase 7 or a separate v2 document.*
