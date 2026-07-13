# FINDINGS — Copa 2026 Group Stage Match Predictor

## 1. Motivating Question

Can a first-principles statistical model — a Poisson attack/defense model fit via MLE on historical international match data — produce well-calibrated win/draw/loss probability forecasts for a 72-match World Cup group stage? And if not, where exactly does it break, and why?

Secondary question: what does a CS theory person learn about applied statistics by building and stress-testing their first real model against live data?

---

## 2. Model Setup

### The Model
- **Type**: Poisson attack/defense model (a.k.a. Dixon-Coles family, though without the Dixon-Coles low-score correlation correction)
- **Parameters per team**: attack strength (α), defensive vulnerability (β)
- **Global parameter**: μ ≈ 1.21 (expected goals per team per match, shared across all matchups)
- **Match intensity**:
  - `λ_home = μ × α_home × β_away`
  - `λ_away = μ × α_away × β_home`
- **Goal distribution**: `Goals_home ~ Poisson(λ_home)`, `Goals_away ~ Poisson(λ_away)`, **assumed independent**
- **Probability computation**: Joint PMF enumerated over 0–10 goals per side (11×11 = 121 scorelines), win/draw/loss read off directly

### Estimation
- **Method**: Maximum Likelihood Estimation (MLE) via L-BFGS-B optimizer (`scipy.optimize.minimize`)
- **Regularization**: L2 penalty on log(α) and log(β), strength = 300, to break the multiplicative scaling symmetry (the αβ identifiability issue)
- **Post-optimization**: geometric mean normalization of α and β to 1.0, then clipping to [0.4, 2.5]

### Training Data
- **Source**: [martj42/international-football-results](https://github.com/martj42/international-football-results)
- **Filter**: January 1, 2014 – June 4, 2026 (≈3 World Cup cycles)
- **Size**: 1,433 matches across 48 qualified teams
- **Weighted by tournament importance**:
  - World Cup: 1.0
  - Continental championships: 0.85
  - Nations League: 0.80
  - WC qualifiers: 0.75
  - Friendlies: 0.50
  - (Full tier system in [poisson_model.py](file:///c:/Users/avila/OneDrive/Escritorio/Summer%20Stuffs/match-predictor/src/poisson_model.py#L14-L48))

### What Was Locked and When
- **72 group stage predictions locked** in [group_stage_predictions_new.csv](file:///c:/Users/avila/OneDrive/Escritorio/Summer%20Stuffs/match-predictor/predictions/group_stage_predictions_new.csv) before June 11, 2026 (tournament start)
- All 72 actual results recorded in [actual_outcomes.csv](file:///c:/Users/avila/OneDrive/Escritorio/Summer%20Stuffs/match-predictor/results/actual_outcomes.csv)
- Predictions were NOT modified after matches were played
- A re-fitted model (incorporating group stage results) was used for knockout predictions, but group stage calibration uses the pre-tournament locked predictions only

---

## 3. What I Expected

- **Outcome accuracy around 50–55%**: picking the most likely outcome should be right about half the time in international football, given how frequent draws and upsets are
- **Brier score in the 0.55–0.62 range**: the ROADMAP cites this as "good sports forecasting" territory
- **Draw rate roughly matching historical**: the training data has a 25.1% draw rate; I expected the WC to track close to that
- **Model should do well on lopsided matches**: Brazil vs Haiti, Germany vs Curaçao — these should be easy
- **Model might struggle on closely-matched teams**: Groups D and F especially

---

## 4. What Actually Happened

### Top-Line Calibration Numbers

| Metric | Value | Reference |
|--------|-------|-----------|
| **Brier Score** | **0.4996** | Perfect: 0.0, Random (1/3 each): 0.667 |
| **Log-Loss** | **0.8478** | Perfect: 0.0, Random (1/3 each): 1.099 |
| **Outcome accuracy** (most likely outcome correct) | **46/72 = 63.9%** | — |
| **Exact scoreline correct** | **10/72 = 13.9%** | — |

> [!IMPORTANT]
> Brier of 0.50 is meaningfully better than random (0.667) — the model is extracting real signal. But it's worse than the 0.55–0.62 "good" range cited in the ROADMAP. The model beat random but did not reach the level of established forecasting systems.

### Outcome Distribution: Predicted vs Actual

| Outcome | Model Predicted (most likely) | Actual | Difference |
|---------|------------------------------|--------|------------|
| Home win | 45 | 34 | −11 |
| Draw | **0** | **20** | **+20** |
| Away win | 27 | 18 | −9 |

> [!CAUTION]
> **The model predicted zero draws as the most likely outcome across all 72 matches.** Twenty actually happened. This is the single biggest structural failure and it's not a bug — it's inherent to the independent Poisson assumption.

### Mean Predicted Probabilities vs Actual Rates

| Outcome | Mean Predicted P | Actual Rate | Gap |
|---------|-----------------|-------------|-----|
| Home win | 42.2% | 47.2% | −5.0 pp |
| Draw | 23.1% | **27.8%** | **−4.7 pp** |
| Away win | 34.8% | 25.0% | +9.8 pp |

- Home advantage was **underestimated**: model predicted 42.2% home win rate, actual was 47.2%
- Draw rate was **underestimated**: predicted 23.1%, actual 27.8%
- Away wins were **overestimated**: predicted 34.8%, actual 25.0%

### Probability Assigned to Actual Outcome

| Subset | Mean P(actual outcome) |
|--------|----------------------|
| All matches | 0.4727 |
| Home wins (n=34) | 0.5262 |
| Draws (n=20) | **0.2472** |
| Away wins (n=18) | 0.6220 |

- When a draw happened, the model had only assigned it a 24.7% probability on average
- When an away win happened, the model gave it 62.2% — actually well-calibrated for decisive away results

### Goal-Scoring Statistics

| Stat | Value |
|------|-------|
| Total goals (group stage) | 215 |
| Average goals/match | 2.99 |
| Historical avg (training data) | 2.58 |
| Home goals | 129 (60.0%) |
| Away goals | 86 (40.0%) |

- The 2026 World Cup group stage was **higher-scoring than the training data** (2.99 vs 2.58 goals/match). The model's μ = 1.21 expected ~2.42 goals/match. The tournament ran hotter.

### Confidence and Errors

| Stat | Value |
|------|-------|
| Matches with >50% confidence | 45/72 |
| Of those, wrong | 12 (26.7%) |
| Matches where actual outcome was LEAST likely | 8 (11.1%) |

### Brier Score by Matchday

| Period | Brier | n | Draws |
|--------|-------|---|-------|
| MD1 (Jun 11–17) | **0.6062** | 24 | 9 (37.5%) |
| MD2 (Jun 18–23) | 0.4538 | 24 | 5 (20.8%) |
| MD3 (Jun 24–27) | 0.4387 | 24 | 6 (25.0%) |

- MD1 was near-random quality (0.606 vs random 0.667) — **9 of 24 matches were draws**
- Performance improved dramatically in MD2 and MD3 as the draw rate fell closer to historical norms

---

## 5. Where It Broke

### Failure Mode 1: Structural Draw Underestimation

**The evidence:**
- The model **never** predicted draw as the most likely outcome in any of 72 matches
- Maximum P(draw) across all 72 matches: **32.0%** (Czechia vs South Africa)
- Mean P(draw): 23.1% vs actual draw rate 27.8%
- 10 "surprise draws" where P(draw) < 25% and the match drew anyway
- 8 of the 10 worst Brier-contributing matches were draws the model gave low probability to

**The top 10 worst predictions were ALL draws:**

| Match | P(draw) | Actual | Brier contribution |
|-------|---------|--------|--------------------|
| Spain vs Cabo Verde | 16.5% | 0–0 | 1.2535 |
| Portugal vs DR Congo | 16.7% | 1–1 | 1.2420 |
| Ecuador vs Curaçao | 19.0% | 0–0 | 1.1481 |
| Qatar vs Switzerland | 19.7% | 1–1 | 1.0842 |
| England vs Ghana | 20.9% | 0–0 | 1.0663 |
| Belgium vs IR Iran | 22.8% | 0–0 | 1.0010 |
| Belgium vs Egypt | 22.7% | 1–1 | 0.9984 |
| Uruguay vs Cabo Verde | 24.1% | 2–2 | 0.9620 |
| Netherlands vs Japan | 21.8% | 2–2 | 0.9576 |
| Saudi Arabia vs Uruguay | 25.6% | 1–1 | 0.9345 |

**Why this happens (structural):**
- The model assumes Goals_home and Goals_away are **independent** Poisson random variables
- Under independence, P(X = Y) is always bounded by a sum of products of individual Poisson PMFs
- For typical λ values (1.0–2.0), the draw probability maxes out around 25–30%, even when teams are closely matched
- In reality, goals in football are **positively correlated** — both teams respond to the game state (a team that concedes a goal plays more attacking football to equalize). This inflates draws beyond what independent Poisson can produce
- The **Dixon-Coles correction** (1997) specifically addresses this by adding a correlation parameter ρ for low-scoring outcomes (0–0, 1–0, 0–1, 1–1). This project does not implement it.

**Draw calibration detail:**

| P(draw) bin | Predicted mean | Observed draw rate | n |
|------------|---------------|-------------------|---|
| [0.05, 0.10) | 0.079 | 0.000 | 2 |
| [0.10, 0.15) | 0.129 | 0.000 | 4 |
| [0.15, 0.20) | 0.176 | **0.308** | 13 |
| [0.20, 0.25) | 0.229 | **0.286** | 21 |
| [0.25, 0.30) | 0.272 | 0.241 | 29 |
| [0.30, 0.35) | 0.319 | **1.000** | 3 |

- In the 0.15–0.25 range (where most predictions land), the actual draw rate is 29–31% but the model predicts 18–23%. Systematic underestimation.

### Failure Mode 2: Home Advantage Underestimation

**The evidence:**
- Actual home win rate: **47.2%**; model predicted: **42.2%** (−5 pp)
- Actual away win rate: **25.0%**; model predicted: **34.8%** (+9.8 pp)
- Home teams scored 60% of all goals
- The model does not include a home advantage parameter — all teams are treated symmetrically

**Why this matters:**
- The model structure `λ_home = μ × α_home × β_away` and `λ_away = μ × α_away × β_home` is **perfectly symmetric** — there's no factor boosting the home team
- Yet the first-listed team won 47.2% of the time vs 25.0% for second-listed
- Even at neutral venues (all group stage matches are in USA/Canada/Mexico), there appears to be a listing-order / "home designation" effect
- A simple multiplicative home advantage factor (e.g., `λ_home *= 1.1`) would have partially corrected both the home and away win rate biases

### Failure Mode 3: Overconfidence on Lopsided Matches

**The evidence:**
- Of 45 matches where the model had >50% confidence, **12 were wrong (26.7%)**
- Several confidently-predicted favorites drew unexpectedly:
  - Spain (74.0% to win) drew Cabo Verde 0–0
  - Portugal (73.4% to win) drew DR Congo 1–1
  - Ecuador (69.2% to win) drew Curaçao 0–0
  - England (64.7% to win) drew Ghana 0–0

**Why this happens:**
- The model concentrates too much probability on the favorite's win, partly because it steals from draw probability (see Failure Mode 1)
- If the true draw rate is ~28% and the model says ~23%, that 5 pp is redistributed to the favorite, making them look more dominant than they are
- This is not independent of the draw problem — fix the draws and the overconfidence partially resolves

### Failure Mode 4: Potential Recency Bias / Stale Parameters

**The evidence:**
- Training data spans 2014–2026 with tournament-type weights but **no time-decay weighting**
- A friendly from 2015 with weight 0.50 still counts the same as a friendly from 2025 with weight 0.50
- Notable misses that suggest stale parameters:
  - **Australia beat Türkiye 2–0** (model: 28.8% home win, 47.0% Türkiye win) — Australia's 2026 form was better than historical
  - **Ecuador beat Germany 2–1** (model: 24.2% Ecuador, 52.3% Germany) — Ecuador improving rapidly
- The re-fit after group stage showed **significant parameter shifts** for teams whose tournament performance diverged from historical:
  - Norway α: +0.126 (much more attacking than history suggested)
  - Senegal α: +0.101
  - Iraq β: +0.181 (much leakier than history suggested)
  - New Zealand β: +0.178

**Why this matters:**
- International football teams change drastically between World Cup cycles — squad turnover, coaching changes, tactical evolution
- The model weights by tournament importance but not by time recency
- An exponential time decay (e.g., half-life of 2 years) on top of the tournament weights would down-weight the 2014–2020 data appropriately

### Matchday 1 Was Especially Bad

- Brier = 0.606 on MD1 (vs 0.454 and 0.439 on MD2/MD3)
- 9 of 24 MD1 matches were draws (37.5%)
- This is partly variance (small sample) but also pattern: the first round of a World Cup historically produces more conservative, cautious football → more draws
- The model has no concept of tournament dynamics or round-specific behavior

---

## 6. What That Tells Me

### The model works, but with a known structural ceiling
- Brier of 0.50 beats random (0.667) by a comfortable margin
- 63.9% accuracy on most-likely-outcome is respectable for international football
- The independent Poisson assumption extracts meaningful signal from attack/defense decomposition

### The independent Poisson ceiling is real and quantifiable
- The independence assumption caps draw probability at ~30%, even for perfectly matched teams
- The actual draw rate (27.8%) consistently exceeds predicted draw probability (23.1%)
- **This is the single highest-leverage improvement**: implementing the Dixon-Coles ρ correction for low-scoring draws (0–0, 1–0, 0–1, 1–1) would directly address the biggest calibration gap
- The draw problem cascades: underestimated draws → overestimated favorite wins → inflated Brier score

### Home advantage exists even at "neutral" venues
- Predicted 42% home win rate, actual 47% — a consistent and correctable bias
- Adding a single multiplicative home advantage parameter is straightforward and would improve calibration

### Time decay is the next structural fix after Dixon-Coles
- The model treats a 2015 friendly the same as a 2025 friendly (given same tournament weight)
- Teams like Norway, Australia, and Ecuador showed tournament-vs-historical divergence that a time decay could partially capture
- The re-fit evidence confirms this: when group stage data was added (weight 1.0, most recent), significant parameter shifts occurred

### 13.9% exact scoreline accuracy is actually decent
- Given 121 possible scorelines (0–10 × 0–10), picking the right one 13.9% of the time (10/72) is well above the ~3% random baseline for realistic scorelines
- The Poisson model's scoreline predictions are one of its genuine strengths

### The model is honest about what it doesn't know
- Mean P(actual outcome) = 0.47 — the model assigns almost half its probability mass to the thing that actually happens
- For away wins specifically, P(actual) = 0.62 — the model is well-calibrated on decisive away results
- The gap is almost entirely on draws (P(actual) = 0.25 for draws vs 0.53/0.62 for wins)

---

## Appendix: Key File References

| File | Purpose |
|------|---------|
| [poisson_model.py](file:///c:/Users/avila/OneDrive/Escritorio/Summer%20Stuffs/match-predictor/src/poisson_model.py) | Model definition, MLE fitting, prediction function |
| [app.py](file:///c:/Users/avila/OneDrive/Escritorio/Summer%20Stuffs/match-predictor/app.py) | Streamlit dashboard with Brier/log-loss/reliability diagram |
| [group_stage_predictions_new.csv](file:///c:/Users/avila/OneDrive/Escritorio/Summer%20Stuffs/match-predictor/predictions/group_stage_predictions_new.csv) | 72 locked pre-tournament predictions |
| [actual_outcomes.csv](file:///c:/Users/avila/OneDrive/Escritorio/Summer%20Stuffs/match-predictor/results/actual_outcomes.csv) | 72 actual group stage results |
| [team_ratings_new.csv](file:///c:/Users/avila/OneDrive/Escritorio/Summer%20Stuffs/match-predictor/data/processed/team_ratings_new.csv) | Fitted α, β parameters for 48 teams |
| [matches_filtered.csv](file:///c:/Users/avila/OneDrive/Escritorio/Summer%20Stuffs/match-predictor/data/processed/matches_filtered.csv) | 1,433 historical matches used for training |
| [knockout_prediction_diff.md](file:///c:/Users/avila/OneDrive/Escritorio/Summer%20Stuffs/match-predictor/predictions/knockout_prediction_diff.md) | Re-fit parameter shifts after group stage |

---

*Compiled July 12, 2026 from actual project data. Numbers pulled directly from CSVs and computed via analysis scripts, not from memory.*
