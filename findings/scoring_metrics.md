# Scoring Metrics — Brier Score & Log Loss Comparison

## Overview

All models are evaluated on the same **72 completed group-stage matches** of the 2026 FIFA World Cup. Two proper scoring rules are used:

- **Brier Score** — measures the mean squared error of the probability vector against the one-hot outcome vector. Range: 0 (perfect) to 2 (maximally wrong). A model predicting 1/3 for each outcome scores exactly 2/3 ≈ 0.6667.
- **Log Loss** — measures the negative log-likelihood of the observed outcome under the model's predicted probabilities. Range: 0 (perfect) to +∞. Penalizes confident wrong predictions much more heavily than Brier. A uniform model scores ln(3) ≈ 1.0986.

Both metrics are **lower-is-better**.

---

## Results

| Model | Brier Score | Log Loss | Brier vs Uniform | Log Loss vs Uniform |
|-------|------------|----------|-----------------|-------------------|
| **Poisson MLE** | **0.4996** | **0.8478** | −25.1% | −22.8% |
| Elo-Gap Baseline | 0.5313 | 0.9013 | −20.3% | −18.0% |
| Base-Rate Baseline | 0.6402 | 1.0617 | −4.0% | −3.4% |
| Uniform Baseline | 0.6667 | 1.0986 | — | — |

> [!NOTE]
> "Brier vs Uniform" and "Log Loss vs Uniform" columns show percentage improvement over the uninformative 1/3-each baseline.

---

## Model Descriptions

### Uniform Baseline
Assigns equal probability to all three outcomes: P(home win) = P(draw) = P(away win) = 1/3. This is the theoretical floor — any model with domain knowledge should beat it. Serves as a sanity check.

- Brier = 6/9 = 0.6667 (exact)
- Log Loss = ln(3) ≈ 1.0986 (exact)

### Base-Rate Baseline
Uses historical outcome proportions from the training set (all international matches in `matches_filtered.csv` prior to the 2026 tournament):

| Outcome | Historical Rate |
|---------|----------------|
| Home Win | 45.6% |
| Draw | 25.1% |
| Away Win | 29.3% |

These proportions are applied identically to every match — no per-match differentiation. The base-rate baseline only marginally beats uniform because the training-set outcome distribution is close to uniform to begin with.

> [!IMPORTANT]
> The base-rate uses only pre-tournament training data. An earlier version of `app.py` leaked 2026 outcomes into this calculation; that was fixed by loading historical rates from `elo_match_log.csv` with a date cutoff of 2026-06-11.

### Elo-Gap Baseline
Maintains a running Elo rating per team (K-factor scaled by tournament tier), computes the absolute Elo gap between opponents, and maps that gap to outcome probabilities via historically observed bucket frequencies:

| Elo Gap Bucket | P(Favored Win) | P(Draw) | P(Underdog Win) | N (training) |
|----------------|---------------|---------|-----------------|-------------|
| 0–50 | 42.3% | 27.9% | 29.8% | (large) |
| 50–100 | 47.0% | 25.0% | 28.0% | (large) |
| 100–150 | 52.7% | 23.7% | 23.6% | (large) |
| 150–200 | 56.8% | 23.2% | 20.0% | (moderate) |
| 200+ | 63.3% | 18.4% | 18.3% | (moderate) |

This baseline adds per-match information (team strength differential) but uses a coarse mapping. It meaningfully beats the static baselines but is substantially worse than the Poisson model.

### Poisson MLE (Primary Model)
Fits per-team attack (α) and defense (β) ratings via maximum likelihood estimation on historical match scorelines, with exponential time-decay weighting. For each match, computes expected goals as λ = μ · α_attack · β_defense for each side, then derives win/draw/loss probabilities by summing over the bivariate Poisson scoreline grid (0–10 × 0–10 goals).

This model captures per-team offensive and defensive quality at a much finer granularity than the Elo-gap approach, resulting in the best scores across both metrics.

---

## Interpretation

1. **The Poisson model meaningfully outperforms all baselines.** A 25% Brier improvement and 23% log loss improvement over uniform is substantial for a 3-outcome prediction problem with 72 matches.

2. **Elo-gap sits cleanly between the static baselines and Poisson.** This is expected — Elo captures team strength ordering but loses information about the magnitude of offensive/defensive asymmetries that the Poisson attack/defense decomposition preserves.

3. **Base-rate barely beats uniform.** This confirms that the historical outcome distribution (≈46/25/29) is close enough to uniform (33/33/33) that simply knowing "home wins more often" adds almost no predictive value — consistent with the finding that home/away labels are meaningless for this tournament.

4. **Log loss separates the models slightly more than Brier.** The Poisson model's log loss advantage over Elo-gap (0.8478 vs 0.9013, Δ = 0.0535) is relatively larger than its Brier advantage (0.4996 vs 0.5313, Δ = 0.0317). This suggests the Poisson model's edge comes partly from being more confidently correct on its best predictions — exactly the kind of signal log loss rewards.

---

*Values computed from `predictions/group_stage_predictions_new.csv`, `predictions/elo_gap_baseline_predictions.csv`, and `results/actual_outcomes.csv` over 72 group-stage matches.*
