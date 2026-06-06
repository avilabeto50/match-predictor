# Copa 2026 Match Predictor

A probabilistic match outcome prediction model for the 2026 FIFA World Cup, built around classical statistical methods from sports analytics and quantitative modeling. Given any two national teams, the model estimates win, draw, and loss probabilities and tracks how well-calibrated those predictions are as the tournament unfolds in real time.

## What This Is

This project is not about picking winners. It is about estimating a probability distribution over match outcomes and measuring the quality of those estimates against observed results. The core question is not "who will win?" but "how confident should we be, and are we right to be that confident?"

The modeling approach draws on three ideas that map directly to quantitative risk work:

**Team strength as a prior.** FIFA World Rankings and historically derived Elo ratings give a baseline estimate of relative team quality before a single ball is kicked. This is the same conceptual move as a baseline risk model — a structured prior built from observable data.

**Attack and defense decomposition.** Each team is characterized by two latent parameters: an offensive strength and a defensive vulnerability. These are estimated from historical international match results via maximum likelihood. The resulting decomposition is conceptually identical to decomposing credit risk into probability of default and loss given default — separating the two dimensions of a team's profile rather than collapsing them into a single number.

**Poisson goal model.** Goals scored by each team in a match are modeled as independent Poisson random variables, with rates determined by the attack and defense ratings of both teams. Win, draw, and loss probabilities are derived analytically from the joint Poisson distribution. This is a closed-form solution that directly teaches working with probability distributions in a real forecasting context.

**Calibration as the ground truth.** After each round of the tournament, predicted probabilities are evaluated against actual outcomes using reliability diagrams and Brier scores. A model that says "60% win probability" should win about 60% of the time across those predictions. Measuring this gap — and understanding why it exists — is the core analytical output of the project.

## Data Sources

All data used in this project is publicly available at no cost:

- **Historical international match results** (1872–present): the `martj42/international-football-results-from-1872-to-2017` dataset, available on GitHub and Kaggle. Approximately 45,000 matches covering all FIFA-recognized national teams.
- **FIFA World Rankings**: official ranking points published by FIFA, available via the FIFA website and mirrored on Kaggle as historical CSVs.

## Scope

The model covers all 104 matches of the 2026 FIFA World Cup (June 11 – July 19, 2026), including the group stage (72 matches across 12 groups of 4 teams), round of 32, quarterfinals, semifinals, and final. Predictions are generated before each round and evaluated after results are known.

48 teams compete across 12 groups (A–L), with the top 2 from each group plus the best 8 third-place finishers advancing to the knockout stage.

---

*Environment setup, model details, and usage instructions will be added as the project is built out. See `ROADMAP.md` for the full development plan.*
