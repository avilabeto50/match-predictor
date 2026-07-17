# Betting on the Underdog? - TITLE WIP

I've recently been introduced to the world of [Kalshi](https://www.kalshi.com/), a prediction market where users can bet on the outcome of events. Im not much of an avid gambler, but i became interested in the "why" behind certain predictions and how one could gain an edge in these markets. During the NBA finals  I saw the Knicks go from 4% to winning the game during multiple games in the series. This got me thinking on whether that was just bad luck, or whether the market's confidence was actually miscalibrated.

With the World Cup around the corner, this seemed like a great opportunnity to learn more about what is going on under the hood of these prediction markets for soccer games. I got curious about how exactly can you assign a number to the outcome of a soccer match. Sure, prediction markets have their own dynamics (liqudity, arbitrage, herding behavior, etc), but at the core it all comes down to the likelyness of an outcome... right?

<!-- insert kashi image -->


#### What I did
I decided to build a simple statistical model, based on the Poisson distribution, that estimates each participating team's attack and defense strength based on historical matches. Then, I used these estimates to predict the probability of each team winning, losing, or drawing the match. Finally, I compared my model's predictions to the actual outcomes of the matches and evaluated its performance. This is by no means a production level model. It served better as a great "Hello World" to modeling and sport statistics. This write up is meant to be a short companion to my project, highlighting my train of thought, the process, insights on the results, and some limitations for possible improvements.

## Data

I sourced raw data from [martj42's international results](https://github.com/martj42/international_results). This repo contains the results of 49,459 matches since 1872. I filtered this data to only include matches played since 2014. This gives exactly 3 world cup cycles of data (2014, 2018, 2022) plus all the matches in between. Then, I went into [FIFA's](https://www.fifa.com/en/tournaments/mens/worldcup/canadamexicousa2026/teams) website and extracted the teams for the 2026 world cup along with the fixtures of all 72 group stage matches. 

<!-- csv with all wc teams and fixtures here -->

With these processed data sets, I could now look at any historical matchups between any two countries that will be articipating in the 2026 world cup. This left 1,433 matches to work with.

<!-- csv with filtered matches here -->

## The Model

Given my background as a theorist, I reached straight into my "probability and statistics" bag. I wanted to really understand the fundamental concepts rather than just calling some function I wont know the math behind. I first tried to "decompose" the question of "who will win" into simpler, more tangible variables. Intuitively, the winner of the game is the team who scores more goals. So, the question becomes: "who is more likely to score more goals?". Goals in a match are discrete, rare (3+ in a game is already surprising), and somewhat independent events. This screams "Poisson Distribution".

Recall that a random variable $X$ counting the number of events in a fixed interval follows a Poisson distribution if two conditions hold:
1. The events occur independently,
2. The probability of an event occurring is constant throughout the interval.

The independence condition is one to consider. It means that the number of goals scored by a team in a match is independent of the number of goals scored by the other team. This is not entirely true in reality. Often teams adjust tactics after a goal, and momentum does play a huge role in soccer. However, it is a reasonable starting approximation for the sake of simplicity.


### Attack and Defense parameters

Given our assumptions, the next logical step is to quantify the expected number of goals each team will score. Intuitively, the number of goals a team scores depends on two factors: how strong their attack is, and how strong the opponent's defense is. We can capture this using a simple multiplicative model:

$$ \text{E}[X_{ij}] = \lambda_{ij} = \mu \cdot \alpha_i \cdot \beta_j $$

where:
- $X_{ij}$ is the number of goals scored by team $i$ against team $j$.
- $\lambda_{ij}$ is the expected number of goals for team $i$ against team $j$.
- $\mu$ is a global average rate of scoring, representing the baseline scoring rate for an average team against an average defense.
- $\alpha_i$ is a parameter representing the relative attacking strength of team $i$ compared to the average.
- $\beta_j$ is a parameter representing the relative defensive weakness of team $j$ against the average. A higher value indicates a weaker defense (more goals conceded).

To estimate these parameters, we use maximum likelihood estimation (MLE). We define the likelihood of observing the actual score $S_{ij} = (x_{ij}, y_{ij})$ as the product of the probabilities of observing each score, based on the Poisson distribution:

$$ P(X_{ij}=x_{ij}, X_{ji}=y_{ij} | \alpha, \beta) = \frac{e^{-\lambda_{ij}}\lambda_{ij}^{x_{ij}}}{x_{ij}!} \cdot \frac{e^{-\lambda_{ji}}\lambda_{ji}^{y_{ij}}}{y_{ij}!} $$

### Building the Likelihood

A single match does not tell us much. It's only by looking across many matches, against many different opponents, that the model can start to isolate each team's attack and defense strength individually, and start to estimate these parameters with any confidence. Assuming matches are independent, the likelihood of the whole dataset is the product of the likelihood of each match. However, not every match should be counted the same. Friendies and Invitationals do not carry the same weight as a World Cup match. To account for this, we introduce a weighting system based on the "tournament" attribute for each game. Let $w_{ij}$ represent the weight for the match between teams $i$ and $j$, assigned as follows:

| Tier | Competition | Weight |
|------|-------------|--------|
| 1 — World Cup Final Tournament | FIFA World Cup | 1.00 |
|  | Confederations Cup | 0.90 |
| 2 — Continental Championships | UEFA Euro | 0.85 |
|  | Copa América | 0.85 |
|  | African Cup of Nations | 0.85 |
|  | AFC Asian Cup | 0.85 |
|  | Gold Cup | 0.85 |
|  | Arab Cup | 0.75 |
|  | Gulf Cup | 0.75 |
|  | EAFF Championship | 0.75 |
|  | WAFF Championship | 0.75 |
|  | COSAFA Cup | 0.70 |
|  | CAFA Nations Cup | 0.70 |
|   3 — Nations Leagues | UEFA Nations League | 0.80 |
|  | CONCACAF Nations League | 0.80 |
| 4 — WC & Continental Qualification | FIFA World Cup qualification | 0.75 |
|  | UEFA Euro qualification | 0.65 |
|  | AFCON qualification | 0.65 |
| 5 — FIFA-Sanctioned / Bilateral | FIFA Series | 0.60 |
|  | Superclásico de las Américas | 0.65 |
| 6 — Invitational / Regional Cups | Kirin Challenge Cup | 0.50 |
|  | Kirin Cup | 0.50 |
|  | Al Ain International Cup | 0.50 |
|  | Canadian Shield | 0.50 |
|  | Soccer Ashes | 0.50 |
| 7 — Friendlies | Friendly | 0.50 |

Now, we can write the likelihood of the whole dataset $L(\alpha, \beta, \mu)$ as the product of the individual likelihoods, weighted by the tournament weights $w_{ij}$:

$$ L(\alpha, \beta, \mu) = \prod_{(i,j) \in \text{Matches}} P(X_{ij}=x_{ij}, X_{ji}=y_{ij} | \alpha, \beta, \mu)^{w_{ij}} $$

### Fitting the Model

Multiplying all those probabilities gets messy fast. The product shrinks so fast that the likelihood is extremely close to zero. A common mathematical "trick" to get around this is to work in log-space. The log transform turns a product into a sum, which allows the product of 1433 probabilities to stay in a much better shape. This log-likelihood is mathematically equivalent to the original likelihood, whatever values of $\alpha, \beta, \mu$ maximize the likelihood also maximize the log-likelihood. By convention, this is usually flipped into a negative log-likelihood, turning "find the maximum" into "find the minimum". Taking the log of the weighted likelihood turns each match's exponent $w_{ij}$ into a coefficient instead so the full negative log-likelihood becomes:

$$\text{NLL}(\alpha, \beta, \mu) = - \sum_{(i,j) \in \text{Matches}} w_{ij} \cdot \left[ \ln \left( \frac{e^{-\lambda_{ij}}\lambda_{ij}^{x_{ij}}}{x_{ij}!} \right) + \ln \left( \frac{e^{-\lambda_{ji}}\lambda_{ji}^{y_{ij}}}{y_{ij}!} \right) \right] $$



As written, the model is underdetermined: doubling every team's α while halving every team's β leaves every λij, and therefore every prediction, completely unchanged. There are infinitely many equally good solutions sitting on this scaling symmetry, and nothing in the likelihood itself picks one. To break the tie, I add a regularization term that penalizes parameters for drifting away from a shared baseline, effectively anchoring the average team's α and β near 1. It's a small addition, but I'll come back to it later. To actually implement this, I used `scipy.optimize.minimize` with the `L-BFGS-B` method. This explores the roughly 97-dimensional space of α, β, and μ values and locates whichever combination minimizes the (regularized) negative log-likelihood. The output is a single pair of numbers for every team, the attack strength (alpha) and defensive weakness (beta), which is all the model needs to predict any matchup at all.
 
<!-- INSERT SCATTER PLOT OF alpha vs beta here  -->

<!-- INSERT csv with alpha and beta for each team here. -->


### Predicting a Match

With the parameters estimated, we can now predict the probability of any score (x, y) in a match between any two teams, say team i and team j. We can define the expected number of goals for team i against team j as:

$$ \lambda_{ij} = \mu \cdot \alpha_i \cdot \beta_j $$

This defines a full Poisson distribution over how many goals each team might score. Since we assume that the goals scored by each team are independent events, the probability of a specific score (x, y), for example 2-1, is given by:

$$ P(X_{ij}=2, X_{ji}=1) = \frac{e^{-\lambda_{ij}}\lambda_{ij}^{2}}{2!} \cdot \frac{e^{-\lambda_{ji}}\lambda_{ji}^{1}}{1!} $$

With this, we can calculate the probability of any score for a given matchup. I cap this at 10-10 since anything higher is negligible. Once every scoreline has a probability, win, draw, and loss probabilities fall out as sums over three groups of scorelines that add up to 1: every scoreline where team i scores more goals contributes to i's win probability, every scoreline where the two teams tie contributes to the draw probability, and every scoreline where team j scores more contributes to j's win probability. The model never directly computes 'the probability team i wins'. It's derived from enumerating every possible final score and adding them up. The output is a clean P(win), P(draw), P(loss) for any matchup, along with whichever single scoreline turned out the most probable.

<!-- insert figure with a probability matrix for a given match -->

## Results

### The Predictions

Running the model over the 72 group stage matches of the 2026 FIFA World Cup yielded the following predictions:

<!-- INSERT TABLE OF ALL 72 MATCH PREDICTIONS HERE -->

Now, I compare these predicted probabilities against 3 different baselines, each knowing progressively more about the two teams involved:
1. a uniform baseline that knows nothing and predicts 1/3 for every outcome,
2. a base-rate baseline that knows the historical split of outcomes, but nothing about the two specific teams playing,
3. an Elo-gap baseline that knows the two teams' relative strength, but nothing about their attacking or defensive tendencies specifically.

A team's elo is calcuated historically. Every team starts at 1500 and a team's elo is updated after each match based on the outcome of the match and the elo of the opponent. The formula for updating a team's elo is as follows:

$$ ELO_{new} = ELO_{old} + K \cdot (S - E) $$

where:

- $ELO_{new}$ is the new elo rating
- $ELO_{old}$ is the old elo rating
- $K$ is the k-factor, which is the maximum change in elo for a single match. This scales with importance of competition similar to the weights used in the Poisson MLE model. For example, friendlies use $K=20$ while world cup matches use $K=60$.
- $S$ is the actual score of the match (1 for a win, 0.5 for a draw, 0 for a loss)
- $E$ is the expected score of the match, calculated as:
  $$ E = \frac{1}{1 + 10^{(ELO_{opponent} - ELO_{self})/400}} $$


<!-- insert the elo ratings here -->
<!-- insert the elo predictions here -->


### Metrics

I use 2 different metrics to compare the different predictions, Brier score and Log-loss. These two are metrics where a model can't improve its score by beign overconfident, only by actually being closer to what happened. I'll also analyze the calibration of the Poisson MLE model, to check whether its stated confidence was actually trustworthy.

The *Brier score* is a measure of the mean squared error of the probability vector against the one-hot outcome vector.  For a 3-outcome match, the formula sums the squared error across all three outcomes, then averages across matches:

$$ Brier Score = \frac{1}{N} \sum_{t=1}^{N} \sum_{j=1}^{3} (P_{j,t} - O_{j,t})^2 $$

where $N$ is the number of matches, $P_{j,t}$ is the predicted probability of outcome $j$ in match $t$, and $O_{j,t}$ is 1 if outcome $j$ actually happened and 0 otherwise. The Brier score ranges from 0 (perfect) to 2 (maximally wrong), with a uniform 1/3-each prediction scoring exactly 6/9 ≈ 0.6667. This serves as a useful baseline for comparison. Lower is better.

The *log loss* also compares predicted probabilities against the one-hot outcome vector, but instead of squared distance, it looks at negative log-probability:
$$ \text{Log Loss} = -\frac{1}{N}\sum_{t=1}^{N}\sum_{j=1}^{3} O_{tj}\log(P_{tj}) $$

Because $O_{tj}$ is 1 for exactly one outcome per match and 0 for the other two, this inner sum collapses to a single term: the log of whatever probability the model assigned to the outcome that actually happened.

$$ \text{Log Loss} = -\frac{1}{N}\sum_{t=1}^{N}\log(P_t) $$

where $P_t$ is the predicted probability the model assigned to the actual outcome of match $t$. Log loss ranges from 0 (perfect) to +∞, and punishes confident wrong predictions far more severely than Brier score does (this is because the log of a small number is a large negative number). A uniform model scores exactly ln(3) ≈ 1.0986.

The table below summarizes the scores for the 2026 FIFA World Cup group stage predictions:

| Model | Brier Score | Log Loss | Brier vs Uniform | Log Loss vs Uniform |
|-------|------------|----------|-----------------|-------------------|
| Uniform Baseline | 0.6667 | 1.0986 | — | — |
| Base-Rate Baseline | 0.6402 | 1.0617 | −4.0% | −3.4% |
| Elo-Gap Baseline | 0.5313 | 0.9013 | −20.3% | −18.0% |
| **Poisson MLE** | **0.4996** | **0.8478** | −25.1% | −22.8% |

The Poisson MLE outperformed all other models in both metrics, reducing the Brier score from 0.5313 (Elo-gap) to 0.4996, and log loss from 0.9013 to 0.8478. This demonstrates that explicitly modeling goal-scoring probabilities with a Poisson distribution gets more accurate predictions than models that only consider the relative strength of the two teams or historical base rates. However, a big jump also comes from the Elo-gap baseline over the uniform and base-rate baselines, which shows the importance of accounting for relative team strength. This improved the Brier score by roughly 17%. Going from Elo-gap to the full Poisson model, with its explicit attack and defense parameters, bought another 6%. This tells us that knowing the relative strength of two teams does most of the work in predicting the outcome of a match, and the Poisson model's extra improvement comes from capturing that information through the attack and defense parameters, instead of an elo rating. 

One thing worth noting is that the base rate baseline was built from historical home/away outcome rates. World cup matches usually do not have home advantage. Most games can be considered neutral, unless rare cases like the host country games. The fixture's "home" and "away" designation is just a formality. I kept this baseline rather than discard it because it shows that the home/away win spread is not much different from uniform (a 4% Brier score improvement and 3.4% log loss improvement over uniform). This is also the reason that motivated an elo-gap baseline, as this would account for which country has an edge even in neutral locations, which is exactly what we're trying to predict with these models.

### Calibration

Beating the baselines tells us how good the model was on average, but it does not tell us whether the model's stated confidence was actually honest. A model with a good Brier score can still be consistently wrong about one specific kind of match. That mistake just gets averaged away and hidden inside a the good brier score. Calibration checks for this directly. The idea is: Pull out every match where the model said something like "20 to 25% chance of a draw" , and check what fraction of those matches actually ended in a draw. If the model is honest, the two numbers should be close.

| Outcome | Predicted | Observed | Diff |
|---|---|---|---|
| Favorite Win | 56.5% | 63.9% | +7.4% |
| Underdog Win | 20.5% | 8.3% | −12.1% |
| Draw | 23.1% | 27.8% | +4.7% |

The model underrates draws by about 5 points and overrates underdogs by a much larger margin, about 12 points. These two patterns are connected. Since a model's three probabilities for any match always sum to 1, on average, whatever probability the model is failing to assign to draws has to be landing somewhere else, and most of it appears to be landing on underdogs.

Looking more in depth, I wanted to which matches were being miscalibrated. I grouped matches together into buckets by predicted probability, say every match where the model said something between 20% and 25%. I bucket by favorite and underdog. Whichever team the model gave a higher win probability is the favorite, the other is the underdog. This label comes from the model's own prediction, not from the fixture sheet.

#### Favorite Win
[chart]
| Bucket | N | Predicted | Observed | Diff |
|---|---|---|---|---|
| 0-45% | 19 | 40.3% | 47.4% | +7.1% |
| 45-50% | 8 | 47.4% | 50.0% | +2.6% |
| 50-60% | 18 | 54.9% | 83.3% | +28.5% |
| 60-70% | 12 | 64.2% | 41.7% | −22.5% |
| 70-100% | 15 | 77.7% | 86.7% | +9.0% |
#### Underdog Win
[chart]
| Bucket | N | Predicted | Observed | Diff |
|---|---|---|---|---|
| 0-10% | 13 | 7.1% | 0.0% | −7.1% |
| 10-20% | 23 | 15.6% | 0.0% | −15.6% |
| 20-25% | 10 | 22.9% | 10.0% | −12.9% |
| 25-30% | 14 | 28.3% | 21.4% | −6.8% |
| 30-100% | 12 | 33.0% | 16.7% | −16.3% |
#### Draw
[chart]
| Bucket | N | Predicted | Observed | Diff |
|---|---|---|---|---|
| 0-20% | 19 | 15.6% | 21.1% | +5.5% |
| 20-25% | 21 | 22.9% | 28.6% | +5.7% |
| 25-30% | 29 | 27.2% | 24.1% | −3.0% |


<small>(The 30%+ draw bucket only had 3 matches, too small to draw any conclusion from, so it is left out here.)</small>


Two patterns hold up across the buckets. The model consistently overrates underdogs. Every reliable underdog bucket predicts a higher win rate than actually occurred. The model also consistently underrates draws, in two of three reliable buckets. The largest single gap in the tables, the 50-60% favorite bucket, is really the same pattern seen most sharply. Right around a coin flip is exactly where underdog overconfidence shows up most, since that's where the model's uncertainty about who's actually better is highest. As it turns out, the top 10 worst predictions were all draws 

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

### Insights

The two issues, overrating underdogs and underrating draws, are connected. The Poisson model treats each team's goals as independent of the other's. This is the same independence assumption flagged at the start, and it turns out to matter. Real teams do not play that way. A team that is winning protects a lead by parking the bus, and a team that is behind pushes forward, which pulls both teams' scores toward each other more than independence allows. That tendency would explain both patterns at once: draws happen more than the model expects because of this score convergence, and underdogs win less than the model expects because their occasional big result is treated as independent of the favorite tightening up, when in practice it usually is not.

## Discussion: Limitations and Other directions

There are many more metrics, observations, and additions that could be made to this project. A lot of limitations also need to be addressed. For example:

- This year's group stage gave me only 72 matches to evaluate against, and this particular World Cup was a very crazy one. With that few matches, and several calibration buckets holding as few as 3 to 10 of them, it is hard to fully separate a real bias in the model from ordinary noise in an unusually upset-heavy tournament.
- I filtered the historical dataset down to matches between two teams that both qualified for the 2026 World Cup. In hindsight, this probably removed a lot of useful data. Any historical match where a 2026 team played a team that did not qualify still carries information about that team's attack and defense strength. This probably hurt the weaker team's as their matches against stronger teams likely worsened their scores. Keeping every match involving at least one 2026 team would have given every team's parameters a more complete score.
- A natural next step would be folding in some player-level signal, like the availability of a team's top scorers or defenders, rather than treating every team as a single fixed number. Players like Messi, Mbappé, or Haaland are key players that can change the game with multiple goals. Having them weighted in the model would likely improve the model's predictions.
-  I also want to try blending in Elo ratings directly, since the Elo-gap baseline did a lot of the predictive work almost on its own, and combining that along with the attack and defense structure might close some of the gap without needing a more complicated model

Although it was a very simple and straightforward approach, I was surprised at how well it performed! By no means is it perfect. But it allowed me to get a good grasp of the fundamentals of modeling in a real world context. I look forward to learning more about this topic and building more cool projects like this.

Future me talking: Something I found interesting is the way the odds change during a live event. For Instance, when a penalty is called for a team, the odds of them winning increase. my naive thought process was "oh, their odds increase because they basically score a goal" but it's more nuanced than that. Those odds bake in a lot of factors, like the probability of a player missing a penalty, the probabilitites of the goalie blocking the penalty, the time left in the game, etc. There's a lot of "hidden" factors that are not immediately obvious. This is what makes modeling such an interesting topic to me. Stay tuned for a penalty prediction model ....