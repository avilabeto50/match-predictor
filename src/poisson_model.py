import pandas as pd
import numpy as np
from scipy.optimize import minimize
from scipy.special import gammaln
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_and_weight_data(filtered_path):
    """Load matches_filtered.csv and assign weights by tournament type."""
    df = pd.read_csv(filtered_path)
    weight_map = {
        # ── Tier 1 : FIFA World Cup final tournament ──────────────────────────────
        'FIFA World Cup':                       1.00,
        'Confederations Cup':                   0.90,
        # ── Tier 2 : Continental championships (finals) ───────────────────────────
        'UEFA Euro':                            0.85,
        'Copa América':                         0.85,
        'African Cup of Nations':               0.85,
        'AFC Asian Cup':                        0.85,
        'Gold Cup':                             0.85,
        'Arab Cup':                             0.75,
        'Gulf Cup':                             0.75,
        'EAFF Championship':                    0.75,
        'WAFF Championship':                    0.75,
        'COSAFA Cup':                           0.70,
        'CAFA Nations Cup':                     0.70,
        # ── Tier 3 : Nations Leagues ──────────────────────────────────────────────
        'UEFA Nations League':                  0.80,
        'CONCACAF Nations League':              0.80,
        # ── Tier 4 : World Cup & continental qualification ────────────────────────
        'FIFA World Cup qualification':         0.75,
        'UEFA Euro qualification':              0.65,
        'African Cup of Nations qualification': 0.65,
        # ── Tier 5 : FIFA-sanctioned series / bilateral ───────────────────────────
        'FIFA Series':                          0.60,
        'Superclásico de las Américas':         0.65,
        # ── Tier 6 : Invitational / regional cups ─────────────────────────────────
        'Kirin Challenge Cup':                  0.50,
        'Kirin Cup':                            0.50,
        'Al Ain International Cup':             0.50,
        'Canadian Shield':                      0.50,
        'Soccer Ashes':                         0.50,
        # ── Tier 7 : Friendlies ───────────────────────────────────────────────────
        'Friendly':                             0.50,
    }
    # Raise immediately if any tournament in the data isn't in the map
    unknown = set(df['tournament'].unique()) - set(weight_map.keys())
    if unknown:
        raise ValueError(f"Unknown tournaments (add to weight_map): {unknown}")
    df['weight'] = df['tournament'].map(weight_map)
    logger.info(f"Loaded {len(df)} matches. Weight distribution:\n{df['weight'].value_counts()}")
    return df


def build_team_index(df):
    """Extract unique teams and create bidirectional mapping."""
    teams = pd.concat([df['home_team'], df['away_team']]).unique()
    teams = sorted(teams)
    team_to_idx = {t: i for i, t in enumerate(teams)}
    idx_to_team = {i: t for t, i in team_to_idx.items()}
    logger.info(f"Found {len(teams)} unique teams")
    return teams, team_to_idx, idx_to_team


def poisson_pmf_log(k, lam):
    """Log PMF of Poisson(lam) at k. Uses gammaln for numerical stability."""
    if lam <= 0:
        return -1e10
    return k * np.log(lam) - lam - gammaln(k + 1)


def neg_log_likelihood(params, n_teams, matches_data, team_to_idx):
    """
    Compute weighted negative log-likelihood with regularization to break scaling symmetry.
    params = [mu, alpha_1, ..., alpha_n, beta_1, ..., beta_n]
    """
    mu = params[0]
    alphas = params[1:n_teams + 1]
    betas = params[n_teams + 1:]

    # Hard bounds check
    if mu < 0.01 or mu > 10:
        return 1e10
    if np.any(alphas < 0.01) or np.any(alphas > 10):
        return 1e10
    if np.any(betas < 0.01) or np.any(betas > 10):
        return 1e10
    if np.any(np.isnan(alphas)) or np.any(np.isnan(betas)):
        return 1e10

    nll = 0.0
    for _, row in matches_data.iterrows():
        home_idx = team_to_idx[row['home_team']]
        away_idx = team_to_idx[row['away_team']]
        weight = row['weight']

        goals_home = int(row['home_score'])
        goals_away = int(row['away_score'])

        lambda_home = mu * alphas[home_idx] * betas[away_idx]
        lambda_away = mu * alphas[away_idx] * betas[home_idx]

        # Safety check on lambdas
        if lambda_home <= 0 or lambda_away <= 0:
            return 1e10

        log_prob_home = poisson_pmf_log(goals_home, lambda_home)
        log_prob_away = poisson_pmf_log(goals_away, lambda_away)

        # Check for nan in likelihood
        if np.isnan(log_prob_home) or np.isnan(log_prob_away):
            return 1e10

        nll -= weight * (log_prob_home + log_prob_away)

    # Regularization: penalize deviation from mean of 1.0 in log-space
    # This forces the geometric mean of alphas and betas to be 1.0, breaking scaling symmetry
    log_alphas = np.log(np.clip(alphas, 0.01, 10))
    log_betas = np.log(np.clip(betas, 0.01, 10))
    
    # Sum of squared deviations from 0 (in log space), normalized
    reg_strength = 300.0
    reg = reg_strength * (np.sum(log_alphas ** 2) + np.sum(log_betas ** 2)) / (2 * n_teams)

    total = nll + reg
    
    if np.isnan(total) or np.isinf(total):
        return 1e10
    
    return total


def fit_poisson_model(df, team_to_idx, n_teams):
    """Fit attack/defense model using MLE with L-BFGS-B."""
    logger.info("Starting MLE optimization...")

    n_params = 1 + 2 * n_teams
    x0 = np.ones(n_params)
    x0[0] = 1.4

    bounds = [(0.5, 3.0)] + [(0.3, 3.0)] * (2 * n_teams)

    result = minimize(
        neg_log_likelihood,
        x0,
        args=(n_teams, df, team_to_idx),
        method='L-BFGS-B',
        bounds=bounds,
        options={'maxiter': 500, 'ftol': 1e-6}
    )

    logger.info(f"Optimization complete. Success: {result.success}, NLL: {result.fun:.2f}")

    params = result.x
    
    # Safety check: if optimization failed or produced nan, return sensible defaults
    if not result.success or np.any(np.isnan(params)):
        logger.warning("Optimization failed or produced nan. Using default parameters (all 1.0).")
        mu = 1.4
        alphas = np.ones(n_teams)
        betas = np.ones(n_teams)
    else:
        mu = params[0]
        alphas = params[1:n_teams + 1]
        betas = params[n_teams + 1:]

        # Normalize alphas to have geometric mean 1.0
        alpha_gmean = np.exp(np.mean(np.log(alphas)))
        alphas = alphas / alpha_gmean
        mu = mu * alpha_gmean
        
        # Normalize betas to have geometric mean 1.0
        beta_gmean = np.exp(np.mean(np.log(betas)))
        betas = betas / beta_gmean
        mu = mu * beta_gmean

        logger.info(f"After normalization: mu={mu:.4f}, alpha_mean={np.mean(alphas):.4f}, beta_mean={np.mean(betas):.4f}")

        # Clip to reasonable bounds
        alphas = np.clip(alphas, 0.4, 2.5)
        betas = np.clip(betas, 0.4, 2.5)
        mu = np.clip(mu, 0.8, 2.0)

        logger.info(f"After clipping: mu={mu:.4f}, alpha range=[{alphas.min():.2f}, {alphas.max():.2f}], beta range=[{betas.min():.2f}, {betas.max():.2f}]")

    return mu, alphas, betas


def predict_match(team_a, team_b, mu, alphas, betas, teams, team_to_idx):
    """
    Predict match outcome. Returns dict with win/draw/loss probs and top 5 scorelines.
    """
    idx_a = team_to_idx[team_a]
    idx_b = team_to_idx[team_b]

    lambda_a = mu * alphas[idx_a] * betas[idx_b]
    lambda_b = mu * alphas[idx_b] * betas[idx_a]

    p_win_a = 0.0
    p_draw = 0.0
    p_win_b = 0.0
    scorelines = []

    for g_a in range(11):
        for g_b in range(11):
            prob = np.exp(poisson_pmf_log(g_a, lambda_a) + poisson_pmf_log(g_b, lambda_b))
            scorelines.append((g_a, g_b, prob))

            if g_a > g_b:
                p_win_a += prob
            elif g_a == g_b:
                p_draw += prob
            else:
                p_win_b += prob

    scorelines.sort(key=lambda x: x[2], reverse=True)
    top_5 = scorelines[:5]

    return {
        'team_a': team_a,
        'team_b': team_b,
        'p_win_a': p_win_a,
        'p_draw': p_draw,
        'p_win_b': p_win_b,
        'top_scorelines': top_5
    }


def main():
    """Main entry point: load data, fit model, save results."""

    filtered_path = 'data/processed/matches_filtered.csv'
    output_path = 'data/processed/team_ratings_new.csv'

    df = load_and_weight_data(filtered_path)
    teams, team_to_idx, idx_to_team = build_team_index(df)
    n_teams = len(teams)

    mu, alphas, betas = fit_poisson_model(df, team_to_idx, n_teams)

    results = []
    for i, team in enumerate(teams):
        results.append({
            'team': team,
            'attack': alphas[i],
            'defense': betas[i],
            'mu': mu
        })

    results_df = pd.DataFrame(results)
    results_df.to_csv(output_path, index=False)
    logger.info(f"Team ratings saved to {output_path}")

    logger.info("\n=== Sanity Checks ===")
    try:
        pred_arg_sau = predict_match('Argentina', 'Saudi Arabia', mu, alphas, betas, teams, team_to_idx)
        logger.info(f"Argentina vs Saudi Arabia: P(win)={pred_arg_sau['p_win_a']:.3f}, P(draw)={pred_arg_sau['p_draw']:.3f}, P(loss)={pred_arg_sau['p_win_b']:.3f}")
        logger.info(f"  Top scoreline: {pred_arg_sau['top_scorelines'][0]}")

        pred_fra_bra = predict_match('France', 'Brazil', mu, alphas, betas, teams, team_to_idx)
        logger.info(f"France vs Brazil: P(win)={pred_fra_bra['p_win_a']:.3f}, P(draw)={pred_fra_bra['p_draw']:.3f}, P(loss)={pred_fra_bra['p_win_b']:.3f}")
    except KeyError as e:
        logger.warning(f"Sanity check failed: team {e} not in dataset")

    return results_df


if __name__ == '__main__':
    main()