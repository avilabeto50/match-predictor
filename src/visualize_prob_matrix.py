"""
Visualize the Poisson probability matrix for a given matchup.

Usage:
    python src/visualize_prob_matrix.py "Mexico" "Korea Republic"
"""

import sys
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.patches import FancyBboxPatch
from poisson_model import (
    load_and_weight_data,
    build_team_index,
    fit_poisson_model,
    poisson_pmf_log,
)


def build_prob_matrix(team_a, team_b, mu, alphas, betas, team_to_idx, max_goals=7):
    """Build an (max_goals x max_goals) probability matrix for a matchup."""
    idx_a = team_to_idx[team_a]
    idx_b = team_to_idx[team_b]

    lambda_a = mu * alphas[idx_a] * betas[idx_b]
    lambda_b = mu * alphas[idx_b] * betas[idx_a]

    matrix = np.zeros((max_goals, max_goals))
    for ga in range(max_goals):
        for gb in range(max_goals):
            log_p = poisson_pmf_log(ga, lambda_a) + poisson_pmf_log(gb, lambda_b)
            matrix[gb, ga] = np.exp(log_p)  # row = away goals, col = home goals

    return matrix, lambda_a, lambda_b


def plot_probability_matrix(matrix, team_a, team_b, lambda_a, lambda_b, save_path=None):
    """
    Plot the probability matrix as a heatmap with colored win/draw/loss regions.
    Styled after the reference image.
    """
    max_goals = matrix.shape[0]

    # --- Compute outcome probabilities ---
    p_win_a, p_draw, p_win_b = 0.0, 0.0, 0.0
    for ga in range(max_goals):
        for gb in range(max_goals):
            p = matrix[gb, ga]
            if ga > gb:
                p_win_a += p
            elif ga == gb:
                p_draw += p
            else:
                p_win_b += p

    # Find the most probable scoreline
    best_idx = np.unravel_index(np.argmax(matrix), matrix.shape)
    best_away, best_home = best_idx
    best_prob = matrix[best_idx]

    # --- Build the figure ---
    fig, ax = plt.subplots(figsize=(9, 8))
    fig.patch.set_facecolor('#fafafa')
    ax.set_facecolor('#fafafa')

    # --- Color the background regions ---
    # Home win region (below diagonal): light green
    # Draw region (diagonal): light yellow
    # Away win region (above diagonal): light red/salmon
    for ga in range(max_goals):
        for gb in range(max_goals):
            if ga > gb:      # home wins (col > row => below diagonal when row=away)
                color = '#d4edda'  # soft green
            elif ga == gb:
                color = '#fff3cd'  # soft yellow
            else:             # away wins
                color = '#f8d7da'  # soft red/salmon
            rect = plt.Rectangle((ga - 0.5, gb - 0.5), 1, 1,
                                 facecolor=color, edgecolor='white',
                                 linewidth=1.5, zorder=0)
            ax.add_patch(rect)

    # --- Draw probability values in each cell ---
    for ga in range(max_goals):
        for gb in range(max_goals):
            prob = matrix[gb, ga]
            pct = prob * 100

            if pct >= 1.0:
                fontsize = 11
                fontweight = 'bold'
                alpha = 1.0
            elif pct >= 0.1:
                fontsize = 9
                fontweight = 'normal'
                alpha = 0.7
            else:
                fontsize = 8
                fontweight = 'normal'
                alpha = 0.4

            text = f"{pct:.1f}%" if pct >= 0.1 else f"{pct:.2f}%"

            ax.text(ga, gb, text,
                    ha='center', va='center',
                    fontsize=fontsize, fontweight=fontweight,
                    color='#2c3e50', alpha=alpha, zorder=2)

    # --- Highlight the most probable scoreline ---
    highlight = FancyBboxPatch(
        (best_home - 0.42, best_away - 0.42), 0.84, 0.84,
        boxstyle="round,pad=0.05",
        facecolor='none', edgecolor='#2980b9',
        linewidth=2.5, zorder=3
    )
    ax.add_patch(highlight)

    # --- Axis setup ---
    ax.set_xlim(-0.5, max_goals - 0.5)
    ax.set_ylim(-0.5, max_goals - 0.5)
    ax.set_xticks(range(max_goals))
    ax.set_yticks(range(max_goals))
    ax.set_xlabel(f'{team_a} goals', fontsize=13, fontweight='bold', labelpad=10)
    ax.set_ylabel(f'{team_b} goals', fontsize=13, fontweight='bold', labelpad=10)
    ax.tick_params(axis='both', labelsize=11)
    ax.invert_yaxis()
    ax.set_aspect('equal')

    # Remove spines
    for spine in ax.spines.values():
        spine.set_visible(False)

    # --- Annotation box: most probable scoreline + outcome probs ---
    info_text = (
        f"Most likely: {best_home}–{best_away}  ({best_prob*100:.1f}%)\n"
        f"\n"
        f"P({team_a} win) = {p_win_a*100:.1f}%\n"
        f"P(Draw) = {p_draw*100:.1f}%\n"
        f"P({team_b} win) = {p_win_b*100:.1f}%"
    )

    props = dict(boxstyle='round,pad=0.6', facecolor='white',
                 edgecolor='#bdc3c7', alpha=0.95)
    ax.text(max_goals - 0.6, max_goals - 0.6, info_text,
            transform=ax.transData,
            fontsize=10, verticalalignment='bottom', horizontalalignment='right',
            bbox=props, zorder=5, family='monospace')

    # --- Legend for regions ---
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#d4edda', edgecolor='#aaa', label=f'{team_a} win region'),
        Patch(facecolor='#fff3cd', edgecolor='#aaa', label='Draw (diagonal)'),
        Patch(facecolor='#f8d7da', edgecolor='#aaa', label=f'{team_b} win region'),
    ]
    ax.legend(handles=legend_elements, loc='lower left', fontsize=9,
              framealpha=0.9, edgecolor='#bdc3c7')

    # --- Title ---
    fig.suptitle(f'Score Probability Matrix', fontsize=16, fontweight='bold', y=0.97)
    ax.set_title(f'{team_a}  vs  {team_b}    (λ_home={lambda_a:.2f}, λ_away={lambda_b:.2f})',
                 fontsize=12, color='#555', pad=12)

    plt.tight_layout(rect=[0, 0, 1, 0.94])

    if save_path:
        plt.savefig(save_path, dpi=200, bbox_inches='tight', facecolor=fig.get_facecolor())
        print(f"Saved to {save_path}")
    else:
        plt.show()

    plt.close(fig)


def main():
    team_a = sys.argv[1] if len(sys.argv) > 1 else "Mexico"
    team_b = sys.argv[2] if len(sys.argv) > 2 else "Korea Republic"
    max_goals = int(sys.argv[3]) if len(sys.argv) > 3 else 7

    filtered_path = os.path.join('data', 'processed', 'matches_filtered.csv')
    df = load_and_weight_data(filtered_path)
    teams, team_to_idx, idx_to_team = build_team_index(df)
    n_teams = len(teams)

    mu, alphas, betas = fit_poisson_model(df, team_to_idx, n_teams)

    matrix, lam_a, lam_b = build_prob_matrix(
        team_a, team_b, mu, alphas, betas, team_to_idx, max_goals=max_goals
    )

    save_path = os.path.join('report', f'prob_matrix_{team_a.lower().replace(" ", "_")}_vs_{team_b.lower().replace(" ", "_")}.png')
    plot_probability_matrix(matrix, team_a, team_b, lam_a, lam_b, save_path=save_path)


if __name__ == '__main__':
    main()
