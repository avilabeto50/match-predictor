"""
simulate_bracket.py
Runs the full bracket simulation with BOTH model states:
  - Frozen model (pre-tournament parameters, historical data only)
  - Re-fitted model (historical + 72 group stage results)

Outputs:
  predictions/full_bracket_frozen.csv
  predictions/full_bracket_refitted.csv
  predictions/full_bracket_comparison.md  -- side-by-side diff of every match
"""

import sys
import pandas as pd
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))
from poisson_model import predict_match
import logging
logging.disable(logging.CRITICAL)   # silence poisson_model INFO logs

BASE = Path(__file__).parent

# ─────────────────────────────────────────────────────────────────────────────
# Load both parameter sets
# ─────────────────────────────────────────────────────────────────────────────
def load_model(csv_path):
    df = pd.read_csv(csv_path)
    mu      = float(df["mu"].iloc[0])
    teams   = df["team"].tolist()
    alphas  = df["attack"].values
    betas   = df["defense"].values
    t2i     = {t: i for i, t in enumerate(teams)}
    return mu, teams, alphas, betas, t2i

mu_f, teams_f, alpha_f, beta_f, t2i_f = load_model(
    BASE / "data/processed/team_ratings_frozen.csv")

mu_r, teams_r, alpha_r, beta_r, t2i_r = load_model(
    BASE / "data/processed/team_ratings_knockout_refitted.csv")

NAME_FIX = {
    "Ivory Coast": "Cote d'Ivoire",
    "Cape Verde":  "Cabo Verde",
    "USA":         "United States",
}
# Côte d'Ivoire is in frozen params but with the accent — map Ivory Coast to it
NAME_FIX_FROZEN = {
    "Ivory Coast": "\u00c4\u00f4te d'Ivoire",   # will be fixed below
}
# Actually just look up the exact canonical in the teams list
def canon(name, teams):
    n = NAME_FIX.get(name.strip(), name.strip())
    if n in teams:
        return n
    # fuzzy: try without accents / case insensitive
    nl = n.lower()
    for t in teams:
        if t.lower() == nl:
            return t
    # Cote d'Ivoire special case
    if "ivoire" in nl or "ivory" in nl:
        for t in teams:
            if "ivoire" in t.lower():
                return t
    raise KeyError(f"Team not found: {n!r}")


# ─────────────────────────────────────────────────────────────────────────────
# Predict a match with a given model
# ─────────────────────────────────────────────────────────────────────────────
def pred(home, away, mid, date, round_name, mu, teams, alphas, betas, t2i):
    h = canon(home, teams)
    a = canon(away, teams)
    r = predict_match(h, a, mu, alphas, betas, teams, t2i)
    ph, pd_, pa = r["p_win_a"], r["p_draw"], r["p_win_b"]
    if ph >= pa:
        winner, winner_prob, loser = h, ph, a
    else:
        winner, winner_prob, loser = a, pa, h
    top = r["top_scorelines"][0]
    return dict(
        match_id=mid, round=round_name, date=date,
        home_team=h, away_team=a,
        p_home_win=round(ph,4), p_draw=round(pd_,4), p_away_win=round(pa,4),
        predicted_winner=winner, winner_prob=round(winner_prob,4),
        top_scoreline=f"{int(top[0])}-{int(top[1])}",
        loser=loser,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Bracket template — all 16 R32 matches
# ─────────────────────────────────────────────────────────────────────────────
R32 = [
    (74,  "Germany",       "Paraguay",               "2026-06-29"),
    (77,  "France",        "Sweden",                 "2026-06-30"),
    (73,  "South Africa",  "Canada",                 "2026-06-29"),
    (75,  "Netherlands",   "Morocco",                "2026-06-29"),
    (83,  "Portugal",      "Croatia",                "2026-07-02"),
    (84,  "Spain",         "Austria",                "2026-07-02"),
    (81,  "USA",           "Bosnia and Herzegovina", "2026-07-01"),
    (82,  "Belgium",       "Senegal",                "2026-07-01"),
    (76,  "Brazil",        "Japan",                  "2026-06-29"),
    (78,  "Ivory Coast",   "Norway",                 "2026-06-30"),
    (79,  "Mexico",        "Ecuador",                "2026-06-30"),
    (80,  "England",       "DR Congo",               "2026-07-01"),
    (86,  "Argentina",     "Cape Verde",             "2026-07-03"),
    (88,  "Australia",     "Egypt",                  "2026-07-03"),
    (85,  "Switzerland",   "Algeria",                "2026-07-02"),
    (87,  "Colombia",      "Ghana",                  "2026-07-03"),
]

# ─────────────────────────────────────────────────────────────────────────────
# Simulate the full bracket for one model
# ─────────────────────────────────────────────────────────────────────────────
def simulate(mu, teams, alphas, betas, t2i, label):
    results = {}
    matches = []

    def p(home, away, mid, date, rnd):
        r = pred(home, away, mid, date, rnd, mu, teams, alphas, betas, t2i)
        results[mid] = r
        matches.append(r)
        return r

    def w(mid): return results[mid]["predicted_winner"]
    def l(mid): return results[mid]["loser"]

    # R32
    for mid, home, away, date in R32:
        p(home, away, mid, date, "Round of 32")

    # R16
    r16 = [
        (89, w(74), w(77), "2026-07-05"),
        (90, w(73), w(75), "2026-07-05"),
        (91, w(83), w(84), "2026-07-06"),
        (92, w(81), w(82), "2026-07-06"),
        (93, w(76), w(78), "2026-07-04"),
        (94, w(79), w(80), "2026-07-04"),
        (95, w(86), w(88), "2026-07-07"),
        (96, w(85), w(87), "2026-07-07"),
    ]
    for mid, home, away, date in r16:
        p(home, away, mid, date, "Round of 16")

    # QF
    qf = [
        (97,  w(89), w(90), "2026-07-10"),
        (98,  w(93), w(94), "2026-07-10"),
        (99,  w(91), w(92), "2026-07-09"),
        (100, w(95), w(96), "2026-07-09"),
    ]
    for mid, home, away, date in qf:
        p(home, away, mid, date, "Quarter-final")

    # SF
    sf = [
        (101, w(97),  w(98),  "2026-07-14"),
        (102, w(99),  w(100), "2026-07-14"),
    ]
    for mid, home, away, date in sf:
        p(home, away, mid, date, "Semi-final")

    # 3rd place + final
    p(l(101), l(102), 103, "2026-07-18", "3rd Place")
    p(w(101), w(102), 104, "2026-07-19", "Final")

    return results, matches


# ─────────────────────────────────────────────────────────────────────────────
# Run both simulations
# ─────────────────────────────────────────────────────────────────────────────
print("Simulating frozen model...")
res_f, matches_f = simulate(mu_f, teams_f, alpha_f, beta_f, t2i_f, "frozen")

print("Simulating re-fitted model...")
res_r, matches_r = simulate(mu_r, teams_r, alpha_r, beta_r, t2i_r, "re-fitted")


# ─────────────────────────────────────────────────────────────────────────────
# Print console summary (ASCII-safe)
# ─────────────────────────────────────────────────────────────────────────────
ROUNDS = ["Round of 32", "Round of 16", "Quarter-final", "Semi-final", "3rd Place", "Final"]
ALL_IDS = [r["match_id"] for r in matches_f]

def row_str(r):
    flag = "<-- " if r["predicted_winner"] == r["away_team"] else "--> "
    return (f"{r['home_team']:25s}  {r['p_home_win']:.3f}/{r['p_draw']:.3f}/{r['p_away_win']:.3f}  "
            f"{r['away_team']:25s}  {flag}{r['predicted_winner']} ({r['winner_prob']:.1%})")

print("\n" + "=" * 110)
print("2026 FIFA World Cup -- Full Bracket Simulation (BOTH MODELS)")
print("=" * 110)
print(f"  {'#':>3}  {'--- FROZEN MODEL ---':^55s}   {'--- RE-FITTED MODEL ---':^55s}  FLIP?")
print(f"  {'_'*105}")

for rnd in ROUNDS:
    rnd_matches = [r for r in matches_f if r["round"] == rnd]
    if rnd_matches:
        print(f"\n  [{rnd.upper()}]")
    for rf in rnd_matches:
        mid = rf["match_id"]
        rr = res_r[mid]
        flip = " <<< FLIP" if rf["predicted_winner"] != rr["predicted_winner"] else ""
        print(f"  {mid:>3}  {row_str(rf)}   |  {row_str(rr)}{flip}")

print("\n" + "=" * 110)
print("PREDICTED OUTCOMES")
print("=" * 110)
print(f"  {'':30s}  {'FROZEN':^30s}  {'RE-FITTED':^30s}")
print(f"  {'_'*90}")
for place, label in [("Champion", 104), ("Runner-Up", 104), ("3rd Place", 103), ("4th Place", 103)]:
    if place == "Runner-Up":
        f_team = res_f[label]["loser"]
        r_team = res_r[label]["loser"]
    elif place == "4th Place":
        f_team = res_f[label]["loser"]
        r_team = res_r[label]["loser"]
    elif place == "3rd Place":
        f_team = res_f[label]["predicted_winner"]
        r_team = res_r[label]["predicted_winner"]
    else:  # Champion
        f_team = res_f[label]["predicted_winner"]
        r_team = res_r[label]["predicted_winner"]
    match = "" if f_team == r_team else " <-- DIFFERENT"
    print(f"  {place:12s}  {f_team:^30s}  {r_team:^30s}{match}")
print()


# ─────────────────────────────────────────────────────────────────────────────
# Save CSVs
# ─────────────────────────────────────────────────────────────────────────────
def to_df(matches):
    return pd.DataFrame([{k: v for k, v in r.items() if k != "loser"} for r in matches])

out_dir = BASE / "predictions"
to_df(matches_f).to_csv(out_dir / "full_bracket_frozen.csv", index=False)
to_df(matches_r).to_csv(out_dir / "full_bracket_refitted.csv", index=False)
print("Saved: full_bracket_frozen.csv")
print("Saved: full_bracket_refitted.csv")


# ─────────────────────────────────────────────────────────────────────────────
# Save markdown comparison
# ─────────────────────────────────────────────────────────────────────────────
def md_row(rf, rr):
    flip = " **FLIP**" if rf["predicted_winner"] != rr["predicted_winner"] else ""
    return (
        f"| {rf['match_id']} | {rf['home_team']} vs {rf['away_team']} "
        f"| {rf['p_home_win']:.3f}/{rf['p_draw']:.3f}/{rf['p_away_win']:.3f} "
        f"| **{rf['predicted_winner']}** ({rf['winner_prob']:.1%}) "
        f"| {rr['p_home_win']:.3f}/{rr['p_draw']:.3f}/{rr['p_away_win']:.3f} "
        f"| **{rr['predicted_winner']}** ({rr['winner_prob']:.1%}){flip} |"
    )

hdr = "| # | Match | Frozen P(H)/P(D)/P(A) | Frozen Winner | New P(H)/P(D)/P(A) | New Winner |"
sep = "|---|-------|----------------------|---------------|-------------------|------------|"

lines = [
    "# 2026 FIFA World Cup — Full Bracket Simulation (Both Models)",
    "",
    "| Model | Champion | Runner-Up | 3rd Place | 4th Place |",
    "|-------|----------|-----------|-----------|-----------|",
    f"| **Frozen** | **{res_f[104]['predicted_winner']}** | {res_f[104]['loser']} "
    f"| {res_f[103]['predicted_winner']} | {res_f[103]['loser']} |",
    f"| **Re-Fitted** | **{res_r[104]['predicted_winner']}** | {res_r[104]['loser']} "
    f"| {res_r[103]['predicted_winner']} | {res_r[103]['loser']} |",
    "",
    "---",
]

for rnd in ROUNDS:
    rnd_f = [r for r in matches_f if r["round"] == rnd]
    if not rnd_f:
        continue
    lines += [f"\n## {rnd}", "", hdr, sep]
    for rf in rnd_f:
        lines.append(md_row(rf, res_r[rf["match_id"]]))

lines += [
    "",
    "---",
    "*Frozen model: 1,433 historical matches only.*",
    "*Re-fitted model: 1,433 historical + 72 WC 2026 group stage results.*",
]

(out_dir / "full_bracket_comparison.md").write_text("\n".join(lines), encoding="utf-8")
print("Saved: full_bracket_comparison.md")
