"""
standings.py — Group standings computation and Round of 32 bracket resolution.

Reads actual_outcomes.csv (72 completed group stage results) and resolves all
Round of 32 matchup slots (group winners, runners-up, best third-place teams)
to actual team names, ready for knockout prediction.

FIFA 2026 Third-Place Selection Rule:
  All 12 third-place teams are ranked by: Points > Goal Difference > Goals Scored > Fair Play.
  The best 4 advance to the Round of 32. Each bracket slot specifies which pool of groups
  the qualifying third-place team must come from.
"""

import pandas as pd
import numpy as np
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Group assignment: maps every team to its group
# ─────────────────────────────────────────────────────────────────────────────
GROUP_MAP = {
    # Group A
    "Mexico": "A", "South Africa": "A", "Korea Republic": "A", "Czechia": "A",
    # Group B
    "Canada": "B", "Bosnia and Herzegovina": "B", "Qatar": "B", "Switzerland": "B",
    # Group C
    "Haiti": "C", "Scotland": "C", "Brazil": "C", "Morocco": "C",
    # Group D
    "United States": "D", "Paraguay": "D", "Australia": "D", "Türkiye": "D",
    # Group E
    "Côte d'Ivoire": "E", "Ecuador": "E", "Germany": "E", "Curaçao": "E",
    # Group F
    "Netherlands": "F", "Japan": "F", "Sweden": "F", "Tunisia": "F",
    # Group G
    "Saudi Arabia": "G", "Uruguay": "G", "Spain": "G", "Cabo Verde": "G",
    # Group H
    "IR Iran": "H", "New Zealand": "H", "Belgium": "H", "Egypt": "H",
    # Group I
    "France": "I", "Senegal": "I", "Iraq": "I", "Norway": "I",
    # Group J
    "Argentina": "J", "Algeria": "J", "Austria": "J", "Jordan": "J",
    # Group K
    "Portugal": "K", "DR Congo": "K", "Uzbekistan": "K", "Colombia": "K",
    # Group L
    "Ghana": "L", "Panama": "L", "England": "L", "Croatia": "L",
}

# ─────────────────────────────────────────────────────────────────────────────
# Round of 32 bracket template (from fixtures.txt / data/processed/wc_2026_fixtures.csv)
# Each entry: (match_id, date, slot_home, slot_away, location)
# Slots like "W:A" = winner of Group A, "R:B" = runner-up of Group B,
# "T:ABCDF" = best third-place team from among groups A,B,C,D,F
# ─────────────────────────────────────────────────────────────────────────────
R32_TEMPLATE = [
    (73, "2026-06-28", "R:A",      "R:B",       "Los Angeles Stadium"),
    (74, "2026-06-29", "W:E",      "T:ABCDF",   "Boston Stadium"),
    (75, "2026-06-29", "W:F",      "R:C",       "Estadio Monterrey"),
    (76, "2026-06-29", "W:C",      "R:F",       "Houston Stadium"),
    (77, "2026-06-30", "W:I",      "T:CDFGH",   "New York New Jersey Stadium"),
    (78, "2026-06-30", "R:E",      "R:I",       "Dallas Stadium"),
    (79, "2026-06-30", "W:A",      "T:ACEFHI",  "Mexico City Stadium"),
    (80, "2026-07-01", "W:L",      "T:EHIJK",   "Atlanta Stadium"),
    (81, "2026-07-01", "W:D",      "T:BEFIJ",   "San Francisco Bay Area Stadium"),
    (82, "2026-07-01", "W:G",      "T:AEHIJ",   "Seattle Stadium"),
    (83, "2026-07-02", "R:K",      "R:L",       "Toronto Stadium"),
    (84, "2026-07-02", "W:H",      "R:J",       "Los Angeles Stadium"),
    (85, "2026-07-02", "W:B",      "T:EFGIJ",   "BC Place Vancouver"),
    (86, "2026-07-03", "W:J",      "R:H",       "Miami Stadium"),
    (87, "2026-07-03", "W:K",      "T:DEIJL",   "Kansas City Stadium"),
    (88, "2026-07-03", "R:D",      "R:G",       "Dallas Stadium"),
]


def compute_group_standings(results_path: str) -> dict[str, pd.DataFrame]:
    """
    Reads actual_outcomes.csv and builds a standings table for each group.

    Returns a dict: group_letter -> DataFrame with columns:
        team, group, W, D, L, GF, GA, GD, Pts
    sorted by Pts DESC, GD DESC, GF DESC (standard FIFA tiebreaker).
    """
    df = pd.read_csv(results_path)

    # Keep only the 72 group stage rows (match_id 1–72)
    df = df[df["match_id"] <= 72].copy()

    # Assign group to each row based on home team
    df["group"] = df["home_team"].map(GROUP_MAP)

    # Build per-team records
    records = {}

    def add_result(team, gf, ga):
        if team not in records:
            records[team] = dict(W=0, D=0, L=0, GF=0, GA=0)
        r = records[team]
        r["GF"] += gf
        r["GA"] += ga
        if gf > ga:
            r["W"] += 1
        elif gf == ga:
            r["D"] += 1
        else:
            r["L"] += 1

    for _, row in df.iterrows():
        add_result(row["home_team"], int(row["home_goals"]), int(row["away_goals"]))
        add_result(row["away_team"], int(row["away_goals"]), int(row["home_goals"]))

    # Build DataFrame
    rows = []
    for team, r in records.items():
        pts = r["W"] * 3 + r["D"]
        gd = r["GF"] - r["GA"]
        rows.append({
            "team": team,
            "group": GROUP_MAP[team],
            "W": r["W"], "D": r["D"], "L": r["L"],
            "GF": r["GF"], "GA": r["GA"],
            "GD": gd, "Pts": pts
        })

    standings_df = pd.DataFrame(rows)

    # Split by group and sort
    group_tables = {}
    for grp in sorted(standings_df["group"].unique()):
        t = standings_df[standings_df["group"] == grp].copy()
        t = t.sort_values(["Pts", "GD", "GF"], ascending=False).reset_index(drop=True)
        t["Position"] = t.index + 1
        group_tables[grp] = t

    return group_tables


def get_team_at_position(group_tables: dict, group: str, position: int) -> str:
    """Return the team at the given position (1=winner, 2=runner-up, 3=third) in a group."""
    t = group_tables[group]
    row = t[t["Position"] == position]
    if len(row) == 0:
        raise ValueError(f"No team at position {position} in Group {group}")
    return row.iloc[0]["team"]


def select_best_third_place_teams(group_tables: dict, eligible_groups: str) -> list[str]:
    """
    Given a pool of eligible group letters (e.g. 'ABCDF'), return ALL third-place
    teams from those groups, sorted by Pts DESC, GD DESC, GF DESC.
    The caller decides how many to pick.
    """
    third_place_rows = []
    for grp in eligible_groups:
        t = group_tables[grp]
        third = t[t["Position"] == 3]
        if len(third) == 0:
            logger.warning(f"No third-place team found in group {grp}")
            continue
        third_place_rows.append(third.iloc[0])

    if not third_place_rows:
        return []

    pool = pd.DataFrame(third_place_rows)
    pool = pool.sort_values(["Pts", "GD", "GF"], ascending=False).reset_index(drop=True)
    return pool["team"].tolist()


def resolve_r32_bracket(results_path: str) -> list[dict]:
    """
    Main entry point. Computes group standings and resolves all 16 Round of 32 fixtures.

    Returns a list of 16 dicts with keys:
        match_id, date, home_team, away_team, location, round
    """
    logger.info("Computing group standings from actual results...")
    group_tables = compute_group_standings(results_path)

    # Log standings for all groups
    for grp in sorted(group_tables.keys()):
        t = group_tables[grp]
        logger.info(f"\nGroup {grp}:")
        for _, row in t.iterrows():
            logger.info(f"  {row['Position']}. {row['team']:30s} Pts:{row['Pts']} GD:{row['GD']:+d} GF:{row['GF']}")

    # --- Resolve third-place teams ---
    # Gather ALL 12 third-place teams and rank them globally
    all_groups = "ABCDEFGHIJKL"
    all_thirds = select_best_third_place_teams(group_tables, all_groups)
    logger.info(f"\nAll third-place teams ranked: {all_thirds}")

    # The 4 best third-place teams advance. Their bracket placement is fixed
    # by FIFA rules based on which groups they came from. We determine which
    # of the pre-defined "T:XXXXX" slots each advancing third-place team goes to
    # by matching the team's group letter against the eligible pool in each slot.

    advancing_thirds = all_thirds[:4]
    advancing_thirds_groups = {t: GROUP_MAP[t] for t in advancing_thirds}
    logger.info(f"Advancing third-place teams: {advancing_thirds_groups}")

    def resolve_slot(slot: str) -> str:
        """Resolve a bracket slot string to an actual team name."""
        kind = slot[0]  # W, R, or T
        info = slot[2:]  # group letter or pool letters

        if kind == "W":
            return get_team_at_position(group_tables, info, 1)
        elif kind == "R":
            return get_team_at_position(group_tables, info, 2)
        elif kind == "T":
            # Find which advancing third-place team belongs to this pool
            eligible_pool = list(info)  # e.g. ['A','B','C','D','F']
            candidates = [
                t for t in advancing_thirds
                if advancing_thirds_groups[t] in eligible_pool
            ]
            if not candidates:
                raise ValueError(
                    f"No advancing third-place team found for pool {info}. "
                    f"Advancing thirds groups: {advancing_thirds_groups}"
                )
            if len(candidates) > 1:
                # More than one advancing third fits this pool: pick highest ranked
                logger.warning(
                    f"Multiple advancing thirds for pool {info}: {candidates}. "
                    f"Using highest-ranked: {candidates[0]}"
                )
            return candidates[0]
        else:
            raise ValueError(f"Unknown slot kind: {slot}")

    fixtures = []
    for match_id, date, slot_home, slot_away, location in R32_TEMPLATE:
        home_team = resolve_slot(slot_home)
        away_team = resolve_slot(slot_away)
        fixtures.append({
            "match_id": match_id,
            "date": date,
            "home_team": home_team,
            "away_team": away_team,
            "location": location,
            "round": "Round of 32",
        })
        logger.info(f"Match {match_id}: {home_team} vs {away_team} ({date})")

    logger.info(f"\n✓ Resolved {len(fixtures)} Round of 32 fixtures")
    return fixtures


def print_standings_table(group_tables: dict) -> None:
    """Pretty-print all group standings to stdout."""
    print("\n" + "=" * 70)
    print("2026 FIFA World Cup — Final Group Stage Standings")
    print("=" * 70)
    for grp in sorted(group_tables.keys()):
        t = group_tables[grp]
        print(f"\nGroup {grp}")
        print(f"  {'Pos':<4} {'Team':<30} {'W':>2} {'D':>2} {'L':>2} {'GF':>3} {'GA':>3} {'GD':>4} {'Pts':>4}")
        print(f"  {'-'*65}")
        for _, row in t.iterrows():
            qualifier = "✓" if row["Position"] <= 2 else " "
            print(
                f"  {qualifier}{int(row['Position']):<3} {row['team']:<30} "
                f"{int(row['W']):>2} {int(row['D']):>2} {int(row['L']):>2} "
                f"{int(row['GF']):>3} {int(row['GA']):>3} {int(row['GD']):>+4} "
                f"{int(row['Pts']):>4}"
            )
    print()


if __name__ == "__main__":
    import sys
    results = sys.argv[1] if len(sys.argv) > 1 else "results/actual_outcomes.csv"
    group_tables = compute_group_standings(results)
    print_standings_table(group_tables)
    fixtures = resolve_r32_bracket(results)
    print("\n" + "=" * 70)
    print("Round of 32 Fixtures")
    print("=" * 70)
    for f in fixtures:
        print(f"  Match {f['match_id']} ({f['date']}): {f['home_team']} vs {f['away_team']}")
