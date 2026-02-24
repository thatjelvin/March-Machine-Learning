"""
Data integrity validation – duplicates, ID consistency, chronological ordering.
"""

import warnings
import pandas as pd
from typing import Dict

from src.config import COVID_SEASON, CURRENT_SEASON


def validate_data(data: Dict[str, pd.DataFrame], gender: str = "m") -> bool:
    """
    Run all validation checks. Returns True if no fatal errors.
    Prints warnings for non-fatal issues.
    """
    ok = True

    # ── 1. No duplicate games ─────────────────────────────────────────────
    for key in ["reg_compact", "reg_detailed", "tourney_compact", "tourney_detailed"]:
        df = data[key]
        dup_cols = ["Season", "DayNum", "WTeamID", "LTeamID"]
        dupes = df.duplicated(subset=dup_cols, keep=False)
        n = dupes.sum()
        if n > 0:
            warnings.warn(f"[{gender.upper()}] {key}: {n} duplicate rows on {dup_cols}")
            ok = False
        else:
            print(f"  ✓ {key}: no duplicates")

    # ── 2. WScore > LScore everywhere ─────────────────────────────────────
    for key in ["reg_compact", "reg_detailed", "tourney_compact", "tourney_detailed"]:
        df = data[key]
        bad = (df["WScore"] <= df["LScore"]).sum()
        if bad > 0:
            warnings.warn(f"[{gender.upper()}] {key}: {bad} rows with WScore <= LScore")
            ok = False
        else:
            print(f"  ✓ {key}: WScore > LScore ∀ rows")

    # ── 3. All TeamIDs in team master ─────────────────────────────────────
    valid_ids = set(data["teams"]["TeamID"].unique())
    for key in ["reg_compact", "tourney_compact"]:
        df = data[key]
        all_ids = set(df["WTeamID"].unique()) | set(df["LTeamID"].unique())
        missing = all_ids - valid_ids
        if missing:
            warnings.warn(f"[{gender.upper()}] {key}: unknown TeamIDs: {missing}")
            ok = False
        else:
            print(f"  ✓ {key}: all TeamIDs in master")

    # ── 4. COVID season check ─────────────────────────────────────────────
    tc = data["tourney_compact"]
    covid_tourney = tc[tc["Season"] == COVID_SEASON]
    if len(covid_tourney) > 0:
        warnings.warn(f"[{gender.upper()}] Found {len(covid_tourney)} tourney games in COVID season {COVID_SEASON}")
    else:
        print(f"  ✓ No tournament games in COVID season {COVID_SEASON}")

    # ── 5. 2026 should have no tourney games ──────────────────────────────
    future_tourney = tc[tc["Season"] == CURRENT_SEASON]
    if len(future_tourney) > 0:
        warnings.warn(f"[{gender.upper()}] Found {len(future_tourney)} tourney games in current season {CURRENT_SEASON}")
    else:
        print(f"  ✓ No tournament games in current season {CURRENT_SEASON}")

    # ── 6. Seed integrity ─────────────────────────────────────────────────
    seeds = data["seeds"]
    dup_seeds = seeds.duplicated(subset=["Season", "TeamID"], keep=False).sum()
    if dup_seeds > 0:
        warnings.warn(f"[{gender.upper()}] Seeds: {dup_seeds} duplicate (Season,TeamID) rows")
    else:
        print(f"  ✓ Seeds: no duplicate (Season, TeamID)")

    # Check every tourney team has a seed
    for season in tc["Season"].unique():
        if season == COVID_SEASON:
            continue
        tourney_teams = set(tc[tc["Season"] == season]["WTeamID"]) | \
                        set(tc[tc["Season"] == season]["LTeamID"])
        seeded = set(seeds[seeds["Season"] == season]["TeamID"])
        missing_seeds = tourney_teams - seeded
        if missing_seeds:
            warnings.warn(f"[{gender.upper()}] Season {season}: tourney teams without seeds: {missing_seeds}")
            ok = False

    print(f"\n  {'✓ All checks passed' if ok else '✗ Some checks failed'} for {gender.upper()}")
    return ok
