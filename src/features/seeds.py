"""
Seed-based and tournament meta-features:
  - Seed difference
  - Historical upset frequency by seed pairing
  - Conference strength index
  - Tournament experience proxy
"""

import numpy as np
import pandas as pd
from typing import Dict

from src.data_loader import seed_to_int


def build_seed_features(seeds: pd.DataFrame,
                        tourney_games: pd.DataFrame,
                        elo_snapshot: pd.DataFrame,
                        conferences_map: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    """
    Build seed-related features.
    
    Returns dict:
      'seed_lookup' -> (Season, TeamID) → SeedNum
      'upset_table' -> (SeedA, SeedB) → historical win rate for lower-seeded team
      'conf_strength' -> (Season, ConfAbbrev) → avg ELO
    """
    # ── Seed lookup ──────────────────────────────────────────────────────
    seed_lookup = seeds[["Season", "TeamID", "SeedNum"]].copy()

    # ── Historical upset frequency table ─────────────────────────────────
    # Join seeds to tourney games
    tg = tourney_games.copy()
    tg = tg.merge(
        seed_lookup.rename(columns={"TeamID": "TeamA", "SeedNum": "SeedA"}),
        on=["Season", "TeamA"], how="left"
    )
    tg = tg.merge(
        seed_lookup.rename(columns={"TeamID": "TeamB", "SeedNum": "SeedB"}),
        on=["Season", "TeamB"], how="left"
    )

    # Drop games where seeds are missing
    tg = tg.dropna(subset=["SeedA", "SeedB"])
    tg["SeedA_int"] = tg["SeedA"].astype(int)
    tg["SeedB_int"] = tg["SeedB"].astype(int)

    # For the upset table: use integer seeds, compute win rate for lower seed num (better team)
    # Canonical: lower seed is "favorite"
    tg["FavSeed"] = tg[["SeedA_int", "SeedB_int"]].min(axis=1)
    tg["DogSeed"] = tg[["SeedA_int", "SeedB_int"]].max(axis=1)
    tg["FavWon"] = np.where(
        tg["SeedA_int"] < tg["SeedB_int"],
        tg["target"],   # TeamA is fav, target=1 means fav won
        1 - tg["target"]  # TeamB is fav
    )
    # Handle same-seed matchups
    same_seed = tg["SeedA_int"] == tg["SeedB_int"]
    tg.loc[same_seed, "FavWon"] = tg.loc[same_seed, "target"]

    upset_table = tg.groupby(["FavSeed", "DogSeed"]).agg(
        FavWins=("FavWon", "sum"),
        Games=("FavWon", "count"),
    ).reset_index()
    upset_table["FavWinRate"] = upset_table["FavWins"] / upset_table["Games"]

    # ── Conference strength index ────────────────────────────────────────
    # Average ELO of all teams in each conference per season
    conf_elo = elo_snapshot.merge(conferences_map, on=["Season", "TeamID"], how="left")
    conf_strength = conf_elo.groupby(["Season", "ConfAbbrev"])["Elo"].mean().reset_index()
    conf_strength.rename(columns={"Elo": "ConfStrength"}, inplace=True)

    # ── Tournament experience ────────────────────────────────────────────
    # Count how many times each team appeared in tournament in prior seasons
    tourney_appearances = seeds.groupby("TeamID")["Season"].apply(
        lambda x: sorted(x.tolist())
    ).to_dict()

    return {
        "seed_lookup": seed_lookup,
        "upset_table": upset_table,
        "conf_strength": conf_strength,
        "tourney_appearances": tourney_appearances,
    }


def get_upset_prior(upset_table: pd.DataFrame, seed_a: int, seed_b: int) -> float:
    """
    For a matchup (seedA, seedB), return historical probability that
    the lower-numbered (better) seed wins. If not in table, use 0.5 prior.
    """
    fav = min(seed_a, seed_b)
    dog = max(seed_a, seed_b)
    row = upset_table[(upset_table["FavSeed"] == fav) & (upset_table["DogSeed"] == dog)]
    if len(row) == 0:
        return 0.5
    return row.iloc[0]["FavWinRate"]


def get_tourney_experience(tourney_appearances: dict, tid: int, season: int) -> int:
    """Number of tournament appearances before `season`."""
    apps = tourney_appearances.get(tid, [])
    return sum(1 for s in apps if s < season)
