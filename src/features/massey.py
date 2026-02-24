"""
Massey Ordinals feature extraction (Men's only).
Reads MMasseyOrdinals.csv in chunks, selects top systems,
computes rank-based features per team-season.
"""

import numpy as np
import pandas as pd
from typing import List, Optional

from src.config import FILES


# Systems known to be strong predictors (from historical Kaggle analysis)
TOP_SYSTEMS = ["POM", "SAG", "MOR", "DOL", "COL"]
FALLBACK_SYSTEMS = ["RPI", "WLK", "WOL", "RTH", "AP"]

# Tournament typically starts around DayNum 134
TOURNEY_DAY_CUTOFF = 133


def load_massey_for_season(season: int, systems: Optional[List[str]] = None,
                            max_day: int = TOURNEY_DAY_CUTOFF) -> pd.DataFrame:
    """
    Load Massey ordinals for one season, filtered to pre-tournament data only.
    Uses chunked reading to handle the large file.
    """
    path = FILES["m_massey"]
    if systems is None:
        systems = TOP_SYSTEMS + FALLBACK_SYSTEMS

    chunks = []
    for chunk in pd.read_csv(path, chunksize=500_000):
        filtered = chunk[
            (chunk["Season"] == season) &
            (chunk["RankingDayNum"] <= max_day) &
            (chunk["SystemName"].isin(systems))
        ]
        if len(filtered) > 0:
            chunks.append(filtered)

    if not chunks:
        return pd.DataFrame()

    return pd.concat(chunks, ignore_index=True)


def build_massey_features(seasons: list) -> pd.DataFrame:
    """
    For each season, get the LAST available ranking per system per team
    before the tournament. Pivot to get one row per (Season, TeamID).
    """
    all_records = []

    for season in seasons:
        df = load_massey_for_season(season)
        if df.empty:
            continue

        # Take the last ranking day per system per team
        idx = df.groupby(["SystemName", "TeamID"])["RankingDayNum"].idxmax()
        latest = df.loc[idx]

        # Pivot: rows = (Season, TeamID), columns = SystemName
        pivot = latest.pivot_table(
            index=["Season", "TeamID"],
            columns="SystemName",
            values="OrdinalRank",
        ).reset_index()

        # Flatten column names
        pivot.columns = [f"Massey_{c}" if c not in ("Season", "TeamID") else c
                         for c in pivot.columns]

        # Compute composite rank (mean across available systems)
        rank_cols = [c for c in pivot.columns if c.startswith("Massey_")]
        pivot["Massey_Mean"] = pivot[rank_cols].mean(axis=1)

        all_records.append(pivot)

    if not all_records:
        return pd.DataFrame()

    return pd.concat(all_records, ignore_index=True)


def get_massey_diff_features(massey_df: pd.DataFrame, team_a: int, team_b: int,
                              season: int) -> dict:
    """
    Compute rank differences for a matchup.
    Negative diff = TeamA ranked higher (better) = lower rank number.
    """
    if massey_df.empty:
        return {}

    row_a = massey_df[(massey_df["Season"] == season) &
                       (massey_df["TeamID"] == team_a)]
    row_b = massey_df[(massey_df["Season"] == season) &
                       (massey_df["TeamID"] == team_b)]

    features = {}
    if len(row_a) == 0 or len(row_b) == 0:
        return features

    ra = row_a.iloc[0]
    rb = row_b.iloc[0]

    rank_cols = [c for c in massey_df.columns if c.startswith("Massey_")]
    for col in rank_cols:
        if pd.notna(ra.get(col)) and pd.notna(rb.get(col)):
            features[f"{col}_diff"] = ra[col] - rb[col]

    return features
