"""
Data loading, schema validation, and matchup canonicalization.
"""

import re
import warnings
import numpy as np
import pandas as pd
from typing import Dict, Tuple, Optional

from src.config import FILES, COVID_SEASON, CURRENT_SEASON


# ═══════════════════════════════════════════════════════════════════════════════
#  Loading helpers
# ═══════════════════════════════════════════════════════════════════════════════

def load_csv(key: str, **kwargs) -> pd.DataFrame:
    """Load a CSV by its FILES key."""
    path = FILES[key]
    return pd.read_csv(path, **kwargs)


def parse_seed(seed_str: str) -> Tuple[str, int, bool]:
    """
    Parse seed string like 'W01', 'X16a' → (region, seed_number, is_playin).
    """
    m = re.match(r"([WXYZ])(\d{2})([ab])?", seed_str)
    if not m:
        raise ValueError(f"Cannot parse seed: {seed_str}")
    region = m.group(1)
    num = int(m.group(2))
    playin = m.group(3) is not None
    return region, num, playin


def seed_to_int(seed_str: str) -> float:
    """Convert seed string to numeric – play-in seeds get +0.5."""
    _, num, playin = parse_seed(seed_str)
    return float(num) + (0.5 if playin else 0.0)


# ═══════════════════════════════════════════════════════════════════════════════
#  Load all data for a gender
# ═══════════════════════════════════════════════════════════════════════════════

def load_all(gender: str = "m") -> Dict[str, pd.DataFrame]:
    """
    Load all relevant CSVs for the given gender ('m' or 'w').
    Returns dict of DataFrames keyed by short name.
    """
    g = gender.lower()
    assert g in ("m", "w"), "gender must be 'm' or 'w'"

    data = {}

    # Core results
    data["reg_compact"] = load_csv(f"{g}_reg_compact")
    data["reg_detailed"] = load_csv(f"{g}_reg_detailed")
    data["tourney_compact"] = load_csv(f"{g}_tourney_compact")
    data["tourney_detailed"] = load_csv(f"{g}_tourney_detailed")

    # Seeds & slots
    data["seeds"] = load_csv(f"{g}_seeds")
    data["seeds"]["SeedNum"] = data["seeds"]["Seed"].apply(seed_to_int)
    data["slots"] = load_csv(f"{g}_slots")

    # Metadata
    data["seasons"] = load_csv(f"{g}_seasons")
    data["teams"] = load_csv(f"{g}_teams")
    data["conferences_map"] = load_csv(f"{g}_conferences")
    data["conferences"] = load_csv("conferences")

    # Extra (may not exist for women)
    if g == "m":
        data["coaches"] = load_csv("m_coaches")
        data["seed_round_slots"] = load_csv("m_seed_round_slots")
        # Massey: large file – load lazily or in chunks
        # data["massey"] = load_csv("m_massey")  # deferred

    # Conference tourney & secondary
    data["conf_tourney"] = load_csv(f"{g}_conf_tourney")
    data["secondary_results"] = load_csv(f"{g}_secondary_results")

    # Game cities (optional)
    try:
        data["game_cities"] = load_csv(f"{g}_game_cities")
    except Exception:
        pass

    return data


# ═══════════════════════════════════════════════════════════════════════════════
#  Canonicalize matchups
# ═══════════════════════════════════════════════════════════════════════════════

# Box-score stat columns for winner / loser (detailed results)
_STAT_COLS = ["FGM", "FGA", "FGM3", "FGA3", "FTM", "FTA",
              "OR", "DR", "Ast", "TO", "Stl", "Blk", "PF"]


def canonicalize_game(df: pd.DataFrame, source: str = "regular") -> pd.DataFrame:
    """
    Convert a results DataFrame (compact or detailed, W/L format) into
    canonical form where TeamA < TeamB by ID.

    Returns DataFrame with columns:
        Season, DayNum, TeamA, TeamB, ScoreA, ScoreB, target,
        WLoc, NumOT, source, [+ per-team stat cols for detailed]
    """
    out = pd.DataFrame()
    out["Season"] = df["Season"]
    out["DayNum"] = df["DayNum"]
    out["WTeamID"] = df["WTeamID"]
    out["LTeamID"] = df["LTeamID"]
    out["WScore"] = df["WScore"]
    out["LScore"] = df["LScore"]
    out["WLoc"] = df["WLoc"] if "WLoc" in df.columns else "N"
    out["NumOT"] = df["NumOT"] if "NumOT" in df.columns else 0

    # Determine canonical ordering
    a_is_winner = df["WTeamID"] < df["LTeamID"]

    out["TeamA"] = np.where(a_is_winner, df["WTeamID"], df["LTeamID"])
    out["TeamB"] = np.where(a_is_winner, df["LTeamID"], df["WTeamID"])
    out["ScoreA"] = np.where(a_is_winner, df["WScore"], df["LScore"])
    out["ScoreB"] = np.where(a_is_winner, df["LScore"], df["WScore"])
    out["target"] = np.where(a_is_winner, 1, 0)  # 1 if TeamA won

    # Location from TeamA's perspective
    # WLoc is from winner's perspective: H/A/N
    out["LocA"] = "N"  # default neutral
    # If TeamA is the winner → LocA = WLoc; if TeamA is loser → flip H↔A
    loc_map_flip = {"H": "A", "A": "H", "N": "N"}
    wloc = df["WLoc"].values if "WLoc" in df.columns else np.full(len(df), "N")
    out["LocA"] = np.where(
        a_is_winner,
        wloc,
        [loc_map_flip.get(l, "N") for l in wloc]
    )

    out["source"] = source

    # Detailed stats (if present)
    has_detail = "WFGM" in df.columns
    if has_detail:
        for col in _STAT_COLS:
            out[f"A_{col}"] = np.where(a_is_winner, df[f"W{col}"], df[f"L{col}"])
            out[f"B_{col}"] = np.where(a_is_winner, df[f"L{col}"], df[f"W{col}"])

    return out.reset_index(drop=True)


def build_all_games(data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """
    Build a single canonical game-level DataFrame from all sources:
    regular season, NCAA tourney, conference tourney, secondary tourney.
    """
    frames = []

    # Regular season (use detailed if available, else compact)
    frames.append(canonicalize_game(data["reg_detailed"], source="regular"))
    # Add compact-only seasons (those NOT in detailed)
    detailed_seasons = set(data["reg_detailed"]["Season"].unique())
    compact_extra = data["reg_compact"][
        ~data["reg_compact"]["Season"].isin(detailed_seasons)
    ]
    if len(compact_extra) > 0:
        frames.append(canonicalize_game(compact_extra, source="regular"))

    # NCAA tournament
    frames.append(canonicalize_game(data["tourney_detailed"], source="ncaa_tourney"))
    detailed_t_seasons = set(data["tourney_detailed"]["Season"].unique())
    compact_t_extra = data["tourney_compact"][
        ~data["tourney_compact"]["Season"].isin(detailed_t_seasons)
    ]
    if len(compact_t_extra) > 0:
        frames.append(canonicalize_game(compact_t_extra, source="ncaa_tourney"))

    # Conference tourney (compact only – no detailed box scores)
    if "conf_tourney" in data and len(data["conf_tourney"]) > 0:
        ct = data["conf_tourney"].copy()
        # conf_tourney has: Season, ConfAbbrev, DayNum, WTeamID, LTeamID
        # Missing: WScore, LScore, WLoc, NumOT → fill with NaN/defaults
        ct["WScore"] = np.nan
        ct["LScore"] = np.nan
        ct["WLoc"] = "N"
        ct["NumOT"] = 0
        frames.append(canonicalize_game(ct, source="conf_tourney"))

    # Secondary tourney
    if "secondary_results" in data and len(data["secondary_results"]) > 0:
        frames.append(canonicalize_game(data["secondary_results"], source="secondary"))

    all_games = pd.concat(frames, ignore_index=True)
    all_games.sort_values(["Season", "DayNum", "TeamA", "TeamB"], inplace=True)
    all_games.reset_index(drop=True, inplace=True)

    return all_games


def build_tourney_dataset(data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """NCAA tournament games only, canonical form."""
    frames = []
    frames.append(canonicalize_game(data["tourney_detailed"], source="ncaa_tourney"))
    detailed_seasons = set(data["tourney_detailed"]["Season"].unique())
    extra = data["tourney_compact"][
        ~data["tourney_compact"]["Season"].isin(detailed_seasons)
    ]
    if len(extra) > 0:
        frames.append(canonicalize_game(extra, source="ncaa_tourney"))
    tourney = pd.concat(frames, ignore_index=True)
    tourney.sort_values(["Season", "DayNum"], inplace=True)
    return tourney.reset_index(drop=True)


# ═══════════════════════════════════════════════════════════════════════════════
#  Season-level team aggregates
# ═══════════════════════════════════════════════════════════════════════════════

def build_season_aggregates(all_games: pd.DataFrame) -> pd.DataFrame:
    """
    Build per-team, per-season aggregate stats from regular-season games.
    Returns one row per (Season, TeamID).
    """
    reg = all_games[all_games["source"] == "regular"].copy()

    a_side = reg[["Season", "TeamA", "ScoreA", "ScoreB"]].rename(
        columns={"TeamA": "TeamID", "ScoreA": "Pts", "ScoreB": "OppPts"}
    )
    a_side["Win"] = (a_side["Pts"] > a_side["OppPts"]).astype(int)

    b_side = reg[["Season", "TeamB", "ScoreB", "ScoreA"]].rename(
        columns={"TeamB": "TeamID", "ScoreB": "Pts", "ScoreA": "OppPts"}
    )
    b_side["Win"] = (b_side["Pts"] > b_side["OppPts"]).astype(int)

    team_games = pd.concat([a_side, b_side], ignore_index=True)
    team_games["Margin"] = team_games["Pts"] - team_games["OppPts"]

    agg = team_games.groupby(["Season", "TeamID"]).agg(
        Games=("Win", "count"),
        Wins=("Win", "sum"),
        AvgPts=("Pts", "mean"),
        AvgOppPts=("OppPts", "mean"),
        AvgMargin=("Margin", "mean"),
        StdMargin=("Margin", "std"),
    ).reset_index()
    agg["WinPct"] = agg["Wins"] / agg["Games"]

    return agg
